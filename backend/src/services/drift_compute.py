"""漂移任务的独立 worker 计算：分批扫描、断点续跑、幂等重试。

十万件样本分批扫描，每批按 (任务, 样本编号) 幂等写入，重试不会重复
计入。扫描完成后按班次分组写入已排序的频谱质心，再逐个班次推进动态
规划的已算成本，每一步都写回 checkpoint 与 drift_costs；中断后从最后
一个已算班次继续。
"""
import json
import logging
import time

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from ..models.drift import DriftCost, DriftJob, DriftSample, DriftShift, ShiftAssignment
from ..models.record import Record
from ..models.run import Run
from ..utils.serialization import now
from . import drift_algo
from .drift import claim_job, publish_result, requeue_stale, save_checkpoint, touch_job

SCAN_BATCH = 1000
COST_CHECKPOINT_SECONDS = 2
STALE_SECONDS = 120


def _scan_batch(db, job, references, label):
    """把一批指定故障样本幂等写入中转表。"""
    added = 0
    for start in range(0, len(references), SCAN_BATCH):
        chunk = references[start:start + SCAN_BATCH]
        records = {
            record.reference: record
            for record in db.scalars(
                select(Record).where(Record.source_id == job.source_id, Record.reference.in_(chunk))
            )
        }
        assignments = {
            reference: (order, name)
            for reference, order, name in db.execute(
                select(ShiftAssignment.reference, ShiftAssignment.shift_order, ShiftAssignment.shift_name)
                .where(ShiftAssignment.plan_id == job.plan_id, ShiftAssignment.reference.in_(chunk))
            )
        }
        existing = set(db.scalars(
            select(DriftSample.reference)
            .where(DriftSample.job_id == job.id, DriftSample.reference.in_(chunk))
        ))
        pending = []
        for reference in chunk:
            if reference in existing:
                continue
            record = records.get(reference)
            if record is None or record.values.get('label') != label:
                continue
            assignment = assignments.get(reference)
            if assignment is None:
                raise drift_algo.DriftError(f'样本 {reference} 未归班，无法计算')
            order, name = assignment
            centroid = float(record.values.get('centroid', float('nan')))
            if not np.isfinite(centroid):
                raise drift_algo.DriftError(f'样本 {reference} 的频谱质心不是有限数值')
            pending.append(DriftSample(
                job_id=job.id, reference=reference, shift_order=order,
                shift_name=name, centroid=centroid,
            ))
        if pending:
            db.add_all(pending)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                continue
            added += len(pending)
        touch_job(db, job)
    return added


def _build_shifts(db, job, checkpoint):
    total = db.scalar(select(func.count()).select_from(DriftSample).where(DriftSample.job_id == job.id))
    built = db.scalar(select(func.count()).select_from(DriftShift).where(DriftShift.job_id == job.id))
    expected = checkpoint.get('shift_count')
    if expected and built == expected:
        return
    db.execute(delete(DriftShift).where(DriftShift.job_id == job.id))
    db.commit()
    groups = db.execute(
        select(DriftSample.shift_order, func.min(DriftSample.shift_name), func.count())
        .where(DriftSample.job_id == job.id)
        .group_by(DriftSample.shift_order)
        .order_by(DriftSample.shift_order)
    ).all()
    cursor = 0
    while cursor < len(groups):
        order, name, count = groups[cursor]
        values = list(db.scalars(
            select(DriftSample.centroid)
            .where(DriftSample.job_id == job.id, DriftSample.shift_order == order)
            .order_by(DriftSample.centroid)
        ))
        db.add(DriftShift(
            job_id=job.id, shift_order=order, shift_name=name,
            sample_count=len(values), values=[float(v) for v in values],
        ))
        cursor += 1
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
        touch_job(db, job)
    saved = {
        'phase': 'compute',
        'shift_count': len(groups),
        'raw_sample_count': int(total or 0),
        'shifts': [
            {'order': order, 'name': name, 'count': count}
            for order, name, count in groups
        ],
        'next': len(groups) - 1,
    }
    save_checkpoint(db, job, saved)


def _cost_rows(db, job, k):
    stored = {
        row.shift_index: row
        for row in db.scalars(select(DriftCost).where(DriftCost.job_id == job.id))
    }
    best_cost = [None] * k
    full_stat = [None] * k
    for index, row in stored.items():
        if 0 <= index < k:
            best_cost[index] = row.cost
            if row.full_median is not None:
                full_stat[index] = (row.full_deviation, row.full_median, row.full_count)
    return best_cost, full_stat


def _suffix_state(db, job, k):
    """恢复已算成本的连续后缀：从最小的已算班次再往前一格续跑。"""
    best_cost, full_stat = _cost_rows(db, job, k)
    computed = [index for index in range(k) if best_cost[index] is not None]
    if not computed:
        return k - 1, best_cost, full_stat
    return min(computed) - 1, best_cost, full_stat


class _CheckpointWriter:
    def __init__(self, db, job):
        self.db = db
        self.job = job
        self.last = 0.0
        self.saved = set()

    def backfill(self, best_cost, full_stat):
        for index, cost in enumerate(best_cost):
            if cost is not None and index not in self.saved:
                self.write(index, cost, full_stat[index], commit=False)
        self.db.commit()

    def maybe(self, index, cost, stat):
        moment = time.monotonic()
        if moment - self.last < COST_CHECKPOINT_SECONDS:
            return
        self.last = moment
        self.write(index, cost, stat)

    def write(self, index, cost, stat, commit=True):
        existing = self.db.scalar(
            select(DriftCost).where(DriftCost.job_id == self.job.id, DriftCost.shift_index == index)
        )
        if existing is None:
            self.db.add(DriftCost(
                job_id=self.job.id, shift_index=index, cost=cost,
                full_deviation=stat[0] if stat else None,
                full_median=stat[1] if stat else None,
                full_count=stat[2] if stat else None,
            ))
        else:
            existing.cost = cost
            existing.full_deviation = stat[0] if stat else None
            existing.full_median = stat[1] if stat else None
            existing.full_count = stat[2] if stat else None
        self.saved.add(index)
        save_checkpoint(self.db, self.job, {
            **(self.job.checkpoint or {}),
            'phase': 'compute',
            'next': index - 1,
        }, commit=False)
        if commit:
            try:
                self.db.commit()
            except IntegrityError:
                self.db.rollback()


def _shifts_from_db(db, job):
    rows = list(db.scalars(
        select(DriftShift).where(DriftShift.job_id == job.id).order_by(DriftShift.shift_order)
    ))
    blocks = [np.asarray(row.values, dtype=np.float64) for row in rows]
    return rows, blocks


def _summary(shifts, solution):
    segments = []
    for segment in solution['segments']:
        rows = shifts[segment['start']:segment['end'] + 1]
        segments.append({
            **segment,
            'shift_orders': [row.shift_order for row in rows],
            'shift_names': [row.shift_name for row in rows],
        })
    return {**solution, 'segments': segments}


def compute_job(db, job):
    checkpoint = dict(job.checkpoint or {'phase': 'scan'})
    run = db.get(Run, job.run_id)
    if run is None:
        raise drift_algo.DriftError('研究任务已不存在')
    plan_assignments = set(db.scalars(
        select(ShiftAssignment.reference).where(ShiftAssignment.plan_id == job.plan_id)
    ))
    references = [
        record['reference'] for record in run.snapshot.get('records', [])
        if record['values'].get('label') == job.label and record['reference'] in plan_assignments
    ]
    if checkpoint.get('phase') == 'scan':
        _scan_batch(db, job, references, job.label)
        checkpoint = {**checkpoint, 'phase': 'group', 'scanned': len(references)}
        save_checkpoint(db, job, checkpoint)
    if checkpoint.get('phase') == 'group':
        _build_shifts(db, job, checkpoint)
        checkpoint = dict(job.checkpoint or {})
    shifts, blocks = _shifts_from_db(db, job)
    if not shifts:
        raise drift_algo.DriftError('归班后没有可用于计算的样本')
    k = len(blocks)
    penalty = float(job.parameters['penalty'])
    min_shifts = int(job.parameters.get('min_shifts', drift_algo.MIN_SHIFTS))
    if k < min_shifts:
        raise drift_algo.DriftError(f'归班后只有 {k} 个班次，不足以按每段至少 {min_shifts} 班切分')

    start, best_cost, full_stat = _suffix_state(db, job, k)
    writer = _CheckpointWriter(db, job)
    best_cost, full_stat = drift_algo.compute_suffix_costs(
        blocks, penalty, min_shifts,
        state={'next': start, 'costs': best_cost, 'full': full_stat},
        on_checkpoint=writer.maybe,
    )
    writer.backfill(best_cost, full_stat)
    solution = drift_algo.reconstruct(blocks, penalty, min_shifts, best_cost, full_stat)
    result = _summary(shifts, solution)
    result['plan_id'] = job.plan_id
    result['plan_revision'] = job.plan_revision
    result['label'] = job.label
    result['run_id'] = job.run_id
    result['computed_at'] = now()
    json.dumps(result, allow_nan=False)
    return result


def execute(db, job_id):
    from ..models.audit import Audit
    token = claim_job(db, job_id)
    if not token:
        return
    job = db.get(DriftJob, job_id)
    try:
        result = compute_job(db, job)
        job = db.get(DriftJob, job_id)
        if job.claim_token != token:
            return
        job.result = result
        job.state = 'completed'
        job.error = ''
        job.updated_at = now()
        db.add(Audit(actor='worker', action='drift.completed', target=str(job_id),
                     detail=job.label, created_at=now()))
        db.commit()
        job = db.get(DriftJob, job_id)
        publish_result(db, job, 'worker', reason='drift.published.auto')
    except Exception as error:
        logging.exception('drift job failed')
        db.rollback()
        job = db.get(DriftJob, job_id)
        if job is not None and job.claim_token == token:
            job.state = 'failed'
            job.error = str(error)[:2000]
            job.updated_at = now()
            db.add(Audit(actor='worker', action='drift.failed', target=str(job_id),
                         detail=job.error, created_at=now()))
            db.commit()


def requeue_stale_jobs(db):
    return requeue_stale(db, STALE_SECONDS)
