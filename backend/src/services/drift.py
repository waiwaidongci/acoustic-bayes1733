"""漂移分界的业务服务：归班版本、任务创建、发布。

计算本身在 worker 的 ``drift_compute`` 中执行；这里只做校验、入队与
带版本比较的发布，保证并发下旧计算不会覆盖新版。
"""
import json
import secrets

from fastapi import HTTPException
from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from redis import Redis

from ..config.settings import REDIS_URL
from ..models.drift import DriftCost, DriftJob, DriftResult, DriftSample, DriftShift, ShiftAssignment, ShiftPlan
from ..models.record import Record
from ..models.run import Run
from ..models.source import Source
from ..utils.serialization import now, row
from .audit import log
from .drift_algo import MIN_SHIFTS

queue = Redis.from_url(REDIS_URL)

MAX_REFERENCES = 200000
REFERENCE_CHECK_BATCH = 1000
PUBLISH_RETRY = 3


# ---------------------------------------------------------------- 归班版本

def _shift_counts(db, plan_id):
    return dict(db.execute(
        select(ShiftAssignment.shift_order, func.count())
        .where(ShiftAssignment.plan_id == plan_id)
        .group_by(ShiftAssignment.shift_order)
    ).all())


def _plan_summary(db, plan):
    counts = _shift_counts(db, plan.id)
    return {
        **row(plan),
        'shift_count': len(counts),
        'reference_count': sum(counts.values()),
        'shifts': [{'order': order, 'count': counts[order]} for order in sorted(counts)],
    }


def list_plans(db, source_id=None):
    query = select(ShiftPlan).order_by(ShiftPlan.source_id, ShiftPlan.revision.desc())
    if source_id is not None:
        query = query.where(ShiftPlan.source_id == source_id)
    return [_plan_summary(db, plan) for plan in db.scalars(query)]


def plan_or_404(db, plan_id):
    plan = db.get(ShiftPlan, plan_id)
    if not plan:
        raise HTTPException(404, '未找到归班版本')
    return plan


def get_plan(db, plan_id):
    plan = plan_or_404(db, plan_id)
    summary = _plan_summary(db, plan)
    rows = db.execute(
        select(ShiftAssignment.shift_order, ShiftAssignment.shift_name, func.count())
        .where(ShiftAssignment.plan_id == plan_id)
        .group_by(ShiftAssignment.shift_order, ShiftAssignment.shift_name)
        .order_by(ShiftAssignment.shift_order)
    ).all()
    summary['shifts'] = [
        {'order': order, 'name': name, 'count': count} for order, name, count in rows
    ]
    return summary


def _validate_source_references(db, source_id, references):
    found = set()
    for start in range(0, len(references), REFERENCE_CHECK_BATCH):
        batch = references[start:start + REFERENCE_CHECK_BATCH]
        found.update(db.scalars(
            select(Record.reference).where(Record.source_id == source_id, Record.reference.in_(batch))
        ))
    missing = [ref for ref in references if ref not in found]
    return missing


def create_plan(db, user, data):
    source_id = data.source_id
    if not db.get(Source, source_id):
        raise HTTPException(404, '未找到来源批次')

    seen_names, seen_orders, seen_refs = set(), set(), set()
    total = 0
    shifts = []
    for shift in data.shifts:
        name = shift.name.strip()
        if not name:
            raise HTTPException(422, '班次名称不能为空')
        if name in seen_names:
            raise HTTPException(422, '班次名称重复：' + name)
        if shift.order in seen_orders:
            raise HTTPException(422, '班次顺序重复：' + str(shift.order))
        seen_names.add(name)
        seen_orders.add(shift.order)
        references = []
        for reference in shift.references:
            reference = reference.strip()
            if not reference:
                raise HTTPException(422, '样本编号不能为空')
            if reference in seen_refs:
                raise HTTPException(422, '样本编号重复归班：' + reference)
            seen_refs.add(reference)
            references.append(reference)
        total += len(references)
        shifts.append((shift.order, name, references))

    if total > MAX_REFERENCES:
        raise HTTPException(422, f'单次归班最多 {MAX_REFERENCES} 件样本')
    if not total:
        raise HTTPException(422, '归班至少需要一条样本')
    missing = _validate_source_references(db, source_id, list(seen_refs))
    if missing:
        preview = '、'.join(missing[:5])
        raise HTTPException(422, f'{len(missing)} 个样本编号不在该来源中：{preview}')

    for _ in range(PUBLISH_RETRY):
        revision = (db.scalar(
            select(func.max(ShiftPlan.revision)).where(ShiftPlan.source_id == source_id)
        ) or 0) + 1
        plan = ShiftPlan(
            source_id=source_id, revision=revision, state='draft', note=data.note,
            created_by=user.email, updated_by=user.email, created_at=now(),
        )
        db.add(plan)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            continue
        for order, name, references in shifts:
            db.add_all([
                ShiftAssignment(plan_id=plan.id, reference=ref, shift_name=name, shift_order=order)
                for ref in references
            ])
        log(db, user, 'drift.plan.created', plan.id, f'来源 #{source_id} · 版本 r{revision} · {len(shifts)} 班')
        db.commit()
        return get_plan(db, plan.id)
    raise HTTPException(409, '归班版本并发创建冲突，请重试')


# ---------------------------------------------------------------- 任务

def _latest_plan(db, source_id):
    return db.scalar(
        select(ShiftPlan).where(ShiftPlan.source_id == source_id).order_by(ShiftPlan.revision.desc())
    )


def _plan_assignments(db, plan_id):
    return list(db.execute(
        select(ShiftAssignment.reference, ShiftAssignment.shift_order, ShiftAssignment.shift_name)
        .where(ShiftAssignment.plan_id == plan_id)
    ))


def _run_or_404(db, run_id):
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(404, '未找到分析任务')
    return run


def run_labels(db, run_id):
    run = _run_or_404(db, run_id)
    if run.state != 'completed':
        raise HTTPException(409, '只可为已完成研究补录归班')
    latest = _latest_plan(db, run.source_id)
    assigned = set()
    if latest:
        assigned = set(db.scalars(
            select(ShiftAssignment.reference).where(ShiftAssignment.plan_id == latest.id)
        ))
    counts, unassigned = {}, {}
    for record in run.snapshot.get('records', []):
        label = record['values'].get('label')
        counts[label] = counts.get(label, 0) + 1
        if record['reference'] not in assigned:
            unassigned[label] = unassigned.get(label, 0) + 1
    return {
        'run_id': run.id,
        'source_id': run.source_id,
        'plan_id': latest.id if latest else None,
        'plan_revision': latest.revision if latest else None,
        'labels': [
            {'label': label, 'count': count, 'unassigned': unassigned.get(label, 0)}
            for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        ],
    }


def create_job(db, user, data):
    run = _run_or_404(db, data.run_id)
    if run.state != 'completed':
        raise HTTPException(409, '只可为已完成研究计算漂移分界')
    plan = plan_or_404(db, data.plan_id) if data.plan_id else _latest_plan(db, run.source_id)
    if not plan:
        raise HTTPException(422, '该来源还没有归班版本，请先补录样本归班')
    if plan.source_id != run.source_id:
        raise HTTPException(422, '归班版本与研究的来源不一致')
    if not isinstance(data.penalty, (int, float)) or data.penalty != data.penalty or data.penalty in (float('inf'), float('-inf')):
        raise HTTPException(422, '罚分必须是非负有限数值')
    penalty = float(data.penalty)

    label = data.label.strip()
    records = [
        record for record in run.snapshot.get('records', [])
        if record['values'].get('label') == label
    ]
    if not records:
        raise HTTPException(422, '该研究中没有指定故障的样本：' + label)
    assignments = {
        reference for reference, _, _ in _plan_assignments(db, plan.id)
    }
    missing = [record['reference'] for record in records if record['reference'] not in assignments]
    if missing:
        preview = '、'.join(missing[:5])
        raise HTTPException(422, f'{len(missing)} 件指定故障样本尚未归班：{preview}')

    job = DriftJob(
        run_id=run.id, source_id=run.source_id, plan_id=plan.id, plan_revision=plan.revision,
        label=label, parameters={'penalty': penalty, 'min_shifts': MIN_SHIFTS},
        state='queued', checkpoint={'phase': 'scan'}, created_at=now(), updated_at=now(),
    )
    db.add(job)
    db.flush()
    log(db, user, 'drift.job.queued', job.id, f'来源 #{run.source_id} · {label} · r{plan.revision}')
    db.commit()
    enqueue(db, job)
    return row(job)


def enqueue(db, job):
    try:
        queue.rpush('drift:queue', json.dumps({'kind': 'drift', 'id': job.id}))
    except Exception:
        job.state = 'failed'
        job.error = '队列暂不可用，请稍后重试'
        db.commit()
        raise HTTPException(503, job.error)


def list_jobs(db):
    jobs = db.scalars(select(DriftJob).order_by(DriftJob.id.desc()).limit(200))
    output = []
    for job in jobs:
        item = {key: value for key, value in row(job).items() if key not in ('checkpoint', 'result')}
        result = job.result or {}
        item['summary'] = {
            'boundary_count': len(result.get('boundaries', [])),
            'segment_count': len(result.get('segments', [])),
            'total': result.get('total'),
        }
        output.append(item)
    return output


def get_job(db, job_id):
    job = db.get(DriftJob, job_id)
    if not job:
        raise HTTPException(404, '未找到漂移任务')
    return row(job)


def retry_job(db, user, job_id):
    job = db.get(DriftJob, job_id)
    if not job:
        raise HTTPException(404, '未找到漂移任务')
    if job.state not in ('failed', 'running'):
        raise HTTPException(409, '只有失败或中断的任务可以重试')
    checkpoint = dict(job.checkpoint or {'phase': 'scan'})
    job.state = 'queued'
    job.error = ''
    job.claim_token = ''
    job.checkpoint = checkpoint
    job.updated_at = now()
    log(db, user, 'drift.job.retried', job.id, checkpoint.get('phase', ''))
    db.commit()
    enqueue(db, job)
    return row(job)


# ---------------------------------------------------------------- 发布

def result_complete(result):
    """各段的班次与中位数齐全，且总账对得上，才允许替换已发布版本。"""
    if not isinstance(result, dict):
        return False
    segments = result.get('segments')
    if not isinstance(segments, list) or not segments:
        return False
    try:
        shift_total = int(result['shift_total'])
        sample_total = int(result['sample_total'])
        total = float(result['total'])
        deviation_total = float(result['deviation_total'])
        penalty_total = float(result['penalty_total'])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(value == value and abs(value) != float('inf')
               for value in (total, deviation_total, penalty_total)):
        return False
    if abs(deviation_total + penalty_total - total) > 1e-6 * max(1.0, abs(total)):
        return False
    shift_sum, sample_sum = 0, 0
    for segment in segments:
        orders = segment.get('shift_orders')
        median = segment.get('median')
        if not isinstance(orders, list) or not orders:
            return False
        if int(segment.get('shift_count', 0)) != len(orders):
            return False
        if len(orders) < int(result.get('min_shifts', MIN_SHIFTS)):
            return False
        if median is None or median != median or abs(median) == float('inf'):
            return False
        if int(segment.get('sample_count', 0)) <= 0:
            return False
        shift_sum += len(orders)
        sample_sum += int(segment['sample_count'])
    if shift_sum != shift_total or sample_sum != sample_total:
        return False
    return True


def publish_result(db, job, actor, reason='drift.published'):
    """把已完成任务的结果发布为来源+故障的最新分界。

    以 (归班版本, 任务号) 作为新旧顺序：只有严格更新的计算才能替换。
    比较与写入在同一条 UPDATE 的 WHERE 中完成，并发下旧计算不会盖掉新版。
    """
    if job.state != 'completed' or not job.result:
        return False
    if not result_complete(job.result):
        log(db, actor, 'drift.publish.blocked', job.id, '分段班次或中位数不齐')
        db.commit()
        return False
    key = (int(job.plan_revision), int(job.id))
    values = {
        'run_id': job.run_id,
        'plan_id': job.plan_id,
        'plan_revision': job.plan_revision,
        'job_id': job.id,
        'result': job.result,
        'published_by': actor,
        'published_at': now(),
    }
    for _ in range(PUBLISH_RETRY):
        existing = db.scalar(
            select(DriftResult).where(
                DriftResult.source_id == job.source_id, DriftResult.label == job.label
            )
        )
        if existing is None:
            db.add(DriftResult(source_id=job.source_id, label=job.label, version=1, **values))
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                continue
            log(db, actor, reason, job.id, f'首发 · r{job.plan_revision}')
            db.commit()
            return True
        if (existing.plan_revision, existing.job_id) >= key:
            log(db, actor, 'drift.publish.skipped', job.id,
                f'现有 r{existing.plan_revision}/#{existing.job_id} 不早于本次 r{key[0]}/#{key[1]}')
            db.commit()
            return False
        updated = db.execute(
            update(DriftResult)
            .where(
                DriftResult.id == existing.id,
                or_(
                    DriftResult.plan_revision < key[0],
                    and_(DriftResult.plan_revision == key[0], DriftResult.job_id < key[1]),
                ),
            )
            .values(version=DriftResult.version + 1, **values)
        )
        db.commit()
        if updated.rowcount == 1:
            log(db, actor, reason, job.id, f'r{job.plan_revision} 替换旧版')
            db.commit()
            return True
        return False
    log(db, actor, 'drift.publish.skipped', job.id, '并发冲突，旧计算不覆盖新版')
    db.commit()
    return False


def publish_job(db, user, job_id):
    job = db.get(DriftJob, job_id)
    if not job:
        raise HTTPException(404, '未找到漂移任务')
    if job.state != 'completed':
        raise HTTPException(409, '只有已完成任务可以发布')
    if not result_complete(job.result):
        raise HTTPException(409, '各段的班次与中位数未齐，不能替换已发布版本')
    published = publish_result(db, job, user.email, reason='drift.published.manual')
    if not published:
        raise HTTPException(409, '本次计算不新于已发布版本，未替换')
    return {'published': True, 'source_id': job.source_id, 'label': job.label}


def list_results(db, source_id=None):
    query = select(DriftResult).order_by(DriftResult.published_at.desc())
    if source_id is not None:
        query = query.where(DriftResult.source_id == source_id)
    return [row(result) for result in db.scalars(query)]


def latest_result(db, source_id, label):
    return db.scalar(
        select(DriftResult).where(DriftResult.source_id == source_id, DriftResult.label == label)
    )


# ---------------------------------------------------------------- worker 辅助

def claim_job(db, job_id):
    token = secrets.token_hex(16)
    changed = db.execute(
        update(DriftJob)
        .where(DriftJob.id == job_id, DriftJob.state == 'queued')
        .values(state='running', claim_token=token, updated_at=now())
    )
    db.commit()
    if changed.rowcount != 1:
        return None
    return token


def touch_job(db, job, commit=True):
    db.execute(
        update(DriftJob)
        .where(DriftJob.id == job.id, DriftJob.claim_token == job.claim_token)
        .values(updated_at=now())
    )
    if commit:
        db.commit()


def save_checkpoint(db, job, checkpoint, commit=True):
    job.checkpoint = checkpoint
    db.execute(
        update(DriftJob)
        .where(DriftJob.id == job.id, DriftJob.claim_token == job.claim_token)
        .values(checkpoint=checkpoint, updated_at=now())
    )
    if commit:
        db.commit()


def requeue_stale(db, stale_seconds):
    """把丢失 worker 的 running 任务放回队列；已完成的分段与成本都还在。"""
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)).isoformat()
    rows = list(db.scalars(
        select(DriftJob).where(DriftJob.state == 'running', DriftJob.updated_at < cutoff)
    ))
    for job in rows:
        job.state = 'queued'
        job.claim_token = ''
        job.updated_at = now()
    db.commit()
    for job in rows:
        try:
            queue.rpush('drift:queue', json.dumps({'kind': 'drift', 'id': job.id}))
        except Exception:
            pass
    return [job.id for job in rows]
