import itertools
import types

import numpy as np
import pytest

from src.models.drift import DriftCost, DriftJob, DriftResult, DriftSample, DriftShift, ShiftAssignment, ShiftPlan
from src.models.record import Record
from src.models.run import Run
from src.models.source import Source
from src.schemas.drift import JobInput, PlanInput, ShiftInput
from src.services import drift_algo, drift_compute
from src.services import drift as drift_service
from src.utils.serialization import now


USER = types.SimpleNamespace(email='analyst@acoustic-bayes.demo', role='analyst')


def _brute(blocks, penalty, min_shifts):
    k = len(blocks)
    best, ties = None, []
    for size in range(k):
        for cuts in itertools.combinations(range(1, k), size):
            points = (0,) + cuts + (k,)
            segments = [(points[i], points[i + 1] - 1) for i in range(len(points) - 1)]
            if any(b - a + 1 < min_shifts for a, b in segments):
                continue
            total = sum(
                drift_algo.segment_stat(np.sort(np.concatenate([blocks[t] for t in range(a, b + 1)])))[0]
                for a, b in segments
            ) + penalty * len(cuts)
            boundaries = [b for a, b in segments[:-1]]
            if best is None or total < best - 1e-9:
                best, ties = total, [boundaries]
            elif abs(total - best) <= 1e-9:
                ties.append(boundaries)
    return best, sorted(ties)[0]


def test_matches_brute_force_including_ties():
    rng = np.random.default_rng(21)
    for _ in range(200):
        k = int(rng.integers(3, 9))
        min_shifts = int(rng.integers(2, 4))
        blocks = [
            np.round(rng.normal(1000 + rng.integers(0, 4) * 70, 140, int(rng.integers(1, 5))), 1)
            for _ in range(k)
        ]
        penalty = float(rng.choice([0.0, 20.0, 100.0, 300.0]))
        outcome = drift_algo.solve(blocks, penalty, min_shifts)
        expected_total, expected_boundaries = _brute(blocks, penalty, min_shifts)
        assert abs(outcome['total'] - expected_total) <= 1e-6
        assert outcome['boundaries'] == expected_boundaries


def test_penalty_controls_boundaries():
    blocks = [np.array([1000.0] * 4) for _ in range(3)] + [np.array([2000.0] * 4) for _ in range(3)]
    loose = drift_algo.solve(blocks, 20000.0)
    tight = drift_algo.solve(blocks, 0.0)
    assert loose['boundaries'] == []
    assert tight['boundaries'] == [2]
    # 不分段：24 件样本到中位数 1500 的绝对差共 12000
    assert loose['total'] == 12000.0
    assert tight['total'] == 0.0


def test_min_shifts_enforced():
    blocks = [np.array([float(i)]) for i in range(5)]
    result = drift_algo.solve(blocks, 0.0)
    assert all(segment['shift_count'] >= 3 for segment in result['segments'])
    with pytest.raises(drift_algo.DriftError):
        drift_algo.solve(blocks[:2], 0.0)


def test_rejects_empty_or_nonfinite_shift():
    with pytest.raises(drift_algo.DriftError):
        drift_algo.solve([np.array([1.0]), np.array([]), np.array([2.0])], 0.0)
    with pytest.raises(drift_algo.DriftError):
        drift_algo.solve([np.array([1.0]), np.array([np.nan]), np.array([2.0])], 0.0)
    with pytest.raises(drift_algo.DriftError):
        drift_algo.solve([np.array([1.0])] * 3, -1.0)


def _make_source(db, key='TX-TEST-01'):
    src = Source(key=key, name='测试来源', filename='test.csv', description='')
    db.add(src)
    db.flush()
    return src


def _make_run(db, source, shifts, label_count=None):
    """shifts: [(name, order, [(reference, centroid), ...])]；每个班次样本带有同一故障标签。"""
    records = []
    for name, order, samples in shifts:
        for reference, centroid in samples:
            db.add(Record(source_id=source.id, reference=reference,
                          values={'rms': 1.0, 'centroid': centroid, 'kurtosis': 2.0, 'label': '轴承磨损'}))
            records.append({'reference': reference, 'values': {'centroid': centroid, 'label': '轴承磨损'}})
    db.flush()
    run = Run(source_id=source.id, name='研究', state='completed', parameters={'folds': 3},
              snapshot={'source': {'id': source.id, 'name': source.name, 'filename': source.filename},
                        'records': records},
              created_at=now())
    db.add(run)
    db.commit()
    return run


def _plan(db, source, shifts, note=''):
    payload = PlanInput(source_id=source.id, note=note, shifts=[
        ShiftInput(name=name, order=order, references=[ref for ref, _ in samples])
        for name, order, samples in shifts
    ])
    return drift_service.create_plan(db, USER, payload)


def _shifts_payload(shifts):
    return shifts


def test_plan_revision_is_appended_and_validated(db):
    source = _make_source(db)
    shifts = [('白班', 1, [('R-1', 1000.0), ('R-2', 1010.0)]),
              ('中班', 2, [('R-3', 1200.0)]),
              ('夜班', 3, [('R-4', 1400.0)])]
    _make_run(db, source, shifts)
    first = _plan(db, source, shifts)
    assert first['revision'] == 1
    assert first['reference_count'] == 4
    second = _plan(db, source, shifts, note='更正')
    assert second['revision'] == 2
    plans = drift_service.list_plans(db, source.id)
    assert [p['revision'] for p in plans] == [2, 1]
    # 旧版本仍完整保留
    assert drift_service.get_plan(db, first['id'])['reference_count'] == 4


def test_plan_rejects_unknown_reference_and_duplicates(db):
    source = _make_source(db, 'TX-TEST-02')
    _make_run(db, source, [('白班', 1, [('A-1', 1000.0)]),
                           ('中班', 2, [('A-2', 1100.0)]),
                           ('夜班', 3, [('A-3', 1200.0)])])
    with pytest.raises(Exception) as error:
        _plan(db, source, [('白班', 1, [('MISSING', 1000.0)]),
                           ('中班', 2, [('A-2', 1100.0)]),
                           ('夜班', 3, [('A-3', 1200.0)])])
    assert '不在该来源' in str(error.value)
    with pytest.raises(Exception) as error:
        _plan(db, source, [('白班', 1, [('A-1', 1.0), ('A-1', 2.0)]),
                           ('中班', 2, [('A-2', 1100.0)]),
                           ('夜班', 3, [('A-3', 1200.0)])])
    assert '重复归班' in str(error.value)


def _job(db, run, label='轴承磨损', penalty=100.0):
    return drift_service.create_job(db, USER, JobInput(run_id=run.id, label=label, penalty=penalty))


def _planned_fixture(db, key='TX-TEST-03'):
    source = _make_source(db, key)
    shifts = []
    for shift in range(6):
        samples = [(f'{key}-S{shift}-{i}', 1000.0 + 100.0 * (shift // 3) + i) for i in range(3)]
        shifts.append((f'第{shift + 1}班', shift + 1, samples))
    run = _make_run(db, source, shifts)
    plan = _plan(db, source, shifts)
    return source, run, plan


def test_job_requires_label_present_and_assigned(db):
    source = _make_source(db, 'TX-TEST-04')
    shifts = [('白班', 1, [('B-1', 1000.0)]), ('中班', 2, [('B-2', 1100.0)]),
              ('夜班', 3, [('B-3', 1200.0)])]
    run = _make_run(db, source, shifts)
    _plan(db, source, shifts)
    with pytest.raises(Exception) as error:
        _job(db, run, label='不存在的类型')
    assert '没有指定故障' in str(error.value)
    with pytest.raises(Exception) as error:
        _job(db, run, penalty=float('nan'))
    assert 'finite' in str(error.value).lower() or '罚分' in str(error.value)


def test_job_requires_every_matching_sample_assigned(db):
    source = _make_source(db, 'TX-TEST-05')
    shifts = [('白班', 1, [('C-1', 1000.0)]), ('中班', 2, [('C-2', 1100.0)]),
              ('夜班', 3, [('C-3', 1200.0)])]
    run = _make_run(db, source, shifts)
    _plan(db, source, [('白班', 1, [('C-1', 1000.0)]), ('中班', 2, [('C-2', 1100.0)])])
    with pytest.raises(Exception) as error:
        _job(db, run)
    assert '尚未归班' in str(error.value)


def test_compute_end_to_end_and_publish(db):
    source, run, plan = _planned_fixture(db)
    job = _job(db, run, penalty=100.0)
    job_id = job['id']
    drift_compute.execute(db, job_id)
    stored = db.get(DriftJob, job_id)
    assert stored.state == 'completed', stored.error
    assert stored.result['boundaries'] == [2]
    assert stored.result['shift_total'] == 6
    assert stored.result['sample_total'] == 18
    assert all(segment['median'] is not None for segment in stored.result['segments'])
    published = drift_service.latest_result(db, source.id, '轴承磨损')
    assert published is not None and published.job_id == job_id


def test_retry_does_not_double_count(db):
    source, run, plan = _planned_fixture(db, 'TX-TEST-06')
    job = _job(db, run)
    job_id = job['id']
    drift_compute.execute(db, job_id)
    first = db.get(DriftJob, job_id).result
    # 重放同一个任务：状态已 completed，claim 失败，不重复计入
    drift_compute.execute(db, job_id)
    second = db.get(DriftJob, job_id).result
    assert first == second
    assert db.query(DriftSample).filter(DriftSample.job_id == job_id).count() == 18
    # 模拟失败的重复扫描：清掉样本后重新扫描，也不重复
    db.query(DriftSample).filter(DriftSample.job_id == job_id).delete()
    db.commit()
    references = [r['reference'] for r in run.snapshot['records']]
    drift_compute._scan_batch(db, db.get(DriftJob, job_id), references, '轴承磨损')
    drift_compute._scan_batch(db, db.get(DriftJob, job_id), references, '轴承磨损')
    assert db.query(DriftSample).filter(DriftSample.job_id == job_id).count() == 18


def test_resume_from_checkpoint_matches_uninterrupted(db):
    source, run, plan = _planned_fixture(db, 'TX-TEST-07')
    # 先算一次完好结果作为基准
    baseline_job = _job(db, run, penalty=50.0)
    drift_compute.execute(db, baseline_job['id'])
    baseline = db.get(DriftJob, baseline_job['id']).result

    # 新任务：扫描与分组后只算部分班次，模拟中断
    job = _job(db, run, penalty=50.0)
    job_id = job['id']
    stored = db.get(DriftJob, job_id)
    references = [r['reference'] for r in run.snapshot['records']]
    drift_compute._scan_batch(db, stored, references, '轴承磨损')
    drift_compute._build_shifts(db, stored, {'phase': 'group'})
    shifts, blocks = drift_compute._shifts_from_db(db, stored)
    # 手动只保留最后两个班次的成本
    for index in (len(blocks) - 1, len(blocks) - 2):
        values = np.sort(np.concatenate(blocks[index:]))
        deviation, median = drift_algo.segment_stat(values)
        db.add(DriftCost(job_id=job_id, shift_index=index, cost=float(deviation),
                         full_deviation=deviation, full_median=median, full_count=int(values.size)))
    db.commit()
    drift_compute.execute(db, job_id)
    resumed = db.get(DriftJob, job_id)
    assert resumed.state == 'completed', resumed.error
    assert resumed.result['boundaries'] == baseline['boundaries']
    assert abs(resumed.result['total'] - baseline['total']) <= 1e-6


def test_stale_running_job_is_requeued(db, redis_queue):
    source, run, plan = _planned_fixture(db, 'TX-TEST-08')
    job_id = _job(db, run)['id']
    stored = db.get(DriftJob, job_id)
    stored.state = 'running'
    stored.claim_token = 'dead-worker'
    stored.updated_at = '2000-01-01T00:00:00+00:00'
    db.commit()
    requeued = drift_service.requeue_stale(db, 60)
    assert requeued == [job_id]
    assert db.get(DriftJob, job_id).state == 'queued'
    assert redis_queue.llen('drift:queue') >= 1


def test_completed_job_is_not_claimed_twice(db):
    source, run, plan = _planned_fixture(db, 'TX-TEST-09')
    job_id = _job(db, run)['id']
    drift_compute.execute(db, job_id)
    assert drift_service.claim_job(db, job_id) is None


def _force_result(db, job_id, revision, shift_total=3, total=10.0):
    job = db.get(DriftJob, job_id)
    job.state = 'completed'
    job.plan_revision = revision
    segments = [{'start': 0, 'end': shift_total - 1, 'shift_count': shift_total,
                 'sample_count': 9, 'median': 1000.0, 'deviation': total,
                 'shift_orders': list(range(1, shift_total + 1)),
                 'shift_names': [f'第{i}班' for i in range(1, shift_total + 1)]}]
    job.result = {'total': total, 'deviation_total': total, 'penalty_total': 0.0,
                  'boundaries': [], 'segments': segments, 'penalty': 0.0, 'min_shifts': 3,
                  'shift_total': shift_total, 'sample_total': 9}
    db.commit()
    return job


def test_publish_keeps_newer_and_ignores_stale(db):
    source = _make_source(db, 'TX-TEST-10')
    shifts = [('白班', 1, [('D-1', 1000.0)]), ('中班', 2, [('D-2', 1100.0)]),
              ('夜班', 3, [('D-3', 1200.0)])]
    run = _make_run(db, source, shifts)
    plan = _plan(db, source, shifts)
    older = _force_result(db, _job(db, run)['id'], plan['revision'])
    newer = _force_result(db, _job(db, run)['id'], plan['revision'])
    assert older.id < newer.id
    # 新计算先发布，随后旧计算才完成并尝试发布
    assert drift_service.publish_result(db, newer, 'worker') is True
    assert drift_service.publish_result(db, older, 'worker') is False
    final = drift_service.latest_result(db, source.id, '轴承磨损')
    assert final.job_id == newer.id
    assert final.version == 1


def test_higher_revision_replaces_older_result(db):
    source = _make_source(db, 'TX-TEST-11')
    shifts = [('白班', 1, [('E-1', 1000.0)]), ('中班', 2, [('E-2', 1100.0)]),
              ('夜班', 3, [('E-3', 1200.0)])]
    run = _make_run(db, source, shifts)
    plan1 = _plan(db, source, shifts)
    plan2 = _plan(db, source, shifts, note='更正')
    job1 = _force_result(db, _job(db, run)['id'], plan1['revision'])
    job2 = _force_result(db, _job(db, run)['id'], plan2['revision'])
    assert drift_service.publish_result(db, job1, 'worker') is True
    assert drift_service.publish_result(db, job2, 'worker') is True
    final = drift_service.latest_result(db, source.id, '轴承磨损')
    assert final.job_id == job2.id
    assert final.plan_revision == plan2['revision']
    assert final.version == 2
    # 旧版本的计算再晚到也不覆盖
    assert drift_service.publish_result(db, job1, 'worker') is False
    assert drift_service.latest_result(db, source.id, '轴承磨损').job_id == job2.id


def test_publish_blocked_when_segments_incomplete(db):
    source = _make_source(db, 'TX-TEST-12')
    shifts = [('白班', 1, [('F-1', 1000.0)]), ('中班', 2, [('F-2', 1100.0)]),
              ('夜班', 3, [('F-3', 1200.0)])]
    run = _make_run(db, source, shifts)
    plan = _plan(db, source, shifts)
    job = _force_result(db, _job(db, run)['id'], plan['revision'])
    job.result['segments'][0]['median'] = None
    db.commit()
    assert drift_service.result_complete(job.result) is False
    assert drift_service.publish_result(db, job, 'worker') is False
    assert drift_service.latest_result(db, source.id, '轴承磨损') is None
    job = _force_result(db, _job(db, run)['id'], plan['revision'])
    job.result['segments'][0]['shift_count'] = 2
    db.commit()
    assert drift_service.result_complete(job.result) is False


def test_labels_endpoint_reports_unassigned(db):
    source = _make_source(db, 'TX-TEST-13')
    shifts = [('白班', 1, [('G-1', 1000.0)]), ('中班', 2, [('G-2', 1100.0)]),
              ('夜班', 3, [('G-3', 1200.0)])]
    run = _make_run(db, source, shifts)
    _plan(db, source, [('白班', 1, [('G-1', 1000.0)]), ('中班', 2, [('G-2', 1100.0)])])
    labels = drift_service.run_labels(db, run.id)
    entry = next(item for item in labels['labels'] if item['label'] == '轴承磨损')
    assert entry['count'] == 3
    assert entry['unassigned'] == 1


def test_scan_batch_in_chunks_for_large_input(db):
    source = _make_source(db, 'TX-TEST-14')
    shifts = []
    for shift in range(4):
        samples = [(f'H-{shift}-{i}', 900.0 + 50 * shift + i) for i in range(2500)]
        shifts.append((f'第{shift + 1}班', shift + 1, samples))
    run = _make_run(db, source, shifts)
    _plan(db, source, shifts)
    job_id = _job(db, run, penalty=10.0)['id']
    stored = db.get(DriftJob, job_id)
    references = [r['reference'] for r in run.snapshot['records']]
    added = drift_compute._scan_batch(db, stored, references, '轴承磨损')
    assert added == 10000
    assert db.query(DriftSample).filter(DriftSample.job_id == job_id).count() == 10000
