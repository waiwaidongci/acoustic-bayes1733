import os

import pytest


def _headers(token):
    return {'Authorization': f'Bearer {token}'}


def _seed_source(client, token):
    response = client.post('/sources', json={
        'name': '漂移测试批次', 'filename': 'drift.csv', 'description': '测试'
    }, headers=_headers(token))
    assert response.status_code == 200, response.text
    source = response.json()
    references = []
    for shift in range(6):
        records = []
        for index in range(3):
            reference = f'DR-{shift}-{index}'
            records.append({'reference': reference, 'values': {
                'rms': 1.0, 'centroid': 1000.0 + 120.0 * (shift // 3) + index,
                'kurtosis': 2.0, 'label': '轴承磨损',
            }})
            references.append(reference)
        response = client.post(f"/sources/{source['id']}/records", json={'records': records},
                               headers=_headers(token))
        assert response.status_code == 200, response.text
    return source, references


def _plan_payload(source, references):
    shifts = []
    for shift in range(6):
        shifts.append({'name': f'第{shift + 1}班', 'order': shift + 1,
                       'references': [ref for ref in references if ref.startswith(f'DR-{shift}-')]})
    return {'source_id': source['id'], 'note': '首版', 'shifts': shifts}


def test_viewer_cannot_write_drift(api):
    client, users, Session, _ = api
    source, references = _seed_source(client, users['analyst'])
    response = client.post('/drift/plans', json=_plan_payload(source, references),
                           headers=_headers(users['viewer']))
    assert response.status_code == 403
    assert client.get('/drift/plans', headers={}).status_code == 401


def test_full_flow_plan_job_worker_publish_and_read(api, monkeypatch):
    client, users, Session, fake = api
    from src.services import drift_compute

    source, references = _seed_source(client, users['analyst'])
    response = client.post('/drift/plans', json=_plan_payload(source, references),
                           headers=_headers(users['analyst']))
    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan['revision'] == 1

    response = client.post('/runs', json={
        'source_id': source['id'], 'name': '漂移研究', 'parameters': {'folds': 3}
    }, headers=_headers(users['analyst']))
    assert response.status_code == 200, response.text
    run = response.json()

    # 直接完成研究任务（worker 的常规算法不在本测试范围）
    from src.models.run import Run
    with Session() as db:
        stored = db.get(Run, run['id'])
        stored.state = 'completed'
        stored.result = {'summary': [], 'rows': [], 'columns': []}
        db.commit()

    labels = client.get(f"/drift/runs/{run['id']}/labels", headers=_headers(users['analyst']))
    assert labels.status_code == 200, labels.text
    assert labels.json()['labels'][0]['label'] == '轴承磨损'
    assert labels.json()['labels'][0]['unassigned'] == 0

    response = client.post('/drift/jobs', json={
        'run_id': run['id'], 'label': '轴承磨损', 'penalty': 100.0
    }, headers=_headers(users['analyst']))
    assert response.status_code == 200, response.text
    job = response.json()
    assert job['state'] == 'queued'
    assert fake.llen('drift:queue') == 1

    with Session() as db:
        drift_compute.execute(db, job['id'])

    detail = client.get(f"/drift/jobs/{job['id']}", headers=_headers(users['analyst']))
    assert detail.status_code == 200
    assert detail.json()['state'] == 'completed'
    assert detail.json()['result']['boundaries'] == [2]

    results = client.get(f"/drift/results?source_id={source['id']}", headers=_headers(users['viewer']))
    assert results.status_code == 200
    published = results.json()[0]
    assert published['label'] == '轴承磨损'
    assert published['result']['segments'][0]['median'] is not None
    assert published['job_id'] == job['id']


def test_retry_failed_job_requeues(api):
    client, users, Session, fake = api
    source, references = _seed_source(client, users['analyst'])
    client.post('/drift/plans', json=_plan_payload(source, references), headers=_headers(users['analyst']))
    response = client.post('/runs', json={
        'source_id': source['id'], 'name': '重试研究', 'parameters': {'folds': 3}
    }, headers=_headers(users['analyst']))
    run = response.json()
    from src.models.run import Run
    with Session() as db:
        stored = db.get(Run, run['id'])
        stored.state = 'completed'
        db.commit()
    job = client.post('/drift/jobs', json={
        'run_id': run['id'], 'label': '轴承磨损', 'penalty': 10.0
    }, headers=_headers(users['analyst'])).json()
    from src.models.drift import DriftJob
    with Session() as db:
        stored = db.get(DriftJob, job['id'])
        stored.state = 'failed'
        stored.error = '模拟中断'
        stored.claim_token = ''
        db.commit()
    response = client.post(f"/drift/jobs/{job['id']}/retry", headers=_headers(users['analyst']))
    assert response.status_code == 200, response.text
    assert response.json()['state'] == 'queued'
    assert fake.llen('drift:queue') == 2
    # 已完成任务不可重试
    with Session() as db:
        stored = db.get(DriftJob, job['id'])
        stored.state = 'completed'
        db.commit()
    response = client.post(f"/drift/jobs/{job['id']}/retry", headers=_headers(users['analyst']))
    assert response.status_code == 409


def test_publish_manual_rejects_stale(api):
    client, users, Session, fake = api
    from src.services import drift as drift_service
    from src.models.drift import DriftJob

    source, references = _seed_source(client, users['analyst'])
    first = client.post('/drift/plans', json=_plan_payload(source, references),
                        headers=_headers(users['analyst'])).json()
    second = client.post('/drift/plans', json={**_plan_payload(source, references), 'note': '更正'},
                         headers=_headers(users['analyst'])).json()
    assert second['revision'] == 2
    response = client.post('/runs', json={
        'source_id': source['id'], 'name': '发布研究', 'parameters': {'folds': 3}
    }, headers=_headers(users['analyst']))
    run = response.json()
    from src.models.run import Run
    with Session() as db:
        stored = db.get(Run, run['id'])
        stored.state = 'completed'
        db.commit()

    def force(job_id, revision):
        with Session() as db:
            job = db.get(DriftJob, job_id)
            job.state = 'completed'
            job.plan_revision = revision
            job.result = {
                'total': 1.0, 'deviation_total': 1.0, 'penalty_total': 0.0, 'boundaries': [],
                'segments': [{'start': 0, 'end': 5, 'shift_count': 6, 'sample_count': 18,
                              'median': 1000.0, 'deviation': 1.0,
                              'shift_orders': [1, 2, 3, 4, 5, 6],
                              'shift_names': [f'第{i}班' for i in range(1, 7)]}],
                'penalty': 0.0, 'min_shifts': 3, 'shift_total': 6, 'sample_total': 18,
            }
            db.commit()
        return job_id

    old_job = client.post('/drift/jobs', json={
        'run_id': run['id'], 'label': '轴承磨损', 'penalty': 10.0, 'plan_id': first['id']
    }, headers=_headers(users['analyst'])).json()
    new_job = client.post('/drift/jobs', json={
        'run_id': run['id'], 'label': '轴承磨损', 'penalty': 10.0, 'plan_id': second['id']
    }, headers=_headers(users['analyst'])).json()
    force(old_job['id'], first['revision'])
    force(new_job['id'], second['revision'])

    response = client.post(f"/drift/jobs/{new_job['id']}/publish", headers=_headers(users['analyst']))
    assert response.status_code == 200, response.text
    response = client.post(f"/drift/jobs/{old_job['id']}/publish", headers=_headers(users['analyst']))
    assert response.status_code == 409
    with Session() as db:
        published = drift_service.latest_result(db, source['id'], '轴承磨损')
        assert published.job_id == new_job['id']
        assert published.plan_revision == 2
        assert published.version == 1
