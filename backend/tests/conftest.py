import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 必须在导入 src.config.database 之前指向一次性 SQLite 文件，避免碰到开发库。
_TEST_DB_DIR = tempfile.mkdtemp(prefix='drift-tests-')
os.environ.setdefault('SEED_DEMO', 'false')
os.environ['DATABASE_URL'] = 'sqlite:///' + os.path.join(_TEST_DB_DIR, 'test.db')
os.environ.setdefault('ARTIFACT_DIR', os.path.join(_TEST_DB_DIR, 'artifacts'))

import fakeredis  # noqa: E402
import pytest  # noqa: E402

from src.config import database  # noqa: E402
from src.models import audit, drift, note, record, run, source, user  # noqa: E402
from src.services import drift as drift_service  # noqa: E402
from src.services import runs as runs_service  # noqa: E402


@pytest.fixture()
def db():
    database.Base.metadata.drop_all(database.engine)
    database.Base.metadata.create_all(database.engine)
    fake = fakeredis.FakeRedis()
    drift_service.queue = fake
    runs_service.queue = fake
    session = database.Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def redis_queue(db):
    return drift_service.queue


@pytest.fixture()
def api(db):
    """真实 FastAPI 应用 + 一次性 SQLite + fakeredis。"""
    from fastapi.testclient import TestClient

    from src.main import app
    from src.utils.security import hash_password, token_for

    users = {}
    for role in ('admin', 'analyst', 'viewer'):
        account = user.User(email=f'{role}@acoustic-bayes.demo',
                            password_hash=hash_password('Research2026!'), role=role)
        db.add(account)
        db.flush()
        users[role] = token_for(account)
    db.commit()
    with TestClient(app) as client:
        yield client, users, database.Session, drift_service.queue
