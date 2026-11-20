import json,time,logging
from redis import Redis
from sqlalchemy import select
from .config.database import Session
from .config.settings import REDIS_URL,ARTIFACT_DIR
from .models.run import Run
from .models import source,record,user,note,audit,drift
from .services.algorithm import calculate
from .services.audit import log
from .services.export import csv_response
from .services import drift_compute
queue=Redis.from_url(REDIS_URL)
ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)
def execute(id):
    with Session() as db:
        r=db.get(Run,id)
        if not r or r.state!='queued':return
        r.state='running';db.commit()
        try:
            result=calculate(r.snapshot['records'],r.parameters)
            json.dumps(result,allow_nan=False)
            r.result=result;r.state='completed'
            response=csv_response(result,'run.csv')
            (ARTIFACT_DIR/f'run-{r.id}.csv').write_bytes(response.body)
            log(db,'worker','run.completed',r.id)
        except Exception as e:
            logging.exception('analysis failed');r.state='failed';r.error=str(e)[:2000]
            log(db,'worker','run.failed',r.id,r.error)
        db.commit()
def execute_drift(id):
    with Session() as db:
        drift_compute.execute(db,id)
def main():
    last_sweep=0.0
    while True:
        (ARTIFACT_DIR/'heartbeat').write_text(str(time.time()))
        if time.time()-last_sweep>=30:
            last_sweep=time.time()
            try:
                with Session() as db:drift_compute.requeue_stale_jobs(db)
            except Exception:logging.exception('drift stale sweep failed; worker continues')
        value=queue.blpop(['drift:queue','analysis:queue'],timeout=2)
        if not value:continue
        kind,raw=value
        if kind==b'drift:queue':
            try:message=json.loads(raw)
            except Exception:logging.exception('bad drift message');continue
            try:execute_drift(int(message['id']))
            except Exception:logging.exception('drift task failed; worker continues')
        else:
            execute(int(raw))
if __name__=='__main__':main()
