from fastapi import APIRouter, Depends, Query
from ..config.database import db_session
from ..schemas.drift import JobInput, PlanInput
from ..services.auth import current_user, writer
from ..controllers.drift import DriftController as C

router = APIRouter()


@router.get('/drift/plans')
def plans(source_id: int | None = Query(default=None), db=Depends(db_session), u=Depends(current_user)):
    return C.list_plans(db, source_id)


@router.post('/drift/plans')
def create_plan(data: PlanInput, db=Depends(db_session), u=Depends(writer)):
    return C.create_plan(db, u, data)


@router.get('/drift/plans/{id}')
def plan(id: int, db=Depends(db_session), u=Depends(current_user)):
    return C.get_plan(db, id)


@router.get('/drift/runs/{id}/labels')
def run_labels(id: int, db=Depends(db_session), u=Depends(current_user)):
    return C.run_labels(db, id)


@router.get('/drift/jobs')
def jobs(db=Depends(db_session), u=Depends(current_user)):
    return C.list_jobs(db)


@router.post('/drift/jobs')
def create_job(data: JobInput, db=Depends(db_session), u=Depends(writer)):
    return C.create_job(db, u, data)


@router.get('/drift/jobs/{id}')
def job(id: int, db=Depends(db_session), u=Depends(current_user)):
    return C.get_job(db, id)


@router.post('/drift/jobs/{id}/retry')
def retry_job(id: int, db=Depends(db_session), u=Depends(writer)):
    return C.retry_job(db, u, id)


@router.post('/drift/jobs/{id}/publish')
def publish_job(id: int, db=Depends(db_session), u=Depends(writer)):
    return C.publish_job(db, u, id)


@router.get('/drift/results')
def results(source_id: int | None = Query(default=None), db=Depends(db_session), u=Depends(current_user)):
    return C.list_results(db, source_id)
