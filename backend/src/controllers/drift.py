from ..services import drift
class DriftController:
    list_plans = staticmethod(drift.list_plans)
    get_plan = staticmethod(drift.get_plan)
    create_plan = staticmethod(drift.create_plan)
    run_labels = staticmethod(drift.run_labels)
    list_jobs = staticmethod(drift.list_jobs)
    create_job = staticmethod(drift.create_job)
    retry_job = staticmethod(drift.retry_job)
    publish_job = staticmethod(drift.publish_job)
    list_results = staticmethod(drift.list_results)
    latest_result = staticmethod(drift.latest_result)

    get_job = staticmethod(drift.get_job)
