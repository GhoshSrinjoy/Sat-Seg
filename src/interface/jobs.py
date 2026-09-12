"""Bounded background job queue; cancellation is checked between work units."""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="geo-analysis")
_lock = threading.Lock()
_jobs = {}


class Cancelled(RuntimeError):
    pass


def submit(action, cleanup=None):
    with _lock:
        if sum(j["status"] in {"queued", "running"} for j in _jobs.values()) >= 3:
            raise ValueError("The analysis queue is full. Wait for a job to finish or cancel it.")
        for key, job in list(_jobs.items()):
            if job["status"] in {"completed", "failed", "cancelled"} and time.time()-job["updated"] > 3600:
                del _jobs[key]
        identifier = uuid.uuid4().hex
        _jobs[identifier] = {"job_id": identifier, "status": "queued", "progress": 0, "message": "Queued",
                             "cancel": False, "updated": time.time()}

    def progress(message, value):
        with _lock:
            job = _jobs[identifier]
            if job["cancel"]:
                raise Cancelled("Analysis cancelled.")
            job.update(message=message, progress=value, updated=time.time())

    def run():
        try:
            progress("Starting analysis", .01)
            with _lock:
                _jobs[identifier]["status"] = "running"
            result = action(progress)
            progress("Complete", 1)
            with _lock:
                _jobs[identifier].update(status="completed", result=result)
        except Cancelled as error:
            with _lock:
                _jobs[identifier].update(status="cancelled", message=str(error))
        except Exception as error:
            import logging
            logging.getLogger(__name__).exception("Analysis job failed")
            with _lock:
                _jobs[identifier].update(status="failed", message=str(error) or type(error).__name__)
        finally:
            if cleanup:
                cleanup()
    _executor.submit(run)
    return {"job_id": identifier, "status": "queued"}


def get_job(identifier):
    with _lock:
        job = _jobs.get(identifier)
        return {k: v for k, v in job.items() if k != "cancel"} if job else None


def cancel_job(identifier):
    with _lock:
        if identifier not in _jobs:
            raise ValueError("Unknown analysis job.")
        job = _jobs[identifier]
        if job["status"] in {"queued", "running"}:
            job.update(cancel=True, message="Cancelling after the current operation…")
    return {"message": "Cancellation requested."}


def active():
    with _lock:
        return any(j["status"] in {"queued", "running"} for j in _jobs.values())
