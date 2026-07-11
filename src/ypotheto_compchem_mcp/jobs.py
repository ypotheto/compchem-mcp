import uuid
import time
import logging
import threading
import json
import inspect
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from ypotheto_compchem_mcp.workspace import workspace_manager

class JobState:
    def __init__(
        self,
        job_id: str,
        workspace_id: str,
        estimated_time: int = 10,
        status: str = "running",
        progress_message: str = "Job initialized."
    ):
        self.job_id = job_id
        self.workspace_id = workspace_id
        self.status = status
        self.estimated_time_seconds = estimated_time
        self.progress_message = progress_message
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.results: Dict[str, Any] = {}
        self.warnings: List[Dict[str, str]] = []
        self.interpretation: str = ""
        self.artifacts: List[Dict[str, str]] = []
        self.error: Optional[Dict[str, str]] = None

    @property
    def elapsed_time_seconds(self) -> int:
        end = self.end_time or time.time()
        return int(end - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "status": self.status,
            "estimated_time_seconds": self.estimated_time_seconds,
            "elapsed_time_seconds": self.elapsed_time_seconds,
            "progress_message": self.progress_message,
            "results": self.results,
            "warnings": self.warnings,
            "interpretation": self.interpretation,
            "artifacts": self.artifacts,
            "error": self.error
        }

class JobManager:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: Dict[str, JobState] = {}
        self._lock = threading.Lock()

    def _get_jobs_file(self, workspace_id: str) -> Path:
        return workspace_manager.get_workspace_dir(workspace_id) / "jobs.json"

    def _persist_job(self, job: JobState):
        """Save a single job's state to the workspace jobs file."""
        file_path = self._get_jobs_file(job.workspace_id)
        with self._lock:
            data = {}
            if file_path.exists():
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            data[job.job_id] = job.to_dict()
            file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_job_from_disk(self, workspace_id: str, job_id: str) -> Optional[JobState]:
        """Load job state from disk."""
        file_path = self._get_jobs_file(workspace_id)
        if not file_path.exists():
            return None
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            job_dict = data.get(job_id)
            if not job_dict:
                return None
                
            job = JobState(
                job_id=job_dict["job_id"],
                workspace_id=job_dict["workspace_id"],
                estimated_time=job_dict["estimated_time_seconds"],
                status=job_dict["status"],
                progress_message=job_dict["progress_message"]
            )
            job.results = job_dict.get("results", {})
            job.warnings = job_dict.get("warnings", [])
            job.interpretation = job_dict.get("interpretation", "")
            job.artifacts = job_dict.get("artifacts", [])
            job.error = job_dict.get("error")
            return job
        except Exception:
            return None

    def get_job(self, workspace_id: str, job_id: str) -> Optional[JobState]:
        """Retrieve job state. Checks memory first, then falls back to disk."""
        with self._lock:
            job = self.jobs.get(job_id)
        if job and job.workspace_id == workspace_id:
            return job
        # Fallback to disk (for server restarts / stateless design)
        return self._load_job_from_disk(workspace_id, job_id)

    def submit_job(
        self,
        workspace_id: str,
        func: Callable[..., Dict[str, Any]],
        estimated_time: int,
        *args,
        **kwargs
    ) -> JobState:
        """Submit a computational chemistry calculation task to run in the background."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job = JobState(job_id=job_id, workspace_id=workspace_id, estimated_time=estimated_time)
        
        with self._lock:
            self.jobs[job_id] = job
            
        self._persist_job(job)
        
        def _run_wrapper():
            # Set context variable inside the worker thread
            from ypotheto_compchem_mcp.workspace import current_workspace_id
            token_var = current_workspace_id.set(workspace_id)
            try:
                def progress_callback(msg: str):
                    job.progress_message = msg
                    self._persist_job(job)
                
                # Check if target function takes progress_callback
                sig = inspect.signature(func)
                if "progress_callback" in sig.parameters:
                    kwargs["progress_callback"] = progress_callback
                
                # Execute computational runner
                envelope = func(*args, **kwargs)
                
                job.end_time = time.time()
                if envelope.get("ok", True):
                    job.status = "completed"
                    job.progress_message = "Calculation completed successfully."
                    job.results = envelope.get("results", {})
                    job.warnings = envelope.get("warnings", [])
                    job.interpretation = envelope.get("interpretation", "")
                    job.artifacts = envelope.get("artifacts", [])
                else:
                    job.status = "failed"
                    job.progress_message = "Calculation failed."
                    job.error = envelope.get("error", {"code": "COMPUTE_ERROR", "message": "Failed"})
            except Exception as e:
                logging.error(f"Error executing background job {job_id}: {str(e)}", exc_info=True)
                job.end_time = time.time()
                job.status = "failed"
                job.progress_message = f"Calculation encountered an error: {str(e)}"
                job.error = {
                    "code": "INTERNAL_JOB_ERROR",
                    "message": str(e),
                    "hint": "Please review log files or retry."
                }
            finally:
                current_workspace_id.reset(token_var)
                self._persist_job(job)
                with self._lock:
                    self.jobs.pop(job_id, None)

        self.executor.submit(_run_wrapper)
        return job

job_manager = JobManager()
