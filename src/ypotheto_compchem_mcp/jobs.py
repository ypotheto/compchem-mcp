import uuid
import time
import logging
import threading
import json
import inspect
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from ypotheto_compchem_mcp.workspace import workspace_manager
from ypotheto_compchem_mcp.errors import CompchemError

def _job_error_from_exception(e: Exception) -> Dict[str, Any]:
    """
    Background jobs call engine functions directly, bypassing mcp_tool_decorator.
    Preserve a CompchemError's typed code/hint (e.g. BACKEND_UNAVAILABLE) instead
    of collapsing every failure into a generic INTERNAL_JOB_ERROR - otherwise an
    async caller can't distinguish "backend not installed" from "actually crashed".
    """
    if isinstance(e, CompchemError):
        return {
            "code": e.code,
            "message": str(e),
            "hint": e.hint or "Please review log files or retry."
        }
    return {
        "code": "INTERNAL_JOB_ERROR",
        "message": str(e),
        "hint": "Please review log files or retry."
    }

# Import registry helper functions
_FUNCTIONS_REGISTRY = {}

def get_registered_function(name: str):
    if not _FUNCTIONS_REGISTRY:
        from ypotheto_compchem_mcp.chemistry.qm_engine import run_single_point_engine, optimize_geometry_engine
        from ypotheto_compchem_mcp.chemistry.vib_engine import run_vibrations_engine, simulate_ir_spectrum_engine
        from ypotheto_compchem_mcp.chemistry.md_engine import run_molecular_dynamics_engine
        from ypotheto_compchem_mcp.chemistry.xtb_engine import run_xtb_calculation_engine, run_conformer_search_engine
        from ypotheto_compchem_mcp.chemistry.qm_engine import run_pyscf_properties_engine
        from ypotheto_compchem_mcp.chemistry.ensemble_pipeline import run_ensemble_thermochemistry_engine
        from ypotheto_compchem_mcp.chemistry.thermo_engine import run_mixture_flash_engine, run_reactor_kinetics_engine
        from ypotheto_compchem_mcp.chemistry.polymer_engine import pack_amorphous_cell_engine, run_lammps_simulation_engine
        from ypotheto_compchem_mcp.chemistry.kinetics_engine import run_transition_state_search_engine, run_neb_calculation_engine
        from ypotheto_compchem_mcp.chemistry.periodic_engine import run_periodic_dft_engine
        from ypotheto_compchem_mcp.chemistry.mlff_engine import run_mlff_optimization_engine, run_mlff_molecular_dynamics_engine
        
        _FUNCTIONS_REGISTRY["run_single_point_engine"] = run_single_point_engine
        _FUNCTIONS_REGISTRY["optimize_geometry_engine"] = optimize_geometry_engine
        _FUNCTIONS_REGISTRY["run_vibrations_engine"] = run_vibrations_engine
        _FUNCTIONS_REGISTRY["simulate_ir_spectrum_engine"] = simulate_ir_spectrum_engine
        _FUNCTIONS_REGISTRY["run_molecular_dynamics_engine"] = run_molecular_dynamics_engine
        _FUNCTIONS_REGISTRY["run_xtb_calculation_engine"] = run_xtb_calculation_engine
        _FUNCTIONS_REGISTRY["run_conformer_search_engine"] = run_conformer_search_engine
        _FUNCTIONS_REGISTRY["run_pyscf_properties_engine"] = run_pyscf_properties_engine
        _FUNCTIONS_REGISTRY["run_ensemble_thermochemistry_engine"] = run_ensemble_thermochemistry_engine
        _FUNCTIONS_REGISTRY["run_mixture_flash_engine"] = run_mixture_flash_engine
        _FUNCTIONS_REGISTRY["run_reactor_kinetics_engine"] = run_reactor_kinetics_engine
        _FUNCTIONS_REGISTRY["pack_amorphous_cell_engine"] = pack_amorphous_cell_engine
        _FUNCTIONS_REGISTRY["run_lammps_simulation_engine"] = run_lammps_simulation_engine
        _FUNCTIONS_REGISTRY["run_transition_state_search_engine"] = run_transition_state_search_engine
        _FUNCTIONS_REGISTRY["run_neb_calculation_engine"] = run_neb_calculation_engine
        _FUNCTIONS_REGISTRY["run_periodic_dft_engine"] = run_periodic_dft_engine
        _FUNCTIONS_REGISTRY["run_mlff_optimization_engine"] = run_mlff_optimization_engine
        _FUNCTIONS_REGISTRY["run_mlff_molecular_dynamics_engine"] = run_mlff_molecular_dynamics_engine
        
    return _FUNCTIONS_REGISTRY.get(name)

class HeartbeatThread(threading.Thread):
    """Periodically updates the job lease in the database to prevent other workers from claiming it."""
    def __init__(self, job_id: str, interval: int = 15):
        super().__init__(daemon=True)
        self.job_id = job_id
        self.interval = interval
        self._stop_event = threading.Event()
        
    def stop(self):
        self._stop_event.set()
        
    def run(self):
        from ypotheto_compchem_mcp.database import get_connection
        while not self._stop_event.wait(self.interval):
            conn = get_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "UPDATE compchem.jobs SET lease_timeout = NOW() + INTERVAL '2 minutes' WHERE job_id = %s;",
                        (self.job_id,)
                    )
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as e:
                    logging.error(f"Heartbeat update failed for job {self.job_id}: {str(e)}")

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
        self._stop_event = threading.Event()
        self.polling_threads = []
        self._workers_started = False

    def start_workers(self):
        """Starts the background database polling workers if configured."""
        with self._lock:
            if self._workers_started:
                return
            from ypotheto_compchem_mcp.config import settings
            if settings.database_url:
                self._reclaim_crashed_jobs()
                for i in range(min(2, self.executor._max_workers)):
                    t = threading.Thread(target=self._worker_loop, name=f"db-queue-worker-{i}", daemon=True)
                    t.start()
                    self.polling_threads.append(t)
            self._workers_started = True

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

    def _load_job_from_db(self, workspace_id: str, job_id: str) -> Optional[JobState]:
        """Load job state from PostgreSQL."""
        from ypotheto_compchem_mcp.database import get_connection
        conn = get_connection()
        if not conn:
            return None
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT job_id, workspace_id, status, progress_message, estimated_time_seconds, 
                       started_at, finished_at, results, warnings, error
                FROM compchem.jobs 
                WHERE workspace_id = %s AND job_id = %s;
                """,
                (workspace_id, job_id)
            )
            row = cur.fetchone()
            if not row:
                return None
            job_id, workspace_id, status, progress_message, estimated_time_seconds, started_at, finished_at, results, warnings, error = row
            
            job = JobState(
                job_id=job_id,
                workspace_id=workspace_id,
                estimated_time=estimated_time_seconds,
                status=status,
                progress_message=progress_message
            )
            
            if started_at:
                if isinstance(started_at, datetime.datetime):
                    job.start_time = started_at.timestamp()
            if finished_at:
                if isinstance(finished_at, datetime.datetime):
                    job.end_time = finished_at.timestamp()
                    
            job.results = results or {}
            job.warnings = warnings or []
            job.interpretation = job.results.get("interpretation", "")
            job.artifacts = job.results.get("artifacts", [])
            job.error = error
            
            cur.close()
            conn.close()
            return job
        except Exception as e:
            logging.error(f"Failed to load job {job_id} from DB: {str(e)}", exc_info=True)
            return None

    def get_job(self, workspace_id: str, job_id: str) -> Optional[JobState]:
        """Retrieve job state. Checks database first if configured, otherwise falls back to memory/disk."""
        from ypotheto_compchem_mcp.config import settings
        if settings.database_url:
            return self._load_job_from_db(workspace_id, job_id)
            
        with self._lock:
            job = self.jobs.get(job_id)
        if job and job.workspace_id == workspace_id:
            return job
        return self._load_job_from_disk(workspace_id, job_id)

    def _reclaim_crashed_jobs(self):
        from ypotheto_compchem_mcp.database import get_connection
        conn = get_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE compchem.jobs
                SET status = 'failed', finished_at = NOW(), 
                    error = '{"code": "SERVER_SHUTDOWN", "message": "Worker crashed or server restarted during job execution."}'::jsonb,
                    progress_message = 'Calculation failed due to server restart.'
                WHERE status = 'running' AND (lease_timeout IS NULL OR lease_timeout < NOW());
                """
            )
            conn.commit()
            cur.close()
            conn.close()
            logging.info("Reclaimed crashed/orphaned background jobs.")
        except Exception as e:
            if hasattr(e, "pgcode") and e.pgcode == "42P01":
                logging.info("Database tables not initialized yet during crashed job reclamation.")
            else:
                logging.error(f"Failed to reclaim crashed background jobs: {str(e)}", exc_info=True)

    def _claim_next_job_with_conn(self, conn) -> Optional[Dict[str, Any]]:
        job_info = None
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE compchem.jobs 
                SET status = 'running', started_at = NOW(), lease_timeout = NOW() + INTERVAL '2 minutes'
                WHERE job_id = (
                    SELECT job_id FROM compchem.jobs 
                    WHERE status = 'queued' 
                    ORDER BY created_at ASC 
                    LIMIT 1 
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING job_id, workspace_id, func_name, args, kwargs;
                """
            )
            row = cur.fetchone()
            if row:
                job_id, workspace_id, func_name, args_json, kwargs_json = row
                job_info = {
                    "job_id": job_id,
                    "workspace_id": workspace_id,
                    "func_name": func_name,
                    "args": args_json,
                    "kwargs": kwargs_json
                }
            conn.commit()
            cur.close()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            if hasattr(e, "pgcode") and e.pgcode == "42P01":
                pass
            else:
                logging.error(f"Error claiming next job: {str(e)}", exc_info=True)
                raise e
        return job_info

    def _update_job_status(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress_message: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[Dict[str, str]]] = None,
        error: Optional[Dict[str, str]] = None,
        conn = None
    ):
        should_close = False
        if conn is None:
            from ypotheto_compchem_mcp.database import get_connection
            conn = get_connection()
            should_close = True
            
        if not conn:
            return
            
        try:
            cur = conn.cursor()
            updates = []
            params = []
            if status is not None:
                updates.append("status = %s")
                params.append(status)
                if status in ("completed", "failed"):
                    updates.append("finished_at = NOW()")
            if progress_message is not None:
                updates.append("progress_message = %s")
                params.append(progress_message)
            if results is not None:
                updates.append("results = %s")
                params.append(json.dumps(results))
            if warnings is not None:
                updates.append("warnings = %s")
                params.append(json.dumps(warnings))
            if error is not None:
                updates.append("error = %s")
                params.append(json.dumps(error))
                
            if updates:
                query = f"UPDATE compchem.jobs SET {', '.join(updates)} WHERE job_id = %s;"
                params.append(job_id)
                cur.execute(query, tuple(params))
                conn.commit()
            cur.close()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            logging.error(f"Failed to update job status in DB: {str(e)}", exc_info=True)
            if not should_close:
                raise e
        finally:
            if should_close:
                try:
                    conn.close()
                except Exception:
                    pass

    def _execute_job_with_conn(self, job_info: Dict[str, Any], conn):
        job_id = job_info["job_id"]
        workspace_id = job_info["workspace_id"]
        func_name = job_info["func_name"]
        args = job_info["args"] or []
        kwargs = job_info["kwargs"] or {}
        
        func = get_registered_function(func_name)
        if not func:
            self._update_job_status(
                job_id,
                status="failed",
                progress_message=f"Function {func_name} is not registered or found.",
                error={"code": "FUNCTION_NOT_FOUND", "message": f"Callable {func_name} is missing."},
                conn=conn
            )
            return
            
        heartbeat = HeartbeatThread(job_id)
        heartbeat.start()
        
        from ypotheto_compchem_mcp.workspace import current_workspace_id
        token_var = current_workspace_id.set(workspace_id)
        
        try:
            def progress_callback(msg: str):
                self._update_job_status(job_id, progress_message=msg, conn=conn)
                
            sig = inspect.signature(func)
            if "progress_callback" in sig.parameters:
                kwargs["progress_callback"] = progress_callback
                
            envelope = func(*args, **kwargs)
            
            if envelope.get("ok", True):
                self._update_job_status(
                    job_id,
                    status="completed",
                    progress_message="Calculation completed successfully.",
                    results=envelope,
                    warnings=envelope.get("warnings", []),
                    error=None,
                    conn=conn
                )
            else:
                self._update_job_status(
                    job_id,
                    status="failed",
                    progress_message="Calculation failed.",
                    error=envelope.get("error", {"code": "COMPUTE_ERROR", "message": "Failed"}),
                    conn=conn
                )
        except Exception as e:
            logging.error(f"Error executing background job {job_id}: {str(e)}", exc_info=True)
            self._update_job_status(
                job_id,
                status="failed",
                progress_message=f"Calculation encountered an error: {str(e)}",
                error=_job_error_from_exception(e),
                conn=conn
            )
            raise e
        finally:
            current_workspace_id.reset(token_var)
            heartbeat.stop()

    def _worker_loop(self):
        from ypotheto_compchem_mcp.database import get_connection
        conn = None
        while not self._stop_event.is_set():
            try:
                if conn is None or conn.closed != 0:
                    conn = get_connection()
                if conn:
                    job_to_run = self._claim_next_job_with_conn(conn)
                    if job_to_run:
                        self._execute_job_with_conn(job_to_run, conn)
                    else:
                        time.sleep(1.0)
                else:
                    time.sleep(2.0)
            except Exception as e:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                time.sleep(2.0)
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    def submit_job(
        self,
        workspace_id: str,
        func: Callable[..., Dict[str, Any]],
        estimated_time: int,
        *args,
        **kwargs
    ) -> JobState:
        """Submit a computational chemistry calculation task to run in the background."""
        self.start_workers()
        from ypotheto_compchem_mcp.database import get_connection
        conn = get_connection()
        if conn is not None:
            job_id = f"job_{uuid.uuid4().hex[:8]}"
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO compchem.jobs (job_id, workspace_id, status, progress_message, estimated_time_seconds, func_name, args, kwargs)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        job_id,
                        workspace_id,
                        "queued",
                        "Job queued in database.",
                        estimated_time,
                        getattr(func, "__name__", str(func)),
                        json.dumps(args),
                        json.dumps(kwargs)
                    )
                )
                conn.commit()
                cur.close()
                conn.close()
                
                job = JobState(job_id=job_id, workspace_id=workspace_id, estimated_time=estimated_time, status="queued", progress_message="Job queued in database.")
                return job
            except Exception as e:
                logging.error(f"Failed to submit job to DB queue: {str(e)}", exc_info=True)
                if conn:
                    conn.close()
                    
        # Fallback to local thread execution
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job = JobState(job_id=job_id, workspace_id=workspace_id, estimated_time=estimated_time)
        
        with self._lock:
            self.jobs[job_id] = job
            
        self._persist_job(job)
        
        def _run_wrapper():
            from ypotheto_compchem_mcp.workspace import current_workspace_id
            token_var = current_workspace_id.set(workspace_id)
            try:
                def progress_callback(msg: str):
                    job.progress_message = msg
                    self._persist_job(job)
                
                sig = inspect.signature(func)
                if "progress_callback" in sig.parameters:
                    kwargs["progress_callback"] = progress_callback
                
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
                job.error = _job_error_from_exception(e)
            finally:
                current_workspace_id.reset(token_var)
                self._persist_job(job)
                with self._lock:
                    self.jobs.pop(job_id, None)

        self.executor.submit(_run_wrapper)
        return job

    def stop(self):
        """Signals worker threads to stop and joins them."""
        self._stop_event.set()
        for t in self.polling_threads:
            t.join(timeout=3.0)
        self.executor.shutdown(wait=False)

job_manager = JobManager()
