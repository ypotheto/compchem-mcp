import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

class EngineRunResult:
    def __init__(self, return_code: int, stdout: str, stderr: str, log_file: Path):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.log_file = log_file

class EngineRunner(ABC):
    @abstractmethod
    def run_command(
        self,
        workspace_dir: Path,
        job_id: str,
        command: List[str],
        input_files: Dict[str, str]
    ) -> EngineRunResult:
        pass

class LocalSubprocessRunner(EngineRunner):
    """Runs calculation locally via shell subprocess."""
    def run_command(
        self,
        workspace_dir: Path,
        job_id: str,
        command: List[str],
        input_files: Dict[str, str]
    ) -> EngineRunResult:
        job_dir = workspace_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Write inputs
        for filename, content in input_files.items():
            (job_dir / filename).write_text(content, encoding="utf-8")
            
        log_file = job_dir / "output.log"
        err_file = job_dir / "error.log"
        
        # 2. Execute process
        with open(log_file, "w", encoding="utf-8") as out, open(err_file, "w", encoding="utf-8") as err:
            proc = subprocess.Popen(
                command,
                cwd=str(job_dir),
                stdout=out,
                stderr=err,
                env=os.environ.copy()
            )
            proc.wait()
            
        return EngineRunResult(
            return_code=proc.returncode,
            stdout=log_file.read_text(encoding="utf-8", errors="ignore"),
            stderr=err_file.read_text(encoding="utf-8", errors="ignore"),
            log_file=log_file
        )

class DockerContainerRunner(EngineRunner):
    """Runs calculation in isolated Docker container."""
    def __init__(self, image_name: str = "ypotheto-compchem-mcp:latest"):
        self.image_name = image_name

    def run_command(
        self,
        workspace_dir: Path,
        job_id: str,
        command: List[str],
        input_files: Dict[str, str]
    ) -> EngineRunResult:
        job_dir = workspace_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Write inputs
        for filename, content in input_files.items():
            (job_dir / filename).write_text(content, encoding="utf-8")
            
        log_file = job_dir / "output.log"
        err_file = job_dir / "error.log"
        
        # Map local job directory to container workspace
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{job_dir.absolute().as_posix()}:/workspace",
            "-w", "/workspace",
            self.image_name
        ] + command
        
        # Execute Docker process
        with open(log_file, "w", encoding="utf-8") as out, open(err_file, "w", encoding="utf-8") as err:
            proc = subprocess.Popen(
                docker_cmd,
                stdout=out,
                stderr=err
            )
            proc.wait()
            
        return EngineRunResult(
            return_code=proc.returncode,
            stdout=log_file.read_text(encoding="utf-8", errors="ignore"),
            stderr=err_file.read_text(encoding="utf-8", errors="ignore"),
            log_file=log_file
        )

def get_engine_runner() -> EngineRunner:
    runner_type = os.getenv("COMPCHEM_ENGINE_RUNNER_TYPE", "subprocess").lower()
    if runner_type == "docker":
        return DockerContainerRunner()
    return LocalSubprocessRunner()
