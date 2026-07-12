class CompchemError(Exception):
    """Base class for typed tool errors. `code` is a stable machine-readable string
    returned to the client; `hint` tells the calling LLM what to do next."""
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


class ValidationError(CompchemError):
    code = "INVALID_ARGUMENT"


class MoleculeNotFoundError(CompchemError):
    code = "MOLECULE_NOT_FOUND"


class JobNotFoundError(CompchemError):
    code = "JOB_NOT_FOUND"


class QuotaExceededError(CompchemError):
    code = "QUOTA_EXCEEDED"


class BackendUnavailableError(CompchemError):
    """Raised when a required external backend/binary/model is missing.
    Never substitute a toy potential and report success instead of raising this."""
    code = "BACKEND_UNAVAILABLE"


class CalculationFailedError(CompchemError):
    """Raised when a backend ran but the calculation itself failed/diverged/crashed."""
    code = "CALCULATION_FAILED"
