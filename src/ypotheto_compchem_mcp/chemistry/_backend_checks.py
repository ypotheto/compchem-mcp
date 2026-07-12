"""Shared availability checks for optional external quantum-chemistry backends."""
import shutil

from ypotheto_compchem_mcp.errors import BackendUnavailableError

XTB_INSTALL_HINT = "Install the xtb binary and the xtb-python ASE calculator, or rerun with method='DFT'."


def require_xtb_calculator(context: str):
    """
    Return a configured ASE XTB (GFN2-xTB) calculator.

    Raises BackendUnavailableError with a consistent message/hint if the xtb
    binary is not on PATH. `context` is a short phrase describing the calling
    operation (e.g. "transition-state search"), used in the error message.
    """
    if not shutil.which("xtb"):
        raise BackendUnavailableError(
            f"xTB backend is not available for {context}.",
            hint=XTB_INSTALL_HINT,
        )
    from ase.calculators.xtb import XTB
    return XTB(method="GFN2-xTB")
