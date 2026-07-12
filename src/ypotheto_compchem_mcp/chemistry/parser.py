import cclib
import numpy as np
from pathlib import Path
from rdkit import Chem
from ypotheto_compchem_mcp.chemistry.schemas import QMResultSchema, AtomChargeSchema

HARTREE_TO_EV = 27.211386245988

def parse_qm_log_with_cclib(log_path: Path) -> QMResultSchema:
    """
    Parse a quantum chemistry output log file using cclib and return normalized results.
    """
    try:
        data = cclib.io.ccread(str(log_path))
    except Exception as e:
        return QMResultSchema(
            ok=False,
            energy_ev=0.0,
            energy_hartree=0.0,
            warnings=[{"type": "PARSE_ERROR", "message": f"cclib failed to parse log: {str(e)}"}]
        )

    if data is None:
        return QMResultSchema(
            ok=False,
            energy_ev=0.0,
            energy_hartree=0.0,
            warnings=[{"type": "PARSE_ERROR", "message": "cclib returned empty parsed data."}]
        )

    if not hasattr(data, "scfenergies") or len(data.scfenergies) == 0:
        return QMResultSchema(
            ok=False,
            energy_ev=0.0,
            energy_hartree=0.0,
            warnings=[{"type": "SCF_FAILED", "message": "No SCF energies found in log file."}]
        )

    energy_ev = float(data.scfenergies[-1])
    energy_hartree = energy_ev / HARTREE_TO_EV

    table = Chem.GetPeriodicTable()
    atom_symbols = []
    for z in data.atomnos:
        try:
            symbol = table.GetElementSymbol(int(z))
        except Exception:
            symbol = f"X{z}"
        atom_symbols.append(symbol)

    coords = []
    if hasattr(data, "atomcoords") and len(data.atomcoords) > 0:
        coords = [[float(x) for x in pos] for pos in data.atomcoords[-1]]

    dipole = [0.0, 0.0, 0.0]
    if hasattr(data, "moments") and len(data.moments) > 1:
        try:
            dipole = [float(x) for x in data.moments[1]]
        except Exception:
            pass

    homo_ev = 0.0
    lumo_ev = 0.0
    gap = 0.0
    if hasattr(data, "moenergies") and hasattr(data, "nocc"):
        try:
            mo_energies = data.moenergies
            nocc = data.nocc
            if isinstance(nocc, (list, tuple, np.ndarray)) and len(nocc) == 2:
                homo_a = mo_energies[0][nocc[0] - 1]
                lumo_a = mo_energies[0][nocc[0]]
                homo_b = mo_energies[1][nocc[1] - 1]
                lumo_b = mo_energies[1][nocc[1]]
                homo_ev = max(homo_a, homo_b)
                lumo_ev = min(lumo_a, lumo_b)
            else:
                homo_ev = mo_energies[0][nocc - 1]
                lumo_ev = mo_energies[0][nocc]
            gap = max(0.0, lumo_ev - homo_ev)
        except Exception:
            pass

    charges = []
    if hasattr(data, "atomcharges") and data.atomcharges is not None:
        for charge_type in ["mulliken", "lowdin"]:
            if charge_type in data.atomcharges:
                try:
                    for i, val in enumerate(data.atomcharges[charge_type]):
                        charges.append(AtomChargeSchema(
                            index=i,
                            element=atom_symbols[i] if i < len(atom_symbols) else "X",
                            charge=float(val)
                        ))
                    break
                except Exception:
                    pass

    return QMResultSchema(
        energy_ev=energy_ev,
        energy_hartree=energy_hartree,
        dipole_moment_debye=dipole,
        homo_ev=float(homo_ev),
        lumo_ev=float(lumo_ev),
        homo_lumo_gap_ev=float(gap),
        mulliken_charges=charges,
        atom_symbols=atom_symbols,
        coordinates_angstrom=coords
    )
