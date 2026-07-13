from typing import Any

from pydantic import BaseModel, Field


class AtomChargeSchema(BaseModel):
    index: int
    element: str
    charge: float

class QMResultSchema(BaseModel):
    ok: bool = True
    energy_ev: float = Field(..., description="Total SCF energy in electron-volts")
    energy_hartree: float = Field(..., description="Total SCF energy in Hartree")
    dipole_moment_debye: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    homo_ev: float = 0.0
    lumo_ev: float = 0.0
    homo_lumo_gap_ev: float = 0.0
    mulliken_charges: list[AtomChargeSchema] = Field(default_factory=list)
    atom_symbols: list[str] = Field(default_factory=list)
    coordinates_angstrom: list[list[float]] = Field(default_factory=list)
    forces_ev_angstrom: list[list[float]] | None = None
    warnings: list[dict[str, str]] = Field(default_factory=list)

class VibrationsResultSchema(BaseModel):
    ok: bool = True
    frequencies_cm1: list[float] = Field(default_factory=list)
    zero_point_energy_ev: float = 0.0
    zero_point_energy_kcal: float = 0.0
    imaginary_modes_count: int = 0
    thermochemistry: dict[str, Any] | None = None
    warnings: list[dict[str, str]] = Field(default_factory=list)
