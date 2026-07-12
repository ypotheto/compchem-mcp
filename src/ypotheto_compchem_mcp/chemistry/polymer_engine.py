import json
import logging
import uuid
import os
import shutil
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from ypotheto_compchem_mcp.workspace import workspace_manager
from ypotheto_compchem_mcp.chemistry.builder_engine import _get_molecules_dir, _load_index, _save_index, save_molecule_coords

logger = logging.getLogger(__name__)

# Check binary availability
PACKMOL_AVAILABLE = bool(shutil.which("packmol"))
LAMMPS_AVAILABLE = bool(shutil.which("lammps") or shutil.which("lmp") or shutil.which("lmp_serial") or shutil.which("lmp_mpi") or shutil.which("lmp_aes"))


def _parse_lammps_thermo_lines(lines: Iterable[str]) -> Optional[Dict[str, float]]:
    """
    Parse the final thermo row from LAMMPS output lines produced by
    `thermo_style custom step temp press pe ke etotal density`, which LAMMPS
    prints with the header tokens Step/Temp/Press/PotEng/KinEng/TotEng/Density.
    Returns None if the expected header/rows cannot be found.

    `lines` is consumed once, in order, so callers can stream it from a file
    instead of holding the whole log in memory as one string - long production
    runs with frequent thermo output can otherwise produce a Python string
    many MB-GB in size.

    The generated input script issues an initial `run 0` (before the ensemble
    fix is even applied) followed by the real production `run {steps}` - and
    LAMMPS reprints the thermo header for every `run` invocation. Each time a
    new header line is seen, any previously accumulated table is discarded, so
    the result reflects the LAST `run`'s final row, not the pre-equilibration
    `run 0` row.
    """
    header_tokens = None
    last_row = None
    in_table = False

    for line in lines:
        tokens = line.split()
        if "PotEng" in tokens and "Density" in tokens:
            header_tokens = tokens
            last_row = None
            in_table = True
            continue
        if not in_table:
            continue
        if len(tokens) != len(header_tokens):
            in_table = False
            continue
        try:
            last_row = [float(t) for t in tokens]
        except ValueError:
            in_table = False

    if header_tokens is None or last_row is None:
        return None

    try:
        pe_idx = header_tokens.index("PotEng")
        density_idx = header_tokens.index("Density")
        return {
            "potential_energy_kcal_mol": last_row[pe_idx],
            "final_density_g_cm3": last_row[density_idx],
        }
    except (ValueError, IndexError):
        return None


def _parse_lammps_thermo_log(log_text: str) -> Optional[Dict[str, float]]:
    """Parse the final thermo row from a LAMMPS log held in memory as a string."""
    return _parse_lammps_thermo_lines(log_text.splitlines())


def _parse_lammps_thermo_log_file(log_path: str) -> Optional[Dict[str, float]]:
    """
    Parse the final thermo row from a LAMMPS log file on disk, streaming it
    line-by-line rather than reading it into memory as a single string.
    """
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return _parse_lammps_thermo_lines(f)
    except FileNotFoundError:
        return None

try:
    import MDAnalysis as mda
    MDANALYSIS_AVAILABLE = True
except ImportError:
    MDANALYSIS_AVAILABLE = False

def _get_monomers_dir(workspace_id: str) -> Path:
    """Get the monomers directory for the workspace."""
    path = workspace_manager.get_workspace_dir(workspace_id) / "monomers"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _get_monomer_index_file(workspace_id: str) -> Path:
    """Get the index file path for monomers."""
    return _get_monomers_dir(workspace_id) / "index.json"

def _load_monomer_index(workspace_id: str) -> Dict[str, Any]:
    """Load the monomer index from disk."""
    index_file = _get_monomer_index_file(workspace_id)
    if not index_file.exists():
        return {}
    try:
        return json.loads(index_file.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_monomer_index(workspace_id: str, index: Dict[str, Any]):
    """Save the monomer index to disk."""
    index_file = _get_monomer_index_file(workspace_id)
    index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")

def register_monomer_engine(
    workspace_id: str,
    smiles: str,
    name: str,
    head_idx: Optional[int] = None,
    tail_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Register a monomer repeat unit, setting up attachment points [1*] (head) and [2*] (tail).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid monomer SMILES string: '{smiles}'")

    # Ensure attachment points exist or graft them
    if head_idx is not None and tail_idx is not None:
        # Graft dummy atoms [1*] and [2*] onto specified indices
        editable = Chem.EditableMol(mol)
        
        # Add head dummy
        dummy_h = editable.AddAtom(Chem.Atom(0)) # atomic number 0 is dummy
        editable.AddBond(head_idx, dummy_h, Chem.BondType.SINGLE)
        
        # Add tail dummy
        dummy_t = editable.AddAtom(Chem.Atom(0))
        editable.AddBond(tail_idx, dummy_t, Chem.BondType.SINGLE)
        
        mol = editable.GetMol()
        
        # Set isotopes on the newly added dummy atoms
        # They will be the last two atoms added
        mol.GetAtomWithIdx(dummy_h).SetIsotope(1)
        mol.GetAtomWithIdx(dummy_t).SetIsotope(2)
        
    else:
        # Check if dummy atoms (*) are already present
        dummies = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummies) >= 2:
            # Set isotopes to designate head [1*] and tail [2*]
            dummies[0].SetIsotope(1)
            dummies[1].SetIsotope(2)
        elif len(dummies) == 1:
            raise ValueError("Monomer must have exactly 2 connection points; found only 1.")
        else:
            # Fallback: if no dummies are present and no indices specified, try to find polymerizable terminals
            # For simplicity, warn and require explicit attachment points
            raise ValueError(
                "No attachment points found. Please specify head_idx and tail_idx, "
                "or include connection points '*' in the SMILES string (e.g. '*CC(*)C' for propylene)."
            )

    Chem.SanitizeMol(mol)
    monomer_smiles = Chem.MolToSmiles(mol)
    monomer_id = f"mon_{uuid.uuid4().hex[:8]}"

    # Save to index
    meta = {
        "monomer_id": monomer_id,
        "name": name,
        "smiles": monomer_smiles,
        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "num_atoms": mol.GetNumAtoms()
    }
    index = _load_monomer_index(workspace_id)
    index[monomer_id] = meta
    _save_monomer_index(workspace_id, index)

    return meta

def build_polymer_chain_engine(
    workspace_id: str,
    monomer_id: str,
    dp: int,
    tacticity: str = "isotactic",
    name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a polymer chain of degree of polymerization (DP) by connecting registered
    monomer repeat units head-to-tail, minimizing the 3D cell, and saving it.
    """
    index = _load_monomer_index(workspace_id)
    if monomer_id not in index:
        raise FileNotFoundError(f"Monomer {monomer_id} not found in workspace.")
    
    monomer_smiles = index[monomer_id]["smiles"]
    monomer = Chem.MolFromSmiles(monomer_smiles)
    
    # 1. Setup chain building reaction
    # Connection: connects atom bonded to [2*] to atom bonded to [1*]
    rxn = AllChem.ReactionFromSmarts("[*:1]-[2*].[1*]-[*:2]>>[*:1]-[*:2]")
    
    chain = Chem.Mol(monomer)
    for step in range(dp - 1):
        products = rxn.RunReactants((chain, monomer))
        if not products or not products[0]:
            raise RuntimeError("Polymerization connection failed. Verify monomer head/tail attachment points.")
        chain = products[0][0]
        Chem.SanitizeMol(chain)

    # 2. Cap the chain: replace remaining [1*] and [2*] dummy atoms with Hydrogens (H)
    dummy_query = Chem.MolFromSmarts("[#0]") # Matches atomic number 0
    capped = Chem.ReplaceSubstructs(chain, dummy_query, Chem.MolFromSmiles("[H]"), replaceAll=True)[0]
    
    # Clean up to standard representation
    capped = Chem.RemoveHs(capped)
    Chem.SanitizeMol(capped)
    
    polymer_smiles = Chem.MolToSmiles(capped)
    
    # 3. Generate 3D Conformer coordinates
    mol_3d = Chem.AddHs(capped)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    embed_ok = AllChem.EmbedMolecule(mol_3d, params)
    if embed_ok < 0:
        # Fallback to random coordinates for large molecules
        AllChem.EmbedMolecule(mol_3d, randomCoords=True)
        
    if AllChem.MMFFHasAllMoleculeParams(mol_3d):
        AllChem.MMFFOptimizeMolecule(mol_3d)
        method = "MMFF94"
    else:
        AllChem.UFFOptimizeMolecule(mol_3d)
        method = "UFF"

    # 4. Export SVG
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    mol_2d = Chem.Mol(capped)
    rdDepictor.Compute2DCoords(mol_2d)
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    drawer.DrawMolecule(mol_2d)
    drawer.FinishDrawing()
    svg_data = drawer.GetDrawingText()

    # 5. Save back to molecules directory
    formula = CalcMolFormula(mol_3d)
    xyz_block = Chem.MolToXYZBlock(mol_3d)
    sdf_block = Chem.MolToMolBlock(mol_3d)
    
    polymer_id = f"mol_{uuid.uuid4().hex[:8]}"
    name_str = name or f"Polymer {index[monomer_id]['name']} (DP={dp})"

    meta = {
        "molecule_id": polymer_id,
        "name": name_str,
        "formula": formula,
        "smiles": polymer_smiles,
        "num_atoms": mol_3d.GetNumAtoms(),
        "is_polymer": True,
        "monomer_id": monomer_id,
        "dp": dp,
        "tacticity": tacticity,
        "method": f"Chain Construction ({method})"
    }
    save_molecule_coords(workspace_id, polymer_id, sdf_block, xyz_block, meta)

    return {
        "polymer_molecule_id": polymer_id,
        "name": name_str,
        "formula": formula,
        "smiles": polymer_smiles,
        "num_atoms": mol_3d.GetNumAtoms(),
        "dp": dp,
        "svg_data": svg_data
    }


def pack_amorphous_cell_engine(
    workspace_id: str,
    molecule_ids: List[str],
    counts: List[int],
    density_g_cm3: float = 0.9,
    box_size_angstrom: Optional[float] = None
) -> Dict[str, Any]:
    """
    Pack structures into a periodic simulation cell. Use packmol if available,
    otherwise fallback to a simple geometric packing algorithm.
    """
    workspace_dir = workspace_manager.get_workspace_dir(workspace_id)
    xyz_contents = []
    for mol_id in molecule_ids:
        xyz_path = workspace_dir / "molecules" / f"{mol_id}.xyz"
        if not xyz_path.exists():
            raise FileNotFoundError(f"Molecule {mol_id} coordinates not found.")
        xyz_contents.append(xyz_path.read_text(encoding="utf-8"))

    # Estimate box size if not provided
    if box_size_angstrom is None:
        total_mw = 0.0
        for xyz_text, count in zip(xyz_contents, counts):
            lines = xyz_text.strip().split("\n")
            if len(lines) < 3:
                continue
            mw = 0.0
            for line in lines[2:]:
                parts = line.split()
                if parts:
                    sym = parts[0]
                    mw += {"H": 1.0, "C": 12.0, "N": 14.0, "O": 16.0, "F": 19.0, "S": 32.0, "Cl": 35.5}.get(sym, 12.0)
            total_mw += mw * count
            
        na = 6.022e23
        mass_g = total_mw / na
        vol_cm3 = mass_g / density_g_cm3
        vol_ang3 = vol_cm3 * 1e24
        box_size_angstrom = float(vol_ang3 ** (1.0 / 3.0))
        
    box_size_angstrom = max(10.0, box_size_angstrom)
    packed_xyz = ""
    
    if PACKMOL_AVAILABLE:
        temp_dir = tempfile.mkdtemp()
        try:
            inp_lines = [
                "tolerance 2.0",
                "filetype xyz",
                "output packed.xyz",
                ""
            ]
            for idx, (xyz_text, count) in enumerate(zip(xyz_contents, counts)):
                xyz_path = os.path.join(temp_dir, f"struct_{idx}.xyz")
                with open(xyz_path, "w", encoding="utf-8") as f:
                    f.write(xyz_text)
                    
                inp_lines.extend([
                    f"structure struct_{idx}.xyz",
                    f"  number {count}",
                    f"  inside box 0. 0. 0. {box_size_angstrom} {box_size_angstrom} {box_size_angstrom}",
                    "end structure",
                    ""
                ])
                
            inp_path = os.path.join(temp_dir, "pack.inp")
            with open(inp_path, "w", encoding="utf-8") as f:
                f.write("\n".join(inp_lines))
                
            subprocess.run(["packmol"], stdin=open(inp_path), cwd=temp_dir, check=True, stdout=subprocess.DEVNULL)
            
            packed_path = os.path.join(temp_dir, "packed.xyz")
            if os.path.exists(packed_path):
                with open(packed_path, "r", encoding="utf-8") as f:
                    packed_xyz = f.read()
        except Exception as e:
            logger.warning(f"Packmol run failed: {str(e)}. Falling back to python packing.")
        finally:
            shutil.rmtree(temp_dir)
            
    if not packed_xyz:
        # Simple Python Fallback Packing
        packed_atoms = []
        for idx, (xyz_text, count) in enumerate(zip(xyz_contents, counts)):
            lines = xyz_text.strip().split("\n")
            if len(lines) < 3:
                continue
            natoms = int(lines[0])
            coords = []
            for line in lines[2:]:
                parts = line.split()
                if len(parts) >= 4:
                    coords.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
                    
            c_arr = np.array([[x[1], x[2], x[3]] for x in coords])
            center = np.mean(c_arr, axis=0)
            c_arr_centered = c_arr - center
            
            for c in range(count):
                pos = np.random.rand(3) * (box_size_angstrom - 4.0) + 2.0
                theta = np.random.rand() * 2.0 * np.pi
                phi = np.random.rand() * np.pi
                rot_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                rot_coords = np.dot(c_arr_centered, rot_matrix) + pos
                for i, (sym, _, _, _) in enumerate(coords):
                    packed_atoms.append((sym, rot_coords[i][0], rot_coords[i][1], rot_coords[i][2]))
                    
        xyz_out = [f"{len(packed_atoms)}", f"Amorphous Cell L={box_size_angstrom:.3f}"]
        for sym, x, y, z in packed_atoms:
            xyz_out.append(f"{sym} {x:.4f} {y:.4f} {z:.4f}")
        packed_xyz = "\n".join(xyz_out)
        
    lines = packed_xyz.strip().split("\n")
    num_atoms = int(lines[0])
    
    packed_id = f"cell_{uuid.uuid4().hex[:8]}"
    packed_name = f"Amorphous Cell Density={density_g_cm3} Box={box_size_angstrom:.2f}"
    
    meta = {
        "molecule_id": packed_id,
        "name": packed_name,
        "formula": "",
        "smiles": "",
        "num_atoms": num_atoms,
        "is_amorphous_cell": True,
        "box_size_angstrom": box_size_angstrom,
        "density_g_cm3": density_g_cm3
    }
    save_molecule_coords(workspace_id, packed_id, "", packed_xyz, meta)
    
    return {
        "ok": True,
        "packed_molecule_id": packed_id,
        "name": packed_name,
        "num_atoms": num_atoms,
        "box_size_angstrom": box_size_angstrom,
        "density_g_cm3": density_g_cm3
    }


def run_lammps_simulation_engine(
    workspace_id: str,
    packed_cell_id: str,
    steps: int = 1000,
    timestep_fs: float = 1.0,
    temperature_k: float = 300.0,
    pressure_atm: float = 1.0,
    ensemble: str = "npt"
) -> Dict[str, Any]:
    """
    Run classical MD simulation in LAMMPS. If not available, run using ASE fallback.
    """
    workspace_dir = workspace_manager.get_workspace_dir(workspace_id)
    xyz_path = workspace_dir / "molecules" / f"{packed_cell_id}.xyz"
    if not xyz_path.exists():
        raise FileNotFoundError(f"Packed cell {packed_cell_id} not found.")
    packed_cell_xyz = xyz_path.read_text(encoding="utf-8")
    
    box_size = 15.0
    lines = packed_cell_xyz.strip().split("\n")
    if len(lines) > 1 and "L=" in lines[1]:
        try:
            box_size = float(lines[1].split("L=")[1].split()[0])
        except Exception:
            pass
            
    traj_content = ""
    pot_energy = None
    final_density = None
    engine_used = None
    warnings: List[Dict[str, str]] = []

    if LAMMPS_AVAILABLE:
        temp_dir = tempfile.mkdtemp()
        try:
            atoms = []
            types_map = {}
            for line in lines[2:]:
                parts = line.split()
                if len(parts) >= 4:
                    sym = parts[0]
                    if sym not in types_map:
                        types_map[sym] = len(types_map) + 1
                    atoms.append((types_map[sym], float(parts[1]), float(parts[2]), float(parts[3])))
                    
            data_lines = [
                "LAMMPS Amorphous Cell Data file",
                f"{len(atoms)} atoms",
                f"{len(types_map)} atom types",
                f"0.0 {box_size} xlo xhi",
                f"0.0 {box_size} ylo yhi",
                f"0.0 {box_size} zlo zhi",
                "",
                "Masses",
                ""
            ]
            for sym, t_idx in types_map.items():
                mass = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}.get(sym, 12.011)
                data_lines.append(f"{t_idx} {mass}")
            data_lines.extend(["", "Atoms", ""])
            for i, (t_idx, x, y, z) in enumerate(atoms):
                data_lines.append(f"{i+1} {t_idx} {x:.4f} {y:.4f} {z:.4f}")
                
            with open(os.path.join(temp_dir, "cell.data"), "w", encoding="utf-8") as f:
                f.write("\n".join(data_lines))
                
            in_lines = [
                "units real",
                "atom_style atomic",
                "boundary p p p",
                "read_data cell.data",
                "pair_style lj/cut 8.0",
                "pair_coeff * * 0.1 3.5"
            ]
            for t_idx in range(1, len(types_map) + 1):
                in_lines.append(f"pair_coeff {t_idx} {t_idx} 0.1 3.5")
                
            in_lines.extend([
                "neighbor 2.0 bin",
                "neigh_modify delay 0 every 1 check yes",
                f"velocity all create {temperature_k} 12345",
                f"timestep {timestep_fs}",
                "thermo 100",
                "thermo_style custom step temp press pe ke etotal density",
                f"dump traj all xyz 100 trajectory.xyz",
                "run 0"
            ])
            if ensemble.lower() == "npt":
                in_lines.append(f"fix 1 all npt temp {temperature_k} {temperature_k} 100.0 iso {pressure_atm} {pressure_atm} 1000.0")
            elif ensemble.lower() == "nvt":
                in_lines.append(f"fix 1 all nvt temp {temperature_k} {temperature_k} 100.0")
            else:
                in_lines.append("fix 1 all nve")
                
            in_lines.append(f"run {steps}")
            
            with open(os.path.join(temp_dir, "sim.in"), "w", encoding="utf-8") as f:
                f.write("\n".join(in_lines))
                
            lmp_bin = shutil.which("lammps") or shutil.which("lmp") or shutil.which("lmp_serial") or shutil.which("lmp_mpi") or shutil.which("lmp_aes")
            log_path = os.path.join(temp_dir, "lammps.log")
            with open(log_path, "w", encoding="utf-8") as log_f:
                subprocess.run(
                    [lmp_bin, "-in", "sim.in"], cwd=temp_dir, check=True,
                    stdout=log_f, stderr=subprocess.PIPE, text=True
                )

            traj_path = os.path.join(temp_dir, "trajectory.xyz")
            if os.path.exists(traj_path):
                with open(traj_path, "r", encoding="utf-8") as f:
                    traj_content = f.read()
                engine_used = "lammps"
                thermo = _parse_lammps_thermo_log_file(log_path)
                if thermo is not None:
                    pot_energy = thermo["potential_energy_kcal_mol"]
                    final_density = thermo["final_density_g_cm3"]
                else:
                    logger.warning("LAMMPS ran successfully but thermo output could not be parsed.")
                    warnings.append({
                        "type": "parse_failure",
                        "message": "LAMMPS completed but its thermo output could not be parsed; "
                                   "potential_energy_kcal_mol and final_density_g_cm3 are unavailable."
                    })
        except Exception as e:
            stderr = getattr(e, "stderr", None)
            if stderr:
                logger.warning(f"LAMMPS run failed: {str(e)}. stderr: {stderr}. Falling back to ASE simulation.")
            else:
                logger.warning(f"LAMMPS run failed: {str(e)}. Falling back to ASE simulation.")
        finally:
            shutil.rmtree(temp_dir)

    if not traj_content:
        # ASE Fallback Simulation - uses a generic Lennard-Jones potential, NOT LAMMPS.
        # This produces a plausible-looking trajectory for pipeline/plumbing purposes only;
        # the energies and densities below are not physically meaningful for the requested system.
        from ase import Atoms
        from ase.calculators.lj import LennardJones
        from ase.md.langevin import Langevin
        from ase import units

        symbols = []
        positions = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                symbols.append(parts[0])
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

        atoms = Atoms(symbols=symbols, positions=positions, cell=[box_size, box_size, box_size], pbc=True)
        atoms.calc = LennardJones(sigma=3.5, epsilon=0.01)

        dyn = Langevin(atoms, timestep_fs * units.fs, temperature_K=temperature_k, friction=0.01)

        traj_out = [f"{len(atoms)}", f"ASE trajectory step 0"]
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            traj_out.append(f"{sym} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")

        for _ in range(5):
            dyn.run(steps // 5)
            traj_out.append(f"{len(atoms)}")
            traj_out.append(f"ASE trajectory step {_}")
            for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                traj_out.append(f"{sym} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")

        traj_content = "\n".join(traj_out)
        final_density = float(len(atoms) * 12.011 / (6.022e23 * (box_size * 1e-8) ** 3))
        pot_energy = float(atoms.get_potential_energy() * 23.06)
        engine_used = "ase-lj-fallback"
        warnings.append({
            "type": "fallback",
            "message": "LAMMPS unavailable — trajectory generated with a generic Lennard-Jones "
                       "potential. Energies and densities are NOT physically meaningful for this system."
        })

    # Save trajectory to artifact store
    traj_filename = f"{packed_cell_id}_trajectory.xyz"
    from ypotheto_compchem_mcp.artifacts import register_artifact
    traj_art = register_artifact(traj_filename, traj_content.encode("utf-8"), "trajectory", f"MD Trajectory for {packed_cell_id}")

    # final_density_g_cm3/potential_energy_kcal_mol can be None even when
    # engine_used == "lammps" (a real run whose thermo output failed to parse) -
    # callers must check `warnings` before treating these as trustworthy, not just
    # engine_used.
    results = {
        "final_density_g_cm3": final_density,
        "potential_energy_kcal_mol": pot_energy,
        "trajectory_file_url": traj_art.url,
        "engine_used": engine_used
    }

    if engine_used == "lammps":
        density_str = f"{final_density:.4f}" if final_density is not None else "unavailable"
        energy_str = f"{pot_energy:.2f}" if pot_energy is not None else "unavailable"
        interpretation = (
            f"LAMMPS simulation completed successfully.\n"
            f"Ensemble: {ensemble.upper()}, Steps: {steps}, Temperature: {temperature_k} K.\n"
            f"Final density = {density_str} g/cm3.\n"
            f"Potential Energy = {energy_str} kcal/mol."
        )
    else:
        interpretation = (
            f"LAMMPS was not available — ran a generic Lennard-Jones ASE simulation instead.\n"
            f"Ensemble: {ensemble.upper()}, Steps: {steps}, Temperature: {temperature_k} K.\n"
            f"These results (density = {final_density:.4f} g/cm3, "
            f"potential energy = {pot_energy:.2f} kcal/mol) are illustrative only and "
            f"are NOT physically meaningful for this specific system."
        )

    return {
        "ok": True,
        "results": results,
        "interpretation": interpretation,
        "artifacts": [traj_art],
        "warnings": warnings
    }


def analyze_md_trajectory_engine(
    workspace_id: str,
    trajectory_xyz: str
) -> Dict[str, Any]:
    """
    Parse packed cell MD trajectory and calculate Radius of Gyration (Rg), RDF, and MSD.
    """
    frames = []
    lines = trajectory_xyz.strip().split("\n")
    idx = 0
    while idx < len(lines):
        if not lines[idx].strip():
            idx += 1
            continue
        try:
            natoms = int(lines[idx].strip())
            frame_coords = []
            for k in range(natoms):
                parts = lines[idx + 2 + k].split()
                frame_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            frames.append(np.array(frame_coords))
            idx += natoms + 2
        except Exception:
            break
            
    if not frames:
        raise ValueError("Could not parse trajectory coordinates.")
        
    # 1. Radius of Gyration (Rg) over time
    rgs = []
    for f in frames:
        center = np.mean(f, axis=0)
        sq_dist = np.sum((f - center) ** 2, axis=1)
        rg = np.sqrt(np.mean(sq_dist))
        rgs.append(float(rg))
        
    # 2. Radial Distribution Function (RDF)
    final_frame = frames[-1]
    nat = len(final_frame)
    dists = []
    for i in range(min(nat, 100)):
        for j in range(i + 1, min(nat, 100)):
            d = np.linalg.norm(final_frame[i] - final_frame[j])
            dists.append(d)
            
    hist, bin_edges = np.histogram(dists, bins=20, range=(1.0, 10.0))
    rdf_vals = [float(x) for x in hist]
    rdf_bins = [float(x) for x in bin_edges[:-1]]
    
    # 3. MSD (Mean Squared Displacement)
    msd = []
    initial_frame = frames[0]
    for f in frames:
        sq_disp = np.sum((f - initial_frame) ** 2, axis=1)
        msd.append(float(np.mean(sq_disp)))
        
    results = {
        "radius_of_gyration_angstrom": rgs,
        "mean_squared_displacement_angstrom2": msd,
        "rdf": {
            "values": rdf_vals,
            "bins": rdf_bins
        }
    }
    
    interpretation = (
        f"Trajectory analysis completed successfully (frames analyzed: {len(frames)}).\n"
        f"Initial Rg = {rgs[0]:.2f} A, Final Rg = {rgs[-1]:.2f} A.\n"
        f"Final Mean Squared Displacement = {msd[-1]:.2f} A^2."
    )
    
    return {
        "ok": True,
        "results": results,
        "interpretation": interpretation
    }
