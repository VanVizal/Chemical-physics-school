import os
import sys
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.calculators.mopac import MOPAC
import numpy as np
import pyscf

# =====================================================================
# 1. Automate MOPAC Path Configuration for this Conda Environment
# =====================================================================
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    mopac_bin = os.path.join(conda_prefix, 'bin', 'mopac')
    # Set the legacy ASE environment variable format
    os.environ['ASE_MOPAC_COMMAND'] = f"{mopac_bin} PREFIX.mop"
else:
    print("Warning: No active Conda environment detected. MOPAC might fail.")

# =====================================================================
# 2. Custom Lightweight PySCF Calculator for ASE
# =====================================================================
class PySCFCalculator(Calculator):
    implemented_properties = ['energy']
    
    def __init__(self, method='RHF', basis='6-31g', **kwargs):
        Calculator.__init__(self, **kwargs)
        self.method = method
        self.basis = basis

    def calculate(self, atoms=None, properties=['energy'], system_changes=['positions', 'numbers']):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Convert ASE layout to PySCF format strings
        xyz_coords = []
        for sym, pos in zip(self.atoms.get_chemical_symbols(), self.atoms.get_positions()):
            xyz_coords.append(f"{sym} {pos[0]} {pos[1]} {pos[2]}")
        atom_str = "; ".join(xyz_coords)
        
        # Initialize and build the PySCF Molecule instance
        mol = pyscf.gto.Mole()
        mol.atom = atom_str
        mol.basis = self.basis
        mol.verbose = 0
        mol.build()
        
        # Run Mean Field (HF / DFT) calculation based on setup
        if self.method.upper() == 'RHF':
            mf = pyscf.scf.RHF(mol)
        else:
            mf = pyscf.scf.KS(mol)
            mf.xc = self.method
            
        # Convert Hartree output to eV for ASE compatibility
        hartree_to_ev = 27.211386245988
        self.results['energy'] = mf.kernel() * hartree_to_ev

# =====================================================================
# 3. Execution Block
# =====================================================================
if __name__ == "__main__":
    print("Initializing test molecule (H2O)...")
    atoms = molecule('H2O')

    # Test PySCF Custom Calculator
    print("\n--- Running PySCF Test ---")
    try:
        atoms.calc = PySCFCalculator(method='RHF', basis='6-31g')
        energy_pyscf = atoms.get_potential_energy()
        print(f"PySCF Potential Energy: {energy_pyscf:.4f} eV")
    except Exception as e:
        print(f"PySCF Calculation Failed: {e}")

    # Test MOPAC Calculator (Semi-empirical PM7)
    print("\n--- Running MOPAC Test ---")
    try:
        atoms.calc = MOPAC(method='PM7')
        energy_mopac = atoms.get_potential_energy()
        print(f"MOPAC Potential Energy: {energy_mopac:.4f} eV")
    except Exception as e:
        print(f"MOPAC Calculation Failed:\n{e}")

    print("\nScript execution finished.")

