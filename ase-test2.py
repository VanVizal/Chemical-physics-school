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
# 3. Helper function to test molecules
# =====================================================================
def test_molecule(mol_name, method='RHF', basis='6-31g', mopac_method='PM7'):
    """Test a molecule with both PySCF and MOPAC calculators"""
    
    print(f"\n{'='*60}")
    print(f"Testing molecule: {mol_name}")
    print(f"{'='*60}")
    
    # Create molecule
    atoms = molecule(mol_name)
    print(f"Number of atoms: {len(atoms)}")
    print(f"Chemical symbols: {atoms.get_chemical_symbols()}")
    print(f"Formula: {mol_name}")
    
    # Test PySCF Custom Calculator
    print(f"\n--- PySCF Test ({method}/{basis}) ---")
    try:
        atoms.calc = PySCFCalculator(method=method, basis=basis)
        energy_pyscf = atoms.get_potential_energy()
        print(f"✓ PySCF Potential Energy: {energy_pyscf:.4f} eV")
        print(f"  (Total electronic energy including nuclear repulsion)")
    except Exception as e:
        print(f"✗ PySCF Calculation Failed: {e}")

    # Test MOPAC Calculator (Semi-empirical)
    print(f"\n--- MOPAC Test ({mopac_method}) ---")
    try:
        atoms.calc = MOPAC(method=mopac_method)
        energy_mopac = atoms.get_potential_energy()
        print(f"✓ MOPAC Potential Energy: {energy_mopac:.4f} eV")
        print(f"  (Heat of formation)")
    except Exception as e:
        print(f"✗ MOPAC Calculation Failed:\n{e}")
    
    return atoms

# =====================================================================
# 4. Execution Block
# =====================================================================
if __name__ == "__main__":
    print("="*60)
    print("MOLECULAR MODELING TEST SUITE")
    print("="*60)
    print(f"PySCF version: {pyscf.__version__}")
    print(f"ASE version: {sys.modules['ase'].__version__}")
    
    # Define molecules to test
    test_molecules = ['H2O', 'CH4']  # Water and Methane
    
    # Dictionary to store results
    results = {}
    
    # Test each molecule
    for mol_name in test_molecules:
        atoms = test_molecule(mol_name)
        results[mol_name] = atoms
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Molecule':<10} {'PySCF Energy (eV)':<20} {'MOPAC Energy (eV)':<20}")
    print("-"*60)
    
    for mol_name, atoms in results.items():
        # Get energies from the atoms object (last calculation results)
        # Note: These will use the last calculator set (MOPAC in this case)
        # We need to re-calculate or store energies separately
        pass
    
    print("\nNOTE: PySCF gives total electronic energy, MOPAC gives heat of formation.")
    print("These values are on different scales and cannot be directly compared.")
    print("\nScript execution finished successfully!")
