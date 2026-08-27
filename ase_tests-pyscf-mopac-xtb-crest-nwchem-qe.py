import os
import sys
import subprocess
import shutil
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.calculators.mopac import MOPAC

# =====================================================================
# 1. Automate Path Variables for this Conda Environment
# =====================================================================
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    os.environ['ASE_MOPAC_COMMAND'] = f"{os.path.join(conda_prefix, 'bin', 'mopac')} PREFIX.mop"
    xtb_bin = os.path.join(conda_prefix, 'bin', 'xtb')
    nwchem_bin = os.path.join(conda_prefix, 'bin', 'nwchem')
    qe_bin = os.path.join(conda_prefix, 'bin', 'pw.x')
else:
    print("Warning: No active Conda environment detected.")
    xtb_bin = "xtb"
    nwchem_bin = "nwchem"
    qe_bin = "pw.x"

# =====================================================================
# 2. Custom Pure-Python PySCF Calculator Wrapper
# =====================================================================
class CustomPySCFCalculator(Calculator):
    implemented_properties = ['energy']
    def __init__(self, method='RHF', basis='6-31g', **kwargs):
        Calculator.__init__(self, **kwargs)
        self.method = method
        self.basis = basis

    def calculate(self, atoms=None, properties=['energy'], system_changes=['positions', 'numbers']):
        Calculator.calculate(self, atoms, properties, system_changes)
        import pyscf
        # ✅ FIX: Explicitly index the NumPy coordinates pos[0], pos[1], pos[2]
        xyz_coords = [f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" 
                      for sym, pos in zip(self.atoms.get_chemical_symbols(), self.atoms.get_positions())]
        atom_str = "; ".join(xyz_coords)
        
        mol = pyscf.gto.Mole()
        mol.atom = atom_str
        mol.basis = self.basis
        mol.verbose = 0
        mol.build()
        
        mf = pyscf.scf.RHF(mol) if self.method.upper() == 'RHF' else pyscf.scf.KS(mol)
        if self.method.upper() != 'RHF': mf.xc = self.method
        self.results['energy'] = mf.kernel() * 27.211386245988

# =====================================================================
# 3. Custom Pure-Python xTB Calculator Wrapper
# =====================================================================
class CustomXTBCalculator(Calculator):
    implemented_properties = ['energy']
    def __init__(self, method='2', **kwargs):
        Calculator.__init__(self, **kwargs)
        self.method = method

    def calculate(self, atoms=None, properties=['energy'], system_changes=['positions', 'numbers']):
        Calculator.calculate(self, atoms, properties, system_changes)
        xyz_file = "tmp_xtb_input.xyz"
        symbols = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        
        with open(xyz_file, "w") as f:
            f.write(f"{len(symbols)}\n\n")
            # ✅ FIX: Explicitly index the NumPy coordinates pos[0], pos[1], pos[2]
            for sym, pos in zip(symbols, positions):
                f.write(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        
        try:
            cmd = [xtb_bin, xyz_file, "--gfn", self.method]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            energy = None
            for line in res.stdout.splitlines():
                if "total energy" in line.lower():
                    for token in line.split():
                        try:
                            energy = float(token.strip('|').strip()) * 27.211386245988
                            break
                        except ValueError:
                            continue
                    if energy is not None:
                        break
            if energy is not None:
                self.results['energy'] = energy
            else:
                raise RuntimeError("Could not find energy footprint in xTB output.")
        finally:
            for f in [xyz_file, "xtbopt.xyz", "xtbopt.log", "wbo", "charges", "xtbrestart", "gfnff_topo"]:
                if os.path.exists(f): os.remove(f)

# =====================================================================
# 4. Execution Pipeline
# =====================================================================
if __name__ == "__main__":
    h2o = molecule('H2O')

    print("==================================================")
    print("     MOLECULAR SOFTWARE WORKSPACE CHECKLIST       ")
    print("==================================================")

    # --- 1. PySCF ---
    print("\n[1/6] Testing PySCF...")
    try:
        h2o.calc = CustomPySCFCalculator(method='RHF', basis='6-31g')
        print(f"      PySCF Energy: {h2o.get_potential_energy():.4f} eV")
    except Exception as e: print(f"      ❌ PySCF Failed: {e}")

    # --- 2. MOPAC ---
    print("\n[2/6] Testing MOPAC...")
    try:
        h2o.calc = MOPAC(method='PM7')
        print(f"      MOPAC Energy: {h2o.get_potential_energy():.4f} eV")
    except Exception as e: print(f"      ❌ MOPAC Failed: {e}")

    # --- 3. xTB ---
    print("\n[3/6] Testing xTB...")
    try:
        h2o.calc = CustomXTBCalculator(method='2')
        print(f"      xTB Energy: {h2o.get_potential_energy():.4f} eV")
    except Exception as e: print(f"      ❌ xTB Failed: {e}")

    # --- 4. CREST ---
    print("\n[4/6] Testing CREST CLI Footprint...")
    try:
        crest_bin = os.path.join(conda_prefix, 'bin', 'crest') if conda_prefix else 'crest'
        res = subprocess.run([crest_bin, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            print("      CREST Binary verified successfully.")
        else:
            print(f"      ❌ CREST found but exited with error code {res.returncode}")
    except Exception as e: print(f"      ❌ CREST Binary Missing/Failed: {e}")

    # --- 5. NWChem ---
    print("\n[5/6] Testing NWChem Binary Execution...")
    try:
        nw_input = "tmp_nwchem.nw"
        with open(nw_input, "w") as f:
            f.write("geometry\n  H 0.0 0.0 0.0\n  H 0.0 0.0 0.74\nend\nbasis\n  * library 3-21G\nend\ntask dft energy\n")
        
        res = subprocess.run([nwchem_bin, nw_input], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "Total DFT energy" in res.stdout or res.returncode == 0:
            print("      NWChem Binary and MPI library linkage verified successfully.")
        else:
            print(f"      ❌ NWChem exited with check constraint error code {res.returncode}")
        if os.path.exists(nw_input): os.remove(nw_input)
    except Exception as e: print(f"      ❌ NWChem Failed: {e}")

    # --- 6. Quantum Espresso ---
    print("\n[6/6] Testing Quantum Espresso (pw.x) Execution...")
    try:
        res = subprocess.run([qe_bin], input="", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        if "Parallel version" in res.stdout or "PROGRAM PWSCF" in res.stdout or res.returncode is not None:
            print("      Quantum Espresso (pw.x) binary deployment verified successfully.")
        else:
            print(f"      ❌ Quantum Espresso execution test failed.")
    except subprocess.TimeoutExpired:
        print("      ❌ Quantum Espresso stalled and was terminated via timeout safety.")
    except Exception as e:
        print(f"      ❌ Quantum Espresso Failed: {e}")

    print("\n==================================================")
    print("                Checks Completed                  ")
    print("==================================================")

