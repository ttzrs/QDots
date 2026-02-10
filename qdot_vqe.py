#!/usr/bin/env python3
"""
Simulación VQE con Tangelo para Quantum Dot C5H5N (Piridina)
Optimizado para espacio activo reducido y API actual de Tangelo
"""

import numpy as np

# Constantes
HARTREE_TO_EV = 27.2114
EV_TO_NM = 1240

# Geometría del clúster C5H5N (piridina completa - modelo de QD dopado)
# Piridina tiene 42 electrones (par), compatible con spin=0
c_dot_geometry = [
    ("C", (0.0000,  1.3950, 0.0000)),
    ("C", (1.2075,  0.6975, 0.0000)),
    ("C", (1.2075, -0.6975, 0.0000)),
    ("C", (0.0000, -1.3950, 0.0000)),
    ("C", (-1.2075, -0.6975, 0.0000)),
    ("N", (-1.2075,  0.6975, 0.0000)),
    ("H", (0.0000,  2.4800, 0.0000)),
    ("H", (2.1440,  1.2400, 0.0000)),
    ("H", (2.1440, -1.2400, 0.0000)),
    ("H", (0.0000, -2.4800, 0.0000)),
    ("H", (-2.1440, -1.2400, 0.0000))
]

print("=" * 60)
print("SIMULACIÓN VQE - QUANTUM DOT C5H5N (PIRIDINA)")
print("=" * 60)

print("\nGeometría del clúster:")
for atom, (x, y, z) in c_dot_geometry:
    print(f"  {atom}: ({x:7.4f}, {y:7.4f}, {z:7.4f}) Å")

vqe_success = False

try:
    from tangelo import SecondQuantizedMolecule
    from tangelo.algorithms.variational import VQESolver

    print("\n→ Creando molécula en Tangelo...")
    mol = SecondQuantizedMolecule(
        xyz=c_dot_geometry,
        q=0,
        spin=0,
        basis="sto-3g",
        frozen_orbitals=[]
    )

    print(f"  Electrones: {mol.n_electrons}")
    print(f"  Orbitales moleculares: {mol.n_mos}")
    print(f"  Energía HF: {mol.mf_energy:.6f} Hartree")

    # Espacio activo MÍNIMO para VQE rápido (demo)
    # Solo HOMO y LUMO: 2 electrones en 2 orbitales = 4 qubits
    # HOMO está en orbital 20 (índice), LUMO en 21
    n_occ = mol.n_electrons // 2  # 21 orbitales ocupados
    frozen_core = list(range(n_occ - 1))  # Congelar hasta HOMO-1
    frozen_virtual = list(range(n_occ + 1, mol.n_mos))  # Congelar desde LUMO+1
    frozen_orbitals = frozen_core + frozen_virtual

    print(f"\n→ Configurando VQE con espacio activo reducido...")
    print(f"  Orbitales congelados (core): {len(frozen_core)}")
    print(f"  Orbitales congelados (virtual): {len(frozen_virtual)}")

    # Crear molécula con espacio activo
    mol_active = SecondQuantizedMolecule(
        xyz=c_dot_geometry,
        q=0,
        spin=0,
        basis="sto-3g",
        frozen_orbitals=frozen_orbitals
    )

    print(f"  Electrones activos: {mol_active.n_active_electrons}")
    print(f"  Orbitales activos: {mol_active.n_active_mos}")
    print(f"  Qubits requeridos: {2 * mol_active.n_active_mos}")

    # VQE con ansatz UCCSD
    vqe_options = {
        "molecule": mol_active,
        "qubit_mapping": "jw",
        "up_then_down": True,
        "verbose": False
    }

    print(f"\n→ Construyendo circuito VQE...")
    vqe = VQESolver(vqe_options)
    vqe.build()

    # Obtener recursos del circuito
    resources = vqe.get_resources()
    print(f"  Qubits del circuito: {resources['circuit_width']}")
    print(f"  Profundidad del circuito: {resources['circuit_depth']}")
    print(f"  Compuertas de 2 qubits: {resources['circuit_2qubit_gates']}")
    print(f"  Parámetros variacionales: {resources['vqe_variational_parameters']}")

    print(f"\n→ Ejecutando optimización VQE...")
    print(f"  (Esto puede tardar unos segundos...)")

    energy_vqe = vqe.simulate()
    vqe_success = True

    print(f"\n{'='*60}")
    print("RESULTADOS VQE")
    print("=" * 60)
    print(f"  Energía HF:  {mol.mf_energy:.6f} Hartree")
    print(f"  Energía VQE: {energy_vqe:.6f} Hartree")
    correlation = (energy_vqe - mol.mf_energy) * 1000
    print(f"  Correlación: {correlation:.3f} mHartree")

    # Gap HOMO-LUMO desde energías orbitales HF
    homo_idx = mol.n_electrons // 2 - 1
    lumo_idx = mol.n_electrons // 2
    homo_energy = mol.mo_energies[homo_idx]
    lumo_energy = mol.mo_energies[lumo_idx]
    gap_hl = lumo_energy - homo_energy

    print(f"\n  Orbitales frontera (HF):")
    print(f"    HOMO (#{homo_idx+1}): {homo_energy:.4f} Ha = {homo_energy*HARTREE_TO_EV:.2f} eV")
    print(f"    LUMO (#{lumo_idx+1}): {lumo_energy:.4f} Ha = {lumo_energy*HARTREE_TO_EV:.2f} eV")
    print(f"    Gap HOMO-LUMO: {gap_hl:.4f} Ha = {gap_hl*HARTREE_TO_EV:.2f} eV")

    lambda_emission = EV_TO_NM / (gap_hl * HARTREE_TO_EV)
    print(f"    λ estimada (Koopmans): {lambda_emission:.1f} nm")

except ImportError as e:
    print(f"\n⚠ Tangelo no disponible: {e}")
    print("→ Usando cálculo Hartree-Fock con PySCF...")

except Exception as e:
    print(f"\n⚠ Error en VQE: {e}")
    print("→ Fallback a cálculo Hartree-Fock con PySCF...")

# Fallback a PySCF si VQE no funcionó
if not vqe_success:
    from pyscf import gto, scf

    mol_pyscf = gto.Mole()
    mol_pyscf.atom = [(a, c) for a, c in c_dot_geometry]
    mol_pyscf.basis = 'sto-3g'
    mol_pyscf.charge = 0
    mol_pyscf.spin = 0
    mol_pyscf.verbose = 0
    mol_pyscf.build()

    print(f"  Electrones: {mol_pyscf.nelectron}")
    print(f"  Funciones base: {mol_pyscf.nao}")

    mf = scf.RHF(mol_pyscf)
    mf.verbose = 0
    e_hf = mf.kernel()

    print(f"\n{'='*60}")
    print("RESULTADOS HARTREE-FOCK (PySCF)")
    print("=" * 60)
    print(f"  Energía RHF: {e_hf:.6f} Hartree")
    print(f"             = {e_hf * HARTREE_TO_EV:.4f} eV")

    mo_energies = mf.mo_energy
    n_occ = mol_pyscf.nelectron // 2
    homo = mo_energies[n_occ - 1]
    lumo = mo_energies[n_occ]
    gap = lumo - homo

    print(f"\n  HOMO (#{n_occ}): {homo:.4f} Ha = {homo*HARTREE_TO_EV:.2f} eV")
    print(f"  LUMO (#{n_occ+1}): {lumo:.4f} Ha = {lumo*HARTREE_TO_EV:.2f} eV")
    print(f"  Gap HOMO-LUMO: {gap:.4f} Ha = {gap*HARTREE_TO_EV:.2f} eV")

    lambda_nm = EV_TO_NM / (gap * HARTREE_TO_EV)
    print(f"\n  λ emisión estimada: {lambda_nm:.1f} nm")

# Comparación con valores del documento
print("\n" + "=" * 60)
print("COMPARACIÓN CON DOCUMENTO")
print("=" * 60)
print("""
  Valores del documento:
    ΔE = 0.102 Hartree ≈ 2.77 eV
    λ = 1240/2.77 = 447.65 nm (azul)

  Interpretación:
  - El gap HOMO-LUMO de Koopmans (~14 eV) sobreestima el gap óptico
  - El documento usa el gap óptico experimental (~2.77 eV)
  - Para CQDs de ~2.5 nm, el gap real es ~2.7-3.0 eV

  Factores no capturados por el modelo:
  1. Relajación orbital en estados excitados
  2. Efectos de correlación electrónica (parcialmente en VQE)
  3. Efectos de tamaño finito del quantum dot
  4. Funcionalización de superficie

  Para el control del reactor:
    → Setpoint sensor: 450 nm ± 20 nm
    → Tamaño objetivo: ~2.5 nm
    → FWHM esperado: 30-50 nm
""")
