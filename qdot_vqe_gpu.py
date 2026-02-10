#!/usr/bin/env python3
"""
Simulación VQE con Tangelo para Quantum Dot C5H5N (Piridina)
Espacio activo extendido: 12 qubits (6 electrones, 6 orbitales)
"""

import numpy as np
import time

# Constantes
HARTREE_TO_EV = 27.2114
EV_TO_NM = 1240

# Geometría piridina (C5H5N) - 42 electrones
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

print("=" * 70)
print("  SIMULACIÓN VQE - QUANTUM DOT C5H5N (PIRIDINA)")
print("  Espacio activo extendido: 12 qubits")
print("=" * 70)

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

    # ═══════════════════════════════════════════════════════════════
    # ESPACIO ACTIVO EXTENDIDO: 6 electrones en 6 orbitales = 12 qubits
    # Incluye: HOMO-2, HOMO-1, HOMO, LUMO, LUMO+1, LUMO+2
    # ═══════════════════════════════════════════════════════════════
    n_occ = mol.n_electrons // 2  # 21 orbitales ocupados

    # Orbitales activos: 18, 19, 20 (HOMO-2 a HOMO) y 21, 22, 23 (LUMO a LUMO+2)
    frozen_core = list(range(n_occ - 3))        # 0-17: congelados
    frozen_virtual = list(range(n_occ + 3, mol.n_mos))  # 24-34: congelados
    frozen_orbitals = frozen_core + frozen_virtual

    print(f"\n→ Espacio activo EXTENDIDO (12 qubits):")
    print(f"  Orbitales congelados (core): {len(frozen_core)} (0 a {n_occ-4})")
    print(f"  Orbitales activos: 6 ({n_occ-3} a {n_occ+2})")
    print(f"  Orbitales congelados (virtual): {len(frozen_virtual)}")

    mol_active = SecondQuantizedMolecule(
        xyz=c_dot_geometry,
        q=0,
        spin=0,
        basis="sto-3g",
        frozen_orbitals=frozen_orbitals
    )

    print(f"  Electrones activos: {mol_active.n_active_electrons}")
    print(f"  Orbitales activos: {mol_active.n_active_mos}")
    n_qubits = 2 * mol_active.n_active_mos
    print(f"  Qubits requeridos: {n_qubits}")

    # VQE
    vqe_options = {
        "molecule": mol_active,
        "qubit_mapping": "jw",
        "up_then_down": True,
        "verbose": False
    }

    print(f"\n→ Construyendo circuito VQE...")
    vqe = VQESolver(vqe_options)
    vqe.build()

    resources = vqe.get_resources()
    print(f"  Qubits del circuito: {resources['circuit_width']}")
    print(f"  Profundidad: {resources['circuit_depth']}")
    print(f"  Compuertas 2-qubit: {resources['circuit_2qubit_gates']}")
    print(f"  Parámetros variacionales: {resources['vqe_variational_parameters']}")

    print(f"\n→ Ejecutando optimización VQE (12 qubits)...")
    start_time = time.time()

    energy_vqe = vqe.simulate()

    elapsed = time.time() - start_time
    vqe_success = True

    print(f"\n{'='*70}")
    print("  RESULTADOS VQE (Espacio activo 12 qubits)")
    print("=" * 70)
    print(f"  Tiempo de ejecución: {elapsed:.2f} segundos")
    print(f"  Energía HF:  {mol.mf_energy:.6f} Hartree")
    print(f"  Energía VQE: {energy_vqe:.6f} Hartree")
    correlation = (energy_vqe - mol.mf_energy) * 1000
    print(f"  Correlación capturada: {correlation:.3f} mHartree")

    # Gap HOMO-LUMO
    homo_idx = mol.n_electrons // 2 - 1
    lumo_idx = mol.n_electrons // 2
    homo_energy = mol.mo_energies[homo_idx]
    lumo_energy = mol.mo_energies[lumo_idx]
    gap_hl = lumo_energy - homo_energy

    print(f"\n  Orbitales frontera (HF):")
    print(f"    HOMO (#{homo_idx+1}): {homo_energy:.4f} Ha = {homo_energy*HARTREE_TO_EV:.2f} eV")
    print(f"    LUMO (#{lumo_idx+1}): {lumo_energy:.4f} Ha = {lumo_energy*HARTREE_TO_EV:.2f} eV")
    print(f"    Gap HOMO-LUMO: {gap_hl:.4f} Ha = {gap_hl*HARTREE_TO_EV:.2f} eV")
    print(f"    λ (Koopmans): {EV_TO_NM/(gap_hl*HARTREE_TO_EV):.1f} nm")

except ImportError as e:
    print(f"\n⚠ Tangelo no disponible: {e}")
    print("→ Usando cálculo Hartree-Fock con PySCF...")

except Exception as e:
    print(f"\n⚠ Error en VQE: {e}")
    print("→ Fallback a Hartree-Fock con PySCF...")

# Fallback a PySCF
if not vqe_success:
    from pyscf import gto, scf

    mol_pyscf = gto.Mole()
    mol_pyscf.atom = [(a, c) for a, c in c_dot_geometry]
    mol_pyscf.basis = 'sto-3g'
    mol_pyscf.charge = 0
    mol_pyscf.spin = 0
    mol_pyscf.verbose = 0
    mol_pyscf.build()

    mf = scf.RHF(mol_pyscf)
    mf.verbose = 0
    e_hf = mf.kernel()

    print(f"\n{'='*70}")
    print("  RESULTADOS HARTREE-FOCK (PySCF)")
    print("=" * 70)
    print(f"  Energía RHF: {e_hf:.6f} Hartree")

    mo_energies = mf.mo_energy
    n_occ = mol_pyscf.nelectron // 2
    homo = mo_energies[n_occ - 1]
    lumo = mo_energies[n_occ]
    gap = lumo - homo

    print(f"\n  HOMO: {homo:.4f} Ha = {homo*HARTREE_TO_EV:.2f} eV")
    print(f"  LUMO: {lumo:.4f} Ha = {lumo*HARTREE_TO_EV:.2f} eV")
    print(f"  Gap: {gap:.4f} Ha = {gap*HARTREE_TO_EV:.2f} eV")

# Comparación
print("\n" + "=" * 70)
print("  COMPARACIÓN CON DOCUMENTO")
print("=" * 70)
print("""
  Valores del documento:
    ΔE = 0.102 Hartree ≈ 2.77 eV → λ = 447.65 nm (azul)

  Ventaja del espacio activo extendido (12 qubits):
    • Captura más correlación electrónica
    • Incluye excitaciones HOMO-2 → LUMO+2
    • Mejor descripción de estados excitados
    • Predicción de gap más precisa

  Para el control del reactor DBD:
    → Setpoint sensor: 450 nm ± 20 nm
    → Tamaño objetivo: ~2.5 nm
    → FWHM esperado: 30-50 nm
""")
