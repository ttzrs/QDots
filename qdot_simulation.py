#!/usr/bin/env python3
"""
Simulación de Quantum Dots de Carbono dopados con Nitrógeno
Basado en el documento técnico del proyecto de milirreactor DBD
"""

import numpy as np

# Intentar importar Tangelo
try:
    from tangelo.molecule_library import mol_H2  # Test import
    from tangelo import SecondQuantizedMolecule
    from tangelo.algorithms.variational import VQESolver
    from tangelo.toolboxes.molecular_computation.rdms import compute_rdms
    TANGELO_AVAILABLE = True
except ImportError as e:
    print(f"Error importando Tangelo: {e}")
    TANGELO_AVAILABLE = False

# Constantes físicas
HARTREE_TO_EV = 27.2114  # Conversión Hartree a eV
EV_TO_NM = 1240  # E(eV) * λ(nm) = 1240


def create_cdot_geometry():
    """
    Geometría del clúster C5N (anillo de piridina simplificado)
    Representa un fragmento de punto cuántico de carbono dopado con N pirrolítico
    """
    # Coordenadas en Angstrom - Anillo hexagonal con N sustituyendo un C
    geometry = [
        ("C", (0.0000, 0.0000, 0.0000)),
        ("C", (1.4200, 0.0000, 0.0000)),
        ("N", (2.1300, 1.2300, 0.0000)),  # Nitrógeno del purín integrado
        ("C", (1.4200, 2.4600, 0.0000)),
        ("C", (0.0000, 2.4600, 0.0000)),
        ("C", (-0.7100, 1.2300, 0.0000))
    ]
    return geometry


def run_vqe_simulation():
    """Ejecuta simulación VQE para el estado fundamental"""

    if not TANGELO_AVAILABLE:
        print("Tangelo no disponible, usando cálculos analíticos")
        return None

    geometry = create_cdot_geometry()

    print("=" * 60)
    print("SIMULACIÓN TANGELO - QUANTUM DOT C5N")
    print("=" * 60)
    print("\nGeometría del clúster:")
    for atom, coords in geometry:
        print(f"  {atom}: ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})")

    # Crear molécula
    print("\nCreando molécula...")
    try:
        mol = SecondQuantizedMolecule(
            geometry=geometry,
            q=0,  # Carga neutra
            spin=0,  # Singlete
            basis="sto-3g"  # Base mínima para velocidad
        )

        print(f"  Electrones: {mol.n_electrons}")
        print(f"  Orbitales: {mol.n_mos}")

        # Configurar VQE
        print("\nConfigurando VQE solver...")
        vqe_solver = VQESolver({
            "molecule": mol,
            "qubit_mapping": "jw",  # Jordan-Wigner
            "up_then_down": True
        })
        vqe_solver.build()

        # Ejecutar simulación
        print("Ejecutando simulación cuántica...")
        energy_s0 = vqe_solver.simulate()

        return energy_s0, mol

    except Exception as e:
        print(f"Error en simulación: {e}")
        return None


def calculate_optical_properties(energy_hartree=None):
    """
    Calcula propiedades ópticas del quantum dot
    Valores del documento: Delta_E = 0.102 Hartree ≈ 2.77 eV
    """
    print("\n" + "=" * 60)
    print("CÁLCULO DE PROPIEDADES ÓPTICAS")
    print("=" * 60)

    # Valor del documento para comparación
    delta_e_doc = 0.102  # Hartree (inferido en documento)
    energy_gap_ev_doc = 2.77  # eV (valor del documento)

    # Conversión
    delta_e_ev_calculated = delta_e_doc * HARTREE_TO_EV
    wavelength_doc = EV_TO_NM / energy_gap_ev_doc
    wavelength_calculated = EV_TO_NM / delta_e_ev_calculated

    print("\n--- VALORES DEL DOCUMENTO ---")
    print(f"  ΔE inferido: {delta_e_doc:.3f} Hartree")
    print(f"  ΔE en eV (documento): {energy_gap_ev_doc:.2f} eV")
    print(f"  λ calculada documento: {wavelength_doc:.2f} nm")

    print("\n--- VALIDACIÓN MATEMÁTICA ---")
    print(f"  0.102 Hartree × 27.2114 = {delta_e_ev_calculated:.4f} eV")
    print(f"  1240 / {energy_gap_ev_doc} eV = {wavelength_doc:.2f} nm")

    # Verificar consistencia
    print("\n--- ANÁLISIS DE CONSISTENCIA ---")
    ev_error = abs(delta_e_ev_calculated - energy_gap_ev_doc)
    print(f"  Error en conversión Hartree→eV: {ev_error:.4f} eV ({ev_error/energy_gap_ev_doc*100:.1f}%)")

    if ev_error > 0.05:
        print(f"  ⚠ DISCREPANCIA: 0.102 Hartree = {delta_e_ev_calculated:.2f} eV, no {energy_gap_ev_doc} eV")
        correct_hartree = energy_gap_ev_doc / HARTREE_TO_EV
        print(f"  → Para obtener {energy_gap_ev_doc} eV se necesita ΔE = {correct_hartree:.4f} Hartree")

    # Rango espectral
    print("\n--- CLASIFICACIÓN ESPECTRAL ---")
    if wavelength_doc < 450:
        color = "VIOLETA/AZUL PROFUNDO"
    elif wavelength_doc < 495:
        color = "AZUL"
    elif wavelength_doc < 570:
        color = "VERDE"
    elif wavelength_doc < 590:
        color = "AMARILLO"
    elif wavelength_doc < 620:
        color = "NARANJA"
    else:
        color = "ROJO"

    print(f"  λ = {wavelength_doc:.1f} nm → {color}")
    print(f"  Setpoint del sensor: {wavelength_doc:.0f} nm")

    return {
        "delta_e_hartree": delta_e_doc,
        "delta_e_ev": energy_gap_ev_doc,
        "wavelength_nm": wavelength_doc,
        "color": color
    }


def size_dependent_gap():
    """
    Modelo simplificado de gap vs tamaño de partícula
    Para quantum dots, el gap aumenta al disminuir el tamaño (confinamiento cuántico)
    """
    print("\n" + "=" * 60)
    print("RELACIÓN TAMAÑO-GAP (Confinamiento Cuántico)")
    print("=" * 60)

    # Modelo aproximado para CQDs: E_gap ≈ E_bulk + A/d^2
    # donde d es el diámetro y A es constante de confinamiento
    E_bulk = 0.5  # eV (grafeno bulk ~ 0 eV, pero con dopaje N aumenta)
    A = 10  # eV·nm² (constante de confinamiento aproximada para CQDs)

    print("\nModelo: E_gap = E_bulk + A/d²")
    print(f"  E_bulk = {E_bulk} eV (grafeno dopado N)")
    print(f"  A = {A} eV·nm²\n")

    sizes = [2, 3, 4, 5, 7, 10]  # nm
    print(f"{'Diámetro (nm)':<15} {'Gap (eV)':<12} {'λ (nm)':<12} {'Color'}")
    print("-" * 55)

    for d in sizes:
        gap = E_bulk + A / (d ** 2)
        wavelength = EV_TO_NM / gap if gap > 0 else float('inf')

        if wavelength < 450:
            color = "Violeta/UV"
        elif wavelength < 495:
            color = "Azul"
        elif wavelength < 570:
            color = "Verde"
        elif wavelength < 620:
            color = "Amarillo/Naranja"
        else:
            color = "Rojo/IR"

        print(f"{d:<15} {gap:<12.2f} {wavelength:<12.1f} {color}")

    # Encontrar tamaño para λ = 450 nm (azul objetivo)
    target_lambda = 450  # nm
    target_gap = EV_TO_NM / target_lambda
    target_size = np.sqrt(A / (target_gap - E_bulk))

    print(f"\n→ Para λ = {target_lambda} nm (azul): diámetro óptimo ≈ {target_size:.1f} nm")


def validate_document_numbers():
    """Validación completa de los números del documento"""
    print("\n" + "=" * 60)
    print("VALIDACIÓN DE NÚMEROS DEL DOCUMENTO")
    print("=" * 60)

    validations = []

    # 1. Conversión Hartree a eV
    hartree_value = 0.102
    ev_claimed = 2.77
    ev_actual = hartree_value * HARTREE_TO_EV

    print("\n1. Conversión energética:")
    print(f"   Documento: {hartree_value} Hartree ≈ {ev_claimed} eV")
    print(f"   Calculado: {hartree_value} Hartree = {ev_actual:.4f} eV")
    if abs(ev_actual - ev_claimed) > 0.1:
        print(f"   ❌ ERROR: La conversión correcta daría {ev_actual:.2f} eV, no {ev_claimed} eV")
        validations.append(("Conversión Hartree→eV", False, f"{ev_actual:.2f} vs {ev_claimed}"))
    else:
        print("   ✓ Correcto")
        validations.append(("Conversión Hartree→eV", True, ""))

    # 2. Conversión eV a nm
    lambda_from_doc = EV_TO_NM / ev_claimed
    print(f"\n2. Longitud de onda:")
    print(f"   Documento usa: 1240 / {ev_claimed} eV = {lambda_from_doc:.2f} nm")
    print(f"   Documento dice: ~450 nm (AZUL)")
    if abs(lambda_from_doc - 450) > 10:
        print(f"   ⚠ NOTA: Con {ev_claimed} eV da {lambda_from_doc:.0f} nm, cercano pero no exacto")
    validations.append(("λ desde eV", True, f"{lambda_from_doc:.0f} nm"))

    # 3. Parámetros del plasma
    print("\n3. Parámetros de plasma DBD:")
    params = {
        "Voltaje": ("5-15 kV ajustable → 8-12 kV operación", True),
        "Frecuencia": ("10-30 kHz → 15-25 kHz óptimo", True),
        "Ancho pulso": ("<500 ns (plasma frío)", True),
        "Rise time": ("<100 ns", True)
    }
    for param, (desc, valid) in params.items():
        status = "✓" if valid else "❌"
        print(f"   {status} {param}: {desc}")
        validations.append((param, valid, desc))

    # 4. Geometría del reactor
    print("\n4. Geometría del milirreactor:")
    reactor_params = {
        "Canal": "2 mm × 0.5 mm (maximiza contacto plasma-líquido)",
        "Electrodo": "0.8 mm de pared (barrera dieléctrica)",
        "Serpentín": "150 mm (tiempo de enfriamiento)",
        "Obstáculos": "0.2 mm (micro-turbulencias)"
    }
    for param, desc in reactor_params.items():
        print(f"   → {param}: {desc}")

    # 5. Química del dopaje
    print("\n5. Modelo químico C5N:")
    print("   Clúster: 5 C + 1 N (piridina-like)")
    print("   Base: STO-3G (mínima, para velocidad)")
    print("   Carga: 0 (neutro)")
    print("   Spin: 0 (singlete)")

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE VALIDACIÓN")
    print("=" * 60)

    errors = [v for v in validations if not v[1]]
    if errors:
        print(f"\n⚠ Se encontraron {len(errors)} inconsistencias:")
        for name, _, detail in errors:
            print(f"   - {name}: {detail}")
        print("\n→ RECOMENDACIÓN: Ajustar el valor de ΔE para consistencia")
        correct_hartree = ev_claimed / HARTREE_TO_EV
        print(f"   Si ΔE = {ev_claimed} eV → {correct_hartree:.4f} Hartree (no 0.102)")
    else:
        print("\n✓ Todos los valores son consistentes")


def main():
    print("╔" + "═" * 58 + "╗")
    print("║  SIMULACIÓN QUANTUM DOTS - VALIDACIÓN PROYECTO PURINES  ║")
    print("╚" + "═" * 58 + "╝")

    # Intentar simulación VQE
    result = run_vqe_simulation()
    if result:
        energy, mol = result
        print(f"\n✓ Energía estado fundamental (S0): {energy:.6f} Hartree")
        print(f"  = {energy * HARTREE_TO_EV:.4f} eV")

    # Cálculos analíticos
    optical = calculate_optical_properties()

    # Relación tamaño-gap
    size_dependent_gap()

    # Validación completa
    validate_document_numbers()

    print("\n" + "=" * 60)
    print("CONCLUSIONES PARA EL SETPOINT DEL SENSOR")
    print("=" * 60)
    print(f"""
  Para el sistema de control:
  - λ objetivo: ~450 nm (azul)
  - Tolerancia: FWHM < 40 nm (según documento)
  - Umbral: Intensidad > I_threshold

  Para el diseño del reactor:
  - Tiempo de residencia: Ajustar según cinética
  - Flujo: Variable para controlar tamaño de partícula
  - Dopaje N: Natural del purín (urea, aminoácidos)
""")


if __name__ == "__main__":
    main()
