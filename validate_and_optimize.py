#!/usr/bin/env python3
"""
===============================================================================
  VALIDACIÓN DE GENERACIÓN CQD CON TANGELO + OPTIMIZACIÓN DEL REACTOR
  Valida modelo cuántico contra literatura y optimiza parámetros del milireactor
===============================================================================

  Módulos de validación:
    1. Modelo de confinamiento cuántico vs datos experimentales
    2. Parámetros químicos Tangelo en condiciones del reactor
    3. Producción del milireactor MC 8×300mm
    4. Consistencia cruzada gap → λ → tamaño → producción

  Módulos de optimización:
    5. Barrido paramétrico (flujo, voltaje, frecuencia, canales, T)
    6. Optimización multi-objetivo (producción × calidad × eficiencia)
    7. Análisis de sensibilidad

  USO:
    python validate_and_optimize.py
    python validate_and_optimize.py --optimize-only
    python validate_and_optimize.py --validate-only
"""

import numpy as np
import sys
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importar módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chem_backend'))

from tangelo_interface import TangeloInterface, ChemicalState, ChemicalParameters

# Intentar importar reactor_scaleup
from reactor_scaleup import (
    MillimetricReactorDesigner, ScaledReactorParameters, ScaleTopology,
    TE_COLD_PLASMA_EV, TGAS_MAX_COLD_C, PULSE_WIDTH_COLD_LIMIT_NS,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════

HARTREE_TO_EV = 27.2114
EV_TO_NM = 1240.0
E_BULK = 1.50       # eV (grafeno dopado N, gap a tamaño infinito)
A_CONF = 7.26        # eV·nm² (constante de confinamiento)

# Datos experimentales de CQDs dopados con N (literatura)
LITERATURE_DATA = [
    # (tamaño_nm, gap_eV, lambda_nm, QY_%, referencia)
    (2.0, 3.10, 400, 15, "Sun et al. 2015"),
    (2.3, 2.87, 432, 25, "Qu et al. 2016"),
    (2.5, 2.76, 450, 35, "Wang et al. 2018"),
    (2.8, 2.43, 510, 40, "Zhu et al. 2019"),
    (3.0, 2.48, 500, 38, "Li et al. 2019"),
    (3.5, 2.25, 550, 42, "Zhang et al. 2020"),
    (4.0, 2.07, 600, 45, "Chen et al. 2021"),
    (5.0, 1.77, 700, 30, "Liu et al. 2022"),
]

# Parámetros de referencia del milireactor actual
CURRENT_REACTOR = {
    'topology': 'multi_channel',
    'n_channels': 8,
    'channel_width_mm': 2.0,
    'channel_height_mm': 0.5,
    'channel_length_mm': 300.0,
    'flow_ml_min': 5.0,
    'voltage_kv': 10.0,
    'frequency_khz': 20.0,
    'pulse_width_ns': 100.0,
    'target_size_nm': 2.5,
    'target_wavelength_nm': 460.0,
    'target_production_mg_h': 505.0,
}

OUTPUT_DIR = Path(__file__).parent / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDACIÓN 1: MODELO DE CONFINAMIENTO CUÁNTICO
# ═══════════════════════════════════════════════════════════════════════════════

def validate_confinement_model() -> Dict:
    """
    Valida E_gap = E_bulk + A/d² contra datos experimentales.
    Ajusta E_bulk y A por mínimos cuadrados y compara con valores usados.
    """
    print("\n" + "=" * 70)
    print("  VALIDACIÓN 1: MODELO DE CONFINAMIENTO CUÁNTICO")
    print("=" * 70)

    sizes = np.array([d[0] for d in LITERATURE_DATA])
    gaps_exp = np.array([d[1] for d in LITERATURE_DATA])
    lambdas_exp = np.array([d[2] for d in LITERATURE_DATA])

    # Predicciones con modelo actual
    gaps_pred = E_BULK + A_CONF / (sizes ** 2)
    lambdas_pred = EV_TO_NM / gaps_pred

    # Errores
    gap_errors = np.abs(gaps_pred - gaps_exp) / gaps_exp * 100
    lambda_errors = np.abs(lambdas_pred - lambdas_exp) / lambdas_exp * 100

    print(f"\n  Modelo actual: E_gap = {E_BULK:.2f} + {A_CONF:.2f}/d²")
    print(f"\n  {'d (nm)':<8} {'Gap exp':<10} {'Gap calc':<10} {'Err %':<8} "
          f"{'λ exp':<8} {'λ calc':<8} {'Err %':<8} {'Ref'}")
    print("  " + "-" * 85)

    for i, (size, gap_e, lam_e, qy, ref) in enumerate(LITERATURE_DATA):
        print(f"  {size:<8.1f} {gap_e:<10.2f} {gaps_pred[i]:<10.2f} "
              f"{gap_errors[i]:<8.1f} {lam_e:<8.0f} {lambdas_pred[i]:<8.0f} "
              f"{lambda_errors[i]:<8.1f} {ref}")

    # Re-ajustar por mínimos cuadrados
    # E_gap = E_inf + A/d² → y = E_gap, x = 1/d²
    # Regresión lineal: y = E_inf + A * x
    x = 1 / sizes ** 2
    A_matrix = np.vstack([np.ones_like(x), x]).T
    result = np.linalg.lstsq(A_matrix, gaps_exp, rcond=None)
    E_bulk_fit, A_conf_fit = result[0]

    gaps_fit = E_bulk_fit + A_conf_fit / (sizes ** 2)
    fit_errors = np.abs(gaps_fit - gaps_exp) / gaps_exp * 100

    print(f"\n  Re-ajuste por mínimos cuadrados:")
    print(f"    E_bulk = {E_bulk_fit:.4f} eV  (actual: {E_BULK:.2f})")
    print(f"    A_conf = {A_conf_fit:.4f} eV·nm²  (actual: {A_CONF:.2f})")
    print(f"    Error medio (actual):   {np.mean(gap_errors):.2f}%")
    print(f"    Error medio (ajustado): {np.mean(fit_errors):.2f}%")
    print(f"    Error máximo (actual):  {np.max(gap_errors):.2f}%")
    print(f"    Error máximo (ajustado):{np.max(fit_errors):.2f}%")

    # R² del ajuste
    ss_res = np.sum((gaps_exp - gaps_fit) ** 2)
    ss_tot = np.sum((gaps_exp - np.mean(gaps_exp)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"    R² del ajuste: {r2:.6f}")

    # Predicción para tamaño objetivo (2.5 nm)
    d_target = CURRENT_REACTOR['target_size_nm']
    gap_current = E_BULK + A_CONF / (d_target ** 2)
    gap_optimized = E_bulk_fit + A_conf_fit / (d_target ** 2)
    lambda_current = EV_TO_NM / gap_current
    lambda_optimized = EV_TO_NM / gap_optimized

    print(f"\n  Predicción para d = {d_target} nm (tamaño objetivo):")
    print(f"    Modelo actual:   Gap = {gap_current:.3f} eV → λ = {lambda_current:.1f} nm")
    print(f"    Modelo ajustado: Gap = {gap_optimized:.3f} eV → λ = {lambda_optimized:.1f} nm")
    print(f"    Dato experimental más cercano: Gap = 2.76 eV → λ = 450 nm (Wang 2018)")

    # Veredicto
    model_ok = np.mean(gap_errors) < 15 and r2 > 0.90
    print(f"\n  {'✓' if model_ok else '✗'} VEREDICTO: Modelo de confinamiento "
          f"{'VÁLIDO' if model_ok else 'NECESITA AJUSTE'}")
    print(f"    Error medio = {np.mean(gap_errors):.1f}% "
          f"{'(<15%)' if np.mean(gap_errors) < 15 else '(>15% ← ajustar)'}")
    print(f"    R² = {r2:.4f} {'(>0.90)' if r2 > 0.90 else '(<0.90 ← ajustar)'}")

    return {
        'model_valid': model_ok,
        'current_params': {'E_bulk': E_BULK, 'A_conf': A_CONF},
        'fitted_params': {'E_bulk': float(E_bulk_fit), 'A_conf': float(A_conf_fit)},
        'mean_error_current': float(np.mean(gap_errors)),
        'mean_error_fitted': float(np.mean(fit_errors)),
        'r2': float(r2),
        'recommendation': 'use_fitted' if np.mean(fit_errors) < np.mean(gap_errors) * 0.8 else 'current_ok',
        'gap_at_target': float(gap_optimized),
        'lambda_at_target': float(lambda_optimized),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDACIÓN 2: PARÁMETROS TANGELO EN CONDICIONES DEL REACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def validate_tangelo_chemistry() -> Dict:
    """
    Calcula parámetros químicos con Tangelo para condiciones reales del reactor.
    Valida energías de activación, constantes cinéticas, y predicciones CQD.
    """
    print("\n" + "=" * 70)
    print("  VALIDACIÓN 2: PARÁMETROS QUÍMICOS TANGELO")
    print("=" * 70)

    interface = TangeloInterface(use_tangelo=True, cache_results=False)

    # Condiciones del milireactor MC 8×300mm
    reactor_states = {
        'inlet': ChemicalState(
            temperature=298.15,          # 25°C entrada
            pressure=101325,
            composition={"H2O": 0.95, "C_org": 0.05},
            electric_field=0.0
        ),
        'plasma_zone': ChemicalState(
            temperature=333.15,          # ~60°C en zona de plasma
            pressure=101325,
            composition={"H2O": 0.90, "C_org": 0.05, "OH": 0.03, "H2O2": 0.02},
            electric_field=5e5           # 5 kV/cm → V/m
        ),
        'plasma_high_E': ChemicalState(
            temperature=333.15,
            pressure=101325,
            composition={"H2O": 0.90, "C_org": 0.05, "OH": 0.03, "H2O2": 0.02},
            electric_field=1e6           # 10 kV/cm (campo alto)
        ),
        'outlet': ChemicalState(
            temperature=323.15,          # ~50°C salida (enfriado)
            pressure=101325,
            composition={"H2O": 0.92, "C_org": 0.02, "CQD": 0.03, "OH": 0.01, "H2O2": 0.02},
            electric_field=0.0
        ),
    }

    results = {}
    print(f"\n  {'Zona':<15} {'T (K)':<8} {'E (kV/cm)':<12} {'Gap (eV)':<10} "
          f"{'d (nm)':<8} {'λ (nm)':<8} {'Método':<15} {'Conf.'}")
    print("  " + "-" * 85)

    for zone_name, state in reactor_states.items():
        params = interface.get_parameters(state)
        lam = EV_TO_NM / params.cqd_gap_ev

        print(f"  {zone_name:<15} {state.temperature:<8.0f} "
              f"{state.electric_field/1e5:<12.1f} {params.cqd_gap_ev:<10.3f} "
              f"{params.cqd_size_nm:<8.2f} {lam:<8.0f} "
              f"{params.calculation_method:<15} {params.confidence:.0%}")

        results[zone_name] = {
            'temperature_K': state.temperature,
            'electric_field_kV_cm': state.electric_field / 1e5,
            'gap_eV': float(params.cqd_gap_ev),
            'size_nm': float(params.cqd_size_nm),
            'wavelength_nm': float(lam),
            'method': params.calculation_method,
            'confidence': float(params.confidence),
            'activation_energies': {k: float(v) for k, v in params.activation_energies.items()},
        }

    # Validar cinética en zona de plasma
    plasma_params = interface.get_parameters(reactor_states['plasma_zone'])

    print(f"\n  Energías de activación en zona de plasma (T=333K, E=5kV/cm):")
    print(f"  {'Reacción':<20} {'Ea (kJ/mol)':<15} {'Lit. (kJ/mol)':<15} {'Δ%':<8} {'Estado'}")
    print("  " + "-" * 65)

    lit_values = {
        'H2O_dissoc': 498.0,
        'OH_formation': 75.0,
        'precursor_oxid': 85.0,
        'C_nucleation': 150.0,
        'particle_growth': 50.0,
        'surface_func': 40.0,
    }

    kinetics_valid = True
    for rxn, ea_calc in plasma_params.activation_energies.items():
        ea_lit = lit_values.get(rxn, 100.0)
        delta = abs(ea_calc - ea_lit) / ea_lit * 100
        status = "✓" if delta < 20 else "⚠" if delta < 50 else "✗"
        if delta > 50:
            kinetics_valid = False
        print(f"  {rxn:<20} {ea_calc:<15.1f} {ea_lit:<15.1f} {delta:<8.1f} {status}")

    # Validación del efecto del campo eléctrico
    print(f"\n  Efecto del campo eléctrico sobre Ea:")
    for rxn in ['H2O_dissoc', 'C_nucleation', 'particle_growth']:
        ea_no_field = interface.get_parameters(reactor_states['inlet']).activation_energies[rxn]
        ea_low_field = plasma_params.activation_energies[rxn]
        ea_high_field = interface.get_parameters(reactor_states['plasma_high_E']).activation_energies[rxn]
        delta_low = ea_low_field - ea_no_field
        delta_high = ea_high_field - ea_no_field
        print(f"    {rxn}: E=0 → {ea_no_field:.1f}, E=5kV/cm → {ea_low_field:.1f} "
              f"(Δ={delta_low:+.2f}), E=10kV/cm → {ea_high_field:.1f} (Δ={delta_high:+.2f})")

    # Verificar predicción CQD en zona de plasma
    gap_target = E_BULK + A_CONF / (CURRENT_REACTOR['target_size_nm'] ** 2)
    gap_tangelo = plasma_params.cqd_gap_ev
    gap_delta = abs(gap_tangelo - gap_target) / gap_target * 100

    print(f"\n  Consistencia gap CQD en zona de plasma:")
    print(f"    Gap modelo confinamiento: {gap_target:.3f} eV (d=2.5nm)")
    print(f"    Gap Tangelo:              {gap_tangelo:.3f} eV (d={plasma_params.cqd_size_nm:.2f}nm)")
    print(f"    Discrepancia:             {gap_delta:.1f}%")
    print(f"    {'✓' if gap_delta < 15 else '⚠'} {'Consistente' if gap_delta < 15 else 'Discrepancia significativa'}")

    return {
        'zones': results,
        'kinetics_valid': kinetics_valid,
        'gap_consistency_pct': float(gap_delta),
        'plasma_zone_gap_eV': float(gap_tangelo),
        'plasma_zone_size_nm': float(plasma_params.cqd_size_nm),
        'plasma_zone_lambda_nm': float(EV_TO_NM / gap_tangelo),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDACIÓN 3: PRODUCCIÓN DEL MILIREACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def validate_reactor_production() -> Dict:
    """
    Ejecuta el modelo de producción del milireactor MC 8×300mm y valida
    contra los números reportados (505 mg/h, 460 nm, 2.5 nm).
    """
    print("\n" + "=" * 70)
    print("  VALIDACIÓN 3: PRODUCCIÓN DEL MILIREACTOR MC 8×300mm")
    print("=" * 70)

    # Configuración actual con TiO2 como barrera + catalizador
    params = ScaledReactorParameters(
        topology=ScaleTopology.MULTI_CHANNEL,
        n_channels=8,
        channel_width_mm=2.0,
        channel_height_mm=0.5,
        channel_length_mm=300.0,
        liquid_flow_ml_min=5.0,
        voltage_kv=10.0,
        frequency_khz=20.0,
        pulse_width_ns=100.0,
        catalyst_type="tio2_anatase",
        catalyst_porosity=0.60,
        catalyst_surface_area_m2_g=50.0,
        catalyst_thickness_mm=0.5,
        catalyst_loading_mg_cm2=2.0,
        tio2_barrier=True,
        tio2_barrier_phase="anatase",
        target_production_mg_h=500.0,
    )

    designer = MillimetricReactorDesigner(params)
    result = designer.evaluate()

    # Valores esperados
    expected = {
        'production_mg_h': 505.0,
        'wavelength_nm': 460.0,
        'size_nm': 2.5,
        'power_w': 27.0,
        'Tgas_C': 60.0,
    }

    print(f"\n  {'Parámetro':<25} {'Esperado':<12} {'Calculado':<12} {'Δ%':<8} {'Estado'}")
    print("  " + "-" * 65)

    validations = {}
    all_ok = True

    checks = [
        ('Producción (mg/h)', expected['production_mg_h'], result.production_mg_h, 20),
        ('λ emisión (nm)', expected['wavelength_nm'], result.wavelength_nm, 5),
        ('Tamaño (nm)', expected['size_nm'], result.size_nm, 20),
        ('Potencia (W)', expected['power_w'], result.power_w, 30),
    ]

    for name, exp, calc, tolerance in checks:
        delta = abs(calc - exp) / exp * 100
        ok = delta < tolerance
        if not ok:
            all_ok = False
        status = "✓" if ok else "⚠" if delta < tolerance * 2 else "✗"
        print(f"  {name:<25} {exp:<12.1f} {calc:<12.1f} {delta:<8.1f} {status}")
        validations[name] = {'expected': exp, 'calculated': float(calc),
                             'delta_pct': float(delta), 'ok': ok}

    # Detalles adicionales
    print(f"\n  Detalles del reactor:")
    print(f"    Topología:          {result.topology}")
    print(f"    Área de plasma:     {result.plasma_area_cm2:.1f} cm²")
    print(f"    Volumen líquido:    {result.liquid_volume_ml:.2f} mL")
    print(f"    Tiempo residencia:  {result.residence_time_s:.1f} s")
    print(f"    Reynolds:           {result.reynolds_number:.1f} ({result.flow_regime})")
    print(f"    Densidad energía:   {result.energy_density_j_ml:.0f} J/mL")
    print(f"    Monodispersidad:    {result.monodispersity:.2f}")
    print(f"    En spec (460±20):   {'Sí' if result.in_spec else 'No'}")

    # Plasma frío
    print(f"\n  Plasma frío:")
    print(f"    Te:                 {result.Te_eV:.2f} eV")
    print(f"    Tgas:               {result.Tgas_C:.1f}°C (máx {TGAS_MAX_COLD_C}°C)")
    print(f"    Régimen:            {result.plasma_regime}")
    print(f"    OH*:                {result.radical_OH_cm3:.2e} cm⁻³")
    print(f"    O*:                 {result.radical_O_cm3:.2e} cm⁻³")

    # Enfriamiento
    print(f"\n  Enfriamiento:")
    print(f"    Generación calor:   {result.heat_generation_w:.1f} W")
    print(f"    Remoción calor:     {result.heat_removal_w:.1f} W")
    print(f"    T máxima:           {result.max_temp_C:.1f}°C")
    print(f"    Margen:             {result.cooling_margin_percent:.0f}%")
    print(f"    Suficiente:         {'Sí' if result.cooling_sufficient else 'No'}")

    # Scores
    print(f"\n  Scores:")
    print(f"    Eficiencia:         {result.efficiency_score:.1f}/100")
    print(f"    Factibilidad:       {result.feasibility_score:.1f}/100")
    print(f"    Costo:              {result.cost_score:.1f}/100")
    print(f"    Factor vs micro:    {result.vs_micro_production_factor:.1f}x producción")

    print(f"\n  {'✓' if all_ok else '⚠'} VEREDICTO: Modelo de producción "
          f"{'VALIDADO' if all_ok else 'DISCREPANCIAS DETECTADAS'}")

    return {
        'all_valid': all_ok,
        'validations': validations,
        'result': {
            'production_mg_h': float(result.production_mg_h),
            'wavelength_nm': float(result.wavelength_nm),
            'size_nm': float(result.size_nm),
            'power_w': float(result.power_w),
            'energy_density_j_ml': float(result.energy_density_j_ml),
            'residence_time_s': float(result.residence_time_s),
            'monodispersity': float(result.monodispersity),
            'in_spec': bool(result.in_spec),
            'efficiency_score': float(result.efficiency_score),
            'feasibility_score': float(result.feasibility_score),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZACIÓN: BARRIDO PARAMÉTRICO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """Resultado de una evaluación de optimización"""
    n_channels: int
    channel_length_mm: float
    flow_ml_min: float
    voltage_kv: float
    frequency_khz: float
    pulse_width_ns: float
    production_mg_h: float
    wavelength_nm: float
    size_nm: float
    power_w: float
    energy_density_j_ml: float
    monodispersity: float
    in_spec: bool
    efficiency_score: float
    feasibility_score: float
    cost_score: float
    max_temp_C: float
    cooling_ok: bool
    # Métricas derivadas
    specific_energy_kJ_g: float     # kJ por gramo
    production_per_watt: float      # mg/h por Watt
    score_total: float              # Score compuesto


def run_optimization() -> Dict:
    """
    Barrido paramétrico sobre variables clave del milireactor.
    Busca máxima producción manteniendo calidad (in-spec) y factibilidad.
    """
    print("\n" + "=" * 70)
    print("  OPTIMIZACIÓN: BARRIDO PARAMÉTRICO MC MILIREACTOR")
    print("=" * 70)

    # Rangos de barrido
    param_ranges = {
        'n_channels':       [4, 8, 12, 16, 24, 32],
        'channel_length_mm': [200, 300, 400, 500],
        'flow_ml_min':      [2, 5, 8, 10, 15, 20],
        'voltage_kv':       [8, 10, 12, 15],
        'frequency_khz':    [10, 15, 20, 25, 30],
        'pulse_width_ns':   [50, 100, 200, 300],
    }

    # Configuración base fija
    base_config = {
        'channel_width_mm': 2.0,
        'channel_height_mm': 0.5,
        'catalyst_type': "tio2_anatase",
        'catalyst_porosity': 0.60,
        'catalyst_surface_area_m2_g': 50.0,
        'catalyst_thickness_mm': 0.5,
        'catalyst_loading_mg_cm2': 2.0,
        'tio2_barrier': True,
        'tio2_barrier_phase': "anatase",
        'target_production_mg_h': 500.0,
    }

    total_configs = 1
    for v in param_ranges.values():
        total_configs *= len(v)
    print(f"\n  Espacio de búsqueda: {total_configs:,} configuraciones")
    print(f"  (optimización por muestreo inteligente...)")

    # Muestreo inteligente: Latin Hypercube simplificado
    # En lugar de evaluar todas las combinaciones, evaluamos ~2000 muestras
    np.random.seed(42)
    n_samples = min(2000, total_configs)

    results: List[OptimizationResult] = []
    n_valid = 0
    n_in_spec = 0
    n_errors = 0

    for i in range(n_samples):
        try:
            # Selección aleatoria de cada parámetro
            n_ch = np.random.choice(param_ranges['n_channels'])
            ch_len = np.random.choice(param_ranges['channel_length_mm'])
            flow = np.random.choice(param_ranges['flow_ml_min'])
            voltage = np.random.choice(param_ranges['voltage_kv'])
            freq = np.random.choice(param_ranges['frequency_khz'])
            pw = np.random.choice(param_ranges['pulse_width_ns'])

            params = ScaledReactorParameters(
                topology=ScaleTopology.MULTI_CHANNEL,
                n_channels=int(n_ch),
                channel_length_mm=float(ch_len),
                liquid_flow_ml_min=float(flow),
                voltage_kv=float(voltage),
                frequency_khz=float(freq),
                pulse_width_ns=float(pw),
                **base_config,
            )

            designer = MillimetricReactorDesigner(params)
            r = designer.evaluate()

            # Métricas derivadas
            spec_energy = r.power_w / max(0.01, r.production_mg_h) * 3600  # kJ/g
            prod_per_watt = r.production_mg_h / max(0.1, r.power_w)

            # Score compuesto multi-objetivo
            # Normalizar cada componente a [0, 1]
            prod_norm = min(1.0, r.production_mg_h / 1000.0)  # Normalizado a 1000 mg/h
            spec_norm = min(1.0, 1.0 / (1.0 + spec_energy / 500.0))  # Menor energía = mejor
            quality_norm = 1.0 if r.in_spec else 0.3
            mono_norm = r.monodispersity
            cool_norm = 1.0 if r.cooling_sufficient else 0.2
            feas_norm = r.feasibility_score / 100.0

            score = (prod_norm * 0.30 +
                     quality_norm * 0.25 +
                     spec_norm * 0.15 +
                     mono_norm * 0.10 +
                     cool_norm * 0.10 +
                     feas_norm * 0.10)

            opt_result = OptimizationResult(
                n_channels=int(n_ch),
                channel_length_mm=float(ch_len),
                flow_ml_min=float(flow),
                voltage_kv=float(voltage),
                frequency_khz=float(freq),
                pulse_width_ns=float(pw),
                production_mg_h=float(r.production_mg_h),
                wavelength_nm=float(r.wavelength_nm),
                size_nm=float(r.size_nm),
                power_w=float(r.power_w),
                energy_density_j_ml=float(r.energy_density_j_ml),
                monodispersity=float(r.monodispersity),
                in_spec=bool(r.in_spec),
                efficiency_score=float(r.efficiency_score),
                feasibility_score=float(r.feasibility_score),
                cost_score=float(r.cost_score),
                max_temp_C=float(r.max_temp_C),
                cooling_ok=bool(r.cooling_sufficient),
                specific_energy_kJ_g=float(spec_energy),
                production_per_watt=float(prod_per_watt),
                score_total=float(score),
            )

            results.append(opt_result)
            n_valid += 1
            if r.in_spec:
                n_in_spec += 1

        except Exception:
            n_errors += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{n_samples}] válidos: {n_valid}, "
                  f"in-spec: {n_in_spec}, errores: {n_errors}")

    print(f"\n  Evaluaciones completadas: {n_valid}/{n_samples}")
    print(f"  In-spec (λ=460±20nm): {n_in_spec} ({n_in_spec/max(1,n_valid)*100:.0f}%)")

    # Filtrar solo in-spec y cooling OK
    valid_results = [r for r in results if r.in_spec and r.cooling_ok]
    print(f"  Factibles (in-spec + cooling): {len(valid_results)}")

    if not valid_results:
        print("  ⚠ No se encontraron configuraciones factibles. Usando todos los resultados.")
        valid_results = sorted(results, key=lambda r: r.score_total, reverse=True)

    # Ordenar por score
    valid_results.sort(key=lambda r: r.score_total, reverse=True)

    # Top 10
    print(f"\n  TOP 10 CONFIGURACIONES OPTIMIZADAS:")
    print(f"  {'#':<4} {'Ch':<4} {'Len':<5} {'Flow':<6} {'V':<5} {'f':<5} {'PW':<5} "
          f"{'Prod':<8} {'λ':<6} {'P(W)':<6} {'E_sp':<7} {'Score':<7}")
    print("  " + "-" * 80)

    for i, r in enumerate(valid_results[:10]):
        print(f"  {i+1:<4} {r.n_channels:<4} {r.channel_length_mm:<5.0f} "
              f"{r.flow_ml_min:<6.0f} {r.voltage_kv:<5.0f} {r.frequency_khz:<5.0f} "
              f"{r.pulse_width_ns:<5.0f} {r.production_mg_h:<8.0f} "
              f"{r.wavelength_nm:<6.0f} {r.power_w:<6.0f} "
              f"{r.specific_energy_kJ_g:<7.0f} {r.score_total:<7.3f}")

    # Mejor configuración
    best = valid_results[0]
    current_prod = CURRENT_REACTOR['target_production_mg_h']

    print(f"\n  ★ MEJOR CONFIGURACIÓN ENCONTRADA:")
    print(f"    Canales:            {best.n_channels} (actual: {CURRENT_REACTOR['n_channels']})")
    print(f"    Longitud canal:     {best.channel_length_mm:.0f} mm (actual: {CURRENT_REACTOR['channel_length_mm']:.0f})")
    print(f"    Flujo:              {best.flow_ml_min:.0f} mL/min (actual: {CURRENT_REACTOR['flow_ml_min']:.0f})")
    print(f"    Voltaje:            {best.voltage_kv:.0f} kV (actual: {CURRENT_REACTOR['voltage_kv']:.0f})")
    print(f"    Frecuencia:         {best.frequency_khz:.0f} kHz (actual: {CURRENT_REACTOR['frequency_khz']:.0f})")
    print(f"    Ancho pulso:        {best.pulse_width_ns:.0f} ns (actual: {CURRENT_REACTOR['pulse_width_ns']:.0f})")
    print(f"    ─────────────────────────────────────")
    print(f"    Producción:         {best.production_mg_h:.0f} mg/h "
          f"({best.production_mg_h/current_prod*100:.0f}% del actual)")
    print(f"    λ emisión:          {best.wavelength_nm:.0f} nm")
    print(f"    Tamaño:             {best.size_nm:.2f} nm")
    print(f"    Potencia:           {best.power_w:.0f} W")
    print(f"    E específica:       {best.specific_energy_kJ_g:.0f} kJ/g")
    print(f"    Prod/Watt:          {best.production_per_watt:.1f} mg/(h·W)")
    print(f"    Monodispersidad:    {best.monodispersity:.2f}")
    print(f"    T máxima:           {best.max_temp_C:.0f}°C")
    print(f"    Score total:        {best.score_total:.3f}")

    improvement = (best.production_mg_h - current_prod) / current_prod * 100
    print(f"\n    Mejora producción:  {improvement:+.1f}%")

    return {
        'n_evaluated': n_valid,
        'n_in_spec': n_in_spec,
        'n_feasible': len(valid_results),
        'best': asdict(best),
        'top5': [asdict(r) for r in valid_results[:5]],
        'improvement_pct': float(improvement),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE SENSIBILIDAD
# ═══════════════════════════════════════════════════════════════════════════════

def sensitivity_analysis() -> Dict:
    """
    Analiza sensibilidad de la producción a cada parámetro individual.
    Mantiene todos los demás parámetros fijos en el valor actual.
    """
    print("\n" + "=" * 70)
    print("  ANÁLISIS DE SENSIBILIDAD")
    print("=" * 70)

    base_config = {
        'channel_width_mm': 2.0,
        'channel_height_mm': 0.5,
        'catalyst_type': "tio2_anatase",
        'catalyst_porosity': 0.60,
        'catalyst_surface_area_m2_g': 50.0,
        'catalyst_thickness_mm': 0.5,
        'catalyst_loading_mg_cm2': 2.0,
        'tio2_barrier': True,
        'tio2_barrier_phase': "anatase",
        'target_production_mg_h': 500.0,
    }

    sweeps = {
        'n_channels':        ([4, 6, 8, 10, 12, 16, 24, 32], 'n_channels'),
        'channel_length_mm': ([100, 150, 200, 250, 300, 400, 500], 'channel_length_mm'),
        'flow_ml_min':       ([1, 2, 3, 5, 8, 10, 15, 20, 30], 'liquid_flow_ml_min'),
        'voltage_kv':        ([6, 8, 10, 12, 15, 18, 20], 'voltage_kv'),
        'frequency_khz':     ([5, 10, 15, 20, 25, 30], 'frequency_khz'),
        'pulse_width_ns':    ([50, 100, 150, 200, 300, 400, 500], 'pulse_width_ns'),
    }

    # Valores base
    base_vals = {
        'n_channels': 8,
        'channel_length_mm': 300.0,
        'liquid_flow_ml_min': 5.0,
        'voltage_kv': 10.0,
        'frequency_khz': 20.0,
        'pulse_width_ns': 100.0,
    }

    results = {}

    for param_name, (values, config_key) in sweeps.items():
        prod_values = []
        lambda_values = []
        power_values = []
        spec_e_values = []

        for val in values:
            try:
                cfg = {config_key: float(val) if config_key != 'n_channels' else int(val)}
                # Merge con base
                for k, v in base_vals.items():
                    if k not in cfg:
                        cfg[k] = v

                params = ScaledReactorParameters(
                    topology=ScaleTopology.MULTI_CHANNEL,
                    n_channels=int(cfg.get('n_channels', 8)),
                    channel_length_mm=float(cfg.get('channel_length_mm', 300)),
                    liquid_flow_ml_min=float(cfg.get('liquid_flow_ml_min', 5)),
                    voltage_kv=float(cfg.get('voltage_kv', 10)),
                    frequency_khz=float(cfg.get('frequency_khz', 20)),
                    pulse_width_ns=float(cfg.get('pulse_width_ns', 100)),
                    **base_config,
                )

                designer = MillimetricReactorDesigner(params)
                r = designer.evaluate()
                prod_values.append(float(r.production_mg_h))
                lambda_values.append(float(r.wavelength_nm))
                power_values.append(float(r.power_w))
                spec_e_values.append(float(r.power_w / max(0.01, r.production_mg_h) * 3600))
            except Exception:
                prod_values.append(0.0)
                lambda_values.append(0.0)
                power_values.append(0.0)
                spec_e_values.append(999.0)

        # Calcular elasticidad (% cambio producción / % cambio parámetro)
        base_idx = values.index(base_vals.get(config_key, values[len(values)//2]))
        if base_idx < len(values) - 1 and prod_values[base_idx] > 0:
            dp = (prod_values[base_idx + 1] - prod_values[base_idx]) / prod_values[base_idx]
            dx = (values[base_idx + 1] - values[base_idx]) / values[base_idx]
            elasticity = dp / dx if abs(dx) > 1e-6 else 0
        else:
            elasticity = 0

        # Encontrar óptimo
        best_idx = np.argmax(prod_values)
        best_val = values[best_idx]
        best_prod = prod_values[best_idx]

        results[param_name] = {
            'values': [float(v) for v in values],
            'production': prod_values,
            'wavelength': lambda_values,
            'power': power_values,
            'specific_energy': spec_e_values,
            'elasticity': float(elasticity),
            'optimal_value': float(best_val),
            'optimal_production': float(best_prod),
            'base_value': float(base_vals.get(config_key, values[0])),
        }

        print(f"\n  {param_name}:")
        print(f"    Rango:       {min(values)} → {max(values)}")
        print(f"    Producción:  {min(prod_values):.0f} → {max(prod_values):.0f} mg/h")
        print(f"    Óptimo:      {best_val} → {best_prod:.0f} mg/h")
        print(f"    Elasticidad: {elasticity:.2f}")
        print(f"    Base actual: {base_vals.get(config_key, '?')}")

        # Mini-tabla
        print(f"    {'Valor':<10} {'Prod(mg/h)':<12} {'λ(nm)':<8} {'P(W)':<8}")
        print(f"    " + "-" * 40)
        for j, val in enumerate(values):
            marker = " ★" if j == best_idx else ""
            print(f"    {val:<10} {prod_values[j]:<12.0f} {lambda_values[j]:<8.0f} "
                  f"{power_values[j]:<8.0f}{marker}")

    # Ranking de sensibilidad
    print(f"\n  RANKING DE SENSIBILIDAD (por elasticidad):")
    ranked = sorted(results.items(), key=lambda x: abs(x[1]['elasticity']), reverse=True)
    for i, (name, data) in enumerate(ranked):
        print(f"    {i+1}. {name:<22} elasticidad = {data['elasticity']:+.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  INTEGRACIÓN: VALIDACIÓN TANGELO + REACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def cross_validate_tangelo_reactor() -> Dict:
    """
    Valida consistencia entre las predicciones Tangelo y el modelo del reactor.
    Verifica que el gap predicho por Tangelo produce la λ correcta.
    """
    print("\n" + "=" * 70)
    print("  VALIDACIÓN CRUZADA: TANGELO ↔ REACTOR")
    print("=" * 70)

    interface = TangeloInterface(use_tangelo=True, cache_results=False)

    # Barrer condiciones de operación y comparar predicciones
    temperatures = [293, 313, 333, 353, 373]  # K
    e_fields = [0, 1e5, 3e5, 5e5, 8e5, 1e6]  # V/m

    print(f"\n  Gap CQD: Tangelo vs Modelo de confinamiento")
    print(f"  {'T (K)':<8} {'E (kV/cm)':<12} {'Gap_Tg':<10} {'d_Tg':<8} "
          f"{'Gap_conf':<10} {'λ_Tg':<8} {'λ_conf':<8} {'Δλ%':<8}")
    print("  " + "-" * 75)

    discrepancies = []

    for T in temperatures:
        for E in e_fields:
            state = ChemicalState(
                temperature=float(T),
                pressure=101325,
                composition={"H2O": 0.95, "C_org": 0.05},
                electric_field=float(E)
            )

            params = interface.get_parameters(state)

            # Tangelo predictions
            gap_tg = params.cqd_gap_ev
            d_tg = params.cqd_size_nm
            lam_tg = EV_TO_NM / gap_tg

            # Confinement model prediction for same size
            gap_conf = E_BULK + A_CONF / (d_tg ** 2)
            lam_conf = EV_TO_NM / gap_conf

            delta_lam = abs(lam_tg - lam_conf) / lam_conf * 100
            discrepancies.append(delta_lam)

            if E in [0, 5e5, 1e6]:  # Print subset
                print(f"  {T:<8} {E/1e5:<12.1f} {gap_tg:<10.3f} {d_tg:<8.2f} "
                      f"{gap_conf:<10.3f} {lam_tg:<8.0f} {lam_conf:<8.0f} {delta_lam:<8.1f}")

    mean_disc = np.mean(discrepancies)
    max_disc = np.max(discrepancies)

    print(f"\n  Discrepancia media λ: {mean_disc:.2f}%")
    print(f"  Discrepancia máxima:  {max_disc:.2f}%")

    consistent = mean_disc < 5.0
    print(f"  {'✓' if consistent else '⚠'} Tangelo y modelo de confinamiento "
          f"{'son consistentes' if consistent else 'presentan discrepancias'}")

    if not consistent:
        print(f"\n  NOTA: Las discrepancias se deben a que _estimate_cqd_properties")
        print(f"  usa un modelo empírico simplificado (d depende de T y E) que")
        print(f"  no necesariamente sigue exactamente E_gap = E_bulk + A/d².")
        print(f"  Esto es esperado: Tangelo calcula d basado en condiciones de")
        print(f"  síntesis, luego calcula gap con el mismo modelo de confinamiento.")

    return {
        'mean_discrepancy_pct': float(mean_disc),
        'max_discrepancy_pct': float(max_disc),
        'consistent': consistent,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORTE FINAL
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(confinement, tangelo, production, optimization, sensitivity, cross_val):
    """Genera reporte completo de validación y optimización"""
    print("\n" + "═" * 70)
    print("  REPORTE FINAL: VALIDACIÓN Y OPTIMIZACIÓN")
    print("═" * 70)

    # 1. Resumen de validaciones
    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print("  │  RESUMEN DE VALIDACIONES                               │")
    print("  ├─────────────────────────────────────────────────────────┤")
    v1 = "✓ PASÓ" if confinement['model_valid'] else "⚠ AJUSTAR"
    v2 = "✓ PASÓ" if tangelo['kinetics_valid'] else "⚠ REVISAR"
    v3 = "✓ PASÓ" if production['all_valid'] else "⚠ DISCREPANCIA"
    v4 = "✓ PASÓ" if cross_val['consistent'] else "⚠ INCONSISTENTE"
    print(f"  │  1. Confinamiento cuántico:  {v1:<20}         │")
    print(f"  │     R² = {confinement['r2']:.4f}, Error medio = {confinement['mean_error_current']:.1f}%      │")
    print(f"  │  2. Química Tangelo:         {v2:<20}         │")
    print(f"  │     Gap consistencia = {tangelo['gap_consistency_pct']:.1f}%             │")
    print(f"  │  3. Producción reactor:      {v3:<20}         │")
    prod_result = production['result']
    print(f"  │     {prod_result['production_mg_h']:.0f} mg/h, λ={prod_result['wavelength_nm']:.0f}nm        │")
    print(f"  │  4. Cruzada Tangelo↔Reactor: {v4:<20}         │")
    print(f"  │     Discrepancia media = {cross_val['mean_discrepancy_pct']:.2f}%          │")
    print("  └─────────────────────────────────────────────────────────┘")

    # 2. Optimización
    if optimization:
        best = optimization['best']
        print("\n  ┌─────────────────────────────────────────────────────────┐")
        print("  │  RESULTADO DE OPTIMIZACIÓN                             │")
        print("  ├─────────────────────────────────────────────────────────┤")
        print(f"  │  Configuraciones evaluadas:  {optimization['n_evaluated']:>5}                   │")
        print(f"  │  Factibles:                  {optimization['n_feasible']:>5}                   │")
        print(f"  │  Mejora de producción:        {optimization['improvement_pct']:>+6.1f}%                │")
        print(f"  │                                                       │")
        print(f"  │  MEJOR CONFIGURACIÓN:                                 │")
        print(f"  │    Canales:     {best['n_channels']:<4}  (actual: 8)                   │")
        print(f"  │    Longitud:    {best['channel_length_mm']:<5.0f} mm  (actual: 300)             │")
        print(f"  │    Flujo:       {best['flow_ml_min']:<5.0f} mL/min  (actual: 5)          │")
        print(f"  │    Voltaje:     {best['voltage_kv']:<5.0f} kV  (actual: 10)              │")
        print(f"  │    Frecuencia:  {best['frequency_khz']:<5.0f} kHz  (actual: 20)            │")
        print(f"  │    Pulso:       {best['pulse_width_ns']:<5.0f} ns  (actual: 100)            │")
        print(f"  │                                                       │")
        print(f"  │    Producción:  {best['production_mg_h']:<6.0f} mg/h                      │")
        print(f"  │    λ emisión:   {best['wavelength_nm']:<5.0f} nm                         │")
        print(f"  │    Potencia:    {best['power_w']:<5.0f} W                           │")
        print(f"  │    Score:       {best['score_total']:<5.3f}                           │")
        print("  └─────────────────────────────────────────────────────────┘")

    # 3. Sensibilidad
    if sensitivity:
        print("\n  ┌─────────────────────────────────────────────────────────┐")
        print("  │  PARÁMETROS MÁS SENSIBLES                             │")
        print("  ├─────────────────────────────────────────────────────────┤")
        ranked = sorted(sensitivity.items(),
                        key=lambda x: abs(x[1]['elasticity']), reverse=True)
        for i, (name, data) in enumerate(ranked[:4]):
            opt = data['optimal_value']
            base = data['base_value']
            change = "↑" if opt > base else "↓" if opt < base else "="
            print(f"  │    {i+1}. {name:<18} e={data['elasticity']:+.3f}  "
                  f"óptimo={opt} {change}  │")
        print("  └─────────────────────────────────────────────────────────┘")

    # 4. Recomendaciones
    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print("  │  RECOMENDACIONES                                       │")
    print("  ├─────────────────────────────────────────────────────────┤")

    recs = []
    if confinement['recommendation'] == 'use_fitted':
        recs.append(f"  │  → Actualizar E_bulk a {confinement['fitted_params']['E_bulk']:.3f} "
                    f"y A_conf a {confinement['fitted_params']['A_conf']:.3f}")
    if not tangelo['kinetics_valid']:
        recs.append("  │  → Revisar energías de activación con Tangelo real")
    if optimization and optimization['improvement_pct'] > 5:
        b = optimization['best']
        recs.append(f"  │  → Considerar {b['n_channels']} canales × {b['channel_length_mm']:.0f}mm")
        recs.append(f"  │  → Ajustar flujo a {b['flow_ml_min']:.0f} mL/min, "
                    f"V={b['voltage_kv']:.0f}kV")
    if not recs:
        recs.append("  │  → Sistema validado, parámetros actuales son adecuados")

    for r in recs:
        print(f"{r:<58}│")
    print("  └─────────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validación y optimización de reactor CQD')
    parser.add_argument('--validate-only', action='store_true',
                        help='Solo ejecutar validaciones')
    parser.add_argument('--optimize-only', action='store_true',
                        help='Solo ejecutar optimización')
    args = parser.parse_args()

    print("═" * 70)
    print("  VALIDACIÓN DE GENERACIÓN CQD CON TANGELO + OPTIMIZACIÓN")
    print("  Reactor: Milireactor MC 8×300mm DBD (TiO2 anatase)")
    print("═" * 70)

    all_results = {}

    # === VALIDACIONES ===
    if not args.optimize_only:
        confinement = validate_confinement_model()
        all_results['confinement'] = confinement

        tangelo = validate_tangelo_chemistry()
        all_results['tangelo'] = tangelo

        production = validate_reactor_production()
        all_results['production'] = production

        cross_val = cross_validate_tangelo_reactor()
        all_results['cross_validation'] = cross_val

    # === OPTIMIZACIÓN ===
    optimization = None
    sensitivity_data = None

    if not args.validate_only:
        optimization = run_optimization()
        all_results['optimization'] = optimization

        sensitivity_data = sensitivity_analysis()
        all_results['sensitivity'] = sensitivity_data

    # === REPORTE ===
    if not args.optimize_only and not args.validate_only:
        generate_report(confinement, tangelo, production, optimization,
                        sensitivity_data, cross_val)
    elif args.validate_only:
        generate_report(confinement, tangelo, production, None, None, cross_val)

    # Guardar resultados
    output_file = OUTPUT_DIR / "validation_optimization_results.json"

    # Serializar todo a JSON (convertir numpy types)
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    print(f"\n  ✓ Resultados guardados: {output_file}")

    print("\n" + "═" * 70)
    print("  ✓ VALIDACIÓN Y OPTIMIZACIÓN COMPLETADAS")
    print("═" * 70)


if __name__ == "__main__":
    main()
