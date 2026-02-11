#!/usr/bin/env python3
"""
===============================================================================
  OPTIMIZACIÓN DEL REACTOR DBD CON CANTERA + TANGELO
  Cinética detallada de plasma + química cuántica para CQDs
===============================================================================

  Integra:
    - Cantera: cinética química detallada del plasma DBD
    - Tangelo: propiedades cuánticas de CQDs (gap, tamaño)
    - PyTorch: surrogate DNN para evaluación rápida (DeepFlame-style)
    - Reactor model: producción, térmica, clasificación

  Pipeline:
    1. Definir mecanismo de plasma en Cantera (H2O/C/N/O + e⁻)
    2. Simular cinética a condiciones del reactor DBD
    3. Extraer concentraciones de radicales (OH*, O*, H2O2)
    4. Alimentar modelo de producción CQD
    5. Optimizar parámetros con Bayesian optimization
    6. Entrenar surrogate DNN para evaluación rápida

  USO:
    python optimize_cantera_reactor.py
    python optimize_cantera_reactor.py --quick      # Solo cinética, sin surrogate
    python optimize_cantera_reactor.py --surrogate  # Entrenar DNN surrogate
"""

import numpy as np
import json
import sys
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chem_backend'))

from tangelo_interface import TangeloInterface, ChemicalState

OUTPUT_DIR = Path(__file__).parent / "optimization_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════

HARTREE_TO_EV = 27.2114
EV_TO_NM = 1240.0
E_BULK = 1.50
A_CONF = 7.26

# Parámetros del reactor actual
REACTOR_DEFAULTS = {
    'n_channels': 8,
    'channel_width_mm': 2.0,
    'channel_height_mm': 0.5,
    'channel_length_mm': 300.0,
    'flow_ml_min': 5.0,
    'voltage_kv': 10.0,
    'frequency_khz': 20.0,
    'Te_eV': 1.5,           # Temperatura electrónica
    'Tgas_K': 333.0,         # ~60°C
    'pressure_Pa': 101325.0,
    'precursor_conc_g_L': 2.0,  # Purín diluido
    'pulse_width_ns': 100.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  MECANISMO DE PLASMA EN CANTERA
# ═══════════════════════════════════════════════════════════════════════════════

def create_plasma_mechanism():
    """
    Crea un mecanismo de cinética de plasma DBD para síntesis de CQDs.

    Especies:
      - H2O, OH, H, O, H2O2, HO2 (agua y radicales)
      - C_org (precursor orgánico de purín)
      - C_rad (radical de carbono)
      - C_nuc (núcleo de CQD)
      - CQD (quantum dot formado)
      - N2, O2, Ar (gas de trabajo)

    Reacciones clave en plasma DBD:
      R1: e⁻ + H2O → OH + H + e⁻     (disociación por impacto electrónico)
      R2: OH + OH → H2O2               (recombinación)
      R3: e⁻ + C_org → C_rad + e⁻     (fragmentación del precursor)
      R4: C_rad + C_rad → C_nuc        (nucleación)
      R5: C_nuc + C_rad → CQD          (crecimiento)
      R6: CQD + OH → CQD_func          (funcionalización)
    """
    try:
        import cantera as ct

        # Crear solución ideal de gas con especies del plasma
        species_names = 'H2O OH H O H2O2 HO2 H2 O2 N2 AR'

        # Usar mecanismo GRI-Mech simplificado como base (H/O chemistry)
        # Cantera incluye gri30.yaml con especies H2O/OH/H/O/H2O2/etc.
        try:
            gas = ct.Solution('gri30.yaml')
            print("  ✓ Mecanismo GRI-Mech 3.0 cargado (53 especies, 325 reacciones)")
            mechanism_type = 'gri30'
        except Exception:
            # Fallback: mecanismo H2/O2 simplificado
            gas = ct.Solution('h2o2.yaml')
            print("  ✓ Mecanismo H2/O2 cargado")
            mechanism_type = 'h2o2'

        return gas, mechanism_type

    except ImportError:
        print("  ⚠ Cantera no disponible - usando modelo cinético simplificado")
        return None, 'simplified'


def simulate_plasma_chemistry(gas, reactor_params: Dict) -> Dict:
    """
    Simula la cinética del plasma DBD frío con Cantera.

    Enfoque: El plasma DBD genera radicales por impacto electrónico a Te~1.5eV,
    pero el gas permanece frío (Tgas < 60°C). Modelamos esto en dos fases:

    Fase 1: Calcular producción de radicales basada en potencia del plasma
            (modelo de sección eficaz de impacto electrónico)
    Fase 2: Simular recombinación térmica con Cantera a Tgas baja
            (OH + OH → H2O2, etc.)

    Esto evita la divergencia de GRI-Mech a temperaturas irreales.
    """
    import cantera as ct

    T = reactor_params.get('Tgas_K', 333.0)
    P = reactor_params.get('pressure_Pa', 101325.0)
    t_res = reactor_params.get('residence_time_s', 20.0)

    # ─── Fase 1: Producción de radicales por plasma ──────────────
    voltage_kv = reactor_params.get('voltage_kv', 10.0)
    freq_khz = reactor_params.get('frequency_khz', 20.0)
    Te_eV = reactor_params.get('Te_eV', 1.5)

    # Densidad de potencia (W/cm³) — modelo capacitivo DBD
    power_density = voltage_kv * freq_khz * 0.01

    # Fracción de disociación de H2O por impacto electrónico
    # σ(e⁻ + H2O → OH + H) ≈ 1e-16 cm² a Te=1.5eV
    # Tasa de disociación ∝ ne * σ * ve * [H2O]
    # ne ≈ 1e11 cm⁻³ (típico DBD atmosférico)
    ne = 1e11 * (power_density / 2.0)  # Escala con potencia
    sigma_dissoc = 1e-16  # cm²
    ve = np.sqrt(2 * Te_eV * 1.6e-19 / 9.1e-31) * 100  # cm/s
    n_H2O = P / (1.38e-23 * T) * 0.90 * 1e-6  # cm⁻³

    # Tasa de producción de OH (cm⁻³/s)
    R_OH = ne * sigma_dissoc * ve * n_H2O
    R_OH = min(R_OH, 1e18)  # Limitar a valores físicos

    # Fracción de OH producida en t_res
    OH_fraction = min(0.01, R_OH * t_res / n_H2O)  # Máx 1%

    # ─── Fase 2: Recombinación térmica con Cantera ───────────────
    # Iniciar con radicales producidos por el plasma
    initial_comp = (f'H2O:{0.90 - OH_fraction*2}, OH:{OH_fraction}, '
                    f'H:{OH_fraction*0.5}, O:{OH_fraction*0.3}, '
                    f'H2:{OH_fraction*0.2}, '
                    f'N2:0.05, O2:0.04, AR:0.01')

    gas.TPX = T, P, initial_comp

    # Reactor a presión constante (sistema abierto a la atmósfera)
    reactor = ct.IdealGasConstPressureReactor(gas, energy='off')  # Isotérmico (plasma frío)
    sim = ct.ReactorNet([reactor])
    sim.rtol = 1e-8
    sim.atol = 1e-12

    # Evolucionar — solo recombinación de radicales (sin inyección de calor)
    n_steps = 50
    dt = t_res / n_steps
    times = []
    temperatures = []
    species_history = {}

    key_species = ['OH', 'H', 'O', 'H2O2', 'HO2', 'H2O', 'H2', 'O2']
    for sp in key_species:
        if sp in gas.species_names:
            species_history[sp] = []

    for i in range(n_steps):
        t = (i + 1) * dt
        try:
            sim.advance(t)
        except Exception:
            break
        times.append(t)
        temperatures.append(reactor.T)

        for sp in species_history:
            idx = gas.species_index(sp)
            species_history[sp].append(gas.X[idx])

    # Concentraciones finales
    final_concentrations = {}
    for sp in species_history:
        if species_history[sp]:
            final_concentrations[sp] = float(species_history[sp][-1])
        else:
            final_concentrations[sp] = 0.0

    # OH en cm⁻³ (combina plasma generation + Cantera recombination)
    n_total = P / (1.38e-23 * T) * 1e-6  # moléculas/cm³
    OH_cm3 = final_concentrations.get('OH', OH_fraction) * n_total

    return {
        'final_T_K': float(T),
        'final_T_C': float(T - 273.15),
        'final_concentrations': final_concentrations,
        'OH_cm3': float(OH_cm3),
        'H2O2_fraction': float(final_concentrations.get('H2O2', 0)),
        'OH_fraction_plasma': float(OH_fraction),
        'power_density_W_cm3': float(power_density),
        'ne_cm3': float(ne),
        'R_OH_cm3_s': float(R_OH),
        'residence_time_s': float(t_res),
        'species_history': {k: [float(v) for v in vals] for k, vals in species_history.items()},
        'times': [float(t) for t in times],
        'temperatures': [float(t) for t in temperatures],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELO INTEGRADO: CANTERA + TANGELO + REACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_reactor_config(params: Dict, gas, tangelo_interface) -> Dict:
    """
    Evaluación completa de una configuración del reactor usando
    Cantera (cinética) + Tangelo (propiedades cuánticas) + modelo de producción.
    """
    # ─── 1. Geometría ─────────────────────────────────────────────
    n_ch = params.get('n_channels', 8)
    ch_w = params.get('channel_width_mm', 2.0)
    ch_h = params.get('channel_height_mm', 0.5)
    ch_l = params.get('channel_length_mm', 300.0)
    flow = params.get('flow_ml_min', 5.0)

    # Volumen y tiempo de residencia
    ch_vol_ml = ch_w * ch_h * ch_l / 1000.0
    liquid_fraction = 0.6
    total_liquid_ml = ch_vol_ml * liquid_fraction * n_ch
    flow_per_channel = flow / n_ch
    liquid_depth = ch_h * liquid_fraction
    v_mm_s = (flow_per_channel / 60.0 * 1000.0) / (ch_w * liquid_depth)
    t_res = ch_l / v_mm_s if v_mm_s > 0 else 999

    # Área de plasma
    plasma_area_cm2 = ch_w * ch_l * n_ch / 100.0

    # ─── 2. Eléctrico ────────────────────────────────────────────
    voltage_kv = params.get('voltage_kv', 10.0)
    freq_khz = params.get('frequency_khz', 20.0)
    gap_mm = ch_h * (1 - liquid_fraction)

    E_field = voltage_kv * 1e3 / (gap_mm * 1e-3)  # V/m

    # Calibrated DBD power: specific_power (W/cm²) scales with V² and f
    # 0.25 W/cm² at 10kV, 20kHz → ~12W total for 8ch×300mm (matches literature)
    specific_power = 0.25 * (voltage_kv / 10.0)**2 * (freq_khz / 20.0)
    area_per_ch_cm2 = (ch_w / 10.0) * (ch_l / 10.0)
    power_per_channel = specific_power * area_per_ch_cm2
    power_w = power_per_channel * n_ch
    energy_density_j_ml = power_w / (flow / 60.0) if flow > 0 else 0

    # ─── 3. Cinética Cantera ──────────────────────────────────────
    cantera_results = None
    OH_cm3 = 1e15  # Default
    H2O2_frac = 0.001

    if gas is not None:
        try:
            reactor_params = {
                'Tgas_K': params.get('Tgas_K', 333.0),
                'pressure_Pa': params.get('pressure_Pa', 101325.0),
                'residence_time_s': t_res,
                'voltage_kv': voltage_kv,
                'frequency_khz': freq_khz,
            }
            cantera_results = simulate_plasma_chemistry(gas, reactor_params)
            OH_cm3 = cantera_results['OH_cm3']
            H2O2_frac = cantera_results['H2O2_fraction']
        except Exception as e:
            cantera_results = {'error': str(e)}

    # ─── 4. Propiedades CQD (Tangelo) ────────────────────────────
    state = ChemicalState(
        temperature=params.get('Tgas_K', 333.0),
        pressure=params.get('pressure_Pa', 101325.0),
        composition={"H2O": 0.95, "C_org": 0.05},
        electric_field=E_field
    )
    tangelo_params = tangelo_interface.get_parameters(state)

    # CQD size from energy density (more physical for reactor optimization)
    # Higher energy density → more nucleation → smaller particles
    # Calibrated: d=2.5nm at E_density=450 J/mL, t_res=20s
    E_opt = 450.0  # J/mL optimal energy density
    if energy_density_j_ml > 10:
        size_nm = 2.5 * (E_opt / energy_density_j_ml)**0.15 * (t_res / 20.0)**0.08
        size_nm = max(1.5, min(5.0, size_nm))
        gap_ev = E_BULK + A_CONF / size_nm**2
        wavelength_nm = EV_TO_NM / gap_ev
    else:
        size_nm = tangelo_params.cqd_size_nm
        gap_ev = tangelo_params.cqd_gap_ev
        wavelength_nm = EV_TO_NM / gap_ev

    # ─── 5. Producción ───────────────────────────────────────────
    # Concentración base con correcciones
    base_conc = 0.3  # mg/mL

    # Factor de energía (óptimo 300-600 J/mL)
    optimal_energy = 450
    if energy_density_j_ml < 100:
        energy_factor = energy_density_j_ml / 100 * 0.3
    elif energy_density_j_ml > 1000:
        energy_factor = 0.5
    else:
        energy_factor = np.exp(-((energy_density_j_ml - optimal_energy) / 300) ** 2)

    # Factor de residencia (óptimo 10-30 s)
    optimal_res = 20
    if t_res < 3:
        res_factor = t_res / 3 * 0.3
    elif t_res > 60:
        res_factor = 0.5
    else:
        res_factor = np.exp(-((t_res - optimal_res) / 20) ** 2)

    # Factor de área (normalizado a 5 cm²)
    area_factor = min(2.0, plasma_area_cm2 / 5.0)

    # Factor de radicales (Cantera-enhanced)
    OH_ref = 1e15
    radical_factor = 1.0 + 0.15 * (OH_cm3 / OH_ref - 1.0)
    radical_factor = max(0.7, min(1.5, radical_factor))

    # Factor catalítico TiO2
    catalyst_factor = 1.35  # Sinergia plasma-fotocatálisis

    # Producción final
    concentration = (base_conc * energy_factor * res_factor * area_factor *
                     radical_factor * catalyst_factor)
    concentration = max(0.01, min(3.0, concentration))
    production_mg_h = concentration * flow * 60

    # ─── 6. Calidad ──────────────────────────────────────────────
    in_spec = abs(wavelength_nm - 460) < 20
    monodispersity = 0.85 if t_res > 5 else 0.60

    # ─── 7. Térmico ─────────────────────────────────────────────
    # Cooling: forced water convection through milli-channel walls
    heat_gen = power_w * 0.30  # 30% of electrical power → heat
    A_cooling = n_ch * (ch_l * 1e-3) * ((ch_w + 2 * ch_h) * 1e-3)  # m²
    h_conv = 300  # W/(m²·K) laminar forced convection in milli-channels
    cooling_capacity = h_conv * A_cooling  # W/K
    delta_T = heat_gen / max(0.1, cooling_capacity)
    max_temp_C = params.get('Tgas_K', 333) - 273.15 + delta_T
    cooling_ok = max_temp_C < 70  # <70°C safe for CQD synthesis (water-based)

    # ─── 8. Score multi-objetivo ─────────────────────────────────
    prod_norm = min(1.0, production_mg_h / 1000.0)
    quality_norm = 1.0 if in_spec else 0.3
    efficiency_norm = min(1.0, 1.0 / (1.0 + (power_w / max(0.01, production_mg_h) * 3600) / 500))
    cool_norm = 1.0 if cooling_ok else 0.2

    score = (prod_norm * 0.35 + quality_norm * 0.30 +
             efficiency_norm * 0.20 + cool_norm * 0.15)

    return {
        # Parámetros
        'n_channels': n_ch,
        'channel_length_mm': ch_l,
        'flow_ml_min': flow,
        'voltage_kv': voltage_kv,
        'frequency_khz': freq_khz,
        # Geometría
        'plasma_area_cm2': float(plasma_area_cm2),
        'residence_time_s': float(t_res),
        'energy_density_j_ml': float(energy_density_j_ml),
        # Producción
        'production_mg_h': float(production_mg_h),
        'concentration_mg_ml': float(concentration),
        'wavelength_nm': float(wavelength_nm),
        'size_nm': float(size_nm),
        'gap_ev': float(gap_ev),
        'monodispersity': float(monodispersity),
        'in_spec': bool(in_spec),
        # Factores
        'energy_factor': float(energy_factor),
        'residence_factor': float(res_factor),
        'area_factor': float(area_factor),
        'radical_factor': float(radical_factor),
        'catalyst_factor': float(catalyst_factor),
        # Cantera
        'OH_cm3': float(OH_cm3),
        'H2O2_fraction': float(H2O2_frac),
        'cantera_available': cantera_results is not None and 'error' not in (cantera_results or {}),
        # Eléctrico
        'power_w': float(power_w),
        'E_field_V_m': float(E_field),
        # Térmico
        'max_temp_C': float(max_temp_C),
        'cooling_ok': bool(cooling_ok),
        # Score
        'score': float(score),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZACIÓN BAYESIANA
# ═══════════════════════════════════════════════════════════════════════════════

def bayesian_optimization(gas, tangelo_interface, n_iterations: int = 200) -> Dict:
    """
    Optimización Bayesiana del reactor usando Cantera + Tangelo.
    """
    print("\n" + "=" * 70)
    print("  OPTIMIZACIÓN BAYESIANA CON CANTERA + TANGELO")
    print("=" * 70)

    # Espacio de búsqueda
    param_space = {
        'n_channels':        (4, 32),
        'channel_length_mm': (150, 500),
        'flow_ml_min':       (2, 20),
        'voltage_kv':        (8, 20),
        'frequency_khz':     (10, 30),
    }

    best_score = -1
    best_params = None
    best_result = None
    all_results = []

    print(f"\n  Iteraciones: {n_iterations}")
    print(f"  Parámetros: {list(param_space.keys())}")

    np.random.seed(42)

    for i in range(n_iterations):
        # Muestreo (aleatorio en primeras 50, luego guiado por mejores)
        if i < 50 or best_params is None:
            # Exploración
            params = {
                'n_channels': int(np.random.uniform(*param_space['n_channels'])),
                'channel_length_mm': np.random.uniform(*param_space['channel_length_mm']),
                'flow_ml_min': np.random.uniform(*param_space['flow_ml_min']),
                'voltage_kv': np.random.uniform(*param_space['voltage_kv']),
                'frequency_khz': np.random.uniform(*param_space['frequency_khz']),
            }
        else:
            # Explotación: perturbar mejor configuración
            params = {
                'n_channels': max(4, min(32, int(best_params['n_channels'] +
                                                  np.random.randint(-4, 5)))),
                'channel_length_mm': max(150, min(500, best_params['channel_length_mm'] +
                                                   np.random.uniform(-50, 50))),
                'flow_ml_min': max(2, min(20, best_params['flow_ml_min'] +
                                           np.random.uniform(-3, 3))),
                'voltage_kv': max(8, min(20, best_params['voltage_kv'] +
                                          np.random.uniform(-2, 2))),
                'frequency_khz': max(10, min(30, best_params['frequency_khz'] +
                                              np.random.uniform(-5, 5))),
            }

        try:
            result = evaluate_reactor_config(params, gas, tangelo_interface)
            all_results.append(result)

            if result['score'] > best_score:
                best_score = result['score']
                best_params = params
                best_result = result

                if (i + 1) <= 10 or (i + 1) % 50 == 0:
                    print(f"  [{i+1:4d}] ★ Score={result['score']:.3f} "
                          f"Prod={result['production_mg_h']:.0f}mg/h "
                          f"λ={result['wavelength_nm']:.0f}nm "
                          f"P={result['power_w']:.0f}W")
        except Exception:
            pass

        if (i + 1) % 50 == 0 and (i + 1) > 10:
            print(f"  [{i+1:4d}]   Best so far: Score={best_score:.3f} "
                  f"Prod={best_result['production_mg_h']:.0f}mg/h")

    # Resultados
    print(f"\n  ★ MEJOR CONFIGURACIÓN (Cantera + Tangelo):")
    print(f"    Canales:        {best_params['n_channels']}")
    print(f"    Longitud:       {best_params['channel_length_mm']:.0f} mm")
    print(f"    Flujo:          {best_params['flow_ml_min']:.1f} mL/min")
    print(f"    Voltaje:        {best_params['voltage_kv']:.1f} kV")
    print(f"    Frecuencia:     {best_params['frequency_khz']:.1f} kHz")
    print(f"    ─────────────────────────────────────")
    print(f"    Producción:     {best_result['production_mg_h']:.0f} mg/h")
    print(f"    λ emisión:      {best_result['wavelength_nm']:.0f} nm")
    print(f"    Tamaño CQD:     {best_result['size_nm']:.2f} nm")
    print(f"    Potencia:       {best_result['power_w']:.0f} W")
    print(f"    E específica:   {best_result['power_w']/max(0.01, best_result['production_mg_h'])*3600:.0f} kJ/g")
    print(f"    OH*:            {best_result['OH_cm3']:.2e} cm⁻³")
    print(f"    E. densidad:    {best_result['energy_density_j_ml']:.0f} J/mL")
    print(f"    T máxima:       {best_result['max_temp_C']:.0f}°C")
    print(f"    In-spec:        {'Sí' if best_result['in_spec'] else 'No'}")
    print(f"    Score:          {best_result['score']:.3f}")

    # Top 5
    valid = [r for r in all_results if r['in_spec'] and r['cooling_ok']]
    valid.sort(key=lambda r: r['score'], reverse=True)

    print(f"\n  TOP 5 FACTIBLES (in-spec + cooling):")
    print(f"  {'#':<4} {'Ch':<4} {'Len':<5} {'Flow':<6} {'V':<5} {'f':<5} "
          f"{'Prod':<8} {'λ':<6} {'Score':<7}")
    print("  " + "-" * 55)

    for i, r in enumerate(valid[:5]):
        print(f"  {i+1:<4} {r['n_channels']:<4} {r['channel_length_mm']:<5.0f} "
              f"{r['flow_ml_min']:<6.1f} {r['voltage_kv']:<5.1f} "
              f"{r['frequency_khz']:<5.1f} {r['production_mg_h']:<8.0f} "
              f"{r['wavelength_nm']:<6.0f} {r['score']:<7.3f}")

    return {
        'best_params': best_params,
        'best_result': best_result,
        'top5': [r for r in valid[:5]],
        'n_iterations': n_iterations,
        'n_feasible': len(valid),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DNN SURROGATE (DeepFlame-style)
# ═══════════════════════════════════════════════════════════════════════════════

def train_surrogate(all_results: List[Dict]) -> Optional[Dict]:
    """
    Entrena un modelo DNN surrogate al estilo DeepFlame
    para evaluación rápida del reactor.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("  ⚠ PyTorch no disponible - saltando surrogate DNN")
        return None

    print("\n" + "=" * 70)
    print("  DNN SURROGATE (DeepFlame-style)")
    print("=" * 70)

    # Preparar datos
    feature_keys = ['n_channels', 'channel_length_mm', 'flow_ml_min',
                     'voltage_kv', 'frequency_khz']
    target_keys = ['production_mg_h', 'wavelength_nm', 'power_w',
                    'energy_density_j_ml', 'max_temp_C']

    X = np.array([[r[k] for k in feature_keys] for r in all_results], dtype=np.float32)
    Y = np.array([[r[k] for k in target_keys] for r in all_results], dtype=np.float32)

    # Normalizar
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-8
    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std

    # Split
    n_train = int(0.8 * len(X))
    idx = np.random.permutation(len(X))
    X_train = torch.tensor(X_norm[idx[:n_train]])
    Y_train = torch.tensor(Y_norm[idx[:n_train]])
    X_test = torch.tensor(X_norm[idx[n_train:]])
    Y_test = torch.tensor(Y_norm[idx[n_train:]])

    # Modelo
    class ReactorDNN(nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_out),
            )

        def forward(self, x):
            return self.net(x)

    model = ReactorDNN(len(feature_keys), len(target_keys))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"  Device: {device}")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Modelo: {sum(p.numel() for p in model.parameters())} parámetros")

    # Entrenar
    epochs = 500
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, Y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_loss = criterion(test_pred, Y_test)
            print(f"  Epoch {epoch+1:4d}: Train MSE={loss.item():.4f}, "
                  f"Test MSE={test_loss.item():.4f}")

    # Evaluación final
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test).cpu().numpy()
        Y_true = Y_test.cpu().numpy()

    # Desnormalizar
    Y_pred_real = Y_pred * Y_std + Y_mean
    Y_true_real = Y_true * Y_std + Y_mean

    print(f"\n  Precisión del surrogate:")
    print(f"  {'Output':<25} {'MAE':<12} {'RMSE':<12} {'R²':<10}")
    print("  " + "-" * 55)

    metrics = {}
    for i, name in enumerate(target_keys):
        mae = np.mean(np.abs(Y_pred_real[:, i] - Y_true_real[:, i]))
        rmse = np.sqrt(np.mean((Y_pred_real[:, i] - Y_true_real[:, i]) ** 2))
        ss_res = np.sum((Y_true_real[:, i] - Y_pred_real[:, i]) ** 2)
        ss_tot = np.sum((Y_true_real[:, i] - np.mean(Y_true_real[:, i])) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        print(f"  {name:<25} {mae:<12.2f} {rmse:<12.2f} {r2:<10.4f}")
        metrics[name] = {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}

    # Guardar modelo
    model_data = {
        'model_state': model.state_dict(),
        'X_mean': X_mean.tolist(),
        'X_std': X_std.tolist(),
        'Y_mean': Y_mean.tolist(),
        'Y_std': Y_std.tolist(),
        'feature_keys': feature_keys,
        'target_keys': target_keys,
    }
    model_path = OUTPUT_DIR / "reactor_surrogate_dnn.pt"
    try:
        torch.save(model_data, model_path)
        print(f"\n  ✓ Modelo guardado: {model_path}")
    except (RuntimeError, OSError):
        model_path = Path("/tmp/reactor_surrogate_dnn.pt")
        torch.save(model_data, model_path)
        print(f"\n  ✓ Modelo guardado: {model_path} (fallback)")

    return {
        'metrics': metrics,
        'device': str(device),
        'model_path': str(model_path),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Solo cinética Cantera')
    parser.add_argument('--surrogate', action='store_true', help='Entrenar DNN surrogate')
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()

    print("═" * 70)
    print("  OPTIMIZACIÓN DEL REACTOR DBD CON CANTERA + TANGELO")
    print("  Reactor: Milireactor MC 8×300mm (TiO2 anatase)")
    print("═" * 70)

    # Inicializar herramientas
    print("\n→ Inicializando...")
    gas, mechanism_type = create_plasma_mechanism()
    tangelo_interface = TangeloInterface(use_tangelo=True, cache_results=False)

    # ─── Validación rápida con Cantera ───────────────────────────
    if gas is not None:
        print("\n→ Simulación de cinética de plasma con Cantera...")
        cantera_result = simulate_plasma_chemistry(gas, REACTOR_DEFAULTS)
        print(f"  T final:       {cantera_result['final_T_C']:.1f}°C")
        print(f"  OH*:           {cantera_result['OH_cm3']:.2e} cm⁻³")
        print(f"  H2O2:          {cantera_result['H2O2_fraction']:.2e}")
        print(f"  t_residencia:  {cantera_result['residence_time_s']:.1f} s")

    # ─── Evaluación del reactor actual ───────────────────────────
    print("\n→ Evaluando configuración actual...")
    current = evaluate_reactor_config(REACTOR_DEFAULTS, gas, tangelo_interface)
    print(f"  Producción:    {current['production_mg_h']:.0f} mg/h")
    print(f"  λ emisión:     {current['wavelength_nm']:.0f} nm")
    print(f"  Potencia:      {current['power_w']:.0f} W")
    print(f"  E. densidad:   {current['energy_density_j_ml']:.0f} J/mL")
    print(f"  Score:         {current['score']:.3f}")

    if args.quick:
        print("\n✓ Modo rápido completado")
        return

    # ─── Optimización Bayesiana ──────────────────────────────────
    opt_result = bayesian_optimization(gas, tangelo_interface, n_iterations=args.iterations)

    # ─── DNN Surrogate ───────────────────────────────────────────
    surrogate_result = None
    if args.surrogate or True:  # Siempre entrenar si hay PyTorch
        # Generar dataset más grande para el surrogate
        print("\n→ Generando dataset para surrogate DNN...")
        dataset = []
        for _ in range(1000):
            params = {
                'n_channels': int(np.random.uniform(4, 32)),
                'channel_length_mm': np.random.uniform(150, 500),
                'flow_ml_min': np.random.uniform(2, 20),
                'voltage_kv': np.random.uniform(8, 20),
                'frequency_khz': np.random.uniform(10, 30),
            }
            try:
                result = evaluate_reactor_config(params, gas, tangelo_interface)
                dataset.append(result)
            except Exception:
                pass
        print(f"  Dataset: {len(dataset)} muestras")

        surrogate_result = train_surrogate(dataset)

    # ─── Guardar resultados ──────────────────────────────────────
    output = {
        'current_config': current,
        'optimization': {
            'best_params': opt_result['best_params'],
            'best_result': opt_result['best_result'],
            'n_feasible': opt_result['n_feasible'],
        },
        'surrogate': surrogate_result,
        'mechanism_type': mechanism_type,
    }

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_file = OUTPUT_DIR / "cantera_optimization_results.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(convert(output), f, indent=2)
        print(f"\n  ✓ Resultados: {output_file}")
    except (PermissionError, OSError):
        output_file = Path("/tmp/cantera_optimization_results.json")
        with open(output_file, 'w') as f:
            json.dump(convert(output), f, indent=2)
        print(f"\n  ✓ Resultados: {output_file} (fallback)")

    # ─── Comparación final ───────────────────────────────────────
    print("\n" + "═" * 70)
    print("  COMPARACIÓN: ACTUAL vs OPTIMIZADO (Cantera + Tangelo)")
    print("═" * 70)

    best = opt_result['best_result']
    print(f"\n  {'Parámetro':<25} {'Actual':<15} {'Optimizado':<15} {'Cambio'}")
    print("  " + "-" * 65)

    comparisons = [
        ('Canales', current['n_channels'], best['n_channels'], ''),
        ('Longitud (mm)', f"{REACTOR_DEFAULTS['channel_length_mm']:.0f}",
         f"{opt_result['best_params']['channel_length_mm']:.0f}", ''),
        ('Flujo (mL/min)', f"{REACTOR_DEFAULTS['flow_ml_min']:.0f}",
         f"{opt_result['best_params']['flow_ml_min']:.1f}", ''),
        ('Voltaje (kV)', f"{REACTOR_DEFAULTS['voltage_kv']:.0f}",
         f"{opt_result['best_params']['voltage_kv']:.1f}", ''),
        ('Frecuencia (kHz)', f"{REACTOR_DEFAULTS['frequency_khz']:.0f}",
         f"{opt_result['best_params']['frequency_khz']:.1f}", ''),
        ('─' * 20, '─' * 10, '─' * 10, ''),
        ('Producción (mg/h)', f"{current['production_mg_h']:.0f}",
         f"{best['production_mg_h']:.0f}",
         f"{(best['production_mg_h']/current['production_mg_h']-1)*100:+.0f}%"),
        ('λ emisión (nm)', f"{current['wavelength_nm']:.0f}",
         f"{best['wavelength_nm']:.0f}", ''),
        ('Potencia (W)', f"{current['power_w']:.0f}",
         f"{best['power_w']:.0f}", ''),
        ('Score', f"{current['score']:.3f}",
         f"{best['score']:.3f}", ''),
    ]

    for name, curr, opt, change in comparisons:
        print(f"  {name:<25} {str(curr):<15} {str(opt):<15} {change}")

    print("\n" + "═" * 70)
    print("  ✓ OPTIMIZACIÓN CON CANTERA + TANGELO COMPLETADA")
    print("═" * 70)


if __name__ == "__main__":
    main()
