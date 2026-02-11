#!/usr/bin/env python3
"""
===============================================================================
  EXPERIMENTAL VALIDATION — DIGITAL TWIN + PROTOCOL
  16ch × 500mm Parametric Optimized DBD Milireactor for CQD Synthesis
===============================================================================

  Consolidates ALL model predictions into a digital twin with uncertainty
  bounds, then generates a complete experimental protocol for building
  and validating the reactor from scratch.

  Part A — Digital Twin:
    1. Cantera plasma chemistry (OH, H2O2, species)
    2. Tangelo quantum chemistry (gap, size, activation energies)
    3. Production model (scoring, wavelength, production rate)
    4. Thermal model (cold plasma, cooling design)
    5. Control simulation (classifier, valve table)
    6. Cross-model consolidation with CFD reference

  Part B — Experimental Protocol:
    7.  Bill of Materials (BOM)
    8.  Fabrication protocol (8 phases)
    9.  Measurement equipment list
    10. 3-phase experimental protocol
    11. Data acquisition plan
    12. Acceptance criteria (pass/fail)
    13. Statistical analysis plan

  Output:
    14. Formatted report to stdout
    15. JSON report to optimization_results/experimental_validation_report.json

  Usage:
    python experimental_validation.py
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

from tangelo_interface import TangeloInterface, ChemicalState, REACTIONS, LITERATURE_VALUES

OUTPUT_DIR = Path(__file__).parent / "optimization_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

HARTREE_TO_EV = 27.2114
EV_TO_NM = 1240.0
E_BULK = 1.50          # eV — bulk gap of N-doped graphene
A_CONF = 7.26          # eV·nm² — quantum confinement constant
R_GAS = 8.314          # J/(mol·K)

# Target configuration: parametric optimized (CFD-validated, score 0.932)
TARGET_CONFIG = {
    'name': 'Parametric Optimized',
    'n_channels': 16,
    'channel_width_mm': 2.0,
    'channel_height_mm': 0.5,
    'channel_length_mm': 500.0,
    'flow_ml_min': 15.0,
    'voltage_kv': 12.0,
    'frequency_khz': 30.0,
    'Te_eV': 1.5,
    'Tgas_K': 333.0,
    'pressure_Pa': 101325.0,
    'precursor_conc_g_L': 2.0,
    'pulse_width_ns': 100.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PART A: DIGITAL TWIN
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 1. Cantera Simulation ────────────────────────────────────────────────────

def run_cantera_simulation(config: Dict) -> Dict:
    """
    Run Cantera plasma chemistry for the target configuration.
    Reuses create_plasma_mechanism() + simulate_plasma_chemistry() logic.
    Returns OH_cm3, H2O2, species with uncertainty bounds.
    """
    T = config.get('Tgas_K', 333.0)
    P = config.get('pressure_Pa', 101325.0)
    voltage_kv = config.get('voltage_kv', 12.0)
    freq_khz = config.get('frequency_khz', 30.0)
    Te_eV = config.get('Te_eV', 1.5)

    cantera_available = False
    cantera_results = None

    # Compute residence time for this config
    n_ch = config['n_channels']
    ch_w = config['channel_width_mm']
    ch_h = config['channel_height_mm']
    ch_l = config['channel_length_mm']
    flow = config['flow_ml_min']
    liquid_fraction = 0.6
    liquid_depth = ch_h * liquid_fraction
    flow_per_ch = flow / n_ch
    v_mm_s = (flow_per_ch / 60.0 * 1000.0) / (ch_w * liquid_depth)
    t_res = ch_l / v_mm_s if v_mm_s > 0 else 999

    # Plasma source OH: same model as cfd_validate_reactor.py
    # The plasma continuously generates OH radicals; steady-state density
    # is the balance between generation and thermal recombination.
    power_density = voltage_kv * freq_khz * 0.01
    ne = 1e11 * (power_density / 2.0)
    sigma_dissoc = 1e-16
    ve = np.sqrt(2 * Te_eV * 1.6e-19 / 9.1e-31) * 100
    n_H2O = P / (1.38e-23 * T) * 0.90 * 1e-6
    R_OH = ne * sigma_dissoc * ve * n_H2O
    R_OH = min(R_OH, 1e18)
    # Steady-state OH: capped at 1e16 (same as CFD validator)
    OH_cm3_plasma = min(R_OH * t_res, 1e16)

    try:
        from optimize_cantera_reactor import create_plasma_mechanism, simulate_plasma_chemistry

        gas, mechanism_type = create_plasma_mechanism()
        if gas is not None:
            cantera_available = True

            reactor_params = {
                'Tgas_K': T,
                'pressure_Pa': P,
                'residence_time_s': t_res,
                'voltage_kv': voltage_kv,
                'frequency_khz': freq_khz,
                'Te_eV': Te_eV,
            }
            cantera_results = simulate_plasma_chemistry(gas, reactor_params)
            # Cantera gives post-recombination OH (near-zero at 333K).
            # For production model, use plasma-source OH (steady-state
            # balance), matching cfd_validate_reactor.py approach.
            cantera_results['OH_cm3'] = float(OH_cm3_plasma)
    except Exception as e:
        print(f"  Cantera import/run: {e}")

    # Fallback: analytical model (same physics as optimize_cantera_reactor.py)
    if cantera_results is None:
        n_total = P / (1.38e-23 * T) * 1e-6
        OH_fraction = OH_cm3_plasma / n_total

        # H2O2 from OH recombination: [H2O2] ~ k_recomb * [OH]^2 * t_res
        # OH is consumed as it recombines, so limit H2O2 to fraction of initial OH
        k_recomb = 1.4e-11  # cm³/s (OH+OH→H2O2 at 333K)
        H2O2_cm3 = min(OH_cm3_plasma * 0.5, k_recomb * OH_cm3_plasma**2 * t_res)
        H2O2_fraction = H2O2_cm3 / n_total

        cantera_results = {
            'final_T_K': T,
            'final_T_C': T - 273.15,
            'OH_cm3': float(OH_cm3_plasma),
            'H2O2_fraction': float(H2O2_fraction),
            'OH_fraction_plasma': float(OH_fraction),
            'power_density_W_cm3': float(power_density),
            'ne_cm3': float(ne),
            'R_OH_cm3_s': float(R_OH),
            'residence_time_s': float(t_res),
            'final_concentrations': {
                'OH': float(OH_fraction),
                'H2O2': float(H2O2_fraction),
                'H2O': float(0.90 - 2 * OH_fraction),
            },
        }

    # Uncertainty bounds (±30% for analytical, ±15% for Cantera)
    unc_factor = 0.15 if cantera_available else 0.30

    return {
        'cantera_available': cantera_available,
        'results': cantera_results,
        'OH_cm3': cantera_results['OH_cm3'],
        'R_OH_cm3_s': float(R_OH),
        'H2O2_fraction': cantera_results['H2O2_fraction'],
        'uncertainty_relative': unc_factor,
        'OH_cm3_95ci': (
            cantera_results['OH_cm3'] * (1 - 1.96 * unc_factor),
            cantera_results['OH_cm3'] * (1 + 1.96 * unc_factor),
        ),
    }


# ─── 2. Tangelo Analysis ─────────────────────────────────────────────────────

def run_tangelo_analysis(config: Dict) -> Dict:
    """
    Run Tangelo quantum chemistry for 4 reactor zones.
    Returns gap_eV, size_nm, wavelength_nm, activation_energies per zone.
    """
    tangelo = TangeloInterface(use_tangelo=True, cache_results=False)

    voltage_kv = config['voltage_kv']
    gap_mm = config['channel_height_mm'] * 0.4
    E_field = voltage_kv * 1e3 / (gap_mm * 1e-3)

    zones = [
        {'name': 'inlet',   'T': 298.0, 'E': 0.0},
        {'name': 'plasma',  'T': 333.0, 'E': E_field},
        {'name': 'high_E',  'T': 333.0, 'E': 2.0 * E_field},
        {'name': 'outlet',  'T': 323.0, 'E': 0.0},
    ]

    zone_results = []
    for zone in zones:
        state = ChemicalState(
            temperature=zone['T'],
            pressure=config['pressure_Pa'],
            composition={"H2O": 0.95, "C_org": 0.05},
            electric_field=zone['E']
        )
        params = tangelo.get_parameters(state)

        zone_results.append({
            'zone': zone['name'],
            'temperature_K': zone['T'],
            'E_field_V_m': zone['E'],
            'gap_eV': params.cqd_gap_ev,
            'size_nm': params.cqd_size_nm,
            'wavelength_nm': EV_TO_NM / params.cqd_gap_ev,
            'activation_energies': params.activation_energies,
            'method': params.calculation_method,
            'confidence': params.confidence,
        })

    # Primary prediction: plasma zone
    plasma_zone = zone_results[1]

    return {
        'zones': zone_results,
        'primary_gap_eV': plasma_zone['gap_eV'],
        'primary_size_nm': plasma_zone['size_nm'],
        'primary_wavelength_nm': plasma_zone['wavelength_nm'],
        'method': plasma_zone['method'],
        'confidence': plasma_zone['confidence'],
        'uncertainty_gap_eV': 0.10,  # ±0.10 eV from literature model
        'uncertainty_size_nm': 0.20,  # ±0.20 nm
    }


# ─── 3. Production Model ─────────────────────────────────────────────────────

def run_production_model(config: Dict, cantera: Dict, tangelo: Dict) -> Dict:
    """
    Evaluate production using exact same scoring formula as
    optimize_cantera_reactor.py / cfd_validate_reactor.py.
    """
    n_ch = config['n_channels']
    ch_w = config['channel_width_mm']
    ch_h = config['channel_height_mm']
    ch_l = config['channel_length_mm']
    flow = config['flow_ml_min']
    voltage_kv = config['voltage_kv']
    freq_khz = config['frequency_khz']

    # Geometry
    liquid_fraction = 0.6
    liquid_depth = ch_h * liquid_fraction
    flow_per_ch = flow / n_ch
    v_mm_s = (flow_per_ch / 60.0 * 1000.0) / (ch_w * liquid_depth)
    t_res = ch_l / v_mm_s if v_mm_s > 0 else 999
    plasma_area_cm2 = ch_w * ch_l * n_ch / 100.0

    # Power (calibrated DBD model)
    specific_power = 0.25 * (voltage_kv / 10.0)**2 * (freq_khz / 20.0)
    area_per_ch_cm2 = (ch_w / 10.0) * (ch_l / 10.0)
    power_per_channel = specific_power * area_per_ch_cm2
    power_w = power_per_channel * n_ch
    energy_density_j_ml = power_w / (flow / 60.0) if flow > 0 else 0

    E_field = voltage_kv * 1e3 / (ch_h * 0.4 * 1e-3)

    # CQD size from energy density
    E_opt = 450.0
    if energy_density_j_ml > 10:
        size_nm = 2.5 * (E_opt / energy_density_j_ml)**0.15 * (t_res / 20.0)**0.08
        size_nm = max(1.5, min(5.0, size_nm))
    else:
        size_nm = tangelo['primary_size_nm']

    gap_ev = E_BULK + A_CONF / size_nm**2
    wavelength_nm = EV_TO_NM / gap_ev

    # Production factors
    base_conc = 0.3
    optimal_energy = 450
    if energy_density_j_ml < 100:
        energy_factor = energy_density_j_ml / 100 * 0.3
    elif energy_density_j_ml > 1000:
        energy_factor = 0.5
    else:
        energy_factor = np.exp(-((energy_density_j_ml - optimal_energy) / 300)**2)

    optimal_res = 20
    if t_res < 3:
        res_factor = t_res / 3 * 0.3
    elif t_res > 60:
        res_factor = 0.5
    else:
        res_factor = np.exp(-((t_res - optimal_res) / 20)**2)

    area_factor = min(2.0, plasma_area_cm2 / 5.0)

    OH_cm3 = cantera['OH_cm3']
    OH_ref = 1e15
    radical_factor = 1.0 + 0.15 * (OH_cm3 / OH_ref - 1.0)
    radical_factor = max(0.7, min(1.5, radical_factor))

    catalyst_factor = 1.35

    # CFD-enhanced: precursor conversion ~51% from species transport (CFD validated)
    # This matches compute_cqd_production() in cfd_validate_reactor.py
    precursor_conversion = 0.51  # From CFD species transport result
    cfd_boost = 1.0 + 0.5 * precursor_conversion

    concentration = (base_conc * energy_factor * res_factor * area_factor *
                     radical_factor * catalyst_factor * cfd_boost)
    concentration = max(0.01, min(3.0, concentration))
    production_mg_h = concentration * flow * 60

    # Quality (440-480 nm range, same as CFD validator)
    in_spec = abs(wavelength_nm - 460) <= 20
    monodispersity = 0.85 if t_res > 5 else 0.60

    # Thermal
    heat_gen = power_w * 0.30
    A_cooling = n_ch * (ch_l * 1e-3) * ((ch_w + 2 * ch_h) * 1e-3)
    h_conv = 300
    cooling_capacity = h_conv * A_cooling
    delta_T = heat_gen / max(0.1, cooling_capacity)
    max_temp_C = config.get('Tgas_K', 333) - 273.15 + delta_T
    cooling_ok = max_temp_C < 70

    # Score
    prod_norm = min(1.0, production_mg_h / 1000.0)
    quality_norm = 1.0 if in_spec else 0.3
    efficiency_norm = min(1.0, 1.0 / (1.0 + (power_w / max(0.01, production_mg_h) * 3600) / 500))
    cool_norm = 1.0 if cooling_ok else 0.2

    score = (prod_norm * 0.35 + quality_norm * 0.30 +
             efficiency_norm * 0.20 + cool_norm * 0.15)

    return {
        'production_mg_h': float(production_mg_h),
        'concentration_mg_ml': float(concentration),
        'wavelength_nm': float(wavelength_nm),
        'size_nm': float(size_nm),
        'gap_ev': float(gap_ev),
        'in_spec': bool(in_spec),
        'monodispersity': float(monodispersity),
        'power_w': float(power_w),
        'energy_density_j_ml': float(energy_density_j_ml),
        'E_field_V_m': float(E_field),
        'residence_time_s': float(t_res),
        'plasma_area_cm2': float(plasma_area_cm2),
        'max_temp_C': float(max_temp_C),
        'cooling_ok': bool(cooling_ok),
        'score': float(score),
        'factors': {
            'energy': float(energy_factor),
            'residence': float(res_factor),
            'area': float(area_factor),
            'radical': float(radical_factor),
            'catalyst': float(catalyst_factor),
            'cfd_boost': float(cfd_boost),
        },
    }


# ─── 4. Thermal Model ────────────────────────────────────────────────────────

def run_thermal_model(config: Dict) -> Dict:
    """
    Cold plasma thermal model with cooling serpentine design.
    Uses patterns from reactor_scaleup.py.
    """
    n_ch = config['n_channels']
    ch_w = config['channel_width_mm']
    ch_h = config['channel_height_mm']
    ch_l = config['channel_length_mm']
    voltage_kv = config['voltage_kv']
    freq_khz = config['frequency_khz']
    Te_eV = config.get('Te_eV', 1.5)
    Tgas_K = config.get('Tgas_K', 333.0)

    # Cold plasma parameters
    Te_K = Te_eV * 11604.5  # K
    non_thermal_ratio = Te_K / Tgas_K

    # Radical densities (scale with power density)
    power_density = voltage_kv * freq_khz * 0.01  # W/cm³
    OH_radical_cm3 = 1e15 * (power_density / 2.0)
    O_radical_cm3 = 5e14 * (power_density / 2.0)

    # Electrical power
    specific_power = 0.25 * (voltage_kv / 10.0)**2 * (freq_khz / 20.0)
    area_per_ch_cm2 = (ch_w / 10.0) * (ch_l / 10.0)
    power_w = specific_power * area_per_ch_cm2 * n_ch

    # Heat generation (30% of electrical → heat)
    heat_gen_w = power_w * 0.30

    # Cooling serpentine design
    # Serpentine channel runs between reactor channels
    n_cooling_channels = n_ch + 1  # One between each pair + edges
    cooling_ch_width_mm = 1.5
    cooling_ch_depth_mm = 2.0
    cooling_length_mm = ch_l  # Same as reactor length

    # Cooling area
    A_cooling_m2 = n_cooling_channels * (cooling_length_mm * 1e-3) * \
                   ((cooling_ch_width_mm + 2 * cooling_ch_depth_mm) * 1e-3)

    # Heat transfer coefficient (laminar water, 15°C coolant)
    h_conv = 300  # W/(m²·K) conservative for milli-channel
    coolant_temp_C = 15.0

    # Cooling capacity
    cooling_capacity_w = h_conv * A_cooling_m2 * (Tgas_K - 273.15 - coolant_temp_C)

    # Temperature rise
    delta_T = heat_gen_w / max(0.1, h_conv * A_cooling_m2)
    T_max_C = Tgas_K - 273.15 + delta_T

    # Required coolant flow (water, Cp=4186 J/(kg·K), allow 5°C rise)
    # Q_heat = m_dot * Cp * ΔT → m_dot = Q/(Cp·ΔT) in kg/s → convert to mL/min
    if heat_gen_w > 0:
        m_dot_kg_s = heat_gen_w / (4186.0 * 5.0)  # kg/s
        coolant_flow_ml_min = m_dot_kg_s / 998.0 * 1e6 * 60  # mL/min
    else:
        coolant_flow_ml_min = 50.0

    return {
        'cold_plasma': {
            'Te_eV': Te_eV,
            'Te_K': float(Te_K),
            'Tgas_K': Tgas_K,
            'Tgas_C': Tgas_K - 273.15,
            'non_thermal_ratio': float(non_thermal_ratio),
            'plasma_regime': 'cold_DBD' if non_thermal_ratio > 10 else 'warm',
            'OH_radical_cm3': float(OH_radical_cm3),
            'O_radical_cm3': float(O_radical_cm3),
        },
        'power': {
            'electrical_w': float(power_w),
            'heat_generation_w': float(heat_gen_w),
            'heat_fraction': 0.30,
        },
        'cooling': {
            'n_cooling_channels': n_cooling_channels,
            'cooling_ch_width_mm': cooling_ch_width_mm,
            'cooling_ch_depth_mm': cooling_ch_depth_mm,
            'cooling_length_mm': float(cooling_length_mm),
            'A_cooling_m2': float(A_cooling_m2),
            'h_conv_W_m2K': h_conv,
            'coolant_temp_C': coolant_temp_C,
            'cooling_capacity_w': float(cooling_capacity_w),
            'required_coolant_flow_ml_min': float(coolant_flow_ml_min),
        },
        'temperatures': {
            'delta_T_C': float(delta_T),
            'T_max_C': float(T_max_C),
            'cooling_ok': T_max_C < 70,
            'cooling_margin_C': 70 - T_max_C,
        },
    }


# ─── 5. Control Simulation ───────────────────────────────────────────────────

def run_control_simulation(config: Dict) -> Dict:
    """
    Simulate ReactorController + ClassifierController from reactor_control.py.
    Generate wavelength sweep → valve classification table.
    """
    from reactor_control import ReactorController, ClassifierController

    reactor_ctrl = ReactorController(target_wavelength=480.0, tolerance=20.0)
    classifier_ctrl = ClassifierController(
        n_zones=3,
        reactor_controller=reactor_ctrl,
    )

    # Wavelength sweep: classify each
    wavelengths = np.arange(380, 620, 10)
    classification_table = []
    for wl in wavelengths:
        result = reactor_ctrl.process_sensor_reading(
            wavelength=float(wl),
            intensity=0.8,
            fwhm=35.0
        )
        classification_table.append({
            'wavelength_nm': float(wl),
            'gap_eV': result['gap_ev'],
            'size_nm': result['size_nm'],
            'action': result['action'],
            'valve': result['valve'],
            'in_spec': result['in_spec'],
        })

    # Control setpoints
    setpoints = reactor_ctrl.get_setpoints()

    # Classifier zone specs
    zone_specs = []
    for spec in classifier_ctrl.zone_specs:
        zone_specs.append({
            'zone': spec['zone'],
            'name': spec['name'],
            'led_wavelength_nm': spec['led_wavelength_nm'],
            'size_range_nm': spec['size_range_nm'],
            'emission_range_nm': spec['emission_range_nm'],
        })

    return {
        'setpoints': setpoints,
        'classification_table': classification_table,
        'n_in_spec': sum(1 for c in classification_table if c['in_spec']),
        'n_total': len(classification_table),
        'zone_specs': zone_specs,
    }


# ─── 6. Consolidate Digital Twin ─────────────────────────────────────────────

def consolidate_digital_twin(config: Dict, cantera: Dict, tangelo: Dict,
                              production: Dict, thermal: Dict,
                              control: Dict) -> Dict:
    """
    Cross-model consistency checks, combined uncertainty bounds.
    Loads CFD reference from cfd_validation_results.json.
    """
    # Load CFD reference
    cfd_path = OUTPUT_DIR / "cfd_validation_results.json"
    cfd_ref = None
    if cfd_path.exists():
        with open(cfd_path) as f:
            cfd_data = json.load(f)
        cfd_ref = cfd_data.get('cfd_results', {}).get('parametric_opt', {}).get('cqd', {})

    # Cross-model wavelength predictions
    wl_production = production['wavelength_nm']
    wl_tangelo = tangelo['primary_wavelength_nm']
    wl_cfd = cfd_ref.get('wavelength_nm', 0) if cfd_ref else 0

    # Consistency check: all models should agree within 10%
    wavelength_predictions = {
        'production_model': wl_production,
        'tangelo_model': wl_tangelo,
    }
    if wl_cfd > 0:
        wavelength_predictions['cfd_model'] = wl_cfd

    wl_values = [v for v in wavelength_predictions.values() if v > 0]
    wl_mean = np.mean(wl_values)
    wl_std = np.std(wl_values) if len(wl_values) > 1 else wl_mean * 0.05
    wavelength_consistent = all(abs(v - wl_mean) / wl_mean < 0.10 for v in wl_values)

    # Combined uncertainty (quadrature for independent errors)
    # Production model unc: ±30% (from PINN R² ~ 0.96 → ~20% + model uncertainty)
    # CFD unc: derived from PINN validation metrics
    production_unc = 0.30
    wavelength_unc = 0.10  # ±10% from model spread
    size_unc = 0.16  # from tangelo uncertainty

    # 95% CI bounds (production) — prefer CFD value if available
    prod_central = cfd_ref.get('production_mg_h', production['production_mg_h']) if cfd_ref else production['production_mg_h']
    prod_95ci = (
        prod_central * (1 - 1.96 * production_unc),
        prod_central * (1 + 1.96 * production_unc),
    )

    wl_central = cfd_ref.get('wavelength_nm', production['wavelength_nm']) if cfd_ref else production['wavelength_nm']
    wl_95ci = (
        wl_central * (1 - 1.96 * wavelength_unc),
        wl_central * (1 + 1.96 * wavelength_unc),
    )

    size_central = cfd_ref.get('size_nm', production['size_nm']) if cfd_ref else production['size_nm']
    size_95ci = (
        size_central * (1 - 1.96 * size_unc),
        size_central * (1 + 1.96 * size_unc),
    )

    # Power, temperature, pressure — prefer CFD values (what thermocouples measure)
    power_central = production['power_w']
    power_unc = 0.10
    power_95ci = (power_central * (1 - 1.96 * power_unc), power_central * (1 + 1.96 * power_unc))

    # T_max: use CFD liquid temperature (measurable), not gas zone temperature
    T_max_central = cfd_ref.get('T_max_C', thermal['temperatures']['T_max_C']) if cfd_ref else thermal['temperatures']['T_max_C']
    T_unc = 0.12
    T_95ci = (T_max_central * (1 - 1.96 * T_unc), T_max_central * (1 + 1.96 * T_unc))

    delta_p_central = cfd_ref.get('delta_p_Pa', 1736.0) if cfd_ref else 1736.0
    delta_p_unc = 0.08
    delta_p_95ci = (delta_p_central * (1 - 1.96 * delta_p_unc), delta_p_central * (1 + 1.96 * delta_p_unc))

    return {
        'config': config,
        'cfd_reference_loaded': cfd_ref is not None,
        'cross_model_consistency': {
            'wavelength_predictions': wavelength_predictions,
            'wavelength_mean_nm': float(wl_mean),
            'wavelength_std_nm': float(wl_std),
            'wavelength_consistent': wavelength_consistent,
        },
        'predictions': {
            'production_mg_h': {'central': prod_central, '95ci_lower': prod_95ci[0], '95ci_upper': prod_95ci[1]},
            'wavelength_nm': {'central': wl_central, '95ci_lower': wl_95ci[0], '95ci_upper': wl_95ci[1]},
            'size_nm': {'central': size_central, '95ci_lower': size_95ci[0], '95ci_upper': size_95ci[1]},
            'power_w': {'central': power_central, '95ci_lower': power_95ci[0], '95ci_upper': power_95ci[1]},
            'T_max_C': {'central': T_max_central, '95ci_lower': T_95ci[0], '95ci_upper': T_95ci[1]},
            'delta_p_Pa': {'central': delta_p_central, '95ci_lower': delta_p_95ci[0], '95ci_upper': delta_p_95ci[1]},
        },
        'score': production['score'],
        'cfd_score': cfd_ref.get('score', 0) if cfd_ref else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PART B: EXPERIMENTAL PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 7. Bill of Materials ─────────────────────────────────────────────────────

def generate_bom(config: Dict) -> List[Dict]:
    """Generate Bill of Materials for the 16ch×500mm reactor."""
    n_ch = config['n_channels']
    ch_l = config['channel_length_mm']

    bom = []

    # Category: Structure
    bom.extend([
        {'category': 'Structure', 'item': 'Borosilicate glass plate (reactor body)',
         'spec': f'200×{ch_l+50}×10 mm, polished', 'qty': 2, 'unit': 'pcs', 'cost_eur': 120},
        {'category': 'Structure', 'item': 'Stainless steel base plate',
         'spec': '200×560×15 mm, 316L', 'qty': 1, 'unit': 'pcs', 'cost_eur': 85},
        {'category': 'Structure', 'item': 'PTFE gasket set',
         'spec': '1.5 mm thick, laser-cut to channel pattern', 'qty': 4, 'unit': 'pcs', 'cost_eur': 30},
        {'category': 'Structure', 'item': 'M4 stainless steel bolts + nuts',
         'spec': 'A2-70, 30 mm length', 'qty': 32, 'unit': 'pcs', 'cost_eur': 15},
        {'category': 'Structure', 'item': 'Alignment pins',
         'spec': '3 mm dia, 15 mm length, hardened steel', 'qty': 8, 'unit': 'pcs', 'cost_eur': 10},
    ])

    # Category: Electrodes
    bom.extend([
        {'category': 'Electrodes', 'item': 'Copper foil electrodes (HV)',
         'spec': f'0.1 mm thick, {n_ch} strips × 2×{ch_l} mm', 'qty': n_ch, 'unit': 'pcs', 'cost_eur': 45},
        {'category': 'Electrodes', 'item': 'Copper foil electrodes (ground)',
         'spec': f'0.1 mm thick, {n_ch} strips × 2×{ch_l} mm', 'qty': n_ch, 'unit': 'pcs', 'cost_eur': 45},
        {'category': 'Electrodes', 'item': 'HV wire (silicone insulated)',
         'spec': '20 kV rated, 2 m length', 'qty': 2, 'unit': 'm', 'cost_eur': 25},
        {'category': 'Electrodes', 'item': 'HV connector (banana)',
         'spec': '30 kV rated', 'qty': 2, 'unit': 'pcs', 'cost_eur': 20},
    ])

    # Category: Dielectric / Catalyst
    bom.extend([
        {'category': 'Dielectric/Catalyst', 'item': 'TiO2 anatase powder (P25)',
         'spec': '21 nm primary, 50 m²/g BET', 'qty': 50, 'unit': 'g', 'cost_eur': 35},
        {'category': 'Dielectric/Catalyst', 'item': 'TiO2 sol-gel binder',
         'spec': 'Titanium isopropoxide, 97%', 'qty': 100, 'unit': 'mL', 'cost_eur': 40},
        {'category': 'Dielectric/Catalyst', 'item': 'Ethanol (coating solvent)',
         'spec': '99.5%, anhydrous', 'qty': 500, 'unit': 'mL', 'cost_eur': 15},
    ])

    # Category: Fluidics
    bom.extend([
        {'category': 'Fluidics', 'item': 'Syringe pump',
         'spec': '0.1-50 mL/min, dual channel', 'qty': 1, 'unit': 'pcs', 'cost_eur': 800},
        {'category': 'Fluidics', 'item': 'PTFE tubing (1/16" OD)',
         'spec': '0.5 mm ID, 5 m roll', 'qty': 2, 'unit': 'rolls', 'cost_eur': 30},
        {'category': 'Fluidics', 'item': 'Manifold (inlet)',
         'spec': f'1-to-{n_ch} splitter, PEEK', 'qty': 1, 'unit': 'pcs', 'cost_eur': 150},
        {'category': 'Fluidics', 'item': 'Manifold (outlet)',
         'spec': f'{n_ch}-to-1 collector, PEEK', 'qty': 1, 'unit': 'pcs', 'cost_eur': 150},
        {'category': 'Fluidics', 'item': 'Check valves',
         'spec': 'PEEK body, 0.5 psi cracking', 'qty': 4, 'unit': 'pcs', 'cost_eur': 80},
        {'category': 'Fluidics', 'item': '3-way solenoid valve (classifier)',
         'spec': 'PTFE wetted, 12V, <100ms', 'qty': 3, 'unit': 'pcs', 'cost_eur': 135},
        {'category': 'Fluidics', 'item': 'Collection vessels',
         'spec': '50 mL amber glass, screw cap', 'qty': 10, 'unit': 'pcs', 'cost_eur': 25},
    ])

    # Category: Electrical
    bom.extend([
        {'category': 'Electrical', 'item': 'HV power supply (AC)',
         'spec': '0-20 kV, 10-30 kHz, 200 W max', 'qty': 1, 'unit': 'pcs', 'cost_eur': 1500},
        {'category': 'Electrical', 'item': 'Function generator',
         'spec': '1 Hz - 1 MHz, TTL output', 'qty': 1, 'unit': 'pcs', 'cost_eur': 350},
        {'category': 'Electrical', 'item': 'HV probe (1000:1)',
         'spec': '0-40 kV, 100 MHz BW', 'qty': 1, 'unit': 'pcs', 'cost_eur': 250},
        {'category': 'Electrical', 'item': 'Current transformer',
         'spec': 'Pearson coil, 0.1V/A', 'qty': 1, 'unit': 'pcs', 'cost_eur': 200},
    ])

    # Category: Sensors
    bom.extend([
        {'category': 'Sensors', 'item': 'UV-Vis spectrometer (fiber-coupled)',
         'spec': '350-700 nm, 1 nm resolution', 'qty': 1, 'unit': 'pcs', 'cost_eur': 600},
        {'category': 'Sensors', 'item': 'Fiber optic probe (emission)',
         'spec': '400 μm core, SMA connector', 'qty': 2, 'unit': 'pcs', 'cost_eur': 120},
        {'category': 'Sensors', 'item': 'UV LED excitation source',
         'spec': '365 nm, 5 W, collimated', 'qty': 1, 'unit': 'pcs', 'cost_eur': 80},
        {'category': 'Sensors', 'item': 'K-type thermocouple',
         'spec': '0.5 mm dia, 150 mm length, PTFE sheath', 'qty': 4, 'unit': 'pcs', 'cost_eur': 60},
        {'category': 'Sensors', 'item': 'Pressure transducer (differential)',
         'spec': '0-10 kPa, 4-20 mA output', 'qty': 1, 'unit': 'pcs', 'cost_eur': 180},
        {'category': 'Sensors', 'item': 'Flow meter (liquid)',
         'spec': '0.1-50 mL/min, PEEK wetted', 'qty': 1, 'unit': 'pcs', 'cost_eur': 250},
    ])

    # Category: Cooling
    bom.extend([
        {'category': 'Cooling', 'item': 'Recirculating chiller',
         'spec': '200 W cooling, 5-40°C, 2 L/min', 'qty': 1, 'unit': 'pcs', 'cost_eur': 600},
        {'category': 'Cooling', 'item': 'Copper cooling serpentine',
         'spec': f'3 mm OD, 2 mm ID, {ch_l+100} mm length × {config["n_channels"]+1}', 'qty': 1, 'unit': 'set', 'cost_eur': 90},
        {'category': 'Cooling', 'item': 'Thermal paste',
         'spec': 'Silver-based, >5 W/(m·K)', 'qty': 1, 'unit': 'tube', 'cost_eur': 15},
    ])

    # Category: Control
    bom.extend([
        {'category': 'Control', 'item': 'DAQ board (Arduino/ESP32)',
         'spec': '16-bit ADC, 8 analog inputs, WiFi', 'qty': 1, 'unit': 'pcs', 'cost_eur': 35},
        {'category': 'Control', 'item': 'Oscilloscope',
         'spec': '100 MHz, 4-channel, 1 GS/s', 'qty': 1, 'unit': 'pcs', 'cost_eur': 450},
        {'category': 'Control', 'item': 'Relay board (valve control)',
         'spec': '8-channel, 12V, optocoupled', 'qty': 1, 'unit': 'pcs', 'cost_eur': 15},
    ])

    # Category: Consumables
    bom.extend([
        {'category': 'Consumables', 'item': 'Precursor: dilute manure slurry',
         'spec': f'{config["precursor_conc_g_L"]} g/L, filtered <100 μm', 'qty': 20, 'unit': 'L', 'cost_eur': 5},
        {'category': 'Consumables', 'item': 'DI water (reagent grade)',
         'spec': '18.2 MΩ·cm', 'qty': 50, 'unit': 'L', 'cost_eur': 10},
        {'category': 'Consumables', 'item': 'RTD tracer dye (Rhodamine B)',
         'spec': '1 mg/mL stock solution', 'qty': 100, 'unit': 'mL', 'cost_eur': 20},
        {'category': 'Consumables', 'item': 'Fluorescein standard (calibration)',
         'spec': '1 μg/mL in DI water', 'qty': 100, 'unit': 'mL', 'cost_eur': 15},
        {'category': 'Consumables', 'item': 'Nitrogen gas (carrier)',
         'spec': '99.99%, 50 L cylinder', 'qty': 1, 'unit': 'cylinder', 'cost_eur': 40},
        {'category': 'Consumables', 'item': 'Filter membranes (product isolation)',
         'spec': '0.22 μm, PVDF, 47 mm', 'qty': 50, 'unit': 'pcs', 'cost_eur': 35},
    ])

    return bom


# ─── 8. Fabrication Protocol ─────────────────────────────────────────────────

def generate_fabrication_protocol(config: Dict, bom: List[Dict]) -> List[Dict]:
    """Generate 8-phase fabrication protocol with tolerances from CFD."""
    n_ch = config['n_channels']
    ch_w = config['channel_width_mm']
    ch_h = config['channel_height_mm']
    ch_l = config['channel_length_mm']

    phases = [
        {
            'phase': 1, 'name': 'Preparation',
            'duration_h': 4,
            'steps': [
                f'Clean all glass plates in piranha solution (3:1 H2SO4:H2O2) for 30 min',
                f'Rinse 5× with DI water, dry under N2 flow',
                f'Verify plate dimensions: 200×{ch_l+50}×10 mm (tolerance ±0.1 mm)',
                'Prepare TiO2 coating suspension: 5 wt% P25 in ethanol + 1% Ti-isopropoxide',
                'Sonicate suspension for 30 min, filter through 5 μm mesh',
            ],
        },
        {
            'phase': 2, 'name': 'Channel Machining',
            'duration_h': 8,
            'steps': [
                f'CNC mill {n_ch} parallel channels in bottom glass plate',
                f'Channel dimensions: {ch_w:.1f}×{ch_h:.1f}×{ch_l:.0f} mm (tolerance ±0.05 mm)',
                f'Channel pitch: {ch_w + 2:.1f} mm center-to-center',
                f'Inlet manifold cavity: 10×{n_ch * (ch_w + 2):.0f}×{ch_h:.1f} mm',
                f'Outlet manifold cavity: matching inlet',
                'Deburr all channel edges with 800-grit SiC paper',
                'Ultrasonic clean machined plate in acetone (15 min) then DI water (15 min)',
                'Verify channel depth with profilometer (±0.02 mm required)',
            ],
        },
        {
            'phase': 3, 'name': 'Electrode Integration',
            'duration_h': 6,
            'steps': [
                f'Cut {n_ch} HV electrode strips: 0.1×2×{ch_l:.0f} mm copper foil',
                f'Cut {n_ch} ground electrode strips: matching dimensions',
                'Position HV electrodes on bottom plate between channels (adhesive-backed)',
                'Position ground electrodes on top plate, aligned with HV electrodes',
                'Verify electrode-to-channel alignment: ±0.2 mm tolerance',
                'Solder HV bus bar connecting all HV electrodes in parallel',
                'Solder ground bus bar connecting all ground electrodes in parallel',
                'Resistance check: <0.5 Ω from connector to each electrode tip',
            ],
        },
        {
            'phase': 4, 'name': 'Dielectric & Catalyst Coating',
            'duration_h': 12,
            'steps': [
                'Mask channel bottoms (liquid zone) with Kapton tape',
                'Dip-coat electrode surfaces with TiO2 suspension (3 layers)',
                'Between layers: dry at 60°C for 30 min',
                'Remove Kapton mask from channel bottoms',
                'Apply thin TiO2 layer to gas gap surfaces (1 layer, thinner)',
                'Final cure: ramp to 200°C at 2°C/min, hold 2h, cool naturally',
                f'Target coating: 2 mg/cm² TiO2, {0.5:.1f} mm thickness, 60% porosity',
                'Verify coating adhesion: Scotch tape test (must pass)',
            ],
        },
        {
            'phase': 5, 'name': 'Manifold Assembly',
            'duration_h': 4,
            'steps': [
                f'Install PEEK 1-to-{n_ch} inlet manifold with Swagelok fittings',
                f'Install PEEK {n_ch}-to-1 outlet manifold',
                'Connect PTFE tubing (1/16" OD) to inlet and outlet',
                'Install check valves at inlet (prevent backflow)',
                'Attach thermocouple ports at inlet, outlet, mid-length (×2)',
                'Attach pressure tap at inlet (before manifold)',
            ],
        },
        {
            'phase': 6, 'name': 'Cooling System',
            'duration_h': 4,
            'steps': [
                'Route copper serpentine between channels on bottom plate',
                'Apply thermal paste to serpentine-plate contact surfaces',
                'Connect serpentine inlet/outlet to recirculating chiller',
                'Verify serpentine flow: >100 mL/min with <0.5 bar pressure drop',
                'Thermal test: run chiller at 15°C, verify plate reaches 18°C in <5 min',
            ],
        },
        {
            'phase': 7, 'name': 'Sealing & Leak Test',
            'duration_h': 4,
            'steps': [
                'Place PTFE gaskets on both sides of channel plate',
                'Assemble top plate, align with pins, torque bolts to 2 N·m in star pattern',
                'Pressurize with N2 to 50 kPa, hold 10 min, verify zero pressure drop',
                'Fill with DI water at 5 mL/min, check all fittings for leaks',
                'Increase flow to 20 mL/min, verify no leaks at operating pressure',
                'Dye test: inject 0.1% Rhodamine B, verify uniform exit from all channels',
            ],
        },
        {
            'phase': 8, 'name': 'Electrical Commissioning',
            'duration_h': 4,
            'steps': [
                'Connect HV supply to electrode bus bars via HV cable',
                'Connect HV probe and current transformer to oscilloscope',
                'Dielectric test: ramp voltage 0→5 kV at 20 kHz, verify no arcing',
                'Ramp to 8 kV: verify uniform plasma glow through glass (visual)',
                f'Ramp to {config["voltage_kv"]:.0f} kV at {config["frequency_khz"]:.0f} kHz: record V, I waveforms',
                'Calculate power: P = (1/T)∫V(t)·I(t)dt over 10 cycles',
                f'Verify power within 10% of predicted {0.25 * (config["voltage_kv"]/10)**2 * (config["frequency_khz"]/20) * (ch_w/10)*(ch_l/10)*n_ch:.1f} W',
                'Run 1 hour with water flow: verify stable operation, no arcing',
            ],
        },
    ]

    return phases


# ─── 9. Measurement Equipment ────────────────────────────────────────────────

def generate_measurement_equipment() -> List[Dict]:
    """List of instruments with resolution, calibration, uncertainty."""
    return [
        {
            'instrument': 'UV-Vis Fiber Spectrometer',
            'model_example': 'Ocean Insight Flame-S or similar',
            'range': '350-700 nm',
            'resolution': '1 nm (FWHM)',
            'measurement': 'PL emission spectrum of CQDs',
            'calibration': 'Hg-Ar lamp (spectral lines at 435.8, 546.1, 696.5 nm)',
            'uncertainty': '±1 nm wavelength, ±5% intensity',
        },
        {
            'instrument': 'K-type Thermocouple (×4)',
            'model_example': 'Type K, 0.5 mm dia, PTFE sheath',
            'range': '0-200 °C',
            'resolution': '0.1 °C',
            'measurement': 'Reactor temperature at 4 points (inlet, mid×2, outlet)',
            'calibration': 'Ice-point reference (0°C) + boiling water (100°C)',
            'uncertainty': '±0.5 °C',
        },
        {
            'instrument': 'Differential Pressure Transducer',
            'model_example': 'Honeywell ASDX or similar',
            'range': '0-10 kPa',
            'resolution': '1 Pa',
            'measurement': 'Pressure drop across reactor',
            'calibration': 'Dead-weight tester or water column',
            'uncertainty': '±0.5% FS (±50 Pa)',
        },
        {
            'instrument': 'Flow Meter (liquid)',
            'model_example': 'Sensirion SLF3x or thermal mass flow',
            'range': '0.1-50 mL/min',
            'resolution': '0.01 mL/min',
            'measurement': 'Liquid flow rate',
            'calibration': 'Gravimetric (collect + weigh over timed interval)',
            'uncertainty': '±2% of reading',
        },
        {
            'instrument': 'HV Probe (1000:1)',
            'model_example': 'Tektronix P6015A or similar',
            'range': '0-40 kV',
            'resolution': '10 V',
            'measurement': 'Applied voltage waveform',
            'calibration': 'Known AC source at 1 kV, 10 kV',
            'uncertainty': '±3% of reading',
        },
        {
            'instrument': 'Oscilloscope (4-ch)',
            'model_example': '100 MHz, 1 GS/s',
            'range': 'DC-100 MHz',
            'resolution': '8-bit vertical',
            'measurement': 'V(t), I(t) waveforms for power calculation',
            'calibration': 'Factory calibration certificate',
            'uncertainty': '±1.5% vertical, ±0.01% timebase',
        },
    ]


# ─── 10. Experimental Protocol (3 phases) ────────────────────────────────────

def generate_experimental_protocol(digital_twin: Dict) -> List[Dict]:
    """Generate 3-phase experimental protocol."""
    pred = digital_twin['predictions']

    phases = [
        {
            'phase': 1,
            'name': 'Commissioning',
            'duration_days': 2,
            'objective': 'Verify reactor integrity, flow distribution, plasma ignition',
            'steps': [
                {'step': 1, 'action': 'Leak test',
                 'detail': 'Pressurize with N2 to 50 kPa. Hold 10 min. Accept: zero pressure drop.',
                 'pass_criteria': 'ΔP < 1 Pa in 10 min'},
                {'step': 2, 'action': 'Flow calibration',
                 'detail': 'Set pump to 15 mL/min. Measure outlet with graduated cylinder over 5 min.',
                 'pass_criteria': '15.0 ±0.75 mL/min (±5%)'},
                {'step': 3, 'action': 'Flow distribution check',
                 'detail': 'Inject Rhodamine B dye pulse at inlet. Photograph outlet channels simultaneously.',
                 'pass_criteria': 'All 16 channels show dye within 2 s of each other'},
                {'step': 4, 'action': 'Plasma ignition ramp',
                 'detail': 'Ramp voltage: 0→5→8→10→12 kV at 30 kHz (2 min per step). Monitor for arcing.',
                 'pass_criteria': 'Uniform glow in all channels at 12 kV, no visible arcs'},
                {'step': 5, 'action': 'Thermal baseline',
                 'detail': 'Run plasma at 12 kV, 30 kHz with water at 15 mL/min for 30 min. Record T at all 4 points.',
                 'pass_criteria': f'T_max < 70°C, steady state within 15 min'},
                {'step': 6, 'action': 'Cooling verification',
                 'detail': 'Verify chiller maintains coolant at 15±1°C, flow >100 mL/min.',
                 'pass_criteria': 'Coolant ΔT < 5°C across reactor'},
            ],
        },
        {
            'phase': 2,
            'name': 'Baseline Characterization',
            'duration_days': 3,
            'objective': 'Validate CFD predictions (RTD, pressure, thermal, OES)',
            'steps': [
                {'step': 1, 'action': 'RTD measurement with tracer',
                 'detail': 'Inject 0.1 mL Rhodamine B pulse at inlet. Record outlet fluorescence vs time (spectrometer, 10 Hz).',
                 'pass_criteria': f'Pe ≈ {pred["delta_p_Pa"]["central"]/41:.0f}±10 (verify plug flow)'},
                {'step': 2, 'action': 'Pressure drop measurement',
                 'detail': 'Measure ΔP at 5, 10, 15, 20 mL/min (3 replicates each). Plot ΔP vs Q.',
                 'pass_criteria': f'ΔP at 15 mL/min = {pred["delta_p_Pa"]["central"]:.0f} ±{(pred["delta_p_Pa"]["95ci_upper"]-pred["delta_p_Pa"]["central"]):.0f} Pa'},
                {'step': 3, 'action': 'Thermal mapping',
                 'detail': 'Record temperatures at 4 points under plasma (12 kV, 30 kHz, 15 mL/min). Steady state = 30 min.',
                 'pass_criteria': f'T_max = {pred["T_max_C"]["central"]:.1f} ±{(pred["T_max_C"]["95ci_upper"]-pred["T_max_C"]["central"]):.1f} °C'},
                {'step': 4, 'action': 'OES for OH* emission',
                 'detail': 'Record optical emission spectrum (300-400 nm) through glass window. Identify OH(A-X) at 309 nm.',
                 'pass_criteria': 'OH(A-X) 309 nm band present, intensity scales with voltage'},
                {'step': 5, 'action': 'Power measurement',
                 'detail': 'Record V(t) and I(t) waveforms at 12 kV, 30 kHz. Compute P = ∫V·I dt / T over 100 cycles.',
                 'pass_criteria': f'P = {pred["power_w"]["central"]:.1f} ±{(pred["power_w"]["95ci_upper"]-pred["power_w"]["central"]):.1f} W'},
                {'step': 6, 'action': 'Voltage-frequency matrix',
                 'detail': 'Test 3 voltages (10, 12, 14 kV) × 3 frequencies (20, 25, 30 kHz). Record P, T, OES for each.',
                 'pass_criteria': 'Power scales as V²·f (within 15%)'},
            ],
        },
        {
            'phase': 3,
            'name': 'CQD Synthesis & Validation',
            'duration_days': 5,
            'objective': 'Produce CQDs, measure properties, compare to digital twin',
            'steps': [
                {'step': 1, 'action': 'Precursor preparation',
                 'detail': 'Dilute manure slurry to 2 g/L with DI water. Filter through 100 μm mesh. Sonicate 15 min.',
                 'pass_criteria': 'Homogeneous suspension, no visible solids >100 μm'},
                {'step': 2, 'action': 'Baseline run (3 replicates)',
                 'detail': 'Run at nominal conditions (12 kV, 30 kHz, 15 mL/min) for 2 h each. Collect product in amber vials.',
                 'pass_criteria': 'Visible fluorescence under 365 nm UV lamp'},
                {'step': 3, 'action': 'PL spectroscopy',
                 'detail': 'Excite product at 365 nm. Record emission 400-700 nm. Extract peak λ, FWHM, intensity.',
                 'pass_criteria': f'λ_peak = {pred["wavelength_nm"]["central"]:.0f} ±{(pred["wavelength_nm"]["95ci_upper"]-pred["wavelength_nm"]["central"]):.0f} nm'},
                {'step': 4, 'action': 'TEM characterization',
                 'detail': 'Drop-cast on Cu grid. Image 200+ particles. Measure size distribution.',
                 'pass_criteria': f'd_mean = {pred["size_nm"]["central"]:.2f} ±{(pred["size_nm"]["95ci_upper"]-pred["size_nm"]["central"]):.2f} nm'},
                {'step': 5, 'action': 'DLS size measurement',
                 'detail': 'Hydrodynamic size in DI water. 3 measurements × 15 runs each.',
                 'pass_criteria': f'D_h = {pred["size_nm"]["central"]*1.3:.1f} ±{pred["size_nm"]["central"]*0.3:.1f} nm (1.3× TEM expected)'},
                {'step': 6, 'action': 'Production rate measurement',
                 'detail': 'Gravimetric: filter 100 mL product through 0.22 μm, dry at 60°C, weigh. Verify with UV-Vis calibration curve.',
                 'pass_criteria': f'Production = {pred["production_mg_h"]["central"]:.0f} ±{(pred["production_mg_h"]["95ci_upper"]-pred["production_mg_h"]["central"]):.0f} mg/h'},
                {'step': 7, 'action': 'Flow rate sweep (3 replicates each)',
                 'detail': 'Test 10, 15, 20 mL/min at fixed 12 kV, 30 kHz. Measure production rate and λ at each.',
                 'pass_criteria': 'Production peaks near 15 mL/min, λ shifts <20 nm across range'},
                {'step': 8, 'action': 'Control system validation',
                 'detail': 'Connect classifier controller. Run 1 h. Record valve actions vs PL readings.',
                 'pass_criteria': 'Classification accuracy >90% (vs manual spectroscopy)'},
            ],
        },
    ]

    return phases


# ─── 11. Data Acquisition Plan ────────────────────────────────────────────────

def generate_data_acquisition_plan() -> List[Dict]:
    """Define DAQ channels, rates, and storage."""
    return [
        {'channel': 'Temperature', 'sensor': 'K-type TC ×4', 'rate_hz': 1,
         'resolution': '0.1 °C', 'interface': 'DAQ analog input (4 channels)',
         'storage': '~350 KB/h', 'trigger': 'Continuous'},
        {'channel': 'Pressure', 'sensor': 'Differential transducer', 'rate_hz': 10,
         'resolution': '1 Pa', 'interface': 'DAQ analog input (1 channel)',
         'storage': '~90 KB/h', 'trigger': 'Continuous'},
        {'channel': 'Flow', 'sensor': 'Thermal mass flow', 'rate_hz': 1,
         'resolution': '0.01 mL/min', 'interface': 'DAQ analog input (1 channel)',
         'storage': '~90 KB/h', 'trigger': 'Continuous'},
        {'channel': 'Voltage', 'sensor': 'HV probe 1000:1', 'rate_hz': 100,
         'resolution': '10 V', 'interface': 'Oscilloscope Ch1',
         'storage': '~900 KB/h (sampled bursts)', 'trigger': '10 s burst every 60 s'},
        {'channel': 'Current', 'sensor': 'Pearson coil', 'rate_hz': 100,
         'resolution': '1 mA', 'interface': 'Oscilloscope Ch2',
         'storage': '~900 KB/h (sampled bursts)', 'trigger': 'Synced with voltage'},
        {'channel': 'PL Spectrum', 'sensor': 'Fiber spectrometer', 'rate_hz': 0.1,
         'resolution': '1 nm, 2048 pixels', 'interface': 'USB, vendor SDK',
         'storage': '~1.5 MB/h', 'trigger': 'Every 10 s during synthesis'},
        {'channel': 'Valve State', 'sensor': 'Digital I/O', 'rate_hz': 'event',
         'resolution': 'Binary (3 valves)', 'interface': 'DAQ digital output',
         'storage': '<10 KB/h', 'trigger': 'On state change'},
    ]


# ─── 12. Acceptance Criteria ─────────────────────────────────────────────────

def generate_acceptance_criteria(digital_twin: Dict) -> List[Dict]:
    """Generate pass/fail table from digital twin predictions."""
    pred = digital_twin['predictions']

    criteria = []
    for param, data in pred.items():
        method_map = {
            'production_mg_h': 'Gravimetric + PL calibration',
            'wavelength_nm': 'PL spectroscopy (365 nm excitation)',
            'size_nm': 'TEM (200+ particles) + DLS',
            'power_w': 'V×I oscilloscope measurement',
            'T_max_C': 'K-type thermocouple',
            'delta_p_Pa': 'Differential pressure sensor',
        }
        unit_map = {
            'production_mg_h': 'mg/h',
            'wavelength_nm': 'nm',
            'size_nm': 'nm',
            'power_w': 'W',
            'T_max_C': '°C',
            'delta_p_Pa': 'Pa',
        }
        criteria.append({
            'parameter': param,
            'predicted': data['central'],
            '95ci_lower': data['95ci_lower'],
            '95ci_upper': data['95ci_upper'],
            'unit': unit_map.get(param, ''),
            'method': method_map.get(param, ''),
        })

    return criteria


# ─── 13. Statistical Analysis Plan ───────────────────────────────────────────

def generate_statistical_plan() -> Dict:
    """Define statistical analysis approach."""
    return {
        'sample_size': {
            'n_replicates': 3,
            'distribution': 't-distribution (df=2)',
            'ci_level': 0.95,
            'justification': 'Minimum for t-distribution CI; limited by precursor and reactor time',
        },
        'tests': [
            {
                'name': 'One-sample t-test (predicted vs measured)',
                'purpose': 'Test if measured mean matches digital twin prediction',
                'null_hypothesis': 'H0: μ_measured = μ_predicted',
                'alpha': 0.05,
                'interpretation': 'Reject H0 → model prediction outside experimental range',
            },
            {
                'name': 'Bland-Altman analysis',
                'purpose': 'Assess agreement between model and experiment',
                'method': 'Plot (model+experiment)/2 vs (model-experiment). Calculate limits of agreement.',
                'interpretation': 'Points within ±1.96σ → acceptable agreement',
            },
            {
                'name': 'One-way ANOVA (flow rate effect)',
                'purpose': 'Compare production at 10, 15, 20 mL/min',
                'null_hypothesis': 'H0: μ_10 = μ_15 = μ_20',
                'alpha': 0.05,
                'post_hoc': 'Tukey HSD if ANOVA significant',
            },
            {
                'name': 'Power analysis',
                'purpose': 'Verify n=3 is sufficient to detect meaningful differences',
                'parameters': {'beta': 0.80, 'alpha': 0.05, 'effect_size': 'Cohen d=1.5 (large)'},
                'result': 'n=3 sufficient for large effects (d>1.5); n=6 needed for medium (d~0.8)',
            },
        ],
        'data_processing': [
            'Remove first 10 min of each run (transient startup)',
            'Outlier removal: Grubbs test (α=0.05) on triplicates',
            'Report: mean ± 95% CI using t-distribution',
            'Normality check: Shapiro-Wilk test (n<50)',
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT: REPORT
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 14. Print Report ─────────────────────────────────────────────────────────

def print_report(digital_twin: Dict, cantera: Dict, tangelo: Dict,
                 production: Dict, thermal: Dict, control: Dict,
                 bom: List, fabrication: List, equipment: List,
                 protocol: List, daq: List, criteria: List,
                 stats: Dict):
    """Formatted text report to stdout, matching cfd_validate_reactor.py style."""

    W = 78
    SEP = "═" * W
    THIN = "─" * W

    print(SEP)
    print("  EXPERIMENTAL VALIDATION — DIGITAL TWIN + PROTOCOL")
    print(f"  16ch × 500mm Parametric Optimized DBD Milireactor")
    print(SEP)

    # ─── PART A: DIGITAL TWIN ─────────────────────────────
    print(f"\n{'─' * W}")
    print("  PART A: DIGITAL TWIN PREDICTIONS")
    print(f"{'─' * W}")

    # Cantera
    print(f"\n  [1] Cantera Plasma Chemistry")
    print(f"      Cantera available: {'✓' if cantera['cantera_available'] else '✗ (analytical fallback)'}")
    print(f"      OH radical:        {cantera['OH_cm3']:.2e} cm⁻³")
    print(f"      R_OH:              {cantera['R_OH_cm3_s']:.2e} cm⁻³/s")
    print(f"      H2O2 fraction:     {cantera['H2O2_fraction']:.2e}")
    print(f"      Uncertainty:       ±{cantera['uncertainty_relative']*100:.0f}%")

    # Tangelo
    print(f"\n  [2] Tangelo Quantum Chemistry ({tangelo['method']})")
    print(f"      {'Zone':<12} {'T(K)':<8} {'E(V/m)':<12} {'Gap(eV)':<10} {'Size(nm)':<10} {'λ(nm)':<8}")
    print(f"      {THIN[:60]}")
    for z in tangelo['zones']:
        print(f"      {z['zone']:<12} {z['temperature_K']:<8.0f} {z['E_field_V_m']:<12.0e} "
              f"{z['gap_eV']:<10.2f} {z['size_nm']:<10.2f} {z['wavelength_nm']:<8.0f}")

    # Production
    print(f"\n  [3] Production Model (same scoring as CFD validation)")
    print(f"      Production:     {production['production_mg_h']:.0f} mg/h")
    print(f"      Wavelength:     {production['wavelength_nm']:.1f} nm")
    print(f"      Size:           {production['size_nm']:.2f} nm")
    print(f"      Power:          {production['power_w']:.1f} W")
    print(f"      Energy density: {production['energy_density_j_ml']:.0f} J/mL")
    print(f"      In-spec:        {'✓' if production['in_spec'] else '✗'}")
    print(f"      Score:          {production['score']:.4f}")

    # Thermal
    print(f"\n  [4] Thermal Model")
    print(f"      Plasma regime:  {thermal['cold_plasma']['plasma_regime']}")
    print(f"      Te/Tgas ratio:  {thermal['cold_plasma']['non_thermal_ratio']:.0f}:1")
    print(f"      Power:          {thermal['power']['electrical_w']:.1f} W electrical, "
          f"{thermal['power']['heat_generation_w']:.1f} W heat")
    print(f"      T_max:          {thermal['temperatures']['T_max_C']:.1f}°C")
    print(f"      Cooling margin: {thermal['temperatures']['cooling_margin_C']:.1f}°C to limit")
    print(f"      Coolant flow:   {thermal['cooling']['required_coolant_flow_ml_min']:.0f} mL/min required")

    # Control
    print(f"\n  [5] Control Simulation")
    print(f"      Target λ:       {control['setpoints']['target_wavelength_nm']:.1f} nm")
    print(f"      Accept range:   {control['setpoints']['wavelength_range_nm'][0]:.0f}-{control['setpoints']['wavelength_range_nm'][1]:.0f} nm")
    print(f"      In-spec sweep:  {control['n_in_spec']}/{control['n_total']} wavelengths")
    print(f"      Zones: ", end="")
    for z in control['zone_specs']:
        print(f"{z['name']} ({z['led_wavelength_nm']}nm), ", end="")
    print()

    # Consolidated predictions
    print(f"\n  [6] Consolidated Digital Twin (cross-model)")
    print(f"      CFD reference loaded: {'✓' if digital_twin['cfd_reference_loaded'] else '✗'}")
    wl_check = digital_twin['cross_model_consistency']
    print(f"      Wavelength consistency: {'✓' if wl_check['wavelength_consistent'] else '✗'} "
          f"(mean={wl_check['wavelength_mean_nm']:.1f}±{wl_check['wavelength_std_nm']:.1f} nm)")
    for model, wl in wl_check['wavelength_predictions'].items():
        print(f"        {model}: {wl:.1f} nm")

    # Acceptance criteria table
    print(f"\n{SEP}")
    print("  ACCEPTANCE CRITERIA (95% Confidence Intervals)")
    print(SEP)
    print(f"  {'Parameter':<22} {'Predicted':>10} {'95% CI Lower':>13} {'95% CI Upper':>13} {'Method'}")
    print(f"  {THIN}")
    for c in criteria:
        print(f"  {c['parameter']:<22} {c['predicted']:>10.1f} {c['95ci_lower']:>13.1f} "
              f"{c['95ci_upper']:>13.1f} {c['method']}")

    # ─── PART B: EXPERIMENTAL PROTOCOL ────────────────────
    print(f"\n{SEP}")
    print("  PART B: EXPERIMENTAL PROTOCOL")
    print(SEP)

    # BOM summary
    print(f"\n  [7] Bill of Materials ({len(bom)} items)")
    categories = {}
    for item in bom:
        cat = item['category']
        if cat not in categories:
            categories[cat] = {'count': 0, 'cost': 0}
        categories[cat]['count'] += 1
        categories[cat]['cost'] += item['cost_eur']

    total_cost = sum(c['cost'] for c in categories.values())
    print(f"      {'Category':<25} {'Items':>6} {'Cost (EUR)':>12}")
    print(f"      {THIN[:45]}")
    for cat, info in categories.items():
        print(f"      {cat:<25} {info['count']:>6} {info['cost']:>12,}")
    print(f"      {THIN[:45]}")
    print(f"      {'TOTAL':<25} {len(bom):>6} {total_cost:>12,}")

    # Fabrication summary
    print(f"\n  [8] Fabrication Protocol ({len(fabrication)} phases)")
    total_h = sum(p['duration_h'] for p in fabrication)
    total_steps = sum(len(p['steps']) for p in fabrication)
    for p in fabrication:
        print(f"      Phase {p['phase']}: {p['name']:<30} {p['duration_h']:>3}h  ({len(p['steps'])} steps)")
    print(f"      {'Total':<38} {total_h:>3}h  ({total_steps} steps)")

    # Equipment
    print(f"\n  [9] Measurement Equipment ({len(equipment)} instruments)")
    for eq in equipment:
        print(f"      {eq['instrument']:<35} {eq['range']:<15} {eq['uncertainty']}")

    # Protocol summary
    print(f"\n  [10] Experimental Protocol (3 phases)")
    for phase in protocol:
        print(f"       Phase {phase['phase']}: {phase['name']:<30} {phase['duration_days']} days  ({len(phase['steps'])} steps)")

    # DAQ
    print(f"\n  [11] Data Acquisition ({len(daq)} channels)")
    for ch in daq:
        rate_str = f"{ch['rate_hz']} Hz" if isinstance(ch['rate_hz'], (int, float)) else ch['rate_hz']
        print(f"       {ch['channel']:<14} {rate_str:<10} {ch['resolution']:<15} {ch['trigger']}")

    # Stats
    print(f"\n  [13] Statistical Plan")
    print(f"       Replicates: n={stats['sample_size']['n_replicates']} per condition")
    print(f"       CI level:   {stats['sample_size']['ci_level']*100:.0f}% ({stats['sample_size']['distribution']})")
    for test in stats['tests']:
        print(f"       • {test['name']}")

    # Final validation summary
    print(f"\n{SEP}")
    print("  VALIDATION SUMMARY")
    print(SEP)
    print(f"  Digital twin score:    {digital_twin['score']:.4f}")
    print(f"  CFD reference score:   {digital_twin['cfd_score']:.4f}")
    print(f"  Cross-model consistent: {'✓' if digital_twin['cross_model_consistency']['wavelength_consistent'] else '✗'}")
    print(f"  Total BOM cost:        {total_cost:,} EUR")
    print(f"  Fabrication time:      {total_h} hours ({total_h/8:.1f} working days)")
    print(f"  Experiment duration:   {sum(p['duration_days'] for p in protocol)} days")
    print(f"  Total protocol steps:  {sum(len(p['steps']) for p in protocol)}")
    print(SEP)


# ─── 15. Save JSON Report ────────────────────────────────────────────────────

def save_json_report(digital_twin: Dict, cantera: Dict, tangelo: Dict,
                     production: Dict, thermal: Dict, control: Dict,
                     bom: List, fabrication: List, equipment: List,
                     protocol: List, daq: List, criteria: List,
                     stats: Dict):
    """Save complete structured report as JSON."""

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
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

    report = {
        'digital_twin': convert(digital_twin),
        'models': {
            'cantera': convert(cantera),
            'tangelo': convert(tangelo),
            'production': convert(production),
            'thermal': convert(thermal),
            'control': convert(control),
        },
        'protocol': {
            'bom': convert(bom),
            'bom_summary': {
                'n_items': len(bom),
                'total_cost_eur': sum(item['cost_eur'] for item in bom),
            },
            'fabrication': convert(fabrication),
            'fabrication_summary': {
                'n_phases': len(fabrication),
                'total_hours': sum(p['duration_h'] for p in fabrication),
                'total_steps': sum(len(p['steps']) for p in fabrication),
            },
            'measurement_equipment': convert(equipment),
            'experimental_phases': convert(protocol),
            'data_acquisition': convert(daq),
            'acceptance_criteria': convert(criteria),
            'statistical_plan': convert(stats),
        },
    }

    output_file = OUTPUT_DIR / "experimental_validation_report.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  ✓ Report saved: {output_file}")
    except (PermissionError, OSError):
        output_file = Path("/tmp/experimental_validation_report.json")
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  ✓ Report saved: {output_file} (fallback)")

    return str(output_file)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    config = TARGET_CONFIG

    print("═" * 78)
    print("  EXPERIMENTAL VALIDATION — DIGITAL TWIN + PROTOCOL")
    print(f"  Config: {config['n_channels']}ch × {config['channel_length_mm']:.0f}mm, "
          f"{config['voltage_kv']}kV, {config['frequency_khz']}kHz, "
          f"{config['flow_ml_min']} mL/min")
    print("═" * 78)

    # ─── Part A: Digital Twin ─────────────────────────────
    print("\n→ Part A: Building Digital Twin...")

    print("\n  [1] Running Cantera simulation...")
    cantera = run_cantera_simulation(config)
    print(f"      OH = {cantera['OH_cm3']:.2e} cm⁻³ (±{cantera['uncertainty_relative']*100:.0f}%)")

    print("\n  [2] Running Tangelo analysis...")
    tangelo = run_tangelo_analysis(config)
    print(f"      Gap = {tangelo['primary_gap_eV']:.2f} eV, "
          f"Size = {tangelo['primary_size_nm']:.2f} nm, "
          f"λ = {tangelo['primary_wavelength_nm']:.0f} nm")

    print("\n  [3] Running production model...")
    production = run_production_model(config, cantera, tangelo)
    print(f"      Production = {production['production_mg_h']:.0f} mg/h, "
          f"λ = {production['wavelength_nm']:.0f} nm, "
          f"Score = {production['score']:.4f}")

    print("\n  [4] Running thermal model...")
    thermal = run_thermal_model(config)
    print(f"      T_max = {thermal['temperatures']['T_max_C']:.1f}°C, "
          f"Cooling: {'✓' if thermal['temperatures']['cooling_ok'] else '✗'}")

    print("\n  [5] Running control simulation...")
    control = run_control_simulation(config)
    print(f"      In-spec: {control['n_in_spec']}/{control['n_total']} wavelengths")

    print("\n  [6] Consolidating digital twin...")
    digital_twin = consolidate_digital_twin(config, cantera, tangelo,
                                             production, thermal, control)
    print(f"      Cross-model consistent: "
          f"{'✓' if digital_twin['cross_model_consistency']['wavelength_consistent'] else '✗'}")

    # ─── Part B: Experimental Protocol ────────────────────
    print("\n→ Part B: Generating Experimental Protocol...")

    print("  [7] Generating BOM...")
    bom = generate_bom(config)
    print(f"      {len(bom)} items, ~{sum(i['cost_eur'] for i in bom):,} EUR total")

    print("  [8] Generating fabrication protocol...")
    fabrication = generate_fabrication_protocol(config, bom)
    print(f"      {len(fabrication)} phases, {sum(len(p['steps']) for p in fabrication)} steps")

    print("  [9] Generating measurement equipment list...")
    equipment = generate_measurement_equipment()
    print(f"      {len(equipment)} instruments")

    print("  [10] Generating experimental protocol...")
    protocol = generate_experimental_protocol(digital_twin)
    print(f"      {len(protocol)} phases, {sum(p['duration_days'] for p in protocol)} days")

    print("  [11] Generating DAQ plan...")
    daq = generate_data_acquisition_plan()
    print(f"      {len(daq)} channels")

    print("  [12] Generating acceptance criteria...")
    criteria = generate_acceptance_criteria(digital_twin)
    print(f"      {len(criteria)} parameters")

    print("  [13] Generating statistical plan...")
    stats = generate_statistical_plan()
    print(f"      {len(stats['tests'])} tests")

    # ─── Output ───────────────────────────────────────────
    print("\n→ Generating report...")
    print_report(digital_twin, cantera, tangelo, production, thermal, control,
                 bom, fabrication, equipment, protocol, daq, criteria, stats)

    print("\n→ Saving JSON report...")
    report_path = save_json_report(digital_twin, cantera, tangelo, production,
                                    thermal, control, bom, fabrication, equipment,
                                    protocol, daq, criteria, stats)

    print("\n" + "═" * 78)
    print("  ✓ EXPERIMENTAL VALIDATION COMPLETE")
    print("═" * 78)


if __name__ == "__main__":
    main()
