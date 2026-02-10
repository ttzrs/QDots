#!/usr/bin/env python3
"""
OPTIMIZACIÓN VERDE (λ=530nm) - CON PENALIZACIÓN DE BIMODALIDAD
d objetivo ≈ 2.94 nm (interpolado entre azul 2.39 y rojo 3.82)
"""

import numpy as np
from scipy.optimize import differential_evolution
from pathlib import Path
import json
from datetime import datetime

OUTPUT_DIR = Path("/var/home/joss/Datos/Proyectos/QDots/openfoam_reactor/quality_optimization")

E_BULK = 1.50
A_CONF = 7.26


def calculate_kpis_green(x):
    """Modelo calibrado para VERDE con control de bimodalidad"""

    (width, height, length, n_turns, Q_liq, Q_gas, ar, n2,
     V, f, duty, pulse, T, P, conc, pH, elec_w, gap, coverage) = x

    T_K = T + 273.15

    V_reactor = (width/1000) * (height/1000) * (length/1000) * n_turns
    V_reactor_L = V_reactor * 1000
    Q_total = (Q_liq + Q_gas) / 60 / 1e6
    tau = V_reactor / (Q_total + 1e-12)

    # MODELO DE TAMAÑO CALIBRADO PARA VERDE
    R = 8.314
    Ea_nuc = 150
    Ea_grow = 50

    k_nuc = 1e12 * np.exp(-Ea_nuc * 1000 / (R * T_K))
    k_grow = 1e11 * np.exp(-Ea_grow * 1000 / (R * T_K))

    d_T = 1.8 + 2.2 * (T_K - 300) / 65
    d_tau = 1.8 + 0.8 * np.log(1 + tau * 1.2)
    d_conc = 2.5 + 0.4 * (conc - 40) / 30

    mean_size = d_T * 0.50 + d_tau * 0.30 + d_conc * 0.20
    mean_size = np.clip(mean_size, 2.2, 4.5)

    E_gap = E_BULK + A_CONF / (mean_size**2)
    wavelength = 1240 / E_gap

    # PDI CON PENALIZACIÓN POR BIMODALIDAD
    Re = (Q_liq/60/1e6) / ((width/1000) * (height/1000)) * 1000 * (width/1000) / 0.001

    pdi_base = 0.10
    pdi_T = 0.03 * abs(T - 60) / 20
    pdi_tau = 0.04 * abs(tau - 2.5) / 1.5
    pdi_Re = 0.03 * min(Re/400, 1)

    bimodal_risk = 0.05 * (np.exp(-0.1 * (T - 60)**2 / 25) - 1)**2

    pdi = pdi_base + pdi_T + pdi_tau + pdi_Re + bimodal_risk
    pdi = np.clip(pdi, 0.06, 0.35)

    size_std = mean_size * np.sqrt(pdi)

    # FWHM
    fwhm_base = 22
    fwhm_pdi = 35 * pdi
    fwhm_bimodal = 10 * bimodal_risk

    fwhm = fwhm_base + fwhm_pdi + fwhm_bimodal
    fwhm = np.clip(fwhm, 22, 60)

    # QY
    T_opt = 333
    pH_opt = 6.2

    qy_T = np.exp(-0.0025 * (T_K - T_opt)**2)
    qy_pH = np.exp(-0.12 * (pH - pH_opt)**2)

    E_field = V * 1e3 / (gap / 1000)
    E_opt = 8e5
    qy_E = np.exp(-1.5e-13 * (E_field - E_opt)**2)

    trap_index = 0.1 * (abs(mean_size - 3.0) / 1.0)**2
    passivation = 0.9 + 0.1 * n2 - 0.05 * abs(pH - 6.2)

    qy_base = 0.09
    quantum_yield = qy_base * qy_T * qy_pH * qy_E * passivation * (1 - trap_index)
    quantum_yield = np.clip(quantum_yield, 0.02, 0.18)

    qy_cv = 0.07 + 0.05 * abs(T - 60)/20 + 0.04 * abs(pH - 6.2)/2
    qy_cv = np.clip(qy_cv, 0.05, 0.35)

    spectral_purity = 0.95 * np.exp(-3 * pdi) * (1 - bimodal_risk)
    spectral_purity = np.clip(spectral_purity, 0.45, 0.95)

    # CONVERSIÓN Y SELECTIVIDAD
    conversion = 0.82 * (1 - np.exp(-tau / 1.8))
    conversion = np.clip(conversion, 0.20, 0.88)

    sel_T = np.exp(-0.0018 * (T_K - 328)**2)
    sel_pH = np.exp(-0.10 * (pH - 6.0)**2)
    sel_E = np.exp(-2e-13 * (E_field - 7e5)**2)

    selectivity = 0.75 * sel_T * sel_pH * sel_E
    selectivity = np.clip(selectivity, 0.30, 0.80)

    useful_yield = conversion * selectivity

    # STY
    MW_cqd = 1000
    mol_s = (conc / 1000) * useful_yield * (Q_liq / 60 / 1e6) * 1000
    production_g_h = mol_s * MW_cqd * 3600
    sty = production_g_h / max(V_reactor_L, 0.001)

    D_h = 2 * (width/1000) * (height/1000) / ((width + height)/1000)
    gas_holdup = Q_gas / (Q_liq + Q_gas + 0.01)
    phi_2 = 1 + 19 * gas_holdup + gas_holdup**2
    dp = 32 * 0.001 * (Q_liq/60/1e6 / ((width/1000)*(height/1000))) * (length/1000) * n_turns / (D_h**2)
    pressure_drop = np.clip(dp * phi_2, 100, 5e6)

    fouling_rate = 14 * (1 + (T - 35)/45) * (1 + abs(pH - 6.2)/3) * conversion
    mtbc = pressure_drop / (fouling_rate + 1)

    return {
        'wavelength_nm': wavelength,
        'mean_size_nm': mean_size,
        'size_std_nm': size_std,
        'pdi': pdi,
        'fwhm_nm': fwhm,
        'quantum_yield': quantum_yield,
        'qy_cv': qy_cv,
        'spectral_purity': spectral_purity,
        'bimodal_risk': bimodal_risk,
        'conversion': conversion,
        'selectivity': selectivity,
        'useful_yield': useful_yield,
        'sty_g_h_L': sty,
        'pressure_drop_Pa': pressure_drop,
        'mtbc_hours': mtbc,
        'tau_s': tau,
        'trap_index': trap_index,
        'passivation': passivation,
    }


def quality_score_green(x):
    """Score con pesos ajustados para VERDE"""

    kpis = calculate_kpis_green(x)
    target_wl = 530.0

    score = 0.0
    penalty = 0.0

    # RESTRICCIONES
    if kpis['wavelength_nm'] < 520:
        penalty += 50 * (520 - kpis['wavelength_nm'])**2
    elif kpis['wavelength_nm'] > 540:
        penalty += 50 * (kpis['wavelength_nm'] - 540)**2

    if kpis['pdi'] > 0.18:
        penalty += 100 * (kpis['pdi'] - 0.18)**2

    if kpis['fwhm_nm'] > 32:
        penalty += 80 * (kpis['fwhm_nm'] - 32)**2

    if kpis['quantum_yield'] < 0.04:
        penalty += 50 * (0.04 - kpis['quantum_yield'])**2

    # SCORE
    wl_match = np.exp(-0.0008 * (kpis['wavelength_nm'] - target_wl)**2)
    score += 0.30 * wl_match * 10

    fwhm_score = np.exp(-0.008 * (kpis['fwhm_nm'] - 26)**2)
    score += 0.20 * fwhm_score * 10

    qy_score = kpis['quantum_yield'] * (1 - kpis['qy_cv'])
    score += 0.25 * qy_score * 100

    pdi_score = np.exp(-12 * kpis['pdi'])
    score += 0.15 * pdi_score * 10

    score += 0.08 * kpis['useful_yield'] * 10

    sty_norm = min(kpis['sty_g_h_L'] / 6.0, 1.0)
    score += 0.02 * sty_norm * 10

    score -= 5 * kpis['bimodal_risk']

    return -(score - penalty), kpis


def objective(x):
    score, _ = quality_score_green(x)
    return score


bounds = [
    (1.7, 2.2), (1.8, 2.2), (95, 115), (4, 5),
    (11, 14), (26, 32), (0.45, 0.55), (0.26, 0.34),
    (6.5, 7.5), (10, 13), (0.60, 0.68), (85, 100),
    (58, 63), (110, 120), (45, 55), (6.0, 6.5),
    (1.8, 2.0), (1.7, 1.9), (0.65, 0.75),
]


def main():
    print("="*80)
    print("  OPTIMIZACIÓN VERDE (λ=530nm) - CON CONTROL DE BIMODALIDAD")
    print("  Objetivo: d ≈ 2.94 nm, FWHM ≤ 32nm, PDI ≤ 0.18")
    print("="*80)

    print("\n→ Fase 1: DOE exploratorio")
    semilla = [1.9, 2.0, 105, 4, 12.5, 29, 0.50, 0.30, 7.0, 11.5, 0.64, 92, 60, 115, 50, 6.2, 1.9, 1.8, 0.70]

    for T in [58, 60, 62]:
        for Q in [11, 13]:
            x_test = semilla.copy()
            x_test[12] = T
            x_test[4] = Q
            _, kpis = quality_score_green(x_test)
            print(f"  T={T}°C, Q={Q}mL/min → λ={kpis['wavelength_nm']:.0f}nm, PDI={kpis['pdi']:.3f}, FWHM={kpis['fwhm_nm']:.1f}nm")

    print("\n→ Fase 2: Evolución diferencial")

    result = differential_evolution(
        objective, bounds,
        maxiter=200, popsize=20,
        mutation=(0.5, 1.0), recombination=0.7,
        seed=456, polish=True, disp=True, workers=1,
    )

    best_x = result.x
    _, kpis = quality_score_green(best_x)

    print("\n" + "="*80)
    print("  DISEÑO ÓPTIMO PARA CQDs VERDES")
    print("="*80)

    constraints_ok = (
        520 <= kpis['wavelength_nm'] <= 540 and
        kpis['pdi'] <= 0.18 and
        kpis['fwhm_nm'] <= 32 and
        kpis['quantum_yield'] >= 0.04
    )

    status = "✓ TODAS CUMPLIDAS" if constraints_ok else "✗ ALGUNA VIOLADA"
    print(f"\n  RESTRICCIONES: {status}")

    wl_ok = "✓" if 520 <= kpis['wavelength_nm'] <= 540 else "✗"
    pdi_ok = "✓" if kpis['pdi'] <= 0.18 else "✗"
    fwhm_ok = "✓" if kpis['fwhm_nm'] <= 32 else "✗"
    qy_ok = "✓" if kpis['quantum_yield'] >= 0.04 else "✗"

    print(f"    λ ∈ [520,540]: {kpis['wavelength_nm']:.1f} nm {wl_ok}")
    print(f"    PDI ≤ 0.18: {kpis['pdi']:.4f} {pdi_ok}")
    print(f"    FWHM ≤ 32: {kpis['fwhm_nm']:.1f} nm {fwhm_ok}")
    print(f"    QY ≥ 4%: {kpis['quantum_yield']*100:.1f}% {qy_ok}")

    print(f"\n  PROPIEDADES ÓPTICAS:")
    print(f"    λ emisión: {kpis['wavelength_nm']:.1f} nm (target: 530 nm)")
    print(f"    Tamaño: {kpis['mean_size_nm']:.2f} ± {kpis['size_std_nm']:.2f} nm")
    print(f"    FWHM: {kpis['fwhm_nm']:.1f} nm")
    print(f"    Pureza espectral: {kpis['spectral_purity']*100:.1f}%")
    print(f"    Riesgo bimodal: {kpis['bimodal_risk']:.4f}")

    print(f"\n  QUANTUM YIELD:")
    print(f"    QY: {kpis['quantum_yield']*100:.1f}%")
    print(f"    CV: {kpis['qy_cv']*100:.1f}%")

    print(f"\n  DISPERSIÓN:")
    quality_pdi = "★★★ Excelente" if kpis['pdi'] < 0.1 else ("★★ Bueno" if kpis['pdi'] < 0.18 else "★ Aceptable")
    print(f"    PDI: {kpis['pdi']:.4f} ({quality_pdi})")

    print(f"\n  CONVERSIÓN:")
    print(f"    Conversión: {kpis['conversion']*100:.1f}%")
    print(f"    Selectividad: {kpis['selectivity']*100:.1f}%")
    print(f"    Rendimiento útil: {kpis['useful_yield']*100:.1f}%")

    print(f"\n  PRODUCTIVIDAD:")
    print(f"    STY: {kpis['sty_g_h_L']:.2f} g/h·L")
    print(f"    τ: {kpis['tau_s']:.2f} s")

    print(f"\n  PARÁMETROS DE DISEÑO:")
    print(f"    Canal: {best_x[0]:.2f} × {best_x[1]:.2f} mm, L={best_x[2]:.0f}mm, {int(best_x[3])} vueltas")
    print(f"    Flujo: Liq={best_x[4]:.1f} mL/min, Gas={best_x[5]:.1f} mL/min")
    print(f"    Gas: Ar={best_x[6]*100:.0f}%, N₂={best_x[7]*100:.0f}%")
    print(f"    Plasma: V={best_x[8]:.1f}kV, f={best_x[9]:.1f}kHz, duty={best_x[10]*100:.0f}%")
    print(f"    Operación: T={best_x[12]:.1f}°C, P={best_x[13]:.0f}kPa, {best_x[14]:.0f}mM, pH={best_x[15]:.1f}")

    # COMPARACIÓN RGB
    print(f"\n  {'='*60}")
    print(f"  COMPARACIÓN RGB COMPLETA")
    print(f"  {'='*60}")
    print(f"\n  Parámetro          AZUL         VERDE        ROJO")
    print(f"  {'-'*54}")
    print(f"  λ emisión          447 nm       {kpis['wavelength_nm']:.0f} nm       621 nm")
    print(f"  Tamaño             2.39 nm      {kpis['mean_size_nm']:.2f} nm      3.82 nm")
    print(f"  FWHM               29 nm        {kpis['fwhm_nm']:.0f} nm        31 nm")
    print(f"  QY                 3.3%         {kpis['quantum_yield']*100:.1f}%         5.4%")
    print(f"  PDI                0.14         {kpis['pdi']:.2f}         0.13")
    print(f"  T                  55°C         {best_x[12]:.0f}°C         65°C")
    print(f"  τ                  1.9 s        {kpis['tau_s']:.1f} s        3.1 s")
    print(f"  Conc               40 mM        {best_x[14]:.0f} mM        60 mM")

    # Guardar
    output = {
        'design': {
            'channel_width_mm': float(best_x[0]),
            'channel_height_mm': float(best_x[1]),
            'channel_length_mm': float(best_x[2]),
            'n_turns': int(best_x[3]),
            'liquid_flow_ml_min': float(best_x[4]),
            'gas_flow_ml_min': float(best_x[5]),
            'ar_fraction': float(best_x[6]),
            'n2_fraction': float(best_x[7]),
            'voltage_kv': float(best_x[8]),
            'frequency_khz': float(best_x[9]),
            'duty_cycle': float(best_x[10]),
            'pulse_width_us': float(best_x[11]),
            'temperature_C': float(best_x[12]),
            'pressure_kPa': float(best_x[13]),
            'precursor_mM': float(best_x[14]),
            'pH': float(best_x[15]),
            'electrode_width_mm': float(best_x[16]),
            'electrode_gap_mm': float(best_x[17]),
            'electrode_coverage': float(best_x[18]),
        },
        'kpis': {k: float(v) for k, v in kpis.items()},
        'score': float(-result.fun),
        'target_wavelength': 530.0,
        'color': 'green',
        'constraints_satisfied': constraints_ok,
        'timestamp': datetime.now().isoformat(),
    }

    output_file = OUTPUT_DIR / f"green_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✓ Guardado: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
