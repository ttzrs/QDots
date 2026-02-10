#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  OPTIMIZACIÓN PARA CALIDAD AZUL (λ=460nm)
  KPIs reales: QY+estabilidad > Espectro > Selectividad > PDI > STY
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from pathlib import Path
import json
from datetime import datetime

OUTPUT_DIR = Path("/var/home/joss/Datos/Proyectos/QDots/openfoam_reactor/quality_optimization")
OUTPUT_DIR.mkdir(exist_ok=True)

# Constantes físicas
E_BULK = 1.50    # eV - gap del grafeno dopado N
A_CONF = 7.26    # eV·nm² - constante de confinamiento

# Para λ=460nm: E_gap = 1240/460 = 2.70 eV
# d = sqrt(A_CONF / (E_gap - E_BULK)) = sqrt(7.26/1.20) = 2.46 nm

def calculate_kpis(x):
    """Calcula todos los KPIs de calidad"""

    # Desempaquetar
    (width, height, length, n_turns, Q_liq, Q_gas, ar, n2,
     V, f, duty, pulse, T, P, conc, pH, elec_w, gap, coverage) = x

    T_K = T + 273.15

    # ─────────────────────────────────────────────────────────────────────
    # VOLUMEN Y TIEMPO DE RESIDENCIA
    # ─────────────────────────────────────────────────────────────────────
    V_reactor = (width/1000) * (height/1000) * (length/1000) * n_turns  # m³
    V_reactor_L = V_reactor * 1000  # L
    Q_total = (Q_liq + Q_gas) / 60 / 1e6  # m³/s
    tau = V_reactor / (Q_total + 1e-12)  # s

    # ─────────────────────────────────────────────────────────────────────
    # TAMAÑO DE PARTÍCULA - Modelo para control de tamaño
    # ─────────────────────────────────────────────────────────────────────
    # Para CQDs azules necesitamos d ≈ 2.4 nm
    # Factores que reducen tamaño: T baja, tau corto, alta nucleación

    # Cinética de Arrhenius
    R = 8.314
    Ea_nuc = 150  # kJ/mol
    Ea_grow = 50  # kJ/mol

    k_nuc = 1e12 * np.exp(-Ea_nuc * 1000 / (R * T_K))
    k_grow = 1e11 * np.exp(-Ea_grow * 1000 / (R * T_K))

    # Ratio nucleación/crecimiento determina tamaño
    # Alto ratio → muchos núcleos pequeños
    ratio = k_nuc / (k_grow + 1e-20)

    # Tamaño: modelo empírico calibrado
    # T baja: más nucleación relativa → menor tamaño
    # tau corto: menos tiempo de crecimiento
    # conc alta: supersaturación → más núcleos

    d_T = 1.5 + 2.5 * (T_K - 300) / 70  # 1.5nm a 27°C, ~4nm a 97°C
    d_tau = 1.5 + 1.0 * np.log(1 + tau)  # crece logarítmicamente con tau
    d_conc = 1.8 - 0.3 * (conc - 50) / 50  # menor tamaño a mayor conc (más núcleos)

    mean_size = (d_T * 0.5 + d_tau * 0.3 + d_conc * 0.2)
    mean_size = np.clip(mean_size, 1.8, 5.0)

    # Emisión por confinamiento cuántico
    E_gap = E_BULK + A_CONF / (mean_size**2)
    wavelength = 1240 / E_gap

    # ─────────────────────────────────────────────────────────────────────
    # PDI (Índice de Polidispersidad)
    # ─────────────────────────────────────────────────────────────────────
    # Mejor PDI: flujo laminar, T estable, tau uniforme

    Re = (Q_liq/60/1e6) / ((width/1000) * (height/1000)) * 1000 * (width/1000) / 0.001

    pdi_base = 0.10
    pdi_T = 0.05 * abs(T - 45) / 25  # óptimo ~45°C
    pdi_Re = 0.05 * min(Re/500, 1)  # turbulencia aumenta dispersión
    pdi_tau = 0.03 * abs(tau - 1.5)  # tau óptimo ~1.5s

    pdi = pdi_base + pdi_T + pdi_Re + pdi_tau
    pdi = np.clip(pdi, 0.05, 0.40)

    size_std = mean_size * np.sqrt(pdi)

    # ─────────────────────────────────────────────────────────────────────
    # FWHM (Anchura a Media Altura)
    # ─────────────────────────────────────────────────────────────────────
    # Correlación con PDI: mayor dispersión → espectro más ancho
    fwhm = 22 + 50 * pdi  # 22nm mínimo para CQDs perfectos
    fwhm = np.clip(fwhm, 22, 70)

    # ─────────────────────────────────────────────────────────────────────
    # QUANTUM YIELD
    # ─────────────────────────────────────────────────────────────────────
    # Óptimo: T moderada (~50°C), pH 5-7, campo E moderado

    T_opt = 323  # 50°C
    pH_opt = 6.0
    E_field = V * 1e3 / (gap / 1000)  # V/m
    E_opt = 12e5  # V/m óptimo

    qy_T = np.exp(-0.003 * (T_K - T_opt)**2)
    qy_pH = np.exp(-0.12 * (pH - pH_opt)**2)
    qy_E = np.exp(-1e-13 * (E_field - E_opt)**2)

    # QY base ~10% para condiciones óptimas
    qy_base = 0.10
    quantum_yield = qy_base * qy_T * qy_pH * qy_E
    quantum_yield = np.clip(quantum_yield, 0.02, 0.22)

    # CV del QY (estabilidad)
    qy_cv = 0.06 + 0.08 * abs(T - 50)/25 + 0.05 * abs(pH - 6)/2
    qy_cv = np.clip(qy_cv, 0.05, 0.35)

    # ─────────────────────────────────────────────────────────────────────
    # PUREZA ESPECTRAL
    # ─────────────────────────────────────────────────────────────────────
    # Baja dispersión de tamaños → espectro limpio
    spectral_purity = 0.95 * np.exp(-3 * pdi)
    spectral_purity = np.clip(spectral_purity, 0.50, 0.95)

    # ─────────────────────────────────────────────────────────────────────
    # CONVERSIÓN Y SELECTIVIDAD
    # ─────────────────────────────────────────────────────────────────────
    # Conversión: tau suficiente, T adecuada
    tau_half = 1.5  # tiempo para 50% conversión
    conversion = 0.9 * (1 - np.exp(-tau / tau_half))
    conversion = np.clip(conversion, 0.15, 0.92)

    # Selectividad: T baja y E moderado → menos carbonilla
    sel_T = np.exp(-0.002 * (T_K - 315)**2)  # óptimo ~42°C
    sel_E = np.exp(-2e-13 * (E_field - 10e5)**2)
    sel_pH = np.exp(-0.08 * (pH - 5.5)**2)

    selectivity = 0.80 * sel_T * sel_E * sel_pH
    selectivity = np.clip(selectivity, 0.30, 0.88)

    useful_yield = conversion * selectivity

    # ─────────────────────────────────────────────────────────────────────
    # PRODUCTIVIDAD (STY)
    # ─────────────────────────────────────────────────────────────────────
    MW_cqd = 800  # g/mol efectivo
    mol_s = (conc / 1000) * useful_yield * (Q_liq / 60 / 1e6) * 1000  # mol/s
    production_g_h = mol_s * MW_cqd * 3600  # g/h

    sty = production_g_h / max(V_reactor_L, 0.001)  # g/h·L

    # ─────────────────────────────────────────────────────────────────────
    # OPERABILIDAD
    # ─────────────────────────────────────────────────────────────────────
    D_h = 2 * (width/1000) * (height/1000) / ((width + height)/1000)
    mu = 0.001

    gas_holdup = Q_gas / (Q_liq + Q_gas + 0.01)
    phi_2 = 1 + 20 * gas_holdup + gas_holdup**2

    dp = 32 * mu * (Q_liq/60/1e6 / ((width/1000)*(height/1000))) * (length/1000) * n_turns / (D_h**2)
    pressure_drop = dp * phi_2
    pressure_drop = np.clip(pressure_drop, 100, 5e6)

    fouling_rate = 15 * (1 + (T - 30)/40) * (1 + abs(pH - 6)/3) * conversion
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
        'conversion': conversion,
        'selectivity': selectivity,
        'useful_yield': useful_yield,
        'sty_g_h_L': sty,
        'pressure_drop_Pa': pressure_drop,
        'mtbc_hours': mtbc,
        'tau_s': tau,
    }


def quality_score(x, target_wavelength=460.0):
    """Score compuesto para calidad premium"""

    kpis = calculate_kpis(x)

    # Pesos según prioridades para producto premium:
    # 1. QY + estabilidad: 25%
    # 2. Espectro (λ, FWHM, pureza): 35%  <-- más peso para color correcto
    # 3. Selectividad: 15%
    # 4. PDI: 20%
    # 5. STY: 5%

    score = 0.0

    # 1. QY + estabilidad (25%)
    qy_score = kpis['quantum_yield'] * (1 - kpis['qy_cv'])
    score += 0.25 * qy_score * 100  # escalar a ~2.5 max

    # 2. Espectro (35%)
    wl_match = np.exp(-0.0003 * (kpis['wavelength_nm'] - target_wavelength)**2)
    fwhm_score = np.exp(-0.003 * (kpis['fwhm_nm'] - 28)**2)  # FWHM óptimo ~28nm
    spectral_score = wl_match * fwhm_score * kpis['spectral_purity']
    score += 0.35 * spectral_score * 10  # escalar a ~3.5 max

    # 3. Selectividad (15%)
    score += 0.15 * kpis['useful_yield'] * 10  # escalar a ~1.5 max

    # 4. PDI (20%)
    pdi_score = np.exp(-10 * kpis['pdi'])
    score += 0.20 * pdi_score * 10  # escalar a ~2 max

    # 5. STY (5%)
    sty_norm = min(kpis['sty_g_h_L'] / 8.0, 1.0)
    score += 0.05 * sty_norm * 10  # escalar a ~0.5 max

    return -score, kpis


def objective(x):
    score, _ = quality_score(x, target_wavelength=460.0)
    return score


# Límites de diseño
bounds = [
    (1.5, 4.0),    # width mm
    (0.8, 2.5),    # height mm
    (80, 220),     # length mm
    (4, 8),        # n_turns
    (15, 55),      # Q_liq mL/min
    (8, 35),       # Q_gas mL/min
    (0.3, 0.7),    # ar fraction
    (0.2, 0.5),    # n2 fraction
    (8, 15),       # V kV
    (8, 18),       # f kHz
    (0.35, 0.65),  # duty
    (40, 100),     # pulse μs
    (38, 55),      # T °C  <-- rango para partículas pequeñas
    (105, 125),    # P kPa
    (40, 90),      # conc mM
    (5.2, 6.8),    # pH
    (1.0, 2.0),    # electrode width
    (1.0, 1.8),    # gap
    (0.55, 0.80),  # coverage
]


def main():
    print("="*80)
    print("  OPTIMIZACIÓN PARA CALIDAD AZUL (λ=460nm)")
    print("  KPIs: QY+estabilidad > Espectro > Selectividad > PDI > STY")
    print("="*80)

    print("\n→ Ejecutando evolución diferencial (300 generaciones)...")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=300,
        popsize=25,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        disp=True,
        workers=1,  # Single process
    )

    best_x = result.x
    _, kpis = quality_score(best_x, 460.0)

    print("\n" + "="*80)
    print("  DISEÑO ÓPTIMO PARA CQDs AZULES DE ALTA CALIDAD")
    print("="*80)

    print(f"\n  ┌──────────────────────────────────────────────────────────────────┐")
    print(f"  │ PROPIEDADES ÓPTICAS                                              │")
    print(f"  ├──────────────────────────────────────────────────────────────────┤")
    print(f"  │   λ emisión: {kpis['wavelength_nm']:6.1f} nm  (target: 460 nm)                    │")
    print(f"  │   Tamaño: {kpis['mean_size_nm']:5.2f} ± {kpis['size_std_nm']:.2f} nm                                    │")
    print(f"  │   FWHM: {kpis['fwhm_nm']:5.1f} nm                                                │")
    print(f"  │   Pureza espectral: {kpis['spectral_purity']*100:5.1f}%                               │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌──────────────────────────────────────────────────────────────────┐")
    print(f"  │ QUANTUM YIELD Y ESTABILIDAD                                      │")
    print(f"  ├──────────────────────────────────────────────────────────────────┤")
    print(f"  │   QY: {kpis['quantum_yield']*100:5.1f}%                                                   │")
    print(f"  │   CV: {kpis['qy_cv']*100:5.1f}%                                                    │")
    print(f"  │   Score QY×(1-CV): {kpis['quantum_yield']*(1-kpis['qy_cv']):.4f}                              │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌──────────────────────────────────────────────────────────────────┐")
    print(f"  │ DISPERSIÓN Y MONODISPERSIDAD                                     │")
    print(f"  ├──────────────────────────────────────────────────────────────────┤")
    print(f"  │   PDI: {kpis['pdi']:.4f}                                                   │")
    quality_pdi = "★★★ Excelente" if kpis['pdi'] < 0.1 else ("★★ Bueno" if kpis['pdi'] < 0.2 else "★ Aceptable")
    print(f"  │   Evaluación: {quality_pdi:<20}                             │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌──────────────────────────────────────────────────────────────────┐")
    print(f"  │ CONVERSIÓN Y SELECTIVIDAD                                        │")
    print(f"  ├──────────────────────────────────────────────────────────────────┤")
    print(f"  │   Conversión: {kpis['conversion']*100:5.1f}%                                          │")
    print(f"  │   Selectividad: {kpis['selectivity']*100:5.1f}%                                        │")
    print(f"  │   Rendimiento útil: {kpis['useful_yield']*100:5.1f}%                                   │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌──────────────────────────────────────────────────────────────────┐")
    print(f"  │ PRODUCTIVIDAD                                                    │")
    print(f"  ├──────────────────────────────────────────────────────────────────┤")
    print(f"  │   STY: {kpis['sty_g_h_L']:6.2f} g/h·L                                             │")
    print(f"  │   τ residencia: {kpis['tau_s']:5.2f} s                                         │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌──────────────────────────────────────────────────────────────────┐")
    print(f"  │ OPERABILIDAD                                                     │")
    print(f"  ├──────────────────────────────────────────────────────────────────┤")
    print(f"  │   ΔP: {kpis['pressure_drop_Pa']:8.0f} Pa                                            │")
    print(f"  │   MTBC: {kpis['mtbc_hours']:7.1f} h                                               │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ═══════════════════════════════════════════════════════════════════")
    print(f"  PARÁMETROS DE DISEÑO")
    print(f"  ═══════════════════════════════════════════════════════════════════")

    print(f"\n  GEOMETRÍA:")
    print(f"    Canal: {best_x[0]:.2f} × {best_x[1]:.2f} mm")
    print(f"    Longitud: {best_x[2]:.1f} mm, {int(best_x[3])} vueltas")

    print(f"\n  ELECTRODOS:")
    print(f"    Ancho: {best_x[16]:.2f} mm, Gap: {best_x[17]:.2f} mm")
    print(f"    Cobertura: {best_x[18]*100:.0f}%")

    print(f"\n  FLUJO:")
    print(f"    Líquido: {best_x[4]:.1f} mL/min")
    print(f"    Gas: {best_x[5]:.1f} mL/min (Ar:{best_x[6]*100:.0f}%, N₂:{best_x[7]*100:.0f}%)")

    print(f"\n  PLASMA:")
    print(f"    Voltaje: {best_x[8]:.1f} kV")
    print(f"    Frecuencia: {best_x[9]:.1f} kHz")
    print(f"    Duty: {best_x[10]*100:.0f}%")
    print(f"    Pulso: {best_x[11]:.0f} μs")

    print(f"\n  OPERACIÓN:")
    print(f"    Temperatura: {best_x[12]:.1f}°C")
    print(f"    Presión: {best_x[13]:.0f} kPa")
    print(f"    Precursor: {best_x[14]:.1f} mM")
    print(f"    pH: {best_x[15]:.1f}")

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
        'target_wavelength': 460.0,
        'optimization': {
            'method': 'differential_evolution',
            'iterations': result.nit,
            'function_evaluations': result.nfev,
        },
        'timestamp': datetime.now().isoformat(),
    }

    output_file = OUTPUT_DIR / f"blue_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✓ Guardado: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
