#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  OPTIMIZACIÓN PARA CALIDAD - KPIs REALES

  Prioridad (producto premium: fotónica/bio/tinta):
  1. QY + estabilidad (CV)
  2. λ pico + FWHM + estabilidad espectral
  3. Selectividad (producto útil vs carbonilla)
  4. Dispersión de tamaño (PDI)
  5. STY sin degradar lo anterior
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path("/var/home/joss/Datos/Proyectos/QDots/openfoam_reactor/pinn_training")
OUTPUT_DIR = Path("/var/home/joss/Datos/Proyectos/QDots/openfoam_reactor/quality_optimization")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════════
#  DATACLASS PARA KPIs DE CALIDAD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityKPIs:
    """KPIs reales para optimización de calidad"""

    # 1. RENDIMIENTO CUÁNTICO Y ESTABILIDAD
    quantum_yield: float           # QY (fracción 0-1)
    qy_cv: float                   # Coeficiente de variación del QY (%)
    qy_stability_score: float      # Score combinado QY×(1-CV)

    # 2. PROPIEDADES ESPECTRALES
    emission_peak_nm: float        # λ pico
    fwhm_nm: float                 # Anchura a media altura
    spectral_purity: float         # 1 - (área colas / área total)
    stokes_shift_nm: float         # Diferencia absorción-emisión

    # 3. CONVERSIÓN Y SELECTIVIDAD
    precursor_conversion: float    # % precursor → productos
    selectivity: float             # % productos → CQDs útiles (vs carbonilla)
    useful_yield: float            # conversion × selectivity

    # 4. TAMAÑO Y DISPERSIÓN
    mean_size_nm: float            # Tamaño medio
    size_std_nm: float             # Desviación estándar
    pdi: float                     # Índice de polidispersidad (σ/mean)²
    size_emission_correlation: float  # R² entre tamaño y λ

    # 5. PRODUCTIVIDAD (sin sacrificar calidad)
    sty_g_h_L: float              # Space-Time Yield (g/h·L)
    energy_efficiency_g_kWh: float # Eficiencia energética

    # 6. ESTABILIDAD OPERACIONAL
    pressure_drop_Pa: float        # ΔP actual
    fouling_rate: float            # dΔP/dt (Pa/h)
    mtbc_hours: float              # Mean time between cleaning

    def quality_score(self, target_wavelength: float = 460.0) -> float:
        """
        Score compuesto para CALIDAD (producto premium).

        Pesos según prioridades:
        - QY + estabilidad: 30%
        - Espectro (λ, FWHM, pureza): 30%
        - Selectividad: 20%
        - PDI: 15%
        - STY (bonus): 5%
        """
        score = 0.0

        # 1. QY + estabilidad (30%)
        qy_component = self.quantum_yield * (1 - min(self.qy_cv, 0.5))
        score += 0.30 * qy_component * 10  # Escalar a ~0-3

        # 2. Espectro (30%)
        # Penalizar desviación del target
        wavelength_match = np.exp(-0.001 * (self.emission_peak_nm - target_wavelength)**2)
        # FWHM óptimo: 25-40nm para CQDs de calidad
        fwhm_score = np.exp(-0.01 * (self.fwhm_nm - 30)**2) if self.fwhm_nm > 0 else 0
        spectral_component = wavelength_match * fwhm_score * self.spectral_purity
        score += 0.30 * spectral_component * 10

        # 3. Selectividad (20%)
        score += 0.20 * self.useful_yield * 10

        # 4. PDI (15%) - menor es mejor
        pdi_score = np.exp(-5 * self.pdi) if self.pdi > 0 else 1.0
        score += 0.15 * pdi_score * 10

        # 5. STY bonus (5%) - solo si no sacrifica calidad
        sty_normalized = min(self.sty_g_h_L / 10.0, 1.0)  # Normalizar a ~10 g/h·L
        score += 0.05 * sty_normalized * 10

        return score


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELOS FÍSICOS PARA KPIs
# ═══════════════════════════════════════════════════════════════════════════════

class CQDQualityModel:
    """
    Modelo físico-empírico para calcular KPIs de calidad.
    Basado en literatura y correlaciones establecidas.
    """

    # Constantes físicas
    E_BULK = 1.50        # eV - gap del grafeno dopado N
    A_CONF = 7.26        # eV·nm² - constante de confinamiento
    H_PLANCK = 4.136e-15 # eV·s
    C_LIGHT = 3e17       # nm/s

    def __init__(self):
        # Parámetros empíricos (calibrados con literatura)
        self.k_nucleation_ref = 1e12   # 1/s a 350K
        self.k_growth_ref = 1e11       # 1/s a 350K
        self.Ea_nucleation = 150.0     # kJ/mol
        self.Ea_growth = 50.0          # kJ/mol

    def calculate_kpis(self, design: Dict) -> QualityKPIs:
        """Calcula todos los KPIs a partir del diseño"""

        # Extraer parámetros
        T = design['temperature_C'] + 273.15  # K
        P = design['pressure_kPa'] * 1000     # Pa
        V = design['voltage_kv']              # kV
        f = design['frequency_khz']           # kHz
        duty = design['duty_cycle']

        conc = design['precursor_mM']         # mM
        pH = design['pH']

        w = design['channel_width_mm'] / 1000   # m
        h = design['channel_height_mm'] / 1000  # m
        L = design['channel_length_mm'] / 1000  # m

        Q_liq = design['liquid_flow_ml_min'] / 60 / 1e6  # m³/s
        Q_gas = design['gas_flow_ml_min'] / 60 / 1e6     # m³/s

        gap = design['electrode_gap_mm'] / 1000  # m

        # ─────────────────────────────────────────────────────────────────────
        # CÁLCULOS FÍSICOS
        # ─────────────────────────────────────────────────────────────────────

        # Volumen del reactor
        V_reactor = w * h * L * design['n_turns']  # m³
        V_reactor_L = V_reactor * 1000             # L

        # Tiempo de residencia
        tau = V_reactor / (Q_liq + Q_gas + 1e-12)  # s

        # Campo eléctrico
        E_field = V * 1e3 / gap  # V/m
        E_kV_cm = E_field / 1e5  # kV/cm

        # Potencia plasma (modelo Manley simplificado)
        C_diel = 8.85e-12 * 4 / (design.get('dielectric_thickness_mm', 1) / 1000)
        P_plasma = 4 * f * 1000 * C_diel * (V * 1000)**2 * duty  # W aproximado
        P_plasma = max(0.1, min(P_plasma, 1000))  # Limitar rango realista

        # ─────────────────────────────────────────────────────────────────────
        # 1. QUANTUM YIELD Y ESTABILIDAD
        # ─────────────────────────────────────────────────────────────────────

        # QY depende de: T (óptimo ~50°C), pH (óptimo 5-7), E (moderado mejor)
        T_opt = 323  # K (50°C)
        pH_opt = 6.0
        E_opt = 15   # kV/cm

        qy_T = np.exp(-0.002 * (T - T_opt)**2)
        qy_pH = np.exp(-0.1 * (pH - pH_opt)**2)
        qy_E = np.exp(-0.005 * (E_kV_cm - E_opt)**2)

        # QY base ~10% para buenas condiciones, escala con factores
        qy_base = 0.10
        quantum_yield = qy_base * qy_T * qy_pH * qy_E
        quantum_yield = np.clip(quantum_yield, 0.01, 0.25)  # 1-25%

        # CV del QY (estabilidad): mejor a T baja, pH neutro
        qy_cv = 0.05 + 0.1 * abs(T - T_opt)/50 + 0.05 * abs(pH - pH_opt)/3
        qy_cv = np.clip(qy_cv, 0.05, 0.5)

        qy_stability_score = quantum_yield * (1 - qy_cv)

        # ─────────────────────────────────────────────────────────────────────
        # 2. PROPIEDADES ESPECTRALES
        # ─────────────────────────────────────────────────────────────────────

        # Tamaño de partícula (modelo cinético)
        R = 8.314
        k_nuc = self.k_nucleation_ref * np.exp(-self.Ea_nucleation * 1000 / (R * T))
        k_grow = self.k_growth_ref * np.exp(-self.Ea_growth * 1000 / (R * T))

        # Tamaño ~ (k_grow/k_nuc)^0.33 * tau^0.5 * conc^0.2
        size_factor = (k_grow / (k_nuc + 1e-10))**0.33
        mean_size_nm = 1.5 + 0.5 * size_factor * (tau**0.3) * ((conc/50)**0.2)
        mean_size_nm = np.clip(mean_size_nm, 1.5, 5.0)

        # Dispersión: mejor control a T baja, flujo uniforme
        Re = (Q_liq / (w * h)) * 1000 * w / 0.001  # Reynolds aproximado
        pdi_base = 0.15
        pdi = pdi_base * (1 + 0.5 * (T - 300)/100) * (1 + 0.3 * min(Re/1000, 1))
        pdi = np.clip(pdi, 0.05, 0.5)

        size_std_nm = mean_size_nm * np.sqrt(pdi)

        # Emisión por confinamiento cuántico
        E_gap = self.E_BULK + self.A_CONF / (mean_size_nm**2)
        emission_peak_nm = 1240 / E_gap

        # FWHM: dispersión de tamaños → dispersión espectral
        # FWHM típico: 25-50nm para CQDs
        fwhm_nm = 25 + 50 * pdi
        fwhm_nm = np.clip(fwhm_nm, 20, 80)

        # Pureza espectral: penalizar condiciones extremas que dan mezclas
        spectral_purity = 0.9 * qy_T * qy_pH
        spectral_purity = np.clip(spectral_purity, 0.5, 0.95)

        # Stokes shift: ~50-80nm típico para CQDs
        stokes_shift_nm = 50 + 30 * (1 - quantum_yield/0.2)

        # ─────────────────────────────────────────────────────────────────────
        # 3. CONVERSIÓN Y SELECTIVIDAD
        # ─────────────────────────────────────────────────────────────────────

        # Conversión: depende de tau, T, potencia plasma
        tau_opt = 2.0  # s óptimo
        conversion_tau = 1 - np.exp(-tau / tau_opt)
        conversion_T = np.exp(-0.001 * (T - 340)**2)
        conversion_P = 1 - np.exp(-P_plasma / 10)

        precursor_conversion = 0.8 * conversion_tau * conversion_T * conversion_P
        precursor_conversion = np.clip(precursor_conversion, 0.1, 0.95)

        # Selectividad: T baja y E moderado → menos carbonilla
        selectivity_T = np.exp(-0.001 * (T - 320)**2)
        selectivity_E = np.exp(-0.002 * (E_kV_cm - 12)**2)

        selectivity = 0.7 * selectivity_T * selectivity_E
        selectivity = np.clip(selectivity, 0.3, 0.9)

        useful_yield = precursor_conversion * selectivity

        # ─────────────────────────────────────────────────────────────────────
        # 4. CORRELACIÓN TAMAÑO-EMISIÓN
        # ─────────────────────────────────────────────────────────────────────

        # Si PDI bajo → buena correlación (control por confinamiento)
        # Si PDI alto → mala correlación (defectos superficiales dominan)
        size_emission_correlation = 0.95 * np.exp(-5 * pdi)

        # ─────────────────────────────────────────────────────────────────────
        # 5. PRODUCTIVIDAD (STY)
        # ─────────────────────────────────────────────────────────────────────

        # Producción másica: conc * conversion * selectivity * Q_liq * MW
        MW_cqd = 1000  # g/mol efectivo para CQD típico
        production_mol_s = (conc / 1000) * useful_yield * Q_liq * 1000  # mol/s
        production_g_h = production_mol_s * MW_cqd * 3600 / 1000  # g/h
        production_g_h = max(0.001, production_g_h)

        # STY = producción / volumen reactor
        sty_g_h_L = production_g_h / max(V_reactor_L, 0.001)

        # Eficiencia energética
        energy_efficiency_g_kWh = production_g_h / (P_plasma / 1000 + 0.001)

        # ─────────────────────────────────────────────────────────────────────
        # 6. ESTABILIDAD OPERACIONAL
        # ─────────────────────────────────────────────────────────────────────

        # ΔP (Hagen-Poiseuille + bifásico)
        mu = 0.001  # Pa·s (agua)
        D_h = 2 * w * h / (w + h)  # Diámetro hidráulico

        # Gas holdup
        gas_holdup = Q_gas / (Q_liq + Q_gas + 1e-12)

        # Factor bifásico (Lockhart-Martinelli simplificado)
        phi_2 = 1 + 20 * gas_holdup + gas_holdup**2

        dp_single = 32 * mu * (Q_liq / (w * h)) * L * design['n_turns'] / D_h**2
        pressure_drop_Pa = dp_single * phi_2
        pressure_drop_Pa = max(100, min(pressure_drop_Pa, 1e7))

        # Fouling rate: mayor a T alta, pH extremo, alta conversión
        fouling_rate = 10 * (1 + (T - 300)/50) * (1 + abs(pH - 6)/3) * precursor_conversion
        fouling_rate = np.clip(fouling_rate, 1, 100)  # Pa/h

        # MTBC: cuando ΔP duplica → limpieza
        mtbc_hours = pressure_drop_Pa / (fouling_rate + 0.1)
        mtbc_hours = np.clip(mtbc_hours, 1, 1000)

        # ─────────────────────────────────────────────────────────────────────
        # CONSTRUIR RESULTADO
        # ─────────────────────────────────────────────────────────────────────

        return QualityKPIs(
            quantum_yield=quantum_yield,
            qy_cv=qy_cv,
            qy_stability_score=qy_stability_score,
            emission_peak_nm=emission_peak_nm,
            fwhm_nm=fwhm_nm,
            spectral_purity=spectral_purity,
            stokes_shift_nm=stokes_shift_nm,
            precursor_conversion=precursor_conversion,
            selectivity=selectivity,
            useful_yield=useful_yield,
            mean_size_nm=mean_size_nm,
            size_std_nm=size_std_nm,
            pdi=pdi,
            size_emission_correlation=size_emission_correlation,
            sty_g_h_L=sty_g_h_L,
            energy_efficiency_g_kWh=energy_efficiency_g_kWh,
            pressure_drop_Pa=pressure_drop_Pa,
            fouling_rate=fouling_rate,
            mtbc_hours=mtbc_hours,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZADOR BAYESIANO CON KPIs REALES
# ═══════════════════════════════════════════════════════════════════════════════

# Límites de diseño refinados
BOUNDS = {
    'channel_width_mm': (1.0, 5.0),
    'channel_height_mm': (0.5, 3.0),
    'channel_length_mm': (50.0, 300.0),
    'n_turns': (3, 10),
    'liquid_flow_ml_min': (10.0, 80.0),
    'gas_flow_ml_min': (5.0, 50.0),
    'ar_fraction': (0.2, 0.8),
    'n2_fraction': (0.1, 0.6),
    'voltage_kv': (5.0, 18.0),
    'frequency_khz': (5.0, 25.0),
    'duty_cycle': (0.2, 0.8),
    'pulse_width_us': (20.0, 150.0),
    'temperature_C': (25.0, 70.0),
    'pressure_kPa': (100.0, 140.0),
    'precursor_mM': (20.0, 120.0),
    'pH': (4.0, 8.0),
    'electrode_width_mm': (0.8, 2.5),
    'electrode_gap_mm': (0.8, 2.5),
    'electrode_coverage': (0.4, 0.85),
}


class QualityOptimizer:
    """Optimizador Bayesiano para KPIs de calidad"""

    def __init__(self, target_wavelength: float = 460.0):
        self.model = CQDQualityModel()
        self.target_wavelength = target_wavelength
        self.bounds = BOUNDS
        self.bounds_array = np.array([BOUNDS[k] for k in BOUNDS.keys()])
        self.keys = list(BOUNDS.keys())

        # GP surrogate
        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(0.1)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

        self.X_observed = []
        self.Y_observed = []
        self.kpis_history = []
        self.best_score = -np.inf
        self.best_design = None
        self.best_kpis = None

    def vector_to_design(self, x: np.ndarray) -> Dict:
        return {self.keys[i]: x[i] for i in range(len(self.keys))}

    def evaluate(self, x: np.ndarray) -> Tuple[float, QualityKPIs]:
        """Evalúa diseño y retorna score de calidad"""
        design = self.vector_to_design(x)
        kpis = self.model.calculate_kpis(design)
        score = kpis.quality_score(self.target_wavelength)
        return score, kpis

    def acquisition_ei(self, x: np.ndarray, xi: float = 0.01) -> float:
        """Expected Improvement"""
        x = x.reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)

        if sigma == 0:
            return 0.0

        imp = mu - self.best_score - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -ei[0]

    def suggest_next(self, n_restarts: int = 15) -> np.ndarray:
        """Sugiere siguiente punto"""
        best_x = None
        best_acq = float('inf')

        for _ in range(n_restarts):
            x0 = np.random.uniform(self.bounds_array[:, 0], self.bounds_array[:, 1])

            from scipy.optimize import minimize
            result = minimize(
                self.acquisition_ei,
                x0,
                bounds=self.bounds_array,
                method='L-BFGS-B'
            )

            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x

        return best_x

    def optimize(self, n_init: int = 40, n_iter: int = 80, verbose: bool = True) -> Dict:
        """Ejecuta optimización"""

        if verbose:
            print("\n" + "="*80)
            print("  OPTIMIZACIÓN PARA CALIDAD - KPIs REALES")
            print("="*80)
            print(f"\n  Target λ: {self.target_wavelength} nm")
            print(f"  Prioridades: QY+estabilidad > Espectro > Selectividad > PDI > STY")

        # Fase 1: Exploración LHS
        if verbose:
            print(f"\n→ Fase 1: Exploración ({n_init} puntos)")

        for i in range(n_init):
            x = np.zeros(len(self.keys))
            for j, (lo, hi) in enumerate(self.bounds_array):
                x[j] = lo + (hi - lo) * (i + np.random.random()) / n_init

            score, kpis = self.evaluate(x)

            self.X_observed.append(x)
            self.Y_observed.append(score)
            self.kpis_history.append(kpis)

            if score > self.best_score:
                self.best_score = score
                self.best_design = self.vector_to_design(x)
                self.best_kpis = kpis

            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_init}] Best: {self.best_score:.2f} | QY={self.best_kpis.quantum_yield*100:.1f}% | λ={self.best_kpis.emission_peak_nm:.0f}nm | PDI={self.best_kpis.pdi:.3f}")

        # Ajustar GP
        self.gp.fit(np.array(self.X_observed), np.array(self.Y_observed))

        # Fase 2: Optimización Bayesiana
        if verbose:
            print(f"\n→ Fase 2: Optimización Bayesiana ({n_iter} iteraciones)")

        for i in range(n_iter):
            x_next = self.suggest_next()
            score, kpis = self.evaluate(x_next)

            self.X_observed.append(x_next)
            self.Y_observed.append(score)
            self.kpis_history.append(kpis)

            improved = False
            if score > self.best_score:
                self.best_score = score
                self.best_design = self.vector_to_design(x_next)
                self.best_kpis = kpis
                improved = True

            if (i + 1) % 5 == 0:
                self.gp.fit(np.array(self.X_observed), np.array(self.Y_observed))

            if verbose and (improved or (i + 1) % 20 == 0):
                mark = "★" if improved else " "
                print(f"  [{i+1}/{n_iter}] {mark} Score: {score:.2f} | QY={kpis.quantum_yield*100:.1f}% | λ={kpis.emission_peak_nm:.0f}nm | FWHM={kpis.fwhm_nm:.0f}nm | PDI={kpis.pdi:.3f} | STY={kpis.sty_g_h_L:.2f}")

        return {
            'best_design': self.best_design,
            'best_kpis': self.best_kpis,
            'best_score': self.best_score,
            'n_evaluations': len(self.X_observed),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*80)
    print("  OPTIMIZACIÓN DE REACTOR DBD - ENFOQUE CALIDAD PREMIUM")
    print("="*80)

    # Optimizar para emisión azul (460nm)
    optimizer = QualityOptimizer(target_wavelength=460.0)
    result = optimizer.optimize(n_init=50, n_iter=100, verbose=True)

    # Mostrar resultado
    kpis = result['best_kpis']
    design = result['best_design']

    print("\n" + "="*80)
    print("  DISEÑO ÓPTIMO PARA CALIDAD")
    print("="*80)

    print(f"\n  SCORE TOTAL: {result['best_score']:.2f}")
    print(f"  Evaluaciones: {result['n_evaluations']}")

    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │ 1. QUANTUM YIELD Y ESTABILIDAD                                  │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │   QY: {kpis.quantum_yield*100:6.2f}%                                           │")
    print(f"  │   CV: {kpis.qy_cv*100:6.2f}%  (variabilidad)                             │")
    print(f"  │   Score estabilidad: {kpis.qy_stability_score:.4f}                           │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │ 2. PROPIEDADES ESPECTRALES                                      │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │   λ pico: {kpis.emission_peak_nm:6.1f} nm  (target: 460 nm)                   │")
    print(f"  │   FWHM: {kpis.fwhm_nm:6.1f} nm                                           │")
    print(f"  │   Pureza espectral: {kpis.spectral_purity*100:5.1f}%                              │")
    print(f"  │   Stokes shift: {kpis.stokes_shift_nm:5.1f} nm                                  │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │ 3. CONVERSIÓN Y SELECTIVIDAD                                    │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │   Conversión precursor: {kpis.precursor_conversion*100:5.1f}%                          │")
    print(f"  │   Selectividad: {kpis.selectivity*100:5.1f}%                                     │")
    print(f"  │   Rendimiento útil: {kpis.useful_yield*100:5.1f}%                                │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │ 4. TAMAÑO Y DISPERSIÓN                                          │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │   Tamaño medio: {kpis.mean_size_nm:5.2f} nm                                    │")
    print(f"  │   Std: {kpis.size_std_nm:5.2f} nm                                             │")
    print(f"  │   PDI: {kpis.pdi:6.4f}  (< 0.1 excelente, < 0.2 bueno)            │")
    print(f"  │   Correlación size-λ: {kpis.size_emission_correlation:.2f}                           │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │ 5. PRODUCTIVIDAD (sin sacrificar calidad)                       │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │   STY: {kpis.sty_g_h_L:6.2f} g/h·L                                        │")
    print(f"  │   Eficiencia: {kpis.energy_efficiency_g_kWh:8.1f} g/kWh                           │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │ 6. ESTABILIDAD OPERACIONAL                                      │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │   ΔP: {kpis.pressure_drop_Pa:8.0f} Pa                                       │")
    print(f"  │   Fouling rate: {kpis.fouling_rate:5.1f} Pa/h                                 │")
    print(f"  │   MTBC: {kpis.mtbc_hours:6.1f} horas                                       │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  ═══════════════════════════════════════════════════════════════════")
    print(f"  PARÁMETROS DE DISEÑO ÓPTIMO")
    print(f"  ═══════════════════════════════════════════════════════════════════")

    print(f"\n  GEOMETRÍA:")
    print(f"    Canal: {design['channel_width_mm']:.2f} × {design['channel_height_mm']:.2f} mm")
    print(f"    Longitud: {design['channel_length_mm']:.1f} mm")
    print(f"    Vueltas: {int(design['n_turns'])}")

    print(f"\n  ELECTRODOS:")
    print(f"    Ancho: {design['electrode_width_mm']:.2f} mm")
    print(f"    Gap: {design['electrode_gap_mm']:.2f} mm")
    print(f"    Cobertura: {design['electrode_coverage']*100:.0f}%")

    print(f"\n  FLUJO:")
    print(f"    Líquido: {design['liquid_flow_ml_min']:.1f} mL/min")
    print(f"    Gas: {design['gas_flow_ml_min']:.1f} mL/min")
    print(f"    Ar: {design['ar_fraction']*100:.0f}%, N₂: {design['n2_fraction']*100:.0f}%")

    print(f"\n  PLASMA:")
    print(f"    Voltaje: {design['voltage_kv']:.1f} kV")
    print(f"    Frecuencia: {design['frequency_khz']:.1f} kHz")
    print(f"    Duty: {design['duty_cycle']*100:.0f}%")
    print(f"    Pulso: {design['pulse_width_us']:.0f} μs")

    print(f"\n  OPERACIÓN:")
    print(f"    Temperatura: {design['temperature_C']:.1f}°C")
    print(f"    Presión: {design['pressure_kPa']:.1f} kPa")
    print(f"    Precursor: {design['precursor_mM']:.1f} mM")
    print(f"    pH: {design['pH']:.1f}")

    # Guardar resultado
    output = {
        'design': {k: float(v) for k, v in design.items()},
        'kpis': {
            'quantum_yield': kpis.quantum_yield,
            'qy_cv': kpis.qy_cv,
            'emission_peak_nm': kpis.emission_peak_nm,
            'fwhm_nm': kpis.fwhm_nm,
            'spectral_purity': kpis.spectral_purity,
            'precursor_conversion': kpis.precursor_conversion,
            'selectivity': kpis.selectivity,
            'useful_yield': kpis.useful_yield,
            'mean_size_nm': kpis.mean_size_nm,
            'pdi': kpis.pdi,
            'sty_g_h_L': kpis.sty_g_h_L,
            'energy_efficiency_g_kWh': kpis.energy_efficiency_g_kWh,
            'pressure_drop_Pa': kpis.pressure_drop_Pa,
            'mtbc_hours': kpis.mtbc_hours,
        },
        'score': result['best_score'],
        'target_wavelength': 460.0,
        'timestamp': datetime.now().isoformat(),
    }

    output_file = OUTPUT_DIR / f"quality_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✓ Resultado guardado: {output_file}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
