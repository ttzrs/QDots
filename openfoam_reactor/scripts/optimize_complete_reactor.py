#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  OPTIMIZADOR COMPLETO DE REACTOR DBD - VERSIÓN EXTENDIDA
  Incluye: flujo bifásico, turbulencia, transferencia de masa/calor, cinética
═══════════════════════════════════════════════════════════════════════════════

  VARIABLES DE OPTIMIZACIÓN:

  1. GEOMETRÍA DEL CANAL
     - channel_width, channel_height, channel_length, n_turns
     - mixer_type (none, baffles, serpentine_3d)
     - aspect_ratio_optimization

  2. ELECTRODOS
     - electrode_width, electrode_gap, electrode_coverage
     - electrode_material (Cu, Al, stainless)
     - dielectric_thickness, dielectric_material

  3. FLUJO BIFÁSICO (LÍQUIDO + GAS)
     - liquid_flow_rate (mL/min)
     - gas_flow_rate (mL/min)
     - gas_composition (Ar%, N2%, O2%)
     - bubble_injection (sparger_type, bubble_size_target)

  4. PLASMA DBD
     - voltage_kv, frequency_khz, duty_cycle, pulse_width_us
     - waveform (sinusoidal, pulsed, bipolar)

  5. CONDICIONES DE OPERACIÓN
     - temperature_C (liquid inlet)
     - pressure_kPa
     - precursor_concentration_mM
     - pH

  6. SISTEMA DE ENFRIAMIENTO
     - cooling_required, cooling_flow_rate
     - max_temperature_allowed

  7. MODELO DE TURBULENCIA
     - turbulence_model (laminar, k-epsilon, k-omega)
     - mixing_enhancement_factor
"""

import json
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import hashlib
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

CASE_DIR = Path(__file__).parent.parent
DATA_EXPORT_DIR = CASE_DIR / "data_export_complete"

# ═══════════════════════════════════════════════════════════════════════════════
#  ENUMERACIONES
# ═══════════════════════════════════════════════════════════════════════════════

class MixerType(Enum):
    NONE = "none"
    BAFFLES = "baffles"
    SERPENTINE_3D = "serpentine_3d"
    HERRINGBONE = "herringbone"

class ElectrodeMaterial(Enum):
    COPPER = "copper"
    ALUMINUM = "aluminum"
    STAINLESS = "stainless_steel"
    TITANIUM = "titanium"

class DielectricMaterial(Enum):
    GLASS = "glass"
    QUARTZ = "quartz"
    ALUMINA = "alumina"
    RESIN_HT = "high_temp_resin"
    PTFE = "ptfe"

class TurbulenceModel(Enum):
    LAMINAR = "laminar"
    K_EPSILON = "k-epsilon"
    K_OMEGA = "k-omega"
    K_OMEGA_SST = "k-omega-SST"

class WaveformType(Enum):
    SINUSOIDAL = "sinusoidal"
    PULSED_UNIPOLAR = "pulsed_unipolar"
    PULSED_BIPOLAR = "pulsed_bipolar"
    NANOSECOND = "nanosecond"

class SpargerType(Enum):
    NONE = "none"
    SINGLE_ORIFICE = "single_orifice"
    MULTI_ORIFICE = "multi_orifice"
    POROUS_PLATE = "porous_plate"
    MEMBRANE = "membrane"

# ═══════════════════════════════════════════════════════════════════════════════
#  LÍMITES DE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

DESIGN_BOUNDS = {
    # Geometría del canal
    "channel_width_mm": (1.0, 10.0),
    "channel_height_mm": (0.5, 5.0),
    "channel_length_mm": (50.0, 500.0),
    "n_turns": (1, 15),

    # Electrodos
    "electrode_width_mm": (0.5, 5.0),
    "electrode_gap_mm": (0.3, 3.0),
    "electrode_coverage": (0.2, 0.95),
    "dielectric_thickness_mm": (0.3, 2.0),

    # Flujo líquido
    "liquid_flow_rate_ml_min": (0.5, 50.0),

    # Flujo gas (inyección de burbujas)
    "gas_flow_rate_ml_min": (0.0, 100.0),
    "ar_fraction": (0.0, 1.0),
    "n2_fraction": (0.0, 1.0),
    # O2 = 1 - Ar - N2 (aire)
    "bubble_diameter_target_um": (50, 2000),

    # Plasma
    "voltage_kv": (3.0, 30.0),
    "frequency_khz": (0.5, 100.0),
    "duty_cycle": (0.05, 1.0),
    "pulse_width_us": (0.1, 1000.0),

    # Condiciones operación
    "temperature_inlet_C": (10.0, 60.0),
    "pressure_kPa": (90.0, 200.0),
    "precursor_conc_mM": (0.1, 100.0),
    "pH": (3.0, 11.0),

    # Enfriamiento
    "cooling_flow_rate_ml_min": (0.0, 200.0),
    "max_temp_allowed_C": (40.0, 80.0),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  PROPIEDADES DE MATERIALES
# ═══════════════════════════════════════════════════════════════════════════════

MATERIAL_PROPERTIES = {
    # Electrodos: conductividad (S/m), trabajo de salida (eV)
    "copper": {"conductivity": 5.96e7, "work_function": 4.65, "cost_factor": 1.0},
    "aluminum": {"conductivity": 3.77e7, "work_function": 4.28, "cost_factor": 0.5},
    "stainless_steel": {"conductivity": 1.45e6, "work_function": 4.4, "cost_factor": 0.8},
    "titanium": {"conductivity": 2.38e6, "work_function": 4.33, "cost_factor": 3.0},

    # Dieléctricos: permitividad relativa, rigidez dieléctrica (kV/mm)
    "glass": {"epsilon_r": 7.0, "breakdown": 10.0, "cost_factor": 0.5},
    "quartz": {"epsilon_r": 3.8, "breakdown": 25.0, "cost_factor": 2.0},
    "alumina": {"epsilon_r": 9.0, "breakdown": 15.0, "cost_factor": 1.5},
    "high_temp_resin": {"epsilon_r": 4.0, "breakdown": 20.0, "cost_factor": 1.0},
    "ptfe": {"epsilon_r": 2.1, "breakdown": 60.0, "cost_factor": 1.2},

    # Gases: masa molar, energía ionización (eV)
    "Ar": {"M": 39.95, "ionization_eV": 15.76, "metastable_eV": 11.55},
    "N2": {"M": 28.01, "ionization_eV": 15.58, "metastable_eV": 6.17},
    "O2": {"M": 32.00, "ionization_eV": 12.07, "metastable_eV": None},
    "air": {"M": 28.97, "ionization_eV": 14.0, "metastable_eV": None},
}

# ═══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompleteDesignPoint:
    """Punto de diseño completo con todas las variables"""
    # Geometría
    channel_width_mm: float
    channel_height_mm: float
    channel_length_mm: float
    n_turns: int
    mixer_type: str = "none"

    # Electrodos
    electrode_width_mm: float = 1.5
    electrode_gap_mm: float = 1.0
    electrode_coverage: float = 0.8
    electrode_material: str = "copper"
    dielectric_thickness_mm: float = 0.8
    dielectric_material: str = "high_temp_resin"

    # Flujo líquido
    liquid_flow_rate_ml_min: float = 5.0

    # Flujo gas
    gas_flow_rate_ml_min: float = 10.0
    ar_fraction: float = 0.0
    n2_fraction: float = 0.79
    sparger_type: str = "multi_orifice"
    bubble_diameter_target_um: float = 500.0

    # Plasma
    voltage_kv: float = 15.0
    frequency_khz: float = 20.0
    duty_cycle: float = 0.5
    pulse_width_us: float = 50.0
    waveform: str = "sinusoidal"

    # Operación
    temperature_inlet_C: float = 25.0
    pressure_kPa: float = 101.325
    precursor_conc_mM: float = 10.0
    precursor_type: str = "citric_acid"
    pH: float = 7.0

    # Enfriamiento
    cooling_flow_rate_ml_min: float = 50.0
    max_temp_allowed_C: float = 60.0

    # Turbulencia
    turbulence_model: str = "laminar"

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:8]


@dataclass
class FlowResults:
    """Resultados del modelo de flujo bifásico"""
    # Líquido
    liquid_velocity_m_s: float
    liquid_residence_time_s: float
    reynolds_liquid: float

    # Gas
    gas_holdup: float              # fracción volumétrica de gas
    bubble_frequency_hz: float     # frecuencia de burbujas
    bubble_velocity_m_s: float
    gas_residence_time_s: float

    # Interfacial
    interfacial_area_m2_m3: float  # área específica gas-líquido
    mass_transfer_coeff_m_s: float # k_L (coeficiente de transferencia)

    # Presión y mezcla
    pressure_drop_Pa: float
    mixing_efficiency: float       # 0-1
    flow_regime: str              # "bubble", "slug", "annular"


@dataclass
class ThermalResults:
    """Resultados del modelo térmico"""
    plasma_heat_generation_W: float
    liquid_temp_rise_C: float
    outlet_temperature_C: float
    cooling_required_W: float
    thermal_efficiency: float


@dataclass
class PlasmaChemResults:
    """Resultados del modelo de plasma y química"""
    # Plasma
    electric_field_kV_cm: float
    power_density_W_cm2: float
    total_power_W: float
    electron_density_cm3: float
    electron_temp_eV: float

    # Especies reactivas
    oh_concentration_M: float      # [OH] en líquido
    h2o2_concentration_M: float    # [H2O2] generado
    ozone_concentration_ppm: float # O3 en gas

    # Cinética
    reaction_rate_mol_L_s: float   # tasa de conversión de precursor
    conversion_fraction: float     # conversión del precursor


@dataclass
class CQDProductResults:
    """Resultados de producción de CQDs"""
    production_rate_mg_h: float
    yield_percent: float           # rendimiento respecto a precursor
    energy_efficiency_mg_kWh: float

    # Propiedades del producto
    mean_size_nm: float
    size_std_nm: float
    emission_wavelength_nm: float
    quantum_yield_percent: float

    # Calidad
    quality_score: float
    monodispersity: float          # 1 - (std/mean)


@dataclass
class CompleteSimulationRecord:
    """Registro completo de simulación"""
    design: CompleteDesignPoint
    flow: FlowResults
    thermal: ThermalResults
    plasma_chem: PlasmaChemResults
    cqd: CQDProductResults
    timestamp: str
    simulation_id: str
    objective_score: float

    def to_dict(self) -> dict:
        return {
            "design": self.design.to_dict(),
            "flow": asdict(self.flow),
            "thermal": asdict(self.thermal),
            "plasma_chem": asdict(self.plasma_chem),
            "cqd": asdict(self.cqd),
            "timestamp": self.timestamp,
            "simulation_id": self.simulation_id,
            "objective_score": self.objective_score,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELO DE FLUJO BIFÁSICO
# ═══════════════════════════════════════════════════════════════════════════════

class TwoPhaseFlowModel:
    """
    Modelo de flujo bifásico gas-líquido en microcanal.

    Basado en correlaciones para microcanales con inyección de burbujas.
    """

    def __init__(self):
        # Propiedades del agua a 25°C
        self.rho_L = 997.0      # kg/m³
        self.mu_L = 8.9e-4      # Pa·s
        self.sigma = 0.072      # N/m (tensión superficial)

        # Propiedades del gas (aire/Ar/N2)
        self.rho_G = 1.2        # kg/m³
        self.mu_G = 1.8e-5      # Pa·s

    def calculate(self, design: CompleteDesignPoint) -> FlowResults:
        """Calcula parámetros de flujo bifásico"""

        # Geometría
        W = design.channel_width_mm * 1e-3    # m
        H = design.channel_height_mm * 1e-3   # m
        L = design.channel_length_mm * 1e-3   # m
        A_cross = W * H                        # m²
        D_h = 2 * W * H / (W + H)             # diámetro hidráulico

        # Caudales volumétricos
        Q_L = design.liquid_flow_rate_ml_min / 60 * 1e-6  # m³/s
        Q_G = design.gas_flow_rate_ml_min / 60 * 1e-6     # m³/s

        # Velocidades superficiales
        j_L = Q_L / A_cross  # m/s
        j_G = Q_G / A_cross if Q_G > 0 else 0

        # Velocidad total y holdup (correlación de Armand)
        j_total = j_L + j_G
        if j_G > 0:
            beta = j_G / j_total  # fracción volumétrica de gas en entrada
            # Holdup (correlación para microcanales)
            gas_holdup = 0.833 * beta  # Armand correlation
            gas_holdup = min(0.9, max(0.01, gas_holdup))
        else:
            gas_holdup = 0.0
            beta = 0.0

        # Velocidades reales
        liquid_velocity = j_L / (1 - gas_holdup) if gas_holdup < 1 else j_L
        bubble_velocity = j_G / gas_holdup if gas_holdup > 0 else 0

        # Reynolds
        Re_L = self.rho_L * liquid_velocity * D_h / self.mu_L

        # Tiempos de residencia
        liquid_res_time = L / liquid_velocity if liquid_velocity > 0 else 1e6
        gas_res_time = L / bubble_velocity if bubble_velocity > 0 else 0

        # Régimen de flujo (basado en mapa de flujo para microcanales)
        if gas_holdup < 0.1:
            flow_regime = "liquid_only"
        elif gas_holdup < 0.3:
            flow_regime = "bubble"
        elif gas_holdup < 0.6:
            flow_regime = "slug"
        elif gas_holdup < 0.8:
            flow_regime = "churn"
        else:
            flow_regime = "annular"

        # Diámetro de burbuja (limitado por geometría)
        d_b = min(design.bubble_diameter_target_um * 1e-6, min(W, H) * 0.8)

        # Área interfacial específica
        if gas_holdup > 0 and d_b > 0:
            a_i = 6 * gas_holdup / d_b  # m²/m³
        else:
            a_i = 0

        # Frecuencia de burbujas
        if gas_holdup > 0 and d_b > 0:
            V_bubble = (4/3) * np.pi * (d_b/2)**3
            bubble_freq = Q_G / V_bubble
        else:
            bubble_freq = 0

        # Coeficiente de transferencia de masa (correlación de Higbie)
        D_O2 = 2.1e-9  # m²/s (difusividad O2 en agua)
        if bubble_velocity > 0:
            contact_time = d_b / bubble_velocity
            k_L = 2 * np.sqrt(D_O2 / (np.pi * contact_time))
        else:
            k_L = 1e-5  # valor base para difusión

        # Caída de presión (Lockhart-Martinelli para bifásico)
        f_L = 64 / Re_L if Re_L > 0 else 0.1  # factor de fricción laminar
        dP_L = f_L * (L / D_h) * 0.5 * self.rho_L * liquid_velocity**2

        if gas_holdup > 0:
            # Multiplicador bifásico
            X_tt = ((1-beta)/beta)**0.9 * (self.rho_G/self.rho_L)**0.5 * (self.mu_L/self.mu_G)**0.1
            phi_L2 = 1 + 20/X_tt + 1/X_tt**2
            dP_total = dP_L * phi_L2
        else:
            dP_total = dP_L

        # Eficiencia de mezcla (basada en turbulencia y burbujas)
        if design.mixer_type == "baffles":
            mix_factor = 1.3
        elif design.mixer_type == "serpentine_3d":
            mix_factor = 1.5
        elif design.mixer_type == "herringbone":
            mix_factor = 1.4
        else:
            mix_factor = 1.0

        # Mezcla por burbujas
        if gas_holdup > 0.05:
            bubble_mixing = 0.2 + 0.8 * min(1, gas_holdup / 0.3)
        else:
            bubble_mixing = 0.1

        mixing_efficiency = min(1.0, (0.5 + bubble_mixing) * mix_factor * min(1, Re_L/500))

        return FlowResults(
            liquid_velocity_m_s=liquid_velocity,
            liquid_residence_time_s=liquid_res_time,
            reynolds_liquid=Re_L,
            gas_holdup=gas_holdup,
            bubble_frequency_hz=bubble_freq,
            bubble_velocity_m_s=bubble_velocity,
            gas_residence_time_s=gas_res_time,
            interfacial_area_m2_m3=a_i,
            mass_transfer_coeff_m_s=k_L,
            pressure_drop_Pa=dP_total,
            mixing_efficiency=mixing_efficiency,
            flow_regime=flow_regime
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELO TÉRMICO
# ═══════════════════════════════════════════════════════════════════════════════

class ThermalModel:
    """Modelo de transferencia de calor en el reactor"""

    def __init__(self):
        self.cp_water = 4186  # J/(kg·K)
        self.rho_water = 997  # kg/m³

    def calculate(self, design: CompleteDesignPoint, plasma_power_W: float) -> ThermalResults:
        """Calcula balance térmico"""

        # Caudal másico
        m_dot = design.liquid_flow_rate_ml_min / 60 * 1e-6 * self.rho_water  # kg/s

        # Calor generado por plasma (parte va al líquido)
        plasma_to_liquid_fraction = 0.3  # ~30% del calor va al líquido
        Q_plasma = plasma_power_W * plasma_to_liquid_fraction

        # Subida de temperatura sin enfriamiento
        if m_dot > 0:
            dT_rise = Q_plasma / (m_dot * self.cp_water)
        else:
            dT_rise = 100  # valor alto si no hay flujo

        # Temperatura de salida
        T_out = design.temperature_inlet_C + dT_rise

        # Enfriamiento necesario
        if T_out > design.max_temp_allowed_C:
            Q_cooling = m_dot * self.cp_water * (T_out - design.max_temp_allowed_C)
            T_out = design.max_temp_allowed_C
        else:
            Q_cooling = 0

        # Eficiencia térmica (fracción de energía usada en reacción, no calor)
        thermal_efficiency = 1 - plasma_to_liquid_fraction

        return ThermalResults(
            plasma_heat_generation_W=Q_plasma,
            liquid_temp_rise_C=dT_rise,
            outlet_temperature_C=T_out,
            cooling_required_W=Q_cooling,
            thermal_efficiency=thermal_efficiency
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELO DE PLASMA Y QUÍMICA
# ═══════════════════════════════════════════════════════════════════════════════

class PlasmaChemistryModel:
    """
    Modelo de plasma DBD y química de especies reactivas.

    Incluye generación de OH, H2O2, O3 y cinética de formación de CQDs.
    """

    def __init__(self):
        self.breakdown_field = 3.0  # kV/mm en aire a 1 atm

    def calculate(self, design: CompleteDesignPoint, flow: FlowResults) -> PlasmaChemResults:
        """Calcula parámetros de plasma y química"""

        # Gap total (dieléctrico + canal)
        gap_mm = design.electrode_gap_mm + design.channel_height_mm
        gap_cm = gap_mm / 10

        # Propiedades del dieléctrico
        diel_props = MATERIAL_PROPERTIES.get(design.dielectric_material,
                                             {"epsilon_r": 4.0, "breakdown": 20.0})
        epsilon_r = diel_props["epsilon_r"]

        # Campo eléctrico (considerando dieléctrico)
        # V_gap = V_total * d_gas / (d_gas + d_diel/epsilon_r)
        d_gas = design.channel_height_mm
        d_diel = design.dielectric_thickness_mm
        V_gap = design.voltage_kv * d_gas / (d_gas + d_diel/epsilon_r)
        E_field = V_gap / d_gas * 10  # kV/cm

        # Área del electrodo
        electrode_length = design.channel_length_mm * design.electrode_coverage
        electrode_area_cm2 = (design.electrode_width_mm/10) * (electrode_length/10)

        # Potencia del plasma (modelo de Manley simplificado)
        C_d = 8.854e-12 * epsilon_r / (design.dielectric_thickness_mm * 1e-3)  # F/m²
        V_breakdown = self.breakdown_field * d_gas  # kV

        if design.voltage_kv > V_breakdown:
            # Potencia por ciclo
            P_cycle = 4 * C_d * electrode_area_cm2 * 1e-4 * \
                     design.voltage_kv * 1000 * (design.voltage_kv - V_breakdown) * 1000
            power_W = P_cycle * design.frequency_khz * 1000 * design.duty_cycle
            power_W = min(power_W, 500)  # limitar
        else:
            power_W = 0.1

        power_density = power_W / (electrode_area_cm2 + 1e-6)

        # Densidad de electrones (modelo empírico)
        if E_field > 10:
            n_e = 1e12 * (power_density / 1.0)**0.5 * (design.frequency_khz / 10)**0.3
        else:
            n_e = 1e10
        n_e = min(n_e, 1e15)

        # Temperatura de electrones (típico para DBD)
        T_e = 1 + 0.5 * np.log10(E_field + 1)  # eV, aproximación

        # ═══════════════════════════════════════════════════════════════
        # GENERACIÓN DE ESPECIES REACTIVAS
        # ═══════════════════════════════════════════════════════════════

        # Composición del gas
        O2_fraction = max(0, 1 - design.ar_fraction - design.n2_fraction)

        # Generación de OH (del agua en la interfaz gas-líquido)
        # Basado en energía depositada y área interfacial
        energy_density_J_cm3 = power_W * flow.liquid_residence_time_s / \
                              (design.channel_width_mm * design.channel_height_mm * \
                               design.channel_length_mm * 1e-3)

        # Yield de OH: ~1-10 moléculas/100 eV en plasma-líquido
        G_OH = 5  # moléculas/100 eV
        eV_per_J = 6.242e18
        OH_generated_mol = energy_density_J_cm3 * eV_per_J * G_OH / 100 / 6.022e23

        # Concentración de OH en estado estacionario (reacciona rápido)
        # Vida media ~1 μs, pero se regenera continuamente
        OH_conc_M = OH_generated_mol * 1000 * flow.mixing_efficiency

        # Generación de H2O2 (2 OH -> H2O2)
        H2O2_conc_M = OH_conc_M * 0.1 * flow.liquid_residence_time_s

        # Generación de O3 (si hay O2)
        if O2_fraction > 0.1:
            ozone_ppm = 100 * power_density * O2_fraction * design.duty_cycle
        else:
            ozone_ppm = 0

        # ═══════════════════════════════════════════════════════════════
        # CINÉTICA DE FORMACIÓN DE CQDs
        # ═══════════════════════════════════════════════════════════════

        # Modelo cinético simplificado: Precursor + OH -> Intermedios -> CQDs
        # r = k * [Precursor] * [OH]
        k_reaction = 1e3  # M^-1 s^-1 (constante de velocidad estimada)

        precursor_M = design.precursor_conc_mM / 1000
        reaction_rate = k_reaction * precursor_M * OH_conc_M * flow.mixing_efficiency

        # Conversión
        conversion = 1 - np.exp(-reaction_rate * flow.liquid_residence_time_s)
        conversion = min(0.99, max(0.01, conversion))

        return PlasmaChemResults(
            electric_field_kV_cm=E_field,
            power_density_W_cm2=power_density,
            total_power_W=power_W,
            electron_density_cm3=n_e,
            electron_temp_eV=T_e,
            oh_concentration_M=OH_conc_M,
            h2o2_concentration_M=H2O2_conc_M,
            ozone_concentration_ppm=ozone_ppm,
            reaction_rate_mol_L_s=reaction_rate,
            conversion_fraction=conversion
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELO DE PRODUCCIÓN DE CQDs
# ═══════════════════════════════════════════════════════════════════════════════

class CQDProductionModel:
    """Modelo de producción y calidad de CQDs"""

    def __init__(self):
        # Constantes del modelo de tamaño (confinamiento cuántico)
        self.E_bulk = 1.50   # eV
        self.A_conf = 7.26   # eV·nm²

        # Parámetros de calidad
        self.optimal_size = 2.5  # nm
        self.optimal_wavelength = 460  # nm (azul)

    def calculate(self, design: CompleteDesignPoint,
                 flow: FlowResults,
                 thermal: ThermalResults,
                 plasma: PlasmaChemResults) -> CQDProductResults:
        """Calcula producción y propiedades de CQDs"""

        # Producción basada en conversión y caudal
        precursor_flow_mol_h = (design.liquid_flow_rate_ml_min * 60 / 1000) * \
                               (design.precursor_conc_mM / 1000)  # mol/h

        # Masa molar aproximada de CQD (asumiendo ~100 átomos de C)
        M_CQD = 1200  # g/mol

        # Yield másico (considerando pérdidas)
        yield_factor = plasma.conversion_fraction * flow.mixing_efficiency * \
                      (1 - thermal.liquid_temp_rise_C / 100)  # penaliza alta T
        yield_factor = max(0.1, min(0.9, yield_factor))

        production_mg_h = precursor_flow_mol_h * M_CQD * 1000 * yield_factor * 0.1
        production_mg_h = max(0.1, production_mg_h)

        yield_percent = yield_factor * 100

        # Eficiencia energética
        if plasma.total_power_W > 0:
            efficiency = production_mg_h / (plasma.total_power_W / 1000)  # mg/kWh
        else:
            efficiency = 0

        # ═══════════════════════════════════════════════════════════════
        # TAMAÑO DE PARTÍCULA
        # ═══════════════════════════════════════════════════════════════

        # El tamaño depende de:
        # - Tiempo de residencia (más tiempo = más grande)
        # - Potencia del plasma (más potencia = más pequeño, fragmentación)
        # - Temperatura (más calor = más grande, coalescencia)
        # - Concentración de precursor (más concentrado = más grande)

        size_base = 2.0  # nm

        tau_factor = (flow.liquid_residence_time_s / 10) ** 0.2
        power_factor = (1.0 / (plasma.power_density_W_cm2 + 0.1)) ** 0.15
        temp_factor = (thermal.outlet_temperature_C / 25) ** 0.1
        conc_factor = (design.precursor_conc_mM / 10) ** 0.1

        mean_size = size_base * tau_factor * power_factor * temp_factor * conc_factor
        mean_size = max(1.5, min(6.0, mean_size))

        # Dispersión de tamaño (depende de uniformidad del proceso)
        size_std = mean_size * (1 - flow.mixing_efficiency) * 0.3
        size_std = max(0.1, min(mean_size * 0.5, size_std))

        # Monodispersidad
        monodispersity = 1 - size_std / mean_size

        # ═══════════════════════════════════════════════════════════════
        # PROPIEDADES ÓPTICAS
        # ═══════════════════════════════════════════════════════════════

        # Gap óptico (modelo de confinamiento cuántico)
        E_gap = self.E_bulk + self.A_conf / (mean_size ** 2)
        wavelength = 1240 / E_gap  # nm

        # Quantum yield (depende de calidad de superficie)
        # Mejor QY con N-doping (pH alto) y baja temperatura
        qy_base = 10  # % base
        ph_factor = 1 + 0.5 * (design.pH - 7) / 7  # mejor a pH alto
        temp_penalty = max(0, (thermal.outlet_temperature_C - 40) / 40)
        qy = qy_base * ph_factor * (1 - temp_penalty) * monodispersity
        qy = max(1, min(80, qy))

        # ═══════════════════════════════════════════════════════════════
        # SCORE DE CALIDAD
        # ═══════════════════════════════════════════════════════════════

        # Penalizaciones por desviación del óptimo
        size_score = 1 - abs(mean_size - self.optimal_size) / self.optimal_size
        wavelength_score = 1 - abs(wavelength - self.optimal_wavelength) / 100
        qy_score = qy / 50

        quality_score = (size_score + wavelength_score + qy_score + monodispersity) / 4
        quality_score = max(0, min(1, quality_score))

        return CQDProductResults(
            production_rate_mg_h=production_mg_h,
            yield_percent=yield_percent,
            energy_efficiency_mg_kWh=efficiency,
            mean_size_nm=mean_size,
            size_std_nm=size_std,
            emission_wavelength_nm=wavelength,
            quantum_yield_percent=qy,
            quality_score=quality_score,
            monodispersity=monodispersity
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN OBJETIVO MULTI-CRITERIO
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_objective(design: CompleteDesignPoint,
                       flow: FlowResults,
                       thermal: ThermalResults,
                       plasma: PlasmaChemResults,
                       cqd: CQDProductResults) -> float:
    """
    Función objetivo multi-criterio a MAXIMIZAR.

    Prioridades:
    1. Producción de CQDs (mg/h) - peso alto
    2. Calidad del producto (tamaño, λ, QY) - peso alto
    3. Eficiencia energética (mg/kWh) - peso medio
    4. Eficiencia de recursos (conversión) - peso medio
    5. Penalizaciones: alta presión, alta temperatura, exceso de potencia
    """

    # Pesos
    w = {
        "production": 2.0,
        "quality": 3.0,
        "efficiency": 1.0,
        "conversion": 1.5,
        "pressure_penalty": -0.001,
        "temp_penalty": -0.02,
        "power_penalty": -0.01,
    }

    # Normalizaciones
    production_norm = cqd.production_rate_mg_h / 10.0  # normalizar a 10 mg/h
    quality_norm = cqd.quality_score
    efficiency_norm = cqd.energy_efficiency_mg_kWh / 500  # normalizar a 500 mg/kWh
    conversion_norm = plasma.conversion_fraction

    # Penalizaciones
    pressure_penalty = flow.pressure_drop_Pa
    temp_penalty = max(0, thermal.outlet_temperature_C - design.max_temp_allowed_C)
    power_penalty = max(0, plasma.total_power_W - 50)

    # Score total
    score = (
        w["production"] * production_norm +
        w["quality"] * quality_norm +
        w["efficiency"] * efficiency_norm +
        w["conversion"] * conversion_norm +
        w["pressure_penalty"] * pressure_penalty +
        w["temp_penalty"] * temp_penalty +
        w["power_penalty"] * power_penalty
    )

    return score


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE DISEÑOS
# ═══════════════════════════════════════════════════════════════════════════════

def random_complete_design() -> CompleteDesignPoint:
    """Genera diseño aleatorio completo"""
    ar = np.random.uniform(0, 0.5)
    n2 = np.random.uniform(0, 1 - ar)

    return CompleteDesignPoint(
        channel_width_mm=np.random.uniform(*DESIGN_BOUNDS["channel_width_mm"]),
        channel_height_mm=np.random.uniform(*DESIGN_BOUNDS["channel_height_mm"]),
        channel_length_mm=np.random.uniform(*DESIGN_BOUNDS["channel_length_mm"]),
        n_turns=np.random.randint(*DESIGN_BOUNDS["n_turns"]),
        mixer_type=np.random.choice(["none", "baffles", "serpentine_3d"]),
        electrode_width_mm=np.random.uniform(*DESIGN_BOUNDS["electrode_width_mm"]),
        electrode_gap_mm=np.random.uniform(*DESIGN_BOUNDS["electrode_gap_mm"]),
        electrode_coverage=np.random.uniform(*DESIGN_BOUNDS["electrode_coverage"]),
        electrode_material=np.random.choice(["copper", "aluminum", "stainless_steel"]),
        dielectric_thickness_mm=np.random.uniform(*DESIGN_BOUNDS["dielectric_thickness_mm"]),
        dielectric_material=np.random.choice(["glass", "quartz", "high_temp_resin"]),
        liquid_flow_rate_ml_min=np.random.uniform(*DESIGN_BOUNDS["liquid_flow_rate_ml_min"]),
        gas_flow_rate_ml_min=np.random.uniform(*DESIGN_BOUNDS["gas_flow_rate_ml_min"]),
        ar_fraction=ar,
        n2_fraction=n2,
        sparger_type=np.random.choice(["single_orifice", "multi_orifice", "porous_plate"]),
        bubble_diameter_target_um=np.random.uniform(*DESIGN_BOUNDS["bubble_diameter_target_um"]),
        voltage_kv=np.random.uniform(*DESIGN_BOUNDS["voltage_kv"]),
        frequency_khz=np.random.uniform(*DESIGN_BOUNDS["frequency_khz"]),
        duty_cycle=np.random.uniform(*DESIGN_BOUNDS["duty_cycle"]),
        pulse_width_us=np.random.uniform(*DESIGN_BOUNDS["pulse_width_us"]),
        waveform=np.random.choice(["sinusoidal", "pulsed_unipolar", "pulsed_bipolar"]),
        temperature_inlet_C=np.random.uniform(*DESIGN_BOUNDS["temperature_inlet_C"]),
        pressure_kPa=np.random.uniform(*DESIGN_BOUNDS["pressure_kPa"]),
        precursor_conc_mM=np.random.uniform(*DESIGN_BOUNDS["precursor_conc_mM"]),
        precursor_type=np.random.choice(["citric_acid", "glucose", "sucrose", "peg"]),
        pH=np.random.uniform(*DESIGN_BOUNDS["pH"]),
        cooling_flow_rate_ml_min=np.random.uniform(*DESIGN_BOUNDS["cooling_flow_rate_ml_min"]),
        max_temp_allowed_C=np.random.uniform(*DESIGN_BOUNDS["max_temp_allowed_C"]),
        turbulence_model=np.random.choice(["laminar", "k-epsilon"]),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def save_record(record: CompleteSimulationRecord, export_dir: Path):
    """Guarda registro"""
    export_dir.mkdir(parents=True, exist_ok=True)

    record_file = export_dir / f"sim_{record.simulation_id}.json"
    with open(record_file, 'w') as f:
        json.dump(record.to_dict(), f, indent=2)

    master_file = export_dir / "all_simulations.jsonl"
    with open(master_file, 'a') as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def export_for_pinn(export_dir: Path):
    """Exporta para PINN training"""
    master_file = export_dir / "all_simulations.jsonl"
    if not master_file.exists():
        return

    records = []
    with open(master_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Inputs: todas las variables de diseño numéricas
    X = []
    Y = []

    for r in records:
        d = r['design']
        fl = r['flow']
        th = r['thermal']
        pl = r['plasma_chem']
        cq = r['cqd']

        X.append([
            d['channel_width_mm'], d['channel_height_mm'], d['channel_length_mm'], d['n_turns'],
            d['electrode_width_mm'], d['electrode_gap_mm'], d['electrode_coverage'],
            d['dielectric_thickness_mm'],
            d['liquid_flow_rate_ml_min'], d['gas_flow_rate_ml_min'],
            d['ar_fraction'], d['n2_fraction'], d['bubble_diameter_target_um'],
            d['voltage_kv'], d['frequency_khz'], d['duty_cycle'], d['pulse_width_us'],
            d['temperature_inlet_C'], d['pressure_kPa'], d['precursor_conc_mM'], d['pH'],
        ])

        Y.append([
            fl['liquid_residence_time_s'], fl['gas_holdup'], fl['interfacial_area_m2_m3'],
            fl['mixing_efficiency'], fl['pressure_drop_Pa'],
            th['outlet_temperature_C'], th['cooling_required_W'],
            pl['total_power_W'], pl['electric_field_kV_cm'], pl['oh_concentration_M'],
            pl['conversion_fraction'],
            cq['production_rate_mg_h'], cq['mean_size_nm'], cq['emission_wavelength_nm'],
            cq['quantum_yield_percent'], cq['quality_score'],
        ])

    X = np.array(X)
    Y = np.array(Y)

    np.save(export_dir / "pinn_inputs.npy", X)
    np.save(export_dir / "pinn_outputs.npy", Y)

    metadata = {
        "n_samples": len(records),
        "input_features": [
            "channel_width_mm", "channel_height_mm", "channel_length_mm", "n_turns",
            "electrode_width_mm", "electrode_gap_mm", "electrode_coverage",
            "dielectric_thickness_mm",
            "liquid_flow_rate_ml_min", "gas_flow_rate_ml_min",
            "ar_fraction", "n2_fraction", "bubble_diameter_target_um",
            "voltage_kv", "frequency_khz", "duty_cycle", "pulse_width_us",
            "temperature_inlet_C", "pressure_kPa", "precursor_conc_mM", "pH",
        ],
        "output_features": [
            "liquid_residence_time_s", "gas_holdup", "interfacial_area_m2_m3",
            "mixing_efficiency", "pressure_drop_Pa",
            "outlet_temperature_C", "cooling_required_W",
            "total_power_W", "electric_field_kV_cm", "oh_concentration_M",
            "conversion_fraction",
            "production_rate_mg_h", "mean_size_nm", "emission_wavelength_nm",
            "quantum_yield_percent", "quality_score",
        ],
        "design_bounds": DESIGN_BOUNDS,
        "export_date": datetime.now().isoformat(),
    }

    with open(export_dir / "pinn_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✓ Datos PINN exportados:")
    print(f"    - pinn_inputs.npy: {X.shape}")
    print(f"    - pinn_outputs.npy: {Y.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def run_complete_optimization(n_iterations: int = 100):
    """Ejecuta optimización completa"""
    print("═" * 80)
    print("  OPTIMIZACIÓN COMPLETA DE REACTOR DBD - MODELO EXTENDIDO")
    print("  Flujo bifásico + Turbulencia + Transferencia + Cinética")
    print("═" * 80)

    # Modelos
    flow_model = TwoPhaseFlowModel()
    thermal_model = ThermalModel()
    plasma_model = PlasmaChemistryModel()
    cqd_model = CQDProductionModel()

    # Generar diseños
    print(f"\n→ Generando {n_iterations} diseños aleatorios...")
    designs = [random_complete_design() for _ in range(n_iterations)]

    results_list = []
    best_score = float('-inf')
    best_record = None

    for i, design in enumerate(designs):
        sim_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{design.hash()}"

        # Calcular todos los modelos
        flow = flow_model.calculate(design)
        plasma = plasma_model.calculate(design, flow)
        thermal = thermal_model.calculate(design, plasma.total_power_W)
        cqd = cqd_model.calculate(design, flow, thermal, plasma)

        # Calcular objetivo
        score = calculate_objective(design, flow, thermal, plasma, cqd)

        # Mostrar progreso cada 10 iteraciones
        if (i + 1) % 10 == 0 or score > best_score:
            print(f"\n[{i+1}/{n_iterations}] Score: {score:.3f}")
            print(f"  Producción: {cqd.production_rate_mg_h:.2f} mg/h, "
                  f"λ={cqd.emission_wavelength_nm:.0f}nm, QY={cqd.quantum_yield_percent:.1f}%")

        if score > best_score:
            best_score = score
            print(f"  ★ Nuevo mejor!")

        # Guardar registro
        record = CompleteSimulationRecord(
            design=design,
            flow=flow,
            thermal=thermal,
            plasma_chem=plasma,
            cqd=cqd,
            timestamp=datetime.now().isoformat(),
            simulation_id=sim_id,
            objective_score=score
        )

        if score > best_score - 0.001:
            best_record = record

        save_record(record, DATA_EXPORT_DIR)
        results_list.append(record)

    # Resumen
    print("\n" + "═" * 80)
    print("  RESUMEN DE OPTIMIZACIÓN COMPLETA")
    print("═" * 80)

    if best_record:
        d = best_record.design
        cqd = best_record.cqd
        pl = best_record.plasma_chem
        fl = best_record.flow

        print(f"\n  MEJOR DISEÑO (Score: {best_record.objective_score:.3f}):")
        print(f"\n  GEOMETRÍA:")
        print(f"    Canal: {d.channel_width_mm:.1f} × {d.channel_height_mm:.1f} mm")
        print(f"    Longitud: {d.channel_length_mm:.0f} mm, {d.n_turns} vueltas")
        print(f"    Mezclador: {d.mixer_type}")

        print(f"\n  ELECTRODOS:")
        print(f"    Ancho: {d.electrode_width_mm:.1f} mm, Gap: {d.electrode_gap_mm:.1f} mm")
        print(f"    Cobertura: {d.electrode_coverage*100:.0f}%")
        print(f"    Material: {d.electrode_material}, Dieléctrico: {d.dielectric_material}")

        print(f"\n  FLUJO:")
        print(f"    Líquido: {d.liquid_flow_rate_ml_min:.1f} mL/min")
        print(f"    Gas: {d.gas_flow_rate_ml_min:.1f} mL/min (Ar:{d.ar_fraction*100:.0f}%, N2:{d.n2_fraction*100:.0f}%)")
        print(f"    Gas holdup: {fl.gas_holdup*100:.1f}%, Régimen: {fl.flow_regime}")
        print(f"    τ_líquido: {fl.liquid_residence_time_s:.1f} s")

        print(f"\n  PLASMA:")
        print(f"    Voltaje: {d.voltage_kv:.1f} kV, Frecuencia: {d.frequency_khz:.1f} kHz")
        print(f"    Duty cycle: {d.duty_cycle*100:.0f}%, Pulso: {d.pulse_width_us:.0f} μs")
        print(f"    Potencia: {pl.total_power_W:.1f} W, E={pl.electric_field_kV_cm:.1f} kV/cm")

        print(f"\n  OPERACIÓN:")
        print(f"    T_inlet: {d.temperature_inlet_C:.0f}°C, P: {d.pressure_kPa:.0f} kPa")
        print(f"    Precursor: {d.precursor_conc_mM:.1f} mM {d.precursor_type}, pH={d.pH:.1f}")

        print(f"\n  RESULTADOS CQDs:")
        print(f"    Producción: {cqd.production_rate_mg_h:.2f} mg/h")
        print(f"    Eficiencia: {cqd.energy_efficiency_mg_kWh:.0f} mg/kWh")
        print(f"    Tamaño: {cqd.mean_size_nm:.2f} ± {cqd.size_std_nm:.2f} nm")
        print(f"    λ emisión: {cqd.emission_wavelength_nm:.0f} nm")
        print(f"    Quantum Yield: {cqd.quantum_yield_percent:.1f}%")
        print(f"    Calidad: {cqd.quality_score*100:.0f}%")

        # Guardar mejor diseño
        with open(DATA_EXPORT_DIR / "best_design.json", 'w') as f:
            json.dump(best_record.to_dict(), f, indent=2)

    # Exportar
    print("\n→ Exportando datos para PINN...")
    export_for_pinn(DATA_EXPORT_DIR)

    print("\n" + "═" * 80)
    print("  ✓ Optimización completada")
    print("═" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--iterations", type=int, default=100)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    os.chdir(CASE_DIR)

    if args.export_only:
        export_for_pinn(DATA_EXPORT_DIR)
    else:
        run_complete_optimization(args.iterations)
