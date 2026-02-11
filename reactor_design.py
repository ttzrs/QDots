#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  DISEÑO PARAMÉTRICO DEL REACTOR DBD PARA SÍNTESIS DE CQDs
  Optimización basada en simulación cuántica y física de plasmas
═══════════════════════════════════════════════════════════════════════════════

  Este módulo calcula los parámetros óptimos de construcción para maximizar
  la producción de Carbon Quantum Dots con emisión azul (450 nm).

  Variables de optimización:
    - Geometría de la cámara (forma, dimensiones)
    - Posición y configuración de electrodos
    - Materiales (dieléctrico, electrodos, sellado)
    - Caudal y tiempo de residencia
    - Área de interfaz aire/líquido
    - Parámetros eléctricos (voltaje, frecuencia)

  Basado en:
    - Física DBD (Dielectric Barrier Discharge)
    - Modelo de confinamiento cuántico validado por VQE
    - Literatura de síntesis de CQDs por plasma

  USO:
    python reactor_design.py --optimize
    python reactor_design.py --production-target 100  # mg/hora
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES FÍSICAS Y DE SIMULACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

# Constantes fundamentales
EPSILON_0 = 8.854e-12      # Permitividad del vacío (F/m)
E_ELECTRON = 1.602e-19     # Carga del electrón (C)
K_BOLTZMANN = 1.381e-23    # Constante de Boltzmann (J/K)

# Del modelo VQE y literatura
TARGET_WAVELENGTH_NM = 450          # Longitud de onda objetivo
TARGET_SIZE_NM = 2.4                # Tamaño de partícula óptimo
E_GAP_TARGET_EV = 2.77              # Gap óptico objetivo

# Parámetros de plasma DBD (del documento proyecto.txt)
BREAKDOWN_FIELD_AIR = 3e6           # Campo de ruptura del aire (V/m)
ELECTRON_TEMPERATURE_EV = 1.5       # Temperatura electrónica típica DBD
ION_DENSITY_CM3 = 1e10              # Densidad de iones típica


class ChamberGeometry(Enum):
    """Geometrías de cámara disponibles"""
    RECTANGULAR = "rectangular"      # Canal rectangular simple
    SERPENTINE = "serpentine"        # Serpentín para mayor residencia
    COAXIAL = "coaxial"              # Cilíndrico coaxial
    PARALLEL_PLATE = "parallel_plate" # Placas paralelas
    MESH = "mesh"                    # Malla 3D para máximo contacto


class ElectrodeConfig(Enum):
    """Configuraciones de electrodos"""
    EMBEDDED_WALLS = "embedded_walls"     # Embebidos en paredes
    MESH_SUBMERGED = "mesh_submerged"     # Malla sumergida
    COPLANAR = "coplanar"                 # Coplanares en superficie
    RING = "ring"                         # Anillos concéntricos
    INTERDIGITATED = "interdigitated"    # Interdigitados


@dataclass
class Material:
    """Propiedades de materiales"""
    name: str
    dielectric_constant: float      # Constante dieléctrica relativa
    dielectric_strength: float      # Rigidez dieléctrica (V/m)
    thermal_conductivity: float     # Conductividad térmica (W/m·K)
    max_temperature: float          # Temperatura máxima (°C)
    chemical_resistance: bool       # Resistencia química
    printable_3d: bool              # Imprimible en 3D
    cost_relative: float            # Costo relativo (1 = bajo)


# Base de datos de materiales
MATERIALS = {
    "alumina_ceramic": Material(
        name="Alúmina (Al₂O₃)",
        dielectric_constant=9.8,
        dielectric_strength=15e6,
        thermal_conductivity=30,
        max_temperature=1700,
        chemical_resistance=True,
        printable_3d=True,  # Con impresora cerámica
        cost_relative=3.0
    ),
    "borosilicate_glass": Material(
        name="Vidrio Borosilicato",
        dielectric_constant=4.6,
        dielectric_strength=10e6,
        thermal_conductivity=1.2,
        max_temperature=500,
        chemical_resistance=True,
        printable_3d=False,
        cost_relative=1.5
    ),
    "quartz": Material(
        name="Cuarzo (SiO₂)",
        dielectric_constant=3.8,
        dielectric_strength=25e6,
        thermal_conductivity=1.4,
        max_temperature=1200,
        chemical_resistance=True,
        printable_3d=False,
        cost_relative=4.0
    ),
    "ptfe": Material(
        name="PTFE (Teflón)",
        dielectric_constant=2.1,
        dielectric_strength=60e6,
        thermal_conductivity=0.25,
        max_temperature=260,
        chemical_resistance=True,
        printable_3d=False,
        cost_relative=2.0
    ),
    "peek": Material(
        name="PEEK",
        dielectric_constant=3.3,
        dielectric_strength=20e6,
        thermal_conductivity=0.25,
        max_temperature=250,
        chemical_resistance=True,
        printable_3d=True,
        cost_relative=5.0
    ),
    "resin_high_temp": Material(
        name="Resina Alta Temperatura",
        dielectric_constant=3.5,
        dielectric_strength=15e6,
        thermal_conductivity=0.2,
        max_temperature=200,
        chemical_resistance=False,
        printable_3d=True,  # SLA/DLP
        cost_relative=1.0
    ),
    "ceramic_resin": Material(
        name="Resina Cerámica",
        dielectric_constant=6.0,
        dielectric_strength=12e6,
        thermal_conductivity=1.0,
        max_temperature=300,
        chemical_resistance=True,
        printable_3d=True,
        cost_relative=2.5
    ),
    "tio2_anatase_porous": Material(
        name="TiO₂ Anatase Poroso",
        dielectric_constant=40.0,      # Anatase: 31-48 (poroso reduce)
        dielectric_strength=4e6,       # Menor por porosidad
        thermal_conductivity=4.0,      # Reducido por porosidad (~8.5 denso)
        max_temperature=1200,
        chemical_resistance=True,
        printable_3d=True,             # Extrusión cerámica / sol-gel + molde
        cost_relative=4.0
    ),
    "tio2_rutile_porous": Material(
        name="TiO₂ Rutilo Poroso",
        dielectric_constant=85.0,      # Rutilo: 86-170 (poroso reduce)
        dielectric_strength=5e6,
        thermal_conductivity=5.0,
        max_temperature=1800,
        chemical_resistance=True,
        printable_3d=True,
        cost_relative=4.5
    ),
}


@dataclass
class ReactorParameters:
    """Parámetros completos del reactor"""
    # Geometría de la cámara
    geometry: ChamberGeometry = ChamberGeometry.SERPENTINE
    channel_width_mm: float = 2.0       # Ancho del canal
    channel_height_mm: float = 0.5      # Altura del canal (gap líquido)
    channel_length_mm: float = 150.0    # Longitud total del canal
    n_serpentine_turns: int = 8         # Número de vueltas si es serpentín

    # Electrodos
    electrode_config: ElectrodeConfig = ElectrodeConfig.EMBEDDED_WALLS
    electrode_width_mm: float = 1.5     # Ancho del electrodo
    electrode_gap_mm: float = 1.0       # Separación entre electrodos
    electrode_thickness_mm: float = 0.1 # Espesor del electrodo
    dielectric_thickness_mm: float = 0.8 # Espesor de barrera dieléctrica

    # Materiales
    dielectric_material: str = "ceramic_resin"
    electrode_material: str = "copper"  # Cu, Al, acero inoxidable

    # Catalizador
    catalyst_type: Optional[str] = None          # None, "tio2_anatase", "tio2_rutile"
    catalyst_porosity: float = 0.60              # Porosidad (0-1), típico 0.4-0.7
    catalyst_surface_area_m2_g: float = 50.0     # Área BET (m²/g), 50-200 típico
    catalyst_thickness_mm: float = 0.5           # Espesor de capa catalítica
    catalyst_loading_mg_cm2: float = 2.0         # Carga de TiO2 por cm²

    # Parámetros eléctricos
    voltage_kv: float = 10.0            # Voltaje aplicado
    frequency_khz: float = 20.0         # Frecuencia
    duty_cycle: float = 0.5             # Ciclo de trabajo
    rise_time_ns: float = 50.0          # Tiempo de subida

    # Flujo y proceso
    liquid_flow_ml_min: float = 5.0     # Caudal de líquido
    gas_flow_ml_min: float = 50.0       # Caudal de gas (aire/N₂)
    liquid_depth_mm: float = 0.3        # Profundidad del líquido
    temperature_c: float = 25.0         # Temperatura de operación

    # Producción objetivo
    target_production_mg_h: float = 50.0  # Producción objetivo


@dataclass
class DesignOutput:
    """Salida del diseño optimizado"""
    # Dimensiones calculadas
    total_volume_ml: float = 0.0
    plasma_area_cm2: float = 0.0
    liquid_volume_ml: float = 0.0
    gas_volume_ml: float = 0.0

    # Tiempos
    residence_time_s: float = 0.0
    plasma_exposure_time_s: float = 0.0

    # Eléctricos
    power_w: float = 0.0
    energy_density_j_ml: float = 0.0
    electric_field_v_m: float = 0.0

    # Producción estimada
    estimated_production_mg_h: float = 0.0
    yield_percent: float = 0.0
    specific_energy_j_mg: float = 0.0

    # Calidad
    estimated_wavelength_nm: float = 0.0
    estimated_size_nm: float = 0.0
    monodispersity_index: float = 0.0

    # Térmica
    heat_generation_w: float = 0.0
    cooling_required: bool = False

    # Métricas de diseño
    efficiency_score: float = 0.0
    cost_score: float = 0.0
    feasibility_score: float = 0.0


class ReactorDesigner:
    """
    Motor de diseño paramétrico del reactor DBD.
    Optimiza geometría y parámetros para maximizar producción de CQDs.
    """

    def __init__(self, params: Optional[ReactorParameters] = None):
        self.params = params or ReactorParameters()
        self.output = DesignOutput()

    def calculate_geometry(self) -> Dict:
        """Calcula parámetros geométricos derivados"""
        p = self.params

        # Área de sección transversal del canal
        cross_section_mm2 = p.channel_width_mm * p.channel_height_mm

        # Longitud efectiva según geometría
        if p.geometry == ChamberGeometry.SERPENTINE:
            # Longitud total incluyendo curvas
            turn_length = p.channel_width_mm * 2  # Cada vuelta añade longitud
            effective_length_mm = p.channel_length_mm + p.n_serpentine_turns * turn_length
        else:
            effective_length_mm = p.channel_length_mm

        # Volúmenes
        total_volume_mm3 = cross_section_mm2 * effective_length_mm
        total_volume_ml = total_volume_mm3 / 1000

        # Volumen de líquido (profundidad parcial)
        liquid_fraction = p.liquid_depth_mm / p.channel_height_mm
        liquid_volume_ml = total_volume_ml * liquid_fraction
        gas_volume_ml = total_volume_ml * (1 - liquid_fraction)

        # Área de interfaz plasma-líquido
        interface_area_mm2 = p.channel_width_mm * effective_length_mm
        interface_area_cm2 = interface_area_mm2 / 100

        # Área de electrodo activa
        if p.electrode_config == ElectrodeConfig.EMBEDDED_WALLS:
            # Electrodos en ambas paredes
            electrode_area_cm2 = 2 * interface_area_cm2
        elif p.electrode_config == ElectrodeConfig.INTERDIGITATED:
            # Mayor área efectiva
            n_fingers = int(effective_length_mm / (p.electrode_width_mm + p.electrode_gap_mm))
            electrode_area_cm2 = n_fingers * p.electrode_width_mm * p.channel_width_mm / 100
        else:
            electrode_area_cm2 = interface_area_cm2

        return {
            'effective_length_mm': effective_length_mm,
            'cross_section_mm2': cross_section_mm2,
            'total_volume_ml': total_volume_ml,
            'liquid_volume_ml': liquid_volume_ml,
            'gas_volume_ml': gas_volume_ml,
            'interface_area_cm2': interface_area_cm2,
            'electrode_area_cm2': electrode_area_cm2,
        }

    def calculate_flow_dynamics(self, geometry: Dict) -> Dict:
        """Calcula dinámica de flujo y tiempos de residencia"""
        p = self.params

        # Tiempo de residencia del líquido
        residence_time_s = (geometry['liquid_volume_ml'] / p.liquid_flow_ml_min) * 60

        # Velocidad del líquido
        liquid_velocity_mm_s = (p.liquid_flow_ml_min / 60 * 1000) / \
                               (p.channel_width_mm * p.liquid_depth_mm)

        # Tiempo de exposición al plasma (fracción del canal con plasma)
        plasma_coverage = 0.8  # 80% del canal tiene plasma activo
        plasma_exposure_s = residence_time_s * plasma_coverage

        # Número de Reynolds (para verificar régimen laminar)
        # Re = ρvL/μ, para agua a 25°C: ρ=1000 kg/m³, μ=0.001 Pa·s
        hydraulic_diameter_mm = 2 * p.channel_width_mm * p.liquid_depth_mm / \
                                (p.channel_width_mm + p.liquid_depth_mm)
        reynolds = 1000 * (liquid_velocity_mm_s/1000) * (hydraulic_diameter_mm/1000) / 0.001

        # Tiempo de mezcla (difusión)
        diffusion_coeff = 1e-9  # m²/s para moléculas pequeñas en agua
        mixing_time_s = (p.liquid_depth_mm/1000)**2 / diffusion_coeff

        return {
            'residence_time_s': residence_time_s,
            'liquid_velocity_mm_s': liquid_velocity_mm_s,
            'plasma_exposure_s': plasma_exposure_s,
            'reynolds_number': reynolds,
            'flow_regime': 'laminar' if reynolds < 2300 else 'turbulent',
            'mixing_time_s': mixing_time_s,
            'mixing_complete': mixing_time_s < residence_time_s,
        }

    def calculate_electrical(self, geometry: Dict) -> Dict:
        """Calcula parámetros eléctricos del plasma"""
        p = self.params
        material = MATERIALS[p.dielectric_material]

        # Gap de gas para el plasma (espacio sobre el líquido)
        gas_gap_mm = max(0.1, p.channel_height_mm - p.liquid_depth_mm)

        # Campo eléctrico en el gap de gas
        # Considerando barrera dieléctrica en serie
        effective_gap_mm = gas_gap_mm + p.dielectric_thickness_mm / material.dielectric_constant
        electric_field_v_m = (p.voltage_kv * 1000) / (effective_gap_mm / 1000)

        # Verificar si supera campo de ruptura
        breakdown_achieved = electric_field_v_m > BREAKDOWN_FIELD_AIR

        # Potencia del plasma (modelo empírico de DBD)
        # P típico = 0.1-10 W/cm² de área de electrodo
        # Escala con V² y frecuencia
        power_density_w_cm2 = 0.5 * (p.voltage_kv / 10) ** 2 * (p.frequency_khz / 20)
        power_w = power_density_w_cm2 * geometry['electrode_area_cm2'] * p.duty_cycle
        power_w = max(0.1, min(500, power_w))  # Límites físicos razonables

        # Densidad de energía entregada al líquido
        energy_density_j_ml = power_w / p.liquid_flow_ml_min * 60

        # Capacitancia del sistema
        capacitance_pf = EPSILON_0 * material.dielectric_constant * \
                        (geometry['electrode_area_cm2'] * 1e-4) / \
                        (p.dielectric_thickness_mm * 1e-3) * 1e12

        return {
            'electric_field_v_m': electric_field_v_m,
            'breakdown_achieved': breakdown_achieved,
            'power_w': power_w,
            'energy_density_j_ml': energy_density_j_ml,
            'capacitance_pf': capacitance_pf,
            'dielectric_strength_ok': electric_field_v_m < material.dielectric_strength,
            'gas_gap_mm': gas_gap_mm,
        }

    def calculate_catalyst_effect(self, geometry: Dict, electrical: Dict) -> Dict:
        """
        Calcula efecto sinérgico plasma-TiO2.

        Sinergia plasma-TiO2:
        1. UV del plasma (λ < 388nm para anatase, < 413nm para rutile)
           excita TiO2 → pares e⁻/h⁺
        2. Radicales reactivos: OH•, O•, O₂⁻ atacan precursor
        3. Superficie porosa provee sitios de nucleación → CQDs

        De literatura:
        - Plasma-TiO2 sinergia: 1.5-3x mejora vs plasma solo
        - Nucleación en superficie: +0.05-0.15 monodispersidad
        - Degradación: ~5% por hora de operación (fouling)
        - Regeneración: 400°C por 1h restaura >95%
        """
        if not self.params.catalyst_type:
            return {
                'conversion_factor': 1.0,
                'monodispersity_boost': 0.0,
                'size_shift_nm': 0.0,
                'cat_area_m2': 0.0,
                'fouling_rate_per_h': 0.0,
                'regeneration_temp_C': 0,
                'regeneration_time_h': 0.0,
            }

        # Área reactiva total
        loading = self.params.catalyst_loading_mg_cm2
        porosity = self.params.catalyst_porosity
        bet_area = self.params.catalyst_surface_area_m2_g
        plasma_area = geometry['interface_area_cm2']

        # Área catalítica real = carga × área_BET × porosidad_accesible
        cat_area_m2 = (loading * 1e-3) * bet_area * porosity * (plasma_area * 1e-4)
        # Normalizar a área de referencia (1 cm² de TiO2 denso)
        cat_area_factor = min(5.0, cat_area_m2 / 1e-4)

        # Factor de activación UV
        # Plasma DBD emite UV fuerte (200-400 nm)
        # Anatase: bandgap 3.2 eV (λ < 388 nm) - mejor fotocatálisis
        # Rutile: bandgap 3.0 eV (λ < 413 nm) - más estable
        if 'anatase' in self.params.catalyst_type:
            uv_factor = 1.8   # Más activo bajo UV
        else:
            uv_factor = 1.4   # Rutile: más estable, menos activo

        # Factor de energía: más energía = más UV = más activación
        E = electrical['energy_density_j_ml']
        energy_activation = min(2.0, E / 300)

        # Conversión total
        conversion_factor = 1.0 + (uv_factor - 1.0) * cat_area_factor * energy_activation
        conversion_factor = min(3.5, conversion_factor)  # Cap físico

        # Monodispersidad: nucleación heterogénea en superficie
        mono_boost = 0.05 * porosity * min(1.0, cat_area_factor)

        # Tamaño: nucleación en superficie favorece partículas más pequeñas
        size_shift_nm = -0.2 * min(1.0, cat_area_factor)

        # Fouling: degradación por horas de operación continua
        # Regenerable con calentamiento a 400°C
        fouling_rate_per_h = 0.05  # 5% por hora

        return {
            'conversion_factor': conversion_factor,
            'monodispersity_boost': mono_boost,
            'size_shift_nm': size_shift_nm,
            'cat_area_m2': cat_area_m2,
            'fouling_rate_per_h': fouling_rate_per_h,
            'regeneration_temp_C': 400,
            'regeneration_time_h': 1.0,
        }

    def calculate_production(self, geometry: Dict, flow: Dict, electrical: Dict) -> Dict:
        """Estima producción de CQDs basado en modelo cinético"""
        p = self.params

        # Modelo de producción basado en literatura de síntesis por plasma
        # Típico: 0.1-1 mg/mL con densidades de energía de 100-1000 J/mL

        # Concentración base de producto (mg CQD / mL precursor)
        base_concentration = 0.3

        # Factor de energía: óptimo ~300-600 J/mL
        optimal_energy = 450  # J/mL
        if electrical['energy_density_j_ml'] < 100:
            energy_factor = electrical['energy_density_j_ml'] / 100 * 0.3
        elif electrical['energy_density_j_ml'] > 1000:
            energy_factor = 0.5  # Degradación por exceso de energía
        else:
            # Campana gaussiana centrada en óptimo
            energy_factor = np.exp(-((electrical['energy_density_j_ml'] - optimal_energy) / 300) ** 2)

        # Factor de tiempo de residencia: óptimo 10-30 s
        optimal_residence = 20  # s
        if flow['residence_time_s'] < 3:
            residence_factor = flow['residence_time_s'] / 3 * 0.3
        elif flow['residence_time_s'] > 60:
            residence_factor = 0.5  # Degradación/agregación
        else:
            residence_factor = np.exp(-((flow['residence_time_s'] - optimal_residence) / 20) ** 2)

        # Factor de área de contacto plasma-líquido
        area_factor = min(2.0, geometry['interface_area_cm2'] / 5)

        # Efecto catalítico (sinergia plasma-TiO2)
        catalyst_effect = self.calculate_catalyst_effect(geometry, electrical)

        # Concentración final
        concentration_mg_ml = (base_concentration * energy_factor * residence_factor *
                               area_factor * catalyst_effect['conversion_factor'])
        concentration_mg_ml = max(0.01, min(2.0, concentration_mg_ml))

        # Producción por hora
        production_mg_h = concentration_mg_ml * p.liquid_flow_ml_min * 60

        # Rendimiento (% de carbono convertido, asumiendo 10 mg/mL de precursor)
        yield_percent = concentration_mg_ml / 10 * 100

        # Tamaño de partícula (del modelo de confinamiento cuántico)
        # Mayor energía → fragmentación más intensa → partículas más pequeñas
        # Mayor tiempo → agregación → partículas más grandes
        base_size_nm = 2.5
        energy_effect = -0.3 * (electrical['energy_density_j_ml'] - 450) / 450
        time_effect = 0.2 * (flow['residence_time_s'] - 20) / 20
        size_nm = base_size_nm * (1 + energy_effect + time_effect)
        # Catalizador: nucleación en superficie favorece partículas más pequeñas
        size_nm += catalyst_effect['size_shift_nm']
        size_nm = max(1.5, min(5.0, size_nm))

        # Longitud de onda (modelo VQE validado)
        gap_ev = 1.50 + 7.26 / (size_nm ** 2)
        wavelength_nm = 1240 / gap_ev

        # Monodispersidad
        monodispersity = 0.7
        if flow['flow_regime'] == 'laminar':
            monodispersity += 0.15
        if flow['mixing_complete']:
            monodispersity += 0.1
        # Monodispersidad mejorada por nucleación en superficie del catalizador
        monodispersity += catalyst_effect['monodispersity_boost']
        monodispersity = min(0.95, monodispersity)

        # Energía específica
        specific_energy = electrical['power_w'] / max(0.01, production_mg_h) * 3600

        return {
            'production_mg_h': production_mg_h,
            'yield_percent': yield_percent,
            'concentration_mg_ml': concentration_mg_ml,
            'specific_energy_j_mg': specific_energy,
            'estimated_size_nm': size_nm,
            'estimated_wavelength_nm': wavelength_nm,
            'monodispersity_index': monodispersity,
            'in_spec': abs(wavelength_nm - TARGET_WAVELENGTH_NM) < 20,
        }

    def calculate_thermal(self, electrical: Dict) -> Dict:
        """Calcula balance térmico"""
        p = self.params
        material = MATERIALS[p.dielectric_material]

        # Calor generado por el plasma
        heat_generation_w = electrical['power_w'] * 0.3  # 30% se convierte en calor

        # Calor removido por el flujo de líquido
        # Q = m_dot * Cp * ΔT
        cp_water = 4186  # J/kg·K
        mass_flow_kg_s = p.liquid_flow_ml_min / 60 / 1000  # kg/s
        max_temp_rise = 10  # °C permitido
        heat_removal_capacity_w = mass_flow_kg_s * cp_water * max_temp_rise

        # Necesidad de enfriamiento adicional
        cooling_required = heat_generation_w > heat_removal_capacity_w

        # Temperatura estimada en el reactor
        if not cooling_required:
            temp_rise_c = heat_generation_w / (mass_flow_kg_s * cp_water)
        else:
            temp_rise_c = max_temp_rise  # Con enfriamiento activo

        return {
            'heat_generation_w': heat_generation_w,
            'heat_removal_capacity_w': heat_removal_capacity_w,
            'cooling_required': cooling_required,
            'estimated_temp_rise_c': temp_rise_c,
            'max_temp_c': p.temperature_c + temp_rise_c,
            'material_safe': (p.temperature_c + temp_rise_c) < material.max_temperature,
        }

    def calculate_scores(self, geometry: Dict, flow: Dict, electrical: Dict,
                        production: Dict, thermal: Dict) -> Dict:
        """Calcula puntuaciones de diseño"""
        p = self.params
        material = MATERIALS[p.dielectric_material]

        # Puntuación de eficiencia (0-100)
        efficiency_score = 0
        # Producción vs objetivo (40 puntos max)
        prod_ratio = min(2.0, production['production_mg_h'] / max(0.1, p.target_production_mg_h))
        efficiency_score += prod_ratio * 20

        # Calidad - wavelength en spec (40 puntos max)
        wavelength_error = abs(production['estimated_wavelength_nm'] - TARGET_WAVELENGTH_NM)
        if wavelength_error < 20:
            efficiency_score += 40
        elif wavelength_error < 50:
            efficiency_score += 40 * (1 - (wavelength_error - 20) / 30)
        else:
            efficiency_score += 10

        # Eficiencia energética (20 puntos max)
        # Óptimo: 1000-5000 J/mg
        if 1000 < production['specific_energy_j_mg'] < 5000:
            efficiency_score += 20
        elif production['specific_energy_j_mg'] < 10000:
            efficiency_score += 10

        efficiency_score = min(100, max(0, efficiency_score))

        # Puntuación de costo (0-100)
        cost_score = 100
        cost_score -= material.cost_relative * 10
        cost_score -= min(30, electrical['power_w'] * 0.3)
        if not material.printable_3d:
            cost_score -= 15
        cost_score = min(100, max(0, cost_score))

        # Puntuación de factibilidad (0-100)
        feasibility_score = 100
        if not electrical['breakdown_achieved']:
            feasibility_score -= 40
        if not electrical['dielectric_strength_ok']:
            feasibility_score -= 25
        if thermal['cooling_required']:
            feasibility_score -= 10
        if not thermal['material_safe']:
            feasibility_score -= 35
        if not flow['mixing_complete']:
            feasibility_score -= 5
        feasibility_score = min(100, max(0, feasibility_score))

        return {
            'efficiency_score': efficiency_score,
            'cost_score': cost_score,
            'feasibility_score': feasibility_score,
            'overall_score': (efficiency_score + cost_score + feasibility_score) / 3,
        }

    def design(self) -> DesignOutput:
        """Ejecuta el diseño completo"""
        geometry = self.calculate_geometry()
        flow = self.calculate_flow_dynamics(geometry)
        electrical = self.calculate_electrical(geometry)
        production = self.calculate_production(geometry, flow, electrical)
        thermal = self.calculate_thermal(electrical)
        scores = self.calculate_scores(geometry, flow, electrical, production, thermal)

        # Poblar salida
        self.output = DesignOutput(
            total_volume_ml=geometry['total_volume_ml'],
            plasma_area_cm2=geometry['interface_area_cm2'],
            liquid_volume_ml=geometry['liquid_volume_ml'],
            gas_volume_ml=geometry['gas_volume_ml'],
            residence_time_s=flow['residence_time_s'],
            plasma_exposure_time_s=flow['plasma_exposure_s'],
            power_w=electrical['power_w'],
            energy_density_j_ml=electrical['energy_density_j_ml'],
            electric_field_v_m=electrical['electric_field_v_m'],
            estimated_production_mg_h=production['production_mg_h'],
            yield_percent=production['yield_percent'],
            specific_energy_j_mg=production['specific_energy_j_mg'],
            estimated_wavelength_nm=production['estimated_wavelength_nm'],
            estimated_size_nm=production['estimated_size_nm'],
            monodispersity_index=production['monodispersity_index'],
            heat_generation_w=thermal['heat_generation_w'],
            cooling_required=thermal['cooling_required'],
            efficiency_score=scores['efficiency_score'],
            cost_score=scores['cost_score'],
            feasibility_score=scores['feasibility_score'],
        )

        return self.output

    def optimize(self, target_production_mg_h: float = 50.0,
                max_iterations: int = 1000) -> Tuple[ReactorParameters, DesignOutput]:
        """
        Optimiza parámetros para alcanzar producción objetivo.
        Usa búsqueda en grilla con refinamiento.
        """
        best_params = None
        best_output = None
        best_score = -1

        # Rangos de búsqueda
        search_space = {
            'channel_width_mm': [1.0, 1.5, 2.0, 2.5, 3.0],
            'channel_height_mm': [0.3, 0.5, 0.8, 1.0],
            'channel_length_mm': [100, 150, 200, 250],
            'n_serpentine_turns': [4, 6, 8, 10, 12],
            'voltage_kv': [8, 10, 12, 15],
            'frequency_khz': [15, 20, 25, 30],
            'liquid_flow_ml_min': [2, 5, 10, 15, 20],
            'liquid_depth_mm': [0.2, 0.3, 0.4, 0.5],
            'dielectric_material': list(MATERIALS.keys()),
            'catalyst_type': [None, 'tio2_anatase', 'tio2_rutile'],
            'catalyst_porosity': [0.4, 0.5, 0.6, 0.7],
        }

        # Búsqueda aleatoria (más eficiente que grilla completa)
        np.random.seed(42)
        for i in range(max_iterations):
            # Generar parámetros aleatorios
            cat_type = np.random.choice(search_space['catalyst_type'])
            cat_porosity = np.random.choice(search_space['catalyst_porosity'])
            params = ReactorParameters(
                geometry=ChamberGeometry.SERPENTINE,
                channel_width_mm=np.random.choice(search_space['channel_width_mm']),
                channel_height_mm=np.random.choice(search_space['channel_height_mm']),
                channel_length_mm=np.random.choice(search_space['channel_length_mm']),
                n_serpentine_turns=np.random.choice(search_space['n_serpentine_turns']),
                electrode_config=ElectrodeConfig.EMBEDDED_WALLS,
                voltage_kv=np.random.choice(search_space['voltage_kv']),
                frequency_khz=np.random.choice(search_space['frequency_khz']),
                liquid_flow_ml_min=np.random.choice(search_space['liquid_flow_ml_min']),
                liquid_depth_mm=np.random.choice(search_space['liquid_depth_mm']),
                dielectric_material=np.random.choice(search_space['dielectric_material']),
                catalyst_type=cat_type,
                catalyst_porosity=cat_porosity if cat_type else 0.6,
                target_production_mg_h=target_production_mg_h,
            )

            # Evaluar diseño
            designer = ReactorDesigner(params)
            output = designer.design()

            # Calcular score compuesto
            production_match = 1 - abs(output.estimated_production_mg_h - target_production_mg_h) / target_production_mg_h
            production_match = max(0, production_match)
            wavelength_match = 1 if abs(output.estimated_wavelength_nm - TARGET_WAVELENGTH_NM) < 20 else 0.5

            score = (output.efficiency_score * 0.4 +
                    output.feasibility_score * 0.3 +
                    output.cost_score * 0.15 +
                    production_match * 100 * 0.1 +
                    wavelength_match * 100 * 0.05)

            if score > best_score:
                best_score = score
                best_params = params
                best_output = output

        return best_params, best_output

    def print_design_report(self):
        """Imprime reporte completo del diseño"""
        p = self.params
        o = self.output
        material = MATERIALS[p.dielectric_material]

        print("\n" + "═" * 80)
        print("  REPORTE DE DISEÑO DEL REACTOR DBD PARA CQDs")
        print("═" * 80)

        print("\n┌" + "─" * 78 + "┐")
        print("│  1. GEOMETRÍA DE LA CÁMARA" + " " * 51 + "│")
        print("├" + "─" * 78 + "┤")
        print(f"│  Tipo:                    {p.geometry.value:<50} │")
        print(f"│  Ancho del canal:         {p.channel_width_mm:.1f} mm" + " " * 45 + "│")
        print(f"│  Altura del canal:        {p.channel_height_mm:.1f} mm" + " " * 45 + "│")
        print(f"│  Longitud total:          {p.channel_length_mm:.0f} mm" + " " * 44 + "│")
        print(f"│  Vueltas serpentín:       {p.n_serpentine_turns}" + " " * 51 + "│")
        print(f"│  Volumen total:           {o.total_volume_ml:.2f} mL" + " " * 43 + "│")
        print(f"│  Área plasma-líquido:     {o.plasma_area_cm2:.1f} cm²" + " " * 43 + "│")
        print("└" + "─" * 78 + "┘")

        print("\n┌" + "─" * 78 + "┐")
        print("│  2. ELECTRODOS Y MATERIALES" + " " * 50 + "│")
        print("├" + "─" * 78 + "┤")
        print(f"│  Configuración:           {p.electrode_config.value:<50} │")
        print(f"│  Material dieléctrico:    {material.name:<50} │")
        print(f"│  Constante dieléctrica:   ε_r = {material.dielectric_constant:.1f}" + " " * 40 + "│")
        print(f"│  Espesor dieléctrico:     {p.dielectric_thickness_mm:.1f} mm" + " " * 44 + "│")
        print(f"│  Gap de electrodo:        {p.electrode_gap_mm:.1f} mm" + " " * 45 + "│")
        print(f"│  Imprimible 3D:           {'Sí' if material.printable_3d else 'No':<50} │")
        print("└" + "─" * 78 + "┘")

        print("\n┌" + "─" * 78 + "┐")
        print("│  3. PARÁMETROS ELÉCTRICOS" + " " * 52 + "│")
        print("├" + "─" * 78 + "┤")
        print(f"│  Voltaje:                 {p.voltage_kv:.0f} kV" + " " * 46 + "│")
        print(f"│  Frecuencia:              {p.frequency_khz:.0f} kHz" + " " * 44 + "│")
        print(f"│  Campo eléctrico:         {o.electric_field_v_m/1e6:.1f} MV/m" + " " * 41 + "│")
        print(f"│  Potencia:                {o.power_w:.1f} W" + " " * 46 + "│")
        print(f"│  Densidad de energía:     {o.energy_density_j_ml:.0f} J/mL" + " " * 42 + "│")
        print("└" + "─" * 78 + "┘")

        print("\n┌" + "─" * 78 + "┐")
        print("│  4. FLUJO Y PROCESO" + " " * 58 + "│")
        print("├" + "─" * 78 + "┤")
        print(f"│  Caudal líquido:          {p.liquid_flow_ml_min:.0f} mL/min" + " " * 40 + "│")
        print(f"│  Profundidad líquido:     {p.liquid_depth_mm:.1f} mm" + " " * 44 + "│")
        print(f"│  Tiempo de residencia:    {o.residence_time_s:.1f} s" + " " * 45 + "│")
        print(f"│  Exposición al plasma:    {o.plasma_exposure_time_s:.1f} s" + " " * 44 + "│")
        print(f"│  Volumen líquido:         {o.liquid_volume_ml:.2f} mL" + " " * 42 + "│")
        print("└" + "─" * 78 + "┘")

        print("\n┌" + "─" * 78 + "┐")
        print("│  5. PRODUCCIÓN ESTIMADA" + " " * 54 + "│")
        print("├" + "─" * 78 + "┤")
        print(f"│  Producción:              {o.estimated_production_mg_h:.1f} mg/hora" + " " * 38 + "│")
        print(f"│  Rendimiento:             {o.yield_percent:.1f} %" + " " * 45 + "│")
        print(f"│  Energía específica:      {o.specific_energy_j_mg:.0f} J/mg" + " " * 42 + "│")
        print("└" + "─" * 78 + "┘")

        print("\n┌" + "─" * 78 + "┐")
        print("│  6. CALIDAD DEL PRODUCTO (del modelo VQE)" + " " * 36 + "│")
        print("├" + "─" * 78 + "┤")
        wavelength_status = "✓" if abs(o.estimated_wavelength_nm - TARGET_WAVELENGTH_NM) < 20 else "✗"
        print(f"│  Tamaño partícula:        {o.estimated_size_nm:.2f} nm" + " " * 42 + "│")
        print(f"│  Longitud de onda:        {o.estimated_wavelength_nm:.0f} nm {wavelength_status}" + " " * 40 + "│")
        print(f"│  Objetivo:                {TARGET_WAVELENGTH_NM:.0f} nm ± 20 nm" + " " * 37 + "│")
        print(f"│  Índice monodispersidad:  {o.monodispersity_index:.2f}" + " " * 44 + "│")
        print("└" + "─" * 78 + "┘")

        print("\n┌" + "─" * 78 + "┐")
        print("│  7. TÉRMICO" + " " * 66 + "│")
        print("├" + "─" * 78 + "┤")
        print(f"│  Calor generado:          {o.heat_generation_w:.1f} W" + " " * 45 + "│")
        cooling_status = "Requerido" if o.cooling_required else "No necesario"
        print(f"│  Enfriamiento:            {cooling_status:<50} │")
        print("└" + "─" * 78 + "┘")

        print("\n┌" + "─" * 78 + "┐")
        print("│  8. CATALIZADOR (sinergia plasma-TiO₂)" + " " * 38 + "│")
        print("├" + "─" * 78 + "┤")
        if p.catalyst_type:
            cat_name = "TiO₂ Anatase Poroso" if 'anatase' in p.catalyst_type else "TiO₂ Rutilo Poroso"
            # Recalcular efecto para reporte
            geometry = self.calculate_geometry()
            electrical = self.calculate_electrical(geometry)
            cat_effect = self.calculate_catalyst_effect(geometry, electrical)
            print(f"│  Tipo:                    {cat_name:<50} │")
            print(f"│  Porosidad:               {p.catalyst_porosity:.0%}" + " " * 50 + "│")
            print(f"│  Área BET:                {p.catalyst_surface_area_m2_g:.0f} m²/g" + " " * 42 + "│")
            print(f"│  Carga:                   {p.catalyst_loading_mg_cm2:.1f} mg/cm²" + " " * 41 + "│")
            print(f"│  Área catalítica real:     {cat_effect['cat_area_m2']*1e4:.2f} cm²" + " " * 40 + "│")
            print(f"│  Factor de conversión:     {cat_effect['conversion_factor']:.2f}x" + " " * 42 + "│")
            print(f"│  Mejora monodispersidad:   +{cat_effect['monodispersity_boost']:.3f}" + " " * 42 + "│")
            print(f"│  Shift tamaño:            {cat_effect['size_shift_nm']:+.2f} nm" + " " * 42 + "│")
            print(f"│  Fouling:                 {cat_effect['fouling_rate_per_h']:.0%}/h (regen {cat_effect['regeneration_temp_C']}°C/{cat_effect['regeneration_time_h']:.0f}h)" + " " * 22 + "│")
        else:
            print(f"│  Tipo:                    Sin catalizador" + " " * 35 + "│")
            print(f"│  (Usar --catalyst tio2_anatase o tio2_rutile para activar)" + " " * 20 + "│")
        print("└" + "─" * 78 + "┘")

        print("\n┌" + "─" * 78 + "┐")
        print("│  9. PUNTUACIONES DE DISEÑO" + " " * 51 + "│")
        print("├" + "─" * 78 + "┤")
        print(f"│  Eficiencia:              {o.efficiency_score:.0f}/100" + " " * 43 + "│")
        print(f"│  Costo:                   {o.cost_score:.0f}/100" + " " * 44 + "│")
        print(f"│  Factibilidad:            {o.feasibility_score:.0f}/100" + " " * 41 + "│")
        overall = (o.efficiency_score + o.cost_score + o.feasibility_score) / 3
        print(f"│  PUNTUACIÓN GLOBAL:       {overall:.0f}/100" + " " * 41 + "│")
        print("└" + "─" * 78 + "┘")

    def export_cad_parameters(self) -> Dict:
        """Exporta parámetros para software CAD"""
        p = self.params
        o = self.output

        return {
            "units": "mm",
            "chamber": {
                "type": p.geometry.value,
                "channel_width": float(p.channel_width_mm),
                "channel_height": float(p.channel_height_mm),
                "channel_length": float(p.channel_length_mm),
                "wall_thickness": 2.0,
                "n_turns": int(p.n_serpentine_turns) if p.geometry == ChamberGeometry.SERPENTINE else 0,
                "turn_radius": float(p.channel_width_mm * 1.5),
            },
            "electrodes": {
                "config": p.electrode_config.value,
                "width": float(p.electrode_width_mm),
                "gap": float(p.electrode_gap_mm),
                "thickness": float(p.electrode_thickness_mm),
                "material": p.electrode_material,
            },
            "dielectric": {
                "material": p.dielectric_material,
                "thickness": float(p.dielectric_thickness_mm),
            },
            "connections": {
                "inlet_diameter": 2.0,
                "outlet_diameter": 2.0,
                "gas_inlet_diameter": 1.5,
            },
            "cooling": {
                "required": bool(o.cooling_required),
                "serpentine_length": 150.0 if o.cooling_required else 0.0,
                "serpentine_diameter": 3.0,
            },
            "performance": {
                "estimated_production_mg_h": float(o.estimated_production_mg_h),
                "estimated_wavelength_nm": float(o.estimated_wavelength_nm),
                "estimated_size_nm": float(o.estimated_size_nm),
                "power_w": float(o.power_w),
            }
        }


def print_material_comparison():
    """Imprime comparación de materiales disponibles"""
    print("\n" + "═" * 80)
    print("  COMPARACIÓN DE MATERIALES DIELÉCTRICOS")
    print("═" * 80)
    print(f"\n  {'Material':<25} {'ε_r':<8} {'Rigidez':<12} {'k (W/mK)':<10} {'T_max':<8} {'3D':<5} {'Costo':<6}")
    print("  " + "-" * 75)

    for key, mat in MATERIALS.items():
        print(f"  {mat.name:<25} {mat.dielectric_constant:<8.1f} {mat.dielectric_strength/1e6:<12.0f} "
              f"{mat.thermal_conductivity:<10.2f} {mat.max_temperature:<8.0f} "
              f"{'Sí' if mat.printable_3d else 'No':<5} {mat.cost_relative:<6.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PROGRAMA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diseño paramétrico del reactor DBD")
    parser.add_argument("--optimize", action="store_true", help="Ejecutar optimización")
    parser.add_argument("--production-target", type=float, default=50.0,
                       help="Producción objetivo en mg/hora")
    parser.add_argument("--export-cad", type=str, help="Exportar parámetros CAD a archivo JSON")
    parser.add_argument("--catalyst", type=str, default=None,
                       choices=["tio2_anatase", "tio2_rutile"],
                       help="Tipo de catalizador TiO2 (default: ninguno)")
    parser.add_argument("--porosity", type=float, default=0.6,
                       help="Porosidad del catalizador 0-1 (default: 0.6)")
    args = parser.parse_args()

    print("═" * 80)
    print("  DISEÑADOR PARAMÉTRICO DE REACTOR DBD PARA CQDs")
    print("  Basado en simulación cuántica VQE y física de plasmas")
    print("═" * 80)

    if args.optimize:
        print(f"\n→ Optimizando para producción objetivo: {args.production_target} mg/hora...")
        print("  (Esto puede tardar unos segundos...)\n")

        designer = ReactorDesigner()
        best_params, best_output = designer.optimize(
            target_production_mg_h=args.production_target,
            max_iterations=500
        )

        # Crear diseñador con mejores parámetros
        designer = ReactorDesigner(best_params)
        designer.output = best_output
        designer.print_design_report()

        if args.export_cad:
            cad_params = designer.export_cad_parameters()
            with open(args.export_cad, 'w') as f:
                json.dump(cad_params, f, indent=2)
            print(f"\n✓ Parámetros CAD exportados a: {args.export_cad}")

    else:
        # Diseño con parámetros por defecto
        print("\n→ Ejecutando diseño con parámetros por defecto...")

        params = ReactorParameters(
            target_production_mg_h=args.production_target,
            catalyst_type=args.catalyst,
            catalyst_porosity=args.porosity if args.catalyst else 0.6,
        )
        designer = ReactorDesigner(params)
        designer.design()
        designer.print_design_report()

        print_material_comparison()

        if args.export_cad:
            cad_params = designer.export_cad_parameters()
            with open(args.export_cad, 'w') as f:
                json.dump(cad_params, f, indent=2)
            print(f"\n✓ Parámetros CAD exportados a: {args.export_cad}")

    print("\n" + "═" * 80)
    print("  Uso: python reactor_design.py --optimize --production-target 100")
    print("       python reactor_design.py --export-cad design.json")
    print("═" * 80)
