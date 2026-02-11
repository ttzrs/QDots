#!/usr/bin/env python3
"""
===============================================================================
  ANEXO: REACTOR DBD EN ESCALA MILIMETRICA PARA PROCESAMIENTO DE PURIN
  Escalado del microreactor a canales mm con flujos de mL/min a L/min
===============================================================================

  Contexto:
    El reactor microfluidico (reactor_design.py) opera con:
      - Canal: 2 mm ancho x 0.5 mm alto x 150 mm largo
      - Liquido: 0.3 mm profundidad
      - Flujo: 5 mL/min -> ~50 mg/h CQDs
      - Potencia: ~1.8 W
      - Volumen procesado: 0.3 L/h

    Para procesamiento relevante de purin necesitamos 10-100x mas volumen.
    Este modulo escala el reactor a dimensiones milimetricas manteniendo
    la fisica de produccion validada.

  Topologias de reactor escalado:
    1. FALLING FILM: pelicula delgada de liquido sobre placa, plasma encima
    2. MULTI-CHANNEL: N canales microfluidicos en paralelo (numbering up)
    3. ANNULAR: cilindrico coaxial, liquido en gap anular
    4. BUBBLE COLUMN: plasma en burbujas que ascienden por liquido

  Desafios de escala:
    - Uniformidad del plasma: gaps > 2mm requieren voltaje proporcional
    - Gestion termica: mas potencia = enfriamiento activo obligatorio
    - Mezcla: canales mas grandes = tiempo de difusion >> tiempo residencia
    - Densidad de energia: mantener 300-600 J/mL optimo
    - Calidad: monodispersidad degrada si mezcla es pobre

  USO:
    python reactor_scaleup.py                        # Comparar topologias
    python reactor_scaleup.py --topology falling_film --flow 50
    python reactor_scaleup.py --optimize --target 500
    python reactor_scaleup.py --purin --volume 100   # Procesar 100 L/dia de purin
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reactor_design import (
    ReactorDesigner, ReactorParameters, DesignOutput,
    MATERIALS, EPSILON_0, E_ELECTRON, K_BOLTZMANN,
    TARGET_WAVELENGTH_NM, TARGET_SIZE_NM, E_GAP_TARGET_EV,
    BREAKDOWN_FIELD_AIR, ELECTRON_TEMPERATURE_EV,
    ChamberGeometry, ElectrodeConfig, Material,
)


# =============================================================================
#  CONSTANTES DE ESCALA
# =============================================================================

# Limites fisicos de DBD atmosferico
MAX_GAS_GAP_MM = 10.0          # Mas alla de esto: arcos, no DBD uniforme
MIN_LIQUID_FILM_MM = 0.1       # Minimo para pelicula estable
MAX_VOLTAGE_KV = 30.0          # Limite practico fuentes HV accesibles
MAX_POWER_DENSITY_W_CM2 = 5.0  # Limite termico sin dano al dielectrico

# Purin: composicion tipica
PURIN_TOTAL_SOLIDS_G_L = 30.0       # Solidos totales (fresco)
PURIN_ORGANIC_FRACTION = 0.70       # 70% de solidos son organicos
PURIN_CARBON_FRACTION = 0.45        # 45% del organico es carbono
PURIN_OPTIMAL_DILUTION_G_L = 2.0    # Concentracion optima para DBD
PURIN_DENSITY_KG_M3 = 1020          # Similar al agua

# Plasma frio: parametros non-thermal DBD
TE_COLD_PLASMA_EV = 1.5            # Te tipica DBD
TGAS_MAX_COLD_C = 60.0             # Maximo para "frio"
TLIQUID_MAX_OUTLET_C = 50.0        # Maximo liquido a salida
OH_RADICAL_DENSITY_REF = 1e15      # OH* a 1 W/cm2
O_RADICAL_DENSITY_REF = 5e14       # O* atomico
O2_NEG_DENSITY_REF = 1e13          # O2-
PULSE_WIDTH_COLD_LIMIT_NS = 500    # < 500 ns = frio (proyecto.txt)
FREQUENCY_COLD_MIN_KHZ = 10.0
FREQUENCY_COLD_MAX_KHZ = 30.0
DBD_UV_FRACTION = 0.05             # 5% potencia -> UV
UV_PHOTON_ENERGY_EV = 4.0          # ~310 nm promedio
TIO2_DEFECT_DENSITY_PER_CM2 = 1e13 # Sitios Ti3+

# TiO2 como barrera dielectrica estructural
TIO2_EPSILON_R = {
    'anatase': 40.0,    # Permitividad relativa anatase
    'rutile': 85.0,     # Permitividad relativa rutile
}
TIO2_DIELECTRIC_STRENGTH_MV_M = {
    'anatase': 4.0,     # Rigidez dielectrica (MV/m)
    'rutile': 5.0,
}


# =============================================================================
#  TOPOLOGIAS DE REACTOR ESCALADO
# =============================================================================

class ScaleTopology(Enum):
    """Topologias de reactor para escala milimetrica"""
    FALLING_FILM = "falling_film"      # Pelicula descendente sobre placa
    MULTI_CHANNEL = "multi_channel"    # N canales micro en paralelo
    ANNULAR = "annular"                # Cilindrico coaxial
    BUBBLE_COLUMN = "bubble_column"    # Burbujas de plasma en liquido


@dataclass
class ScaledReactorParameters:
    """Parametros del reactor en escala milimetrica"""
    topology: ScaleTopology = ScaleTopology.FALLING_FILM

    # Dimensiones generales
    reactor_length_mm: float = 300.0       # Longitud del reactor
    reactor_width_mm: float = 50.0         # Ancho (falling film) o diametro

    # Falling film
    film_thickness_mm: float = 0.5         # Espesor de la pelicula liquida
    film_inclination_deg: float = 45.0     # Inclinacion de la placa (0=horizontal)
    gas_gap_mm: float = 3.0               # Espacio para plasma sobre film

    # Multi-channel
    n_channels: int = 16                   # Numero de canales paralelos
    channel_width_mm: float = 2.0          # Ancho de cada canal
    channel_height_mm: float = 0.5         # Altura de cada canal
    channel_length_mm: float = 200.0       # Longitud de cada canal

    # Annular
    outer_diameter_mm: float = 30.0        # Diametro externo del tubo
    inner_diameter_mm: float = 20.0        # Diametro del electrodo central
    annular_length_mm: float = 300.0       # Longitud del reactor anular

    # Bubble column
    column_diameter_mm: float = 30.0       # Diametro de la columna
    column_height_mm: float = 200.0        # Altura de liquido
    bubble_size_mm: float = 2.0            # Tamano de burbuja DBD
    gas_flow_fraction: float = 0.15        # Fraccion volumetrica de gas

    # Electricos
    voltage_kv: float = 15.0              # Voltaje (mayor por gaps mas grandes)
    frequency_khz: float = 20.0           # Frecuencia
    duty_cycle: float = 0.5               # Ciclo de trabajo
    n_electrode_segments: int = 4          # Segmentos para uniformidad

    # Dielectrico
    dielectric_material: str = "borosilicate_glass"
    dielectric_thickness_mm: float = 1.5   # Mas grueso por mayor voltaje

    # Catalizador
    catalyst_type: Optional[str] = None          # None, "tio2_anatase", "tio2_rutile"
    catalyst_porosity: float = 0.60              # Porosidad (0-1), tipico 0.4-0.7
    catalyst_surface_area_m2_g: float = 50.0     # Area BET (m2/g), 50-200 tipico
    catalyst_thickness_mm: float = 0.5           # Espesor de capa catalitica
    catalyst_loading_mg_cm2: float = 2.0         # Carga de TiO2 por cm2

    # Plasma frio
    pulse_width_ns: float = 100.0                # Ancho de pulso (< 500 ns = frio)

    # TiO2 como barrera dielectrica estructural
    tio2_barrier: bool = False                   # TiO2 es la barrera dielectrica misma
    tio2_barrier_phase: str = "anatase"          # Fase del TiO2 barrera

    # Flujo
    liquid_flow_ml_min: float = 50.0       # Caudal de liquido
    gas_flow_ml_min: float = 200.0         # Caudal de gas

    # Enfriamiento
    coolant_flow_ml_min: float = 100.0     # Caudal de refrigerante
    coolant_temp_C: float = 15.0           # Temperatura del refrigerante

    # Produccion
    target_production_mg_h: float = 500.0


@dataclass
class ScaleupResult:
    """Resultado del reactor escalado"""
    topology: str
    # Geometria
    plasma_area_cm2: float
    liquid_volume_ml: float
    gas_volume_ml: float
    # Flujo
    flow_ml_min: float
    residence_time_s: float
    reynolds_number: float
    flow_regime: str
    mixing_quality: str
    # Electrico
    power_w: float
    energy_density_j_ml: float
    electric_field_v_m: float
    voltage_kv: float
    breakdown_ok: bool
    # Produccion
    production_mg_h: float
    concentration_mg_ml: float
    wavelength_nm: float
    size_nm: float
    monodispersity: float
    in_spec: bool
    # Termico
    heat_generation_w: float
    heat_removal_w: float
    cooling_sufficient: bool
    max_temp_C: float
    # Purin
    purin_volume_L_day: float
    purin_bruto_L_day: float
    # Scores
    efficiency_score: float
    feasibility_score: float
    cost_score: float
    # Comparison
    vs_micro_production_factor: float
    vs_micro_efficiency_factor: float
    # Plasma frio
    Te_eV: float = 0.0
    Tgas_C: float = 25.0
    non_thermal_ratio: float = 0.0
    plasma_regime: str = "no_plasma"
    radical_OH_cm3: float = 0.0
    radical_O_cm3: float = 0.0
    radical_O2neg_cm3: float = 0.0
    # Diseno de enfriamiento
    cooling_margin_percent: float = 0.0
    coolant_path_length_mm: float = 0.0
    required_coolant_flow_ml_min: float = 0.0


class MillimetricReactorDesigner:
    """
    Disenador de reactores DBD en escala milimetrica.

    Escala la fisica del microreactor validado a dimensiones mayores,
    manteniendo los modelos de produccion y anadiendo correcciones
    por uniformidad de plasma, mezcla, y gestion termica.
    """

    # Referencia: microreactor a 5 mL/min
    MICRO_PRODUCTION_MG_H = 50.0
    MICRO_FLOW_ML_MIN = 5.0
    MICRO_POWER_W = 1.82

    def __init__(self, params: Optional[ScaledReactorParameters] = None):
        self.params = params or ScaledReactorParameters()

    # =========================================================================
    #  GEOMETRIA POR TOPOLOGIA
    # =========================================================================

    def calculate_geometry(self) -> Dict:
        """Calcula geometria segun topologia"""
        p = self.params
        topology = p.topology

        if topology == ScaleTopology.FALLING_FILM:
            return self._geometry_falling_film()
        elif topology == ScaleTopology.MULTI_CHANNEL:
            return self._geometry_multi_channel()
        elif topology == ScaleTopology.ANNULAR:
            return self._geometry_annular()
        elif topology == ScaleTopology.BUBBLE_COLUMN:
            return self._geometry_bubble_column()

    def _geometry_falling_film(self) -> Dict:
        """
        Reactor de pelicula descendente.

        Liquido fluye como pelicula delgada sobre una placa inclinada.
        Plasma se genera en el espacio de gas sobre la pelicula.
        Gran area de contacto plasma-liquido con film delgado.

            GAS (plasma)     <- electrodos en las paredes
          ==================
          ~~~~~~~~~~~~~~~~~~  <- pelicula liquida (0.3-1 mm)
          ------------------  <- placa con sustrato dielectrico
        """
        p = self.params

        # Area de la placa
        plate_area_mm2 = p.reactor_length_mm * p.reactor_width_mm
        plate_area_cm2 = plate_area_mm2 / 100.0

        # Volumen de liquido (pelicula)
        liquid_vol_mm3 = plate_area_mm2 * p.film_thickness_mm
        liquid_vol_ml = liquid_vol_mm3 / 1000.0

        # Volumen de gas (sobre la pelicula)
        gas_vol_mm3 = plate_area_mm2 * p.gas_gap_mm
        gas_vol_ml = gas_vol_mm3 / 1000.0

        # Velocidad del film por gravedad + bombeo
        # v_gravity = rho*g*sin(theta)*h^2 / (3*mu) (Nusselt film)
        theta_rad = np.radians(p.film_inclination_deg)
        h_m = p.film_thickness_mm * 1e-3
        v_gravity = (998 * 9.81 * np.sin(theta_rad) * h_m**2) / (3 * 0.001)

        # Velocidad por bombeo
        flow_m3_s = p.liquid_flow_ml_min / 60.0 * 1e-6
        cross_section_m2 = p.reactor_width_mm * 1e-3 * p.film_thickness_mm * 1e-3
        v_pump = flow_m3_s / cross_section_m2

        v_total = v_gravity + v_pump

        # Tiempo de residencia
        t_res = (p.reactor_length_mm * 1e-3) / v_total

        # Reynolds del film
        Re_film = 998 * v_total * h_m / 0.001

        return {
            'plasma_area_cm2': plate_area_cm2,
            'liquid_volume_ml': liquid_vol_ml,
            'gas_volume_ml': gas_vol_ml,
            'velocity_m_s': v_total,
            'v_gravity_m_s': v_gravity,
            'v_pump_m_s': v_pump,
            'residence_time_s': t_res,
            'reynolds_number': Re_film,
            'flow_regime': 'laminar' if Re_film < 400 else 'wavy',
            'mixing_quality': 'good' if p.film_thickness_mm < 0.5 else 'moderate',
            'gas_gap_mm': p.gas_gap_mm,
        }

    def _geometry_multi_channel(self) -> Dict:
        """
        Reactor multi-canal (numbering up).

        N canales microfluidicos identicos en paralelo.
        Cada canal opera en regimen validado del micro.
        Escalado lineal y predecible.

          [canal 1] ->
          [canal 2] ->   manifold de
          [canal 3] ->   distribucion
          ...        ->
          [canal N] ->
        """
        p = self.params

        # Geometria de cada canal (usa microreactor validado)
        ch_cross_mm2 = p.channel_width_mm * p.channel_height_mm
        ch_vol_mm3 = ch_cross_mm2 * p.channel_length_mm
        ch_vol_ml = ch_vol_mm3 / 1000.0

        # Volumen total
        liquid_fraction = 0.6  # Profundidad parcial del liquido
        liquid_vol_ml = ch_vol_ml * liquid_fraction * p.n_channels
        gas_vol_ml = ch_vol_ml * (1 - liquid_fraction) * p.n_channels

        # Area de interfaz por canal
        ch_area_mm2 = p.channel_width_mm * p.channel_length_mm
        total_area_cm2 = ch_area_mm2 * p.n_channels / 100.0

        # Flujo por canal
        flow_per_channel = p.liquid_flow_ml_min / p.n_channels

        # Velocidad en cada canal
        liquid_depth = p.channel_height_mm * liquid_fraction
        v_mm_s = (flow_per_channel / 60.0 * 1000.0) / (p.channel_width_mm * liquid_depth)

        # Tiempo de residencia
        t_res = p.channel_length_mm / v_mm_s

        # Reynolds por canal
        Dh = 2 * p.channel_width_mm * liquid_depth / (p.channel_width_mm + liquid_depth)
        Re = 998 * (v_mm_s * 1e-3) * (Dh * 1e-3) / 0.001

        return {
            'plasma_area_cm2': total_area_cm2,
            'liquid_volume_ml': liquid_vol_ml,
            'gas_volume_ml': gas_vol_ml,
            'velocity_m_s': v_mm_s * 1e-3,
            'v_gravity_m_s': 0.0,
            'v_pump_m_s': v_mm_s * 1e-3,
            'residence_time_s': t_res,
            'reynolds_number': Re,
            'flow_regime': 'laminar' if Re < 2300 else 'turbulent',
            'mixing_quality': 'excellent',  # Canales micro = difusion rapida
            'gas_gap_mm': p.channel_height_mm * (1 - liquid_fraction),
            'n_channels': p.n_channels,
            'flow_per_channel_ml_min': flow_per_channel,
        }

    def _geometry_annular(self) -> Dict:
        """
        Reactor anular coaxial.

        Liquido fluye por el gap anular entre dos tubos concentricos.
        Electrodo central + electrodo externo con dielectrico.
        Buena distribucion de plasma para gaps de 2-5 mm.

          |  ext electrode  |
          | [dielectric]    |
          |   gas gap       |
          |   ~~~~~~~~~~~~  | <- interfaz plasma-liquido
          |   liquid flow   |
          |   ~~~~~~~~~~~~  |
          |   gas gap       |
          | [dielectric]    |
          |  int electrode  |
        """
        p = self.params

        r_out = p.outer_diameter_mm / 2.0
        r_in = p.inner_diameter_mm / 2.0
        gap = r_out - r_in

        # Volumen anular total
        vol_total_mm3 = np.pi * (r_out**2 - r_in**2) * p.annular_length_mm
        vol_total_ml = vol_total_mm3 / 1000.0

        # Liquid ocupa la mitad inferior del gap, gas la mitad superior
        # En tubo horizontal el liquido se acumula abajo
        liquid_fraction = 0.5
        liquid_vol_ml = vol_total_ml * liquid_fraction
        gas_vol_ml = vol_total_ml * (1 - liquid_fraction)

        # Area de interfaz (superficie cilindrica media)
        r_interface = (r_out + r_in) / 2.0
        interface_area_mm2 = 2 * np.pi * r_interface * p.annular_length_mm
        interface_area_cm2 = interface_area_mm2 / 100.0

        # Velocidad del liquido
        flow_m3_s = p.liquid_flow_ml_min / 60.0 * 1e-6
        annular_area_m2 = np.pi * ((r_out * 1e-3)**2 - (r_in * 1e-3)**2) * liquid_fraction
        v_m_s = flow_m3_s / annular_area_m2

        # Tiempo de residencia
        t_res = (p.annular_length_mm * 1e-3) / v_m_s

        # Reynolds anular
        Dh = 2 * gap * 1e-3  # Diametro hidraulico del gap anular
        Re = 998 * v_m_s * Dh / 0.001

        return {
            'plasma_area_cm2': interface_area_cm2,
            'liquid_volume_ml': liquid_vol_ml,
            'gas_volume_ml': gas_vol_ml,
            'velocity_m_s': v_m_s,
            'v_gravity_m_s': 0.0,
            'v_pump_m_s': v_m_s,
            'residence_time_s': t_res,
            'reynolds_number': Re,
            'flow_regime': 'laminar' if Re < 2300 else 'turbulent',
            'mixing_quality': 'moderate',
            'gas_gap_mm': gap * (1 - liquid_fraction),
            'annular_gap_mm': gap,
        }

    def _geometry_bubble_column(self) -> Dict:
        """
        Reactor de columna de burbujas.

        Plasma generado dentro de burbujas de gas que ascienden
        por la columna de liquido. Maximo contacto 3D.

          |  oo    o   |  <- burbujas con plasma interno
          |    o  oo   |
          | o    o   o |
          |  oo  o     |
          | ==========  | <- generador de burbujas + electrodo
        """
        p = self.params

        # Volumen de la columna
        r = p.column_diameter_mm / 2.0
        col_vol_mm3 = np.pi * r**2 * p.column_height_mm
        col_vol_ml = col_vol_mm3 / 1000.0

        # Volumen de gas (burbujas)
        gas_vol_ml = col_vol_ml * p.gas_flow_fraction
        liquid_vol_ml = col_vol_ml * (1 - p.gas_flow_fraction)

        # Area de interfaz: suma de superficies de burbujas
        # N_burbujas = gas_vol / vol_burbuja
        r_bubble = p.bubble_size_mm / 2.0
        vol_bubble_mm3 = (4.0 / 3.0) * np.pi * r_bubble**3
        n_bubbles = (gas_vol_ml * 1000.0) / vol_bubble_mm3
        area_per_bubble_mm2 = 4 * np.pi * r_bubble**2
        total_area_cm2 = n_bubbles * area_per_bubble_mm2 / 100.0

        # Velocidad ascendente de burbuja (Stokes para burbujas pequenas)
        # v_b = (rho_l - rho_g) * g * d_b^2 / (18 * mu)
        d_b_m = p.bubble_size_mm * 1e-3
        v_bubble = (998 - 1.2) * 9.81 * d_b_m**2 / (18 * 0.001)
        v_bubble = min(v_bubble, 0.3)  # Limit fisico para burbujas mm

        # Tiempo de residencia del liquido (flujo horizontal por la columna)
        flow_m3_s = p.liquid_flow_ml_min / 60.0 * 1e-6
        cross_m2 = np.pi * (r * 1e-3)**2 * (1 - p.gas_flow_fraction)
        v_liquid = flow_m3_s / cross_m2
        t_res = (p.column_height_mm * 1e-3) / max(v_liquid, 1e-6)

        # Reynolds del liquido
        Re = 998 * v_liquid * (p.column_diameter_mm * 1e-3) / 0.001

        return {
            'plasma_area_cm2': total_area_cm2,
            'liquid_volume_ml': liquid_vol_ml,
            'gas_volume_ml': gas_vol_ml,
            'velocity_m_s': v_liquid,
            'v_gravity_m_s': 0.0,
            'v_pump_m_s': v_liquid,
            'residence_time_s': t_res,
            'reynolds_number': Re,
            'flow_regime': 'turbulent_bubbly',
            'mixing_quality': 'excellent',  # Burbujas mezclan muy bien
            'gas_gap_mm': p.bubble_size_mm,
            'n_bubbles': n_bubbles,
            'v_bubble_m_s': v_bubble,
        }

    # =========================================================================
    #  ELECTRICA (escalada)
    # =========================================================================

    def calculate_electrical(self, geometry: Dict) -> Dict:
        """Parametros electricos escalados"""
        p = self.params
        mat = MATERIALS[p.dielectric_material]

        gas_gap = geometry['gas_gap_mm']

        # Campo electrico — con opcion de barrera TiO2
        if p.tio2_barrier and p.catalyst_type:
            # TiO2 ES la barrera dielectrica: epsilon_r mucho mayor
            phase = p.tio2_barrier_phase
            eps_tio2 = TIO2_EPSILON_R.get(phase, 40.0)
            t_tio2 = p.dielectric_thickness_mm  # Mismo espesor, otro material
            effective_gap_mm = gas_gap + t_tio2 / eps_tio2
            # Verificar rigidez dielectrica del TiO2
            E_in_tio2 = (p.voltage_kv * 1000) / (t_tio2 * 1e-3)  # V/m
            tio2_strength = TIO2_DIELECTRIC_STRENGTH_MV_M.get(phase, 4.0) * 1e6
            tio2_dielectric_ok = E_in_tio2 < tio2_strength
        else:
            effective_gap_mm = gas_gap + p.dielectric_thickness_mm / mat.dielectric_constant
            tio2_dielectric_ok = True  # No aplica

        E_field = (p.voltage_kv * 1000) / (effective_gap_mm * 1e-3)

        breakdown_ok = E_field > BREAKDOWN_FIELD_AIR

        # Potencia por area (modelo empirico DBD)
        # P_density escala con V^2 * f, pero se satura a gaps grandes
        # por no-uniformidad del plasma
        P_density = 0.5 * (p.voltage_kv / 10)**2 * (p.frequency_khz / 20)

        # Penalizacion por gap grande: la uniformidad cae
        if gas_gap > 2.0:
            uniformity_factor = 2.0 / gas_gap  # Decae linealmente
        else:
            uniformity_factor = 1.0

        # Para bubble column, cada burbuja es un micro-reactor con gap pequeno
        if p.topology == ScaleTopology.BUBBLE_COLUMN:
            uniformity_factor = 0.8  # Buena uniformidad dentro de burbujas

        P_density *= uniformity_factor

        # Segmentacion de electrodos mejora uniformidad
        if p.n_electrode_segments > 1 and gas_gap > 2.0:
            segment_boost = min(1.5, 1.0 + 0.1 * p.n_electrode_segments)
            P_density *= segment_boost

        # Potencia total
        power_w = P_density * geometry['plasma_area_cm2'] * p.duty_cycle
        power_w = max(0.5, min(2000, power_w))

        # Densidad de energia
        energy_density = power_w / p.liquid_flow_ml_min * 60

        # Verificar dielectrico
        if p.tio2_barrier and p.catalyst_type:
            dielectric_ok = tio2_dielectric_ok
        else:
            dielectric_ok = E_field < mat.dielectric_strength

        return {
            'electric_field_v_m': E_field,
            'breakdown_ok': breakdown_ok,
            'dielectric_ok': dielectric_ok,
            'power_density_w_cm2': P_density * p.duty_cycle,
            'uniformity_factor': uniformity_factor,
            'power_w': power_w,
            'energy_density_j_ml': energy_density,
            'voltage_kv': p.voltage_kv,
            'effective_gap_mm': effective_gap_mm,
            'tio2_barrier_active': p.tio2_barrier and p.catalyst_type is not None,
        }

    # =========================================================================
    #  PLASMA FRIO (non-thermal DBD)
    # =========================================================================

    def _calculate_cold_plasma(self, electrical: Dict, geometry: Dict) -> Dict:
        """
        Modela el equilibrio non-thermal del plasma DBD.

        En un DBD frio:
        - Te >> Tgas (electrones calientes, gas frio)
        - Pulsos cortos (<500 ns) limitan calentamiento del gas
        - Radicales OH*, O*, O2- escalan con potencia
        - UV generado activa TiO2
        """
        p = self.params

        # --- Te: temperatura electronica ---
        # Te sube con campo electrico, capped a 3 eV para DBD atmosferico
        E_field = electrical['electric_field_v_m']
        E_ratio = E_field / BREAKDOWN_FIELD_AIR
        Te_eV = TE_COLD_PLASMA_EV * (E_ratio ** 0.3)
        Te_eV = min(3.0, max(0.5, Te_eV))

        # --- Tgas: temperatura del gas ---
        # Calentamiento estacionario con correccion por ancho de pulso
        # Tgas = T_amb + P_heat / (h_conv * A) * sqrt(tau / 500ns)
        P_heat = electrical['power_w'] * 0.3  # 30% -> calor
        plasma_area_m2 = geometry['plasma_area_cm2'] * 1e-4
        h_conv = 50.0  # W/m2/K conveccion forzada tipica
        pulse_factor = np.sqrt(p.pulse_width_ns / PULSE_WIDTH_COLD_LIMIT_NS)
        pulse_factor = min(2.0, pulse_factor)

        if plasma_area_m2 > 0 and h_conv > 0:
            dT_gas = P_heat / (h_conv * plasma_area_m2) * pulse_factor
        else:
            dT_gas = 0.0
        Tgas_C = 25.0 + dT_gas

        # --- Ratio non-thermal ---
        # Te en Kelvin / Tgas en Kelvin; debe ser >> 1 (tipico 30-100)
        Te_K = Te_eV * E_ELECTRON / K_BOLTZMANN
        Tgas_K = Tgas_C + 273.15
        non_thermal_ratio = Te_K / Tgas_K if Tgas_K > 0 else 0

        # --- Regimen de plasma ---
        if (Tgas_C < TGAS_MAX_COLD_C and
                p.pulse_width_ns < PULSE_WIDTH_COLD_LIMIT_NS):
            plasma_regime = "frio_DBD"
        elif Tgas_C < 100:
            plasma_regime = "tibio_DBD"
        else:
            plasma_regime = "termico"

        # --- Radicales ---
        # Escalan como (P/P_ref)^0.7 con correccion de pulso corto
        P_density = electrical['power_density_w_cm2']
        P_ref = 1.0  # W/cm2 referencia
        power_scale = (P_density / P_ref) ** 0.7 if P_density > 0 else 0

        # Pulso corto favorece radicales (menos recombinacion termica)
        if p.pulse_width_ns < 200:
            short_pulse_boost = 1.3
        elif p.pulse_width_ns < PULSE_WIDTH_COLD_LIMIT_NS:
            short_pulse_boost = 1.0
        else:
            short_pulse_boost = 0.7

        radical_OH = OH_RADICAL_DENSITY_REF * power_scale * short_pulse_boost
        radical_O = O_RADICAL_DENSITY_REF * power_scale * short_pulse_boost
        radical_O2neg = O2_NEG_DENSITY_REF * power_scale * short_pulse_boost

        # --- UV fotones para activacion TiO2 ---
        # P_uv = P_total * 5%
        # N_photons/s = P_uv / E_photon
        P_uv_w = electrical['power_w'] * DBD_UV_FRACTION
        E_photon_J = UV_PHOTON_ENERGY_EV * E_ELECTRON
        if E_photon_J > 0:
            uv_photon_rate = P_uv_w / E_photon_J  # fotones/s
        else:
            uv_photon_rate = 0
        if plasma_area_m2 > 0:
            uv_flux_cm2 = uv_photon_rate / (plasma_area_m2 * 1e4)  # fotones/s/cm2
        else:
            uv_flux_cm2 = 0

        return {
            'Te_eV': Te_eV,
            'Tgas_C': Tgas_C,
            'Te_K': Te_K,
            'Tgas_K': Tgas_K,
            'non_thermal_ratio': non_thermal_ratio,
            'plasma_regime': plasma_regime,
            'radical_OH_cm3': radical_OH,
            'radical_O_cm3': radical_O,
            'radical_O2neg_cm3': radical_O2neg,
            'uv_photon_rate': uv_photon_rate,
            'uv_flux_cm2': uv_flux_cm2,
            'pulse_factor': pulse_factor,
            'short_pulse_boost': short_pulse_boost,
        }

    # =========================================================================
    #  DISENO DE ENFRIAMIENTO (serpentina)
    # =========================================================================

    def _calculate_cooling_design(self, electrical: Dict, geometry: Dict) -> Dict:
        """
        Disena serpentina de enfriamiento usando modelo Dittus-Boelter.

        Nu = 0.023 * Re^0.8 * Pr^0.4 (regimen turbulento)
        h_coolant = Nu * k_agua / D_tubo
        A_serpentina = Q_heat / (h * LMTD)
        L_path = A / (pi * D_tubo)
        """
        p = self.params

        Q_heat = electrical['power_w'] * 0.3  # Calor a remover

        # Propiedades del agua refrigerante a ~20 C
        rho_w = 998.0       # kg/m3
        cp_w = 4186.0       # J/kg/K
        mu_w = 0.001        # Pa*s
        k_w = 0.60          # W/m/K
        Pr_w = cp_w * mu_w / k_w  # ~7.0

        # Tubo de enfriamiento
        D_tubo_mm = 4.0     # Diametro interno serpentina
        D_tubo_m = D_tubo_mm * 1e-3

        # Flujo de refrigerante
        coolant_m3_s = p.coolant_flow_ml_min / 60.0 * 1e-6
        A_tubo = np.pi * (D_tubo_m / 2) ** 2
        v_coolant = coolant_m3_s / A_tubo if A_tubo > 0 else 0

        # Reynolds del refrigerante
        Re_coolant = rho_w * v_coolant * D_tubo_m / mu_w

        # Coeficiente de transferencia de calor
        if Re_coolant > 2300:
            # Dittus-Boelter (turbulento)
            Nu = 0.023 * (Re_coolant ** 0.8) * (Pr_w ** 0.4)
        else:
            # Laminar en tubo: Nu = 3.66 (pared isotermica)
            Nu = 3.66

        h_coolant = Nu * k_w / D_tubo_m if D_tubo_m > 0 else 0

        # LMTD (Log Mean Temperature Difference)
        T_hot = 25.0 + Q_heat / (cp_w * max(coolant_m3_s * rho_w, 1e-6))
        T_cold_in = p.coolant_temp_C
        T_cold_out = T_cold_in + Q_heat / (cp_w * max(coolant_m3_s * rho_w, 1e-6))
        dT1 = T_hot - T_cold_in
        dT2 = T_hot - T_cold_out
        if dT1 > 0 and dT2 > 0 and abs(dT1 - dT2) > 0.01:
            LMTD = (dT1 - dT2) / np.log(dT1 / dT2)
        elif dT1 > 0:
            LMTD = dT1
        else:
            LMTD = 1.0  # Fallback

        # Area de serpentina necesaria
        if h_coolant > 0 and LMTD > 0:
            A_serpentina_m2 = Q_heat / (h_coolant * LMTD)
        else:
            A_serpentina_m2 = 0

        # Longitud de serpentina
        L_path_m = A_serpentina_m2 / (np.pi * D_tubo_m) if D_tubo_m > 0 else 0
        L_path_mm = L_path_m * 1000

        # Capacidad maxima de enfriamiento
        heat_capacity_w = h_coolant * A_serpentina_m2 * LMTD if A_serpentina_m2 > 0 else 0

        # Margen de enfriamiento
        if Q_heat > 0:
            # Capacidad total = flujo liquido + serpentina
            mass_flow_liquid = p.liquid_flow_ml_min / 60.0 * 1e-3  # kg/s
            heat_liquid_w = mass_flow_liquid * cp_w * 10.0  # 10K delta
            coolant_flow_w = coolant_m3_s * rho_w * cp_w * (25 - p.coolant_temp_C)
            total_capacity = heat_liquid_w + coolant_flow_w
            cooling_margin = (total_capacity - Q_heat) / Q_heat * 100
        else:
            cooling_margin = 100.0

        # Flujo minimo requerido de refrigerante
        if Q_heat > 0 and cp_w > 0 and (25 - p.coolant_temp_C) > 0:
            required_flow_m3_s = Q_heat / (rho_w * cp_w * (25 - p.coolant_temp_C))
            required_flow_ml_min = required_flow_m3_s * 1e6 * 60
        else:
            required_flow_ml_min = 0.0

        return {
            'h_coolant_w_m2k': h_coolant,
            'Re_coolant': Re_coolant,
            'Nu_coolant': Nu,
            'LMTD_K': LMTD,
            'A_serpentina_m2': A_serpentina_m2,
            'coolant_path_length_mm': L_path_mm,
            'cooling_margin_percent': cooling_margin,
            'required_coolant_flow_ml_min': required_flow_ml_min,
            'D_tubo_mm': D_tubo_mm,
        }

    # =========================================================================
    #  CATALIZADOR (sinergia plasma-TiO2)
    # =========================================================================

    def _calculate_catalyst_effect(self, geometry: Dict, electrical: Dict,
                                    cold_plasma: Optional[Dict] = None) -> Dict:
        """
        Sinergia plasma-TiO2 con correcciones por topologia.

        Si cold_plasma data disponible, usa flujo UV calculado y sitios
        de nucleacion Ti3+. Si no, fallback a factores fijos.

        Contacto catalizador-plasma por topologia:
        - Falling film: catalizador en la placa = excelente contacto
        - Multi-channel: catalizador en paredes = bueno
        - Annular: catalizador en electrodo central = bueno
        - Bubble column: catalizador como particulas suspendidas = moderado (slurry)
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
                'topology_contact': 'N/A',
                'nucleation_sites': 0.0,
                'uv_quantum_yield': 0.0,
            }

        # Area reactiva total
        loading = self.params.catalyst_loading_mg_cm2
        porosity = self.params.catalyst_porosity
        bet_area = self.params.catalyst_surface_area_m2_g
        plasma_area = geometry['plasma_area_cm2']

        # Area catalitica real = carga * area_BET * porosidad_accesible
        cat_area_m2 = (loading * 1e-3) * bet_area * porosity * (plasma_area * 1e-4)
        cat_area_factor = min(5.0, cat_area_m2 / 1e-4)

        # Factor UV: modelo basado en fotones si cold_plasma disponible
        if cold_plasma is not None and cold_plasma.get('uv_flux_cm2', 0) > 0:
            # Quantum yield: anatase ~0.04, rutile ~0.02
            if 'anatase' in self.params.catalyst_type:
                qy = 0.04
            else:
                qy = 0.02
            # Absorcion UV por TiO2 (~80% de fotones incidentes)
            absorption = 0.80
            # Factor UV calculado: fotones absorbidos * QY, normalizado
            uv_absorbed = cold_plasma['uv_flux_cm2'] * absorption * qy
            # Referencia: 1e12 fotones efectivos/s/cm2 = factor 1.8
            uv_factor = 1.0 + 0.8 * min(2.0, uv_absorbed / 1e12)
        else:
            # Fallback a valores fijos (backward compatible)
            qy = 0.0
            if 'anatase' in self.params.catalyst_type:
                uv_factor = 1.8
            else:
                uv_factor = 1.4

        # Sitios de nucleacion Ti3+
        nucleation_sites = TIO2_DEFECT_DENSITY_PER_CM2 * plasma_area * porosity
        # Mas sitios -> nucleacion mas uniforme -> mejor monodispersidad
        nucleation_norm = min(1.0, nucleation_sites / 1e16)

        # Factor de energia
        E = electrical['energy_density_j_ml']
        energy_activation = min(2.0, E / 300)

        # Correccion por topologia: calidad del contacto catalizador-plasma
        topo = self.params.topology
        if topo == ScaleTopology.FALLING_FILM:
            topo_factor = 1.0       # Excelente: catalizador en la placa
            topo_contact = 'excelente (placa)'
        elif topo == ScaleTopology.MULTI_CHANNEL:
            topo_factor = 0.9       # Bueno: catalizador en paredes del canal
            topo_contact = 'bueno (paredes)'
        elif topo == ScaleTopology.ANNULAR:
            topo_factor = 0.85      # Bueno: catalizador en electrodo central
            topo_contact = 'bueno (electrodo central)'
        elif topo == ScaleTopology.BUBBLE_COLUMN:
            topo_factor = 0.6       # Moderado: slurry TiO2 en suspension
            topo_contact = 'moderado (slurry)'
        else:
            topo_factor = 0.7
            topo_contact = 'desconocido'

        # Conversion total con correccion topologica
        conversion_factor = 1.0 + (uv_factor - 1.0) * cat_area_factor * energy_activation * topo_factor
        conversion_factor = min(3.5, conversion_factor)

        # Monodispersidad: base + nucleacion Ti3+
        mono_boost = 0.05 * porosity * min(1.0, cat_area_factor) * topo_factor
        mono_boost += 0.02 * nucleation_norm  # Extra por nucleacion uniforme

        # Tamano
        size_shift_nm = -0.2 * min(1.0, cat_area_factor) * topo_factor

        # Fouling
        fouling_rate_per_h = 0.05

        return {
            'conversion_factor': conversion_factor,
            'monodispersity_boost': mono_boost,
            'size_shift_nm': size_shift_nm,
            'cat_area_m2': cat_area_m2,
            'fouling_rate_per_h': fouling_rate_per_h,
            'regeneration_temp_C': 400,
            'regeneration_time_h': 1.0,
            'topology_contact': topo_contact,
            'nucleation_sites': nucleation_sites,
            'uv_quantum_yield': qy,
        }

    # =========================================================================
    #  PRODUCCION (modelo escalado)
    # =========================================================================

    def calculate_production(self, geometry: Dict, electrical: Dict,
                             cold_plasma: Optional[Dict] = None) -> Dict:
        """
        Estima produccion usando el mismo modelo cinetico que el microreactor,
        con correcciones por calidad de mezcla, uniformidad de plasma, y radicales.
        """
        p = self.params

        # Modelo base (identico a reactor_design.py)
        base_concentration = 0.3  # mg CQD / mL precursor

        # Factor de energia: optimo ~300-600 J/mL
        E = electrical['energy_density_j_ml']
        optimal_energy = 450
        if E < 100:
            energy_factor = E / 100 * 0.3
        elif E > 1000:
            energy_factor = 0.5
        else:
            energy_factor = np.exp(-((E - optimal_energy) / 300)**2)

        # Factor de tiempo de residencia: optimo 10-30 s
        t_res = geometry['residence_time_s']
        optimal_residence = 20
        if t_res < 3:
            residence_factor = t_res / 3 * 0.3
        elif t_res > 60:
            residence_factor = 0.5
        else:
            residence_factor = np.exp(-((t_res - optimal_residence) / 20)**2)

        # Factor de area de contacto (normalizado a 5 cm2 del micro)
        area_factor = min(2.0, geometry['plasma_area_cm2'] / 5.0)

        # CORRECCIONES POR ESCALA (no presentes en microreactor)

        # 1. Calidad de mezcla: si el fluido no se mezcla bien,
        #    solo la capa superficial en contacto con plasma reacciona
        mixing = geometry['mixing_quality']
        if mixing == 'excellent':
            mixing_factor = 1.0
        elif mixing == 'good':
            mixing_factor = 0.85
        elif mixing == 'moderate':
            mixing_factor = 0.65
        else:
            mixing_factor = 0.45

        # 2. Uniformidad de plasma: plasma no-uniforme produce
        #    distribucion ancha de tamanos -> peor monodispersidad
        uniformity = electrical['uniformity_factor']

        # 3. Efecto catalitico (sinergia plasma-TiO2, ahora con datos UV)
        catalyst_effect = self._calculate_catalyst_effect(geometry, electrical,
                                                          cold_plasma)

        # 4. Factor de radicales (OH* y O* mejoran fragmentacion del precursor)
        #    Solo aplica con catalizador (sinergia plasma-catalizador)
        radical_factor = 1.0
        if cold_plasma is not None and p.catalyst_type:
            OH_norm = cold_plasma['radical_OH_cm3'] / OH_RADICAL_DENSITY_REF
            O_norm = cold_plasma['radical_O_cm3'] / O_RADICAL_DENSITY_REF
            radical_factor = 1.0 + 0.1 * (OH_norm + O_norm - 2.0)
            radical_factor = max(0.8, min(1.3, radical_factor))

        # Concentracion final
        concentration = (base_concentration * energy_factor * residence_factor *
                         area_factor * mixing_factor * uniformity *
                         catalyst_effect['conversion_factor'] * radical_factor)
        concentration = max(0.01, min(2.0, concentration))

        # Produccion
        production_mg_h = concentration * p.liquid_flow_ml_min * 60

        # Tamano de particula (mismo modelo + correccion mezcla)
        base_size = 2.5
        energy_effect = -0.3 * (E - 450) / 450
        time_effect = 0.2 * (t_res - 20) / 20
        size_nm = base_size * (1 + energy_effect + time_effect)
        # Catalizador: nucleacion en superficie favorece particulas mas pequenas
        size_nm += catalyst_effect['size_shift_nm']
        size_nm = max(1.5, min(5.0, size_nm))

        # Longitud de onda (modelo VQE)
        gap_ev = 1.50 + 7.26 / (size_nm**2)
        wavelength_nm = 1240 / gap_ev

        # Monodispersidad (afectada por mezcla y uniformidad)
        mono = 0.7
        if geometry['flow_regime'] in ('laminar', 'laminar_film'):
            mono += 0.15
        if mixing == 'excellent':
            mono += 0.1
        mono *= uniformity  # Plasma no-uniforme degrada monodispersidad
        # Monodispersidad mejorada por nucleacion en superficie del catalizador
        mono += catalyst_effect['monodispersity_boost']
        mono = max(0.3, min(0.95, mono))

        return {
            'production_mg_h': production_mg_h,
            'concentration_mg_ml': concentration,
            'wavelength_nm': wavelength_nm,
            'size_nm': size_nm,
            'monodispersity': mono,
            'in_spec': abs(wavelength_nm - TARGET_WAVELENGTH_NM) < 20,
            'energy_factor': energy_factor,
            'residence_factor': residence_factor,
            'mixing_factor': mixing_factor,
            'uniformity_factor': uniformity,
            'radical_factor': radical_factor,
        }

    # =========================================================================
    #  TERMICA
    # =========================================================================

    def calculate_thermal(self, electrical: Dict,
                          cold_plasma: Optional[Dict] = None,
                          cooling_design: Optional[Dict] = None) -> Dict:
        """Balance termico con enfriamiento activo y verificacion de plasma frio"""
        p = self.params

        # Calor generado por plasma (30% de potencia electrica)
        heat_gen_w = electrical['power_w'] * 0.3

        # Calor removido por flujo de liquido (sin enfriamiento externo)
        cp = 4186  # J/kg/K (agua/purin diluido)
        mass_flow = p.liquid_flow_ml_min / 60.0 * 1e-3  # kg/s
        dT_max = 10.0  # K permitido
        heat_flow_w = mass_flow * cp * dT_max

        # Calor removido por enfriamiento activo
        coolant_flow = p.coolant_flow_ml_min / 60.0 * 1e-3  # kg/s
        heat_coolant_w = coolant_flow * cp * (25 - p.coolant_temp_C)

        total_removal = heat_flow_w + heat_coolant_w
        cooling_ok = total_removal > heat_gen_w

        if cooling_ok:
            actual_dT = heat_gen_w / (mass_flow * cp) if mass_flow > 0 else 100
        else:
            actual_dT = dT_max + (heat_gen_w - total_removal) / max(mass_flow * cp, 1e-6)

        max_temp_C = 25.0 + actual_dT

        # Verificaciones de plasma frio
        Tgas_C = cold_plasma['Tgas_C'] if cold_plasma else max_temp_C
        Tgas_ok = Tgas_C < TGAS_MAX_COLD_C
        T_liquid_outlet = max_temp_C
        T_liquid_ok = T_liquid_outlet < TLIQUID_MAX_OUTLET_C

        # Margen del sistema de enfriamiento
        if cooling_design:
            cooling_margin = cooling_design['cooling_margin_percent']
        else:
            cooling_margin = (total_removal - heat_gen_w) / max(heat_gen_w, 0.01) * 100

        return {
            'heat_generation_w': heat_gen_w,
            'heat_flow_removal_w': heat_flow_w,
            'heat_coolant_removal_w': heat_coolant_w,
            'total_removal_w': total_removal,
            'cooling_sufficient': cooling_ok,
            'temp_rise_C': actual_dT,
            'max_temp_C': max_temp_C,
            'Tgas_C': Tgas_C,
            'Tgas_ok': Tgas_ok,
            'T_liquid_outlet_C': T_liquid_outlet,
            'T_liquid_ok': T_liquid_ok,
            'cooling_margin_percent': cooling_margin,
        }

    # =========================================================================
    #  PURIN
    # =========================================================================

    def calculate_purin_processing(self) -> Dict:
        """Calcula capacidad de procesamiento de purin"""
        p = self.params

        # Volumen de purin diluido procesado
        vol_ml_h = p.liquid_flow_ml_min * 60
        vol_L_h = vol_ml_h / 1000
        vol_L_day = vol_L_h * 24
        vol_L_month = vol_L_day * 30

        # Purin bruto necesario (antes de dilucion)
        dilution_factor = PURIN_TOTAL_SOLIDS_G_L / PURIN_OPTIMAL_DILUTION_G_L
        purin_bruto_L_day = vol_L_day / dilution_factor

        # Carbono disponible para conversion
        carbon_input_g_h = (p.liquid_flow_ml_min * 60 * 1e-3 *
                            PURIN_OPTIMAL_DILUTION_G_L *
                            PURIN_ORGANIC_FRACTION *
                            PURIN_CARBON_FRACTION)

        # Agua necesaria para dilucion
        water_L_day = vol_L_day * (1 - 1.0 / dilution_factor)

        return {
            'diluido_ml_h': vol_ml_h,
            'diluido_L_h': vol_L_h,
            'diluido_L_day': vol_L_day,
            'diluido_L_month': vol_L_month,
            'purin_bruto_L_day': purin_bruto_L_day,
            'purin_bruto_L_month': purin_bruto_L_day * 30,
            'dilution_factor': dilution_factor,
            'water_needed_L_day': water_L_day,
            'carbon_input_g_h': carbon_input_g_h,
            'solids_concentration_g_L': PURIN_OPTIMAL_DILUTION_G_L,
        }

    # =========================================================================
    #  SCORES
    # =========================================================================

    def calculate_scores(self, geometry: Dict, electrical: Dict,
                         production: Dict, thermal: Dict,
                         cold_plasma: Optional[Dict] = None) -> Dict:
        """Scores de diseno con penalizaciones por plasma no-frio"""

        # Eficiencia (0-100)
        eff = 0
        # Produccion vs objetivo (40 pts)
        p = self.params
        prod_ratio = min(2.0, production['production_mg_h'] /
                         max(0.1, p.target_production_mg_h))
        eff += prod_ratio * 20

        # Wavelength en spec (40 pts)
        wl_err = abs(production['wavelength_nm'] - TARGET_WAVELENGTH_NM)
        if wl_err < 20:
            eff += 40
        elif wl_err < 50:
            eff += 40 * (1 - (wl_err - 20) / 30)
        else:
            eff += 10

        # Eficiencia energetica (20 pts)
        specific_e = electrical['power_w'] / max(0.01, production['production_mg_h']) * 3600
        if 1000 < specific_e < 5000:
            eff += 20
        elif specific_e < 10000:
            eff += 10
        eff = min(100, max(0, eff))

        # Factibilidad (0-100)
        feas = 100
        if not electrical['breakdown_ok']:
            feas -= 40
        if not electrical['dielectric_ok']:
            feas -= 25
        if not thermal['cooling_sufficient']:
            feas -= 20
        if thermal['max_temp_C'] > 60:
            feas -= 15
        if production['monodispersity'] < 0.5:
            feas -= 10

        # Penalizaciones por plasma no-frio
        if cold_plasma is not None:
            if cold_plasma['Tgas_C'] > TGAS_MAX_COLD_C:
                feas -= 20  # Gas demasiado caliente
            if not thermal.get('T_liquid_ok', True):
                feas -= 10  # Liquido sobre 50 C
            if cold_plasma['plasma_regime'] == 'termico':
                feas -= 10  # Plasma no es non-thermal

        feas = max(0, feas)

        # Costo (0-100)
        mat = MATERIALS[p.dielectric_material]
        cost = 100
        cost -= mat.cost_relative * 10
        cost -= min(30, electrical['power_w'] * 0.05)
        if p.topology == ScaleTopology.MULTI_CHANNEL:
            cost -= p.n_channels * 1.5  # Complejidad de fabricacion
        if not mat.printable_3d:
            cost -= 15
        cost = max(0, cost)

        return {
            'efficiency_score': eff,
            'feasibility_score': feas,
            'cost_score': cost,
            'overall_score': (eff + feas + cost) / 3,
        }

    # =========================================================================
    #  EVALUACION COMPLETA
    # =========================================================================

    def evaluate(self) -> ScaleupResult:
        """
        Evaluacion completa de un punto de diseno.

        Pipeline: geom -> elec -> cold_plasma -> cooling_design ->
                  prod(cold_plasma) -> therm(cold_plasma, cooling) -> scores
        """
        p = self.params

        geom = self.calculate_geometry()
        elec = self.calculate_electrical(geom)
        cp_data = self._calculate_cold_plasma(elec, geom)
        cool = self._calculate_cooling_design(elec, geom)
        prod = self.calculate_production(geom, elec, cp_data)
        therm = self.calculate_thermal(elec, cp_data, cool)
        purin = self.calculate_purin_processing()
        scores = self.calculate_scores(geom, elec, prod, therm, cp_data)

        # Comparacion con micro
        vs_prod = prod['production_mg_h'] / self.MICRO_PRODUCTION_MG_H
        micro_specific_e = self.MICRO_POWER_W / self.MICRO_PRODUCTION_MG_H * 3600
        scaled_specific_e = elec['power_w'] / max(0.01, prod['production_mg_h']) * 3600
        vs_eff = micro_specific_e / max(1, scaled_specific_e)

        return ScaleupResult(
            topology=p.topology.value,
            plasma_area_cm2=geom['plasma_area_cm2'],
            liquid_volume_ml=geom['liquid_volume_ml'],
            gas_volume_ml=geom['gas_volume_ml'],
            flow_ml_min=p.liquid_flow_ml_min,
            residence_time_s=geom['residence_time_s'],
            reynolds_number=geom['reynolds_number'],
            flow_regime=geom['flow_regime'],
            mixing_quality=geom.get('mixing_quality', 'unknown'),
            power_w=elec['power_w'],
            energy_density_j_ml=elec['energy_density_j_ml'],
            electric_field_v_m=elec['electric_field_v_m'],
            voltage_kv=elec['voltage_kv'],
            breakdown_ok=elec['breakdown_ok'],
            production_mg_h=prod['production_mg_h'],
            concentration_mg_ml=prod['concentration_mg_ml'],
            wavelength_nm=prod['wavelength_nm'],
            size_nm=prod['size_nm'],
            monodispersity=prod['monodispersity'],
            in_spec=prod['in_spec'],
            heat_generation_w=therm['heat_generation_w'],
            heat_removal_w=therm['total_removal_w'],
            cooling_sufficient=therm['cooling_sufficient'],
            max_temp_C=therm['max_temp_C'],
            purin_volume_L_day=purin['diluido_L_day'],
            purin_bruto_L_day=purin['purin_bruto_L_day'],
            efficiency_score=scores['efficiency_score'],
            feasibility_score=scores['feasibility_score'],
            cost_score=scores['cost_score'],
            vs_micro_production_factor=vs_prod,
            vs_micro_efficiency_factor=vs_eff,
            # Plasma frio
            Te_eV=cp_data['Te_eV'],
            Tgas_C=cp_data['Tgas_C'],
            non_thermal_ratio=cp_data['non_thermal_ratio'],
            plasma_regime=cp_data['plasma_regime'],
            radical_OH_cm3=cp_data['radical_OH_cm3'],
            radical_O_cm3=cp_data['radical_O_cm3'],
            radical_O2neg_cm3=cp_data['radical_O2neg_cm3'],
            # Diseno de enfriamiento
            cooling_margin_percent=cool['cooling_margin_percent'],
            coolant_path_length_mm=cool['coolant_path_length_mm'],
            required_coolant_flow_ml_min=cool['required_coolant_flow_ml_min'],
        )

    # =========================================================================
    #  OPTIMIZACION
    # =========================================================================

    def optimize(self, target_mg_h: float = 500.0) -> Dict:
        """
        Busqueda parametrica por topologia.
        Evalua cada topologia con rangos de parametros apropiados.
        """
        results = []
        best = None
        best_score = -1

        # Parametros de busqueda por topologia
        configs = self._generate_search_configs(target_mg_h)

        print(f"\n  Optimizando para {target_mg_h:.0f} mg/h...")
        print(f"  Evaluando {len(configs)} configuraciones...\n")

        for cfg in configs:
            try:
                designer = MillimetricReactorDesigner(cfg)
                r = designer.evaluate()
                results.append(r)

                # Score compuesto
                prod_match = max(0, 1 - abs(r.production_mg_h - target_mg_h) / target_mg_h)
                score = (r.efficiency_score * 0.3 +
                         r.feasibility_score * 0.3 +
                         r.cost_score * 0.15 +
                         prod_match * 100 * 0.15 +
                         (100 if r.in_spec else 50) * 0.1)

                if score > best_score:
                    best_score = score
                    best = r
            except Exception:
                pass

        results.sort(key=lambda r: (r.efficiency_score + r.feasibility_score +
                                     r.cost_score), reverse=True)

        return {
            'all_results': results,
            'best': best,
            'top5': results[:5],
            'target_mg_h': target_mg_h,
        }

    def _generate_search_configs(self, target_mg_h: float) -> List[ScaledReactorParameters]:
        """Genera configuraciones de busqueda para cada topologia"""
        configs = []

        # Campos de catalizador y plasma del diseñador actual
        cat_fields = {
            'catalyst_type': self.params.catalyst_type,
            'catalyst_porosity': self.params.catalyst_porosity,
            'catalyst_surface_area_m2_g': self.params.catalyst_surface_area_m2_g,
            'catalyst_thickness_mm': self.params.catalyst_thickness_mm,
            'catalyst_loading_mg_cm2': self.params.catalyst_loading_mg_cm2,
            'pulse_width_ns': self.params.pulse_width_ns,
            'tio2_barrier': self.params.tio2_barrier,
            'tio2_barrier_phase': self.params.tio2_barrier_phase,
        }

        # FALLING FILM
        for length in [200, 300, 500]:
            for width in [30, 50, 100]:
                for film in [0.3, 0.5, 1.0]:
                    for gap in [2, 3, 5]:
                        for flow in [20, 50, 100, 200]:
                            for voltage in [12, 15, 20]:
                                configs.append(ScaledReactorParameters(
                                    topology=ScaleTopology.FALLING_FILM,
                                    reactor_length_mm=length,
                                    reactor_width_mm=width,
                                    film_thickness_mm=film,
                                    gas_gap_mm=gap,
                                    liquid_flow_ml_min=flow,
                                    voltage_kv=voltage,
                                    target_production_mg_h=target_mg_h,
                                    **cat_fields,
                                ))

        # MULTI-CHANNEL
        for n_ch in [8, 16, 32, 64]:
            for ch_len in [150, 200, 300]:
                for flow in [20, 50, 100, 200]:
                    for voltage in [10, 12, 15]:
                        configs.append(ScaledReactorParameters(
                            topology=ScaleTopology.MULTI_CHANNEL,
                            n_channels=n_ch,
                            channel_length_mm=ch_len,
                            liquid_flow_ml_min=flow,
                            voltage_kv=voltage,
                            target_production_mg_h=target_mg_h,
                            **cat_fields,
                        ))

        # ANNULAR
        for od in [25, 30, 40, 50]:
            for _id in [15, 20, 30]:
                if _id >= od:
                    continue
                for length in [200, 300, 500]:
                    for flow in [20, 50, 100, 200]:
                        for voltage in [12, 15, 20]:
                            configs.append(ScaledReactorParameters(
                                topology=ScaleTopology.ANNULAR,
                                outer_diameter_mm=od,
                                inner_diameter_mm=_id,
                                annular_length_mm=length,
                                liquid_flow_ml_min=flow,
                                voltage_kv=voltage,
                                target_production_mg_h=target_mg_h,
                                **cat_fields,
                            ))

        # BUBBLE COLUMN
        for diam in [30, 50, 80]:
            for height in [150, 200, 300]:
                for bubble in [1.5, 2.0, 3.0]:
                    for flow in [20, 50, 100, 200]:
                        for voltage in [10, 15, 20]:
                            configs.append(ScaledReactorParameters(
                                topology=ScaleTopology.BUBBLE_COLUMN,
                                column_diameter_mm=diam,
                                column_height_mm=height,
                                bubble_size_mm=bubble,
                                liquid_flow_ml_min=flow,
                                voltage_kv=voltage,
                                target_production_mg_h=target_mg_h,
                                **cat_fields,
                            ))

        return configs

    # =========================================================================
    #  PURIN: DISENO DE PLANTA
    # =========================================================================

    def design_purin_plant(self, target_purin_L_day: float) -> Dict:
        """
        Disena planta para procesar un volumen dado de purin bruto/dia.

        Evalua cuantos reactores en paralelo se necesitan por topologia
        y estima costos y produccion total de CQDs.
        """
        results = {}

        for topo in ScaleTopology:
            # Optimizar para esta topologia
            best = None
            best_score = -1

            flows = [20, 50, 100, 200, 500]
            for flow in flows:
                params = ScaledReactorParameters(
                    topology=topo,
                    liquid_flow_ml_min=flow,
                    catalyst_type=self.params.catalyst_type,
                    catalyst_porosity=self.params.catalyst_porosity,
                    catalyst_surface_area_m2_g=self.params.catalyst_surface_area_m2_g,
                    catalyst_thickness_mm=self.params.catalyst_thickness_mm,
                    catalyst_loading_mg_cm2=self.params.catalyst_loading_mg_cm2,
                    pulse_width_ns=self.params.pulse_width_ns,
                    tio2_barrier=self.params.tio2_barrier,
                    tio2_barrier_phase=self.params.tio2_barrier_phase,
                )
                designer = MillimetricReactorDesigner(params)
                r = designer.evaluate()
                purin = designer.calculate_purin_processing()

                score = r.efficiency_score + r.feasibility_score
                if score > best_score and r.production_mg_h > 0:
                    best_score = score
                    best = {
                        'result': r,
                        'purin': purin,
                        'flow': flow,
                    }

            if best is None:
                continue

            purin_per_reactor = best['purin']['purin_bruto_L_day']
            n_reactors = max(1, int(np.ceil(target_purin_L_day /
                                            max(purin_per_reactor, 0.001))))

            total_production = best['result'].production_mg_h * n_reactors
            total_power = best['result'].power_w * n_reactors

            results[topo.value] = {
                'n_reactors': n_reactors,
                'flow_per_reactor_ml_min': best['flow'],
                'production_per_reactor_mg_h': best['result'].production_mg_h,
                'total_production_mg_h': total_production,
                'total_production_g_day': total_production * 24 / 1000,
                'total_power_w': total_power,
                'total_power_kw': total_power / 1000,
                'energy_kwh_day': total_power * 24 / 1000,
                'purin_per_reactor_L_day': purin_per_reactor,
                'total_purin_L_day': purin_per_reactor * n_reactors,
                'wavelength_nm': best['result'].wavelength_nm,
                'in_spec': best['result'].in_spec,
                'monodispersity': best['result'].monodispersity,
                'cooling_ok': best['result'].cooling_sufficient,
                'feasibility': best['result'].feasibility_score,
            }

        return results

    # =========================================================================
    #  REPORTES
    # =========================================================================

    def print_report(self, result: ScaleupResult):
        """Reporte detallado de un reactor escalado"""
        r = result

        print("\n" + "=" * 90)
        print("  REACTOR DBD ESCALA MILIMETRICA")
        print("=" * 90)

        print(f"\n  {'TOPOLOGIA':=<88}")
        print(f"  Tipo:              {r.topology}")
        print(f"  Area de plasma:    {r.plasma_area_cm2:.1f} cm2")
        print(f"  Volumen liquido:   {r.liquid_volume_ml:.2f} mL")
        print(f"  Volumen gas:       {r.gas_volume_ml:.2f} mL")

        print(f"\n  {'FLUJO':=<88}")
        print(f"  Caudal:            {r.flow_ml_min:.1f} mL/min = {r.flow_ml_min*60/1000:.2f} L/h")
        print(f"  Tiempo residencia: {r.residence_time_s:.2f} s")
        print(f"  Reynolds:          {r.reynolds_number:.0f} ({r.flow_regime})")
        print(f"  Calidad mezcla:    {r.mixing_quality}")

        print(f"\n  {'ELECTRICO':=<88}")
        print(f"  Voltaje:           {r.voltage_kv:.0f} kV")
        print(f"  Campo electrico:   {r.electric_field_v_m/1e6:.1f} MV/m "
              f"({'OK' if r.breakdown_ok else 'INSUFICIENTE'})")
        print(f"  Potencia:          {r.power_w:.1f} W")
        print(f"  Densidad energia:  {r.energy_density_j_ml:.0f} J/mL")

        print(f"\n  {'PRODUCCION':=<88}")
        print(f"  Produccion:        {r.production_mg_h:.1f} mg/h = "
              f"{r.production_mg_h*24/1000:.2f} g/dia")
        print(f"  Concentracion:     {r.concentration_mg_ml:.4f} mg/mL")
        print(f"  Longitud de onda:  {r.wavelength_nm:.1f} nm "
              f"{'[EN SPEC]' if r.in_spec else '[FUERA SPEC]'}")
        print(f"  Tamano:            {r.size_nm:.2f} nm")
        print(f"  Monodispersidad:   {r.monodispersity:.2f}")

        print(f"\n  {'TERMICO':=<88}")
        print(f"  Calor generado:    {r.heat_generation_w:.1f} W")
        print(f"  Calor removido:    {r.heat_removal_w:.1f} W "
              f"({'OK' if r.cooling_sufficient else 'INSUFICIENTE'})")
        print(f"  T maxima:          {r.max_temp_C:.1f} C")

        print(f"\n  {'PURIN':=<88}")
        print(f"  Diluido procesado: {r.purin_volume_L_day:.1f} L/dia")
        print(f"  Purin bruto:       {r.purin_bruto_L_day:.2f} L/dia")

        print(f"\n  {'PLASMA FRIO':=<88}")
        print(f"  Te:                {r.Te_eV:.2f} eV")
        print(f"  Tgas:              {r.Tgas_C:.1f} C "
              f"({'OK < 60C' if r.Tgas_C < TGAS_MAX_COLD_C else 'CALIENTE'})")
        print(f"  Ratio non-thermal: {r.non_thermal_ratio:.1f} "
              f"({'OK >> 1' if r.non_thermal_ratio > 10 else 'BAJO'})")
        print(f"  Regimen:           {r.plasma_regime}")
        print(f"  OH* radicales:     {r.radical_OH_cm3:.2e} cm-3")
        print(f"  O* radicales:      {r.radical_O_cm3:.2e} cm-3")
        print(f"  O2- radicales:     {r.radical_O2neg_cm3:.2e} cm-3")

        print(f"\n  {'DISENO ENFRIAMIENTO':=<88}")
        print(f"  Serpentina:        {r.coolant_path_length_mm:.0f} mm")
        print(f"  Flujo min coolant: {r.required_coolant_flow_ml_min:.1f} mL/min")
        print(f"  Margen enfriamto:  {r.cooling_margin_percent:.1f}%")

        print(f"\n  {'CATALIZADOR':=<88}")
        if self.params.catalyst_type:
            geom = self.calculate_geometry()
            elec = self.calculate_electrical(geom)
            cp_data = self._calculate_cold_plasma(elec, geom)
            cat = self._calculate_catalyst_effect(geom, elec, cp_data)
            cat_name = "TiO2 Anatase Poroso" if 'anatase' in self.params.catalyst_type else "TiO2 Rutilo Poroso"
            print(f"  Tipo:              {cat_name}")
            print(f"  Porosidad:         {self.params.catalyst_porosity:.0%}")
            print(f"  Area BET:          {self.params.catalyst_surface_area_m2_g:.0f} m2/g")
            print(f"  Carga:             {self.params.catalyst_loading_mg_cm2:.1f} mg/cm2")
            print(f"  Area catalitica:   {cat['cat_area_m2']*1e4:.2f} cm2")
            print(f"  Factor conversion: {cat['conversion_factor']:.2f}x")
            print(f"  Contacto topo:     {cat['topology_contact']}")
            print(f"  Mejora mono:       +{cat['monodispersity_boost']:.3f}")
            print(f"  Sitios nucleacion: {cat['nucleation_sites']:.2e}")
            print(f"  Fouling:           {cat['fouling_rate_per_h']:.0%}/h (regen {cat['regeneration_temp_C']}C/{cat['regeneration_time_h']:.0f}h)")
        else:
            print(f"  Sin catalizador (usar --catalyst tio2_anatase o tio2_rutile)")

        if self.params.tio2_barrier and self.params.catalyst_type:
            phase = self.params.tio2_barrier_phase
            eps_r = TIO2_EPSILON_R.get(phase, 40.0)
            print(f"\n  {'BARRERA TiO2 ESTRUCTURAL':=<88}")
            print(f"  Fase:              {phase}")
            print(f"  Epsilon_r:         {eps_r:.0f} (vs vidrio ~4.6)")
            print(f"  Reduccion gap eff: {eps_r / 4.6:.1f}x")
            print(f"  Rigidez diel:      {TIO2_DIELECTRIC_STRENGTH_MV_M.get(phase, 4.0):.0f} MV/m")

        print(f"\n  {'vs MICROREACTOR':=<88}")
        print(f"  Factor produccion: {r.vs_micro_production_factor:.1f}x")
        print(f"  Factor eficiencia: {r.vs_micro_efficiency_factor:.2f}x "
              f"(>1 = mas eficiente)")

        print(f"\n  {'SCORES':=<88}")
        print(f"  Eficiencia:        {r.efficiency_score:.0f}/100")
        print(f"  Factibilidad:      {r.feasibility_score:.0f}/100")
        print(f"  Costo:             {r.cost_score:.0f}/100")
        overall = (r.efficiency_score + r.feasibility_score + r.cost_score) / 3
        print(f"  GLOBAL:            {overall:.0f}/100")
        print("=" * 90)

    def print_topology_comparison(self, results: List[ScaleupResult]):
        """Tabla comparativa de topologias"""
        print("\n" + "=" * 110)
        print("  COMPARACION DE TOPOLOGIAS - ESCALA MILIMETRICA")
        print("=" * 110)

        header = (f"  {'Topologia':<18} {'Flow':>8} {'Prod':>10} {'Conc':>8} "
                  f"{'WL':>6} {'Mono':>6} {'Power':>8} {'T_max':>7} "
                  f"{'Purin':>8} {'Eff':>5} {'Feas':>5} {'Cost':>5}")
        print(header)
        units = (f"  {'':18} {'mL/min':>8} {'mg/h':>10} {'mg/mL':>8} "
                 f"{'nm':>6} {'':>6} {'W':>8} {'C':>7} "
                 f"{'L/dia':>8} {'':>5} {'':>5} {'':>5}")
        print(units)
        print(f"  {'-'*108}")

        for r in results:
            line = (f"  {r.topology:<18} {r.flow_ml_min:>8.0f} "
                    f"{r.production_mg_h:>10.1f} {r.concentration_mg_ml:>8.4f} "
                    f"{r.wavelength_nm:>6.0f} {r.monodispersity:>6.2f} "
                    f"{r.power_w:>8.1f} {r.max_temp_C:>7.1f} "
                    f"{r.purin_bruto_L_day:>8.1f} "
                    f"{r.efficiency_score:>5.0f} {r.feasibility_score:>5.0f} "
                    f"{r.cost_score:>5.0f}")
            print(line)

        print(f"  {'-'*108}")

    def print_purin_plant_report(self, target_L_day: float, plant: Dict):
        """Reporte de planta de procesamiento de purin"""
        print("\n" + "=" * 90)
        print(f"  PLANTA DE PROCESAMIENTO DE PURIN: {target_L_day:.0f} L/dia")
        print("=" * 90)

        print(f"\n  Purin bruto objetivo:  {target_L_day:.0f} L/dia")
        print(f"  Dilucion:              {PURIN_TOTAL_SOLIDS_G_L:.0f} g/L -> "
              f"{PURIN_OPTIMAL_DILUTION_G_L:.0f} g/L "
              f"({PURIN_TOTAL_SOLIDS_G_L/PURIN_OPTIMAL_DILUTION_G_L:.0f}x)")
        print()

        header = (f"  {'Topologia':<18} {'React':>7} {'Prod':>10} {'CQD/dia':>10} "
                  f"{'Power':>8} {'kWh/dia':>9} {'WL':>6} {'Spec':>5} {'Feas':>5}")
        print(header)
        print(f"  {'-'*85}")

        for topo, data in plant.items():
            spec = 'SI' if data['in_spec'] else 'NO'
            line = (f"  {topo:<18} {data['n_reactors']:>7d} "
                    f"{data['total_production_mg_h']:>10.0f} "
                    f"{data['total_production_g_day']:>9.2f}g "
                    f"{data['total_power_kw']:>7.2f}kW "
                    f"{data['energy_kwh_day']:>9.1f} "
                    f"{data['wavelength_nm']:>6.0f} {spec:>5} "
                    f"{data['feasibility']:>5.0f}")
            print(line)

        print(f"\n  {'ANALISIS':=<88}")

        # Encontrar mejor opcion
        best_topo = None
        best_score = -1
        for topo, data in plant.items():
            score = (data['total_production_g_day'] * 10 +
                     data['feasibility'] -
                     data['n_reactors'] * 5 -
                     data['energy_kwh_day'] * 0.5)
            if score > best_score:
                best_score = score
                best_topo = topo

        if best_topo:
            d = plant[best_topo]
            print(f"\n  RECOMENDACION: {best_topo}")
            print(f"    {d['n_reactors']} reactores en paralelo")
            print(f"    Produccion total: {d['total_production_g_day']:.2f} g CQDs/dia")
            print(f"    Consumo: {d['energy_kwh_day']:.1f} kWh/dia")
            print(f"    Costo energia (~$0.12/kWh): ${d['energy_kwh_day']*0.12:.2f}/dia")
            print(f"    Valor CQDs (~$50/mg): ${d['total_production_g_day']*1000*50:.0f}/dia")
            print()

            ratio = d['total_production_g_day'] * 1000 * 50 / max(d['energy_kwh_day'] * 0.12, 0.01)
            print(f"    Ratio valor/costo energia: {ratio:.0f}x")

        # Diagrama de planta
        print(f"\n  {'DIAGRAMA DE PLANTA':=<88}")
        if best_topo:
            d = plant[best_topo]
            n = d['n_reactors']
            print(f"""
                         PURIN BRUTO
                             |
                     [DILUCION {PURIN_TOTAL_SOLIDS_G_L/PURIN_OPTIMAL_DILUTION_G_L:.0f}x]
                             |
                     [PREFILTRO GRUESO]
                             |
                    +--------+--------+
                    |        |        |
                 [DBD-1]  [DBD-2] ... [DBD-{n}]    <- {n} reactores {best_topo}
                    |        |        |
                    +--------+--------+
                             |
                     [CASCADA INLINE]               <- clasificador continuo
                             |
                    +--------+--------+
                    |                 |
                 PRODUCTO          RESIDUO
              {d['total_production_g_day']:.1f} g CQDs/dia     |
              99%+ pureza       [FERTILIZANTE]
""")

        print("=" * 90)

    # =========================================================================
    #  LEYES DE ESCALA
    # =========================================================================

    def print_scaling_laws(self):
        """Imprime las leyes de escalado del microreactor al mm"""
        print("\n" + "=" * 90)
        print("  LEYES DE ESCALADO: MICROFLUIDICO -> MILIMETRICO")
        print("=" * 90)

        print(f"""
  PARAMETRO              MICRO (actual)    MM (escalado)     LEY DE ESCALA
  {'─'*80}
  Canal (ancho)          2 mm              10-100 mm         Lineal
  Canal (alto/gap)       0.5 mm            2-10 mm           Lineal
  Profundidad liq.       0.3 mm            0.5-5 mm          Lineal
  Flujo                  5 mL/min          20-500 mL/min     ~Lineal con area
  Tiempo residencia      13 s              10-30 s           Se mantiene optimo
  Area plasma            3 cm2             15-500 cm2        L * W
  Potencia               1.8 W             5-200 W           ~Lineal con area
  Densidad energia       ~400 J/mL         300-600 J/mL      Se mantiene optimo
  Voltaje                10 kV             12-20 kV          ~Lineal con gap
  Produccion             50 mg/h           100-2000 mg/h     ~Lineal con flujo
  Reynolds               <100              100-5000          Puede ser turbulento
  Mezcla (difusion)      <1 s              10-100 s          ~h^2/D (cuadratico)
  Calor generado         0.5 W             2-60 W            ~Lineal con potencia
  Enfriamiento           Pasivo            Activo requerido  Escala con potencia

  DESAFIOS CRITICOS AL ESCALAR:
  {'─'*80}
  1. MEZCLA: El tiempo de difusion escala con h^2. Un canal de 5mm necesita
     25x mas tiempo que uno de 1mm para mezclar por difusion pura.
     -> Solucion: turbulencia, obstaculos, pelicula delgada, burbujas

  2. UNIFORMIDAD DE PLASMA: El campo electrico no es uniforme en gaps >2mm.
     Filamentos de plasma en vez de descarga difusa.
     -> Solucion: electrodos segmentados, frecuencia alta, burbujas

  3. TERMICO: Mas potencia = mas calor. El ratio area/volumen cae al escalar.
     -> Solucion: enfriamiento activo obligatorio, pelicula delgada

  4. MONODISPERSIDAD: Peor mezcla + plasma no uniforme = distribucion ancha.
     -> Solucion: multi-canal (numbering up) conserva la fisica micro

  RECOMENDACION DE TOPOLOGIA POR OBJETIVO:
  {'─'*80}
  Objetivo                     Topologia recomendada
  Maxima calidad (mono>0.9)    MULTI_CHANNEL (conserva fisica micro)
  Maximo throughput            FALLING_FILM (gran area, film delgado)
  Maxima eficiencia            BUBBLE_COLUMN (contacto 3D, buena mezcla)
  Minimo costo                 ANNULAR (simple, comercial)
  Procesamiento de purin       FALLING_FILM o MULTI_CHANNEL
""")
        print("=" * 90)


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Anexo: Reactor DBD en escala milimetrica')
    parser.add_argument('--topology', type=str, default=None,
                        choices=[t.value for t in ScaleTopology],
                        help='Topologia especifica')
    parser.add_argument('--flow', type=float, default=50.0,
                        help='Flujo en mL/min (default: 50)')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimizar para target de produccion')
    parser.add_argument('--target', type=float, default=500.0,
                        help='Produccion objetivo en mg/h (default: 500)')
    parser.add_argument('--purin', action='store_true',
                        help='Disenar planta para procesamiento de purin')
    parser.add_argument('--volume', type=float, default=100.0,
                        help='Volumen de purin bruto en L/dia (default: 100)')
    parser.add_argument('--compare', action='store_true',
                        help='Comparar todas las topologias')
    parser.add_argument('--scaling-laws', action='store_true',
                        help='Mostrar leyes de escalado')
    parser.add_argument('--catalyst', type=str, default=None,
                        choices=['tio2_anatase', 'tio2_rutile'],
                        help='Tipo de catalizador TiO2 (default: ninguno)')
    parser.add_argument('--porosity', type=float, default=0.6,
                        help='Porosidad del catalizador 0-1 (default: 0.6)')
    parser.add_argument('--pulse-width', type=float, default=100.0,
                        help='Ancho de pulso en ns (default: 100, <500 = frio)')
    parser.add_argument('--tio2-barrier', action='store_true',
                        help='Usar TiO2 como barrera dielectrica estructural')
    parser.add_argument('--tio2-phase', type=str, default='anatase',
                        choices=['anatase', 'rutile'],
                        help='Fase del TiO2 barrera (default: anatase)')

    args = parser.parse_args()

    if args.scaling_laws:
        designer = MillimetricReactorDesigner()
        designer.print_scaling_laws()
        return

    if args.purin:
        params = ScaledReactorParameters(
            catalyst_type=args.catalyst,
            catalyst_porosity=args.porosity if args.catalyst else 0.6,
            pulse_width_ns=args.pulse_width,
            tio2_barrier=args.tio2_barrier,
            tio2_barrier_phase=args.tio2_phase,
        )
        designer = MillimetricReactorDesigner(params)
        plant = designer.design_purin_plant(args.volume)
        designer.print_purin_plant_report(args.volume, plant)
        designer.print_scaling_laws()
        return

    if args.optimize:
        params = ScaledReactorParameters(
            catalyst_type=args.catalyst,
            catalyst_porosity=args.porosity if args.catalyst else 0.6,
            pulse_width_ns=args.pulse_width,
            tio2_barrier=args.tio2_barrier,
            tio2_barrier_phase=args.tio2_phase,
        )
        designer = MillimetricReactorDesigner(params)
        opt = designer.optimize(args.target)

        if opt['top5']:
            designer.print_topology_comparison(opt['top5'])
            print()
            if opt['best']:
                designer.print_report(opt['best'])
        designer.print_scaling_laws()
        return

    if args.compare or args.topology is None:
        # Comparar todas las topologias al mismo flujo
        results = []
        designers = []
        for topo in ScaleTopology:
            params = ScaledReactorParameters(
                topology=topo,
                liquid_flow_ml_min=args.flow,
                catalyst_type=args.catalyst,
                catalyst_porosity=args.porosity if args.catalyst else 0.6,
                pulse_width_ns=args.pulse_width,
                tio2_barrier=args.tio2_barrier,
                tio2_barrier_phase=args.tio2_phase,
            )
            d = MillimetricReactorDesigner(params)
            results.append(d.evaluate())
            designers.append(d)

        designers[0].print_topology_comparison(results)

        # Reporte del mejor
        best_idx = max(range(len(results)),
                       key=lambda i: results[i].efficiency_score + results[i].feasibility_score)
        designers[best_idx].print_report(results[best_idx])
        designers[0].print_scaling_laws()
        return

    # Topologia especifica
    topo = ScaleTopology(args.topology)
    params = ScaledReactorParameters(
        topology=topo,
        liquid_flow_ml_min=args.flow,
        catalyst_type=args.catalyst,
        catalyst_porosity=args.porosity if args.catalyst else 0.6,
        pulse_width_ns=args.pulse_width,
        tio2_barrier=args.tio2_barrier,
        tio2_barrier_phase=args.tio2_phase,
    )
    designer = MillimetricReactorDesigner(params)
    result = designer.evaluate()
    designer.print_report(result)


if __name__ == '__main__':
    main()
