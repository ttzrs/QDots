#!/usr/bin/env python3
"""
===============================================================================
  DISENO PARAMETRICO DEL CLASIFICADOR POR FLOTABILIDAD OPTICA PARA CQDs
  Separacion por excitacion selectiva: fuerza fotoforetica + fototermica
===============================================================================

  Este modulo calcula los parametros optimos de construccion para un
  clasificador post-reactor basado en flotabilidad optica con 3 zonas.

  Principio de operacion:
    - Fluido post-reactor entra horizontalmente al clasificador
    - 3 zonas separadas por membranas-barrera de poro grande (>10 um)
    - Cada zona tiene un array de LEDs a longitud de onda especifica
    - Los QDots que absorben esa lambda experimentan:
      * Fuerza de radiacion (radiation pressure): F = n*P*sigma_abs/c
      * Efecto fototermico: absorcion -> calentamiento local -> conveccion
      * Fotoforesis: particula calentada asimetricamente se desplaza
    - Particulas no-QDot NO absorben -> NO experimentan fuerza -> sedimentan
    - Puertos de coleccion arriba (QDots excitados) y abajo (debris)

  Las membranas-barrera solo separan zonas de diferente densidad/corriente,
  con poros de ~50 um que NO generan caida de presion significativa.

  Variables de optimizacion:
    - Potencia LED por zona
    - Altura y longitud de zona
    - Caudal y tiempo de residencia
    - Longitudes de onda de excitacion

  Basado en:
    - Fuerza de radiacion optica (radiation pressure)
    - Modelo fototermico (calentamiento local + Stokes termico)
    - Sedimentacion de Stokes para particulas densas
    - Difusion browniana como competencia
    - Modelo de confinamiento cuantico validado por VQE

  USO:
    python classifier_design.py --design
    python classifier_design.py --optimize
    python classifier_design.py --export-cad classifier_cad.json
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json


# ===============================================================================
#  CONSTANTES FISICAS Y DE SIMULACION
# ===============================================================================

# Constantes fundamentales
BOLTZMANN = 1.381e-23          # Constante de Boltzmann (J/K)
PLANCK = 6.626e-34             # Constante de Planck (J.s)
SPEED_OF_LIGHT = 3e8           # Velocidad de la luz (m/s)
GRAVITY = 9.81                 # Aceleracion gravitatoria (m/s^2)
VISCOSITY_WATER = 0.001        # Viscosidad del agua a 25C (Pa.s)
DENSITY_WATER = 998            # Densidad del agua a 25C (kg/m^3)
THERMAL_CONDUCTIVITY_WATER = 0.6  # Conductividad termica del agua (W/m/K)
REFRACTIVE_INDEX_WATER = 1.33     # Indice de refraccion del agua

# Propiedades del sustrato de oro (modo opto-termico)
THERMAL_CONDUCTIVITY_GOLD = 317.0   # Conductividad termica del oro (W/m/K)
REFRACTIVE_INDEX_CQD = 1.7          # Indice de refraccion estimado de CQDs

# Propiedades de CQDs
DENSITY_CQD = 1800             # Densidad tipica de carbon quantum dots (kg/m^3)
ABSORPTION_CROSS_SECTION = 1e-20  # Seccion eficaz de absorcion a resonancia (m^2)
QUANTUM_YIELD_FLUORESCENT = 0.40  # Rendimiento cuantico fluorescente
QUANTUM_YIELD_THERMAL = 0.60      # Fraccion de energia convertida en calor (1 - QY_fl)

# Modelo de confinamiento cuantico (VQE + literatura)
E_BULK_EV = 1.50               # Gap del grafeno dopado N (eV)
A_CONFINEMENT = 7.26           # Constante de confinamiento (eV*nm^2)
EV_TO_NM = 1240                # Conversion energia-longitud de onda


# ===============================================================================
#  MODOS DE EXCITACION
# ===============================================================================

class ExcitationMode(Enum):
    """Modo de excitacion del clasificador"""
    LED = "led"                    # Array de LEDs difusos (original)
    LASER_DIRECT = "laser_direct"  # Laser focalizado sin sustrato
    OPTOTHERMAL = "optothermal"    # Laser + sustrato absorbente (oro)


# ===============================================================================
#  ESTRUCTURAS DE DATOS
# ===============================================================================

@dataclass
class BarrierSpec:
    """Especificacion de una membrana-barrera (solo separa zonas)"""
    name: str
    pore_diameter_um: float = 50.0    # Poro grande (~50 um)
    material: str = "mesh_steel"       # Malla de acero o policarbonato
    max_pressure_bar: float = 10.0     # Sin problema de presion


@dataclass
class ZoneSpec:
    """Especificacion de una zona de excitacion"""
    zone_number: int
    target_wavelength_nm: float       # Lambda de excitacion del LED
    target_size_range_nm: tuple       # (min, max) tamano de QDot capturado
    target_emission_range_nm: tuple   # (min, max) emision esperada del QDot
    led_power_mw: float               # Potencia del array LED
    zone_length_mm: float             # Longitud de la zona (eje X)
    zone_height_mm: float             # Altura (eje Z, importante para separacion)
    residence_time_s: float           # Tiempo en zona
    expected_content: str             # Descripcion del contenido


@dataclass
class ClassifierParameters:
    """Parametros completos del clasificador por flotabilidad optica"""
    # Modo de excitacion
    excitation_mode: ExcitationMode = ExcitationMode.OPTOTHERMAL

    # Topologia
    n_zones: int = 3                   # 3 zonas: verde, UV-azul, UV
    zone_length_mm: float = 40.0       # Longitud por zona (eje X)
    zone_height_mm: float = 15.0       # Altura (importante para sedimentacion!)
    zone_width_mm: float = 10.0        # Profundidad (eje Y)
    channel_height_mm: float = 2.0     # Canal de entrada
    wall_thickness_mm: float = 2.0     # Espesor de paredes

    # Barreras entre zonas
    barrier_pore_um: float = 50.0      # Poro grande (sin caida de presion)
    barrier_diameter_mm: float = 13.0  # Diametro de la barrera

    # LEDs por zona (modo LED)
    led_wavelengths_nm: list = field(default_factory=lambda: [520.0, 405.0, 365.0])
    led_power_mw: float = 200.0        # Potencia por LED (mW)
    led_array_count: int = 4           # LEDs por zona (arriba)

    # Parametros laser (modos LASER_DIRECT y OPTOTHERMAL)
    laser_power_mw: float = 500.0      # Potencia laser (mW)
    laser_wavelengths_nm: list = field(default_factory=lambda: [520.0, 405.0, 365.0])
    beam_waist_um: float = 10.0        # Radio del beam focalizado (um)

    # Parametros opto-termicos (modo OPTOTHERMAL)
    substrate_type: str = "gold_film"          # Tipo de sustrato absorbente
    substrate_thickness_nm: float = 50.0       # Espesor pelicula de oro
    substrate_absorptance: float = 0.40        # Fraccion absorbida por sustrato
    soret_coefficient: float = 0.05            # S_T tipico (K^-1) para CQDs

    # Geometria adaptada para microcanales (modos laser)
    channel_width_um: float = 200.0    # Ancho del microcanal (um)
    channel_depth_um: float = 100.0    # Profundidad del microcanal

    # Coleccion
    top_port_diameter_mm: float = 2.0  # Puerto superior (QDots)
    bottom_port_diameter_mm: float = 2.0  # Puerto inferior (debris)

    # Flujo
    flow_rate_ml_min: float = 2.0      # Mas lento para dar tiempo
    temperature_c: float = 25.0        # Temperatura de operacion

    # Conexiones
    inlet_diameter_mm: float = 2.0
    outlet_diameter_mm: float = 2.0
    waste_diameter_mm: float = 2.0

    # Material del cuerpo
    body_material: str = "ceramic_resin"

    # Concentracion de CQDs (mg/mL en solucion post-reactor)
    cqd_concentration_mg_ml: float = 0.1


# Zonas por defecto
DEFAULT_ZONES_CONFIG = [
    {
        'zone_number': 1,
        'led_wavelength_nm': 520.0,      # Verde: excita QDots grandes
        'target_size_range_nm': (3.5, 5.0),
        'target_emission_range_nm': (550.0, 700.0),  # Emiten rojo
        'expected_content': 'qdots_rojos',
    },
    {
        'zone_number': 2,
        'led_wavelength_nm': 405.0,      # UV-azul: excita QDots medios
        'target_size_range_nm': (2.5, 3.5),
        'target_emission_range_nm': (450.0, 550.0),  # Emiten azul-verde
        'expected_content': 'qdots_azules',
    },
    {
        'zone_number': 3,
        'led_wavelength_nm': 365.0,      # UV: excita QDots pequenos
        'target_size_range_nm': (1.5, 2.5),
        'target_emission_range_nm': (350.0, 450.0),  # Emiten UV-violeta
        'expected_content': 'qdots_uv',
    },
]


@dataclass
class ClassifierOutput:
    """Salida del diseno del clasificador"""
    # Dimensiones globales
    total_length_mm: float = 0.0
    total_width_mm: float = 0.0
    total_height_mm: float = 0.0

    # Zonas
    zones: list = field(default_factory=list)

    # Hidraulica
    total_volume_ml: float = 0.0
    residence_time_s: float = 0.0

    # Rendimiento de separacion
    avg_zone_efficiency: float = 0.0
    avg_selectivity: float = 0.0
    overall_recovery: float = 0.0
    overall_purity: float = 0.0

    # Puntuaciones de diseno
    efficiency_score: float = 0.0
    cost_score: float = 0.0
    feasibility_score: float = 0.0


# ===============================================================================
#  FUNCIONES AUXILIARES
# ===============================================================================

def size_to_wavelength(diameter_nm: float) -> float:
    """Convierte diametro de CQD a longitud de onda de emision (nm)"""
    if diameter_nm <= 0:
        return 0.0
    gap_ev = E_BULK_EV + A_CONFINEMENT / (diameter_nm ** 2)
    return EV_TO_NM / gap_ev


def wavelength_to_size(wavelength_nm: float) -> float:
    """Convierte longitud de onda de emision a diametro de CQD (nm)"""
    if wavelength_nm <= 0:
        return 0.0
    gap_ev = EV_TO_NM / wavelength_nm
    if gap_ev <= E_BULK_EV:
        return float('inf')
    return np.sqrt(A_CONFINEMENT / (gap_ev - E_BULK_EV))


# ===============================================================================
#  MOTOR DE DISENO DEL CLASIFICADOR
# ===============================================================================

class ClassifierDesigner:
    """
    Motor de diseno parametrico del clasificador por flotabilidad optica.
    Calcula fuerzas opticas, velocidades fototermicas y eficiencias de
    separacion para cada zona de excitacion.
    """

    def __init__(self, params: Optional[ClassifierParameters] = None,
                 zones_config: Optional[List[Dict]] = None):
        self.params = params or ClassifierParameters()
        self.zones_config = zones_config or DEFAULT_ZONES_CONFIG
        self.output = ClassifierOutput()

    def calculate_zone_geometry(self) -> Dict:
        """
        Calcula parametros geometricos del separador completo.
        Layout: zonas rectangulares en serie con barreras entre ellas.
        """
        p = self.params

        # Longitud total: n zonas + barreras + paredes
        barrier_thickness_mm = 1.0  # Espesor de barrera/malla
        total_length_mm = (p.n_zones * p.zone_length_mm +
                           (p.n_zones - 1) * barrier_thickness_mm +
                           2 * p.wall_thickness_mm)

        # Ancho total: zona + paredes
        total_width_mm = p.zone_width_mm + 2 * p.wall_thickness_mm

        # Altura total: zona + canal entrada + paredes + espacio LED
        led_space_mm = 5.0  # Espacio para el array de LEDs arriba
        total_height_mm = (p.wall_thickness_mm +     # base
                           p.zone_height_mm +        # zona
                           p.wall_thickness_mm +     # techo
                           led_space_mm)             # LEDs

        # Volumen por zona
        zone_volume_mm3 = p.zone_length_mm * p.zone_width_mm * p.zone_height_mm
        zone_volume_ml = zone_volume_mm3 / 1000.0

        # Volumen total interno
        total_volume_ml = p.n_zones * zone_volume_ml

        # Caudal en m^3/s
        Q_m3s = p.flow_rate_ml_min / 60.0 * 1e-6

        # Velocidad del flujo en el canal de entrada
        channel_area_m2 = (p.zone_width_mm * 1e-3) * (p.channel_height_mm * 1e-3)
        channel_velocity_m_s = Q_m3s / channel_area_m2

        # Tiempo de residencia total
        total_volume_m3 = total_volume_ml * 1e-6
        residence_time_s = total_volume_m3 / Q_m3s

        # Tiempo de residencia por zona
        zone_residence_time_s = residence_time_s / p.n_zones

        # Posiciones de cada zona (centro X)
        zone_positions = []
        for i in range(p.n_zones):
            x = (p.wall_thickness_mm + p.zone_length_mm / 2 +
                 i * (p.zone_length_mm + barrier_thickness_mm))
            zone_positions.append({
                'zone': i + 1,
                'x_center_mm': x,
                'y_center_mm': total_width_mm / 2,
                'z_center_mm': p.wall_thickness_mm + p.zone_height_mm / 2,
            })

        return {
            'total_length_mm': total_length_mm,
            'total_width_mm': total_width_mm,
            'total_height_mm': total_height_mm,
            'zone_volume_ml': zone_volume_ml,
            'total_volume_ml': total_volume_ml,
            'residence_time_s': residence_time_s,
            'zone_residence_time_s': zone_residence_time_s,
            'channel_velocity_m_s': channel_velocity_m_s,
            'channel_velocity_mm_s': channel_velocity_m_s * 1e3,
            'zone_positions': zone_positions,
            'n_zones': p.n_zones,
            'barrier_thickness_mm': barrier_thickness_mm,
            'led_space_mm': led_space_mm,
        }

    def calculate_radiation_force(self, led_power_mw: float,
                                  particle_diameter_nm: float,
                                  n_leds: int = 4,
                                  zone_area_mm2: float = None) -> Dict:
        """
        Calcula la fuerza de radiacion (radiation pressure) sobre un QDot.

        F_rad = I * sigma_abs / c

        donde I = P_total / A es la irradiancia en la zona.
        """
        p = self.params
        if zone_area_mm2 is None:
            zone_area_mm2 = p.zone_length_mm * p.zone_width_mm

        # Potencia total del array LED
        P_total_W = led_power_mw * 1e-3 * n_leds

        # Area iluminada (m^2)
        A_m2 = zone_area_mm2 * 1e-6

        # Irradiancia (W/m^2)
        irradiance_W_m2 = P_total_W / A_m2

        # Seccion eficaz de absorcion (depende del tamano)
        # sigma_abs escala aprox. con d^3 para nanoparticulas pequenas
        d_ref_nm = 3.0  # Diametro de referencia
        sigma_abs = ABSORPTION_CROSS_SECTION * (particle_diameter_nm / d_ref_nm) ** 3

        # Fuerza de radiacion
        F_rad_N = irradiance_W_m2 * sigma_abs / SPEED_OF_LIGHT

        # Presion de radiacion equivalente
        P_rad_Pa = irradiance_W_m2 / SPEED_OF_LIGHT

        return {
            'P_total_W': P_total_W,
            'irradiance_W_m2': irradiance_W_m2,
            'sigma_abs_m2': sigma_abs,
            'F_radiation_N': F_rad_N,
            'F_radiation_fN': F_rad_N * 1e15,
            'P_radiation_Pa': P_rad_Pa,
        }

    def calculate_photothermal_velocity(self, led_power_mw: float,
                                        particle_diameter_nm: float,
                                        n_leds: int = 4,
                                        zone_area_mm2: float = None) -> Dict:
        """
        Calcula velocidad fototermica del QDot.

        1. Potencia absorbida por particula: P_abs = I * sigma_abs * QY_thermal
        2. Calentamiento local: dT = P_abs / (4*pi*k_water*r)
        3. Velocidad termica (Stokes termico):
           v_thermal = (2/9) * (dT/T) * (r^2 * g * drho) / mu
        """
        p = self.params
        T_K = p.temperature_c + 273.15

        if zone_area_mm2 is None:
            zone_area_mm2 = p.zone_length_mm * p.zone_width_mm

        # Irradiancia
        P_total_W = led_power_mw * 1e-3 * n_leds
        A_m2 = zone_area_mm2 * 1e-6
        irradiance = P_total_W / A_m2

        # Seccion eficaz (escala con d^3)
        d_ref_nm = 3.0
        sigma_abs = ABSORPTION_CROSS_SECTION * (particle_diameter_nm / d_ref_nm) ** 3

        # Potencia absorbida convertida en calor
        P_absorbed_W = irradiance * sigma_abs * QUANTUM_YIELD_THERMAL

        # Radio de la particula
        r_m = particle_diameter_nm * 1e-9 / 2.0

        # Calentamiento local (modelo esferico)
        # dT = P_abs / (4 * pi * k_water * r)
        dT_K = P_absorbed_W / (4.0 * np.pi * THERMAL_CONDUCTIVITY_WATER * r_m)

        # Diferencia de densidad
        delta_rho = DENSITY_CQD - DENSITY_WATER

        # Velocidad termica ascendente (Stokes termico modificado)
        # La conveccion local alrededor de la particula calentada
        # reduce la densidad efectiva del fluido circundante
        # v_up = (2/9) * (dT/T) * (r^2 * g * delta_rho) / mu
        v_thermal_m_s = (2.0 / 9.0) * (dT_K / T_K) * (r_m ** 2 * GRAVITY * delta_rho) / VISCOSITY_WATER

        # Velocidad fotoforetica adicional
        # Para particulas asimetricas, hay un componente fotoforetico
        # v_photophoretic ~ factor * dT * r / mu  (simplificado)
        # Factor tipico para nanoparticulas en agua: ~1e-4
        photophoretic_factor = 1e-4
        v_photophoretic_m_s = photophoretic_factor * dT_K * r_m / VISCOSITY_WATER

        # Velocidad total ascendente (termica + fotoforetica)
        v_up_total_m_s = v_thermal_m_s + v_photophoretic_m_s

        return {
            'P_absorbed_W': P_absorbed_W,
            'P_absorbed_fW': P_absorbed_W * 1e15,
            'dT_local_K': dT_K,
            'dT_local_mK': dT_K * 1e3,
            'v_thermal_m_s': v_thermal_m_s,
            'v_thermal_um_s': v_thermal_m_s * 1e6,
            'v_photophoretic_m_s': v_photophoretic_m_s,
            'v_photophoretic_um_s': v_photophoretic_m_s * 1e6,
            'v_up_total_m_s': v_up_total_m_s,
            'v_up_total_um_s': v_up_total_m_s * 1e6,
        }

    def calculate_sedimentation(self, particle_diameter_nm: float,
                                particle_density: float = DENSITY_CQD) -> Dict:
        """
        Calcula velocidad de sedimentacion Stokes para particulas no-excitadas.

        v_sed = (2/9) * r^2 * (rho_p - rho_f) * g / mu

        Las particulas debris (no-QDot) y QDots no-excitados sedimentan.
        """
        r_m = particle_diameter_nm * 1e-9 / 2.0
        delta_rho = particle_density - DENSITY_WATER

        # Velocidad de sedimentacion Stokes
        v_sed_m_s = (2.0 / 9.0) * r_m ** 2 * delta_rho * GRAVITY / VISCOSITY_WATER

        return {
            'v_sedimentation_m_s': v_sed_m_s,
            'v_sedimentation_um_s': v_sed_m_s * 1e6,
            'v_sedimentation_nm_s': v_sed_m_s * 1e9,
            'particle_radius_nm': particle_diameter_nm / 2.0,
            'delta_rho_kg_m3': delta_rho,
        }

    def calculate_brownian_diffusion(self, particle_diameter_nm: float) -> Dict:
        """
        Calcula coeficiente de difusion browniana (competencia con fuerzas opticas).

        D = kT / (3*pi*mu*d)

        Para nanoparticulas de 2-5nm, D ~ 10^-10 m^2/s.
        """
        p = self.params
        T_K = p.temperature_c + 273.15
        d_m = particle_diameter_nm * 1e-9

        # Difusion de Stokes-Einstein
        D_m2_s = BOLTZMANN * T_K / (3.0 * np.pi * VISCOSITY_WATER * d_m)

        # Desplazamiento RMS en un tiempo dado
        # x_rms = sqrt(2 * D * t)
        # Para 1 segundo:
        x_rms_1s_m = np.sqrt(2.0 * D_m2_s * 1.0)

        return {
            'D_m2_s': D_m2_s,
            'D_um2_s': D_m2_s * 1e12,
            'x_rms_1s_um': x_rms_1s_m * 1e6,
            'x_rms_1s_mm': x_rms_1s_m * 1e3,
        }

    # ===========================================================================
    #  METODOS PARA MODO LASER Y OPTO-TERMICO
    # ===========================================================================

    def calculate_laser_irradiance(self, laser_power_mw: float = None,
                                   beam_waist_um: float = None) -> Dict:
        """
        Calcula irradiancia pico de un beam gaussiano focalizado.

        I_peak = 2 * P / (pi * w0^2)
        """
        p = self.params
        P_W = (laser_power_mw or p.laser_power_mw) * 1e-3
        w0_m = (beam_waist_um or p.beam_waist_um) * 1e-6

        I_peak = 2.0 * P_W / (np.pi * w0_m ** 2)

        return {
            'P_laser_W': P_W,
            'beam_waist_m': w0_m,
            'beam_waist_um': w0_m * 1e6,
            'I_peak_W_m2': I_peak,
            'I_peak_kW_cm2': I_peak * 1e-7,  # conversion a kW/cm2
            'beam_area_um2': np.pi * (w0_m * 1e6) ** 2,
        }

    def calculate_gradient_force(self, particle_diameter_nm: float,
                                 laser_power_mw: float = None,
                                 beam_waist_um: float = None) -> Dict:
        """
        Calcula fuerza de gradiente optico (regimen Rayleigh, d << lambda).

        F_grad = (2*pi*n_m*r^3/c) * ((m^2-1)/(m^2+2)) * grad(I)
        grad(I) ~ I_peak / w0  para beam gaussiano

        Tambien calcula F_rad = I * sigma_abs / c  (presion de radiacion).
        """
        p = self.params
        P_W = (laser_power_mw or p.laser_power_mw) * 1e-3
        w0_m = (beam_waist_um or p.beam_waist_um) * 1e-6
        r_m = particle_diameter_nm * 1e-9 / 2.0

        # Irradiancia pico
        I_peak = 2.0 * P_W / (np.pi * w0_m ** 2)

        # Gradiente de irradiancia (gaussiano)
        grad_I = I_peak / w0_m

        # Indice de refraccion relativo
        n_m = REFRACTIVE_INDEX_WATER
        n_p = REFRACTIVE_INDEX_CQD
        m = n_p / n_m
        clausius_mossotti = (m ** 2 - 1) / (m ** 2 + 2)

        # Fuerza de gradiente (Rayleigh)
        F_grad = (2.0 * np.pi * n_m * r_m ** 3 / SPEED_OF_LIGHT) * clausius_mossotti * grad_I

        # Seccion eficaz de absorcion (escala con d^3)
        d_ref_nm = 3.0
        sigma_abs = ABSORPTION_CROSS_SECTION * (particle_diameter_nm / d_ref_nm) ** 3

        # Fuerza de radiacion (scattering + absorcion)
        F_rad = I_peak * sigma_abs / SPEED_OF_LIGHT

        # Velocidades resultantes
        stokes_drag = 6.0 * np.pi * VISCOSITY_WATER * r_m
        v_grad = abs(F_grad) / stokes_drag if stokes_drag > 0 else 0
        v_rad = F_rad / stokes_drag if stokes_drag > 0 else 0

        return {
            'I_peak_W_m2': I_peak,
            'grad_I_W_m3': grad_I,
            'clausius_mossotti': clausius_mossotti,
            'F_gradient_N': F_grad,
            'F_gradient_fN': F_grad * 1e15,
            'F_radiation_N': F_rad,
            'F_radiation_fN': F_rad * 1e15,
            'sigma_abs_m2': sigma_abs,
            'v_gradient_m_s': v_grad,
            'v_gradient_um_s': v_grad * 1e6,
            'v_radiation_m_s': v_rad,
            'v_radiation_um_s': v_rad * 1e6,
            'v_total_optical_m_s': v_grad + v_rad,
            'v_total_optical_um_s': (v_grad + v_rad) * 1e6,
        }

    def calculate_optothermal_gradient(self, laser_power_mw: float = None,
                                       beam_waist_um: float = None) -> Dict:
        """
        Calcula gradiente termico generado por calentamiento del sustrato dorado.

        P_heat = P_laser * absorptance
        dT = P_heat / (4*pi*kappa_substrate*w0)   (modelo fuente puntual)
        grad_T = dT / w0
        """
        p = self.params
        P_W = (laser_power_mw or p.laser_power_mw) * 1e-3
        w0_m = (beam_waist_um or p.beam_waist_um) * 1e-6

        # Potencia absorbida por el sustrato
        P_heat = P_W * p.substrate_absorptance

        # Modelo de calentamiento puntual en sustrato dorado
        # dT ~ P_heat / (4*pi*kappa*w0)
        # Usamos conductividad efectiva (promedio agua-oro ponderado)
        kappa_eff = np.sqrt(THERMAL_CONDUCTIVITY_GOLD * THERMAL_CONDUCTIVITY_WATER)
        dT_K = P_heat / (4.0 * np.pi * kappa_eff * w0_m)

        # Gradiente termico
        grad_T = dT_K / w0_m

        return {
            'P_heat_W': P_heat,
            'P_heat_mW': P_heat * 1e3,
            'kappa_effective_W_mK': kappa_eff,
            'dT_K': dT_K,
            'grad_T_K_m': grad_T,
            'grad_T_K_um': grad_T * 1e-6,
            'beam_waist_um': w0_m * 1e6,
        }

    def calculate_thermophoretic_velocity(self, particle_diameter_nm: float,
                                          grad_T_K_m: float = None,
                                          laser_power_mw: float = None,
                                          beam_waist_um: float = None) -> Dict:
        """
        Calcula velocidad termoforetica por efecto Soret.

        D_T = S_T * D_brownian       (movilidad termoforetica)
        v_T = -D_T * grad_T          (velocidad termoforetica)
        """
        p = self.params
        T_K = p.temperature_c + 273.15
        d_m = particle_diameter_nm * 1e-9
        w0_m = (beam_waist_um or p.beam_waist_um) * 1e-6

        # Difusion browniana
        D_brownian = BOLTZMANN * T_K / (3.0 * np.pi * VISCOSITY_WATER * d_m)

        # Gradiente termico (calcular si no se proporciona)
        if grad_T_K_m is None:
            ot = self.calculate_optothermal_gradient(laser_power_mw, beam_waist_um)
            grad_T_K_m = ot['grad_T_K_m']

        # Coeficiente de Soret y movilidad termoforetica
        S_T = p.soret_coefficient
        D_T = S_T * D_brownian

        # Velocidad termoforetica (hacia frio = alejandose del spot)
        v_T = D_T * grad_T_K_m  # magnitud

        # Peclet termoforetico
        Pe_T = v_T * w0_m / D_brownian

        return {
            'D_brownian_m2_s': D_brownian,
            'S_T_K_inv': S_T,
            'D_T_m2_sK': D_T,
            'grad_T_K_m': grad_T_K_m,
            'v_thermophoretic_m_s': v_T,
            'v_thermophoretic_um_s': v_T * 1e6,
            'Pe_thermophoretic': Pe_T,
        }

    def calculate_differential_absorption(self, particle_diameter_nm: float,
                                          excitation_wavelength_nm: float,
                                          laser_power_mw: float = None,
                                          beam_waist_um: float = None) -> Dict:
        """
        Calcula selectividad por absorcion diferencial en modo opto-termico.

        QDot absorbente: calentamiento adicional -> dS_T mayor
        QDot no-absorbente: solo termoforesis de fondo del sustrato
        Factor de selectividad = (v_absorbing - v_background) / v_background
        """
        p = self.params
        P_W = (laser_power_mw or p.laser_power_mw) * 1e-3
        w0_m = (beam_waist_um or p.beam_waist_um) * 1e-6
        r_m = particle_diameter_nm * 1e-9 / 2.0

        # Irradiancia del laser
        I_peak = 2.0 * P_W / (np.pi * w0_m ** 2)

        # Absorcion de este QDot a esta lambda
        sel = self.calculate_selectivity(excitation_wavelength_nm, particle_diameter_nm)
        absorption_relative = sel['absorption_relative']

        # Seccion eficaz de absorcion (escala con d^3)
        d_ref_nm = 3.0
        sigma_abs = ABSORPTION_CROSS_SECTION * (particle_diameter_nm / d_ref_nm) ** 3

        # Calentamiento local adicional del QDot absorbente
        P_abs_particle = I_peak * sigma_abs * QUANTUM_YIELD_THERMAL * absorption_relative
        dT_particle = P_abs_particle / (4.0 * np.pi * THERMAL_CONDUCTIVITY_WATER * r_m)

        # Incremento del coeficiente de Soret por calentamiento local
        # El Soret efectivo aumenta con la temperatura local
        S_T_base = p.soret_coefficient
        S_T_enhanced = S_T_base * (1.0 + dT_particle / (p.temperature_c + 273.15))

        # Velocidad de fondo (sustrato solo, para todas las particulas)
        ot = self.calculate_optothermal_gradient(laser_power_mw, beam_waist_um)
        thermo = self.calculate_thermophoretic_velocity(
            particle_diameter_nm, ot['grad_T_K_m'], laser_power_mw, beam_waist_um)
        v_background = thermo['v_thermophoretic_m_s']

        # Velocidad del QDot absorbente (fondo + contribucion propia)
        D_brownian = thermo['D_brownian_m2_s']
        D_T_enhanced = S_T_enhanced * D_brownian
        v_absorbing = D_T_enhanced * ot['grad_T_K_m']

        # Factor de selectividad
        selectivity_factor = ((v_absorbing - v_background) / v_background
                              if v_background > 0 else 0)

        return {
            'absorption_relative': absorption_relative,
            'absorbs': sel['absorbs'],
            'sigma_abs_m2': sigma_abs,
            'P_absorbed_particle_W': P_abs_particle,
            'dT_particle_K': dT_particle,
            'S_T_base': S_T_base,
            'S_T_enhanced': S_T_enhanced,
            'v_background_m_s': v_background,
            'v_background_um_s': v_background * 1e6,
            'v_absorbing_m_s': v_absorbing,
            'v_absorbing_um_s': v_absorbing * 1e6,
            'selectivity_factor': selectivity_factor,
        }

    def calculate_trapping_potential(self, particle_diameter_nm: float,
                                     dT_K: float = None,
                                     laser_power_mw: float = None,
                                     beam_waist_um: float = None) -> Dict:
        """
        Calcula potencial de atrapamiento en unidades de kT.

        U_trap = S_T * k_B * T * dT    (en Joules)
        U_kT = U_trap / (k_B * T)      (en unidades kT; >1 para atrapamiento estable)
        """
        p = self.params
        T_K = p.temperature_c + 273.15
        S_T = p.soret_coefficient

        if dT_K is None:
            ot = self.calculate_optothermal_gradient(laser_power_mw, beam_waist_um)
            dT_K = ot['dT_K']

        # Potencial de atrapamiento
        U_trap_J = S_T * BOLTZMANN * T_K * dT_K
        U_kT = S_T * dT_K  # = U_trap / (k_B * T)

        # Estabilidad
        if U_kT > 10:
            stability = "fuerte"
        elif U_kT > 1:
            stability = "estable"
        elif U_kT > 0.1:
            stability = "debil"
        else:
            stability = "inestable"

        return {
            'S_T': S_T,
            'dT_K': dT_K,
            'U_trap_J': U_trap_J,
            'U_kT': U_kT,
            'stability': stability,
        }

    # ===========================================================================
    #  METODOS DE DISENO POR MODO
    # ===========================================================================

    def _design_led_mode(self, geometry: Dict) -> List[Dict]:
        """Diseno con LEDs difusos (modo original)"""
        p = self.params
        zones_results = []

        for i, zc in enumerate(self.zones_config[:p.n_zones]):
            led_wl = zc['led_wavelength_nm']
            size_min, size_max = zc['target_size_range_nm']
            representative_size = (size_min + size_max) / 2.0

            rad = self.calculate_radiation_force(
                p.led_power_mw, representative_size, p.led_array_count)
            photo = self.calculate_photothermal_velocity(
                p.led_power_mw, representative_size, p.led_array_count)
            sed = self.calculate_sedimentation(representative_size)
            brownian = self.calculate_brownian_diffusion(representative_size)

            r_m = representative_size * 1e-9 / 2.0
            stokes_drag = 6.0 * np.pi * VISCOSITY_WATER * r_m
            v_radiation_m_s = rad['F_radiation_N'] / stokes_drag if stokes_drag > 0 else 0
            v_up_total = photo['v_up_total_m_s'] + v_radiation_m_s

            sep_time = self.calculate_separation_time(
                v_up_total, sed['v_sedimentation_m_s'])
            eff = self.calculate_zone_efficiency(
                v_up_total, sed['v_sedimentation_m_s'],
                brownian['D_m2_s'], geometry['zone_residence_time_s'])
            sel_target = self.calculate_selectivity(led_wl, representative_size)
            other_size = 2.0 if representative_size > 3.0 else 4.5
            sel_other = self.calculate_selectivity(led_wl, other_size)

            zone_result = {
                'zone_number': i + 1,
                'mode': 'LED',
                'led_wavelength_nm': led_wl,
                'target_size_range_nm': zc['target_size_range_nm'],
                'target_emission_range_nm': zc['target_emission_range_nm'],
                'representative_size_nm': representative_size,
                'expected_content': zc['expected_content'],
                'radiation': rad,
                'photothermal': photo,
                'sedimentation': sed,
                'brownian': brownian,
                'v_radiation_m_s': v_radiation_m_s,
                'v_radiation_um_s': v_radiation_m_s * 1e6,
                'v_up_total_m_s': v_up_total,
                'v_up_total_um_s': v_up_total * 1e6,
                'separation_time': sep_time,
                'efficiency': eff,
                'selectivity_target': sel_target,
                'selectivity_other': sel_other,
            }
            zones_results.append(zone_result)
        return zones_results

    def _design_laser_direct_mode(self, geometry: Dict) -> List[Dict]:
        """Diseno con laser focalizado directo (sin sustrato)"""
        p = self.params
        zones_results = []

        for i, zc in enumerate(self.zones_config[:p.n_zones]):
            led_wl = zc['led_wavelength_nm']
            size_min, size_max = zc['target_size_range_nm']
            representative_size = (size_min + size_max) / 2.0

            # Fuerzas opticas del laser focalizado
            laser_forces = self.calculate_gradient_force(
                representative_size, p.laser_power_mw, p.beam_waist_um)

            # Irradiancia del laser
            laser_irr = self.calculate_laser_irradiance(
                p.laser_power_mw, p.beam_waist_um)

            sed = self.calculate_sedimentation(representative_size)
            brownian = self.calculate_brownian_diffusion(representative_size)

            v_up_total = laser_forces['v_total_optical_m_s']

            # Usar beam_waist como escala de separacion (no zone_height)
            w0_m = p.beam_waist_um * 1e-6
            Pe_up = v_up_total * w0_m / brownian['D_m2_s'] if brownian['D_m2_s'] > 0 else 0

            sep_time = self.calculate_separation_time(
                v_up_total, sed['v_sedimentation_m_s'])
            eff = self.calculate_zone_efficiency(
                v_up_total, sed['v_sedimentation_m_s'],
                brownian['D_m2_s'], geometry['zone_residence_time_s'])

            # Override Pe with laser-scale Pe
            eff['Pe_up'] = Pe_up

            sel_target = self.calculate_selectivity(led_wl, representative_size)
            other_size = 2.0 if representative_size > 3.0 else 4.5
            sel_other = self.calculate_selectivity(led_wl, other_size)

            # Construir resultado compatible con LED pero con datos laser
            zone_result = {
                'zone_number': i + 1,
                'mode': 'LASER_DIRECT',
                'led_wavelength_nm': led_wl,
                'target_size_range_nm': zc['target_size_range_nm'],
                'target_emission_range_nm': zc['target_emission_range_nm'],
                'representative_size_nm': representative_size,
                'expected_content': zc['expected_content'],
                'radiation': {
                    'P_total_W': laser_irr['P_laser_W'],
                    'irradiance_W_m2': laser_irr['I_peak_W_m2'],
                    'sigma_abs_m2': laser_forces['sigma_abs_m2'],
                    'F_radiation_N': laser_forces['F_radiation_N'],
                    'F_radiation_fN': laser_forces['F_radiation_fN'],
                    'P_radiation_Pa': laser_irr['I_peak_W_m2'] / SPEED_OF_LIGHT,
                },
                'photothermal': {
                    'P_absorbed_W': 0, 'P_absorbed_fW': 0,
                    'dT_local_K': 0, 'dT_local_mK': 0,
                    'v_thermal_m_s': 0, 'v_thermal_um_s': 0,
                    'v_photophoretic_m_s': 0, 'v_photophoretic_um_s': 0,
                    'v_up_total_m_s': 0, 'v_up_total_um_s': 0,
                },
                'laser': laser_forces,
                'laser_irradiance': laser_irr,
                'sedimentation': sed,
                'brownian': brownian,
                'v_radiation_m_s': laser_forces['v_radiation_m_s'],
                'v_radiation_um_s': laser_forces['v_radiation_um_s'],
                'v_gradient_m_s': laser_forces['v_gradient_m_s'],
                'v_gradient_um_s': laser_forces['v_gradient_um_s'],
                'v_up_total_m_s': v_up_total,
                'v_up_total_um_s': v_up_total * 1e6,
                'separation_time': sep_time,
                'efficiency': eff,
                'selectivity_target': sel_target,
                'selectivity_other': sel_other,
            }
            zones_results.append(zone_result)
        return zones_results

    def _design_optothermal_mode(self, geometry: Dict) -> List[Dict]:
        """Diseno opto-termico: laser + sustrato dorado + termoforesis"""
        p = self.params
        zones_results = []

        for i, zc in enumerate(self.zones_config[:p.n_zones]):
            led_wl = zc['led_wavelength_nm']
            size_min, size_max = zc['target_size_range_nm']
            representative_size = (size_min + size_max) / 2.0

            # Irradiancia del laser
            laser_irr = self.calculate_laser_irradiance(
                p.laser_power_mw, p.beam_waist_um)

            # Gradiente termico del sustrato
            ot_grad = self.calculate_optothermal_gradient(
                p.laser_power_mw, p.beam_waist_um)

            # Velocidad termoforetica
            thermo = self.calculate_thermophoretic_velocity(
                representative_size, ot_grad['grad_T_K_m'],
                p.laser_power_mw, p.beam_waist_um)

            # Absorcion diferencial (selectividad)
            diff_abs = self.calculate_differential_absorption(
                representative_size, led_wl,
                p.laser_power_mw, p.beam_waist_um)

            # Potencial de atrapamiento
            trap = self.calculate_trapping_potential(
                representative_size, ot_grad['dT_K'],
                p.laser_power_mw, p.beam_waist_um)

            # Fuerzas opticas directas (contribucion menor)
            laser_forces = self.calculate_gradient_force(
                representative_size, p.laser_power_mw, p.beam_waist_um)

            sed = self.calculate_sedimentation(representative_size)
            brownian = self.calculate_brownian_diffusion(representative_size)

            # Velocidad total: termoforesis (dominante) + optica (menor)
            v_thermo = thermo['v_thermophoretic_m_s']
            v_optical = laser_forces['v_total_optical_m_s']
            v_up_total = v_thermo + v_optical

            # Peclet a escala del beam waist
            w0_m = p.beam_waist_um * 1e-6
            Pe_up = v_up_total * w0_m / brownian['D_m2_s'] if brownian['D_m2_s'] > 0 else 0

            sep_time = self.calculate_separation_time(
                v_up_total, sed['v_sedimentation_m_s'])
            eff = self.calculate_zone_efficiency(
                v_up_total, sed['v_sedimentation_m_s'],
                brownian['D_m2_s'], geometry['zone_residence_time_s'])
            eff['Pe_up'] = Pe_up  # Override con Pe a escala laser

            sel_target = self.calculate_selectivity(led_wl, representative_size)
            other_size = 2.0 if representative_size > 3.0 else 4.5
            sel_other = self.calculate_selectivity(led_wl, other_size)

            zone_result = {
                'zone_number': i + 1,
                'mode': 'OPTOTHERMAL',
                'led_wavelength_nm': led_wl,
                'target_size_range_nm': zc['target_size_range_nm'],
                'target_emission_range_nm': zc['target_emission_range_nm'],
                'representative_size_nm': representative_size,
                'expected_content': zc['expected_content'],
                'radiation': {
                    'P_total_W': laser_irr['P_laser_W'],
                    'irradiance_W_m2': laser_irr['I_peak_W_m2'],
                    'sigma_abs_m2': laser_forces['sigma_abs_m2'],
                    'F_radiation_N': laser_forces['F_radiation_N'],
                    'F_radiation_fN': laser_forces['F_radiation_fN'],
                    'P_radiation_Pa': laser_irr['I_peak_W_m2'] / SPEED_OF_LIGHT,
                },
                'photothermal': {
                    'P_absorbed_W': 0, 'P_absorbed_fW': 0,
                    'dT_local_K': ot_grad['dT_K'], 'dT_local_mK': ot_grad['dT_K'] * 1e3,
                    'v_thermal_m_s': v_thermo, 'v_thermal_um_s': v_thermo * 1e6,
                    'v_photophoretic_m_s': 0, 'v_photophoretic_um_s': 0,
                    'v_up_total_m_s': v_thermo, 'v_up_total_um_s': v_thermo * 1e6,
                },
                'optothermal': ot_grad,
                'thermophoresis': thermo,
                'differential_absorption': diff_abs,
                'trapping': trap,
                'laser': laser_forces,
                'laser_irradiance': laser_irr,
                'sedimentation': sed,
                'brownian': brownian,
                'v_thermophoretic_m_s': v_thermo,
                'v_thermophoretic_um_s': v_thermo * 1e6,
                'v_optical_m_s': v_optical,
                'v_optical_um_s': v_optical * 1e6,
                'v_radiation_m_s': laser_forces['v_radiation_m_s'],
                'v_radiation_um_s': laser_forces['v_radiation_um_s'],
                'v_up_total_m_s': v_up_total,
                'v_up_total_um_s': v_up_total * 1e6,
                'separation_time': sep_time,
                'efficiency': eff,
                'selectivity_target': sel_target,
                'selectivity_other': sel_other,
            }
            zones_results.append(zone_result)
        return zones_results

    def calculate_separation_time(self, v_up_m_s: float, v_sed_m_s: float,
                                  zone_height_mm: float = None) -> Dict:
        """
        Calcula tiempo para que QDot suba vs debris baje por la zona.

        t_qdot_up = zone_height / v_up  (QDot excitado sube al puerto superior)
        t_debris_down = zone_height / v_sed  (debris sedimenta al fondo)
        """
        p = self.params
        h_m = (zone_height_mm or p.zone_height_mm) * 1e-3

        # Tiempo para que QDot suba la mitad de la zona (desde el centro)
        half_h_m = h_m / 2.0

        if v_up_m_s > 0:
            t_qdot_up_s = half_h_m / v_up_m_s
        else:
            t_qdot_up_s = float('inf')

        if v_sed_m_s > 0:
            t_debris_down_s = half_h_m / v_sed_m_s
        else:
            t_debris_down_s = float('inf')

        # Tiempo maximo de separacion (el mas lento de los dos)
        t_separation_s = max(t_qdot_up_s, t_debris_down_s)

        return {
            't_qdot_up_s': t_qdot_up_s,
            't_qdot_up_min': t_qdot_up_s / 60.0,
            't_debris_down_s': t_debris_down_s,
            't_debris_down_min': t_debris_down_s / 60.0,
            't_separation_s': t_separation_s,
            't_separation_min': t_separation_s / 60.0,
            'half_height_mm': half_h_m * 1e3,
        }

    def calculate_zone_efficiency(self, v_up_m_s: float, v_sed_m_s: float,
                                  D_m2_s: float, residence_time_s: float,
                                  zone_height_mm: float = None) -> Dict:
        """
        Calcula probabilidad de separacion por zona.

        P(QDot sube) basada en Peclet vertical: Pe = v_up * h / D
        P(debris baja) basada en Pe_sed = v_sed * h / D

        Si Pe >> 1: la fuerza dirigida domina sobre difusion -> alta eficiencia
        Si Pe ~ 1: difusion compite -> baja eficiencia
        """
        p = self.params
        h_m = (zone_height_mm or p.zone_height_mm) * 1e-3
        half_h_m = h_m / 2.0

        # Peclet vertical para QDot excitado
        if D_m2_s > 0:
            Pe_up = v_up_m_s * half_h_m / D_m2_s
        else:
            Pe_up = float('inf')

        # Peclet vertical para sedimentacion
        if D_m2_s > 0:
            Pe_sed = v_sed_m_s * half_h_m / D_m2_s
        else:
            Pe_sed = float('inf')

        # Probabilidad de que QDot alcance el puerto superior
        # Modelo: P = 1 - exp(-Pe * t_res / t_transit)
        t_transit_up = half_h_m / max(v_up_m_s, 1e-20)
        ratio_up = residence_time_s / t_transit_up if t_transit_up < float('inf') else 0
        P_qdot_up = 1.0 - np.exp(-max(0, ratio_up))

        # Probabilidad de que debris alcance el puerto inferior
        t_transit_down = half_h_m / max(v_sed_m_s, 1e-20)
        ratio_down = residence_time_s / t_transit_down if t_transit_down < float('inf') else 0
        P_debris_down = 1.0 - np.exp(-max(0, ratio_down))

        # Eficiencia de zona = promedio de captura QDot y rechazo debris
        efficiency = (P_qdot_up + P_debris_down) / 2.0

        return {
            'Pe_up': Pe_up,
            'Pe_sed': Pe_sed,
            'P_qdot_up': P_qdot_up,
            'P_debris_down': P_debris_down,
            'efficiency': efficiency,
            'residence_time_s': residence_time_s,
        }

    def calculate_selectivity(self, target_wavelength_nm: float,
                              particle_diameter_nm: float) -> Dict:
        """
        Calcula selectividad lambda-especifica.

        QDot que absorbe a la lambda del LED -> fuerza vertical (sube)
        QDot de otro color NO absorbe a esa lambda -> se queda
        Debris NO absorbe -> se queda

        Modelo: la absorcion depende de cuanto se acerca la lambda
        del LED a la primera transicion de absorcion del QDot.
        """
        # Calcular emision del QDot de este tamano
        emission_nm = size_to_wavelength(particle_diameter_nm)

        # La absorcion del QDot tiene un continuo que comienza debajo de su emision
        # (absorbe fotones de energia mayor que su gap)
        gap_ev = E_BULK_EV + A_CONFINEMENT / (particle_diameter_nm ** 2)
        absorption_onset_nm = EV_TO_NM / gap_ev  # = emision

        # El LED de excitacion debe tener lambda < emision (mayor energia)
        # Cuanto mayor el Stokes shift (emision - excitacion), mejor
        stokes_shift_nm = emission_nm - target_wavelength_nm

        # Selectividad: si el LED tiene lambda < onset (el QDot absorbe)
        if target_wavelength_nm < absorption_onset_nm:
            # El QDot absorbe: cuanto mas cerca del onset, mejor absorcion
            # Modelo simplificado: absorcion relativa decae con distancia al onset
            delta_nm = absorption_onset_nm - target_wavelength_nm
            # Absorcion maxima cerca del onset, decae exponencialmente
            absorption_relative = np.exp(-delta_nm / 100.0)
            absorbs = True
        else:
            # Lambda del LED > emision del QDot: NO absorbe (energia insuficiente)
            absorption_relative = 0.0
            absorbs = False

        return {
            'emission_nm': emission_nm,
            'absorption_onset_nm': absorption_onset_nm,
            'stokes_shift_nm': stokes_shift_nm,
            'absorption_relative': absorption_relative,
            'absorbs': absorbs,
            'gap_ev': gap_ev,
        }

    def calculate_scores(self, zones_results: List[Dict],
                         geometry: Dict) -> Dict:
        """Calcula puntuaciones de diseno (0-100)"""
        p = self.params

        # --- Puntuacion de eficiencia (0-100) ---
        efficiency_score = 0.0

        # Eficiencia promedio de zonas (50 puntos max)
        efficiencies = [z['efficiency']['efficiency'] for z in zones_results]
        avg_eff = np.mean(efficiencies) if efficiencies else 0.0
        efficiency_score += min(50.0, avg_eff * 50.0)

        # Selectividad promedio (30 puntos max)
        selectivities = []
        for z in zones_results:
            target_sel = z['selectivity_target']['absorption_relative']
            selectivities.append(target_sel)
        avg_sel = np.mean(selectivities) if selectivities else 0.0
        efficiency_score += min(30.0, avg_sel * 30.0)

        # Separacion vs Browniana (20 puntos max)
        pe_values = [z['efficiency']['Pe_up'] for z in zones_results]
        avg_pe = np.mean(pe_values) if pe_values else 0.0
        if avg_pe > 10:
            efficiency_score += 20.0
        elif avg_pe > 1:
            efficiency_score += 10.0 + 10.0 * (avg_pe - 1) / 9.0
        else:
            efficiency_score += avg_pe * 10.0

        efficiency_score = min(100.0, max(0.0, efficiency_score))

        # --- Puntuacion de costo (0-100) ---
        cost_score = 100.0

        # Mas zonas = mas caro
        cost_score -= max(0, (p.n_zones - 2)) * 5.0

        # LEDs de alta potencia
        total_led_power_W = p.led_power_mw * 1e-3 * p.led_array_count * p.n_zones
        if total_led_power_W > 5.0:
            cost_score -= 15.0
        elif total_led_power_W > 2.0:
            cost_score -= 10.0

        # Tamano total
        if geometry['total_length_mm'] > 200:
            cost_score -= 10.0

        cost_score = min(100.0, max(0.0, cost_score))

        # --- Puntuacion de factibilidad (0-100) ---
        feasibility_score = 100.0

        # Tiempo de separacion vs residencia
        for z in zones_results:
            t_sep = z['separation_time']['t_separation_s']
            t_res = z['efficiency']['residence_time_s']
            if t_sep > t_res:
                # No hay suficiente tiempo para separar
                excess = t_sep / max(t_res, 0.01)
                feasibility_score -= min(15.0, (excess - 1.0) * 10.0)

        # Peclet numbers razonables
        if avg_pe < 0.1:
            feasibility_score -= 30.0  # Difusion domina totalmente
        elif avg_pe < 1.0:
            feasibility_score -= 15.0  # Difusion compite

        # Tamano total razonable
        if geometry['total_length_mm'] > 250:
            feasibility_score -= 10.0

        # Potencia termica excesiva (podria calentar el fluido)
        for z in zones_results:
            dT = z['photothermal']['dT_local_mK']
            if dT > 100:  # >100 mK de calentamiento local
                feasibility_score -= 5.0

        feasibility_score = min(100.0, max(0.0, feasibility_score))

        return {
            'efficiency_score': efficiency_score,
            'cost_score': cost_score,
            'feasibility_score': feasibility_score,
            'overall_score': (efficiency_score + cost_score + feasibility_score) / 3.0,
        }

    def design(self) -> ClassifierOutput:
        """Ejecuta el diseno completo del clasificador"""
        p = self.params

        # 1. Geometria
        geometry = self.calculate_zone_geometry()

        # 2. Fisica por zona (ramificado por modo de excitacion)
        if p.excitation_mode == ExcitationMode.LED:
            zones_results = self._design_led_mode(geometry)
        elif p.excitation_mode == ExcitationMode.LASER_DIRECT:
            zones_results = self._design_laser_direct_mode(geometry)
        else:  # OPTOTHERMAL
            zones_results = self._design_optothermal_mode(geometry)

        # 3. Puntuaciones
        scores = self.calculate_scores(zones_results, geometry)

        # 4. Calcular metricas globales
        efficiencies = [z['efficiency']['efficiency'] for z in zones_results]
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0.0

        selectivities = [z['selectivity_target']['absorption_relative']
                         for z in zones_results]
        avg_selectivity = np.mean(selectivities) if selectivities else 0.0

        # Recuperacion: producto de probabilidades de captura
        p_captures = [z['efficiency']['P_qdot_up'] for z in zones_results]
        overall_recovery = np.mean(p_captures) if p_captures else 0.0

        # Pureza: basada en selectividad y eficiencia
        overall_purity = avg_selectivity * avg_efficiency

        # 5. Poblar salida
        self.output = ClassifierOutput(
            total_length_mm=geometry['total_length_mm'],
            total_width_mm=geometry['total_width_mm'],
            total_height_mm=geometry['total_height_mm'],
            zones=zones_results,
            total_volume_ml=geometry['total_volume_ml'],
            residence_time_s=geometry['residence_time_s'],
            avg_zone_efficiency=avg_efficiency,
            avg_selectivity=avg_selectivity,
            overall_recovery=overall_recovery,
            overall_purity=overall_purity,
            efficiency_score=scores['efficiency_score'],
            cost_score=scores['cost_score'],
            feasibility_score=scores['feasibility_score'],
        )

        # Almacenar resultados intermedios para reporte
        self._geometry = geometry
        self._zones_results = zones_results
        self._scores = scores

        return self.output

    def optimize(self, max_iterations: int = 500) -> Tuple[ClassifierParameters, ClassifierOutput]:
        """
        Optimiza parametros del clasificador por busqueda aleatoria.
        Busca maximizar la puntuacion global.
        """
        best_params = None
        best_output = None
        best_score = -1.0

        search_space = {
            'n_zones': [2, 3, 4],
            'zone_length_mm': [30.0, 40.0, 50.0, 60.0, 80.0],
            'zone_height_mm': [10.0, 15.0, 20.0, 25.0, 30.0],
            'zone_width_mm': [8.0, 10.0, 15.0, 20.0],
            'led_power_mw': [100.0, 200.0, 500.0, 1000.0],
            'led_array_count': [2, 4, 6, 8],
            'flow_rate_ml_min': [0.5, 1.0, 2.0, 5.0],
            'channel_height_mm': [1.0, 2.0, 3.0],
        }

        np.random.seed(42)
        for _ in range(max_iterations):
            n_zones = int(np.random.choice(search_space['n_zones']))
            led_power = float(np.random.choice(search_space['led_power_mw']))

            # Generar wavelengths para n_zones
            all_wavelengths = [520.0, 405.0, 365.0, 340.0]
            led_wavelengths = all_wavelengths[:n_zones]

            params = ClassifierParameters(
                n_zones=n_zones,
                zone_length_mm=float(np.random.choice(search_space['zone_length_mm'])),
                zone_height_mm=float(np.random.choice(search_space['zone_height_mm'])),
                zone_width_mm=float(np.random.choice(search_space['zone_width_mm'])),
                led_power_mw=led_power,
                led_array_count=int(np.random.choice(search_space['led_array_count'])),
                led_wavelengths_nm=led_wavelengths,
                flow_rate_ml_min=float(np.random.choice(search_space['flow_rate_ml_min'])),
                channel_height_mm=float(np.random.choice(search_space['channel_height_mm'])),
            )

            # Configuracion de zonas para n_zones
            zones = DEFAULT_ZONES_CONFIG[:n_zones]

            try:
                designer = ClassifierDesigner(params, zones)
                output = designer.design()

                score = (output.efficiency_score * 0.40 +
                         output.feasibility_score * 0.35 +
                         output.cost_score * 0.25)

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_output = output
            except (ValueError, ZeroDivisionError):
                continue

        return best_params, best_output

    def print_design_report(self):
        """Imprime reporte completo del diseno del clasificador"""
        p = self.params
        o = self.output

        mode_labels = {
            ExcitationMode.LED: "LED difuso",
            ExcitationMode.LASER_DIRECT: "Laser focalizado directo",
            ExcitationMode.OPTOTHERMAL: "Opto-termico (laser + sustrato Au)",
        }
        mode_label = mode_labels.get(p.excitation_mode, str(p.excitation_mode))

        print("\n" + "=" * 80)
        print("  REPORTE DE DISENO DEL CLASIFICADOR POR FLOTABILIDAD OPTICA")
        print(f"  Modo: {mode_label}")
        print("=" * 80)

        # --- 1. GEOMETRIA DEL SEPARADOR ---
        print("\n+" + "-" * 78 + "+")
        print("|  1. GEOMETRIA DEL SEPARADOR" + " " * 50 + "|")
        print("+" + "-" * 78 + "+")
        g = self._geometry
        lines = [
            f"Numero de zonas:        {p.n_zones}",
            f"Longitud total:         {o.total_length_mm:.1f} mm",
            f"Ancho total:            {o.total_width_mm:.1f} mm",
            f"Altura total:           {o.total_height_mm:.1f} mm",
            f"Zona (L x W x H):      {p.zone_length_mm:.0f} x {p.zone_width_mm:.0f} x {p.zone_height_mm:.0f} mm",
            f"Volumen por zona:       {g['zone_volume_ml']:.2f} mL",
            f"Volumen total:          {g['total_volume_ml']:.2f} mL",
            f"Tiempo residencia:      {g['residence_time_s']:.1f} s ({g['residence_time_s']/60:.1f} min)",
            f"Tiempo por zona:        {g['zone_residence_time_s']:.1f} s",
            f"Velocidad canal:        {g['channel_velocity_mm_s']:.2f} mm/s",
            f"Barrera entre zonas:    {p.barrier_pore_um:.0f} um poro (sin dP significativa)",
            f"Material:               {p.body_material}",
        ]
        for line in lines:
            print(f"|  {line:<76}|")
        print("+" + "-" * 78 + "+")

        # --- 2. ZONAS DE EXCITACION ---
        print("\n+" + "-" * 78 + "+")
        print("|  2. ZONAS DE EXCITACION" + " " * 54 + "|")
        print("+" + "-" * 78 + "+")
        header = f"{'Zona':<6} {'LED (nm)':<10} {'QDot (nm)':<14} {'Emision (nm)':<16} {'Contenido':<18}"
        print(f"|  {header:<76}|")
        print("|  " + "-" * 74 + "  |")
        for z in self._zones_results:
            smin, smax = z['target_size_range_nm']
            emin, emax = z['target_emission_range_nm']
            line = (f"{z['zone_number']:<6} {z['led_wavelength_nm']:<10.0f} "
                    f"{smin:.1f}-{smax:.1f} nm{'':<3} "
                    f"{emin:.0f}-{emax:.0f} nm{'':<4} "
                    f"{z['expected_content']:<18}")
            print(f"|  {line:<76}|")
        print("+" + "-" * 78 + "+")

        # --- 3. FISICA DE SEPARACION ---
        print("\n+" + "-" * 78 + "+")
        print("|  3. FISICA DE SEPARACION" + " " * 53 + "|")
        print("+" + "-" * 78 + "+")
        for z in self._zones_results:
            mode_str = z.get('mode', 'LED')
            print(f"|  --- Zona {z['zone_number']}: {z['led_wavelength_nm']:.0f} nm [{mode_str}], "
                  f"QDot {z['representative_size_nm']:.1f} nm ---" +
                  " " * max(1, 78 - 48 - len(mode_str) - len(f"{z['representative_size_nm']:.1f}")) + "|")

            rad = z['radiation']
            photo = z['photothermal']
            sed = z['sedimentation']
            br = z['brownian']

            physics_lines = [
                f"  Irradiancia:          {rad['irradiance_W_m2']:.2e} W/m2",
                f"  sigma_abs:            {rad['sigma_abs_m2']:.2e} m2",
                f"  F_radiacion:          {rad['F_radiation_fN']:.4f} fN",
                f"  v_radiacion:          {z['v_radiation_um_s']:.4f} um/s",
            ]

            if p.excitation_mode == ExcitationMode.LED:
                physics_lines += [
                    f"  P_absorbida (calor):  {photo['P_absorbed_fW']:.2f} fW",
                    f"  dT local:             {photo['dT_local_mK']:.3f} mK",
                    f"  v_fototermica:        {photo['v_thermal_um_s']:.4f} um/s",
                    f"  v_fotoforetica:       {photo['v_photophoretic_um_s']:.4f} um/s",
                ]
            elif p.excitation_mode == ExcitationMode.LASER_DIRECT:
                lf = z.get('laser', {})
                physics_lines += [
                    f"  F_gradiente:          {lf.get('F_gradient_fN', 0):.4f} fN",
                    f"  v_gradiente:          {z.get('v_gradient_um_s', 0):.4f} um/s",
                ]
            else:  # OPTOTHERMAL
                ot = z.get('optothermal', {})
                th = z.get('thermophoresis', {})
                da = z.get('differential_absorption', {})
                tr = z.get('trapping', {})
                physics_lines += [
                    f"  --- Opto-termico ---",
                    f"  P_calor sustrato:     {ot.get('P_heat_mW', 0):.1f} mW",
                    f"  dT sustrato:          {ot.get('dT_K', 0):.1f} K",
                    f"  grad_T:               {ot.get('grad_T_K_m', 0):.2e} K/m",
                    f"  v_termoforetica:      {z.get('v_thermophoretic_um_s', 0):.2f} um/s",
                    f"  Pe_termoforetico:     {th.get('Pe_thermophoretic', 0):.2f}",
                    f"  --- Selectividad diferencial ---",
                    f"  v_absorbente:         {da.get('v_absorbing_um_s', 0):.2f} um/s",
                    f"  v_fondo:              {da.get('v_background_um_s', 0):.2f} um/s",
                    f"  factor_selectividad:  {da.get('selectivity_factor', 0):.3f}",
                    f"  --- Atrapamiento ---",
                    f"  U_trap:               {tr.get('U_kT', 0):.2f} kT ({tr.get('stability', '?')})",
                ]

            physics_lines += [
                f"  v_UP TOTAL:           {z['v_up_total_um_s']:.4f} um/s",
                f"  v_sedimentacion:      {sed['v_sedimentation_um_s']:.6f} um/s",
                f"  D_browniana:          {br['D_um2_s']:.2f} um2/s",
                f"  x_rms (1s):           {br['x_rms_1s_um']:.2f} um",
            ]
            for line in physics_lines:
                print(f"|  {line:<76}|")
            print("|  " + "-" * 74 + "  |")
        print("+" + "-" * 78 + "+")

        # --- 4. TIEMPOS DE SEPARACION ---
        print("\n+" + "-" * 78 + "+")
        print("|  4. TIEMPOS DE SEPARACION" + " " * 52 + "|")
        print("+" + "-" * 78 + "+")
        header = f"{'Zona':<6} {'t_QDot_up':<16} {'t_debris_dn':<16} {'t_separacion':<16} {'t_residencia':<14}"
        print(f"|  {header:<76}|")
        print("|  " + "-" * 74 + "  |")
        for z in self._zones_results:
            st = z['separation_time']
            t_res = z['efficiency']['residence_time_s']
            line = (f"{z['zone_number']:<6} "
                    f"{st['t_qdot_up_min']:.1f} min{'':<7} "
                    f"{st['t_debris_down_min']:.1f} min{'':<7} "
                    f"{st['t_separation_min']:.1f} min{'':<7} "
                    f"{t_res:.0f} s")
            print(f"|  {line:<76}|")
        print("+" + "-" * 78 + "+")

        # --- 5. EFICIENCIA POR ZONA ---
        print("\n+" + "-" * 78 + "+")
        print("|  5. EFICIENCIA POR ZONA" + " " * 54 + "|")
        print("+" + "-" * 78 + "+")
        header = f"{'Zona':<6} {'Pe_up':<10} {'Pe_sed':<10} {'P(QDot^)':<12} {'P(debris v)':<12} {'Eficiencia':<12}"
        print(f"|  {header:<76}|")
        print("|  " + "-" * 74 + "  |")
        for z in self._zones_results:
            e = z['efficiency']
            line = (f"{z['zone_number']:<6} "
                    f"{e['Pe_up']:<10.2f} "
                    f"{e['Pe_sed']:<10.4f} "
                    f"{e['P_qdot_up']*100:<11.1f}% "
                    f"{e['P_debris_down']*100:<11.1f}% "
                    f"{e['efficiency']*100:<11.1f}%")
            print(f"|  {line:<76}|")

        print("|  " + "-" * 74 + "  |")
        print(f"|  {'PROMEDIOS:':<6}" + " " * 20 +
              f"  P(QDot^)={o.overall_recovery*100:.1f}%"
              f"  Efic={o.avg_zone_efficiency*100:.1f}%" +
              " " * 17 + "|")
        print("+" + "-" * 78 + "+")

        # --- 6. SELECTIVIDAD ---
        print("\n+" + "-" * 78 + "+")
        print("|  6. SELECTIVIDAD LAMBDA-ESPECIFICA" + " " * 43 + "|")
        print("+" + "-" * 78 + "+")
        for z in self._zones_results:
            st = z['selectivity_target']
            so = z['selectivity_other']
            lines = [
                f"Zona {z['zone_number']}: LED {z['led_wavelength_nm']:.0f} nm",
                f"  Target ({z['representative_size_nm']:.1f} nm): absorbe={'Si' if st['absorbs'] else 'No'}, "
                f"rel={st['absorption_relative']:.3f}, Stokes={st['stokes_shift_nm']:.0f} nm",
                f"  Otro tamano: absorbe={'Si' if so['absorbs'] else 'No'}, "
                f"rel={so['absorption_relative']:.3f}",
            ]
            for line in lines:
                print(f"|  {line:<76}|")
        print("+" + "-" * 78 + "+")

        # --- 7. PUNTUACIONES ---
        print("\n+" + "-" * 78 + "+")
        print("|  7. PUNTUACIONES DE DISENO" + " " * 51 + "|")
        print("+" + "-" * 78 + "+")
        score_lines = [
            f"Eficiencia:             {o.efficiency_score:.0f}/100",
            f"Costo:                  {o.cost_score:.0f}/100",
            f"Factibilidad:           {o.feasibility_score:.0f}/100",
            f"PUNTUACION GLOBAL:      {self._scores['overall_score']:.0f}/100",
        ]
        for line in score_lines:
            print(f"|  {line:<76}|")
        print("+" + "-" * 78 + "+")

        # --- DIAGRAMA ASCII ---
        self._print_ascii_diagram()

    def _print_ascii_diagram(self):
        """Imprime diagrama ASCII del clasificador"""
        p = self.params
        n = p.n_zones

        print("\n+" + "-" * 78 + "+")
        print("|  DIAGRAMA DEL SEPARADOR POR FLOTABILIDAD OPTICA" + " " * 29 + "|")
        print("+" + "-" * 78 + "+")

        # Construir labels
        zone_labels = []
        for z in self._zones_results:
            zone_labels.append(f"LED {z['led_wavelength_nm']:.0f}nm")

        zone_w = 22
        total_w = n * zone_w + 4

        print("|" + " " * 78 + "|")

        # LEDs arriba
        led_line = "  "
        for label in zone_labels:
            led_line += f"    {label:^14}    "
        print(f"|  {led_line:<76}|")

        arrow_line = "  "
        for _ in range(n):
            arrow_line += "      vvvvvv        "
        print(f"|  {arrow_line:<76}|")

        # Top border
        top = "  +" + ("─" * (zone_w - 1) + "┬") * (n - 1) + "─" * (zone_w - 1) + "+"
        print(f"|  {top:<76}|")

        # QDots suben
        up_line = "  │"
        for i, z in enumerate(self._zones_results):
            content = f" ^ {z['expected_content'][:14]:^14}  "
            up_line += content
            if i < n - 1:
                up_line += "│"
        up_line += "│  <- Coleccion ARRIBA"
        print(f"|  {up_line:<76}|")

        # Zona
        flow_line = "IN->"
        for i in range(n):
            flow_line += f"     Zona {i+1}         "
            if i < n - 1:
                flow_line += "║"
        flow_line += "-> WASTE"
        print(f"|  {flow_line:<76}|")

        # Debris bajan
        dn_line = "  │"
        for i in range(n):
            dn_line += f" v {'sedimenta':^14}  "
            if i < n - 1:
                dn_line += "│"
        dn_line += "│  <- Coleccion ABAJO"
        print(f"|  {dn_line:<76}|")

        # Bottom border
        bot = "  +" + ("─" * (zone_w - 1) + "┴") * (n - 1) + "─" * (zone_w - 1) + "+"
        print(f"|  {bot:<76}|")

        print("|" + " " * 78 + "|")
        print(f"|  Barreras: malla {p.barrier_pore_um:.0f} um (sin caida de presion)" +
              " " * 27 + "|")
        print(f"|  LEDs: {p.led_array_count} x {p.led_power_mw:.0f} mW por zona" +
              " " * (78 - 28 - len(f"{p.led_array_count} x {p.led_power_mw:.0f} mW por zona")) + "|")
        print("+" + "-" * 78 + "+")

    @staticmethod
    def compare_modes(zones_config=None):
        """
        Ejecuta los 3 modos de excitacion y muestra comparacion lado a lado.
        Muy util para demostrar por que el modo opto-termico es el unico viable.
        """
        zones_config = zones_config or DEFAULT_ZONES_CONFIG

        # Ejecutar diseno para cada modo
        results = {}
        for mode in ExcitationMode:
            params = ClassifierParameters(excitation_mode=mode)
            designer = ClassifierDesigner(params, zones_config)
            designer.design()
            results[mode] = {
                'output': designer.output,
                'zones': designer._zones_results,
                'scores': designer._scores,
                'geometry': designer._geometry,
            }

        print("\n" + "=" * 80)
        print("  COMPARACION DE MODOS DE EXCITACION")
        print("  Mismo sistema, 3 aproximaciones fisicas diferentes")
        print("=" * 80)

        # Tabla comparativa por zona representativa (Zona 2: 3.0 nm, la mas tipica)
        zone_idx = 1  # Zona 2
        if len(zones_config) < 2:
            zone_idx = 0

        print(f"\n  Zona representativa: Zona {zone_idx+1} "
              f"(QDot {zones_config[zone_idx]['target_size_range_nm']}nm)")
        print()

        header = f"  {'Parametro':<30} {'LED':<18} {'Laser directo':<18} {'Opto-termico':<18}"
        print(header)
        print("  " + "-" * 82)

        # Extraer datos por modo
        led_z = results[ExcitationMode.LED]['zones'][zone_idx]
        laser_z = results[ExcitationMode.LASER_DIRECT]['zones'][zone_idx]
        ot_z = results[ExcitationMode.OPTOTHERMAL]['zones'][zone_idx]

        rows = [
            ("Irradiancia (W/m2)",
             f"{led_z['radiation']['irradiance_W_m2']:.0f}",
             f"{laser_z['radiation']['irradiance_W_m2']:.2e}",
             f"{ot_z['radiation']['irradiance_W_m2']:.2e}"),
            ("F_radiacion (fN)",
             f"{led_z['radiation']['F_radiation_fN']:.6f}",
             f"{laser_z['radiation']['F_radiation_fN']:.4f}",
             f"{ot_z['radiation']['F_radiation_fN']:.4f}"),
            ("v_up total (um/s)",
             f"{led_z['v_up_total_um_s']:.6f}",
             f"{laser_z['v_up_total_um_s']:.4f}",
             f"{ot_z['v_up_total_um_s']:.2f}"),
            ("D_browniana (um2/s)",
             f"{led_z['brownian']['D_um2_s']:.1f}",
             f"{laser_z['brownian']['D_um2_s']:.1f}",
             f"{ot_z['brownian']['D_um2_s']:.1f}"),
            ("Pe (Peclet)",
             f"{led_z['efficiency']['Pe_up']:.6f}",
             f"{laser_z['efficiency']['Pe_up']:.4f}",
             f"{ot_z['efficiency']['Pe_up']:.2f}"),
            ("Eficiencia zona (%)",
             f"{led_z['efficiency']['efficiency']*100:.1f}",
             f"{laser_z['efficiency']['efficiency']*100:.1f}",
             f"{ot_z['efficiency']['efficiency']*100:.1f}"),
        ]

        # Datos especificos opto-termicos
        if 'optothermal' in ot_z:
            rows += [
                ("dT sustrato (K)",
                 "-", "-",
                 f"{ot_z['optothermal']['dT_K']:.1f}"),
                ("grad_T (K/m)",
                 "-", "-",
                 f"{ot_z['optothermal']['grad_T_K_m']:.2e}"),
                ("v_termoforetica (um/s)",
                 "-", "-",
                 f"{ot_z.get('v_thermophoretic_um_s', 0):.2f}"),
                ("U_trap (kT)",
                 "-", "-",
                 f"{ot_z['trapping']['U_kT']:.2f}"),
            ]

        for label, v_led, v_laser, v_ot in rows:
            print(f"  {label:<30} {v_led:<18} {v_laser:<18} {v_ot:<18}")

        # Puntuaciones globales
        print("\n  " + "-" * 82)
        print(f"  {'PUNTUACION GLOBAL':<30} "
              f"{results[ExcitationMode.LED]['scores']['overall_score']:<18.0f} "
              f"{results[ExcitationMode.LASER_DIRECT]['scores']['overall_score']:<18.0f} "
              f"{results[ExcitationMode.OPTOTHERMAL]['scores']['overall_score']:<18.0f}")

        # Veredicto
        print("\n  " + "=" * 82)
        print("  VEREDICTO:")
        print("    LED:            Pe << 1  ->  Difusion browniana domina completamente")
        print("    Laser directo:  Pe << 1  ->  10^6x mas irradiancia pero fuerzas opticas")
        print("                                 sobre CQDs de 2-5nm siguen siendo insuficientes")
        print("    OPTO-TERMICO:   Pe ~ 1-10 -> Termoforesis via sustrato dorado es VIABLE")
        print("                                 Primera aproximacion con separacion real")
        print("  " + "=" * 82)

    def export_cad_parameters(self) -> Dict:
        """Exporta parametros para software CAD en formato JSON"""
        p = self.params
        o = self.output

        zones_cad = []
        for z in o.zones:
            zones_cad.append({
                'zone_number': z['zone_number'],
                'led_wavelength_nm': z['led_wavelength_nm'],
                'target_size_range_nm': list(z['target_size_range_nm']),
                'expected_content': z['expected_content'],
            })

        return {
            "units": "mm",
            "classifier_type": "optical_buoyancy",
            "n_zones": p.n_zones,
            "global_dimensions": {
                "total_length": float(o.total_length_mm),
                "total_width": float(o.total_width_mm),
                "total_height": float(o.total_height_mm),
            },
            "zone": {
                "length": float(p.zone_length_mm),
                "height": float(p.zone_height_mm),
                "width": float(p.zone_width_mm),
            },
            "channel": {
                "height": float(p.channel_height_mm),
                "wall_thickness": float(p.wall_thickness_mm),
            },
            "barriers": {
                "pore_diameter_um": float(p.barrier_pore_um),
                "diameter_mm": float(p.barrier_diameter_mm),
                "material": "mesh_steel",
            },
            "leds": {
                "wavelengths_nm": [float(w) for w in p.led_wavelengths_nm],
                "power_mw": float(p.led_power_mw),
                "array_count": p.led_array_count,
            },
            "collection_ports": {
                "top_diameter_mm": float(p.top_port_diameter_mm),
                "bottom_diameter_mm": float(p.bottom_port_diameter_mm),
            },
            "connections": {
                "inlet_diameter": float(p.inlet_diameter_mm),
                "outlet_diameter": float(p.outlet_diameter_mm),
                "waste_diameter": float(p.waste_diameter_mm),
            },
            "zones": zones_cad,
            "performance": {
                "total_volume_ml": float(o.total_volume_ml),
                "residence_time_s": float(o.residence_time_s),
                "avg_zone_efficiency": float(o.avg_zone_efficiency),
                "avg_selectivity": float(o.avg_selectivity),
                "overall_recovery": float(o.overall_recovery),
                "overall_purity": float(o.overall_purity),
            },
            "material": p.body_material,
        }


# ===============================================================================
#  PROGRAMA PRINCIPAL
# ===============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diseno parametrico del clasificador por flotabilidad optica para CQDs"
    )
    parser.add_argument("--design", action="store_true",
                        help="Ejecutar diseno con parametros por defecto")
    parser.add_argument("--optimize", action="store_true",
                        help="Ejecutar optimizacion por busqueda aleatoria")
    parser.add_argument("--compare", action="store_true",
                        help="Comparar los 3 modos de excitacion lado a lado")
    parser.add_argument("--mode", type=str, default="optothermal",
                        choices=["led", "laser", "optothermal"],
                        help="Modo de excitacion: led, laser, optothermal (default)")
    parser.add_argument("--export-cad", type=str,
                        help="Exportar parametros CAD a archivo JSON")
    args = parser.parse_args()

    # Mapear argumento de modo a enum
    mode_map = {
        "led": ExcitationMode.LED,
        "laser": ExcitationMode.LASER_DIRECT,
        "optothermal": ExcitationMode.OPTOTHERMAL,
    }
    selected_mode = mode_map[args.mode]

    print("=" * 80)
    print("  CLASIFICADOR POR FLOTABILIDAD OPTICA PARA CQDs")
    print("  Separacion selectiva: LED / Laser directo / Opto-termico")
    print("=" * 80)

    if args.compare:
        ClassifierDesigner.compare_modes()

    elif args.optimize:
        print(f"\n-> Optimizando configuracion del clasificador (modo: {args.mode})...")
        print("  (Esto puede tardar unos segundos...)\n")

        designer = ClassifierDesigner(ClassifierParameters(excitation_mode=selected_mode))
        best_params, best_output = designer.optimize(max_iterations=500)

        # Crear disenador con mejores parametros
        best_params.excitation_mode = selected_mode
        designer = ClassifierDesigner(best_params)
        designer.design()
        designer.print_design_report()

        if args.export_cad:
            cad_params = designer.export_cad_parameters()
            with open(args.export_cad, 'w') as f:
                json.dump(cad_params, f, indent=2)
            print(f"\n-> Parametros CAD exportados a: {args.export_cad}")

    elif args.design:
        print(f"\n-> Ejecutando diseno (modo: {args.mode})...")

        params = ClassifierParameters(excitation_mode=selected_mode)
        designer = ClassifierDesigner(params)
        designer.design()
        designer.print_design_report()

        if args.export_cad:
            cad_params = designer.export_cad_parameters()
            with open(args.export_cad, 'w') as f:
                json.dump(cad_params, f, indent=2)
            print(f"\n-> Parametros CAD exportados a: {args.export_cad}")

    else:
        # Sin argumentos: mostrar diseno con modo seleccionado
        print(f"\n-> Ejecutando diseno (modo: {args.mode})...")

        params = ClassifierParameters(excitation_mode=selected_mode)
        designer = ClassifierDesigner(params)
        designer.design()
        designer.print_design_report()

    print("\n" + "=" * 80)
    print("  Uso: python classifier_design.py --design --mode optothermal")
    print("       python classifier_design.py --design --mode led")
    print("       python classifier_design.py --design --mode laser")
    print("       python classifier_design.py --compare")
    print("       python classifier_design.py --optimize --mode optothermal")
    print("       python classifier_design.py --export-cad classifier_cad.json")
    print("=" * 80)
