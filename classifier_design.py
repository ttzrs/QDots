#!/usr/bin/env python3
"""
===============================================================================
  DISENO PARAMETRICO DEL CLASIFICADOR LABERINTO PARA CQDs
  Separacion por tamano con membranas en cascada y discriminacion por PL
===============================================================================

  Este modulo calcula los parametros optimos de construccion para un
  clasificador post-reactor tipo laberinto con 5 etapas de membrana.

  Principio de operacion:
    - Fluido post-reactor entra al laberinto con mezcla de CQDs + debris
    - 5 membranas en cascada con tamano de poro decreciente
    - Cada membrana devia particulas mayores a camara de coleccion inferior
    - Cada camara tiene LED UV/azul + fotodetector para discriminacion PL
    - CQDs fluorescen bajo excitacion; debris no fluorescente se descarta
    - Producto final: fracciones de CQDs clasificadas por tamano/emision

  Variables de optimizacion:
    - Numero y tamano de poros de membrana
    - Geometria de camaras de coleccion
    - Potencia y longitud de onda de excitacion
    - Caudal y presion de operacion
    - Configuracion de deteccion PL

  Basado en:
    - Modelo de transporte impedido (Zeman-Wales)
    - Ecuacion de Hagen-Poiseuille para poros cilindricos
    - Modelo de confinamiento cuantico validado por VQE
    - Fisica de fotoluminiscencia de CQDs

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
VISCOSITY_WATER = 0.001        # Viscosidad del agua a 25C (Pa.s)
DENSITY_WATER = 998            # Densidad del agua a 25C (kg/m3)

# Del modelo VQE y reactor_control.py
E_BULK_EV = 1.50               # Gap del grafeno dopado N (eV)
A_CONFINEMENT = 7.26           # Constante de confinamiento (eV*nm^2)
EV_TO_NM = 1240                # Conversion energia-longitud de onda

# Fuentes de excitacion
UV_LED_WAVELENGTH_NM = 365     # LED ultravioleta estandar
BLUE_LED_WAVELENGTH_NM = 405   # LED azul-violeta estandar

# Propiedades de CQDs (de literatura y simulaciones)
CQD_DENSITY_KG_M3 = 1800      # Densidad tipica de carbon quantum dots
CQD_ABSORPTION_CROSS_SECTION = 1e-16  # Seccion eficaz de absorcion (cm^2) a 405 nm
CQD_QUANTUM_YIELD = 0.40      # Rendimiento cuantico tipico de CQDs
CQD_WALL_LOSS_PER_STAGE = 0.03  # Perdida por adsorcion en pared por etapa (3%)

# Limites fisicos
MIN_PORE_DIAMETER_NM = 1.0    # Poro minimo practico
MAX_PORE_DIAMETER_NM = 50000  # Poro maximo practico (50 um)
PLANCK_CONSTANT = 6.626e-34   # Constante de Planck (J.s)
SPEED_OF_LIGHT = 3e8          # Velocidad de la luz (m/s)


class MembraneType(Enum):
    """Tipos de membrana disponibles"""
    TRACK_ETCHED = "track_etched"        # Policarbonato con poros cilindricos
    ANODIC_ALUMINA = "anodic_alumina"    # Alumina anodicaordenada
    ULTRAFILTRATION = "ultrafiltration"  # Ceramica de ultrafiltracion


class SeparationMode(Enum):
    """Modos de separacion en las bifurcaciones"""
    ACTIVE_VALVE = "active_valve"    # Valvula activa controlada por PL
    PASSIVE_FLOW = "passive_flow"    # Flujo pasivo por gravedad/presion


class ExcitationSource(Enum):
    """Fuentes de excitacion para fotoluminiscencia"""
    UV_LED_365 = "uv_led_365"       # LED UV 365 nm
    BLUE_LED_405 = "blue_led_405"   # LED azul 405 nm


# ===============================================================================
#  ESTRUCTURAS DE DATOS
# ===============================================================================

@dataclass
class MembraneSpec:
    """Especificacion de una membrana comercial"""
    name: str
    pore_diameter_nm: float           # Diametro nominal de poro
    membrane_thickness_um: float      # Espesor de la membrana
    porosity: float                   # Fraccion de area abierta (0-1)
    material: str                     # "polycarbonate", "alumina", "ceramic"
    diameter_mm: float                # Diametro fisico de la membrana
    max_pressure_bar: float           # Presion maxima de operacion
    chemical_resistance: bool         # Resistencia a solventes
    uv_transparent: bool              # Transparencia a UV (para PL in-situ)


@dataclass
class StageSpec:
    """Especificacion de una etapa del clasificador"""
    stage_number: int
    membrane: MembraneSpec
    target_particle_range_nm: Tuple[float, float]  # (min, max) capturado
    expected_wavelength_range_nm: Tuple[float, float]  # Rango de emision
    collection_volume_ml: float
    has_pl_detection: bool
    expected_content: str  # "debris", "large_aggregates", "red_qdots", etc.


@dataclass
class ClassifierParameters:
    """Parametros completos del clasificador laberinto"""
    # Topologia del laberinto
    n_stages: int = 5
    channel_width_mm: float = 2.0       # Ancho del canal principal
    channel_height_mm: float = 1.0      # Altura del canal principal
    wall_thickness_mm: float = 2.0      # Espesor de paredes
    stage_pitch_mm: float = 25.0        # Distancia entre etapas

    # Membranas
    membrane_diameter_mm: float = 13.0  # Nuclepore estandar 13 mm
    membrane_seat_depth_mm: float = 0.5 # Profundidad del asiento de membrana

    # Camaras de coleccion
    chamber_width_mm: float = 10.0      # Ancho de camara inferior
    chamber_length_mm: float = 15.0     # Largo de camara inferior
    chamber_depth_mm: float = 5.0       # Profundidad de camara inferior

    # Optica
    window_diameter_mm: float = 8.0     # Ventana de cuarzo para PL
    window_thickness_mm: float = 1.0    # Espesor de ventana
    led_diameter_mm: float = 5.0        # Diametro del LED
    detector_diameter_mm: float = 5.0   # Diametro del fotodetector

    # Flujo
    flow_rate_ml_min: float = 5.0       # Caudal de entrada
    operating_pressure_bar: float = 0.5 # Presion de operacion
    temperature_c: float = 25.0         # Temperatura de operacion

    # Conexiones
    inlet_diameter_mm: float = 2.0      # Diametro de entrada
    outlet_diameter_mm: float = 1.5     # Diametro de salida
    waste_diameter_mm: float = 2.0      # Diametro de drenaje

    # Excitacion y deteccion
    excitation_source: ExcitationSource = ExcitationSource.BLUE_LED_405
    excitation_power_mw: float = 50.0   # Potencia del LED (mW)
    min_fluorescence_intensity: float = 0.3  # Umbral minimo de PL

    # Modo de operacion
    separation_mode: SeparationMode = SeparationMode.ACTIVE_VALVE

    # Material del cuerpo
    body_material: str = "ceramic_resin"


@dataclass
class ClassifierOutput:
    """Salida del diseno del clasificador"""
    # Dimensiones globales
    total_length_mm: float = 0.0
    total_width_mm: float = 0.0
    total_height_mm: float = 0.0

    # Etapas
    stages: list = field(default_factory=list)  # Lista de dicts por etapa

    # Hidraulica
    total_pressure_drop_bar: float = 0.0
    total_volume_ml: float = 0.0
    residence_time_s: float = 0.0

    # Rendimiento de separacion
    size_separation_efficiency: float = 0.0
    pl_discrimination_efficiency: float = 0.0
    overall_recovery: float = 0.0
    overall_purity: float = 0.0

    # Puntuaciones de diseno
    efficiency_score: float = 0.0
    cost_score: float = 0.0
    feasibility_score: float = 0.0


# ===============================================================================
#  BASE DE DATOS DE MEMBRANAS COMERCIALES
# ===============================================================================

DEFAULT_MEMBRANES = [
    MembraneSpec(
        name="Nuclepore 10um",
        pore_diameter_nm=10000,
        membrane_thickness_um=10,
        porosity=0.15,
        material="polycarbonate",
        diameter_mm=13.0,
        max_pressure_bar=2.0,
        chemical_resistance=True,
        uv_transparent=False,
    ),
    MembraneSpec(
        name="Nuclepore 5um",
        pore_diameter_nm=5000,
        membrane_thickness_um=10,
        porosity=0.12,
        material="polycarbonate",
        diameter_mm=13.0,
        max_pressure_bar=2.0,
        chemical_resistance=True,
        uv_transparent=False,
    ),
    MembraneSpec(
        name="Anodisc 20nm",
        pore_diameter_nm=20,
        membrane_thickness_um=60,
        porosity=0.50,
        material="alumina",
        diameter_mm=13.0,
        max_pressure_bar=3.0,
        chemical_resistance=True,
        uv_transparent=True,
    ),
    MembraneSpec(
        name="Anodisc 13nm",
        pore_diameter_nm=13,
        membrane_thickness_um=60,
        porosity=0.45,
        material="alumina",
        diameter_mm=13.0,
        max_pressure_bar=3.0,
        chemical_resistance=True,
        uv_transparent=True,
    ),
    MembraneSpec(
        name="UF Ceramica 3nm",
        pore_diameter_nm=3,
        membrane_thickness_um=2000,
        porosity=0.35,
        material="ceramic",
        diameter_mm=13.0,
        max_pressure_bar=5.0,
        chemical_resistance=True,
        uv_transparent=False,
    ),
]


def size_to_wavelength(diameter_nm: float) -> float:
    """
    Convierte diametro de CQD a longitud de onda de emision.
    Usa modelo de confinamiento cuantico: E_gap = E_bulk + A/d^2
    """
    if diameter_nm <= 0:
        return 0.0
    gap_ev = E_BULK_EV + A_CONFINEMENT / (diameter_nm ** 2)
    return EV_TO_NM / gap_ev


def wavelength_to_size(wavelength_nm: float) -> float:
    """
    Convierte longitud de onda de emision a diametro de CQD.
    Inversa del modelo de confinamiento cuantico.
    """
    if wavelength_nm <= 0:
        return 0.0
    gap_ev = EV_TO_NM / wavelength_nm
    if gap_ev <= E_BULK_EV:
        return float('inf')  # Particula bulk
    return np.sqrt(A_CONFINEMENT / (gap_ev - E_BULK_EV))


# ===============================================================================
#  MOTOR DE DISENO DEL CLASIFICADOR
# ===============================================================================

class ClassifierDesigner:
    """
    Motor de diseno parametrico del clasificador laberinto.
    Optimiza geometria y membranas para maximizar separacion y pureza de CQDs.
    """

    def __init__(self, params: Optional[ClassifierParameters] = None,
                 membranes: Optional[List[MembraneSpec]] = None):
        self.params = params or ClassifierParameters()
        self.membranes = membranes or DEFAULT_MEMBRANES
        self.output = ClassifierOutput()

        # Verificar que hay suficientes membranas para las etapas
        if len(self.membranes) < self.params.n_stages:
            raise ValueError(
                f"Se necesitan {self.params.n_stages} membranas pero solo "
                f"hay {len(self.membranes)} disponibles"
            )

    def calculate_maze_geometry(self) -> Dict:
        """
        Calcula parametros geometricos del laberinto completo.
        Incluye dimensiones totales, volumenes y coordenadas por etapa.
        """
        p = self.params

        # Longitud total: n etapas * pitch + paredes laterales
        total_length_mm = p.n_stages * p.stage_pitch_mm + 2 * p.wall_thickness_mm

        # Ancho total: canal principal + camara lateral + paredes
        total_width_mm = (p.channel_width_mm + p.chamber_width_mm +
                          3 * p.wall_thickness_mm)

        # Altura total: canal + camara inferior + ventana + paredes
        total_height_mm = (p.channel_height_mm + p.chamber_depth_mm +
                           p.window_thickness_mm + 3 * p.wall_thickness_mm)

        # Volumen del canal principal
        channel_volume_mm3 = (p.channel_width_mm * p.channel_height_mm *
                              (total_length_mm - 2 * p.wall_thickness_mm))
        channel_volume_ml = channel_volume_mm3 / 1000.0

        # Volumen de cada camara de coleccion
        chamber_volume_mm3 = (p.chamber_width_mm * p.chamber_length_mm *
                              p.chamber_depth_mm)
        chamber_volume_ml = chamber_volume_mm3 / 1000.0

        # Volumen interno total
        total_volume_ml = channel_volume_ml + p.n_stages * chamber_volume_ml

        # Coordenadas de cada etapa (centro de la membrana)
        stage_positions = []
        for i in range(p.n_stages):
            x = p.wall_thickness_mm + (i + 0.5) * p.stage_pitch_mm
            y = p.wall_thickness_mm + p.channel_width_mm / 2.0
            z = p.wall_thickness_mm + p.channel_height_mm
            stage_positions.append({
                'stage': i + 1,
                'x_mm': x,
                'y_mm': y,
                'z_mm': z,
                'membrane_center_x': x,
                'chamber_center_x': x,
                'chamber_center_z': z + p.chamber_depth_mm / 2.0,
            })

        return {
            'total_length_mm': total_length_mm,
            'total_width_mm': total_width_mm,
            'total_height_mm': total_height_mm,
            'channel_volume_ml': channel_volume_ml,
            'chamber_volume_ml': chamber_volume_ml,
            'total_volume_ml': total_volume_ml,
            'stage_positions': stage_positions,
            'n_stages': p.n_stages,
        }

    def calculate_membrane_stages(self) -> List[Dict]:
        """
        Calcula rendimiento de cada etapa de membrana.
        Incluye caida de presion, probabilidad de captura y rango de particulas.

        Modelo de presion: Hagen-Poiseuille para poros cilindricos
        Modelo de captura: transporte impedido de Zeman-Wales
        """
        p = self.params
        stages = []

        # Caudal en m^3/s
        Q_m3s = p.flow_rate_ml_min / 60.0 * 1e-6

        # Area de membrana activa
        A_membrane_m2 = np.pi * (p.membrane_diameter_mm / 2.0 * 1e-3) ** 2

        # Rango de tamanos de particulas a considerar (1-15000 nm)
        test_sizes_nm = np.logspace(0, 4.2, 200)

        for i, membrane in enumerate(self.membranes[:p.n_stages]):
            # --- Caida de presion por Hagen-Poiseuille ---
            d_pore_m = membrane.pore_diameter_nm * 1e-9
            r_pore_m = d_pore_m / 2.0
            L_membrane_m = membrane.membrane_thickness_um * 1e-6

            # Numero de poros
            A_single_pore = np.pi * r_pore_m ** 2
            N_pores = membrane.porosity * A_membrane_m2 / A_single_pore
            N_pores = max(1, N_pores)

            # Caudal por poro
            Q_per_pore = Q_m3s / N_pores

            # dP = 128 * mu * Q * L / (N * pi * d^4) o equivalente por poro
            dP_pa = (128 * VISCOSITY_WATER * Q_per_pore * L_membrane_m /
                     (np.pi * d_pore_m ** 4))
            dP_bar = dP_pa / 1e5

            # --- Probabilidad de captura (Zeman-Wales) ---
            # Para cada tamano de particula de prueba
            capture_probs = []
            for d_part_nm in test_sizes_nm:
                lambda_r = d_part_nm / membrane.pore_diameter_nm
                if lambda_r >= 1.0:
                    # Particula mas grande que el poro: rechazo total
                    R = 1.0
                else:
                    # Modelo de transporte impedido
                    # R = 1 - (1 - lambda_r)^2 * (2 - (1 - lambda_r)^2)
                    term = (1.0 - lambda_r) ** 2
                    R = 1.0 - term * (2.0 - term)
                capture_probs.append(R)

            capture_probs = np.array(capture_probs)

            # Rango de particulas capturadas (R > 0.5)
            captured_mask = capture_probs > 0.5
            if np.any(captured_mask):
                captured_sizes = test_sizes_nm[captured_mask]
                min_captured_nm = float(captured_sizes.min())
                max_captured_nm = float(captured_sizes.max())
            else:
                min_captured_nm = membrane.pore_diameter_nm
                max_captured_nm = MAX_PORE_DIAMETER_NM

            # Longitud de onda de emision esperada para el rango capturado
            # Solo relevante para CQDs (< 20 nm tipicamente)
            if min_captured_nm < 100:
                wl_max = size_to_wavelength(max(1.0, min_captured_nm))
                wl_min = size_to_wavelength(min(20.0, max_captured_nm))
                # wl_max corresponde a particula mas pequena (mayor gap)
                # wl_min corresponde a particula mas grande (menor gap)
                expected_wl_range = (min(wl_min, wl_max), max(wl_min, wl_max))
            else:
                expected_wl_range = (0.0, 0.0)  # No son CQDs

            # Clasificacion del contenido esperado
            pore_nm = membrane.pore_diameter_nm
            if pore_nm >= 5000:
                content = "debris_grande"
            elif pore_nm >= 1000:
                content = "agregados_grandes"
            elif pore_nm >= 15:
                content = "qdots_rojos"    # CQDs grandes, emision roja
            elif pore_nm >= 8:
                content = "qdots_azules"   # CQDs medios, emision azul
            else:
                content = "qdots_uv"       # CQDs pequenos, emision UV

            # Eficiencia de captura media en el rango objetivo
            if min_captured_nm < max_captured_nm:
                target_mask = ((test_sizes_nm >= min_captured_nm) &
                               (test_sizes_nm <= max_captured_nm))
                if np.any(target_mask):
                    avg_capture_eff = float(np.mean(capture_probs[target_mask]))
                else:
                    avg_capture_eff = 0.5
            else:
                avg_capture_eff = 0.5

            # Tipo de membrana
            if membrane.material == "polycarbonate":
                membrane_type = MembraneType.TRACK_ETCHED
            elif membrane.material == "alumina":
                membrane_type = MembraneType.ANODIC_ALUMINA
            else:
                membrane_type = MembraneType.ULTRAFILTRATION

            stages.append({
                'stage_number': i + 1,
                'membrane_name': membrane.name,
                'membrane_type': membrane_type.value,
                'pore_diameter_nm': membrane.pore_diameter_nm,
                'membrane_thickness_um': membrane.membrane_thickness_um,
                'n_pores': N_pores,
                'pressure_drop_bar': dP_bar,
                'pressure_drop_pa': dP_pa,
                'captured_range_nm': (min_captured_nm, max_captured_nm),
                'expected_wavelength_range_nm': expected_wl_range,
                'avg_capture_efficiency': avg_capture_eff,
                'expected_content': content,
                'capture_probs': capture_probs,
                'test_sizes_nm': test_sizes_nm,
                'has_pl_detection': pore_nm < 100,
            })

        return stages

    def calculate_photoluminescence(self, membrane_stages: List[Dict]) -> List[Dict]:
        """
        Calcula parametros del sistema de deteccion de fotoluminiscencia
        para cada etapa con poros < 100 nm (etapas relevantes para CQDs).

        Modelo: excitacion LED -> absorcion -> emision fluorescente -> deteccion
        """
        p = self.params
        pl_results = []

        # Longitud de onda de excitacion
        if p.excitation_source == ExcitationSource.UV_LED_365:
            excitation_wl_nm = UV_LED_WAVELENGTH_NM
        else:
            excitation_wl_nm = BLUE_LED_WAVELENGTH_NM

        # Energia por foton de excitacion
        E_photon_J = PLANCK_CONSTANT * SPEED_OF_LIGHT / (excitation_wl_nm * 1e-9)

        # Flujo de fotones (fotones/s/cm^2)
        power_W = p.excitation_power_mw * 1e-3
        # Area iluminada (aprox diametro del LED sobre la ventana)
        illumination_area_cm2 = np.pi * (p.window_diameter_mm / 20.0) ** 2
        photon_flux = power_W / E_photon_J / illumination_area_cm2

        for stage in membrane_stages:
            pore_nm = stage['pore_diameter_nm']

            # Solo calcular PL para etapas con CQDs potenciales
            if pore_nm >= 100:
                pl_results.append({
                    'stage_number': stage['stage_number'],
                    'has_pl': False,
                    'reason': 'Poro demasiado grande para CQDs',
                })
                continue

            # Tamano representativo de CQD en esta etapa
            captured_min, captured_max = stage['captured_range_nm']
            representative_size_nm = (captured_min + min(captured_max, 20.0)) / 2.0
            representative_size_nm = max(1.0, min(20.0, representative_size_nm))

            # Masa por particula
            radius_m = representative_size_nm * 1e-9 / 2.0
            volume_m3 = (4.0 / 3.0) * np.pi * radius_m ** 3
            mass_per_particle_kg = CQD_DENSITY_KG_M3 * volume_m3

            # Concentracion estimada (asumiendo 0.1 mg/mL de CQDs en solucion)
            concentration_kg_m3 = 0.1  # 0.1 mg/mL = 0.1 kg/m3
            number_density_m3 = concentration_kg_m3 / mass_per_particle_kg

            # Numero de particulas en volumen iluminado
            # Volumen iluminado: area de ventana * profundidad de camara
            illuminated_volume_m3 = (illumination_area_cm2 * 1e-4 *
                                     p.chamber_depth_mm * 1e-3)
            n_particles = number_density_m3 * illuminated_volume_m3

            # Tasa de emision por particula
            # Rate = sigma_abs * photon_flux * QY
            emission_rate_per_particle = (CQD_ABSORPTION_CROSS_SECTION *
                                          photon_flux * CQD_QUANTUM_YIELD)

            # Tasa de emision total
            total_emission_rate = emission_rate_per_particle * n_particles

            # Longitud de onda de emision
            emission_wl_nm = size_to_wavelength(representative_size_nm)

            # Senal en el detector (simplificado)
            # Factor de coleccion geometrica (fraccion solida del detector)
            collection_solid_angle = np.pi * (p.detector_diameter_mm / 2.0) ** 2 / \
                                     (4.0 * np.pi * (p.chamber_depth_mm) ** 2)
            collection_solid_angle = min(0.1, collection_solid_angle)  # Max 10%

            signal_photons_per_s = total_emission_rate * collection_solid_angle

            # Ruido (shot noise + dark current del fotodetector)
            dark_count_rate = 100  # Cuentas/s tipico para fotodiodo Si
            # Ruido de fondo por scattering del LED
            background_rate = photon_flux * illumination_area_cm2 * 1e-8 * \
                              collection_solid_angle
            noise_rate = np.sqrt(dark_count_rate + background_rate)

            # SNR
            snr = signal_photons_per_s / max(1.0, noise_rate)

            # Discriminacion PL: SNR > 3 es detectable
            pl_detectable = snr > 3.0

            # Stokes shift (diferencia entre excitacion y emision)
            stokes_shift_nm = emission_wl_nm - excitation_wl_nm

            pl_results.append({
                'stage_number': stage['stage_number'],
                'has_pl': True,
                'representative_size_nm': representative_size_nm,
                'emission_wavelength_nm': emission_wl_nm,
                'excitation_wavelength_nm': excitation_wl_nm,
                'stokes_shift_nm': stokes_shift_nm,
                'n_particles_illuminated': n_particles,
                'emission_rate_per_particle': emission_rate_per_particle,
                'total_emission_rate': total_emission_rate,
                'signal_photons_per_s': signal_photons_per_s,
                'noise_rate': noise_rate,
                'snr': snr,
                'pl_detectable': pl_detectable,
                'collection_efficiency': collection_solid_angle,
                'mass_per_particle_kg': mass_per_particle_kg,
                'number_density_m3': number_density_m3,
            })

        return pl_results

    def calculate_separation_efficiency(self, membrane_stages: List[Dict],
                                        pl_results: List[Dict]) -> Dict:
        """
        Calcula eficiencia global de separacion del clasificador.

        Incluye:
          - Eficiencia de clasificacion por tamano (producto de capturas)
          - Eficiencia de discriminacion PL (basada en QY y SNR)
          - Recuperacion global (perdidas por pared, errores de clasificacion)
          - Pureza global (analisis bayesiano)
        """
        p = self.params

        # Eficiencia de clasificacion por tamano
        # Producto de las eficiencias de captura individuales
        stage_efficiencies = []
        for stage in membrane_stages:
            stage_efficiencies.append(stage['avg_capture_efficiency'])

        # La eficiencia global es el producto de capturas correctas
        # ponderado por la fraccion de particulas en cada rango
        size_separation_eff = float(np.mean(stage_efficiencies))

        # Eficiencia de discriminacion PL
        pl_stages = [r for r in pl_results if r.get('has_pl', False)]
        if pl_stages:
            snr_values = [r['snr'] for r in pl_stages]
            # Probabilidad de deteccion correcta basada en SNR
            # P(detect) = 1 - erfc(SNR / sqrt(2)) / 2  ~  erf(SNR/sqrt(2))
            detection_probs = []
            for snr in snr_values:
                if snr > 10:
                    detection_probs.append(0.99)
                elif snr > 3:
                    # Aproximacion con funcion error
                    from math import erf
                    detection_probs.append(erf(snr / np.sqrt(2)))
                else:
                    detection_probs.append(0.5)  # Basicamente adivinanza
            pl_discrimination_eff = float(np.mean(detection_probs))
        else:
            pl_discrimination_eff = 0.0

        # Recuperacion global
        # P(correcto) * P(detectado) * (1 - wall_loss)^n_stages
        wall_survival = (1.0 - CQD_WALL_LOSS_PER_STAGE) ** p.n_stages
        p_correct_stage = size_separation_eff
        p_detected = pl_discrimination_eff if pl_stages else 1.0

        overall_recovery = p_correct_stage * p_detected * wall_survival

        # Pureza global (analisis bayesiano)
        # Asumimos que la fraccion de CQDs en la solucion es ~30%
        # y el resto es debris/subproductos
        qdot_fraction = 0.30
        tp_rate = size_separation_eff * (pl_discrimination_eff if pl_stages else 0.8)
        fp_rate = (1.0 - size_separation_eff) * 0.05  # Debris que pasa erroneamente

        # Pureza = TP / (TP + FP)  via Bayes
        tp = tp_rate * qdot_fraction
        fp = fp_rate * (1.0 - qdot_fraction)
        overall_purity = tp / max(1e-10, tp + fp)

        return {
            'size_separation_efficiency': size_separation_eff,
            'pl_discrimination_efficiency': pl_discrimination_eff,
            'wall_survival_factor': wall_survival,
            'overall_recovery': overall_recovery,
            'overall_purity': overall_purity,
            'stage_efficiencies': stage_efficiencies,
            'qdot_fraction_assumed': qdot_fraction,
        }

    def calculate_pressure_drop(self, geometry: Dict,
                                membrane_stages: List[Dict]) -> Dict:
        """
        Calcula caida de presion total del sistema.

        Componentes:
          - Canal principal (Hagen-Poiseuille rectangular)
          - Membranas (ya calculado por etapa)
          - Perdidas menores en bifurcaciones
        """
        p = self.params

        # Caudal en m^3/s
        Q_m3s = p.flow_rate_ml_min / 60.0 * 1e-6

        # --- Caida de presion en el canal (flujo rectangular) ---
        # dP = (12 * mu * Q * L) / (w * h^3)
        w_m = p.channel_width_mm * 1e-3
        h_m = p.channel_height_mm * 1e-3
        L_channel_m = (geometry['total_length_mm'] - 2 * p.wall_thickness_mm) * 1e-3

        dP_channel_pa = (12.0 * VISCOSITY_WATER * Q_m3s * L_channel_m /
                         (w_m * h_m ** 3))
        dP_channel_bar = dP_channel_pa / 1e5

        # --- Caida de presion en membranas (acumulada) ---
        dP_membranes_bar = sum(s['pressure_drop_bar'] for s in membrane_stages)
        dP_membranes_pa = sum(s['pressure_drop_pa'] for s in membrane_stages)

        # --- Perdidas menores en bifurcaciones ---
        # Coeficiente de perdida menor: K = 0.5 por bifurcacion tipica
        K_minor = 0.5
        velocity_m_s = Q_m3s / (w_m * h_m)
        dP_minor_per_stage = 0.5 * DENSITY_WATER * K_minor * velocity_m_s ** 2
        dP_minor_total_pa = dP_minor_per_stage * p.n_stages
        dP_minor_total_bar = dP_minor_total_pa / 1e5

        # --- Total ---
        total_dP_pa = dP_channel_pa + dP_membranes_pa + dP_minor_total_pa
        total_dP_bar = total_dP_pa / 1e5

        # Verificar si es factible con presion de operacion
        pressure_ok = total_dP_bar <= p.operating_pressure_bar

        # Tiempo de residencia
        total_volume_m3 = geometry['total_volume_ml'] * 1e-6
        residence_time_s = total_volume_m3 / Q_m3s

        # Velocidad en el canal
        channel_velocity_mm_s = velocity_m_s * 1e3

        # Numero de Reynolds
        D_h = 2 * w_m * h_m / (w_m + h_m)  # Diametro hidraulico
        Re = DENSITY_WATER * velocity_m_s * D_h / VISCOSITY_WATER

        return {
            'dP_channel_bar': dP_channel_bar,
            'dP_membranes_bar': dP_membranes_bar,
            'dP_minor_bar': dP_minor_total_bar,
            'total_pressure_drop_bar': total_dP_bar,
            'pressure_feasible': pressure_ok,
            'residence_time_s': residence_time_s,
            'channel_velocity_mm_s': channel_velocity_mm_s,
            'reynolds_number': Re,
            'flow_regime': 'laminar' if Re < 2300 else 'turbulento',
            'per_stage_membrane_dP': [s['pressure_drop_bar'] for s in membrane_stages],
        }

    def calculate_scores(self, separation: Dict, pressure: Dict,
                         pl_results: List[Dict]) -> Dict:
        """Calcula puntuaciones de diseno (0-100)"""
        p = self.params

        # --- Puntuacion de eficiencia (0-100) ---
        efficiency_score = 0.0

        # Recuperacion (40 puntos max)
        efficiency_score += min(40.0, separation['overall_recovery'] * 40.0)

        # Pureza (40 puntos max)
        efficiency_score += min(40.0, separation['overall_purity'] * 40.0)

        # Discriminacion PL (20 puntos max)
        pl_stages_with_signal = [r for r in pl_results
                                 if r.get('has_pl') and r.get('pl_detectable')]
        total_pl_stages = [r for r in pl_results if r.get('has_pl')]
        if total_pl_stages:
            pl_fraction = len(pl_stages_with_signal) / len(total_pl_stages)
            efficiency_score += pl_fraction * 20.0
        else:
            efficiency_score += 10.0  # Parcial sin PL

        efficiency_score = min(100.0, max(0.0, efficiency_score))

        # --- Puntuacion de costo (0-100) ---
        cost_score = 100.0

        # Numero de etapas (mas etapas = mas caro)
        cost_score -= max(0, (p.n_stages - 3)) * 8.0

        # Membranas especiales (alumina y ceramica son mas caras)
        for membrane in self.membranes[:p.n_stages]:
            if membrane.material == "alumina":
                cost_score -= 5.0
            elif membrane.material == "ceramic":
                cost_score -= 8.0

        # Sistema PL (LEDs + detectores)
        n_pl_stages = len(total_pl_stages)
        cost_score -= n_pl_stages * 5.0  # Cada sistema PL cuesta

        # Potencia de excitacion alta
        if p.excitation_power_mw > 100:
            cost_score -= 10.0

        # Modo activo es mas caro
        if p.separation_mode == SeparationMode.ACTIVE_VALVE:
            cost_score -= 10.0

        cost_score = min(100.0, max(0.0, cost_score))

        # --- Puntuacion de factibilidad (0-100) ---
        feasibility_score = 100.0

        # Presion factible
        if not pressure['pressure_feasible']:
            excess = pressure['total_pressure_drop_bar'] / p.operating_pressure_bar
            feasibility_score -= min(40.0, (excess - 1.0) * 20.0)

        # Flujo laminar (deseable para separacion limpia)
        if pressure['flow_regime'] == 'turbulento':
            feasibility_score -= 15.0

        # Tiempo de residencia razonable (5-120 s)
        if pressure['residence_time_s'] < 5:
            feasibility_score -= 10.0
        elif pressure['residence_time_s'] > 120:
            feasibility_score -= 10.0

        # SNR suficiente en etapas PL
        for r in pl_results:
            if r.get('has_pl') and not r.get('pl_detectable', True):
                feasibility_score -= 5.0

        # Tamano total razonable (< 200 mm)
        if self.output.total_length_mm > 200:
            feasibility_score -= 10.0

        feasibility_score = min(100.0, max(0.0, feasibility_score))

        return {
            'efficiency_score': efficiency_score,
            'cost_score': cost_score,
            'feasibility_score': feasibility_score,
            'overall_score': (efficiency_score + cost_score + feasibility_score) / 3.0,
        }

    def design(self) -> ClassifierOutput:
        """Ejecuta el diseno completo del clasificador"""
        # 1. Geometria del laberinto
        geometry = self.calculate_maze_geometry()

        # 2. Etapas de membrana
        membrane_stages = self.calculate_membrane_stages()

        # 3. Sistema de fotoluminiscencia
        pl_results = self.calculate_photoluminescence(membrane_stages)

        # 4. Eficiencia de separacion
        separation = self.calculate_separation_efficiency(membrane_stages, pl_results)

        # 5. Caida de presion
        pressure = self.calculate_pressure_drop(geometry, membrane_stages)

        # Poblar salida parcialmente para que calculate_scores pueda usar dimensiones
        self.output.total_length_mm = geometry['total_length_mm']
        self.output.total_width_mm = geometry['total_width_mm']
        self.output.total_height_mm = geometry['total_height_mm']

        # 6. Puntuaciones
        scores = self.calculate_scores(separation, pressure, pl_results)

        # Poblar salida completa
        self.output = ClassifierOutput(
            total_length_mm=geometry['total_length_mm'],
            total_width_mm=geometry['total_width_mm'],
            total_height_mm=geometry['total_height_mm'],
            stages=membrane_stages,
            total_pressure_drop_bar=pressure['total_pressure_drop_bar'],
            total_volume_ml=geometry['total_volume_ml'],
            residence_time_s=pressure['residence_time_s'],
            size_separation_efficiency=separation['size_separation_efficiency'],
            pl_discrimination_efficiency=separation['pl_discrimination_efficiency'],
            overall_recovery=separation['overall_recovery'],
            overall_purity=separation['overall_purity'],
            efficiency_score=scores['efficiency_score'],
            cost_score=scores['cost_score'],
            feasibility_score=scores['feasibility_score'],
        )

        # Almacenar resultados intermedios para el reporte
        self._geometry = geometry
        self._membrane_stages = membrane_stages
        self._pl_results = pl_results
        self._separation = separation
        self._pressure = pressure
        self._scores = scores

        return self.output

    def optimize(self, max_iterations: int = 500) -> Tuple[ClassifierParameters, ClassifierOutput]:
        """
        Optimiza parametros del clasificador por busqueda aleatoria.
        Busca maximizar la puntuacion global (eficiencia + costo + factibilidad).
        """
        best_params = None
        best_output = None
        best_score = -1.0

        # Espacios de busqueda
        search_space = {
            'n_stages': [3, 4, 5],
            'channel_width_mm': [1.5, 2.0, 2.5, 3.0],
            'channel_height_mm': [0.5, 1.0, 1.5, 2.0],
            'stage_pitch_mm': [20.0, 25.0, 30.0, 35.0],
            'chamber_width_mm': [8.0, 10.0, 12.0, 15.0],
            'chamber_length_mm': [12.0, 15.0, 18.0, 20.0],
            'chamber_depth_mm': [3.0, 5.0, 7.0, 10.0],
            'flow_rate_ml_min': [1.0, 2.0, 5.0, 8.0, 10.0],
            'operating_pressure_bar': [0.3, 0.5, 1.0, 2.0],
            'excitation_power_mw': [20.0, 50.0, 100.0],
            'excitation_source': [ExcitationSource.UV_LED_365,
                                  ExcitationSource.BLUE_LED_405],
            'separation_mode': [SeparationMode.ACTIVE_VALVE,
                                SeparationMode.PASSIVE_FLOW],
        }

        np.random.seed(42)
        for _ in range(max_iterations):
            n_stages = int(np.random.choice(search_space['n_stages']))

            # Seleccionar membranas (siempre ordenadas de mayor a menor poro)
            if n_stages <= len(DEFAULT_MEMBRANES):
                # Seleccionar n_stages membranas del set disponible
                indices = sorted(np.random.choice(
                    len(DEFAULT_MEMBRANES), size=n_stages, replace=False
                ))
                selected_membranes = [DEFAULT_MEMBRANES[i] for i in indices]
                # Ordenar por tamano de poro decreciente
                selected_membranes.sort(key=lambda m: m.pore_diameter_nm,
                                        reverse=True)
            else:
                selected_membranes = DEFAULT_MEMBRANES[:n_stages]

            params = ClassifierParameters(
                n_stages=n_stages,
                channel_width_mm=float(np.random.choice(
                    search_space['channel_width_mm'])),
                channel_height_mm=float(np.random.choice(
                    search_space['channel_height_mm'])),
                stage_pitch_mm=float(np.random.choice(
                    search_space['stage_pitch_mm'])),
                chamber_width_mm=float(np.random.choice(
                    search_space['chamber_width_mm'])),
                chamber_length_mm=float(np.random.choice(
                    search_space['chamber_length_mm'])),
                chamber_depth_mm=float(np.random.choice(
                    search_space['chamber_depth_mm'])),
                flow_rate_ml_min=float(np.random.choice(
                    search_space['flow_rate_ml_min'])),
                operating_pressure_bar=float(np.random.choice(
                    search_space['operating_pressure_bar'])),
                excitation_power_mw=float(np.random.choice(
                    search_space['excitation_power_mw'])),
                excitation_source=np.random.choice(
                    search_space['excitation_source']),
                separation_mode=np.random.choice(
                    search_space['separation_mode']),
            )

            try:
                designer = ClassifierDesigner(params, selected_membranes)
                output = designer.design()

                # Score compuesto ponderado
                score = (output.efficiency_score * 0.40 +
                         output.feasibility_score * 0.35 +
                         output.cost_score * 0.25)

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_output = output
                    # Guardar membranas ganadoras
                    self._best_membranes = selected_membranes
            except (ValueError, ZeroDivisionError):
                continue

        return best_params, best_output

    def print_design_report(self):
        """Imprime reporte completo del diseno del clasificador"""
        p = self.params
        o = self.output

        print("\n" + "=" * 80)
        print("  REPORTE DE DISENO DEL CLASIFICADOR LABERINTO PARA CQDs")
        print("=" * 80)

        # --- 1. GEOMETRIA DEL LABERINTO ---
        print("\n+" + "-" * 78 + "+")
        print("|  1. GEOMETRIA DEL LABERINTO" + " " * 50 + "|")
        print("+" + "-" * 78 + "+")
        print(f"|  Numero de etapas:        {p.n_stages}" + " " * 51 + "|")
        print(f"|  Longitud total:          {o.total_length_mm:.1f} mm" + " " * (44 - len(f"{o.total_length_mm:.1f}") + 5) + "|")
        print(f"|  Ancho total:             {o.total_width_mm:.1f} mm" + " " * (44 - len(f"{o.total_width_mm:.1f}") + 5) + "|")
        print(f"|  Altura total:            {o.total_height_mm:.1f} mm" + " " * (44 - len(f"{o.total_height_mm:.1f}") + 5) + "|")
        print(f"|  Ancho de canal:          {p.channel_width_mm:.1f} mm" + " " * (44 - len(f"{p.channel_width_mm:.1f}") + 5) + "|")
        print(f"|  Altura de canal:         {p.channel_height_mm:.1f} mm" + " " * (44 - len(f"{p.channel_height_mm:.1f}") + 5) + "|")
        print(f"|  Pitch entre etapas:      {p.stage_pitch_mm:.1f} mm" + " " * (44 - len(f"{p.stage_pitch_mm:.1f}") + 5) + "|")
        print(f"|  Volumen total interno:   {o.total_volume_ml:.2f} mL" + " " * (43 - len(f"{o.total_volume_ml:.2f}") + 5) + "|")
        print(f"|  Tiempo de residencia:    {o.residence_time_s:.1f} s" + " " * (44 - len(f"{o.residence_time_s:.1f}") + 6) + "|")
        print("+" + "-" * 78 + "+")

        # --- 2. ETAPAS DE MEMBRANA ---
        print("\n+" + "-" * 78 + "+")
        print("|  2. ETAPAS DE MEMBRANA" + " " * 55 + "|")
        print("+" + "-" * 78 + "+")
        print("|  " + f"{'Etapa':<7} {'Membrana':<20} {'Poro':<12} {'dP (bar)':<12} {'Captura':<15}" + "  |")
        print("|  " + "-" * 74 + "  |")
        for stage in o.stages:
            pore_str = f"{stage['pore_diameter_nm']:.0f} nm"
            dp_str = f"{stage['pressure_drop_bar']:.4f}"
            cap_min, cap_max = stage['captured_range_nm']
            if cap_max > 10000:
                cap_str = f">{cap_min:.0f} nm"
            else:
                cap_str = f"{cap_min:.0f}-{cap_max:.0f} nm"
            print(f"|  {stage['stage_number']:<7} {stage['membrane_name']:<20} "
                  f"{pore_str:<12} {dp_str:<12} {cap_str:<15}  |")
        print("|  " + "-" * 74 + "  |")
        for stage in o.stages:
            wl_min, wl_max = stage['expected_wavelength_range_nm']
            if wl_min > 0:
                wl_str = f"{wl_min:.0f}-{wl_max:.0f} nm"
            else:
                wl_str = "N/A (debris)"
            content = stage['expected_content']
            pl_str = "Si" if stage['has_pl_detection'] else "No"
            print(f"|  Etapa {stage['stage_number']}: Emision {wl_str:<18} "
                  f"PL: {pl_str:<4} Contenido: {content:<14} |")
        print("+" + "-" * 78 + "+")

        # --- 3. SISTEMA DE FOTOLUMINISCENCIA ---
        print("\n+" + "-" * 78 + "+")
        print("|  3. SISTEMA DE FOTOLUMINISCENCIA" + " " * 45 + "|")
        print("+" + "-" * 78 + "+")
        excitation_wl = UV_LED_WAVELENGTH_NM if p.excitation_source == ExcitationSource.UV_LED_365 \
            else BLUE_LED_WAVELENGTH_NM
        print(f"|  Fuente de excitacion:    LED {excitation_wl} nm ({p.excitation_source.value})" + " " * (
            78 - 32 - len(f"LED {excitation_wl} nm ({p.excitation_source.value})")) + "|")
        print(f"|  Potencia de excitacion:  {p.excitation_power_mw:.0f} mW" + " " * (
            78 - 28 - len(f"{p.excitation_power_mw:.0f} mW")) + "|")
        print(f"|  Diametro ventana:        {p.window_diameter_mm:.0f} mm" + " " * (
            78 - 28 - len(f"{p.window_diameter_mm:.0f} mm")) + "|")
        print(f"|  Umbral minimo PL:        {p.min_fluorescence_intensity:.1f}" + " " * (
            78 - 28 - len(f"{p.min_fluorescence_intensity:.1f}")) + "|")

        if hasattr(self, '_pl_results'):
            print("|  " + "-" * 74 + "  |")
            print("|  " + f"{'Etapa':<7} {'Tamano':<10} {'Emision':<12} {'SNR':<10} {'Detectable':<12}" + " " * 13 + "  |")
            for r in self._pl_results:
                if r.get('has_pl'):
                    size_str = f"{r['representative_size_nm']:.1f} nm"
                    wl_str = f"{r['emission_wavelength_nm']:.0f} nm"
                    snr_str = f"{r['snr']:.1f}"
                    det_str = "Si" if r['pl_detectable'] else "No"
                    print(f"|  {r['stage_number']:<7} {size_str:<10} {wl_str:<12} "
                          f"{snr_str:<10} {det_str:<12}" + " " * 13 + "  |")
                else:
                    print(f"|  {r['stage_number']:<7} {'---':<10} {'---':<12} "
                          f"{'---':<10} {r.get('reason', 'N/A'):<25}  |")
        print("+" + "-" * 78 + "+")

        # --- 4. EFICIENCIA DE SEPARACION ---
        print("\n+" + "-" * 78 + "+")
        print("|  4. EFICIENCIA DE SEPARACION" + " " * 49 + "|")
        print("+" + "-" * 78 + "+")
        print(f"|  Clasificacion por tamano: {o.size_separation_efficiency * 100:.1f} %" + " " * (
            78 - 29 - len(f"{o.size_separation_efficiency * 100:.1f} %")) + "|")
        print(f"|  Discriminacion PL:        {o.pl_discrimination_efficiency * 100:.1f} %" + " " * (
            78 - 29 - len(f"{o.pl_discrimination_efficiency * 100:.1f} %")) + "|")
        print(f"|  Recuperacion global:      {o.overall_recovery * 100:.1f} %" + " " * (
            78 - 29 - len(f"{o.overall_recovery * 100:.1f} %")) + "|")
        print(f"|  Pureza global:            {o.overall_purity * 100:.1f} %" + " " * (
            78 - 29 - len(f"{o.overall_purity * 100:.1f} %")) + "|")
        print("+" + "-" * 78 + "+")

        # --- 5. CAIDA DE PRESION ---
        print("\n+" + "-" * 78 + "+")
        print("|  5. CAIDA DE PRESION" + " " * 57 + "|")
        print("+" + "-" * 78 + "+")
        if hasattr(self, '_pressure'):
            pr = self._pressure
            print(f"|  Canal principal:          {pr['dP_channel_bar']:.4f} bar" + " " * (
                78 - 29 - len(f"{pr['dP_channel_bar']:.4f} bar")) + "|")
            print(f"|  Membranas (acumulada):    {pr['dP_membranes_bar']:.4f} bar" + " " * (
                78 - 29 - len(f"{pr['dP_membranes_bar']:.4f} bar")) + "|")
            print(f"|  Perdidas menores:         {pr['dP_minor_bar']:.6f} bar" + " " * (
                78 - 29 - len(f"{pr['dP_minor_bar']:.6f} bar")) + "|")
            print(f"|  TOTAL:                    {o.total_pressure_drop_bar:.4f} bar" + " " * (
                78 - 29 - len(f"{o.total_pressure_drop_bar:.4f} bar")) + "|")
            print(f"|  Presion disponible:       {p.operating_pressure_bar:.2f} bar" + " " * (
                78 - 29 - len(f"{p.operating_pressure_bar:.2f} bar")) + "|")
            status = "FACTIBLE" if pr['pressure_feasible'] else "EXCEDE PRESION"
            print(f"|  Estado:                   {status}" + " " * (
                78 - 29 - len(status)) + "|")
            print(f"|  Reynolds:                 {pr['reynolds_number']:.1f} ({pr['flow_regime']})" + " " * (
                78 - 29 - len(f"{pr['reynolds_number']:.1f} ({pr['flow_regime']})")) + "|")
        print("+" + "-" * 78 + "+")

        # --- 6. PUNTUACIONES DE DISENO ---
        print("\n+" + "-" * 78 + "+")
        print("|  6. PUNTUACIONES DE DISENO" + " " * 51 + "|")
        print("+" + "-" * 78 + "+")
        print(f"|  Eficiencia:              {o.efficiency_score:.0f}/100" + " " * (
            78 - 28 - len(f"{o.efficiency_score:.0f}/100")) + "|")
        print(f"|  Costo:                   {o.cost_score:.0f}/100" + " " * (
            78 - 28 - len(f"{o.cost_score:.0f}/100")) + "|")
        print(f"|  Factibilidad:            {o.feasibility_score:.0f}/100" + " " * (
            78 - 28 - len(f"{o.feasibility_score:.0f}/100")) + "|")
        overall = (o.efficiency_score + o.cost_score + o.feasibility_score) / 3
        print(f"|  PUNTUACION GLOBAL:       {overall:.0f}/100" + " " * (
            78 - 28 - len(f"{overall:.0f}/100")) + "|")
        print("+" + "-" * 78 + "+")

    def export_cad_parameters(self) -> Dict:
        """Exporta parametros para software CAD en formato JSON"""
        p = self.params
        o = self.output

        stages_cad = []
        for i, stage in enumerate(o.stages):
            stage_dict = {
                'stage_number': stage['stage_number'],
                'membrane_name': stage['membrane_name'],
                'pore_diameter_nm': stage['pore_diameter_nm'],
                'membrane_type': stage['membrane_type'],
                'has_pl_detection': stage['has_pl_detection'],
                'expected_content': stage['expected_content'],
            }
            stages_cad.append(stage_dict)

        return {
            "units": "mm",
            "classifier_type": "ladder_maze",
            "n_stages": p.n_stages,
            "global_dimensions": {
                "total_length": float(o.total_length_mm),
                "total_width": float(o.total_width_mm),
                "total_height": float(o.total_height_mm),
            },
            "channel": {
                "width": float(p.channel_width_mm),
                "height": float(p.channel_height_mm),
                "wall_thickness": float(p.wall_thickness_mm),
                "stage_pitch": float(p.stage_pitch_mm),
            },
            "membrane_seats": {
                "membrane_diameter": float(p.membrane_diameter_mm),
                "seat_depth": float(p.membrane_seat_depth_mm),
            },
            "collection_chambers": {
                "width": float(p.chamber_width_mm),
                "length": float(p.chamber_length_mm),
                "depth": float(p.chamber_depth_mm),
            },
            "optics": {
                "window_diameter": float(p.window_diameter_mm),
                "window_thickness": float(p.window_thickness_mm),
                "led_diameter": float(p.led_diameter_mm),
                "detector_diameter": float(p.detector_diameter_mm),
                "excitation_source": p.excitation_source.value,
                "excitation_power_mw": float(p.excitation_power_mw),
            },
            "connections": {
                "inlet_diameter": float(p.inlet_diameter_mm),
                "outlet_diameter": float(p.outlet_diameter_mm),
                "waste_diameter": float(p.waste_diameter_mm),
            },
            "stages": stages_cad,
            "performance": {
                "total_pressure_drop_bar": float(o.total_pressure_drop_bar),
                "total_volume_ml": float(o.total_volume_ml),
                "residence_time_s": float(o.residence_time_s),
                "size_separation_efficiency": float(o.size_separation_efficiency),
                "pl_discrimination_efficiency": float(o.pl_discrimination_efficiency),
                "overall_recovery": float(o.overall_recovery),
                "overall_purity": float(o.overall_purity),
            },
            "material": p.body_material,
            "separation_mode": p.separation_mode.value,
        }


def print_membrane_database():
    """Imprime la base de datos de membranas disponibles"""
    print("\n" + "=" * 80)
    print("  BASE DE DATOS DE MEMBRANAS DISPONIBLES")
    print("=" * 80)
    print(f"\n  {'Nombre':<22} {'Poro':<12} {'Espesor':<12} {'Porosidad':<12} "
          f"{'Material':<16} {'P_max':<8}")
    print("  " + "-" * 80)

    for mem in DEFAULT_MEMBRANES:
        pore_str = f"{mem.pore_diameter_nm:.0f} nm"
        thick_str = f"{mem.membrane_thickness_um:.0f} um"
        por_str = f"{mem.porosity:.2f}"
        pmax_str = f"{mem.max_pressure_bar:.1f} bar"
        print(f"  {mem.name:<22} {pore_str:<12} {thick_str:<12} {por_str:<12} "
              f"{mem.material:<16} {pmax_str:<8}")


# ===============================================================================
#  PROGRAMA PRINCIPAL
# ===============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diseno parametrico del clasificador laberinto para CQDs"
    )
    parser.add_argument("--design", action="store_true",
                        help="Ejecutar diseno con parametros por defecto")
    parser.add_argument("--optimize", action="store_true",
                        help="Ejecutar optimizacion por busqueda aleatoria")
    parser.add_argument("--export-cad", type=str,
                        help="Exportar parametros CAD a archivo JSON")
    args = parser.parse_args()

    print("=" * 80)
    print("  DISENADOR PARAMETRICO DE CLASIFICADOR LABERINTO PARA CQDs")
    print("  Separacion por tamano + discriminacion por fotoluminiscencia")
    print("=" * 80)

    if args.optimize:
        print(f"\n-> Optimizando configuracion del clasificador...")
        print("  (Esto puede tardar unos segundos...)\n")

        designer = ClassifierDesigner()
        best_params, best_output = designer.optimize(max_iterations=500)

        # Crear disenador con mejores parametros
        best_membranes = getattr(designer, '_best_membranes', DEFAULT_MEMBRANES)
        designer = ClassifierDesigner(best_params, best_membranes)
        designer.design()
        designer.print_design_report()

        if args.export_cad:
            cad_params = designer.export_cad_parameters()
            with open(args.export_cad, 'w') as f:
                json.dump(cad_params, f, indent=2)
            print(f"\n-> Parametros CAD exportados a: {args.export_cad}")

    elif args.design:
        print("\n-> Ejecutando diseno con parametros por defecto...")

        params = ClassifierParameters()
        designer = ClassifierDesigner(params)
        designer.design()
        designer.print_design_report()

        print_membrane_database()

        if args.export_cad:
            cad_params = designer.export_cad_parameters()
            with open(args.export_cad, 'w') as f:
                json.dump(cad_params, f, indent=2)
            print(f"\n-> Parametros CAD exportados a: {args.export_cad}")

    else:
        # Sin argumentos: mostrar diseno por defecto
        print("\n-> Ejecutando diseno con parametros por defecto...")

        params = ClassifierParameters()
        designer = ClassifierDesigner(params)
        designer.design()
        designer.print_design_report()

        print_membrane_database()

    print("\n" + "=" * 80)
    print("  Uso: python classifier_design.py --design")
    print("       python classifier_design.py --optimize")
    print("       python classifier_design.py --export-cad classifier_cad.json")
    print("=" * 80)
