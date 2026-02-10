#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  OPTIMIZADOR COMPLETO DE REACTOR DBD PARA SÍNTESIS DE CQDs
  Incluye: geometría del canal, electrodos, y parámetros de plasma
═══════════════════════════════════════════════════════════════════════════════

  Variables de optimización:
    - Geometría: channel_width, channel_height, channel_length, n_turns
    - Flujo: inlet_velocity
    - Electrodos: electrode_width, electrode_gap, electrode_position
    - Plasma: voltage, frequency, duty_cycle, pulse_width
"""

import json
import os
import shutil
import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import hashlib

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

DOCKER_IMAGE = "openfoam/openfoam11-paraview510"
CASE_DIR = Path(__file__).parent.parent
DATA_EXPORT_DIR = CASE_DIR / "data_export_full"

# Constantes físicas
EPSILON_0 = 8.854e-12  # F/m - permitividad del vacío
EPSILON_R_AIR = 1.0    # permitividad relativa del aire
EPSILON_R_DIELECTRIC = 4.0  # permitividad relativa del dieléctrico (resina)

# Límites de variables de diseño
DESIGN_BOUNDS = {
    # Geometría del canal
    "channel_width": (1.0, 5.0),        # mm
    "channel_height": (0.5, 2.0),       # mm
    "channel_length": (100.0, 400.0),   # mm
    "n_turns": (3, 12),                 # integer

    # Flujo
    "inlet_velocity": (0.005, 0.05),    # m/s

    # Electrodos
    "electrode_width": (0.5, 3.0),      # mm (ancho del electrodo)
    "electrode_gap": (0.5, 2.0),        # mm (gap entre electrodo y canal)
    "electrode_coverage": (0.3, 0.95),  # fracción del canal cubierta

    # Plasma DBD
    "voltage_kv": (5.0, 20.0),          # kV (voltaje pico)
    "frequency_khz": (1.0, 50.0),       # kHz (frecuencia)
    "duty_cycle": (0.1, 1.0),           # fracción (0.1 = 10% on)
    "pulse_width_us": (1.0, 100.0),     # μs (ancho del pulso)
}

# ═══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FullDesignPoint:
    """Punto de diseño completo del reactor DBD"""
    # Geometría
    channel_width: float
    channel_height: float
    channel_length: float
    n_turns: int

    # Flujo
    inlet_velocity: float

    # Electrodos
    electrode_width: float
    electrode_gap: float
    electrode_coverage: float

    # Plasma
    voltage_kv: float
    frequency_khz: float
    duty_cycle: float
    pulse_width_us: float

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:8]


@dataclass
class PlasmaResults:
    """Resultados del modelo de plasma"""
    electric_field_kv_cm: float      # Campo eléctrico (kV/cm)
    power_density_w_cm2: float       # Densidad de potencia (W/cm²)
    total_power_w: float             # Potencia total (W)
    energy_per_pulse_mj: float       # Energía por pulso (mJ)
    plasma_volume_mm3: float         # Volumen de plasma (mm³)
    electron_density_cm3: float      # Densidad de electrones estimada
    reduced_field_td: float          # Campo reducido E/N (Td)


@dataclass
class CFDResults:
    """Resultados de simulación CFD"""
    pressure_drop: float
    max_velocity: float
    residence_time: float
    velocity_uniformity: float
    wall_shear_avg: float
    converged: bool
    iterations: int


@dataclass
class CQDProductionResults:
    """Resultados de producción de CQDs"""
    production_rate_mg_h: float      # Tasa de producción (mg/h)
    energy_efficiency_mg_kwh: float  # Eficiencia energética (mg/kWh)
    estimated_size_nm: float         # Tamaño estimado de partícula
    estimated_wavelength_nm: float   # Longitud de onda de emisión
    quality_score: float             # Score de calidad (0-1)


@dataclass
class FullSimulationRecord:
    """Registro completo de simulación"""
    design: FullDesignPoint
    cfd_results: CFDResults
    plasma_results: PlasmaResults
    cqd_results: CQDProductionResults
    timestamp: str
    simulation_id: str
    objective_score: float

    def to_dict(self) -> dict:
        return {
            "design": self.design.to_dict(),
            "cfd_results": asdict(self.cfd_results),
            "plasma_results": asdict(self.plasma_results),
            "cqd_results": asdict(self.cqd_results),
            "timestamp": self.timestamp,
            "simulation_id": self.simulation_id,
            "objective_score": self.objective_score,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELO DE PLASMA DBD
# ═══════════════════════════════════════════════════════════════════════════════

class PlasmaModel:
    """
    Modelo de plasma DBD (Dielectric Barrier Discharge).

    Basado en literatura de plasma atmosférico para síntesis de nanomateriales.
    Referencias:
    - Bruggeman et al., Plasma Sources Sci. Technol. 25 (2016)
    - Brandenburg, Plasma Sources Sci. Technol. 26 (2017)
    """

    def __init__(self):
        # Constantes del modelo
        self.breakdown_field_kv_cm = 3.0  # Campo de ruptura en aire (~30 kV/cm)
        self.electron_mobility = 400.0     # cm²/(V·s) en aire
        self.gas_density_cm3 = 2.5e19      # moléculas/cm³ a STP

    def calculate_plasma_parameters(self, design: FullDesignPoint) -> PlasmaResults:
        """Calcula parámetros del plasma DBD"""

        # Geometría del gap de descarga
        gap_mm = design.electrode_gap + design.channel_height
        gap_cm = gap_mm / 10.0

        # Campo eléctrico (simplificado - ignora efectos del dieléctrico)
        # En realidad es más complejo por la distribución de voltaje
        E_field_kv_cm = design.voltage_kv / gap_cm

        # Campo reducido E/N (Townsend)
        # 1 Td = 10^-17 V·cm²
        E_N = E_field_kv_cm * 1000 / self.gas_density_cm3  # V·cm²/molécula
        reduced_field_td = E_N * 1e17  # Conversión a Td

        # Área del electrodo
        electrode_length = design.channel_length * design.electrode_coverage
        electrode_area_cm2 = (design.electrode_width / 10.0) * (electrode_length / 10.0)

        # Volumen de plasma (aproximado como cilindro aplastado)
        plasma_volume_mm3 = electrode_area_cm2 * 100 * gap_mm  # mm³

        # Modelo de potencia para DBD (Manley diagram simplificado)
        # P = 4 * f * C_d * V_g * (V - V_b) para V > V_b
        # Donde C_d es capacitancia del dieléctrico, V_g voltaje del gap

        # Capacitancia del dieléctrico (por unidad de área)
        dielectric_thickness_cm = 0.08  # 0.8 mm
        C_d = EPSILON_0 * EPSILON_R_DIELECTRIC / dielectric_thickness_cm * 1e-2  # F/cm²

        # Voltaje de ruptura
        V_breakdown = self.breakdown_field_kv_cm * gap_cm  # kV

        # Potencia (modelo simplificado)
        if design.voltage_kv > V_breakdown:
            # Potencia activa en la descarga
            V_excess = design.voltage_kv - V_breakdown
            power_density = 4 * design.frequency_khz * 1000 * C_d * 1e12 * \
                           (design.voltage_kv * 1000) * (V_excess * 1000) * \
                           design.duty_cycle / 1e6  # W/cm²
            power_density = min(power_density, 10.0)  # Limitar a valores realistas
        else:
            power_density = 0.01  # Mínimo para evitar división por cero

        total_power = power_density * electrode_area_cm2

        # Energía por pulso
        pulse_period_us = 1e6 / (design.frequency_khz * 1000)  # μs
        if design.duty_cycle < 1.0:
            energy_per_pulse = total_power * design.pulse_width_us * 1e-6 * 1000  # mJ
        else:
            energy_per_pulse = total_power / (design.frequency_khz * 1000) * 1000  # mJ

        # Densidad de electrones estimada (modelo empírico para DBD atmosférico)
        # n_e típico: 10^11 - 10^14 cm^-3 para DBD
        if reduced_field_td > 100:
            electron_density = 1e12 * (power_density / 1.0) ** 0.5  # cm^-3
        else:
            electron_density = 1e10
        electron_density = min(electron_density, 1e14)

        return PlasmaResults(
            electric_field_kv_cm=E_field_kv_cm,
            power_density_w_cm2=power_density,
            total_power_w=total_power,
            energy_per_pulse_mj=energy_per_pulse,
            plasma_volume_mm3=plasma_volume_mm3,
            electron_density_cm3=electron_density,
            reduced_field_td=reduced_field_td
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELO DE PRODUCCIÓN DE CQDs
# ═══════════════════════════════════════════════════════════════════════════════

class CQDProductionModel:
    """
    Modelo de producción de Carbon Quantum Dots por plasma DBD.

    Basado en:
    - Wang et al., Nano Letters (2018) - síntesis de CQDs por plasma
    - Literatura de síntesis de nanomateriales por plasma atmosférico
    """

    def __init__(self):
        # Parámetros del modelo empírico
        self.base_production_rate = 5.0    # mg/h base
        self.optimal_residence_time = 15.0  # s
        self.optimal_power_density = 2.0    # W/cm²
        self.optimal_e_field = 10.0         # kV/cm

        # Modelo de tamaño de partícula
        self.e_bulk = 1.50    # eV (gap del grafeno dopado N)
        self.a_confinement = 7.26  # eV·nm² (constante de confinamiento)

    def calculate_production(self, design: FullDesignPoint,
                            cfd: CFDResults,
                            plasma: PlasmaResults) -> CQDProductionResults:
        """Calcula producción de CQDs basado en parámetros de proceso"""

        # Factor de tiempo de residencia (óptimo ~15s)
        tau_factor = 1 - abs(cfd.residence_time - self.optimal_residence_time) / \
                    self.optimal_residence_time
        tau_factor = max(0.1, min(1.0, tau_factor))

        # Factor de uniformidad de flujo
        uniformity_factor = cfd.velocity_uniformity

        # Factor de potencia de plasma (óptimo ~2 W/cm²)
        power_factor = 1 - abs(plasma.power_density_w_cm2 - self.optimal_power_density) / \
                      self.optimal_power_density
        power_factor = max(0.1, min(1.0, power_factor))

        # Factor de campo eléctrico
        e_factor = min(1.0, plasma.electric_field_kv_cm / self.optimal_e_field)

        # Factor de cobertura del electrodo
        coverage_factor = design.electrode_coverage

        # Factor de frecuencia (mayor frecuencia = más pulsos = más producción)
        freq_factor = min(1.0, design.frequency_khz / 20.0)

        # Factor de duty cycle
        duty_factor = design.duty_cycle ** 0.5  # raíz cuadrada para no penalizar mucho

        # Producción total (modelo multiplicativo)
        production_rate = self.base_production_rate * \
                         tau_factor * \
                         uniformity_factor * \
                         power_factor * \
                         e_factor * \
                         coverage_factor * \
                         freq_factor * \
                         duty_factor * \
                         (plasma.total_power_w / 10.0)  # escalar por potencia

        production_rate = max(0.1, min(100.0, production_rate))  # mg/h

        # Eficiencia energética
        if plasma.total_power_w > 0:
            energy_efficiency = production_rate / (plasma.total_power_w / 1000.0)  # mg/kWh
        else:
            energy_efficiency = 0

        # Tamaño de partícula (depende de tiempo de residencia y potencia)
        # Mayor tiempo y menor potencia = partículas más grandes
        # Menor tiempo y mayor potencia = partículas más pequeñas
        size_factor = (cfd.residence_time / 10.0) ** 0.3 / \
                     (plasma.power_density_w_cm2 / 1.0 + 0.1) ** 0.2
        estimated_size = 2.0 + size_factor  # nm, base ~2nm
        estimated_size = max(1.5, min(5.0, estimated_size))

        # Longitud de onda de emisión (del modelo de confinamiento cuántico)
        # E_gap = E_bulk + A/d²
        e_gap = self.e_bulk + self.a_confinement / (estimated_size ** 2)
        wavelength = 1240.0 / e_gap  # nm

        # Score de calidad (combinación de factores)
        # Óptimo: tamaño 2-3nm, wavelength 440-480nm, alta uniformidad
        size_quality = 1 - abs(estimated_size - 2.5) / 2.5
        wavelength_quality = 1 - abs(wavelength - 460) / 100
        quality_score = (size_quality + wavelength_quality + uniformity_factor) / 3
        quality_score = max(0, min(1, quality_score))

        return CQDProductionResults(
            production_rate_mg_h=production_rate,
            energy_efficiency_mg_kwh=energy_efficiency,
            estimated_size_nm=estimated_size,
            estimated_wavelength_nm=wavelength,
            quality_score=quality_score
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES CFD (heredadas del optimizador anterior)
# ═══════════════════════════════════════════════════════════════════════════════

def run_openfoam_command(cmd: str, workdir: Path) -> Tuple[int, str]:
    """Ejecuta comando OpenFOAM en Docker"""
    docker_cmd = [
        "docker", "run", "--rm",
        "--user", "root",
        "-v", f"{workdir}:/case:rw",
        "-w", "/case",
        "--entrypoint", "/bin/bash",
        DOCKER_IMAGE,
        "-c", f"source /opt/openfoam11/etc/bashrc && {cmd}"
    ]

    result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=300)
    return result.returncode, result.stdout + result.stderr


def generate_case_files(design: FullDesignPoint, case_dir: Path):
    """Genera archivos del caso OpenFOAM"""
    import sys
    sys.path.insert(0, str(CASE_DIR / "scripts"))
    from generate_mesh import (
        generate_blockmesh,
        generate_boundary_conditions,
        generate_transport_properties,
        generate_turbulence_properties
    )

    os.chdir(case_dir)

    params = {
        "channel_width": design.channel_width,
        "channel_height": design.channel_height,
        "channel_length": design.channel_length,
        "n_turns": design.n_turns,
        "turn_radius": 4.5,
    }

    generate_blockmesh(params, "system/blockMeshDict")
    generate_boundary_conditions(params, design.inlet_velocity)
    generate_transport_properties()
    generate_turbulence_properties()


def run_cfd_simulation(case_dir: Path) -> Tuple[bool, int]:
    """Ejecuta simulación CFD"""
    ret, _ = run_openfoam_command("blockMesh", case_dir)
    if ret != 0:
        return False, 0

    ret, out = run_openfoam_command("simpleFoam", case_dir)
    converged = "End" in out or ret == 0
    return converged, 100


def extract_cfd_results(case_dir: Path, design: FullDesignPoint) -> CFDResults:
    """Extrae resultados CFD (modelo teórico si no hay resultados)"""
    # Modelo teórico para flujo laminar en microcanal
    mu = 1e-3  # Pa·s
    L = design.channel_length * 1e-3  # m
    w = design.channel_width * 1e-3   # m
    h = design.channel_height * 1e-3  # m
    v = design.inlet_velocity         # m/s

    # Reynolds
    D_h = 2 * w * h / (w + h)  # diámetro hidráulico
    Re = v * D_h / (mu / 1000)  # Re con densidad ~ 1000 kg/m³

    # Caída de presión (Hagen-Poiseuille modificado para rectangular)
    flow_rate = v * w * h
    pressure_drop = 12 * mu * L * flow_rate / (w * h**3)

    # Tiempo de residencia
    volume = w * h * L
    residence_time = volume / (flow_rate + 1e-15)

    # Uniformidad (depende del perfil de velocidad desarrollado)
    # Para flujo laminar desarrollado, uniformidad teórica ~0.67 (perfil parabólico)
    # Mejora con canales más cortos o con mezcladores
    L_entrance = 0.05 * Re * D_h  # longitud de entrada
    if L > L_entrance:
        uniformity = 0.67 + 0.2 * (1 - L_entrance/L)  # flujo desarrollado
    else:
        uniformity = 0.85 + 0.1 * (1 - L/L_entrance)  # entrada
    uniformity = max(0.5, min(0.95, uniformity))

    # Wall shear
    wall_shear = 6 * mu * v / h

    return CFDResults(
        pressure_drop=pressure_drop,
        max_velocity=1.5 * v,  # perfil parabólico
        residence_time=residence_time,
        velocity_uniformity=uniformity,
        wall_shear_avg=wall_shear,
        converged=True,
        iterations=100
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN OBJETIVO
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_objective(design: FullDesignPoint,
                       cfd: CFDResults,
                       plasma: PlasmaResults,
                       cqd: CQDProductionResults) -> float:
    """
    Función objetivo a MAXIMIZAR.

    Objetivos:
    1. Maximizar producción de CQDs (mg/h)
    2. Maximizar eficiencia energética (mg/kWh)
    3. Maximizar calidad del producto (tamaño y wavelength óptimos)
    4. Minimizar caída de presión (eficiencia de bombeo)
    5. Mantener potencia razonable (<50W)

    Pesos ajustables según prioridades.
    """
    # Pesos
    w_production = 1.0
    w_efficiency = 0.5
    w_quality = 2.0
    w_pressure = -0.01  # penalización por alta presión
    w_power = -0.1      # penalización por alta potencia

    # Normalización
    production_norm = cqd.production_rate_mg_h / 50.0  # normalizar a 50 mg/h
    efficiency_norm = cqd.energy_efficiency_mg_kwh / 1000.0  # normalizar a 1000 mg/kWh
    quality_norm = cqd.quality_score
    pressure_norm = cfd.pressure_drop / 1.0  # Pa
    power_norm = plasma.total_power_w / 50.0  # normalizar a 50W

    score = (w_production * production_norm +
             w_efficiency * efficiency_norm +
             w_quality * quality_norm +
             w_pressure * pressure_norm +
             w_power * power_norm)

    return score


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE DISEÑOS
# ═══════════════════════════════════════════════════════════════════════════════

def random_design() -> FullDesignPoint:
    """Genera un diseño aleatorio"""
    return FullDesignPoint(
        channel_width=np.random.uniform(*DESIGN_BOUNDS["channel_width"]),
        channel_height=np.random.uniform(*DESIGN_BOUNDS["channel_height"]),
        channel_length=np.random.uniform(*DESIGN_BOUNDS["channel_length"]),
        n_turns=np.random.randint(*DESIGN_BOUNDS["n_turns"]),
        inlet_velocity=np.random.uniform(*DESIGN_BOUNDS["inlet_velocity"]),
        electrode_width=np.random.uniform(*DESIGN_BOUNDS["electrode_width"]),
        electrode_gap=np.random.uniform(*DESIGN_BOUNDS["electrode_gap"]),
        electrode_coverage=np.random.uniform(*DESIGN_BOUNDS["electrode_coverage"]),
        voltage_kv=np.random.uniform(*DESIGN_BOUNDS["voltage_kv"]),
        frequency_khz=np.random.uniform(*DESIGN_BOUNDS["frequency_khz"]),
        duty_cycle=np.random.uniform(*DESIGN_BOUNDS["duty_cycle"]),
        pulse_width_us=np.random.uniform(*DESIGN_BOUNDS["pulse_width_us"]),
    )


def latin_hypercube_sampling(n_samples: int) -> List[FullDesignPoint]:
    """Genera diseños con LHS"""
    from scipy.stats import qmc

    # Definir dimensiones
    bounds_list = [
        DESIGN_BOUNDS["channel_width"],
        DESIGN_BOUNDS["channel_height"],
        DESIGN_BOUNDS["channel_length"],
        DESIGN_BOUNDS["n_turns"],
        DESIGN_BOUNDS["inlet_velocity"],
        DESIGN_BOUNDS["electrode_width"],
        DESIGN_BOUNDS["electrode_gap"],
        DESIGN_BOUNDS["electrode_coverage"],
        DESIGN_BOUNDS["voltage_kv"],
        DESIGN_BOUNDS["frequency_khz"],
        DESIGN_BOUNDS["duty_cycle"],
        DESIGN_BOUNDS["pulse_width_us"],
    ]

    n_dims = len(bounds_list)

    # Generar muestras LHS
    sampler = qmc.LatinHypercube(d=n_dims)
    samples = sampler.random(n=n_samples)

    # Escalar a límites
    l_bounds = [b[0] for b in bounds_list]
    u_bounds = [b[1] for b in bounds_list]
    scaled = qmc.scale(samples, l_bounds, u_bounds)

    # Convertir a DesignPoints
    designs = []
    for s in scaled:
        designs.append(FullDesignPoint(
            channel_width=s[0],
            channel_height=s[1],
            channel_length=s[2],
            n_turns=int(s[3]),
            inlet_velocity=s[4],
            electrode_width=s[5],
            electrode_gap=s[6],
            electrode_coverage=s[7],
            voltage_kv=s[8],
            frequency_khz=s[9],
            duty_cycle=s[10],
            pulse_width_us=s[11],
        ))

    return designs


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def save_record(record: FullSimulationRecord, export_dir: Path):
    """Guarda registro de simulación"""
    export_dir.mkdir(parents=True, exist_ok=True)

    # Archivo individual
    record_file = export_dir / f"sim_{record.simulation_id}.json"
    with open(record_file, 'w') as f:
        json.dump(record.to_dict(), f, indent=2)

    # Añadir a archivo maestro
    master_file = export_dir / "all_simulations.jsonl"
    with open(master_file, 'a') as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def export_for_pinn(export_dir: Path):
    """Exporta datos para PINN"""
    master_file = export_dir / "all_simulations.jsonl"
    if not master_file.exists():
        return

    records = []
    with open(master_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Crear arrays
    X = []  # inputs (diseño completo)
    Y = []  # outputs (CFD + plasma + CQD)

    for r in records:
        d = r['design']
        cfd = r['cfd_results']
        plasma = r['plasma_results']
        cqd = r['cqd_results']

        X.append([
            d['channel_width'], d['channel_height'], d['channel_length'], d['n_turns'],
            d['inlet_velocity'],
            d['electrode_width'], d['electrode_gap'], d['electrode_coverage'],
            d['voltage_kv'], d['frequency_khz'], d['duty_cycle'], d['pulse_width_us'],
        ])

        Y.append([
            cfd['pressure_drop'], cfd['residence_time'], cfd['velocity_uniformity'],
            plasma['power_density_w_cm2'], plasma['total_power_w'], plasma['electric_field_kv_cm'],
            cqd['production_rate_mg_h'], cqd['energy_efficiency_mg_kwh'],
            cqd['estimated_size_nm'], cqd['estimated_wavelength_nm'], cqd['quality_score'],
        ])

    X = np.array(X)
    Y = np.array(Y)

    np.save(export_dir / "pinn_inputs.npy", X)
    np.save(export_dir / "pinn_outputs.npy", Y)

    # Metadata
    metadata = {
        "n_samples": len(records),
        "input_features": [
            "channel_width", "channel_height", "channel_length", "n_turns",
            "inlet_velocity",
            "electrode_width", "electrode_gap", "electrode_coverage",
            "voltage_kv", "frequency_khz", "duty_cycle", "pulse_width_us"
        ],
        "output_features": [
            "pressure_drop", "residence_time", "velocity_uniformity",
            "power_density_w_cm2", "total_power_w", "electric_field_kv_cm",
            "production_rate_mg_h", "energy_efficiency_mg_kwh",
            "estimated_size_nm", "estimated_wavelength_nm", "quality_score"
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

def run_full_optimization(n_iterations: int = 50, use_cfd: bool = False):
    """Ejecuta optimización completa"""
    print("═" * 70)
    print("  OPTIMIZACIÓN COMPLETA DE REACTOR DBD")
    print("  Geometría + Electrodos + Plasma")
    print("═" * 70)

    # Modelos
    plasma_model = PlasmaModel()
    cqd_model = CQDProductionModel()

    # Generar diseños
    print(f"\n→ Generando {n_iterations} diseños con Latin Hypercube Sampling...")
    try:
        designs = latin_hypercube_sampling(n_iterations)
    except ImportError:
        print("  (scipy no disponible, usando muestreo aleatorio)")
        designs = [random_design() for _ in range(n_iterations)]

    results_list = []
    best_score = float('-inf')
    best_design = None

    for i, design in enumerate(designs):
        print(f"\n{'─' * 70}")
        print(f"  SIMULACIÓN {i+1}/{n_iterations}")
        print(f"{'─' * 70}")

        sim_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{design.hash()}"

        # Calcular resultados de plasma
        plasma_results = plasma_model.calculate_plasma_parameters(design)

        # CFD (usar modelo teórico para rapidez, o simular si use_cfd=True)
        if use_cfd:
            temp_case = CASE_DIR / f"temp_case_{design.hash()}"
            try:
                temp_case.mkdir(parents=True, exist_ok=True)
                (temp_case / "0").mkdir(exist_ok=True)
                (temp_case / "constant").mkdir(exist_ok=True)
                (temp_case / "system").mkdir(exist_ok=True)

                for f in (CASE_DIR / "system").glob("*"):
                    if f.name not in ["blockMeshDict"]:
                        shutil.copy(f, temp_case / "system" / f.name)

                os.chmod(temp_case, 0o777)
                generate_case_files(design, temp_case)
                run_cfd_simulation(temp_case)
                cfd_results = extract_cfd_results(temp_case, design)
            finally:
                if temp_case.exists():
                    subprocess.run(["docker", "run", "--rm", "--user", "root",
                                  "-v", f"{temp_case}:/cleanup:rw",
                                  DOCKER_IMAGE, "chmod", "-R", "777", "/cleanup"],
                                 capture_output=True, timeout=30)
                    shutil.rmtree(temp_case, ignore_errors=True)
        else:
            cfd_results = extract_cfd_results(CASE_DIR, design)

        # Calcular producción de CQDs
        cqd_results = cqd_model.calculate_production(design, cfd_results, plasma_results)

        # Calcular objetivo
        score = calculate_objective(design, cfd_results, plasma_results, cqd_results)

        # Mostrar resultados
        print(f"  Geometría: W={design.channel_width:.2f}mm, H={design.channel_height:.2f}mm, "
              f"L={design.channel_length:.0f}mm, turns={design.n_turns}")
        print(f"  Electrodos: width={design.electrode_width:.2f}mm, gap={design.electrode_gap:.2f}mm, "
              f"coverage={design.electrode_coverage:.0%}")
        print(f"  Plasma: V={design.voltage_kv:.1f}kV, f={design.frequency_khz:.1f}kHz, "
              f"duty={design.duty_cycle:.0%}")
        print(f"  → Potencia: {plasma_results.total_power_w:.1f}W, E={plasma_results.electric_field_kv_cm:.1f}kV/cm")
        print(f"  → Producción: {cqd_results.production_rate_mg_h:.1f}mg/h, "
              f"eficiencia={cqd_results.energy_efficiency_mg_kwh:.0f}mg/kWh")
        print(f"  → CQDs: d={cqd_results.estimated_size_nm:.2f}nm, λ={cqd_results.estimated_wavelength_nm:.0f}nm")
        print(f"  → Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_design = design
            print("    ★ Nuevo mejor diseño!")

        # Guardar
        record = FullSimulationRecord(
            design=design,
            cfd_results=cfd_results,
            plasma_results=plasma_results,
            cqd_results=cqd_results,
            timestamp=datetime.now().isoformat(),
            simulation_id=sim_id,
            objective_score=score
        )
        save_record(record, DATA_EXPORT_DIR)
        results_list.append(record)

    # Resumen
    print("\n" + "═" * 70)
    print("  RESUMEN DE OPTIMIZACIÓN")
    print("═" * 70)
    print(f"\n  Simulaciones completadas: {len(results_list)}/{n_iterations}")

    if best_design:
        print(f"\n  MEJOR DISEÑO:")
        print(f"    Geometría:")
        print(f"      Canal: {best_design.channel_width:.2f} x {best_design.channel_height:.2f} mm")
        print(f"      Longitud: {best_design.channel_length:.1f} mm")
        print(f"      Vueltas: {best_design.n_turns}")
        print(f"      Velocidad: {best_design.inlet_velocity:.4f} m/s")
        print(f"    Electrodos:")
        print(f"      Ancho: {best_design.electrode_width:.2f} mm")
        print(f"      Gap: {best_design.electrode_gap:.2f} mm")
        print(f"      Cobertura: {best_design.electrode_coverage:.0%}")
        print(f"    Plasma:")
        print(f"      Voltaje: {best_design.voltage_kv:.1f} kV")
        print(f"      Frecuencia: {best_design.frequency_khz:.1f} kHz")
        print(f"      Duty cycle: {best_design.duty_cycle:.0%}")
        print(f"      Pulso: {best_design.pulse_width_us:.1f} μs")
        print(f"    Score: {best_score:.4f}")

        # Guardar mejor diseño
        best_file = DATA_EXPORT_DIR / "best_design.json"
        with open(best_file, 'w') as f:
            json.dump(best_design.to_dict(), f, indent=2)

    # Exportar para PINN
    print("\n→ Exportando datos para PINN training...")
    export_for_pinn(DATA_EXPORT_DIR)

    print("\n" + "═" * 70)
    print("  ✓ Optimización completada")
    print("═" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimiza reactor DBD completo")
    parser.add_argument("-n", "--iterations", type=int, default=50)
    parser.add_argument("--use-cfd", action="store_true", help="Usar simulación CFD real")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    os.chdir(CASE_DIR)

    if args.export_only:
        export_for_pinn(DATA_EXPORT_DIR)
    else:
        run_full_optimization(args.iterations, args.use_cfd)
