#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  MÓDULO DE CONTROL DEL REACTOR DBD PARA SÍNTESIS DE CQDs
  Integra resultados de simulación cuántica con control en tiempo real
═══════════════════════════════════════════════════════════════════════════

USO:
  from reactor_control import ReactorController

  controller = ReactorController()
  action = controller.process_sensor_reading(wavelength=455, intensity=0.8)
  print(action)  # {'action': 'COLLECT', 'valve': 'PRODUCT', ...}
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTES DERIVADAS DE SIMULACIONES CUÁNTICAS
# ═══════════════════════════════════════════════════════════════════════════

# Del documento y validación VQE
HARTREE_TO_EV = 27.2114
EV_TO_NM = 1240

# Valores validados por simulación (qdot_final.py, cqd_literature_model.py)
TARGET_GAP_EV = 2.77          # Gap óptico objetivo (del documento)
TARGET_GAP_HARTREE = 0.102    # En Hartree
TARGET_WAVELENGTH_NM = 447.65 # Longitud de onda objetivo (azul)

# Modelo empírico ajustado (de cqd_literature_model.py)
# E_gap = E_bulk + A/d²
E_BULK_EV = 1.50              # Gap del grafeno dopado N
A_CONFINEMENT = 7.26          # Constante de confinamiento (eV·nm²)

# Tolerancias del sensor
WAVELENGTH_TOLERANCE_NM = 20  # ±20 nm
FWHM_MAX_NM = 50              # Ancho máximo aceptable
INTENSITY_THRESHOLD = 0.3     # Umbral mínimo de intensidad


class ValvePosition(Enum):
    """Posiciones de la válvula de tres vías"""
    PRODUCT = "product"       # Colectar producto bueno
    RECYCLE = "recycle"       # Recircular (partículas pequeñas)
    WASTE = "waste"           # Desechar (partículas grandes o contaminadas)


@dataclass
class SensorReading:
    """Lectura del sensor de fluorescencia"""
    wavelength_nm: float      # Longitud de onda del pico
    intensity: float          # Intensidad normalizada (0-1)
    fwhm_nm: float = 40.0     # Ancho a media altura
    timestamp: float = 0.0    # Tiempo en segundos


@dataclass
class ControlAction:
    """Acción de control del reactor"""
    valve: ValvePosition
    flow_adjustment: float    # -1 a +1 (reducir/aumentar flujo)
    voltage_adjustment: float # -1 a +1 (reducir/aumentar voltaje)
    message: str


class ReactorController:
    """
    Controlador del reactor DBD basado en simulaciones cuánticas.

    Usa el modelo empírico validado por VQE para:
    1. Predecir tamaño de partícula desde λ de emisión
    2. Decidir si colectar, recircular o desechar
    3. Ajustar parámetros de proceso
    """

    def __init__(
        self,
        target_wavelength: float = TARGET_WAVELENGTH_NM,
        tolerance: float = WAVELENGTH_TOLERANCE_NM,
        e_bulk: float = E_BULK_EV,
        a_confinement: float = A_CONFINEMENT
    ):
        self.target_wavelength = target_wavelength
        self.tolerance = tolerance
        self.e_bulk = e_bulk
        self.a_confinement = a_confinement

        # Rango óptimo de longitud de onda
        self.lambda_min = target_wavelength - tolerance
        self.lambda_max = target_wavelength + tolerance

        # Historial para control adaptativo
        self.history = []

    def wavelength_to_gap(self, wavelength_nm: float) -> float:
        """Convierte longitud de onda a gap en eV"""
        return EV_TO_NM / wavelength_nm

    def gap_to_wavelength(self, gap_ev: float) -> float:
        """Convierte gap en eV a longitud de onda"""
        return EV_TO_NM / gap_ev

    def gap_to_size(self, gap_ev: float) -> float:
        """
        Calcula tamaño de partícula desde gap óptico.
        Modelo: E_gap = E_bulk + A/d²
        Despejando: d = sqrt(A / (E_gap - E_bulk))
        """
        if gap_ev <= self.e_bulk:
            return float('inf')  # Partícula muy grande (bulk)
        return np.sqrt(self.a_confinement / (gap_ev - self.e_bulk))

    def size_to_gap(self, diameter_nm: float) -> float:
        """Calcula gap óptico desde tamaño de partícula"""
        return self.e_bulk + self.a_confinement / (diameter_nm ** 2)

    def wavelength_to_size(self, wavelength_nm: float) -> float:
        """Convierte longitud de onda directamente a tamaño"""
        gap = self.wavelength_to_gap(wavelength_nm)
        return self.gap_to_size(gap)

    def classify_product(self, wavelength_nm: float, intensity: float, fwhm_nm: float) -> ValvePosition:
        """
        Clasifica el producto basado en propiedades ópticas.

        Returns:
            PRODUCT: Partículas en rango óptimo (λ ≈ 450 nm)
            RECYCLE: Partículas muy pequeñas (λ < 430 nm, UV)
            WASTE: Partículas muy grandes (λ > 470 nm) o baja calidad
        """
        # Verificar intensidad mínima
        if intensity < INTENSITY_THRESHOLD:
            return ValvePosition.WASTE

        # Verificar monodispersidad
        if fwhm_nm > FWHM_MAX_NM:
            return ValvePosition.WASTE

        # Clasificar por longitud de onda
        if wavelength_nm < self.lambda_min:
            # Partículas muy pequeñas → recircular para crecer
            return ValvePosition.RECYCLE
        elif wavelength_nm > self.lambda_max:
            # Partículas muy grandes → descartar
            return ValvePosition.WASTE
        else:
            # En rango óptimo → colectar
            return ValvePosition.PRODUCT

    def calculate_adjustments(self, wavelength_nm: float) -> Tuple[float, float]:
        """
        Calcula ajustes de flujo y voltaje para alcanzar λ objetivo.

        Returns:
            (flow_adj, voltage_adj): Valores entre -1 y +1
        """
        # Error de longitud de onda
        error = wavelength_nm - self.target_wavelength

        # Normalizar error
        normalized_error = error / self.tolerance
        normalized_error = np.clip(normalized_error, -2, 2)

        # Control proporcional
        # λ alta → partículas grandes → aumentar flujo (menos tiempo)
        # λ baja → partículas pequeñas → reducir flujo (más tiempo)
        flow_adj = normalized_error * 0.3

        # Voltaje: ajuste menor, afecta fragmentación
        voltage_adj = -normalized_error * 0.1

        return flow_adj, voltage_adj

    def process_sensor_reading(
        self,
        wavelength: float,
        intensity: float,
        fwhm: float = 40.0
    ) -> Dict:
        """
        Procesa una lectura del sensor y genera acción de control.

        Args:
            wavelength: Longitud de onda del pico de emisión (nm)
            intensity: Intensidad normalizada (0-1)
            fwhm: Ancho a media altura (nm)

        Returns:
            Dict con acción de control completa
        """
        # Calcular propiedades derivadas
        gap_ev = self.wavelength_to_gap(wavelength)
        size_nm = self.wavelength_to_size(wavelength)

        # Clasificar producto
        valve = self.classify_product(wavelength, intensity, fwhm)

        # Calcular ajustes
        flow_adj, voltage_adj = self.calculate_adjustments(wavelength)

        # Generar mensaje
        if valve == ValvePosition.PRODUCT:
            message = f"✓ COLECTAR: λ={wavelength:.1f}nm, d={size_nm:.2f}nm, E={gap_ev:.2f}eV"
        elif valve == ValvePosition.RECYCLE:
            message = f"↻ RECIRCULAR: Partículas pequeñas (λ={wavelength:.1f}nm < {self.lambda_min}nm)"
        else:
            if intensity < INTENSITY_THRESHOLD:
                message = f"✗ DESCARTAR: Intensidad baja ({intensity:.2f} < {INTENSITY_THRESHOLD})"
            elif fwhm > FWHM_MAX_NM:
                message = f"✗ DESCARTAR: FWHM alto ({fwhm:.1f}nm > {FWHM_MAX_NM}nm)"
            else:
                message = f"✗ DESCARTAR: Partículas grandes (λ={wavelength:.1f}nm > {self.lambda_max}nm)"

        # Guardar en historial
        reading = SensorReading(wavelength, intensity, fwhm)
        self.history.append(reading)

        return {
            'action': valve.name,
            'valve': valve.value,
            'wavelength_nm': wavelength,
            'gap_ev': gap_ev,
            'size_nm': size_nm,
            'flow_adjustment': flow_adj,
            'voltage_adjustment': voltage_adj,
            'in_spec': valve == ValvePosition.PRODUCT,
            'message': message
        }

    def get_setpoints(self) -> Dict:
        """Retorna los setpoints de control basados en simulaciones"""
        target_size = self.wavelength_to_size(self.target_wavelength)

        return {
            'target_wavelength_nm': self.target_wavelength,
            'wavelength_range_nm': (self.lambda_min, self.lambda_max),
            'target_gap_ev': TARGET_GAP_EV,
            'target_size_nm': target_size,
            'max_fwhm_nm': FWHM_MAX_NM,
            'min_intensity': INTENSITY_THRESHOLD,
            'source': 'VQE simulation + literature model'
        }

    def print_control_table(self):
        """Imprime tabla de control para referencia"""
        print("\n" + "=" * 70)
        print("  TABLA DE CONTROL - BASADA EN SIMULACIÓN CUÁNTICA")
        print("=" * 70)
        print(f"\n  Setpoint: λ = {self.target_wavelength:.1f} nm ± {self.tolerance} nm")
        print(f"  Gap objetivo: {TARGET_GAP_EV:.2f} eV ({TARGET_GAP_HARTREE:.3f} Hartree)")
        print(f"\n  {'λ (nm)':<10} {'Gap (eV)':<10} {'Tamaño (nm)':<12} {'Color':<12} {'Acción':<15}")
        print("  " + "-" * 60)

        wavelengths = [380, 400, 420, 440, 450, 460, 480, 500, 550, 600]
        colors = {
            380: 'Violeta', 400: 'Violeta', 420: 'Violeta',
            440: 'Azul', 450: 'Azul', 460: 'Azul',
            480: 'Cyan', 500: 'Verde', 550: 'Verde', 600: 'Naranja'
        }

        for wl in wavelengths:
            gap = self.wavelength_to_gap(wl)
            size = self.wavelength_to_size(wl)
            color = colors.get(wl, 'Rojo')

            if wl < self.lambda_min:
                action = "Recircular"
            elif wl > self.lambda_max:
                action = "Descartar"
            else:
                action = "✓ COLECTAR"

            print(f"  {wl:<10} {gap:<10.2f} {size:<12.2f} {color:<12} {action:<15}")


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE UTILIDAD
# ═══════════════════════════════════════════════════════════════════════════

def simulate_sensor_calibration():
    """Simula calibración del sensor con datos de referencia"""
    print("\n" + "=" * 70)
    print("  CALIBRACIÓN DEL SENSOR DE FLUORESCENCIA")
    print("=" * 70)

    # Datos de calibración (estándar de fluoresceína u otro)
    calibration_points = [
        (420, 2.95),  # λ, Gap esperado
        (450, 2.76),
        (480, 2.58),
        (520, 2.38),
    ]

    print("\n  Puntos de calibración:")
    print(f"  {'λ medido (nm)':<15} {'Gap calc (eV)':<15} {'Gap esperado (eV)':<15} {'Error %':<10}")
    print("  " + "-" * 55)

    for wl, expected_gap in calibration_points:
        calc_gap = EV_TO_NM / wl
        error = abs(calc_gap - expected_gap) / expected_gap * 100
        print(f"  {wl:<15} {calc_gap:<15.3f} {expected_gap:<15.3f} {error:<10.2f}")


def vqe_to_control_parameters():
    """Muestra cómo los resultados VQE informan el control"""
    print("\n" + "=" * 70)
    print("  CONEXIÓN VQE → CONTROL DEL REACTOR")
    print("=" * 70)

    print("""
  Los resultados de simulación VQE se usan así:

  1. VALIDACIÓN TEÓRICA (qdot_final.py):
     • Confirma que 0.102 Ha = 2.77 eV = 447.65 nm es correcto
     • Base matemática para el setpoint del sensor

  2. MODELO EMPÍRICO (cqd_literature_model.py):
     • E_gap = 1.50 + 7.26/d² [eV, nm]
     • Permite calcular tamaño desde λ de emisión
     • Usado en tiempo real para control de flujo

  3. CORRELACIÓN ELECTRÓNICA (VQE 12-24 qubits):
     • Captura efectos cuánticos no incluidos en HF
     • Mejora predicción del gap óptico real
     • Útil para diseño de nuevos dopantes (P, S, B)

  4. PARÁMETROS DE CONTROL DERIVADOS:
     ┌─────────────────────────────────────────────────────────┐
     │  Setpoint λ:     450 nm ± 20 nm                        │
     │  Tamaño d:       2.3 - 2.7 nm                          │
     │  Gap E:          2.6 - 2.9 eV                          │
     │  FWHM máx:       50 nm (monodispersidad)               │
     │  Intensidad mín: 0.3 (señal válida)                    │
     └─────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════
#  EJEMPLO DE USO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  SISTEMA DE CONTROL DEL REACTOR DBD - SÍNTESIS DE CQDs")
    print("  Basado en simulación cuántica VQE")
    print("=" * 70)

    # Crear controlador
    controller = ReactorController()

    # Mostrar setpoints
    setpoints = controller.get_setpoints()
    print(f"\n  Configuración del controlador:")
    print(f"    λ objetivo: {setpoints['target_wavelength_nm']:.1f} nm")
    print(f"    Rango λ: {setpoints['wavelength_range_nm'][0]:.0f} - {setpoints['wavelength_range_nm'][1]:.0f} nm")
    print(f"    Tamaño objetivo: {setpoints['target_size_nm']:.2f} nm")
    print(f"    Gap objetivo: {setpoints['target_gap_ev']:.2f} eV")

    # Mostrar tabla de control
    controller.print_control_table()

    # Simular lecturas del sensor
    print("\n" + "=" * 70)
    print("  SIMULACIÓN DE LECTURAS DEL SENSOR")
    print("=" * 70)

    test_readings = [
        (410, 0.7, 35),   # Partículas pequeñas
        (448, 0.85, 38),  # Óptimo
        (455, 0.9, 42),   # Óptimo
        (490, 0.6, 55),   # Partículas grandes, FWHM alto
        (520, 0.4, 45),   # Muy grandes
    ]

    print("\n  Lecturas simuladas:")
    for wl, intensity, fwhm in test_readings:
        result = controller.process_sensor_reading(wl, intensity, fwhm)
        print(f"\n  {result['message']}")
        if result['in_spec']:
            print(f"    → Tamaño estimado: {result['size_nm']:.2f} nm")
        print(f"    → Ajuste flujo: {result['flow_adjustment']:+.2f}, Ajuste voltaje: {result['voltage_adjustment']:+.2f}")

    # Mostrar conexión VQE → Control
    vqe_to_control_parameters()

    # Calibración
    simulate_sensor_calibration()
