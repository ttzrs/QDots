#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  OPTIMIZACIÓN BAYESIANA CON PINN SURROGATE
  Encuentra diseño óptimo de reactor DBD para síntesis de CQDs
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path("/var/home/joss/Datos/Proyectos/QDots/openfoam_reactor/pinn_training")
OUTPUT_DIR = Path("/var/home/joss/Datos/Proyectos/QDots/openfoam_reactor/bayesian_opt")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Límites de diseño (basados en datos de entrenamiento)
BOUNDS = {
    'channel_width_mm': (1.0, 6.0),
    'channel_height_mm': (0.5, 5.0),
    'channel_length_mm': (50.0, 400.0),
    'n_turns': (2, 12),
    'liquid_flow_ml_min': (5.0, 100.0),
    'gas_flow_ml_min': (5.0, 100.0),
    'ar_fraction': (0.1, 0.9),
    'n2_fraction': (0.1, 0.8),
    'voltage_kv': (3.0, 20.0),
    'frequency_khz': (1.0, 30.0),
    'duty_cycle': (0.1, 0.9),
    'pulse_width_us': (10.0, 200.0),
    'temperature_C': (20.0, 80.0),
    'pressure_kPa': (90.0, 150.0),
    'precursor_mM': (10.0, 150.0),
    'pH': (3.0, 10.0),
    'electrode_width_mm': (0.5, 3.0),
    'electrode_gap_mm': (0.5, 3.0),
    'electrode_coverage': (0.3, 0.9),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  CARGAR MODELO PINN
# ═══════════════════════════════════════════════════════════════════════════════

class CQDProductionPINN(nn.Module):
    """Modelo PINN (misma arquitectura que en entrenamiento)"""

    def __init__(self, input_dim=27, output_dim=19, hidden_dims=[128, 256, 256, 128]):
        super().__init__()

        self.E_bulk = 1.50
        self.A_conf = 7.26

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        self.head_production = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.GELU(),
            nn.Linear(64, 6)
        )

        self.head_flow = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.GELU(),
            nn.Linear(64, 8)
        )

        self.head_chemistry = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        h = self.encoder(x)
        out_prod = self.head_production(h)
        out_flow = self.head_flow(h)
        out_chem = self.head_chemistry(h)
        return torch.cat([out_prod, out_flow, out_chem], dim=1)


def load_pinn_model():
    """Carga el modelo PINN entrenado"""
    checkpoint = torch.load(DATA_DIR / "cqd_pinn_pytorch.pt", map_location=DEVICE)

    model = CQDProductionPINN().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    norm_params = checkpoint['norm_params']
    X_mean = np.array(norm_params['X_mean'])
    X_std = np.array(norm_params['X_std'])
    Y_mean = np.array(norm_params['Y_mean'])
    Y_std = np.array(norm_params['Y_std'])

    return model, (X_mean, X_std, Y_mean, Y_std)


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE QUÍMICA (Tangelo simplificado)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_tangelo_features(design: Dict) -> List[float]:
    """Calcula features de Tangelo para un diseño dado"""

    T_K = design['temperature_C'] + 273.15
    # Campo eléctrico aproximado
    E_field = design['voltage_kv'] * 1e3 / (design['electrode_gap_mm'] * 1e-3)  # V/m

    # Energías de activación base (kJ/mol) - ajustadas por campo
    alpha = 1e-20
    delta_Ea = -0.5 * alpha * E_field**2 * 2625.5 / 27.2114

    Ea_H2O_dissoc = max(498.0 + delta_Ea, 10.0)
    Ea_precursor_oxid = max(85.0 + delta_Ea * 0.5, 10.0)
    Ea_nucleation = max(150.0 + delta_Ea * 0.3, 10.0)
    Ea_growth = max(50.0 + delta_Ea * 0.2, 10.0)

    # Constantes de velocidad
    R = 8.314
    k_nucleation = 1e12 * np.exp(-Ea_nucleation * 1000 / (R * T_K))
    k_growth = 1e11 * np.exp(-Ea_growth * 1000 / (R * T_K))

    return [Ea_H2O_dissoc, Ea_precursor_oxid, Ea_nucleation, Ea_growth, k_nucleation, k_growth]


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN OBJETIVO
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectiveFunction:
    """Función objetivo para optimización bayesiana"""

    def __init__(self, model, norm_params, target_wavelength: Optional[float] = None):
        self.model = model
        self.X_mean, self.X_std, self.Y_mean, self.Y_std = norm_params
        self.target_wavelength = target_wavelength  # None = maximizar producción
        self.n_calls = 0
        self.history = []

    def design_to_input(self, design: Dict) -> np.ndarray:
        """Convierte diccionario de diseño a vector de input"""

        # O2 fraction
        o2_frac = max(0, 1 - design['ar_fraction'] - design['n2_fraction'])

        # Mixer type (0=none)
        mixer = 0

        # Features base (21)
        x_base = [
            design['channel_width_mm'],
            design['channel_height_mm'],
            design['channel_length_mm'],
            design['n_turns'],
            design['liquid_flow_ml_min'],
            design['gas_flow_ml_min'],
            design['ar_fraction'],
            design['n2_fraction'],
            o2_frac,
            design['voltage_kv'],
            design['frequency_khz'],
            design['duty_cycle'],
            design['pulse_width_us'],
            design['temperature_C'],
            design['pressure_kPa'],
            design['precursor_mM'],
            design['pH'],
            design['electrode_width_mm'],
            design['electrode_gap_mm'],
            design['electrode_coverage'],
            mixer,
        ]

        # Features Tangelo (6)
        x_tangelo = calculate_tangelo_features(design)

        return np.array(x_base + x_tangelo, dtype=np.float32)

    def predict(self, design: Dict) -> Dict:
        """Predice outputs para un diseño"""
        x = self.design_to_input(design)
        x_norm = (x - self.X_mean) / self.X_std

        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)
            y_pred_norm = self.model(x_tensor).cpu().numpy()[0]

        y_pred = y_pred_norm * self.Y_std + self.Y_mean

        return {
            'production_mg_h': y_pred[0],
            'size_nm': y_pred[1],
            'size_std_nm': y_pred[2],
            'wavelength_nm': y_pred[3],
            'quantum_yield': y_pred[4],
            'quality_score': y_pred[5],
            'gas_holdup': y_pred[6],
            'reynolds': y_pred[10],
            'pressure_drop_Pa': y_pred[13],
        }

    def __call__(self, x_vector: np.ndarray) -> float:
        """Evalúa el objetivo (a minimizar)"""
        self.n_calls += 1

        # Convertir vector a diseño
        keys = list(BOUNDS.keys())
        design = {keys[i]: x_vector[i] for i in range(len(keys))}

        # Predecir
        pred = self.predict(design)

        # Calcular score (negativo porque minimizamos)
        prod = max(pred['production_mg_h'], 0)
        qy = np.clip(pred['quantum_yield'], 0, 1)
        quality = np.clip(pred['quality_score'], 0, 1)
        wl = pred['wavelength_nm']

        if self.target_wavelength:
            # Penalizar desviación del target
            wl_penalty = np.exp(-0.001 * (wl - self.target_wavelength)**2)
            score = prod * qy * quality * wl_penalty
        else:
            # Solo maximizar producción y calidad
            score = prod * qy * quality

        # Penalizar presiones extremas
        dp = abs(pred['pressure_drop_Pa'])
        if dp > 1e6:
            score *= 0.1

        # Guardar histórico
        self.history.append({
            'design': design,
            'prediction': pred,
            'score': score,
        })

        return -score  # Negativo para minimizar


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZACIÓN BAYESIANA
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianOptimizer:
    """Optimizador Bayesiano con GP surrogate"""

    def __init__(self, objective: ObjectiveFunction, bounds: Dict):
        self.objective = objective
        self.bounds = bounds
        self.bounds_array = np.array([bounds[k] for k in bounds.keys()])

        # Gaussian Process
        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=42
        )

        self.X_observed = []
        self.Y_observed = []
        self.best_x = None
        self.best_y = float('inf')

    def acquisition_ei(self, x: np.ndarray, xi: float = 0.01) -> float:
        """Expected Improvement acquisition function"""
        x = x.reshape(1, -1)

        mu, sigma = self.gp.predict(x, return_std=True)

        if sigma == 0:
            return 0.0

        imp = self.best_y - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        return -ei[0]  # Negativo porque minimizamos

    def acquisition_ucb(self, x: np.ndarray, kappa: float = 2.0) -> float:
        """Upper Confidence Bound acquisition function"""
        x = x.reshape(1, -1)

        mu, sigma = self.gp.predict(x, return_std=True)

        return mu[0] - kappa * sigma[0]  # Negativo porque minimizamos objetivo

    def suggest_next(self, n_restarts: int = 10) -> np.ndarray:
        """Sugiere el siguiente punto a evaluar"""

        best_x = None
        best_acq = float('inf')

        for _ in range(n_restarts):
            # Punto inicial aleatorio
            x0 = np.random.uniform(self.bounds_array[:, 0], self.bounds_array[:, 1])

            # Optimizar acquisition function
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

    def optimize(self, n_init: int = 20, n_iter: int = 50, verbose: bool = True) -> Dict:
        """Ejecuta optimización bayesiana"""

        if verbose:
            print("\n" + "="*70)
            print("  OPTIMIZACIÓN BAYESIANA")
            print("="*70)

        # Fase 1: Exploración inicial (Latin Hypercube)
        if verbose:
            print(f"\n→ Fase 1: Exploración inicial ({n_init} puntos)")

        for i in range(n_init):
            # Latin Hypercube Sampling
            x = np.zeros(len(self.bounds))
            for j, (key, (lo, hi)) in enumerate(self.bounds.items()):
                x[j] = lo + (hi - lo) * (i + np.random.random()) / n_init

            y = self.objective(x)
            self.X_observed.append(x)
            self.Y_observed.append(y)

            if y < self.best_y:
                self.best_y = y
                self.best_x = x.copy()

            if verbose and (i + 1) % 5 == 0:
                print(f"  [{i+1}/{n_init}] Score: {-y:.2f}")

        # Ajustar GP inicial
        self.gp.fit(np.array(self.X_observed), np.array(self.Y_observed))

        # Fase 2: Optimización bayesiana
        if verbose:
            print(f"\n→ Fase 2: Optimización bayesiana ({n_iter} iteraciones)")

        for i in range(n_iter):
            # Sugerir siguiente punto
            x_next = self.suggest_next()

            # Evaluar
            y_next = self.objective(x_next)

            # Actualizar observaciones
            self.X_observed.append(x_next)
            self.Y_observed.append(y_next)

            # Actualizar mejor
            if y_next < self.best_y:
                self.best_y = y_next
                self.best_x = x_next.copy()
                if verbose:
                    print(f"  [{i+1}/{n_iter}] ★ Nuevo mejor! Score: {-y_next:.2f}")

            # Re-ajustar GP cada 5 iteraciones
            if (i + 1) % 5 == 0:
                self.gp.fit(np.array(self.X_observed), np.array(self.Y_observed))
                if verbose and (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{n_iter}] Score actual: {-y_next:.2f}, Mejor: {-self.best_y:.2f}")

        # Construir resultado
        keys = list(self.bounds.keys())
        best_design = {keys[i]: self.best_x[i] for i in range(len(keys))}

        return {
            'best_design': best_design,
            'best_score': -self.best_y,
            'n_evaluations': len(self.X_observed),
            'history': self.objective.history,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("  OPTIMIZACIÓN BAYESIANA - REACTOR DBD CQDs")
    print("="*70)

    # Cargar modelo PINN
    print("\n→ Cargando modelo PINN...")
    model, norm_params = load_pinn_model()
    print("  ✓ Modelo cargado")

    # Crear función objetivo
    # Opción 1: Maximizar producción general
    # objective = ObjectiveFunction(model, norm_params)

    # Opción 2: Optimizar para emisión azul (450-480 nm)
    print("\n→ Objetivo: Maximizar producción con λ ≈ 460nm (azul)")
    objective = ObjectiveFunction(model, norm_params, target_wavelength=460)

    # Crear optimizador
    optimizer = BayesianOptimizer(objective, BOUNDS)

    # Ejecutar optimización
    result = optimizer.optimize(n_init=30, n_iter=70, verbose=True)

    # Mostrar resultado
    print("\n" + "="*70)
    print("  RESULTADO ÓPTIMO")
    print("="*70)

    best = result['best_design']
    pred = objective.predict(best)

    print(f"\n  Score: {result['best_score']:.2f}")
    print(f"  Evaluaciones: {result['n_evaluations']}")

    print(f"\n  GEOMETRÍA:")
    print(f"    Canal: {best['channel_width_mm']:.2f} × {best['channel_height_mm']:.2f} mm")
    print(f"    Longitud: {best['channel_length_mm']:.1f} mm, {int(best['n_turns'])} vueltas")

    print(f"\n  ELECTRODOS:")
    print(f"    Ancho: {best['electrode_width_mm']:.2f} mm, Gap: {best['electrode_gap_mm']:.2f} mm")
    print(f"    Cobertura: {best['electrode_coverage']*100:.0f}%")

    print(f"\n  FLUJO:")
    print(f"    Líquido: {best['liquid_flow_ml_min']:.1f} mL/min")
    print(f"    Gas: {best['gas_flow_ml_min']:.1f} mL/min (Ar:{best['ar_fraction']*100:.0f}%, N2:{best['n2_fraction']*100:.0f}%)")

    print(f"\n  PLASMA:")
    print(f"    Voltaje: {best['voltage_kv']:.1f} kV")
    print(f"    Frecuencia: {best['frequency_khz']:.1f} kHz")
    print(f"    Duty cycle: {best['duty_cycle']*100:.0f}%")
    print(f"    Pulso: {best['pulse_width_us']:.0f} μs")

    print(f"\n  OPERACIÓN:")
    print(f"    Temperatura: {best['temperature_C']:.1f}°C")
    print(f"    Presión: {best['pressure_kPa']:.1f} kPa")
    print(f"    Precursor: {best['precursor_mM']:.1f} mM, pH={best['pH']:.1f}")

    print(f"\n  PREDICCIONES PINN:")
    print(f"    Producción: {pred['production_mg_h']:.1f} mg/h")
    print(f"    Tamaño: {pred['size_nm']:.2f} ± {pred['size_std_nm']:.2f} nm")
    print(f"    λ emisión: {pred['wavelength_nm']:.0f} nm")
    print(f"    Quantum Yield: {pred['quantum_yield']*100:.1f}%")
    print(f"    Calidad: {pred['quality_score']*100:.0f}%")

    # Guardar resultados
    output_file = OUTPUT_DIR / f"bayesian_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convertir numpy a float para JSON
    result_json = {
        'best_design': {k: float(v) for k, v in result['best_design'].items()},
        'best_score': float(result['best_score']),
        'predictions': {k: float(v) for k, v in pred.items()},
        'n_evaluations': result['n_evaluations'],
        'target_wavelength': objective.target_wavelength,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_file, 'w') as f:
        json.dump(result_json, f, indent=2)

    print(f"\n  ✓ Resultado guardado: {output_file}")

    # Guardar histórico completo
    history_file = OUTPUT_DIR / "optimization_history.json"
    history_data = []
    for h in result['history']:
        history_data.append({
            'design': {k: float(v) for k, v in h['design'].items()},
            'prediction': {k: float(v) for k, v in h['prediction'].items()},
            'score': float(h['score']),
        })

    with open(history_file, 'w') as f:
        json.dump(history_data, f)

    # Mostrar top 5
    print("\n" + "="*70)
    print("  TOP 5 DISEÑOS")
    print("="*70)

    sorted_history = sorted(result['history'], key=lambda x: -x['score'])[:5]

    print(f"\n{'#':>3} | {'Score':>10} | {'Prod':>10} | {'λ':>6} | {'QY':>6} | {'Q':>5}")
    print("-"*55)

    for i, h in enumerate(sorted_history):
        p = h['prediction']
        print(f"{i+1:3d} | {h['score']:10.2f} | {p['production_mg_h']:10.1f} | {p['wavelength_nm']:6.0f} | {p['quantum_yield']*100:5.1f}% | {p['quality_score']*100:4.0f}%")

    print("\n" + "="*70)
    print("  ✓ OPTIMIZACIÓN COMPLETADA")
    print("="*70)


if __name__ == "__main__":
    main()
