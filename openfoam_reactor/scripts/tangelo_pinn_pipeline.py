#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  PIPELINE TANGELO → PINN
  Re-calcula química con condiciones optimizadas y entrena surrogate
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
sys.path.insert(0, '/var/home/joss/Datos/Proyectos/QDots/chem_backend')

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Importar interfaz Tangelo
from tangelo_interface import TangeloInterface, ChemicalState, ChemicalParameters

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path("/var/home/joss/Datos/Proyectos/QDots/openfoam_reactor/data_export_complete")
OUTPUT_DIR = Path("/var/home/joss/Datos/Proyectos/QDots/openfoam_reactor/pinn_training")
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PASO 1: CARGAR DATOS OPTIMIZADOS DE OPENFOAM
# ═══════════════════════════════════════════════════════════════════════════════

def load_openfoam_data() -> List[Dict]:
    """Carga todos los diseños optimizados"""
    data = []
    jsonl_file = DATA_DIR / "all_simulations.jsonl"

    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    print(f"✓ Cargados {len(data)} diseños de OpenFOAM")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  PASO 2: RE-CALCULAR QUÍMICA CON TANGELO
# ═══════════════════════════════════════════════════════════════════════════════

def recalculate_chemistry(data: List[Dict], interface: TangeloInterface) -> List[Dict]:
    """
    Re-calcula parámetros químicos para cada diseño usando sus condiciones específicas.
    """
    enhanced_data = []

    print("\n→ Re-calculando química con Tangelo...")

    for i, sim in enumerate(data):
        design = sim['design']
        plasma = sim['plasma_chem']

        # Crear estado químico desde condiciones del diseño
        T_kelvin = design['temperature_inlet_C'] + 273.15
        E_field = plasma['electric_field_kV_cm'] * 1e5  # kV/cm → V/m

        # Composición basada en gas
        ar_frac = design['ar_fraction']
        n2_frac = design['n2_fraction']
        o2_frac = 1 - ar_frac - n2_frac

        state = ChemicalState(
            temperature=T_kelvin,
            pressure=design['pressure_kPa'] * 1000,
            composition={
                "H2O": 0.90,  # Fase líquida mayoritaria
                "Ar": ar_frac * 0.05,
                "N2": n2_frac * 0.05,
                "O2": o2_frac * 0.05,
            },
            electric_field=E_field
        )

        # Obtener parámetros químicos
        params = interface.get_parameters(state)

        # Calcular constantes de velocidad
        k_values = {}
        for reaction in params.activation_energies.keys():
            k_values[reaction] = interface.get_rate_constant(reaction, state)

        # Combinar datos
        enhanced = {
            **sim,
            "tangelo_chemistry": {
                "state": {
                    "T_K": T_kelvin,
                    "P_Pa": design['pressure_kPa'] * 1000,
                    "E_Vm": E_field,
                },
                "activation_energies_kJ_mol": params.activation_energies,
                "arrhenius_A": params.arrhenius_A,
                "rate_constants": k_values,
                "cqd_gap_ev_tangelo": params.cqd_gap_ev,
                "cqd_size_nm_tangelo": params.cqd_size_nm,
                "emission_nm_tangelo": 1240 / params.cqd_gap_ev,
                "method": params.calculation_method,
                "confidence": params.confidence,
            }
        }

        enhanced_data.append(enhanced)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(data)}] T={T_kelvin:.0f}K, E={E_field/1e5:.1f}kV/cm → λ={1240/params.cqd_gap_ev:.0f}nm")

    print(f"✓ Química re-calculada para {len(enhanced_data)} diseños")
    return enhanced_data


# ═══════════════════════════════════════════════════════════════════════════════
#  PASO 3: PREPARAR DATOS PARA PINN
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_pinn_data(enhanced_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Prepara arrays numpy para entrenamiento PINN con features extendidos.

    Inputs (27 features):
      [0-20] Parámetros de diseño originales
      [21-26] Parámetros químicos de Tangelo

    Outputs (19 features):
      [0-15] Outputs originales
      [16-18] Predicciones Tangelo (gap, size, λ)
    """

    X = []  # Inputs extendidos
    Y = []  # Outputs extendidos

    for sim in enhanced_data:
        d = sim['design']
        fl = sim['flow']
        th = sim['thermal']
        pl = sim['plasma_chem']
        cq = sim['cqd']
        tg = sim['tangelo_chemistry']

        # Inputs originales (21)
        x_base = [
            d['channel_width_mm'],
            d['channel_height_mm'],
            d['channel_length_mm'],
            d['n_turns'],
            d['liquid_flow_rate_ml_min'],
            d['gas_flow_rate_ml_min'],
            d['ar_fraction'],
            d['n2_fraction'],
            1 - d['ar_fraction'] - d['n2_fraction'],
            d['voltage_kv'],
            d['frequency_khz'],
            d['duty_cycle'],
            d['pulse_width_us'],
            d['temperature_inlet_C'],
            d['pressure_kPa'],
            d['precursor_conc_mM'],
            d['pH'],
            d['electrode_width_mm'],
            d['electrode_gap_mm'],
            d['electrode_coverage'],
            {'none': 0, 'baffle': 1, 'herringbone': 2, 'tesla': 3}.get(d.get('mixer_type', 'none'), 0),
        ]

        # Inputs de Tangelo (6)
        ea = tg['activation_energies_kJ_mol']
        x_tangelo = [
            ea.get('H2O_dissoc', 498.0),
            ea.get('precursor_oxid', 85.0),
            ea.get('C_nucleation', 150.0),
            ea.get('particle_growth', 50.0),
            tg['rate_constants'].get('C_nucleation', 1e6),
            tg['rate_constants'].get('particle_growth', 1e8),
        ]

        X.append(x_base + x_tangelo)

        # Outputs originales (16)
        y_base = [
            cq['production_rate_mg_h'],
            cq['mean_size_nm'],
            cq.get('size_std_nm', 0.1),
            cq['emission_wavelength_nm'],
            cq['quantum_yield_percent'] / 100,
            cq['quality_score'],
            fl['gas_holdup'],
            fl['bubble_frequency_hz'],
            fl['interfacial_area_m2_m3'],
            fl['mass_transfer_coeff_m_s'] * fl['interfacial_area_m2_m3'],
            fl['reynolds_liquid'],
            0.05,
            th['outlet_temperature_C'],
            fl['pressure_drop_Pa'],
            pl['oh_concentration_M'],
            pl['h2o2_concentration_M'],
        ]

        # Outputs de Tangelo (3)
        y_tangelo = [
            tg['cqd_gap_ev_tangelo'],
            tg['cqd_size_nm_tangelo'],
            tg['emission_nm_tangelo'],
        ]

        Y.append(y_base + y_tangelo)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # Metadata
    metadata = {
        "n_samples": len(X),
        "input_features": 27,
        "output_features": 19,
        "input_names": [
            "channel_width_mm", "channel_height_mm", "channel_length_mm", "n_turns",
            "liquid_flow_ml_min", "gas_flow_ml_min", "ar_fraction", "n2_fraction", "o2_fraction",
            "voltage_kv", "frequency_khz", "duty_cycle", "pulse_width_us",
            "temperature_C", "pressure_kPa", "precursor_mM", "pH",
            "electrode_width_mm", "electrode_gap_mm", "electrode_coverage", "mixer_type",
            "Ea_H2O_dissoc", "Ea_precursor_oxid", "Ea_nucleation", "Ea_growth",
            "k_nucleation", "k_growth"
        ],
        "output_names": [
            "production_mg_h", "size_nm", "size_std_nm", "wavelength_nm", "quantum_yield",
            "quality_score", "gas_holdup", "bubble_freq_hz", "interfacial_area",
            "kLa", "reynolds", "turbulence_int", "T_outlet_C", "pressure_drop_Pa",
            "OH_conc_M", "H2O2_conc_M",
            "gap_ev_tangelo", "size_nm_tangelo", "wavelength_nm_tangelo"
        ],
        "timestamp": datetime.now().isoformat(),
    }

    return X, Y, metadata


# ═══════════════════════════════════════════════════════════════════════════════
#  PASO 4: MODELO PINN
# ═══════════════════════════════════════════════════════════════════════════════

class CQDProductionPINN:
    """
    Physics-Informed Neural Network para predicción de síntesis de CQDs.

    Incorpora restricciones físicas:
    - Conservación de masa
    - Modelo de confinamiento cuántico: E_gap = E_bulk + A/d²
    - Cinética de Arrhenius
    """

    def __init__(self, input_dim: int = 27, hidden_dims: List[int] = [128, 256, 128, 64]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Parámetros físicos
        self.E_bulk = 1.50   # eV (gap del grafeno dopado N)
        self.A_conf = 7.26   # eV·nm² (constante de confinamiento)

        # Normalización
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None

        # Pesos del modelo (inicialización)
        self.weights = None
        self.biases = None

    def _init_weights(self):
        """Inicializa pesos con Xavier initialization"""
        np.random.seed(42)

        self.weights = []
        self.biases = []

        dims = [self.input_dim] + self.hidden_dims + [19]  # 19 outputs

        for i in range(len(dims) - 1):
            # Xavier init
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / (dims[i] + dims[i+1]))
            b = np.zeros(dims[i+1])
            self.weights.append(w.astype(np.float32))
            self.biases.append(b.astype(np.float32))

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass"""
        h = X
        for i in range(len(self.weights) - 1):
            h = self._relu(h @ self.weights[i] + self.biases[i])
        # Última capa sin activación
        out = h @ self.weights[-1] + self.biases[-1]
        return out

    def _physics_loss(self, X: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Pérdida basada en restricciones físicas.
        """
        loss = 0.0

        # 1. Confinamiento cuántico: E_gap = E_bulk + A/d²
        size_pred = Y_pred[:, 1]  # tamaño predicho
        gap_pred = Y_pred[:, 16]  # gap predicho (de Tangelo)

        gap_theory = self.E_bulk + self.A_conf / (size_pred**2 + 1e-6)
        loss += np.mean((gap_pred - gap_theory)**2)

        # 2. Wavelength = 1240 / gap
        wl_pred = Y_pred[:, 3]
        wl_theory = 1240.0 / (gap_pred + 1e-6)
        loss += 0.1 * np.mean((wl_pred - wl_theory)**2 / (wl_theory + 1e-6)**2)

        # 3. Producción proporcional a k * concentración * tiempo
        T_inlet = X[:, 13] + 273.15  # K
        conc = X[:, 15]  # mM
        k_growth = X[:, 26]  # constante de crecimiento

        prod_pred = Y_pred[:, 0]
        prod_theory = k_growth * conc * 0.001  # escala aproximada
        loss += 0.01 * np.mean((np.log(prod_pred + 1) - np.log(prod_theory + 1))**2)

        return loss

    def fit(self, X: np.ndarray, Y: np.ndarray,
            epochs: int = 1000,
            learning_rate: float = 0.001,
            physics_weight: float = 0.1,
            batch_size: int = 32,
            verbose: bool = True):
        """
        Entrena el PINN con gradiente descendente.
        """
        # Normalizar datos
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-8
        self.Y_mean = Y.mean(axis=0)
        self.Y_std = Y.std(axis=0) + 1e-8

        X_norm = (X - self.X_mean) / self.X_std
        Y_norm = (Y - self.Y_mean) / self.Y_std

        # Inicializar pesos
        self._init_weights()

        n_samples = len(X)
        history = {'loss': [], 'mse': [], 'physics': []}

        if verbose:
            print(f"\n→ Entrenando PINN ({epochs} epochs, batch={batch_size})...")

        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n_samples)
            X_shuffled = X_norm[idx]
            Y_shuffled = Y_norm[idx]

            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_physics = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]

                # Forward
                Y_pred_norm = self._forward(X_batch)

                # MSE Loss
                mse = np.mean((Y_pred_norm - Y_batch)**2)

                # Physics loss (en espacio original)
                Y_pred = Y_pred_norm * self.Y_std + self.Y_mean
                X_orig = X_batch * self.X_std + self.X_mean
                physics = self._physics_loss(X_orig, Y_pred)

                # Total loss
                loss = mse + physics_weight * physics

                # Backprop simplificado (gradient descent numérico)
                eps = 1e-5
                for layer in range(len(self.weights)):
                    # Gradiente numérico para weights
                    grad_w = np.zeros_like(self.weights[layer])
                    for j in range(min(5, self.weights[layer].shape[0])):  # Sample gradients
                        for k in range(min(5, self.weights[layer].shape[1])):
                            self.weights[layer][j, k] += eps
                            loss_plus = np.mean((self._forward(X_batch) - Y_batch)**2)
                            self.weights[layer][j, k] -= 2*eps
                            loss_minus = np.mean((self._forward(X_batch) - Y_batch)**2)
                            self.weights[layer][j, k] += eps
                            grad_w[j, k] = (loss_plus - loss_minus) / (2*eps)

                    # Update con learning rate adaptativo
                    lr = learning_rate / (1 + epoch/200)
                    self.weights[layer] -= lr * grad_w

                epoch_loss += loss
                epoch_mse += mse
                epoch_physics += physics
                n_batches += 1

            history['loss'].append(epoch_loss / n_batches)
            history['mse'].append(epoch_mse / n_batches)
            history['physics'].append(epoch_physics / n_batches)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:4d}: Loss={history['loss'][-1]:.4f}, MSE={history['mse'][-1]:.4f}, Physics={history['physics'][-1]:.4f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice outputs para nuevos inputs"""
        X_norm = (X - self.X_mean) / self.X_std
        Y_pred_norm = self._forward(X_norm)
        Y_pred = Y_pred_norm * self.Y_std + self.Y_mean
        return Y_pred

    def save(self, filepath: Path):
        """Guarda modelo"""
        data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'X_mean': self.X_mean.tolist(),
            'X_std': self.X_std.tolist(),
            'Y_mean': self.Y_mean.tolist(),
            'Y_std': self.Y_std.tolist(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"✓ Modelo guardado: {filepath}")

    def load(self, filepath: Path):
        """Carga modelo"""
        with open(filepath) as f:
            data = json.load(f)

        self.weights = [np.array(w, dtype=np.float32) for w in data['weights']]
        self.biases = [np.array(b, dtype=np.float32) for b in data['biases']]
        self.X_mean = np.array(data['X_mean'], dtype=np.float32)
        self.X_std = np.array(data['X_std'], dtype=np.float32)
        self.Y_mean = np.array(data['Y_mean'], dtype=np.float32)
        self.Y_std = np.array(data['Y_std'], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  PASO 5: EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model: CQDProductionPINN, X_test: np.ndarray, Y_test: np.ndarray):
    """Evalúa modelo en conjunto de test"""
    Y_pred = model.predict(X_test)

    # Métricas por output
    print("\n" + "="*70)
    print("  EVALUACIÓN DEL MODELO PINN")
    print("="*70)

    output_names = [
        "Producción (mg/h)", "Tamaño (nm)", "Std (nm)", "λ (nm)", "QY",
        "Calidad", "Gas holdup", "f_burbuja", "A_inter", "kLa", "Re",
        "Turb", "T_out", "ΔP", "OH", "H2O2",
        "Gap_Tg", "Size_Tg", "λ_Tg"
    ]

    print(f"\n{'Output':<20} {'MAE':>12} {'RMSE':>12} {'R²':>10}")
    print("-"*56)

    for i, name in enumerate(output_names):
        mae = np.mean(np.abs(Y_pred[:, i] - Y_test[:, i]))
        rmse = np.sqrt(np.mean((Y_pred[:, i] - Y_test[:, i])**2))
        ss_res = np.sum((Y_test[:, i] - Y_pred[:, i])**2)
        ss_tot = np.sum((Y_test[:, i] - np.mean(Y_test[:, i]))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        print(f"{name:<20} {mae:>12.4f} {rmse:>12.4f} {r2:>10.4f}")

    return Y_pred


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("  PIPELINE: TANGELO + OPENFOAM → PINN")
    print("="*70)

    # Paso 1: Cargar datos OpenFOAM
    data = load_openfoam_data()

    # Paso 2: Re-calcular química con Tangelo
    interface = TangeloInterface(use_tangelo=False, cache_results=True)
    enhanced_data = recalculate_chemistry(data, interface)

    # Guardar datos aumentados
    enhanced_file = OUTPUT_DIR / "enhanced_data.jsonl"
    with open(enhanced_file, 'w') as f:
        for item in enhanced_data:
            f.write(json.dumps(item) + '\n')
    print(f"✓ Datos aumentados: {enhanced_file}")

    # Paso 3: Preparar arrays para PINN
    X, Y, metadata = prepare_pinn_data(enhanced_data)

    # Guardar arrays
    np.save(OUTPUT_DIR / "X_tangelo_pinn.npy", X)
    np.save(OUTPUT_DIR / "Y_tangelo_pinn.npy", Y)
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Datos preparados:")
    print(f"    X shape: {X.shape} (inputs)")
    print(f"    Y shape: {Y.shape} (outputs)")

    # Paso 4: Split train/test
    n_train = int(0.8 * len(X))
    idx = np.random.permutation(len(X))

    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    Y_train, Y_test = Y[idx[:n_train]], Y[idx[n_train:]]

    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    # Paso 5: Entrenar PINN
    model = CQDProductionPINN(input_dim=27, hidden_dims=[64, 128, 64, 32])
    history = model.fit(
        X_train, Y_train,
        epochs=500,
        learning_rate=0.01,
        physics_weight=0.1,
        batch_size=16,
        verbose=True
    )

    # Guardar modelo
    model.save(OUTPUT_DIR / "cqd_pinn_model.json")

    # Paso 6: Evaluar
    Y_pred = evaluate_model(model, X_test, Y_test)

    # Guardar predicciones
    np.save(OUTPUT_DIR / "Y_test_pred.npy", Y_pred)
    np.save(OUTPUT_DIR / "Y_test_true.npy", Y_test)

    # Mostrar algunas predicciones
    print("\n" + "="*70)
    print("  EJEMPLOS DE PREDICCIÓN")
    print("="*70)
    print(f"\n{'#':>3} | {'Prod_real':>10} | {'Prod_pred':>10} | {'λ_real':>8} | {'λ_pred':>8} | {'QY_real':>8} | {'QY_pred':>8}")
    print("-"*70)

    for i in range(min(10, len(Y_test))):
        print(f"{i+1:3d} | {Y_test[i,0]:10.1f} | {Y_pred[i,0]:10.1f} | {Y_test[i,3]:8.0f} | {Y_pred[i,3]:8.0f} | {Y_test[i,4]*100:8.1f}% | {Y_pred[i,4]*100:8.1f}%")

    print("\n" + "="*70)
    print("  ✓ PIPELINE COMPLETADO")
    print("="*70)
    print(f"\n  Archivos generados en {OUTPUT_DIR}:")
    for f in OUTPUT_DIR.glob("*"):
        print(f"    - {f.name}")


if __name__ == "__main__":
    main()
