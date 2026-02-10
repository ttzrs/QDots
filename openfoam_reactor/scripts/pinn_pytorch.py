#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  PINN CON PYTORCH - SÍNTESIS DE CQDs
  Physics-Informed Neural Network optimizado
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("/var/home/joss/Datos/Proyectos/QDots/openfoam_reactor/pinn_training")
print(f"Using device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════════
#  MODELO PINN
# ═══════════════════════════════════════════════════════════════════════════════

class CQDProductionPINN(nn.Module):
    """
    Physics-Informed Neural Network para síntesis de CQDs.

    Arquitectura:
    - Input: 27 features (diseño + química)
    - Hidden: 3 capas residuales
    - Output: 19 targets (propiedades de producción)

    Restricciones físicas:
    - Confinamiento cuántico: E_gap = E_bulk + A/d²
    - λ = 1240/E_gap
    - Arrhenius: k = A·exp(-Ea/RT)
    """

    def __init__(self, input_dim=27, output_dim=19, hidden_dims=[256, 512, 256, 128]):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Constantes físicas
        self.E_bulk = 1.50   # eV (gap del grafeno dopado N)
        self.A_conf = 7.26   # eV·nm² (constante de confinamiento)

        # Encoder
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

        # Heads para diferentes outputs
        # Head 1: Producción y tamaño (0-5)
        self.head_production = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.GELU(),
            nn.Linear(64, 6)
        )

        # Head 2: Propiedades de flujo (6-13)
        self.head_flow = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.GELU(),
            nn.Linear(64, 8)
        )

        # Head 3: Química (14-18)
        self.head_chemistry = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 5)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.encoder(x)

        out_prod = self.head_production(h)
        out_flow = self.head_flow(h)
        out_chem = self.head_chemistry(h)

        return torch.cat([out_prod, out_flow, out_chem], dim=1)

    def physics_loss(self, x, y_pred, y_true):
        """
        Calcula pérdida de física.
        """
        loss = torch.tensor(0.0, device=x.device)

        # Extraer predicciones
        size_pred = y_pred[:, 1]       # tamaño CQD (nm)
        wl_pred = y_pred[:, 3]         # λ emisión (nm)
        gap_tangelo = y_pred[:, 16]    # gap Tangelo (eV)
        size_tangelo = y_pred[:, 17]   # size Tangelo (nm)
        wl_tangelo = y_pred[:, 18]     # λ Tangelo (nm)

        # 1. Confinamiento cuántico: coherencia entre size y gap
        gap_from_size = self.E_bulk + self.A_conf / (size_pred**2 + 0.1)
        wl_from_gap = 1240.0 / (gap_from_size + 0.1)

        # Pérdida: λ predicho vs λ teórico de size
        loss += 0.01 * torch.mean((wl_pred - wl_from_gap)**2 / (wl_from_gap + 1e-6)**2)

        # 2. Consistencia Tangelo: size_tangelo debería correlacionar con size
        loss += 0.1 * torch.mean((size_pred - size_tangelo)**2)

        # 3. Producción positiva
        prod_pred = y_pred[:, 0]
        loss += 0.1 * torch.mean(torch.relu(-prod_pred))  # Penalizar negativos

        # 4. QY entre 0 y 1
        qy_pred = y_pred[:, 4]
        loss += 0.1 * torch.mean(torch.relu(qy_pred - 1.0) + torch.relu(-qy_pred))

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Carga datos preparados"""
    X = np.load(DATA_DIR / "X_tangelo_pinn.npy")
    Y = np.load(DATA_DIR / "Y_tangelo_pinn.npy")

    # Normalizar
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-8

    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std

    # Convertir a tensores
    X_tensor = torch.FloatTensor(X_norm)
    Y_tensor = torch.FloatTensor(Y_norm)

    return X_tensor, Y_tensor, (X_mean, X_std, Y_mean, Y_std)


def train_model(model, train_loader, val_loader, epochs=500, lr=1e-3, physics_weight=0.1):
    """Entrena el modelo PINN"""

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    mse_loss = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': [], 'physics_loss': []}

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50

    print(f"\n→ Entrenando PINN ({epochs} epochs)")
    print("-" * 60)

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        phys_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

            optimizer.zero_grad()

            Y_pred = model(X_batch)
            loss_mse = mse_loss(Y_pred, Y_batch)
            loss_physics = model.physics_loss(X_batch, Y_pred, Y_batch)

            loss = loss_mse + physics_weight * loss_physics
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss_mse.item()
            phys_loss += loss_physics.item()

        scheduler.step()
        train_loss /= len(train_loader)
        phys_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                Y_pred = model(X_batch)
                val_loss += mse_loss(Y_pred, Y_batch).item()
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['physics_loss'].append(phys_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), DATA_DIR / "best_model.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:4d}: Train={train_loss:.4f}, Val={val_loss:.4f}, Physics={phys_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    # Cargar mejor modelo
    model.load_state_dict(torch.load(DATA_DIR / "best_model.pt"))
    return history


def evaluate_model(model, X_test, Y_test, norm_params):
    """Evalúa el modelo"""
    X_mean, X_std, Y_mean, Y_std = norm_params

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test.to(DEVICE)
        Y_pred_norm = model(X_test_dev).cpu().numpy()

    # Desnormalizar
    Y_pred = Y_pred_norm * Y_std + Y_mean
    Y_true = Y_test.numpy() * Y_std + Y_mean

    output_names = [
        "Producción (mg/h)", "Tamaño (nm)", "Std (nm)", "λ (nm)", "QY",
        "Calidad", "Gas holdup", "f_burbuja", "A_inter", "kLa", "Re",
        "Turb", "T_out", "ΔP", "OH", "H2O2",
        "Gap_Tg", "Size_Tg", "λ_Tg"
    ]

    print("\n" + "="*70)
    print("  EVALUACIÓN DEL MODELO PINN (PyTorch)")
    print("="*70)
    print(f"\n{'Output':<20} {'MAE':>12} {'RMSE':>12} {'R²':>10}")
    print("-"*56)

    r2_scores = []
    for i, name in enumerate(output_names):
        mae = np.mean(np.abs(Y_pred[:, i] - Y_true[:, i]))
        rmse = np.sqrt(np.mean((Y_pred[:, i] - Y_true[:, i])**2))
        ss_res = np.sum((Y_true[:, i] - Y_pred[:, i])**2)
        ss_tot = np.sum((Y_true[:, i] - np.mean(Y_true[:, i]))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        r2_scores.append(r2)

        print(f"{name:<20} {mae:>12.4f} {rmse:>12.4f} {r2:>10.4f}")

    print("-"*56)
    print(f"{'Promedio R²':<20} {'':<12} {'':<12} {np.mean(r2_scores):>10.4f}")

    return Y_pred, Y_true


def predict_optimal_design(model, norm_params):
    """Usa el modelo para predecir diseño óptimo"""
    X_mean, X_std, Y_mean, Y_std = norm_params

    # Cargar todos los datos originales
    X = np.load(DATA_DIR / "X_tangelo_pinn.npy")

    # Predecir para todos
    model.eval()
    with torch.no_grad():
        X_norm = (X - X_mean) / X_std
        X_tensor = torch.FloatTensor(X_norm).to(DEVICE)
        Y_pred_norm = model(X_tensor).cpu().numpy()

    Y_pred = Y_pred_norm * Y_std + Y_mean

    # Encontrar diseño con mejor score
    # Score = Producción * QY * Calidad / (1 + log(1 + ΔP))
    prod = Y_pred[:, 0]
    qy = Y_pred[:, 4]
    quality = Y_pred[:, 5]
    dp = Y_pred[:, 13]

    scores = prod * np.clip(qy, 0, 1) * np.clip(quality, 0, 1) / (1 + np.log(1 + np.abs(dp)/1000))

    idx_best = np.argmax(scores)

    print("\n" + "="*70)
    print("  DISEÑO ÓPTIMO (según PINN)")
    print("="*70)
    print(f"\n  Índice: {idx_best+1}")
    print(f"  Score: {scores[idx_best]:.2f}")
    print(f"\n  Predicciones:")
    print(f"    Producción: {Y_pred[idx_best, 0]:.1f} mg/h")
    print(f"    Tamaño: {Y_pred[idx_best, 1]:.2f} nm")
    print(f"    λ emisión: {Y_pred[idx_best, 3]:.0f} nm")
    print(f"    QY: {Y_pred[idx_best, 4]*100:.1f}%")
    print(f"    Calidad: {Y_pred[idx_best, 5]*100:.0f}%")
    print(f"    ΔP: {Y_pred[idx_best, 13]:.0f} Pa")

    return idx_best, Y_pred[idx_best]


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("  PINN PYTORCH - SÍNTESIS DE CQDs")
    print("="*70)

    # Cargar datos
    X, Y, norm_params = load_data()
    print(f"\nDatos cargados: X={X.shape}, Y={Y.shape}")

    # Split
    n = len(X)
    idx = torch.randperm(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # DataLoaders
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Modelo
    model = CQDProductionPINN(
        input_dim=27,
        output_dim=19,
        hidden_dims=[128, 256, 256, 128]
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {total_params:,} parámetros")

    # Entrenar
    history = train_model(
        model, train_loader, val_loader,
        epochs=1000,
        lr=5e-4,
        physics_weight=0.05
    )

    # Evaluar
    Y_pred, Y_true = evaluate_model(model, X_test, Y_test, norm_params)

    # Ejemplos
    print("\n" + "="*70)
    print("  EJEMPLOS DE PREDICCIÓN")
    print("="*70)
    print(f"\n{'#':>3} | {'Prod_real':>10} | {'Prod_pred':>10} | {'λ_real':>8} | {'λ_pred':>8} | {'QY_real':>8} | {'QY_pred':>8}")
    print("-"*75)

    for i in range(min(10, len(Y_true))):
        print(f"{i+1:3d} | {Y_true[i,0]:10.1f} | {Y_pred[i,0]:10.1f} | {Y_true[i,3]:8.0f} | {Y_pred[i,3]:8.0f} | {Y_true[i,4]*100:8.1f}% | {Y_pred[i,4]*100:8.1f}%")

    # Encontrar óptimo
    predict_optimal_design(model, norm_params)

    # Guardar modelo final
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params': {
            'X_mean': norm_params[0].tolist(),
            'X_std': norm_params[1].tolist(),
            'Y_mean': norm_params[2].tolist(),
            'Y_std': norm_params[3].tolist(),
        },
        'history': history,
    }, DATA_DIR / "cqd_pinn_pytorch.pt")

    print("\n" + "="*70)
    print("  ✓ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\n  Modelo guardado: {DATA_DIR / 'cqd_pinn_pytorch.pt'}")


if __name__ == "__main__":
    main()
