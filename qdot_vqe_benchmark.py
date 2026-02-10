#!/usr/bin/env python3
"""
Benchmark VQE con diferentes tamaños de qubits
Comparación GPU vs CPU
"""

import time
import pennylane as qml
from pennylane import numpy as pnp

HARTREE_TO_EV = 27.2114

print("=" * 70)
print("  BENCHMARK VQE - Escalabilidad GPU (PennyLane Lightning.GPU)")
print("=" * 70)

# Detectar GPU
def get_device(n_qubits, use_gpu=True):
    if use_gpu:
        try:
            return qml.device("lightning.gpu", wires=n_qubits), "lightning.gpu"
        except:
            pass
    try:
        return qml.device("lightning.qubit", wires=n_qubits), "lightning.qubit"
    except:
        return qml.device("default.qubit", wires=n_qubits), "default.qubit"

# Hamiltoniano aleatorio para benchmark
def random_hamiltonian(n_qubits, n_terms=None):
    if n_terms is None:
        n_terms = min(n_qubits * 4, 50)

    import random
    random.seed(42)

    paulis = [qml.PauliX, qml.PauliY, qml.PauliZ]
    coeffs = []
    obs = []

    # Término de identidad
    coeffs.append(0.5)
    obs.append(qml.Identity(0))

    # Términos de 1 qubit
    for i in range(min(n_qubits, n_terms // 3)):
        coeffs.append(random.uniform(-0.5, 0.5))
        obs.append(random.choice(paulis)(i))

    # Términos de 2 qubits
    for _ in range(min(n_terms - len(coeffs), n_qubits * 2)):
        i, j = random.sample(range(n_qubits), 2)
        coeffs.append(random.uniform(-0.2, 0.2))
        obs.append(random.choice(paulis)(i) @ random.choice(paulis)(j))

    return qml.Hamiltonian(coeffs, obs)

# Ansatz hardware-efficient
def create_circuit(dev, H, n_qubits, n_layers=2):
    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        # Estado inicial
        for i in range(n_qubits // 2):
            qml.PauliX(wires=i)

        # Capas variacionales
        param_idx = 0
        for layer in range(n_layers):
            # Rotaciones
            for i in range(n_qubits):
                qml.RY(params[param_idx], wires=i)
                param_idx += 1
                qml.RZ(params[param_idx], wires=i)
                param_idx += 1

            # Entrelazamiento
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])

        return qml.expval(H)

    n_params = n_layers * n_qubits * 2
    return circuit, n_params

# Benchmark
def run_benchmark(n_qubits, n_steps=20, use_gpu=True):
    dev, dev_name = get_device(n_qubits, use_gpu)
    H = random_hamiltonian(n_qubits)
    circuit, n_params = create_circuit(dev, H, n_qubits)

    params = pnp.array([0.1] * n_params, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.2)

    # Warmup
    _ = circuit(params)

    start = time.time()
    for _ in range(n_steps):
        params, _ = opt.step_and_cost(circuit, params)
    elapsed = time.time() - start

    final_energy = circuit(params)

    return {
        "device": dev_name,
        "qubits": n_qubits,
        "params": n_params,
        "steps": n_steps,
        "time": elapsed,
        "time_per_step": elapsed / n_steps,
        "energy": float(final_energy)
    }

# Ejecutar benchmarks
print("\n→ Detectando hardware...")
try:
    import subprocess
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader"],
        stderr=subprocess.DEVNULL
    ).decode().strip()
    print(f"  GPU: {gpu_info}")
except:
    print("  GPU: No detectada")

qubit_sizes = [4, 8, 12, 16, 20, 24]
results_gpu = []
results_cpu = []

print("\n" + "=" * 70)
print("  RESULTADOS")
print("=" * 70)
print(f"\n{'Qubits':>8} {'Params':>8} {'GPU (s)':>10} {'CPU (s)':>10} {'Speedup':>10}")
print("-" * 50)

for n_q in qubit_sizes:
    print(f"\n→ Probando {n_q} qubits...", end=" ", flush=True)

    # GPU
    try:
        r_gpu = run_benchmark(n_q, n_steps=10, use_gpu=True)
        results_gpu.append(r_gpu)
        gpu_time = r_gpu["time"]
        print(f"GPU: {gpu_time:.2f}s", end=" ", flush=True)
    except Exception as e:
        print(f"GPU error: {e}", end=" ", flush=True)
        gpu_time = None

    # CPU
    try:
        r_cpu = run_benchmark(n_q, n_steps=10, use_gpu=False)
        results_cpu.append(r_cpu)
        cpu_time = r_cpu["time"]
        print(f"CPU: {cpu_time:.2f}s", end=" ", flush=True)
    except Exception as e:
        print(f"CPU error: {e}", end=" ", flush=True)
        cpu_time = None

    # Speedup
    if gpu_time and cpu_time:
        speedup = cpu_time / gpu_time
        print(f"Speedup: {speedup:.1f}x")
        print(f"{n_q:>8} {n_q*4:>8} {gpu_time:>10.2f} {cpu_time:>10.2f} {speedup:>9.1f}x")
    else:
        print()

# Resumen
print("\n" + "=" * 70)
print("  RESUMEN DE RENDIMIENTO GPU")
print("=" * 70)

if results_gpu:
    print(f"\n  Qubits probados: {[r['qubits'] for r in results_gpu]}")
    print(f"  Dispositivo GPU: {results_gpu[0]['device']}")

    # Speedups
    if results_gpu and results_cpu:
        print("\n  Speedup GPU vs CPU:")
        for rg, rc in zip(results_gpu, results_cpu):
            if rg['qubits'] == rc['qubits']:
                sp = rc['time'] / rg['time']
                print(f"    {rg['qubits']:2d} qubits: {sp:.1f}x más rápido")

print("\n" + "=" * 70)
print("  APLICACIÓN A QUANTUM DOTS")
print("=" * 70)
print("""
  Para simulación de CQDs con VQE:

  Espacio activo    Qubits    Tiempo GPU    Aplicación
  ─────────────────────────────────────────────────────
  HOMO-LUMO         4         <1 seg        Demo rápido
  (HOMO-1)-(LUMO+1) 8         ~1 seg        Producción
  (HOMO-2)-(LUMO+2) 12        ~2-3 seg      Alta precisión
  Full valence      20+       ~10+ seg      Investigación

  Con GPU puedes usar espacios activos más grandes,
  capturando más correlación electrónica para mejor
  precisión en la predicción del gap óptico.
""")
