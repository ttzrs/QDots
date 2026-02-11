# QDots: Sintesis de Carbon Quantum Dots mediante Milirreactor de Plasma Frio

Produccion automatizada de CQDs a partir de materia organica (purines de cerdo) utilizando plasma frio DBD con barrera catalitica de TiO2, clasificacion optica en paralelo y recirculacion de waste.

## Estructura del Proyecto

### 1. Milirreactor DBD (Principal)

Escala el microreactor validado a dimensiones milimetricas con 4 topologias. Configuracion principal: multi-channel 8x300mm con TiO2 anatase como barrera dielectrica y fotocatalizador. Plasma frio no-termico (100 ns pulsos).

- `reactor_scaleup.py`: Milirreactor escalado (4 topologias, plasma frio, TiO2, enfriamiento)
- `reactor_design.py`: Microreactor base (diseno parametrico, ~50 mg/h)
- `reactor_3d_cadquery.py` / `reactor_3d.scad`: Modelos CAD
- `reactor_optimized.json`: Parametros optimizados del microreactor

### 2. Clasificacion y Produccion

Sistema de clasificacion optica por termoforesis con cascada inline y clasificadores en paralelo.

- `parallel_classifier.py`: Sistema completo (N clasificadores paralelos + waste recirculation + economia)
- `continuous_production.py`: Cascada inline (reactor + clasificador al mismo flujo)
- `classifier_design.py`: Diseno parametrico del clasificador (3 modos: LED, laser, opto-termico)
- `classifier_3d_cadquery.py` / `classifier_3d.scad`: Modelos CAD del clasificador

### 3. Simulacion Cuantica (VQE)

Algoritmo VQE para predecir propiedades opticas (Gap HOMO-LUMO) de clusteres de carbono dopados con nitrogeno. Define el setpoint del sensor optico.

- `qdot_vqe.py`: Implementacion base con Tangelo y PySCF
- `qdot_vqe_gpu.py`: Version GPU con Qulacs (Docker)
- `qdot_vqe_24q.py`: Simulacion extendida a 24 qubits
- `qdot_vqe_pennylane.py`: Implementacion con PennyLane
- `qdot_vqe_benchmark.py`: Benchmarks entre backends
- `cqd_literature_model.py`: Modelo basado en literatura
- `chem_backend/tangelo_interface.py`: Interfaz del gemelo digital quimico

### 4. Simulacion de Fluidos (OpenFOAM)

Ubicado en `openfoam_reactor/`. Simulaciones de flujo laminar y optimizacion bayesiana de geometria.

- `run_openfoam.sh`: Ejecucion de simulacion
- `scripts/bayesian_optimization.py`: Optimizacion de forma del canal
- `scripts/pinn_pytorch.py`: Physics-Informed Neural Network

### 5. Control y Operacion

- `reactor_control.py`: Control en tiempo real (reactor + clasificador + laser)
- `qdot_final.py`: Integracion completa del sistema

## Tecnologias

- **Simulacion cuantica**: Tangelo, PySCF, Qulacs, PennyLane
- **CAD**: CadQuery, OpenSCAD
- **CFD**: OpenFOAM 11
- **ML/Optimizacion**: PyTorch (PINN), scikit-optimize (Bayesian)
- **Lenguaje**: Python 3.x
- **Infra**: Docker (GPU + OpenFOAM)

## Comandos Clave

```bash
# Sistema completo (milirreactor + 50 clasificadores paralelos)
python parallel_classifier.py

# Milirreactor: comparar topologias
python reactor_scaleup.py

# Produccion continua: cascada inline
python continuous_production.py

# Microreactor base
python reactor_design.py --optimize --production-target 100

# Clasificador optico
python classifier_design.py --design --mode optothermal

# VQE (CPU)
python qdot_vqe.py

# VQE (GPU)
./run_gpu.sh

# CFD
cd openfoam_reactor && ./run_openfoam.sh
```

## Convenciones

- **Gemelo Digital**: Todas las decisiones de diseno fisico deben estar respaldadas por simulaciones previas en Tangelo (quimica) u OpenFOAM (fluidos).
- **Materiales**: TiO2 anatase como barrera dielectrica principal. Resinas de alta temperatura como alternativa.
- **Control**: El setpoint del sensor optico se deriva del gap energetico calculado en los scripts VQE.
- **Economia**: Precio de referencia 30 EUR/g para calculo de payback.
