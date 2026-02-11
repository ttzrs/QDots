# QDots

Sintesis automatizada de **Carbon Quantum Dots (CQDs)** a partir de purines de cerdo mediante un milirreactor de plasma frio (DBD) con barrera catalitica de TiO2, clasificacion optica en paralelo y recirculacion de waste. Guiado por un gemelo digital cuantico (VQE).

## Produccion

| Metrica | Valor |
|---------|-------|
| Produccion bruta (reactor) | **505 mg/h** |
| Produccion neta (purificada) | **495 mg/h** |
| Produccion diaria | **11.9 g/dia** |
| Pureza | **99.7%** |
| Emision | **460 nm** (azul, en spec) |
| Recovery clasificacion | **91%** |
| Payback estimado | **38 dias** (@30 EUR/g) |

## Arquitectura

```
Purin -> [Pre-tratamiento] -> [Milirreactor DBD] -> [Manifold] -> [50x Clasificadores] -> Producto
                                     |                                      |
                               MC 8x300mm                           0.1 mL/min c/u
                               TiO2 anatase                         800 mW laser
                               Plasma frio 100ns                    4S + 3WR + 2R
                               5 mL/min                                    |
                                     |                                     v
                                     +<---- 80% waste recirculado ---  WASTE
                                                                       20% -> fertilizante
```

El milirreactor multi-canal (8 canales x 300 mm) con barrera dielectrica de TiO2 anatase opera con plasma frio (DBD, pulsos de 100 ns). Produce 505 mg/h de CQDs a 5 mL/min. 50 clasificadores opticos en paralelo purifican el producto con 91% de recovery. El 80% del waste se recircula al reactor (boost 1.08x en estado estacionario).

## Modulos

### 1. Milirreactor DBD (principal)

Escala el microreactor validado a dimensiones milimetricas con 4 topologias. La configuracion principal es multi-channel con TiO2 como barrera dielectrica estructural + catalizador.

| Script | Descripcion |
|--------|-------------|
| `reactor_scaleup.py` | **Milirreactor**: 4 topologias (falling film, multi-channel, annular, bubble column), plasma frio, TiO2, enfriamiento activo |
| `reactor_design.py` | Microreactor base: diseno parametrico y optimizacion (~50 mg/h) |
| `reactor_3d_cadquery.py` | Generacion de modelos CAD (CadQuery) |
| `reactor_3d.scad` / `reactor_body.scad` / `reactor_electrodes.scad` | Modelos OpenSCAD |
| `reactor_optimized.json` | Parametros optimizados del microreactor base |

**Especificaciones del milirreactor (multi-channel 8x300mm):**
- Topologia: 8 canales paralelos, 2 mm ancho x 0.5 mm alto x 300 mm largo
- Barrera dielectrica: TiO2 anatase (epsilon_r = 40, sinergia plasma-fotocatalisis)
- Plasma: DBD frio (Te ~1.5 eV, Tgas < 60 C, pulsos 100 ns)
- Flujo: 5 mL/min, energia: 27 W
- Produccion: **505 mg/h** a **460 nm**
- Enfriamiento: activo (serpentin de refrigerante)

### 2. Clasificacion y Produccion

Sistema de clasificacion optica por termoforesis (efecto Soret) con cascada inline y clasificadores en paralelo.

| Script | Descripcion |
|--------|-------------|
| `parallel_classifier.py` | **Sistema completo**: N clasificadores paralelos + waste recirculation + economia |
| `continuous_production.py` | Cascada inline: reactor + clasificador al mismo flujo |
| `classifier_design.py` | Diseno parametrico del clasificador optico (3 modos: LED, laser, opto-termico) |
| `classifier_3d_cadquery.py` | Modelo 3D del clasificador (CadQuery) |
| `classifier_3d.scad` | Modelo 3D OpenSCAD |

**Clasificador optico (modo opto-termico):**
- Principio: laser focalizado sobre sustrato Au (50 nm) genera gradiente termico (~10^6 K/m)
- Efecto Soret: CQDs migran por termoforesis diferencial (Pe >= 1)
- Cascada por clasificador: PF + 4 sep (800 mW) + 3 waste recapture + 2 refinamiento = 10 camaras
- 50 clasificadores en paralelo absorben los 5 mL/min del reactor (0.1 mL/min c/u)
- Recovery: 91%, pureza: 99.7%
- Waste recirculation: 80% del waste vuelve al reactor (boost 1.08x estacionario)

### 3. Simulacion Cuantica (VQE)

Calcula el gap HOMO-LUMO de clusteres de carbono dopados con nitrogeno para predecir la longitud de onda de emision. Define el setpoint del sensor optico.

| Script | Descripcion |
|--------|-------------|
| `qdot_vqe.py` | Implementacion base con Tangelo + PySCF |
| `qdot_vqe_gpu.py` | Version GPU con Qulacs (Docker) |
| `qdot_vqe_24q.py` | Simulacion extendida a 24 qubits |
| `qdot_vqe_pennylane.py` | Implementacion alternativa con PennyLane |
| `qdot_vqe_benchmark.py` | Benchmarks comparativos entre backends |
| `cqd_literature_model.py` | Modelo basado en datos de literatura |
| `chem_backend/tangelo_interface.py` | Interfaz del gemelo digital cuantico |

### 4. Simulacion CFD (OpenFOAM)

Simulacion de flujo laminar y optimizacion bayesiana de la geometria del canal.

```
openfoam_reactor/
├── 0/                  # Condiciones iniciales
├── constant/           # Propiedades de transporte y malla
├── system/             # Configuracion del solver (simpleFoam)
├── stl/                # Geometrias RGB (reactores azul, verde, rojo)
├── scripts/            # Optimizacion bayesiana y PINN
│   ├── bayesian_optimization.py
│   ├── optimize_*.py   # Optimizadores por color/calidad
│   ├── pinn_pytorch.py # Physics-Informed Neural Network
│   └── tangelo_pinn_pipeline.py
└── run_openfoam.sh
```

### 5. Control y Operacion

| Script | Descripcion |
|--------|-------------|
| `reactor_control.py` | Control del reactor + clasificador optico (ReactorController + ClassifierController) |
| `qdot_final.py` | Integracion completa del sistema |

## Uso

```bash
# === SISTEMA COMPLETO (milirreactor + clasificadores paralelos) ===
python parallel_classifier.py                                  # Default: MC 8ch + 50 clasificadores
python parallel_classifier.py --optimize                       # Grid search
python parallel_classifier.py --sweep-recovery                 # Comparar parametros
python parallel_classifier.py --no-recirculation               # Sin waste loop
python parallel_classifier.py --diagram                        # Diagrama de arquitectura

# === MILIRREACTOR (reactor escalado) ===
python reactor_scaleup.py                                      # Comparar 4 topologias
python reactor_scaleup.py --topology multi_channel --flow 5    # MC especifico
python reactor_scaleup.py --optimize --target 500              # Optimizar para 500 mg/h

# === PRODUCCION CONTINUA (reactor + cascada inline) ===
python continuous_production.py                                # Optimizar flujo + volumen
python continuous_production.py --flow 0.1 --stage-vol 5       # Evaluar punto

# === MICROREACTOR BASE ===
python reactor_design.py --optimize --production-target 100    # Optimizar micro

# === CLASIFICADOR OPTICO ===
python classifier_design.py --design --mode optothermal        # Diseno opto-termico
python classifier_design.py --optimize --mode optothermal      # Optimizar

# === SIMULACION VQE ===
python qdot_vqe.py                                             # CPU
./run_gpu.sh                                                   # GPU (Docker)

# === CFD ===
cd openfoam_reactor && ./run_openfoam.sh
```

### Docker (GPU)

```bash
docker compose -f docker-compose.gpu.yml up
```

## Stack

- **Simulacion cuantica**: Tangelo, PySCF, Qulacs, PennyLane
- **CAD**: CadQuery, OpenSCAD
- **CFD**: OpenFOAM 11
- **ML/Optimizacion**: PyTorch (PINN), scikit-optimize (Bayesian)
- **Lenguaje**: Python 3.x
- **Infra**: Docker (GPU + OpenFOAM)

## Estado del Proyecto

| Componente | Estado | Siguiente paso |
|:-----------|:------:|:---------------|
| Milirreactor DBD (MC 8ch, TiO2) | Simulado (505 mg/h) | Fabricar prototipo fisico |
| Plasma frio (non-thermal DBD) | Modelado | Validar Te/Tgas experimentalmente |
| Clasificadores paralelos (50x) | Simulado (91% recovery) | Fabricar primer clasificador chip |
| Waste recirculation loop | Modelado (boost 1.08x) | Implementar manifold + bomba peristaltica |
| Microreactor base | Optimizado (~50 mg/h) | Base para escalar al milirreactor |
| Simulacion VQE | Completada | Validar con clusteres 24+ qubits en GPU |
| Simulacion CFD | Validada | Verificar con fluidos no-newtonianos |
| Clasificador optico (diseno) | 3 modos implementados | Modo opto-termico viable (Pe~1-10) |
| Control y operacion | Implementado | Conectar a actuadores fisicos |
| Hardware | Pendiente | Fabricar milirreactor + clasificadores |

## Licencia

Proyecto de investigacion. Contactar al autor para uso o colaboracion.
