# QDots

Síntesis automatizada de **Carbon Quantum Dots (CQDs)** a partir de purines de cerdo mediante un milirreactor de plasma frío (DBD), guiado por un gemelo digital cuántico.

## Concepto

El sistema convierte materia orgánica residual (purines) en nanomateriales fluorescentes de alto valor. Un plasma de descarga de barrera dieléctrica (DBD) fragmenta la materia orgánica, genera radicales de carbono y los recombina en puntos cuánticos con emisión óptica controlada (~450 nm, azul).

La clave es que **ningún parámetro se elige por ensayo y error**: cada decisión de diseño (geometría del canal, voltaje, tiempo de residencia, setpoint del sensor) se deriva de simulaciones cuánticas (VQE) y fluidodinámicas (CFD) previas.

```
Purín → [Pre-tratamiento] → [Reactor DBD] → [Sensor óptico] → [Válvula selectiva]
                                    ↑                              ↓
                              Gemelo Digital              Producto / Descarte
                           (Tangelo + OpenFOAM)
```

## Arquitectura

### 1. Simulación Cuántica (VQE)

Calcula el gap HOMO-LUMO de clústeres de carbono dopados con nitrógeno para predecir la longitud de onda de emisión. Ese valor define el setpoint del sensor óptico del reactor.

| Script | Descripción |
|--------|-------------|
| `qdot_vqe.py` | Implementación base con Tangelo + PySCF |
| `qdot_vqe_gpu.py` | Versión GPU con Qulacs (Docker) |
| `qdot_vqe_24q.py` | Simulación extendida a 24 qubits |
| `qdot_vqe_pennylane.py` | Implementación alternativa con PennyLane |
| `qdot_vqe_benchmark.py` | Benchmarks comparativos entre backends |
| `cqd_literature_model.py` | Modelo basado en datos de literatura |
| `chem_backend/tangelo_interface.py` | Interfaz del gemelo digital químico |

### 2. Diseño del Reactor

Diseño paramétrico del milirreactor DBD para fabricación en impresión 3D (DLP/cerámica).

| Script | Descripción |
|--------|-------------|
| `reactor_design.py` | Motor de diseño paramétrico y optimización |
| `reactor_3d_cadquery.py` | Generación de modelos CAD (CadQuery) |
| `reactor_3d.scad` / `reactor_body.scad` / `reactor_electrodes.scad` | Modelos OpenSCAD |
| `reactor_optimized.json` | Parámetros optimizados exportados |

**Parámetros de diseño actuales** (de `reactor_optimized.json`):
- Canal serpentina: 3.0 mm ancho, 1.0 mm alto, 250 mm longitud, 6 vueltas
- Electrodos de cobre embebidos, gap de 1.0 mm
- Dieléctrico: resina de alta temperatura, 0.8 mm
- Producción estimada: **~30 mg/h** a **~463 nm**

### 3. Simulación CFD (OpenFOAM)

Simulación de flujo laminar y optimización bayesiana de la geometría del canal.

```
openfoam_reactor/
├── 0/                  # Condiciones iniciales
├── constant/           # Propiedades de transporte y malla
├── system/             # Configuración del solver (simpleFoam)
├── stl/                # Geometrías RGB (reactores azul, verde, rojo)
├── scripts/            # Optimización bayesiana y PINN
│   ├── bayesian_optimization.py
│   ├── optimize_*.py   # Optimizadores por color/calidad
│   ├── pinn_pytorch.py # Physics-Informed Neural Network
│   └── tangelo_pinn_pipeline.py
└── run_openfoam.sh
```

### 4. Control y Operación

| Script | Descripción |
|--------|-------------|
| `reactor_control.py` | Bucle de control sensor-válvula en tiempo real |
| `qdot_final.py` | Integración completa del sistema |

## Uso

```bash
# Simulación VQE (CPU)
python qdot_vqe.py

# Simulación VQE (GPU, requiere Docker)
./run_gpu.sh

# Optimizar diseño del reactor
python reactor_design.py --optimize --production-target 100

# Simulación CFD
cd openfoam_reactor && ./run_openfoam.sh
```

### Docker (GPU)

```bash
docker compose -f docker-compose.gpu.yml up
```

## Stack

- **Simulación cuántica**: Tangelo, PySCF, Qulacs, PennyLane
- **CAD**: CadQuery, OpenSCAD
- **CFD**: OpenFOAM 11
- **ML/Optimización**: PyTorch (PINN), scikit-optimize (Bayesian)
- **Lenguaje**: Python 3.x
- **Infra**: Docker (GPU + OpenFOAM)

## Estado del Proyecto

| Componente | Estado | Siguiente paso |
|:-----------|:------:|:---------------|
| Simulación VQE | Completada | Validar con clústeres de 24+ qubits en GPU |
| Diseño CAD | Optimizado | Imprimir en resina de alta temperatura |
| Simulación CFD | Validada | Verificar con fluidos no-newtonianos |
| Lógica de control | Implementada | Conectar a actuadores físicos |
| Hardware | Pendiente | Fabricar reactor e integrar electrónica |

## Licencia

Proyecto de investigación. Contactar al autor para uso o colaboración.
