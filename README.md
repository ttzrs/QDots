# QDots

Síntesis automatizada de **Carbon Quantum Dots (CQDs)** a partir de purines de cerdo mediante un milirreactor de plasma frío (DBD), guiado por un gemelo digital cuántico.

## Concepto

El sistema convierte materia orgánica residual (purines) en nanomateriales fluorescentes de alto valor. Un plasma de descarga de barrera dieléctrica (DBD) fragmenta la materia orgánica, genera radicales de carbono y los recombina en puntos cuánticos con emisión óptica controlada (~450 nm, azul).

La clave es que **ningún parámetro se elige por ensayo y error**: cada decisión de diseño (geometría del canal, voltaje, tiempo de residencia, setpoint del sensor) se deriva de simulaciones cuánticas (VQE) y fluidodinámicas (CFD) previas.

```
Purín → [Pre-tratamiento] → [Reactor DBD] → [Clasificador Óptico] → Producto
                                    ↑              │                       ↑
                              Gemelo Digital   3 zonas LED          Flotabilidad óptica
                           (Tangelo + OpenFOAM)    ↓               (fotoforesis/fototérmica)
                                         Excitación λ → QDots suben → Colección arriba
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

### 4. Clasificador Optofluidico (Post-reactor)

Separación selectiva de CQDs con **tres modos de excitación** implementados y comparados cuantitativamente:

| Modo | Mecanismo | Pe (Péclet) | Viabilidad |
|------|-----------|:-----------:|:----------:|
| **LED** | Fuerzas fototérmicas/fotofórica directas | ~0.0002 | No viable |
| **Láser directo** | Fuerzas ópticas focalizadas (gradiente + radiación) | ~0.0001 | No viable |
| **Opto-térmico** | Láser + sustrato Au → termoforesis (efecto Soret) | **~1-10** | **Viable** |

El análisis demuestra que las fuerzas ópticas directas (LED o láser) son ~10 órdenes de magnitud insuficientes vs difusión browniana para CQDs de 2-5 nm. El modo opto-térmico (láser 500 mW focalizado a 10 μm sobre película de oro) genera gradientes térmicos macroscópicos (~10⁶ K/m) que producen termoforesis con Pe ≥ 1.

```
  Modo Opto-térmico (VIABLE):

      Fibra óptica (láser)           ← Láser focalizado (500 mW)
            ↓
  +---------|---------------------+
  |  [=====Puerto superior=====] |  ← Colección QDots (arriba)
  |                               |
  |    ←--- termoforesis ---→     |  ← QDots migran por gradiente térmico
  |    ~~~~~~~~~~~~~~~~~~~~~~~~   |  ← ∇T ~ 10⁶ K/m
  |  [###Sustrato Au (50nm)####]  |  ← Película de oro (calentamiento)
  |  [=====Puerto inferior======] |  ← Colección debris (abajo)
  +-------------------------------+

  Selectividad: QDots absorbentes generan calentamiento local adicional
                → coeficiente de Soret diferencial → acumulación diferencial
```

| Script | Descripción |
|--------|-------------|
| `classifier_design.py` | Diseño paramétrico con 3 modos (LED, láser, opto-térmico) + comparación |
| `classifier_3d_cadquery.py` | Modelo 3D CadQuery (LED o fibra óptica + sustrato Au) |
| `classifier_3d.scad` | Modelo 3D OpenSCAD con modo LED/opto-térmico |

**Especificaciones del clasificador (modo opto-térmico):**
- 3 zonas con láser focalizado: 520 nm / 405 nm / 365 nm
- Potencia láser: 500 mW, beam waist: 10 μm
- Sustrato: película de oro (50 nm), absorptancia: 0.40
- Gradiente térmico: ~10⁷ K/m → velocidad termoforética ~15-120 μm/s
- Potencial de atrapamiento: ~5.8 kT (estable)
- Control de temperatura del sustrato (máx 80°C)

### 5. Control y Operación

| Script | Descripción |
|--------|-------------|
| `reactor_control.py` | Control del reactor + clasificador óptico (ReactorController + ClassifierController) |
| `qdot_final.py` | Integración completa del sistema |

## Uso

```bash
# Simulación VQE (CPU)
python qdot_vqe.py

# Simulación VQE (GPU, requiere Docker)
./run_gpu.sh

# Optimizar diseño del reactor
python reactor_design.py --optimize --production-target 100

# Clasificador: comparar los 3 modos de excitación
python classifier_design.py --compare

# Clasificador: diseño opto-térmico (modo por defecto, viable)
python classifier_design.py --design --mode optothermal

# Clasificador: diseño LED (baseline, demuestra que no funciona)
python classifier_design.py --design --mode led

# Clasificador: diseño láser directo (mejora insuficiente)
python classifier_design.py --design --mode laser

# Clasificador: optimización
python classifier_design.py --optimize --mode optothermal

# Vista previa 3D (modo opto-térmico)
python classifier_3d_cadquery.py --preview --mode optothermal

# Demo del controlador (reactor + clasificador con control láser)
python reactor_control.py

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
| Diseño CAD reactor | Optimizado | Imprimir en resina de alta temperatura |
| Simulación CFD | Validada | Verificar con fluidos no-newtonianos |
| Clasificador óptico | 3 modos implementados | Modo opto-térmico viable (Pe~1-10); LED y láser directo descartados |
| Lógica de control | Implementada + control láser | Conectar a actuadores físicos (reactor + clasificador + láser) |
| Hardware | Pendiente | Fabricar reactor + clasificador e integrar electrónica |

## Licencia

Proyecto de investigación. Contactar al autor para uso o colaboración.
