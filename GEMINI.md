# QDots: Síntesis de Carbon Quantum Dots mediante Micro-reactores de Plasma

Este proyecto integra simulación cuántica, diseño paramétrico de ingeniería y dinámica de fluidos computacional para la síntesis automatizada de Carbon Quantum Dots (CQDs) a partir de materia orgánica (purines de cerdo) utilizando plasma de descarga de barrera dieléctrica (DBD).

## Estructura del Proyecto

### 1. Simulación Cuántica (Módulo Químico)
Utiliza el algoritmo **VQE (Variational Quantum Eigensolver)** para predecir las propiedades ópticas (Gap HOMO-LUMO) de los clústeres de carbono dopados con nitrógeno.
- `qdot_vqe.py`: Implementación base con Tangelo y PySCF.
- `qdot_vqe_gpu.py`: Versión optimizada para GPU utilizando Qulacs (vía Docker).
- `chem_backend/tangelo_interface.py`: Interfaz para el gemelo digital químico.

### 2. Diseño del Reactor (Ingeniería)
Diseño paramétrico y optimización del micro-reactor DBD para fabricación mediante impresión 3D (DLP/Cerámica).
- `reactor_design.py`: Motor de diseño paramétrico, optimización de geometría, materiales y parámetros eléctricos.
- `reactor_3d_cadquery.py` / `reactor_3d.scad`: Generación de modelos CAD para fabricación.
- `reactor_optimized.json`: Parámetros de diseño exportados.

### 3. Simulación de Fluidos (OpenFOAM)
Ubicado en `openfoam_reactor/`, contiene las simulaciones de flujo laminar y transporte de especies dentro del reactor.
- `run_openfoam.sh`: Script de ejecución de la simulación.
- `bayesian_opt/`: Resultados de optimización de forma del canal.

### 4. Control y Operación
- `reactor_control.py`: Lógica de control en tiempo real para el bucle de realimentación sensor-válvula.
- `qdot_final.py`: Integración completa del sistema.

## Tecnologías Principales
- **Simulación Cuántica**: Tangelo, PySCF, Qulacs.
- **Ingeniería/CAD**: CadQuery, OpenSCAD.
- **CFD**: OpenFOAM 11.
- **Lenguaje**: Python 3.x.
- **Infraestructura**: Docker (para soporte GPU y OpenFOAM).

## Comandos Clave

### Ejecutar Simulación VQE (CPU)
```bash
python qdot_vqe.py
```

### Ejecutar Simulación VQE (GPU)
```bash
./run_gpu.sh
```

### Optimizar Diseño del Reactor
```bash
python reactor_design.py --optimize --production-target 100
```

### Ejecutar Simulación OpenFOAM
```bash
cd openfoam_reactor && ./run_openfoam.sh
```

## Convenciones de Desarrollo
- **Gemelo Digital**: Todas las decisiones de diseño físico deben estar respaldadas por simulaciones previas en Tangelo (química) u OpenFOAM (fluidos).
- **Materiales**: Preferencia por Alúmina o resinas de alta temperatura debido a la naturaleza del plasma DBD.
- **Control**: El setpoint del sensor óptico se deriva directamente del gap energético calculado en los scripts VQE.
