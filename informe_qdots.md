# Informe Tecnico: Proyecto QDots
## Sintesis de CQDs mediante Milirreactor de Plasma Frio con Clasificacion Paralela
**Fecha:** 11 de febrero de 2026
**Estado:** Gemelo Digital Validado / Prototipado Pendiente

---

## 1. Vision General

El proyecto automatiza la produccion de **Carbon Quantum Dots (CQDs)** a partir de purines de cerdo. Un milirreactor de plasma frio (DBD, Descarga de Barrera Dielectrica) con barrera catalitica de TiO2 anatase fragmenta materia organica y nuclea puntos cuanticos con emision azul (~460 nm). Un sistema de 50 clasificadores opticos en paralelo purifica el producto con 91% de recovery y 99.7% de pureza.

**Cifras clave:**
- Produccion bruta: **505 mg/h** (milirreactor MC 8x300mm)
- Produccion neta purificada: **495 mg/h** (11.9 g/dia)
- Precio de referencia: **30 EUR/g**
- Payback estimado: **38 dias**

## 2. Arquitectura del Sistema

### Pipeline Completo

```
PURIN -> Pre-tratamiento -> MILIRREACTOR DBD -> MANIFOLD -> 50x CLASIFICADORES -> PRODUCTO
         (dilucion)         MC 8ch, TiO2      distribuidor   0.1 mL/min c/u      495 mg/h
                            5 mL/min, 27 W                   800 mW laser         99.7%
                            505 mg/h                         4S+3WR+2R
                                 ^                                |
                                 |                                v
                                 +<--- 80% waste recirculado --- WASTE
                                                                 20% -> fertilizante
```

### A. Milirreactor DBD (Principal)

Ubicado en `reactor_scaleup.py`. Escala el microreactor base (`reactor_design.py`, ~50 mg/h) a dimensiones milimetricas.

**Configuracion optimizada (multi-channel 8x300mm):**
- 8 canales paralelos: 2 mm ancho x 0.5 mm alto x 300 mm largo
- Barrera dielectrica: TiO2 anatase (epsilon_r = 40)
  - Doble funcion: barrera dielectrica + fotocatalizador activado por UV del plasma
- Plasma: DBD frio no-termico
  - Te ~ 1.5 eV (electrones calientes), Tgas < 60 C (gas frio)
  - Pulsos de 100 ns (< 500 ns limite para frio)
  - Radicales: OH*, O*, O2- escalan con potencia
- Enfriamiento activo: serpentin de refrigerante integrado
- Produccion: 505 mg/h a 460 nm, 27 W

**Topologias disponibles:**

| Topologia | Flujo | Produccion | Ventaja |
|-----------|-------|------------|---------|
| Multi-channel (8ch) | 5 mL/min | 505 mg/h | Escalado lineal, validado |
| Falling film | 50 mL/min | ~300 mg/h | Gran area, simple |
| Annular | 50 mL/min | ~250 mg/h | Compacto |
| Bubble column | 50 mL/min | ~200 mg/h | Maximo mezclado 3D |

### B. Clasificacion Optica Paralela

Ubicado en `parallel_classifier.py` (sistema completo), `continuous_production.py` (cascada inline), `classifier_design.py` (diseno base).

**Problema resuelto:** El reactor opera a 5 mL/min pero el clasificador funciona optimamente a 0.1 mL/min. Solucion: 50 clasificadores en paralelo (numbering-up).

**Parametros mejorados vs originales:**

| Parametro | Original | Mejorado | Efecto |
|-----------|----------|----------|--------|
| Volumen camara | 5 uL | 20 uL | t_res 4x mayor |
| Potencia laser | 500 mW | 800 mW | v_thermo +60% |
| Etapas separacion | 3 | 4 | +1 etapa |
| Etapas refinamiento | 3 | 2 | -1 (menos perdida) |
| Waste recapture | 2 | 3 | +1 etapa |
| **Recovery** | **52%** | **91%** | **+39 pp** |

**Waste recirculation (estado estacionario):**
```
boost = 1 / (1 - 0.80 * (1 - 0.91)) = 1.08x
net   = 505 * 0.91 * 1.08 = 495 mg/h
```

### C. Modulo Quimico (Simulacion Cuantica)

Ubicado en `qdot_vqe*.py`.
- **Tecnologia:** Tangelo (IBM/GoodChemistry) + PySCF + Qulacs.
- **Funcion:** Calcula el Gap HOMO-LUMO de un cluster de carbono dopado con nitrogeno.
- La energia del gap (~2.77 eV) define el setpoint del sensor optico.

### D. Modulo de Fluidos (CFD)

Ubicado en `openfoam_reactor/`.
- **Simulacion:** simpleFoam.
- **Optimizacion:** Busqueda bayesiana para maximizar area de contacto plasma-liquido.

## 3. Estado de los Componentes

| Componente | Estado | Accion Necesaria |
|:-----------|:-------|:-----------------|
| **Milirreactor DBD (MC 8ch, TiO2)** | Simulado, 505 mg/h | Fabricar prototipo fisico |
| **Plasma frio (non-thermal)** | Modelado | Validar Te/Tgas experimentalmente |
| **50 clasificadores paralelos** | Simulado, 91% recovery | Fabricar primer chip clasificador |
| **Waste recirculation** | Modelado, boost 1.08x | Implementar manifold + bomba |
| **Microreactor base** | Optimizado, ~50 mg/h | Escalado al milirreactor (hecho) |
| **Simulacion VQE** | Completada | Validar con 24+ qubits en GPU |
| **Simulacion CFD** | Validada | Verificar con fluidos no-newtonianos |
| **Control** | Implementado | Conectar a actuadores fisicos |

## 4. Economia

| Concepto | Valor |
|----------|-------|
| Costo reactor | ~$5,000 |
| Costo clasificadores (50x) | ~$10,000 |
| Total hardware | ~$15,000 |
| Produccion diaria | 11.9 g/dia |
| Valor produccion (@30 EUR/g) | ~$392/dia |
| Costo energia (269 W, 24h) | ~$0.77/dia |
| Payback | ~38 dias |

## 5. Guia de Continuacion

1. **Milirreactor:** Fabricar el reactor multi-channel con TiO2 como barrera dielectrica. Parametros en `reactor_scaleup.py`.
2. **Clasificador:** Fabricar un chip clasificador opto-termico (sustrato Au 50 nm, laser 800 mW). Validar recovery con muestra sintetica.
3. **Integracion:** Conectar reactor -> manifold -> clasificadores con bomba peristaltica y tubing de PTFE.
4. **Calibracion:** Usar `reactor_control.py` para el bucle de control. Calibrar sensor optico con muestra patron antes de introducir purin.
5. **Waste loop:** Implementar recirculacion con bomba secundaria y mezclador estatico en el inlet.

## 6. Comandos Principales

```bash
# Sistema completo (milirreactor + clasificadores paralelos)
python parallel_classifier.py

# Milirreactor: comparar topologias
python reactor_scaleup.py

# Produccion continua: cascada inline
python continuous_production.py

# Microreactor base
python reactor_design.py --optimize

# Clasificador optico
python classifier_design.py --design --mode optothermal

# Simulacion VQE
python qdot_vqe.py
```
