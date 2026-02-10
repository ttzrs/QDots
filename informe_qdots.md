# Informe Técnico: Proyecto QDots
## Síntesis de CQDs mediante Micro-reactores de Plasma Frío
**Fecha:** 10 de febrero de 2026
**Estado:** Prototipado Técnico / Gemelo Digital Validado

---

## 1. Visión General
El proyecto automatiza la producción de **Carbon Quantum Dots (CQDs)** a partir de purines de cerdo. Utiliza plasma de descarga de barrera dieléctrica (DBD) para fragmentar materia orgánica y nuclear puntos cuánticos con propiedades ópticas específicas (emisión azul ~450 nm).

## 2. Arquitectura del Software

### A. Módulo Químico (Simulación Cuántica)
Ubicado en los scripts `qdot_vqe*.py`.
- **Tecnología:** Tangelo (IBM/GoodChemistry) + PySCF + Qulacs.
- **Función:** Calcula el Gap HOMO-LUMO de un clúster de carbono dopado con Nitrógeno.
- **Importante:** La energía del gap (~2.77 eV) define el *setpoint* del sensor óptico. Si la simulación predice 450nm, el hardware solo recogerá material que emita en esa banda.

### B. Módulo de Ingeniería (Diseño Paramétrico)
Ubicado en `reactor_design.py` y `reactor_3d_cadquery.py`.
- **Función:** Traduce la cinética química en dimensiones físicas.
- **Variables Clave:**
  - Ancho de canal: 2.0 mm (optimizado para penetración de plasma).
  - Profundidad de líquido: 0.3 - 0.5 mm.
  - Material: Alúmina o Resina Cerámica (por su constante dieléctrica $\epsilon_r \approx 6-10$).

### C. Módulo de Fluidos (CFD)
Ubicado en `openfoam_reactor/`.
- **Simulación:** `simpleFoam`.
- **Optimización:** Se ha realizado una búsqueda bayesiana para maximizar el área de contacto plasma-líquido evitando zonas de estancamiento.

## 3. Estado de los Componentes

| Componente | Estado | Acción Necesaria |
| :--- | :--- | :--- |
| **Simulación VQE** | Completada | Validar con clústeres más grandes (24 qubits) en GPU. |
| **Diseño CAD** | Optimizado | Imprimir `reactor_body.stl` en resina de alta temperatura. |
| **Simulación CFD** | Validada | Verificar caída de presión con fluidos no-newtonianos (purines espesos). |
| **Lógica de Control** | Implementada | Conectar `reactor_control.py` a los actuadores físicos. |

## 4. Guía de Continuación para el Desarrollador

1. **Hardware:** Imprimir el reactor utilizando los parámetros exportados en `reactor_optimized.json`.
2. **Entorno:** Usar `run_gpu.sh` para ejecutar simulaciones químicas pesadas; el entorno Docker ya tiene todas las dependencias (Tangelo, Qulacs-GPU).
3. **Calibración:** El script `qdot_final.py` es el punto de entrada para la operación continua. Debe calibrarse el sensor óptico con una muestra patrón antes de introducir purín.

## 5. Contacto y Recursos
- **Simulaciones:** Ver carpetas `temp_case_*` para datos de sensibilidad.
- **Documentación adicional:** Consultar `GEMINI.md` en el root del proyecto.
