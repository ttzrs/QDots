# Informe Tecnico: Proyecto QDots
## Sintesis de CQDs mediante Milirreactor de Plasma Frio con Clasificacion Paralela
**Fecha:** 12 de febrero de 2026
**Estado:** Gemelo Digital Validado + Protocolo Experimental Completo / Prototipado Pendiente

---

## 1. Vision General

El proyecto automatiza la produccion de **Carbon Quantum Dots (CQDs)** a partir de purines de cerdo. Un milirreactor de plasma frio (DBD, Descarga de Barrera Dielectrica) con barrera catalitica de TiO2 anatase fragmenta materia organica y nuclea puntos cuanticos con emision azul (~480 nm). Un sistema de clasificadores opticos en paralelo purifica el producto.

**Cifras clave (configuracion optimizada 16ch x 500mm):**
- Produccion bruta: **1211 mg/h** (milirreactor MC 16x500mm, validado CFD)
- Longitud de onda: **480 nm** (emision azul-verde)
- Tamano CQD: **2.59 nm** (confinamiento cuantico fuerte)
- Potencia: **86.4 W** (12 kV, 30 kHz)
- Score de optimizacion: **0.932** (CFD-validado)
- Precio de referencia: **30 EUR/g**

### Evolucion del diseno

| Version | Config | Produccion | Longitud de onda | Score |
|---------|--------|------------|------------------|-------|
| Base (microreactor) | 1ch, 0.5mm | ~50 mg/h | ~460 nm | - |
| MC v1 | 8ch x 300mm | 505 mg/h | 460 nm | 0.423 |
| **Optimizado (actual)** | **16ch x 500mm** | **1211 mg/h** | **480 nm** | **0.932** |
| Cantera+Tangelo Opt | 22ch x 500mm | 1826 mg/h | 467 nm | 0.928 |

---

## 2. Arquitectura del Sistema

### Pipeline Completo

```
PURIN -> Pre-tratamiento -> MILIRREACTOR DBD -> MANIFOLD -> CLASIFICADORES -> PRODUCTO
         (2 g/L dilucion)   MC 16ch, TiO2     distribuidor   opticos          1211 mg/h
                             15 mL/min, 86 W                  paralelos        480 nm
                             1211 mg/h                                          2.59 nm
                                 ^                                |
                                 |                                v
                                 +<--- 80% waste recirculado --- WASTE
                                                                 20% -> fertilizante
```

### A. Milirreactor DBD — Configuracion Optimizada (16ch x 500mm)

Validado por: `cfd_validate_reactor.py` (2D FV Navier-Stokes + especies + termico + RTD)
Gemelo digital: `experimental_validation.py`
Datos CFD: `optimization_results/cfd_validation_results.json`

**Parametros del reactor:**

| Parametro | Valor | Tolerancia |
|-----------|-------|------------|
| Canales paralelos | 16 | - |
| Ancho de canal | 2.0 mm | +/-0.05 mm |
| Alto de canal | 0.5 mm | +/-0.02 mm |
| Largo de canal | 500 mm | +/-0.5 mm |
| Paso entre canales | 4.0 mm c-a-c | +/-0.1 mm |
| Flujo total | 15.0 mL/min | +/-5% |
| Voltaje | 12.0 kV | +/-0.5 kV |
| Frecuencia | 30.0 kHz | +/-0.5 kHz |
| Concentracion precursor | 2.0 g/L | +/-0.1 g/L |
| Ancho de pulso | 100 ns | < 500 ns |

**Predicciones del gemelo digital (95% CI):**

| Parametro | Prediccion | IC 95% inferior | IC 95% superior | Metodo de medida |
|-----------|------------|-----------------|-----------------|------------------|
| Produccion (mg/h) | 1211 | 499 | 1923 | Gravimetrico + PL |
| Longitud de onda (nm) | 480 | 386 | 574 | Espectroscopia PL (exc. 365 nm) |
| Tamano (nm) | 2.59 | 1.78 | 3.40 | TEM (200+ particulas) + DLS |
| Potencia (W) | 86.4 | 69.5 | 103.3 | V x I (osciloscopio) |
| T_max (C) | 21.3 | 16.3 | 26.4 | Termopar tipo K |
| Delta_P (Pa) | 1736 | 1464 | 2008 | Sensor diferencial |

**Hidrodinamica (CFD):**
- Reynolds: Re = 13.6 (laminar)
- Peclet: Pe = 41.9 (flujo piston dominante)
- Velocidad media: 26.1 mm/s
- Tiempo de residencia: 18.8 s
- Conversion de precursor: 51%

**Plasma frio no-termico:**
- Te ~ 1.5 eV (electrones calientes)
- Tgas ~ 333 K (gas frio, Delta_T < 2 C en liquido)
- Densidad electronica: ne ~ 1.8 x 10^11 cm^-3
- Radicales OH: ~10^16 cm^-3 (estado estacionario plasma)
- Barrera dielectrica: TiO2 anatase (epsilon_r = 40)
  - Doble funcion: barrera dielectrica + fotocatalizador activado por UV del plasma

**Modelo de confinamiento cuantico:**
```
E_gap = E_bulk + A_conf / d^2
      = 1.50 eV + 7.26 eV*nm^2 / (2.59 nm)^2
      = 2.58 eV -> 480 nm
```

### B. Clasificacion Optica Paralela

Ubicado en `parallel_classifier.py`, `reactor_control.py`.

**Zonas del clasificador:**

| Zona | Longitud de onda | LED | Accion valvula |
|------|------------------|-----|----------------|
| Green QDots | > 500 nm | 520 nm | Coleccion principal |
| Blue QDots | 440-500 nm | 405 nm | Coleccion premium |
| UV QDots | < 440 nm | 365 nm | Recirculacion |

### C. Modulo Quimico (Simulacion Cuantica)

Ubicado en `qdot_vqe*.py`, `chem_backend/tangelo_interface.py`.
- **Tecnologia:** Tangelo 0.4.3 (IBM/GoodChemistry) + PySCF + Qulacs
- **Funcion:** Calcula el Gap HOMO-LUMO de clusters de carbono dopados con nitrogeno
- **Resultados Tangelo para 16ch reactor:**
  - 4 zonas simuladas: inlet (298K), plasma (333K + E), high_E (333K + 2E), outlet (323K)
  - Energias de activacion: oxidacion, nucleacion, crecimiento por zona
  - Gap estimado Tangelo: ~2.78 eV (446 nm) — complementa modelo de confinamiento

### D. Validacion CFD (2D Finite-Volume)

Ubicado en `cfd_validate_reactor.py`. Solver Navier-Stokes 2D acoplado:
- **Flujo:** Perfil de Poiseuille rectangular, caida de presion analitica
- **Especies:** Transporte adveccion-difusion de OH, C_org, CQD con cinetica de oxidacion/nucleacion/crecimiento
- **Termico:** Calentamiento por plasma + conveccion + enfriamiento
- **RTD:** Modelo de Taylor-Aris 1D explicito (2000 pasos)
- **PINN surrogate:** Red 5->128->256->128->8 (GELU+BatchNorm), R^2 > 0.95 en 6/8 outputs

Comparativa de configuraciones CFD:

| Config | Produccion | Lambda | Score | In-spec |
|--------|------------|--------|-------|---------|
| 8ch x 300mm (base) | 156 mg/h | 528 nm | 0.423 | No |
| **16ch x 500mm (param)** | **1211 mg/h** | **480 nm** | **0.932** | **Si** |
| 22ch x 500mm (Cantera) | 1826 mg/h | 467 nm | 0.928 | Si |

---

## 3. Bill of Materials (BOM) — Reactor 16ch x 500mm

**41 items | Costo total estimado: 6,775 EUR**

### 3.1 Estructura

| Item | Especificacion | Cant | Ud | EUR |
|------|---------------|------|----|-----|
| Placa de vidrio borosilicato (cuerpo) | 200 x 550 x 10 mm, pulida | 2 | pcs | 120 |
| Placa base acero inoxidable | 200 x 560 x 15 mm, 316L | 1 | pcs | 85 |
| Set de juntas PTFE | 1.5 mm espesor, corte laser patron canales | 4 | pcs | 30 |
| Tornillos + tuercas M4 inox | A2-70, 30 mm longitud | 32 | pcs | 15 |
| Pines de alineacion | 3 mm dia, 15 mm longitud, acero endurecido | 8 | pcs | 10 |
| | | | **Subtotal** | **260** |

### 3.2 Electrodos

| Item | Especificacion | Cant | Ud | EUR |
|------|---------------|------|----|-----|
| Electrodos cobre (HV) | 0.1 mm espesor, 16 tiras x 2 x 500 mm | 16 | pcs | 45 |
| Electrodos cobre (tierra) | 0.1 mm espesor, 16 tiras x 2 x 500 mm | 16 | pcs | 45 |
| Cable HV (silicona) | 20 kV rated, 2 m longitud | 2 | m | 25 |
| Conector HV (banana) | 30 kV rated | 2 | pcs | 20 |
| | | | **Subtotal** | **135** |

### 3.3 Dielectrico / Catalizador

| Item | Especificacion | Cant | Ud | EUR |
|------|---------------|------|----|-----|
| TiO2 anatase P25 | 21 nm primario, 50 m^2/g BET | 50 | g | 35 |
| Binder sol-gel TiO2 | Titanio isopropoxido, 97% | 100 | mL | 40 |
| Etanol (solvente coating) | 99.5%, anhidro | 500 | mL | 15 |
| | | | **Subtotal** | **90** |

### 3.4 Fluidica

| Item | Especificacion | Cant | Ud | EUR |
|------|---------------|------|----|-----|
| Bomba de jeringa | 0.1-50 mL/min, doble canal | 1 | pcs | 800 |
| Tubing PTFE (1/16" OD) | 0.5 mm ID, rollo 5 m | 2 | rolls | 30 |
| Manifold inlet | 1-a-16 splitter, PEEK | 1 | pcs | 150 |
| Manifold outlet | 16-a-1 colector, PEEK | 1 | pcs | 150 |
| Valvulas check | PEEK body, 0.5 psi cracking | 4 | pcs | 80 |
| Valvulas solenoide 3-vias (clasificador) | PTFE wetted, 12V, <100 ms | 3 | pcs | 135 |
| Frascos de coleccion | 50 mL vidrio ambar, tapa rosca | 10 | pcs | 25 |
| | | | **Subtotal** | **1,370** |

### 3.5 Electrica

| Item | Especificacion | Cant | Ud | EUR |
|------|---------------|------|----|-----|
| Fuente HV (AC) | 0-20 kV, 10-30 kHz, 200 W max | 1 | pcs | 1,500 |
| Generador de funciones | 1 Hz - 1 MHz, salida TTL | 1 | pcs | 350 |
| Sonda HV (1000:1) | 0-40 kV, 100 MHz BW | 1 | pcs | 250 |
| Transformador de corriente | Pearson coil, 0.1 V/A | 1 | pcs | 200 |
| | | | **Subtotal** | **2,300** |

### 3.6 Sensores

| Item | Especificacion | Cant | Ud | EUR |
|------|---------------|------|----|-----|
| Espectrometro UV-Vis (fibra) | 350-700 nm, 1 nm resolucion | 1 | pcs | 600 |
| Sonda fibra optica (emision) | 400 um core, conector SMA | 2 | pcs | 120 |
| Fuente LED UV (excitacion) | 365 nm, 5 W, colimada | 1 | pcs | 80 |
| Termopar tipo K | 0.5 mm dia, 150 mm, funda PTFE | 4 | pcs | 60 |
| Transductor de presion diferencial | 0-10 kPa, salida 4-20 mA | 1 | pcs | 180 |
| Caudalimetro (liquido) | 0.1-50 mL/min, PEEK wetted | 1 | pcs | 250 |
| | | | **Subtotal** | **1,290** |

### 3.7 Refrigeracion

| Item | Especificacion | Cant | Ud | EUR |
|------|---------------|------|----|-----|
| Chiller recirculante | 200 W refrigeracion, 5-40 C, 2 L/min | 1 | pcs | 600 |
| Serpentin cobre refrigeracion | 3 mm OD, 2 mm ID, 600 mm x 17 pasos | 1 | set | 90 |
| Pasta termica | Base plata, >5 W/(m*K) | 1 | tubo | 15 |
| | | | **Subtotal** | **705** |

### 3.8 Control

| Item | Especificacion | Cant | Ud | EUR |
|------|---------------|------|----|-----|
| DAQ board (Arduino/ESP32) | ADC 16-bit, 8 entradas analogicas, WiFi | 1 | pcs | 35 |
| Osciloscopio | 100 MHz, 4 canales, 1 GS/s | 1 | pcs | 450 |
| Placa de reles (control valvulas) | 8 canales, 12V, optoacoplada | 1 | pcs | 15 |
| | | | **Subtotal** | **500** |

### 3.9 Consumibles

| Item | Especificacion | Cant | Ud | EUR |
|------|---------------|------|----|-----|
| Precursor: purin diluido | 2.0 g/L, filtrado <100 um | 20 | L | 5 |
| Agua DI (grado reactivo) | 18.2 MOhm*cm | 50 | L | 10 |
| Trazador RTD (Rodamina B) | 1 mg/mL solucion stock | 100 | mL | 20 |
| Estandar fluoresceina (calibracion) | 1 ug/mL en agua DI | 100 | mL | 15 |
| Nitrogeno gas (carrier) | 99.99%, cilindro 50 L | 1 | cil | 40 |
| Membranas de filtro (aislamiento producto) | 0.22 um, PVDF, 47 mm | 50 | pcs | 35 |
| | | | **Subtotal** | **125** |

### Resumen BOM

| Categoria | Items | Costo (EUR) |
|-----------|-------|-------------|
| Estructura | 5 | 260 |
| Electrodos | 4 | 135 |
| Dielectrico/Catalizador | 3 | 90 |
| Fluidica | 7 | 1,370 |
| Electrica | 4 | 2,300 |
| Sensores | 6 | 1,290 |
| Refrigeracion | 3 | 705 |
| Control | 3 | 500 |
| Consumibles | 6 | 125 |
| **TOTAL** | **41** | **6,775** |

---

## 4. Protocolo de Fabricacion (8 fases, 54 pasos, ~46 horas)

### Fase 1: Preparacion (4 h)
1. Limpiar placas de vidrio en solucion piranha (3:1 H2SO4:H2O2) durante 30 min
2. Enjuagar 5x con agua DI, secar bajo flujo de N2
3. Verificar dimensiones de placa: 200 x 550 x 10 mm (tolerancia +/-0.1 mm)
4. Preparar suspension de coating TiO2: 5 wt% P25 en etanol + 1% Ti-isopropoxido
5. Sonicar suspension 30 min, filtrar por malla de 5 um

### Fase 2: Mecanizado de Canales (8 h)
1. Fresar CNC 16 canales paralelos en placa de vidrio inferior
2. Dimensiones de canal: 2.0 x 0.5 x 500 mm (**tolerancia +/-0.05 mm**)
3. Paso entre canales: 4.0 mm centro-a-centro
4. Cavidad manifold inlet: 10 x 64 x 0.5 mm
5. Cavidad manifold outlet: identica al inlet
6. Desbarbar bordes de canales con papel SiC grano 800
7. Limpieza ultrasonica de placa mecanizada en acetona (15 min) + agua DI (15 min)
8. Verificar profundidad con perfilometro (+/-0.02 mm requerido)

### Fase 3: Integracion de Electrodos (6 h)
1. Cortar 16 tiras de electrodo HV: 0.1 x 2 x 500 mm cobre
2. Cortar 16 tiras de electrodo tierra: dimensiones identicas
3. Posicionar electrodos HV en placa inferior entre canales (adhesivo)
4. Posicionar electrodos tierra en placa superior, alineados con HV
5. Verificar alineacion electrodo-canal: **tolerancia +/-0.2 mm**
6. Soldar bus bar HV conectando todos los electrodos HV en paralelo
7. Soldar bus bar tierra conectando todos los electrodos tierra en paralelo
8. Verificacion de resistencia: <0.5 Ohm del conector a cada punta de electrodo

### Fase 4: Coating Dielectrico y Catalizador (12 h)
1. Enmascarar fondos de canal (zona liquida) con cinta Kapton
2. Dip-coat superficies de electrodo con suspension TiO2 (3 capas)
3. Entre capas: secar a 60 C durante 30 min
4. Retirar mascara Kapton de fondos de canal
5. Aplicar capa fina de TiO2 a superficies de gap de gas (1 capa, mas delgada)
6. Curado final: rampa a 200 C a 2 C/min, mantener 2 h, enfriar naturalmente
7. Objetivo coating: 2 mg/cm^2 TiO2, 0.5 mm espesor, 60% porosidad
8. Verificar adhesion del coating: test de cinta Scotch (debe pasar)

### Fase 5: Ensamblaje del Manifold (4 h)
1. Instalar manifold PEEK 1-a-16 inlet con racores Swagelok
2. Instalar manifold PEEK 16-a-1 outlet
3. Conectar tubing PTFE (1/16" OD) a inlet y outlet
4. Instalar valvulas check en inlet (prevenir reflujo)
5. Montar puertos de termopar en inlet, outlet, mitad de largo (x2)
6. Montar toma de presion en inlet (antes del manifold)

### Fase 6: Sistema de Refrigeracion (4 h)
1. Enrutar serpentin de cobre entre canales en placa inferior
2. Aplicar pasta termica en superficies de contacto serpentin-placa
3. Conectar inlet/outlet del serpentin al chiller recirculante
4. Verificar flujo del serpentin: >100 mL/min con <0.5 bar caida de presion
5. Test termico: chiller a 15 C, verificar que placa alcanza 18 C en <5 min

### Fase 7: Sellado y Test de Fugas (4 h)
1. Colocar juntas PTFE en ambos lados de la placa de canales
2. Ensamblar placa superior, alinear con pines, apretar tornillos a 2 N*m en patron estrella
3. Presurizar con N2 a 50 kPa, mantener 10 min, verificar zero caida de presion
4. Llenar con agua DI a 5 mL/min, verificar fugas en todos los racores
5. Incrementar flujo a 20 mL/min, verificar sin fugas a presion de operacion
6. Test con colorante: inyectar 0.1% Rodamina B, verificar salida uniforme de todos los canales

### Fase 8: Puesta en Marcha Electrica (4 h)
1. Conectar fuente HV a bus bars de electrodos via cable HV
2. Conectar sonda HV y transformador de corriente al osciloscopio
3. Test dielectrico: rampa voltaje 0 -> 5 kV a 20 kHz, verificar sin arcos
4. Rampa a 8 kV: verificar brillo uniforme del plasma a traves del vidrio (visual)
5. Rampa a 12 kV a 30 kHz: grabar formas de onda V, I
6. Calcular potencia: P = (1/T) integral V(t)*I(t)dt sobre 10 ciclos
7. Verificar potencia dentro del 10% del predicho **86.4 W**
8. Operar 1 hora con flujo de agua: verificar operacion estable, sin arcos

---

## 5. Instrumentacion de Medida

| Instrumento | Modelo/Ejemplo | Rango | Resolucion | Medicion | Calibracion | Incertidumbre |
|-------------|---------------|-------|------------|----------|-------------|---------------|
| Espectrometro UV-Vis fibra | Ocean Insight Flame-S | 350-700 nm | 1 nm FWHM | Espectro PL de CQDs | Lampara Hg-Ar (435.8, 546.1, 696.5 nm) | +/-1 nm lambda, +/-5% intens. |
| Termopar tipo K (x4) | 0.5 mm dia, funda PTFE | 0-200 C | 0.1 C | T reactor en 4 puntos | Ref. hielo (0 C) + agua hirviendo (100 C) | +/-0.5 C |
| Transductor presion dif. | Honeywell ASDX | 0-10 kPa | 1 Pa | Caida presion reactor | Dead-weight tester o columna agua | +/-0.5% FS (+/-50 Pa) |
| Caudalimetro | Sensirion SLF3x | 0.1-50 mL/min | 0.01 mL/min | Caudal liquido | Gravimetrico (recoger + pesar) | +/-2% lectura |
| Sonda HV (1000:1) | Tektronix P6015A | 0-40 kV | 10 V | Forma onda voltaje | Fuente AC conocida 1 kV, 10 kV | +/-3% lectura |
| Osciloscopio (4ch) | 100 MHz, 1 GS/s | DC-100 MHz | 8-bit vertical | V(t), I(t) para calculo potencia | Certificado fabrica | +/-1.5% vert, +/-0.01% base tiempo |

### Adquisicion de Datos (DAQ)

| Canal | Sensor | Frecuencia | Resolucion | Interfaz | Almacenamiento | Trigger |
|-------|--------|------------|------------|----------|----------------|---------|
| Temperatura | TC tipo K x4 | 1 Hz | 0.1 C | DAQ analogico (4ch) | ~350 KB/h | Continuo |
| Presion | Transductor dif. | 10 Hz | 1 Pa | DAQ analogico (1ch) | ~90 KB/h | Continuo |
| Flujo | Masa termica | 1 Hz | 0.01 mL/min | DAQ analogico (1ch) | ~90 KB/h | Continuo |
| Voltaje | Sonda HV 1000:1 | 100 Hz | 10 V | Osciloscopio Ch1 | ~900 KB/h (bursts) | 10 s burst cada 60 s |
| Corriente | Pearson coil | 100 Hz | 1 mA | Osciloscopio Ch2 | ~900 KB/h (bursts) | Sincronizado con voltaje |
| Espectro PL | Espectrometro fibra | 0.1 Hz | 1 nm, 2048 px | USB, SDK vendor | ~1.5 MB/h | Cada 10 s durante sintesis |
| Estado valvulas | Digital I/O | Evento | Binario (3 valv) | DAQ digital | <10 KB/h | Cambio de estado |

---

## 6. Protocolo Experimental de Validacion (3 fases, 10 dias)

### Fase 1: Puesta en Marcha (2 dias)

**Objetivo:** Verificar integridad del reactor, distribucion de flujo, ignicion del plasma.

| Paso | Accion | Detalle | Criterio de aceptacion |
|------|--------|---------|------------------------|
| 1 | Test de fugas | Presurizar con N2 a 50 kPa. Mantener 10 min. | Delta_P < 1 Pa en 10 min |
| 2 | Calibracion de flujo | Bomba a 15 mL/min. Medir salida con probeta 5 min. | 15.0 +/-0.75 mL/min (+/-5%) |
| 3 | Distribucion de flujo | Inyectar pulso Rodamina B en inlet. Fotografiar canales outlet. | 16 canales con colorante en <2 s entre si |
| 4 | Rampa ignicion plasma | Rampa: 0->5->8->10->12 kV a 30 kHz (2 min/paso). Monitor arcos. | Brillo uniforme en todos los canales a 12 kV, sin arcos |
| 5 | Baseline termico | Plasma 12 kV, 30 kHz + agua 15 mL/min durante 30 min. Grabar T x4. | T_max < 70 C, estado estacionario en <15 min |
| 6 | Verificacion refrigeracion | Chiller mantiene refrigerante a 15+/-1 C, flujo >100 mL/min. | Delta_T refrigerante < 5 C a traves del reactor |

### Fase 2: Caracterizacion Baseline (3 dias)

**Objetivo:** Validar predicciones CFD (RTD, presion, termico, OES).

| Paso | Accion | Detalle | Criterio de aceptacion |
|------|--------|---------|------------------------|
| 1 | RTD con trazador | Pulso 0.1 mL Rodamina B en inlet. Grabar fluorescencia outlet (10 Hz). | Pe ~ 42 +/-10 (confirmar flujo piston) |
| 2 | Caida de presion | Medir Delta_P a 5, 10, 15, 20 mL/min (3 replicas c/u). Graficar. | Delta_P a 15 mL/min = **1736 +/-272 Pa** |
| 3 | Mapa termico | Grabar T en 4 puntos bajo plasma (12 kV, 30 kHz, 15 mL/min). SS = 30 min. | T_max = **21.3 +/-5.0 C** |
| 4 | OES para emision OH* | Espectro OES (300-400 nm) a traves de ventana vidrio. Identificar OH(A-X) 309 nm. | Banda OH(A-X) 309 nm presente, intensidad escala con V |
| 5 | Medida de potencia | Grabar V(t) e I(t) a 12 kV, 30 kHz. Calcular P = integral V*I dt/T (100 ciclos). | P = **86.4 +/-16.9 W** |
| 6 | Matriz voltaje-frecuencia | 3 voltajes (10, 12, 14 kV) x 3 frecuencias (20, 25, 30 kHz). Grabar P, T, OES. | Potencia escala como V^2*f (dentro de 15%) |

### Fase 3: Sintesis de CQDs y Validacion (5 dias)

**Objetivo:** Producir CQDs, medir propiedades, comparar con gemelo digital.

| Paso | Accion | Detalle | Criterio de aceptacion |
|------|--------|---------|------------------------|
| 1 | Preparacion precursor | Diluir purin a 2 g/L con agua DI. Filtrar 100 um. Sonicar 15 min. | Suspension homogenea, sin solidos >100 um |
| 2 | Run baseline (3 replicas) | Condiciones nominales (12 kV, 30 kHz, 15 mL/min) 2 h c/u. Recoger en viales ambar. | Fluorescencia visible bajo lampara UV 365 nm |
| 3 | Espectroscopia PL | Excitar producto a 365 nm. Grabar emision 400-700 nm. Extraer lambda pico, FWHM. | lambda_pico = **480 +/-94 nm** |
| 4 | Caracterizacion TEM | Drop-cast en grid Cu. Imagar 200+ particulas. Medir distribucion tamano. | d_mean = **2.59 +/-0.81 nm** |
| 5 | Medida tamano DLS | Tamano hidrodinamico en agua DI. 3 medidas x 15 runs c/u. | D_h = 3.4 +/-0.8 nm (1.3x TEM esperado) |
| 6 | Tasa de produccion | Gravimetrico: filtrar 100 mL producto (0.22 um), secar 60 C, pesar. + curva UV-Vis. | Produccion = **1211 +/-712 mg/h** |
| 7 | Barrido caudal (3 replicas c/u) | Test 10, 15, 20 mL/min a 12 kV fijo. Medir produccion y lambda. | Produccion pico ~15 mL/min, shift lambda <20 nm |
| 8 | Validacion sistema control | Conectar clasificador. Operar 1 h. Grabar acciones valvula vs lecturas PL. | Precision clasificacion >90% (vs espectroscopia manual) |

---

## 7. Criterios de Aceptacion — Tabla Resumen

| Parametro | Prediccion | IC 95% Inf | IC 95% Sup | Metodo | Pass/Fail |
|-----------|------------|------------|------------|--------|-----------|
| Produccion (mg/h) | 1211 | 499 | 1923 | Gravimetrico + PL | Medida dentro IC |
| Longitud de onda (nm) | 480 | 386 | 574 | PL spectroscopy | Medida dentro IC |
| Tamano (nm) | 2.59 | 1.78 | 3.40 | TEM + DLS | Medida dentro IC |
| Potencia (W) | 86.4 | 69.5 | 103.3 | V x I osciloscopio | Medida dentro IC |
| T_max (C) | 21.3 | 16.3 | 26.4 | Termopar tipo K | Medida dentro IC |
| Delta_P (Pa) | 1736 | 1464 | 2008 | Sensor diferencial | Medida dentro IC |

**Validaciones cruzadas:**
- Consistencia longitud de onda: modelo produccion (480 nm), Tangelo (446 nm), CFD (480 nm) — media 469 +/-16 nm
- Consistencia termica: CFD predice T_max = 21.3 C (liquido), T_gas plasma = 333 K (60 C separado)
- Score optimizacion: 0.932 (CFD-validado) vs 0.722 (gemelo digital sin CFD boost)

---

## 8. Plan Estadistico

### Tamano de Muestra
- **n = 3 replicas** minimo por condicion
- Distribucion t-Student (df = 2)
- Nivel de confianza: 95%
- Justificacion: minimo para CI con t-distribution; limitado por precursor y tiempo de reactor

### Tests Estadisticos

| Test | Proposito | Hipotesis | alpha |
|------|-----------|-----------|-------|
| t-test una muestra | Comparar media medida con prediccion gemelo digital | H0: mu_medida = mu_predicho | 0.05 |
| Bland-Altman | Evaluar acuerdo modelo-experimento | Graficar (M+E)/2 vs (M-E). LoA = +/-1.96*sigma | - |
| ANOVA una via | Comparar produccion a 10, 15, 20 mL/min | H0: mu_10 = mu_15 = mu_20 | 0.05 |
| Analisis de potencia | Verificar n=3 suficiente | beta=0.80, alpha=0.05, d=1.5 (grande) | - |

**Resultado analisis de potencia:** n=3 suficiente para efectos grandes (d>1.5); n=6 necesario para medios (d~0.8).

### Procesamiento de Datos
1. Descartar primeros 10 min de cada run (transitorio arranque)
2. Eliminacion outliers: test de Grubbs (alpha=0.05) en triplicados
3. Reportar: media +/- IC 95% usando distribucion t
4. Verificacion normalidad: test Shapiro-Wilk (n<50)

---

## 9. Estado de los Componentes

| Componente | Estado | Accion Necesaria |
|:-----------|:-------|:-----------------|
| **Milirreactor DBD (MC 16ch x 500mm)** | Gemelo digital validado, 1211 mg/h, score 0.932 | Fabricar prototipo segun BOM (Sec 3) |
| **Validacion CFD (2D FV)** | Completada: 3 configs comparadas, PINN R^2>0.95 | Verificar vs experimento (Fase 2) |
| **Plasma frio (non-thermal)** | Modelado: Te=1.5 eV, OH~10^16 cm^-3 | Validar OES + espectro emision (Fase 2.4) |
| **Sistema control + clasificador** | Implementado en `reactor_control.py` | Validar precision >90% (Fase 3.8) |
| **Simulacion Cantera (quimica)** | Ejecutada: GRI-Mech 3.0, 53 especies | OH termico ~0 (correcto), plasma-source OH usado |
| **Simulacion Tangelo (cuantica)** | Ejecutada: 4 zonas, E_activacion por zona | Validar con >24 qubits en GPU |
| **Protocolo experimental** | Completo: 3 fases, 20 pasos, 10 dias | Ejecutar fases secuencialmente |
| **BOM** | 41 items, 6775 EUR estimado | Procurar materiales |

---

## 10. Economia (Reactor Optimizado 16ch x 500mm)

| Concepto | Valor |
|----------|-------|
| Costo reactor (BOM completo) | 6,775 EUR |
| Costo clasificadores paralelos | ~10,000 EUR |
| Total hardware | ~16,775 EUR |
| Produccion bruta | 1211 mg/h = 29.1 g/dia |
| Valor produccion (@30 EUR/g) | ~873 EUR/dia |
| Costo energia (86.4 W reactor, 24h) | ~0.25 EUR/dia |
| **Payback reactor solo** | **~8 dias** |
| **Payback sistema completo** | **~19 dias** |

---

## 11. Comandos Principales

```bash
# Gemelo digital + protocolo experimental (reactor optimizado 16ch)
python experimental_validation.py

# Validacion CFD: 2D FV solver + PINN surrogate
python cfd_validate_reactor.py

# Optimizacion Cantera+Tangelo con quimica de plasma calibrada
python optimize_cantera_reactor.py

# Sistema completo (milirreactor + clasificadores paralelos)
python parallel_classifier.py

# Milirreactor: comparar topologias
python reactor_scaleup.py

# Produccion continua: cascada inline
python continuous_production.py

# Microreactor base
python reactor_design.py --optimize

# Control: ReactorController + ClassifierController
python reactor_control.py

# Simulacion VQE
python qdot_vqe.py
```

### Ejecucion en contenedor (con Cantera + Tangelo)

```bash
podman run --gpus all --security-opt label=disable \
  -v ./:/app:Z -w /app \
  --entrypoint python3 \
  localhost/qdots-cfd:latest \
  experimental_validation.py
```

---

## 12. Archivos del Proyecto

| Archivo | Descripcion |
|---------|-------------|
| `experimental_validation.py` | Gemelo digital + protocolo experimental completo |
| `cfd_validate_reactor.py` | Solver CFD 2D FV + PINN surrogate |
| `optimize_cantera_reactor.py` | Optimizacion Bayesiana + Cantera GRI-Mech 3.0 |
| `reactor_scaleup.py` | Topologias de milirreactor escalado |
| `reactor_design.py` | Diseno parametrico base + materiales |
| `reactor_control.py` | Control PID + clasificador optico por zonas |
| `parallel_classifier.py` | 50 clasificadores en paralelo (numbering-up) |
| `continuous_production.py` | Produccion continua cascada inline |
| `classifier_design.py` | Diseno base clasificador opto-termico |
| `chem_backend/tangelo_interface.py` | Interfaz Tangelo VQE para quimica cuantica |
| `qdot_vqe*.py` | Simulaciones VQE de gaps HOMO-LUMO |
| `optimization_results/cfd_validation_results.json` | Resultados CFD pre-computados (3 configs) |
| `optimization_results/experimental_validation_report.json` | Reporte completo gemelo digital + protocolo |
