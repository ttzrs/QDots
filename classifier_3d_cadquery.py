#!/usr/bin/env python3
"""
===============================================================================
  MODELO 3D DEL CLASIFICADOR POR FLOTABILIDAD OPTICA - CadQuery
  Genera STL y STEP desde parametros optimizados
===============================================================================

  Clasificador con zonas de excitacion optica para separacion selectiva
  de QDots. Cada zona tiene:
    - Array de LEDs en la parte superior
    - Puerto de coleccion superior (QDots excitados que flotan)
    - Puerto de coleccion inferior (debris que sedimenta)
    - Barreras verticales (malla de poro grande) entre zonas

  Requisitos: pip install cadquery-ocp
  Uso: python classifier_3d_cadquery.py --export-stl --export-step
       python classifier_3d_cadquery.py --preview
"""

import json
import math
from pathlib import Path

try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    print("! CadQuery no instalado. Instalarlo con: pip install cadquery-ocp")

# ===============================================================================
#  PARAMETROS (cargados desde JSON o valores por defecto)
# ===============================================================================

DEFAULT_PARAMS = {
    # Zonas
    "n_zones": 3,
    "zone_length": 40.0,         # mm - longitud por zona (eje X)
    "zone_height": 15.0,         # mm - altura (eje Z, critica para separacion)
    "zone_width": 10.0,          # mm - profundidad (eje Y)

    # Canal
    "channel_height": 2.0,       # mm - canal de entrada horizontal
    "wall_thickness": 2.0,       # mm - espesor de paredes

    # Barreras
    "barrier_thickness": 1.0,    # mm - espesor de la malla/barrera
    "barrier_pore_um": 50.0,     # um - poro grande (sin dP)

    # LEDs
    "led_diameter": 5.0,         # mm - diametro del LED
    "led_mount_depth": 3.0,      # mm - profundidad del alojamiento
    "led_count_per_zone": 4,     # LEDs por zona
    "led_spacing": 8.0,          # mm - distancia entre LEDs

    # Puertos de coleccion
    "top_port_diameter": 2.0,    # mm - coleccion QDots (arriba)
    "bottom_port_diameter": 2.0, # mm - coleccion debris (abajo)

    # Conexiones
    "inlet_diameter": 2.0,       # mm
    "outlet_diameter": 2.0,      # mm
    "waste_diameter": 2.0,       # mm
    "connector_length": 8.0,     # mm

    # Ventanas laterales (observacion)
    "window_diameter": 8.0,      # mm
    "window_thickness": 1.0,     # mm
    "has_windows": True,

    # Tornillos tapa (M3)
    "screw_diameter": 3.0,
    "screw_head_diameter": 5.5,

    # Optothermal mode parameters
    "excitation_mode": "led",
    "laser_power_mw": 500,
    "beam_waist_um": 10.0,
    "substrate_type": "gold_film",
    "channel_width_um": 200,
    "channel_depth_um": 100,
}


def load_params(json_file: str = "reactor_optimized.json") -> dict:
    """Carga parametros desde la seccion 'classifier' del archivo JSON"""
    params = DEFAULT_PARAMS.copy()

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if 'classifier' in data:
            clf = data['classifier']
            params['n_zones'] = clf.get('n_zones', params['n_zones'])
            params['zone_length'] = clf.get('zone_length', params['zone_length'])
            params['zone_height'] = clf.get('zone_height', params['zone_height'])
            params['zone_width'] = clf.get('zone_width', params['zone_width'])
            params['channel_height'] = clf.get('channel_height', params['channel_height'])
            params['wall_thickness'] = clf.get('wall_thickness', params['wall_thickness'])

            if 'barriers' in clf:
                params['barrier_pore_um'] = clf['barriers'].get(
                    'pore_diameter_um', params['barrier_pore_um'])

            if 'collection_ports' in clf:
                params['top_port_diameter'] = clf['collection_ports'].get(
                    'top_diameter_mm', params['top_port_diameter'])
                params['bottom_port_diameter'] = clf['collection_ports'].get(
                    'bottom_diameter_mm', params['bottom_port_diameter'])

            if 'connections' in clf:
                params['inlet_diameter'] = clf['connections'].get(
                    'inlet_diameter', params['inlet_diameter'])
                params['outlet_diameter'] = clf['connections'].get(
                    'outlet_diameter', params['outlet_diameter'])
                params['waste_diameter'] = clf['connections'].get(
                    'waste_diameter', params['waste_diameter'])

            # Optothermal mode parameters
            if 'excitation_mode' in clf:
                params['excitation_mode'] = clf['excitation_mode']
            if 'laser' in clf:
                params['laser_power_mw'] = clf['laser'].get('power_mw', 500)
                params['beam_waist_um'] = clf['laser'].get('beam_waist_um', 10.0)
            if 'optothermal' in clf:
                params['substrate_type'] = clf['optothermal'].get('substrate_type', 'gold_film')
            if 'microchannel' in clf:
                params['channel_width_um'] = clf['microchannel'].get('width_um', 200)
                params['channel_depth_um'] = clf['microchannel'].get('depth_um', 100)

            print(f"+ Parametros del clasificador cargados desde {json_file}")
        else:
            print(f"! Seccion 'classifier' no encontrada en {json_file}, usando valores por defecto")

    except FileNotFoundError:
        print(f"! Archivo {json_file} no encontrado, usando valores por defecto")

    return params


class ClassifierModel:
    """Genera modelo 3D del clasificador por flotabilidad optica"""

    def __init__(self, params: dict):
        self.p = params
        self.mode = params.get('excitation_mode', 'led')
        self._calculate_dimensions()

    def _calculate_dimensions(self):
        """Calcula dimensiones derivadas del clasificador"""
        p = self.p

        # Longitud total del cuerpo
        self.body_length = (p['n_zones'] * p['zone_length'] +
                            (p['n_zones'] - 1) * p['barrier_thickness'] +
                            2 * p['wall_thickness'])
        self.total_length = self.body_length + p['connector_length'] * 2

        # Ancho total
        self.total_width = p['zone_width'] + 2 * p['wall_thickness']

        # Altura total: base + zona + techo + espacio LEDs
        self.led_space = p['led_mount_depth'] + 2.0  # espacio para LEDs
        self.total_height = (p['wall_thickness'] +   # base
                             p['zone_height'] +      # zona
                             p['wall_thickness'] +   # techo
                             self.led_space)          # LEDs

        # Alturas de referencia
        self.zone_base_z = p['wall_thickness']
        self.zone_top_z = p['wall_thickness'] + p['zone_height']
        self.led_base_z = self.zone_top_z + p['wall_thickness']

        # Centro Y
        self.center_y = self.total_width / 2.0

        print(f"\n  Dimensiones del clasificador:")
        print(f"    Largo:  {self.total_length:.1f} mm")
        print(f"    Ancho:  {self.total_width:.1f} mm")
        print(f"    Alto:   {self.total_height:.1f} mm")
        print(f"    Zonas:  {p['n_zones']}")

    def _zone_x_center(self, zone_idx: int) -> float:
        """Posicion X del centro de una zona"""
        p = self.p
        return (p['connector_length'] + p['wall_thickness'] +
                zone_idx * (p['zone_length'] + p['barrier_thickness']) +
                p['zone_length'] / 2)

    def _zone_x_start(self, zone_idx: int) -> float:
        """Posicion X del inicio de una zona"""
        p = self.p
        return (p['connector_length'] + p['wall_thickness'] +
                zone_idx * (p['zone_length'] + p['barrier_thickness']))

    def create_body(self) -> 'cq.Workplane':
        """Crea el cuerpo principal del clasificador"""
        body = (
            cq.Workplane("XY")
            .box(self.total_length, self.total_width, self.total_height)
            .translate((self.total_length / 2, self.total_width / 2,
                        self.total_height / 2))
        )
        return body

    def create_zone_cavity(self, zone_idx: int) -> 'cq.Workplane':
        """Crea la cavidad rectangular de una zona"""
        p = self.p
        x_start = self._zone_x_start(zone_idx)

        cavity = (
            cq.Workplane("XY")
            .box(p['zone_length'], p['zone_width'], p['zone_height'])
            .translate((x_start + p['zone_length'] / 2,
                        self.center_y,
                        self.zone_base_z + p['zone_height'] / 2))
        )
        return cavity

    def create_barrier_slot(self, barrier_idx: int) -> 'cq.Workplane':
        """Crea ranura para insertar malla-barrera entre zonas"""
        p = self.p
        # La barrera va entre zona barrier_idx y barrier_idx+1
        x = (p['connector_length'] + p['wall_thickness'] +
             (barrier_idx + 1) * p['zone_length'] +
             barrier_idx * p['barrier_thickness'] +
             p['barrier_thickness'] / 2)

        slot = (
            cq.Workplane("XY")
            .box(p['barrier_thickness'] + 0.2,  # +0.2 tolerancia
                 p['zone_width'] - 1.0,  # ligeramente menor para asentar
                 p['zone_height'])
            .translate((x, self.center_y,
                        self.zone_base_z + p['zone_height'] / 2))
        )
        return slot

    def create_led_bores(self, zone_idx: int) -> 'cq.Workplane':
        """Crea los alojamientos de LEDs en la parte superior de una zona"""
        p = self.p
        x_center = self._zone_x_center(zone_idx)
        n_leds = p['led_count_per_zone']

        bores = None
        for i in range(n_leds):
            # Distribuir LEDs a lo largo del eje X
            offset_x = (i - (n_leds - 1) / 2.0) * p['led_spacing']
            led_x = x_center + offset_x

            # Solo si el LED cae dentro de la zona
            x_start = self._zone_x_start(zone_idx)
            if led_x < x_start + 3.0 or led_x > x_start + p['zone_length'] - 3.0:
                continue

            bore = (
                cq.Workplane("XY")
                .cylinder(p['led_mount_depth'] + p['wall_thickness'] + 1,
                          p['led_diameter'] / 2)
                .translate((led_x, self.center_y,
                            self.total_height - p['led_mount_depth'] / 2))
            )

            if bores is None:
                bores = bore
            else:
                bores = bores.union(bore)

        return bores

    def create_top_collection_port(self, zone_idx: int) -> 'cq.Workplane':
        """Crea puerto de coleccion superior (para QDots excitados)"""
        p = self.p
        x_center = self._zone_x_center(zone_idx)

        # Puerto vertical desde el techo de la zona hacia arriba
        port = (
            cq.Workplane("XY")
            .cylinder(p['wall_thickness'] + self.led_space + 1,
                      p['top_port_diameter'] / 2)
            .translate((x_center - p['zone_length'] / 4,  # desplazado del centro
                        self.center_y,
                        self.zone_top_z + (p['wall_thickness'] + self.led_space) / 2))
        )
        return port

    def create_bottom_collection_port(self, zone_idx: int) -> 'cq.Workplane':
        """Crea puerto de coleccion inferior (para debris)"""
        p = self.p
        x_center = self._zone_x_center(zone_idx)

        # Puerto vertical desde la base de la zona hacia abajo
        port = (
            cq.Workplane("XY")
            .cylinder(p['wall_thickness'] + p['connector_length'],
                      p['bottom_port_diameter'] / 2)
            .translate((x_center + p['zone_length'] / 4,  # desplazado del centro
                        self.center_y,
                        -p['connector_length'] / 2))
        )
        return port

    def create_observation_window(self, zone_idx: int) -> 'cq.Workplane':
        """Crea ventana lateral de observacion"""
        p = self.p
        if not p.get('has_windows', True):
            return None

        x_center = self._zone_x_center(zone_idx)
        zone_center_z = self.zone_base_z + p['zone_height'] / 2

        # Ventana en la cara frontal (Y=0)
        window = (
            cq.Workplane("XY")
            .cylinder(p['wall_thickness'] + 1, p['window_diameter'] / 2)
            .rotateAboutCenter((1, 0, 0), 90)
            .translate((x_center, p['wall_thickness'] / 2, zone_center_z))
        )
        return window

    def create_fiber_port(self, zone_idx):
        """Crea puerto para fibra optica (modo optotermico, reemplaza LEDs)"""
        p = self.p
        x_center = self._zone_x_center(zone_idx)
        # Single fiber port instead of LED array
        port = (
            cq.Workplane("XY")
            .cylinder(p['wall_thickness'] + self.led_space + 1, 1.0)  # 1mm radius for fiber
            .translate((x_center, self.center_y,
                        self.total_height - (p['wall_thickness'] + self.led_space) / 2))
        )
        return port

    def create_gold_substrate_indicator(self, zone_idx):
        """Representa la capa de sustrato dorado en la pared inferior de la zona"""
        p = self.p
        x_start = self._zone_x_start(zone_idx)
        # Thin layer at the bottom of the zone
        substrate = (
            cq.Workplane("XY")
            .box(p['zone_length'] - 1.0, p['zone_width'] - 1.0, 0.1)  # 100nm -> 0.1mm for visibility
            .translate((x_start + p['zone_length'] / 2,
                        self.center_y,
                        self.zone_base_z + 0.05))
        )
        return substrate

    def create_inlet_outlet(self) -> 'cq.Workplane':
        """Crea conectores de entrada (inlet) y salida (waste)"""
        p = self.p
        zone_center_z = self.zone_base_z + p['zone_height'] / 2

        # Inlet - lado izquierdo, a media altura de la zona
        inlet = (
            cq.Workplane("XY")
            .cylinder(p['connector_length'] + p['wall_thickness'],
                      p['inlet_diameter'] / 2)
            .rotateAboutCenter((0, 1, 0), 90)
            .translate((-p['connector_length'] / 2, self.center_y, zone_center_z))
        )

        # Waste outlet - lado derecho
        waste = (
            cq.Workplane("XY")
            .cylinder(p['connector_length'] + p['wall_thickness'],
                      p['waste_diameter'] / 2)
            .rotateAboutCenter((0, 1, 0), 90)
            .translate((self.total_length + p['connector_length'] / 2,
                        self.center_y, zone_center_z))
        )

        return inlet.union(waste)

    def assemble(self) -> 'cq.Workplane':
        """Ensambla el clasificador completo"""
        print("\n-> Ensamblando modelo 3D del clasificador...")
        p = self.p

        # Cuerpo base
        body = self.create_body()
        print("  + Cuerpo principal")

        # Sustraer cavidades de zona
        for i in range(p['n_zones']):
            body = body.cut(self.create_zone_cavity(i))
            print(f"  + Zona {i+1}: cavidad")

        # Sustraer ranuras de barrera
        for i in range(p['n_zones'] - 1):
            body = body.cut(self.create_barrier_slot(i))
            print(f"  + Barrera {i+1}: ranura para malla")

        # Sustraer elementos por zona
        for i in range(p['n_zones']):
            if self.mode == 'optothermal':
                fiber = self.create_fiber_port(i)
                if fiber is not None:
                    body = body.cut(fiber)
            else:
                bores = self.create_led_bores(i)
                if bores is not None:
                    body = body.cut(bores)
            body = body.cut(self.create_top_collection_port(i))
            body = body.cut(self.create_bottom_collection_port(i))
            window = self.create_observation_window(i)
            if window is not None:
                body = body.cut(window)
            mode_str = "fibra optica + Au" if self.mode == 'optothermal' else "LEDs"
            print(f"  + Zona {i+1}: {mode_str} + puertos + ventana")

        # Sustraer conectores de entrada/salida
        body = body.cut(self.create_inlet_outlet())
        print("  + Conectores inlet/waste")

        return body

    def export_stl(self, filename: str = "classifier.stl"):
        """Exporta a formato STL"""
        body = self.assemble()
        cq.exporters.export(body, filename)
        print(f"  + Exportado: {filename}")

    def export_step(self, filename: str = "classifier.step"):
        """Exporta a formato STEP"""
        body = self.assemble()
        cq.exporters.export(body, filename)
        print(f"  + Exportado: {filename}")

    def export_all(self, base_name: str = "classifier"):
        """Exporta cuerpo en STL y STEP"""
        body = self.assemble()

        body_stl = f"{base_name}_body.stl"
        cq.exporters.export(body, body_stl)
        print(f"  + Cuerpo STL: {body_stl}")

        body_step = f"{base_name}_body.step"
        cq.exporters.export(body, body_step)
        print(f"  + Cuerpo STEP: {body_step}")


def generate_ascii_preview(params: dict):
    """Genera vista previa ASCII del clasificador"""
    p = params
    n = p['n_zones']

    print("\n" + "=" * 70)
    print("  CLASIFICADOR POR FLOTABILIDAD OPTICA - Vista superior")
    print("=" * 70)

    # Leyenda de zonas
    zone_info = [
        {"led": "520nm (verde)", "qdot": "3.5-5.0 nm", "emission": "rojo"},
        {"led": "405nm (UV-azul)", "qdot": "2.5-3.5 nm", "emission": "azul-verde"},
        {"led": "365nm (UV)", "qdot": "1.5-2.5 nm", "emission": "UV-violeta"},
    ]

    # Vista superior
    zone_w = 20
    total_w = n * zone_w + 8

    print()
    # LEDs arriba
    led_line = "         "
    for i in range(min(n, len(zone_info))):
        led_line += f"  LED {zone_info[i]['led']:^14}"
    print(led_line)

    arrow_line = "         "
    for _ in range(n):
        arrow_line += "      vvvvvv        "
    print(arrow_line)

    # Top border
    print("  +" + ("=" * zone_w + "+") * n)

    # Top collection
    top_line = "  |"
    for i in range(n):
        top_line += f"  ^ QDots {'RVU'[i]}       |"
    top_line += "  <- Coleccion ARRIBA"
    print(top_line)

    # Flow
    flow_line = "IN->"
    for i in range(n):
        flow_line += f"    Zona {i+1}          "
        if i < n - 1:
            flow_line += "|"
    flow_line += "-> WASTE"
    print(flow_line)

    # Bottom collection
    bot_line = "  |"
    for i in range(n):
        bot_line += f"  v debris          |"
    bot_line += "  <- Coleccion ABAJO"
    print(bot_line)

    # Bottom border
    print("  +" + ("=" * zone_w + "+") * n)

    print(f"\n  Barreras entre zonas: malla {p['barrier_pore_um']:.0f} um (sin caida de presion)")

    # Zonas detalle
    print("\n  Zonas:")
    for i in range(min(n, len(zone_info))):
        zi = zone_info[i]
        print(f"    [Zona {i+1}] LED {zi['led']} -> excita QDots {zi['qdot']} -> emision {zi['emission']}")

    print("\n" + "=" * 70)
    print("  CLASIFICADOR - Corte transversal (una zona)")
    print("=" * 70)

    print("""
      LED  LED  LED  LED         <- Array de LEDs (arriba)
       v    v    v    v
  +----|----|----|----|-----------+
  |  [=====Puerto superior=====] |  <- Coleccion QDots (arriba)
  |                               |
  |    ^    ^    ^    ^           |  <- QDots excitados SUBEN
  |    |    |    |    |           |
  |    :    :    :    :           |     (fuerza fototermica +
  |    :    :    :    :           |      radiacion optica)
  |    |    |    |    |           |
  |    v    v    v    v           |  <- Debris NO excitado BAJA
  |                               |     (sedimentacion Stokes)
  |  [=====Puerto inferior======] |  <- Coleccion debris (abajo)
  +-------------------------------+
     Ventana lateral (observacion)""")

    print(f"\n  Dimensiones totales:")
    total_l = (n * p['zone_length'] + (n - 1) * p.get('barrier_thickness', 1.0) +
               2 * p['wall_thickness'] + 2 * p['connector_length'])
    total_w_mm = p['zone_width'] + 2 * p['wall_thickness']
    led_space = p.get('led_mount_depth', 3.0) + 2.0
    total_h = p['wall_thickness'] + p['zone_height'] + p['wall_thickness'] + led_space
    print(f"    {total_l:.1f} x {total_w_mm:.1f} x {total_h:.1f} mm")
    print(f"    Zona: {p['zone_length']:.0f} x {p['zone_width']:.0f} x {p['zone_height']:.0f} mm")

    mode = params.get('excitation_mode', 'led')
    if mode == 'optothermal':
        print("\n" + "=" * 70)
        print("  CLASIFICADOR OPTO-TERMICO - Corte transversal (una zona)")
        print("=" * 70)
        print("""
      Fibra optica (laser)           <- Laser focalizado (arriba)
            v
  +---------|---------------------+
  |  [=====Puerto superior=====] |  <- Coleccion QDots (arriba)
  |                               |
  |    <--- termoforesis --->     |  <- QDots migran por gradiente termico
  |                               |
  |    ~~~~~~~~~~~~~~~~~~~~~~~~~  |  <- Gradiente de temperatura
  |  [###Sustrato Au (50nm)####]  |  <- Pelicula de oro (calentamiento)
  |  [=====Puerto inferior======] |  <- Coleccion debris (abajo)
  +-------------------------------+
     Ventana lateral (observacion)

  Principio: Laser calienta Au -> grad_T -> termoforesis (Soret)
  Pe ~ 1-10 -> separacion VIABLE para CQDs 2-5nm""")


# ===============================================================================
#  PROGRAMA PRINCIPAL
# ===============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generador 3D del clasificador por flotabilidad optica")
    parser.add_argument("--params", type=str, default="reactor_optimized.json",
                        help="Archivo JSON con parametros")
    parser.add_argument("--mode", type=str, default="led",
                        choices=["led", "optothermal"],
                        help="Excitation mode: led or optothermal")
    parser.add_argument("--export-stl", action="store_true", help="Exportar STL")
    parser.add_argument("--export-step", action="store_true", help="Exportar STEP")
    parser.add_argument("--export-all", action="store_true",
                        help="Exportar todos los formatos")
    parser.add_argument("--preview", action="store_true",
                        help="Solo mostrar vista previa ASCII")
    args = parser.parse_args()

    print("=" * 70)
    print("  GENERADOR 3D - CLASIFICADOR POR FLOTABILIDAD OPTICA")
    print("=" * 70)

    # Cargar parametros
    params = load_params(args.params)

    # Override mode from CLI if specified
    if args.mode:
        params['excitation_mode'] = args.mode

    # Vista previa ASCII (siempre)
    generate_ascii_preview(params)

    if args.preview:
        print("\n  (Usar --export-stl o --export-step para generar archivos)")
        exit(0)

    if not CADQUERY_AVAILABLE:
        print("\n  ! Para generar modelos 3D, instalar CadQuery:")
        print("    pip install cadquery-ocp")
        print("\n  Alternativa: usar el archivo OpenSCAD (classifier_3d.scad)")
        exit(1)

    # Crear modelo
    model = ClassifierModel(params)

    if args.export_all:
        model.export_all("classifier_optical")
    else:
        if args.export_stl:
            model.export_stl("classifier_optical.stl")
        if args.export_step:
            model.export_step("classifier_optical.step")

    if not (args.export_stl or args.export_step or args.export_all):
        print("\n  Usar --export-stl, --export-step o --export-all para generar archivos")

    print("\n" + "=" * 70)
