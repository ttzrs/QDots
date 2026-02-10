#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  MODELO 3D DEL CLASIFICADOR DE LABERINTO - CadQuery
  Genera STL y STEP desde parámetros optimizados
═══════════════════════════════════════════════════════════════════════════════

  Clasificador de escalera con membranas en cascada para separación de QDots
  por tamaño, con cámaras de detección por fotoluminiscencia.

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
    print("⚠ CadQuery no instalado. Instalarlo con: pip install cadquery-ocp")

# ═══════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS (cargados desde JSON o valores por defecto)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    # Etapas
    "n_stages": 5,
    "stage_pitch": 25.0,

    # Canal principal
    "channel_width": 2.0,
    "channel_height": 1.0,
    "wall_thickness": 2.0,

    # Membrana
    "membrane_diameter": 13.0,
    "membrane_seat_depth": 0.5,
    "membrane_clamp_thickness": 1.0,

    # Cámara de colección
    "chamber_width": 10.0,
    "chamber_length": 15.0,
    "chamber_depth": 5.0,

    # Puertos ópticos
    "window_diameter": 8.0,
    "window_thickness": 1.0,
    "window_seat_depth": 0.3,
    "led_diameter": 5.0,
    "led_mount_depth": 3.0,
    "detector_diameter": 5.0,
    "detector_mount_depth": 3.0,

    # Conexiones
    "inlet_diameter": 2.0,
    "outlet_diameter": 1.5,
    "waste_diameter": 2.0,
    "connector_length": 8.0,

    # Tornillos tapa (M3)
    "screw_diameter": 3.0,
    "screw_head_diameter": 5.5,
}


def load_params(json_file: str = "reactor_optimized.json") -> dict:
    """Carga parámetros desde la sección 'classifier' del archivo JSON"""
    params = DEFAULT_PARAMS.copy()

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if 'classifier' in data:
            clf = data['classifier']
            params['n_stages'] = clf.get('n_stages', params['n_stages'])
            params['stage_pitch'] = clf.get('stage_pitch', params['stage_pitch'])
            params['channel_width'] = clf.get('channel_width', params['channel_width'])
            params['channel_height'] = clf.get('channel_height', params['channel_height'])
            params['wall_thickness'] = clf.get('wall_thickness', params['wall_thickness'])
            params['membrane_diameter'] = clf.get('membrane_diameter', params['membrane_diameter'])

            if 'chambers' in clf:
                params['chamber_width'] = clf['chambers'].get('width', params['chamber_width'])
                params['chamber_length'] = clf['chambers'].get('length', params['chamber_length'])
                params['chamber_depth'] = clf['chambers'].get('depth', params['chamber_depth'])

            if 'optics' in clf:
                params['window_diameter'] = clf['optics'].get('window_diameter', params['window_diameter'])
                params['led_diameter'] = clf['optics'].get('led_diameter', params['led_diameter'])
                params['detector_diameter'] = clf['optics'].get('detector_diameter', params['detector_diameter'])

            if 'connections' in clf:
                params['inlet_diameter'] = clf['connections'].get('inlet_diameter', params['inlet_diameter'])
                params['outlet_diameter'] = clf['connections'].get('outlet_diameter', params['outlet_diameter'])
                params['waste_diameter'] = clf['connections'].get('waste_diameter', params['waste_diameter'])

            print(f"✓ Parámetros del clasificador cargados desde {json_file}")
        else:
            print(f"⚠ Sección 'classifier' no encontrada en {json_file}, usando valores por defecto")

    except FileNotFoundError:
        print(f"⚠ Archivo {json_file} no encontrado, usando valores por defecto")

    return params


class ClassifierModel:
    """Genera modelo 3D del clasificador de laberinto"""

    def __init__(self, params: dict):
        self.p = params
        self._calculate_dimensions()

    def _calculate_dimensions(self):
        """Calcula dimensiones derivadas del clasificador"""
        p = self.p

        # Dimensiones totales
        self.total_length = p['n_stages'] * p['stage_pitch'] + 2 * p['wall_thickness'] + p['connector_length'] * 2
        self.total_width = max(
            p['chamber_length'] + 2 * p['wall_thickness'],
            p['membrane_diameter'] + 2 * p['wall_thickness']
        )
        self.total_height = (
            p['wall_thickness'] +           # base
            p['chamber_depth'] +            # cámara de colección
            p['wall_thickness'] +           # separador
            p['channel_height'] +           # canal principal
            p['wall_thickness'] +           # tapa
            p['membrane_clamp_thickness']   # tapa removible
        )

        # Altura del canal principal (desde base)
        self.channel_z = p['wall_thickness'] + p['chamber_depth'] + p['wall_thickness'] + p['channel_height'] / 2
        # Altura de la cámara (desde base)
        self.chamber_z = p['wall_thickness'] + p['chamber_depth'] / 2

        print(f"\n  Dimensiones del clasificador:")
        print(f"    Largo:  {self.total_length:.1f} mm")
        print(f"    Ancho:  {self.total_width:.1f} mm")
        print(f"    Alto:   {self.total_height:.1f} mm")
        print(f"    Etapas: {p['n_stages']}")

    def _stage_x(self, stage_idx: int) -> float:
        """Posición X del centro de una etapa"""
        return self.p['connector_length'] + self.p['wall_thickness'] + \
               stage_idx * self.p['stage_pitch'] + self.p['stage_pitch'] / 2

    def create_body(self) -> 'cq.Workplane':
        """Crea el cuerpo principal del clasificador"""
        body = (
            cq.Workplane("XY")
            .box(self.total_length, self.total_width,
                 self.total_height - self.p['membrane_clamp_thickness'])
            .translate((self.total_length / 2, self.total_width / 2,
                       (self.total_height - self.p['membrane_clamp_thickness']) / 2))
        )
        return body

    def create_main_channel(self) -> 'cq.Workplane':
        """Crea el canal principal de flujo (lineal, de inlet a waste)"""
        p = self.p

        channel_length = p['n_stages'] * p['stage_pitch'] + 2 * p['wall_thickness']
        channel = (
            cq.Workplane("XY")
            .box(channel_length, p['channel_width'], p['channel_height'])
            .translate((
                p['connector_length'] + channel_length / 2,
                self.total_width / 2,
                self.channel_z
            ))
        )
        return channel

    def create_membrane_slot(self, stage_idx: int) -> 'cq.Workplane':
        """Crea el asiento circular de membrana para una etapa"""
        p = self.p
        x = self._stage_x(stage_idx)

        # Asiento de la membrana (cilindro que atraviesa desde canal hasta cámara)
        slot = (
            cq.Workplane("XY")
            .cylinder(
                p['channel_height'] + p['wall_thickness'] + p['membrane_seat_depth'],
                p['membrane_diameter'] / 2 + 0.1  # +0.1mm tolerancia
            )
            .translate((
                x,
                self.total_width / 2,
                self.channel_z - p['channel_height'] / 2 - p['wall_thickness'] / 2
            ))
        )

        # Acceso desde arriba para insertar membrana
        access = (
            cq.Workplane("XY")
            .cylinder(
                p['membrane_clamp_thickness'] + p['wall_thickness'],
                p['membrane_diameter'] / 2 + 0.5  # holgura para inserción
            )
            .translate((
                x,
                self.total_width / 2,
                self.total_height - p['membrane_clamp_thickness']
            ))
        )

        return slot.union(access)

    def create_collection_chamber(self, stage_idx: int) -> 'cq.Workplane':
        """Crea la cámara de colección bajo cada membrana"""
        p = self.p
        x = self._stage_x(stage_idx)

        chamber = (
            cq.Workplane("XY")
            .box(p['chamber_width'], p['chamber_length'], p['chamber_depth'])
            .translate((x, self.total_width / 2, self.chamber_z))
        )
        return chamber

    def create_optical_ports(self, stage_idx: int) -> 'cq.Workplane':
        """
        Crea puertos ópticos para una etapa:
        - Ventana de cuarzo (frontal)
        - LED UV (lateral izquierdo)
        - Fotodetector (lateral derecho, a 90° del LED)
        """
        p = self.p
        x = self._stage_x(stage_idx)

        # Ventana de cuarzo - cara frontal (Y=0)
        window = (
            cq.Workplane("XY")
            .cylinder(p['wall_thickness'] + 1, p['window_diameter'] / 2)
            .rotateAboutCenter((1, 0, 0), 90)
            .translate((x, p['wall_thickness'] / 2, self.chamber_z))
        )

        # Asiento de ventana (rebaje para disco de cuarzo)
        window_seat = (
            cq.Workplane("XY")
            .cylinder(p['window_seat_depth'], p['window_diameter'] / 2 + 0.5)
            .rotateAboutCenter((1, 0, 0), 90)
            .translate((x, p['window_seat_depth'] / 2, self.chamber_z))
        )

        # LED UV - cara lateral izquierda (X negativo respecto al centro)
        led_bore = (
            cq.Workplane("XY")
            .cylinder(p['led_mount_depth'] + p['wall_thickness'], p['led_diameter'] / 2)
            .rotateAboutCenter((0, 1, 0), 90)
            .translate((
                x - p['chamber_width'] / 2 - p['wall_thickness'] / 2,
                self.total_width / 2,
                self.chamber_z
            ))
        )

        # Fotodetector - cara lateral derecha (X positivo, 90° del LED)
        detector_bore = (
            cq.Workplane("XY")
            .cylinder(p['detector_mount_depth'] + p['wall_thickness'], p['detector_diameter'] / 2)
            .rotateAboutCenter((0, 1, 0), 90)
            .translate((
                x + p['chamber_width'] / 2 + p['wall_thickness'] / 2,
                self.total_width / 2,
                self.chamber_z
            ))
        )

        return window.union(window_seat).union(led_bore).union(detector_bore)

    def create_stage_outlet(self, stage_idx: int) -> 'cq.Workplane':
        """Crea el puerto de salida en el fondo de cada cámara"""
        p = self.p
        x = self._stage_x(stage_idx)

        outlet = (
            cq.Workplane("XY")
            .cylinder(p['wall_thickness'] + p['connector_length'], p['outlet_diameter'] / 2)
            .translate((x, self.total_width / 2, -p['connector_length'] / 2))
        )
        return outlet

    def create_screw_holes(self, stage_idx: int) -> 'cq.Workplane':
        """Crea los agujeros de tornillo M3 alrededor de cada membrana"""
        p = self.p
        x = self._stage_x(stage_idx)
        r = p['membrane_diameter'] / 2 + p['wall_thickness'] / 2

        holes = None
        for angle in [0, 90, 180, 270]:
            rad = math.radians(angle)
            hx = x + r * math.cos(rad)
            hy = self.total_width / 2 + r * math.sin(rad)

            hole = (
                cq.Workplane("XY")
                .cylinder(self.total_height, p['screw_diameter'] / 2)
                .translate((hx, hy, self.total_height / 2))
            )

            if holes is None:
                holes = hole
            else:
                holes = holes.union(hole)

        return holes

    def create_inlet_outlet(self) -> 'cq.Workplane':
        """Crea conectores de entrada (inlet) y salida (waste)"""
        p = self.p

        # Inlet - lado izquierdo
        inlet = (
            cq.Workplane("XY")
            .cylinder(p['connector_length'], p['inlet_diameter'] / 2)
            .rotateAboutCenter((0, 1, 0), 90)
            .translate((-p['connector_length'] / 2, self.total_width / 2, self.channel_z))
        )

        # Waste outlet - lado derecho
        waste = (
            cq.Workplane("XY")
            .cylinder(p['connector_length'], p['waste_diameter'] / 2)
            .rotateAboutCenter((0, 1, 0), 90)
            .translate((
                self.total_length + p['connector_length'] / 2,
                self.total_width / 2,
                self.channel_z
            ))
        )

        return inlet.union(waste)

    def create_top_plate(self) -> 'cq.Workplane':
        """Crea la placa superior removible para acceso a membranas"""
        p = self.p

        plate_length = p['n_stages'] * p['stage_pitch'] + 2 * p['wall_thickness']

        plate = (
            cq.Workplane("XY")
            .box(plate_length, self.total_width, p['membrane_clamp_thickness'])
            .translate((
                p['connector_length'] + plate_length / 2,
                self.total_width / 2,
                self.total_height - p['membrane_clamp_thickness'] / 2
            ))
        )

        # Sustraer aberturas de membrana y agujeros de tornillo
        for i in range(p['n_stages']):
            x = self._stage_x(i)

            # Abertura para membrana
            opening = (
                cq.Workplane("XY")
                .cylinder(p['membrane_clamp_thickness'] + 1, p['membrane_diameter'] / 2 + 0.2)
                .translate((x, self.total_width / 2, self.total_height - p['membrane_clamp_thickness'] / 2))
            )
            plate = plate.cut(opening)

            # Agujeros de tornillo
            r = p['membrane_diameter'] / 2 + p['wall_thickness'] / 2
            for angle in [0, 90, 180, 270]:
                rad = math.radians(angle)
                hx = x + r * math.cos(rad)
                hy = self.total_width / 2 + r * math.sin(rad)

                screw = (
                    cq.Workplane("XY")
                    .cylinder(p['membrane_clamp_thickness'] + 1, p['screw_diameter'] / 2)
                    .translate((hx, hy, self.total_height - p['membrane_clamp_thickness'] / 2))
                )
                plate = plate.cut(screw)

        return plate

    def assemble(self) -> 'cq.Workplane':
        """Ensambla el clasificador completo (cuerpo principal)"""
        print("\n→ Ensamblando modelo 3D del clasificador...")
        p = self.p

        # Cuerpo base
        body = self.create_body()
        print("  ✓ Cuerpo principal")

        # Sustraer canal principal
        body = body.cut(self.create_main_channel())
        print("  ✓ Canal principal")

        # Sustraer elementos por etapa
        for i in range(p['n_stages']):
            body = body.cut(self.create_membrane_slot(i))
            body = body.cut(self.create_collection_chamber(i))
            body = body.cut(self.create_optical_ports(i))
            body = body.cut(self.create_stage_outlet(i))
            body = body.cut(self.create_screw_holes(i))
            print(f"  ✓ Etapa {i+1}: membrana + cámara + óptica + salida")

        # Sustraer conectores de entrada/salida
        body = body.cut(self.create_inlet_outlet())
        print("  ✓ Conectores inlet/waste")

        return body

    def export_stl(self, filename: str = "classifier.stl"):
        """Exporta a formato STL"""
        body = self.assemble()
        cq.exporters.export(body, filename)
        print(f"  ✓ Exportado: {filename}")

    def export_step(self, filename: str = "classifier.step"):
        """Exporta a formato STEP"""
        body = self.assemble()
        cq.exporters.export(body, filename)
        print(f"  ✓ Exportado: {filename}")

    def export_all(self, base_name: str = "classifier"):
        """Exporta cuerpo y tapa por separado"""
        # Cuerpo principal
        body = self.assemble()
        body_stl = f"{base_name}_body.stl"
        cq.exporters.export(body, body_stl)
        print(f"  ✓ Cuerpo: {body_stl}")

        body_step = f"{base_name}_body.step"
        cq.exporters.export(body, body_step)
        print(f"  ✓ STEP: {body_step}")

        # Tapa removible
        top = self.create_top_plate()
        top_stl = f"{base_name}_top_plate.stl"
        cq.exporters.export(top, top_stl)
        print(f"  ✓ Tapa: {top_stl}")


def generate_ascii_preview(params: dict):
    """Genera vista previa ASCII del clasificador"""
    p = params
    n = p['n_stages']

    print("\n" + "═" * 70)
    print("  CLASIFICADOR DE LABERINTO - Vista superior")
    print("═" * 70)

    # Vista superior
    stage_w = 7
    total_w = n * stage_w + 4

    print()
    print("  ┌" + "─" * total_w + "┐")
    print("  │ IN", end="")
    for i in range(n):
        print(f"═══╤═══", end="")
    print(" OUT │")
    print("  │   ", end="")
    for i in range(n):
        print(f"   │   ", end="")
    print("     │")
    print("  │   ", end="")
    for i in range(n):
        print(f"  [{i+1}]  ", end="")
    print("     │")
    print("  │   ", end="")
    for i in range(n):
        print(f"   │   ", end="")
    print("     │")
    print("  │   ", end="")
    for i in range(n):
        print(f"   ▼   ", end="")
    print("     │")
    print("  └" + "─" * total_w + "┘")

    # Leyenda de etapas
    pore_sizes = [10000, 5000, 20, 13, 3]
    labels = ["10μm", "5μm", "20nm", "13nm", "3nm"]
    types = ["PC", "PC", "AAO", "AAO", "UF"]

    print()
    print("  Etapas:")
    for i in range(min(n, len(labels))):
        qdot = " ← OBJETIVO" if i == 3 else ""
        print(f"    [{i+1}] Membrana {labels[i]} ({types[i]}){qdot}")

    print("\n" + "═" * 70)
    print("  CLASIFICADOR DE LABERINTO - Corte transversal (Etapa QDot)")
    print("═" * 70)

    print("""
     Tapa removible (M3)
  ┌──●────────────────●──┐
  │  ┌────────────────┐  │  ← Canal principal
  │  │  flujo → ║memb║  │  │
  │  └──────────╫════╫──┘  │
  │             ║    ║      │
  │  ┌──────────╨────╨──┐  │
  │  │                  │  │
  ○──│  Cámara colección│──□  ← LED (○) y detector (□) a 90°
  │  │    [ventana]     │  │
  │  └────────┬─────────┘  │
  └───────────┼────────────┘
              │  salida""")

    print(f"\n  Dimensiones totales:")
    total_l = n * p['stage_pitch'] + 2 * p['wall_thickness'] + 2 * p['connector_length']
    total_w_mm = max(p['chamber_length'] + 2 * p['wall_thickness'],
                     p['membrane_diameter'] + 2 * p['wall_thickness'])
    total_h = (p['wall_thickness'] + p['chamber_depth'] + p['wall_thickness'] +
               p['channel_height'] + p['wall_thickness'] + p['membrane_clamp_thickness'])
    print(f"    {total_l:.1f} x {total_w_mm:.1f} x {total_h:.1f} mm")


# ═══════════════════════════════════════════════════════════════════════════════
#  PROGRAMA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generador 3D del clasificador de laberinto")
    parser.add_argument("--params", type=str, default="reactor_optimized.json",
                       help="Archivo JSON con parámetros")
    parser.add_argument("--export-stl", action="store_true", help="Exportar STL")
    parser.add_argument("--export-step", action="store_true", help="Exportar STEP")
    parser.add_argument("--export-all", action="store_true", help="Exportar todos los formatos")
    parser.add_argument("--preview", action="store_true", help="Solo mostrar vista previa ASCII")
    args = parser.parse_args()

    print("═" * 70)
    print("  GENERADOR 3D - CLASIFICADOR DE LABERINTO PARA QDots")
    print("═" * 70)

    # Cargar parámetros
    params = load_params(args.params)

    # Vista previa ASCII (siempre)
    generate_ascii_preview(params)

    if args.preview:
        print("\n  (Usar --export-stl o --export-step para generar archivos)")
        exit(0)

    if not CADQUERY_AVAILABLE:
        print("\n  ⚠ Para generar modelos 3D, instalar CadQuery:")
        print("    pip install cadquery-ocp")
        print("\n  Alternativa: usar el archivo OpenSCAD (classifier_3d.scad)")
        exit(1)

    # Crear modelo
    model = ClassifierModel(params)

    if args.export_all:
        model.export_all("classifier_maze")
    else:
        if args.export_stl:
            model.export_stl("classifier_maze.stl")
        if args.export_step:
            model.export_step("classifier_maze.step")

    if not (args.export_stl or args.export_step or args.export_all):
        print("\n  Usar --export-stl, --export-step o --export-all para generar archivos")

    print("\n" + "═" * 70)
