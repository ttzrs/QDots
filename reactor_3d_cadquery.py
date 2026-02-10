#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  MODELO 3D DEL REACTOR DBD - CadQuery
  Genera STL y STEP desde parámetros optimizados
═══════════════════════════════════════════════════════════════════════════════

  Requisitos: pip install cadquery-ocp
  Uso: python reactor_3d_cadquery.py --export-stl --export-step
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
    "channel_width": 3.0,
    "channel_height": 1.0,
    "channel_length": 250.0,
    "wall_thickness": 2.0,
    "n_turns": 6,
    "turn_radius": 4.5,
    "electrode_width": 1.5,
    "electrode_gap": 1.0,
    "electrode_thickness": 0.1,
    "dielectric_thickness": 0.8,
    "inlet_diameter": 2.0,
    "outlet_diameter": 2.0,
    "gas_inlet_diameter": 1.5,
    "connector_length": 8.0,
}


def load_params(json_file: str = "reactor_optimized.json") -> dict:
    """Carga parámetros desde archivo JSON"""
    params = DEFAULT_PARAMS.copy()

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Mapear desde estructura JSON
        if 'chamber' in data:
            params['channel_width'] = data['chamber'].get('channel_width', params['channel_width'])
            params['channel_height'] = data['chamber'].get('channel_height', params['channel_height'])
            params['channel_length'] = data['chamber'].get('channel_length', params['channel_length'])
            params['wall_thickness'] = data['chamber'].get('wall_thickness', params['wall_thickness'])
            params['n_turns'] = data['chamber'].get('n_turns', params['n_turns'])
            params['turn_radius'] = data['chamber'].get('turn_radius', params['turn_radius'])

        if 'electrodes' in data:
            params['electrode_width'] = data['electrodes'].get('width', params['electrode_width'])
            params['electrode_gap'] = data['electrodes'].get('gap', params['electrode_gap'])
            params['electrode_thickness'] = data['electrodes'].get('thickness', params['electrode_thickness'])

        if 'dielectric' in data:
            params['dielectric_thickness'] = data['dielectric'].get('thickness', params['dielectric_thickness'])

        if 'connections' in data:
            params['inlet_diameter'] = data['connections'].get('inlet_diameter', params['inlet_diameter'])
            params['outlet_diameter'] = data['connections'].get('outlet_diameter', params['outlet_diameter'])
            params['gas_inlet_diameter'] = data['connections'].get('gas_inlet_diameter', params['gas_inlet_diameter'])

        print(f"✓ Parámetros cargados desde {json_file}")

    except FileNotFoundError:
        print(f"⚠ Archivo {json_file} no encontrado, usando valores por defecto")

    return params


class ReactorModel:
    """Genera modelo 3D del reactor DBD"""

    def __init__(self, params: dict):
        self.p = params
        self._calculate_dimensions()

    def _calculate_dimensions(self):
        """Calcula dimensiones derivadas"""
        p = self.p

        # Longitud de cada tramo recto
        self.straight_length = p['channel_length'] / (2 * p['n_turns'])

        # Dimensiones totales del cuerpo
        self.total_width = self.straight_length + 2 * p['wall_thickness']
        self.total_depth = p['n_turns'] * (p['channel_width'] + p['wall_thickness']) + p['wall_thickness']
        self.total_height = p['channel_height'] + 2 * p['wall_thickness'] + 2 * p['dielectric_thickness']

        print(f"\n  Dimensiones calculadas:")
        print(f"    Ancho: {self.total_width:.1f} mm")
        print(f"    Profundidad: {self.total_depth:.1f} mm")
        print(f"    Altura: {self.total_height:.1f} mm")

    def create_body(self) -> 'cq.Workplane':
        """Crea el cuerpo principal del reactor"""
        p = self.p

        # Bloque base
        body = (
            cq.Workplane("XY")
            .box(self.total_width, self.total_depth, self.total_height)
            .translate((self.total_width/2, self.total_depth/2, self.total_height/2))
        )

        return body

    def create_channel(self) -> 'cq.Workplane':
        """Crea el canal serpentín (para sustracción)"""
        p = self.p

        # Crear perfil del canal
        channel_profile = (
            cq.Workplane("XY")
            .rect(p['channel_width'], p['channel_height'])
        )

        # Path del serpentín
        points = []
        z_channel = p['wall_thickness'] + p['dielectric_thickness'] + p['channel_height']/2

        for i in range(p['n_turns']):
            y_base = p['wall_thickness'] + p['channel_width']/2 + i * (p['channel_width'] + p['wall_thickness'])

            # Inicio del tramo
            x_start = p['wall_thickness'] + p['channel_width']/2
            x_end = p['wall_thickness'] + self.straight_length - p['channel_width']/2

            if i % 2 == 0:
                # Tramo hacia la derecha
                points.append((x_start, y_base, z_channel))
                points.append((x_end, y_base, z_channel))
            else:
                # Tramo hacia la izquierda
                points.append((x_end, y_base, z_channel))
                points.append((x_start, y_base, z_channel))

            # Conexión al siguiente nivel (excepto en el último)
            if i < p['n_turns'] - 1:
                y_next = y_base + p['channel_width']/2 + p['wall_thickness']/2
                if i % 2 == 0:
                    points.append((x_end, y_next, z_channel))
                else:
                    points.append((x_start, y_next, z_channel))

        # Crear canal como serie de cubos (simplificado)
        channel = cq.Workplane("XY")

        for i in range(p['n_turns']):
            y_pos = p['wall_thickness'] + i * (p['channel_width'] + p['wall_thickness'])

            channel = (
                channel
                .union(
                    cq.Workplane("XY")
                    .box(self.straight_length - p['wall_thickness'],
                         p['channel_width'],
                         p['channel_height'])
                    .translate((
                        p['wall_thickness'] + self.straight_length/2,
                        y_pos + p['channel_width']/2,
                        p['wall_thickness'] + p['dielectric_thickness'] + p['channel_height']/2
                    ))
                )
            )

            # Conexión vertical
            if i < p['n_turns'] - 1:
                x_conn = self.straight_length if i % 2 == 0 else p['wall_thickness'] * 2
                channel = (
                    channel
                    .union(
                        cq.Workplane("XY")
                        .box(p['channel_width'],
                             p['wall_thickness'] + p['channel_width'],
                             p['channel_height'])
                        .translate((
                            x_conn,
                            y_pos + p['channel_width'] + p['wall_thickness']/2,
                            p['wall_thickness'] + p['dielectric_thickness'] + p['channel_height']/2
                        ))
                    )
                )

        return channel

    def create_inlet_outlet(self) -> 'cq.Workplane':
        """Crea los conectores de entrada y salida"""
        p = self.p

        # Posiciones
        inlet_x = p['wall_thickness'] + self.straight_length/2
        inlet_y = p['wall_thickness'] + p['channel_width']/2

        outlet_x = inlet_x
        outlet_y = self.total_depth - p['wall_thickness'] - p['channel_width']/2

        # Cilindros para entrada y salida
        inlet = (
            cq.Workplane("XY")
            .cylinder(p['connector_length'], p['inlet_diameter']/2)
            .translate((inlet_x, inlet_y, -p['connector_length']/2))
        )

        outlet = (
            cq.Workplane("XY")
            .cylinder(p['connector_length'], p['outlet_diameter']/2)
            .translate((outlet_x, outlet_y, -p['connector_length']/2))
        )

        return inlet.union(outlet)

    def create_electrode(self, length: float) -> 'cq.Workplane':
        """Crea un electrodo"""
        p = self.p
        return (
            cq.Workplane("XY")
            .box(length, p['electrode_width'], p['electrode_thickness'])
        )

    def create_electrodes(self) -> 'cq.Workplane':
        """Crea el sistema de electrodos"""
        p = self.p
        electrodes = None

        electrode_length = self.straight_length - 2 * p['wall_thickness']

        for i in range(p['n_turns']):
            y_pos = p['wall_thickness'] + (p['channel_width'] - p['electrode_width'])/2 + \
                   i * (p['channel_width'] + p['wall_thickness'])

            x_pos = p['wall_thickness'] * 2 + electrode_length/2

            # Electrodo superior
            z_top = p['wall_thickness'] + p['dielectric_thickness'] + p['channel_height'] + \
                   p['dielectric_thickness'] + p['electrode_thickness']/2

            electrode_top = (
                self.create_electrode(electrode_length)
                .translate((x_pos, y_pos + p['electrode_width']/2, z_top))
            )

            # Electrodo inferior
            z_bottom = p['wall_thickness'] - p['electrode_thickness']/2

            electrode_bottom = (
                self.create_electrode(electrode_length)
                .translate((x_pos, y_pos + p['electrode_width']/2, z_bottom))
            )

            if electrodes is None:
                electrodes = electrode_top.union(electrode_bottom)
            else:
                electrodes = electrodes.union(electrode_top).union(electrode_bottom)

        return electrodes

    def assemble(self) -> 'cq.Workplane':
        """Ensambla el reactor completo"""
        print("\n→ Ensamblando modelo 3D...")

        # Cuerpo con canal sustraído
        body = self.create_body()
        channel = self.create_channel()
        connectors = self.create_inlet_outlet()

        # Sustracción del canal del cuerpo
        reactor = body.cut(channel)

        # Sustracción de orificios para conectores
        reactor = reactor.cut(connectors)

        print("  ✓ Cuerpo principal creado")

        # Los electrodos se exportan por separado
        # (en la práctica se insertan después de imprimir)

        return reactor

    def export_stl(self, filename: str = "reactor.stl"):
        """Exporta a formato STL"""
        reactor = self.assemble()
        cq.exporters.export(reactor, filename)
        print(f"  ✓ Exportado: {filename}")

    def export_step(self, filename: str = "reactor.step"):
        """Exporta a formato STEP"""
        reactor = self.assemble()
        cq.exporters.export(reactor, filename)
        print(f"  ✓ Exportado: {filename}")

    def export_all(self, base_name: str = "reactor"):
        """Exporta todos los formatos"""
        reactor = self.assemble()

        # STL para impresión 3D
        stl_file = f"{base_name}.stl"
        cq.exporters.export(reactor, stl_file)
        print(f"  ✓ STL: {stl_file}")

        # STEP para CAD
        step_file = f"{base_name}.step"
        cq.exporters.export(reactor, step_file)
        print(f"  ✓ STEP: {step_file}")

        # Electrodos por separado
        electrodes = self.create_electrodes()
        electrodes_file = f"{base_name}_electrodes.stl"
        cq.exporters.export(electrodes, electrodes_file)
        print(f"  ✓ Electrodos: {electrodes_file}")


def generate_ascii_preview(params: dict):
    """Genera una vista previa ASCII del reactor"""
    p = params
    straight = p['channel_length'] / (2 * p['n_turns'])

    print("\n" + "═" * 60)
    print("  VISTA PREVIA DEL REACTOR (Vista superior)")
    print("═" * 60)
    print()

    # Generar vista superior simplificada
    width_chars = 40
    depth_chars = int(p['n_turns'] * 3)

    # Borde superior
    print("  ┌" + "─" * width_chars + "┐")

    for i in range(depth_chars):
        turn = i // 3
        pos_in_turn = i % 3

        if pos_in_turn == 0:
            # Inicio de tramo
            if turn % 2 == 0:
                line = "│" + "═" * (width_chars - 2) + "→│"
            else:
                line = "│←" + "═" * (width_chars - 2) + "│"
        elif pos_in_turn == 1:
            # Conexión
            if turn < p['n_turns'] - 1:
                if turn % 2 == 0:
                    line = "│" + " " * (width_chars - 2) + "↓│"
                else:
                    line = "│↓" + " " * (width_chars - 2) + "│"
            else:
                line = "│" + " " * width_chars + "│"
        else:
            line = "│" + " " * width_chars + "│"

        print("  " + line)

    # Borde inferior
    print("  └" + "─" * width_chars + "┘")

    # Leyenda
    print()
    print("  ═══ Canal de flujo")
    print("  → Dirección del líquido")
    print("  ↓ Conexión entre tramos")
    print()
    print(f"  Dimensiones: {straight + 2*p['wall_thickness']:.1f} x "
          f"{p['n_turns'] * (p['channel_width'] + p['wall_thickness']) + p['wall_thickness']:.1f} x "
          f"{p['channel_height'] + 2*p['wall_thickness'] + 2*p['dielectric_thickness']:.1f} mm")


# ═══════════════════════════════════════════════════════════════════════════════
#  PROGRAMA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generador de modelo 3D del reactor DBD")
    parser.add_argument("--params", type=str, default="reactor_optimized.json",
                       help="Archivo JSON con parámetros")
    parser.add_argument("--export-stl", action="store_true", help="Exportar STL")
    parser.add_argument("--export-step", action="store_true", help="Exportar STEP")
    parser.add_argument("--export-all", action="store_true", help="Exportar todos los formatos")
    parser.add_argument("--preview", action="store_true", help="Solo mostrar vista previa ASCII")
    args = parser.parse_args()

    print("═" * 60)
    print("  GENERADOR DE MODELO 3D - REACTOR DBD")
    print("═" * 60)

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
        print("\n  Alternativa: usar el archivo OpenSCAD (reactor_3d.scad)")
        exit(1)

    # Crear modelo
    model = ReactorModel(params)

    if args.export_all:
        model.export_all("reactor_dbd")
    else:
        if args.export_stl:
            model.export_stl("reactor_dbd.stl")
        if args.export_step:
            model.export_step("reactor_dbd.step")

    if not (args.export_stl or args.export_step or args.export_all):
        print("\n  Usar --export-stl, --export-step o --export-all para generar archivos")

    print("\n═" * 60)
