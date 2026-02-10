#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  GENERADOR DE MALLA PARAMÉTRICA PARA REACTOR DBD
  Crea blockMeshDict para OpenFOAM desde parámetros optimizados
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import math
from pathlib import Path

# Cargar parámetros
def load_params(json_file: str = "../reactor_optimized.json") -> dict:
    """Carga parámetros del reactor"""
    default = {
        "channel_width": 3.0,      # mm
        "channel_height": 1.0,     # mm
        "channel_length": 250.0,   # mm
        "wall_thickness": 2.0,     # mm
        "n_turns": 6,
        "turn_radius": 4.5,        # mm
    }

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        if 'chamber' in data:
            default.update({
                "channel_width": data['chamber'].get('channel_width', default['channel_width']),
                "channel_height": data['chamber'].get('channel_height', default['channel_height']),
                "channel_length": data['chamber'].get('channel_length', default['channel_length']),
                "n_turns": data['chamber'].get('n_turns', default['n_turns']),
                "turn_radius": data['chamber'].get('turn_radius', default['turn_radius']),
            })
        print(f"✓ Parámetros cargados desde {json_file}")
    except FileNotFoundError:
        print(f"⚠ Usando parámetros por defecto")

    return default


def generate_blockmesh(params: dict, output_file: str = "system/blockMeshDict") -> str:
    """
    Genera blockMeshDict para canal serpentín simplificado.
    Para el primer MVP usamos un canal recto equivalente.
    """
    p = params

    # Convertir mm a m (SI para OpenFOAM)
    w = p['channel_width'] / 1000.0      # ancho del canal
    h = p['channel_height'] / 1000.0     # altura del canal
    L = p['channel_length'] / 1000.0     # longitud total

    # Resolución de malla (celdas)
    nx = int(L / (w/10))  # ~10 celdas por ancho en dirección del flujo
    ny = max(10, int(w / (w/10)))
    nz = max(5, int(h / (h/5)))

    # Grading para capa límite
    grading_y = 1.0
    grading_z = 1.0

    blockmesh = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  11
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "constant/polyMesh";
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// Reactor DBD - Canal equivalente
// Parámetros: W={w*1000:.1f}mm, H={h*1000:.1f}mm, L={L*1000:.1f}mm

scale   1;

vertices
(
    // Cara inferior (z=0)
    (0      0      0)       // 0
    ({L:.6f} 0      0)       // 1
    ({L:.6f} {w:.6f} 0)       // 2
    (0      {w:.6f} 0)       // 3

    // Cara superior (z=h)
    (0      0      {h:.6f})  // 4
    ({L:.6f} 0      {h:.6f})  // 5
    ({L:.6f} {w:.6f} {h:.6f})  // 6
    (0      {w:.6f} {h:.6f})  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading ({grading_y} {grading_y} {grading_z})
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}

    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}

    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4)   // bottom-front
            (3 7 6 2)   // top-back
        );
    }}

    electrode_top
    {{
        type wall;
        faces
        (
            (4 5 6 7)   // top (electrode zone)
        );
    }}

    electrode_bottom
    {{
        type wall;
        faces
        (
            (0 3 2 1)   // bottom (electrode zone)
        );
    }}
);

// ************************************************************************* //
"""

    # Guardar archivo
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(blockmesh)

    print(f"✓ blockMeshDict generado: {output_file}")
    print(f"  Dimensiones: {L*1000:.1f} x {w*1000:.1f} x {h*1000:.1f} mm")
    print(f"  Malla: {nx} x {ny} x {nz} = {nx*ny*nz:,} celdas")

    return blockmesh


def generate_boundary_conditions(params: dict, inlet_velocity: float = 0.01):
    """
    Genera archivos de condiciones de frontera en 0/
    inlet_velocity en m/s (0.01 m/s = 1 cm/s típico para microcanales)
    """
    p = params

    # U (velocidad)
    u_file = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  11
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({inlet_velocity} 0 0);
    }}

    outlet
    {{
        type            zeroGradient;
    }}

    walls
    {{
        type            noSlip;
    }}

    electrode_top
    {{
        type            noSlip;
    }}

    electrode_bottom
    {{
        type            noSlip;
    }}
}}

// ************************************************************************* //
"""

    # p (presión)
    p_file = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  11
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }

    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }

    walls
    {
        type            zeroGradient;
    }

    electrode_top
    {
        type            zeroGradient;
    }

    electrode_bottom
    {
        type            zeroGradient;
    }
}

// ************************************************************************* //
"""

    # Guardar archivos
    Path("0").mkdir(exist_ok=True)

    with open("0/U", 'w') as f:
        f.write(u_file)
    print("✓ 0/U generado")

    with open("0/p", 'w') as f:
        f.write(p_file)
    print("✓ 0/p generado")


def generate_transport_properties():
    """Genera propiedades de transporte para agua"""
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  11
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Agua a 25°C
transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] 1e-06;  // viscosidad cinemática m²/s

// ************************************************************************* //
"""
    Path("constant").mkdir(exist_ok=True)
    with open("constant/transportProperties", 'w') as f:
        f.write(content)
    print("✓ constant/transportProperties generado")


def generate_turbulence_properties():
    """Genera propiedades de turbulencia (laminar para microcanales)"""
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  11
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      momentumTransport;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Flujo laminar en microcanales (Re << 2300)
simulationType  laminar;

// ************************************************************************* //
"""
    with open("constant/momentumTransport", 'w') as f:
        f.write(content)
    print("✓ constant/momentumTransport generado")


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Genera malla OpenFOAM para reactor DBD")
    parser.add_argument("--params", type=str, default="../reactor_optimized.json",
                       help="Archivo JSON con parámetros")
    parser.add_argument("--velocity", type=float, default=0.01,
                       help="Velocidad de entrada (m/s)")
    args = parser.parse_args()

    print("═" * 60)
    print("  GENERADOR DE CASO OPENFOAM - REACTOR DBD")
    print("═" * 60)

    # Cambiar al directorio del caso
    script_dir = Path(__file__).parent
    case_dir = script_dir.parent
    os.chdir(case_dir)
    print(f"\n  Directorio del caso: {case_dir}")

    # Cargar parámetros
    params = load_params(args.params)

    # Generar archivos
    print("\n→ Generando archivos del caso...")
    generate_blockmesh(params, "system/blockMeshDict")
    generate_boundary_conditions(params, args.velocity)
    generate_transport_properties()
    generate_turbulence_properties()

    print("\n═" * 60)
    print("  ✓ Caso OpenFOAM generado")
    print("  Ejecutar: blockMesh && simpleFoam")
    print("═" * 60)
