/*
 * ELECTRODOS DEL REACTOR DBD
 * Para impresión con material conductor (filamento de cobre/grafeno)
 */

// Parámetros (copiados de reactor_3d.scad)
channel_width = 3.0;
channel_height = 1.0;
channel_length = 250;
wall_thickness = 2.0;
n_turns = 6;
electrode_width = 1.5;
electrode_thickness = 0.1;
dielectric_thickness = 0.8;

$fn = 32;

// Cálculos
straight_length = channel_length / (2 * n_turns);
total_depth = n_turns * (channel_width + wall_thickness * 2) + wall_thickness;
total_height = channel_height + wall_thickness * 2 + dielectric_thickness * 2;

// Electrodo individual
module electrode(length, width, thickness) {
    cube([length, width, thickness]);
}

// Par de electrodos para un tramo
module electrode_pair(length) {
    // Electrodo superior
    translate([0, 0, channel_height + dielectric_thickness])
        electrode(length, electrode_width, electrode_thickness);

    // Electrodo inferior
    translate([0, 0, -dielectric_thickness - electrode_thickness])
        electrode(length, electrode_width, electrode_thickness);
}

// Sistema completo de electrodos
module electrodes_only() {
    for (i = [0:n_turns-1]) {
        y_offset = wall_thickness + (channel_width - electrode_width)/2 +
                  i * (channel_width * 2 + wall_thickness);

        translate([wall_thickness + wall_thickness,
                  y_offset,
                  wall_thickness]) {
            electrode_pair(straight_length - wall_thickness * 2);
        }
    }
}

// Renderizar solo electrodos
electrodes_only();

echo("═══════════════════════════════════════════════════════════════");
echo("  ELECTRODOS - Para impresión con filamento conductor");
echo("═══════════════════════════════════════════════════════════════");
echo(str("  Número de pares: ", n_turns));
echo(str("  Longitud electrodo: ", straight_length - wall_thickness * 2, " mm"));
echo(str("  Ancho: ", electrode_width, " mm"));
echo(str("  Espesor: ", electrode_thickness, " mm"));
echo("═══════════════════════════════════════════════════════════════");
