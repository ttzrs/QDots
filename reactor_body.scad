/*
 * CUERPO DEL REACTOR DBD (sin electrodos ni accesorios)
 * Para impresión con resina de alta temperatura
 */

// Parámetros (copiados de reactor_3d.scad)
channel_width = 3.0;
channel_height = 1.0;
channel_length = 250;
wall_thickness = 2.0;
n_turns = 6;
turn_radius = 4.5;
dielectric_thickness = 0.8;
inlet_diameter = 2.0;
outlet_diameter = 2.0;
gas_inlet_diameter = 1.5;

$fn = 32;

// Cálculos
straight_length = channel_length / (2 * n_turns);
total_width = straight_length + 2 * wall_thickness;
total_depth = n_turns * (channel_width + wall_thickness * 2) + wall_thickness;
total_height = channel_height + wall_thickness * 2 + dielectric_thickness * 2;

inlet_pos = [wall_thickness + straight_length/2, wall_thickness + channel_width/2, 0];
outlet_pos = [wall_thickness + straight_length/2, total_depth - wall_thickness - channel_width/2, 0];

// Canal serpentín
module serpentine_channel() {
    for (i = [0:n_turns-1]) {
        y_offset = wall_thickness + channel_width/2 + i * (channel_width + wall_thickness);

        translate([wall_thickness, y_offset, wall_thickness + dielectric_thickness])
            cube([straight_length, channel_width, channel_height]);

        if (i < n_turns - 1) {
            translate([wall_thickness + straight_length - turn_radius,
                      y_offset + channel_width/2,
                      wall_thickness + dielectric_thickness])
                cube([turn_radius, channel_width + wall_thickness, channel_height]);

            translate([wall_thickness,
                      y_offset + channel_width,
                      wall_thickness + dielectric_thickness])
                cube([straight_length, wall_thickness, channel_height]);
        }
    }
}

// Cuerpo principal
module reactor_body_only() {
    difference() {
        // Bloque exterior
        cube([total_width, total_depth, total_height]);

        // Canal serpentín
        serpentine_channel();

        // Entrada de líquido
        translate([inlet_pos[0], inlet_pos[1], -0.1])
            cylinder(d = inlet_diameter, h = wall_thickness + 0.2);

        // Salida de líquido
        translate([outlet_pos[0], outlet_pos[1], -0.1])
            cylinder(d = outlet_diameter, h = wall_thickness + 0.2);

        // Entrada de gas
        translate([-0.1, total_depth/2, total_height/2])
            rotate([0, 90, 0])
                cylinder(d = gas_inlet_diameter, h = wall_thickness + 0.2);
    }
}

// Renderizar solo cuerpo
reactor_body_only();

echo("═══════════════════════════════════════════════════════════════");
echo("  CUERPO DEL REACTOR - Para impresión con resina");
echo("═══════════════════════════════════════════════════════════════");
echo(str("  Dimensiones: ", total_width, " x ", total_depth, " x ", total_height, " mm"));
echo(str("  Volumen canal: ", channel_width * channel_height * channel_length / 1000, " mL"));
echo("═══════════════════════════════════════════════════════════════");
