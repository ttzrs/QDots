/*
 * ═══════════════════════════════════════════════════════════════════════════════
 *  REACTOR DBD PARA SÍNTESIS DE CARBON QUANTUM DOTS
 *  Modelo 3D paramétrico generado desde simulación VQE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *  Parámetros optimizados para:
 *    - Producción: ~30 mg/h de CQDs
 *    - Longitud de onda: 463 nm (azul)
 *    - Tamaño partícula: 2.48 nm
 *
 *  Para generar STL: openscad -o reactor.stl reactor_3d.scad
 *  Para visualizar: openscad reactor_3d.scad
 */

// ═══════════════════════════════════════════════════════════════════════════════
//  PARÁMETROS DEL REACTOR (desde reactor_optimized.json)
// ═══════════════════════════════════════════════════════════════════════════════

// Geometría del canal
channel_width = 3.0;          // mm - ancho del canal
channel_height = 1.0;         // mm - altura del canal
channel_length = 250;         // mm - longitud total desenrollada
wall_thickness = 2.0;         // mm - espesor de paredes
n_turns = 6;                  // número de vueltas del serpentín
turn_radius = 4.5;            // mm - radio de las curvas

// Electrodos
electrode_width = 1.5;        // mm
electrode_gap = 1.0;          // mm
electrode_thickness = 0.1;    // mm

// Dieléctrico
dielectric_thickness = 0.8;   // mm

// Conexiones
inlet_diameter = 2.0;         // mm
outlet_diameter = 2.0;        // mm
gas_inlet_diameter = 1.5;     // mm
connector_length = 8.0;       // mm

// Enfriamiento
cooling_channel_diameter = 3.0;  // mm
cooling_serpentine_length = 150; // mm

// Parámetros de visualización
$fn = 32;                     // Resolución de curvas
show_electrodes = true;       // Mostrar electrodos
show_channels = true;         // Mostrar canales internos
show_cooling = true;          // Mostrar sistema de enfriamiento
exploded_view = false;        // Vista explosionada
cross_section = false;        // Vista en corte

// ═══════════════════════════════════════════════════════════════════════════════
//  CÁLCULOS DERIVADOS
// ═══════════════════════════════════════════════════════════════════════════════

// Dimensiones del cuerpo
straight_length = channel_length / (2 * n_turns);  // Longitud de cada tramo recto
total_width = straight_length + 2 * wall_thickness;
total_depth = n_turns * (channel_width + wall_thickness * 2) + wall_thickness;
total_height = channel_height + wall_thickness * 2 + dielectric_thickness * 2;

// Posiciones de las conexiones
inlet_pos = [wall_thickness + straight_length/2, wall_thickness + channel_width/2, 0];
outlet_pos = [wall_thickness + straight_length/2, total_depth - wall_thickness - channel_width/2, 0];

// ═══════════════════════════════════════════════════════════════════════════════
//  MÓDULOS
// ═══════════════════════════════════════════════════════════════════════════════

// Canal serpentín (sustracción)
module serpentine_channel() {
    for (i = [0:n_turns-1]) {
        y_offset = wall_thickness + channel_width/2 + i * (channel_width + wall_thickness);

        // Tramo recto hacia la derecha
        translate([wall_thickness, y_offset, wall_thickness + dielectric_thickness])
            cube([straight_length, channel_width, channel_height]);

        // Curva superior (si no es la última vuelta)
        if (i < n_turns - 1) {
            // Conexión en U entre tramos
            translate([wall_thickness + straight_length - turn_radius,
                      y_offset + channel_width/2,
                      wall_thickness + dielectric_thickness])
                cube([turn_radius, channel_width + wall_thickness, channel_height]);

            // Tramo de retorno parcial
            translate([wall_thickness,
                      y_offset + channel_width,
                      wall_thickness + dielectric_thickness])
                cube([straight_length, wall_thickness, channel_height]);
        }
    }
}

// Canal serpentín mejorado con curvas suaves
module serpentine_channel_smooth() {
    union() {
        for (i = [0:n_turns-1]) {
            y_pos = wall_thickness + channel_width/2 + i * (channel_width * 2 + wall_thickness);

            // Tramo horizontal
            translate([wall_thickness, y_pos, wall_thickness + dielectric_thickness])
                cube([straight_length, channel_width, channel_height]);

            // Conexión curva al siguiente nivel
            if (i < n_turns - 1) {
                // Semicírculo de conexión
                translate([wall_thickness + straight_length,
                          y_pos + channel_width,
                          wall_thickness + dielectric_thickness]) {
                    rotate([0, 0, -90])
                        rotate_extrude(angle = 180, $fn = 24)
                            translate([channel_width/2 + wall_thickness/2, 0, 0])
                                square([channel_width, channel_height]);
                }
            }
        }
    }
}

// Electrodo individual
module electrode(length, width, thickness) {
    color("orange", 0.8)
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

// Sistema de electrodos completo
module electrodes_system() {
    if (show_electrodes) {
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
}

// Conector de entrada/salida
module connector(diameter, length, type="inlet") {
    color(type == "inlet" ? "blue" : "green", 0.6) {
        difference() {
            cylinder(d = diameter + wall_thickness, h = length);
            translate([0, 0, -0.1])
                cylinder(d = diameter, h = length + 0.2);
        }

        // Collar de conexión
        translate([0, 0, length - 2])
            difference() {
                cylinder(d = diameter + wall_thickness * 2, h = 2);
                translate([0, 0, -0.1])
                    cylinder(d = diameter, h = 2.2);
            }
    }
}

// Sistema de enfriamiento
module cooling_system() {
    if (show_cooling) {
        color("cyan", 0.4) {
            // Serpentín de enfriamiento debajo del reactor
            translate([total_width/2, wall_thickness, -cooling_channel_diameter - 2]) {
                for (i = [0:3]) {
                    y_pos = i * (total_depth - wall_thickness * 2) / 4;

                    // Tramo horizontal
                    translate([-total_width/2 + wall_thickness, y_pos, 0])
                        rotate([0, 90, 0])
                            cylinder(d = cooling_channel_diameter,
                                    h = total_width - wall_thickness * 2);

                    // Conexiones
                    if (i < 3) {
                        x_pos = (i % 2 == 0) ? total_width/2 - wall_thickness :
                                              -total_width/2 + wall_thickness;
                        translate([x_pos, y_pos, 0])
                            rotate([-90, 0, 0])
                                cylinder(d = cooling_channel_diameter,
                                        h = (total_depth - wall_thickness * 2) / 4);
                    }
                }
            }
        }
    }
}

// Cuerpo principal del reactor
module reactor_body() {
    difference() {
        // Bloque exterior
        color("lightgray", 0.9)
            cube([total_width, total_depth, total_height]);

        // Canal serpentín (sustracción)
        if (show_channels) {
            color("white")
                serpentine_channel();
        }

        // Entrada de líquido
        translate([inlet_pos[0], inlet_pos[1], -0.1])
            cylinder(d = inlet_diameter, h = wall_thickness + 0.2);

        // Salida de líquido
        translate([outlet_pos[0], outlet_pos[1], -0.1])
            cylinder(d = outlet_diameter, h = wall_thickness + 0.2);

        // Entrada de gas (lateral)
        translate([-0.1, total_depth/2, total_height/2])
            rotate([0, 90, 0])
                cylinder(d = gas_inlet_diameter, h = wall_thickness + 0.2);

        // Corte transversal para visualización
        if (cross_section) {
            translate([-1, total_depth/2, -1])
                cube([total_width + 2, total_depth, total_height + 2]);
        }
    }
}

// Tapa superior (separable para acceso)
module top_cover() {
    cover_height = wall_thickness;

    color("darkgray", 0.8)
    translate([0, 0, total_height]) {
        difference() {
            cube([total_width, total_depth, cover_height]);

            // Ranuras para electrodos
            for (i = [0:n_turns-1]) {
                y_offset = wall_thickness + (channel_width - electrode_width)/2 +
                          i * (channel_width * 2 + wall_thickness);
                translate([wall_thickness * 2, y_offset, -0.1])
                    cube([straight_length - wall_thickness * 3,
                          electrode_width,
                          cover_height + 0.2]);
            }

            // Agujeros para tornillos de fijación
            for (x = [wall_thickness, total_width - wall_thickness])
                for (y = [wall_thickness, total_depth - wall_thickness])
                    translate([x, y, -0.1])
                        cylinder(d = 2, h = cover_height + 0.2);
        }
    }
}

// Conectores eléctricos
module electrical_connectors() {
    color("gold", 0.9) {
        // Conector positivo
        translate([total_width + 2, total_depth/3, total_height/2])
            rotate([0, 90, 0]) {
                cylinder(d = 4, h = 5);
                translate([0, 0, 5])
                    cylinder(d = 6, h = 2);
            }

        // Conector negativo
        translate([total_width + 2, 2*total_depth/3, total_height/2])
            rotate([0, 90, 0]) {
                cylinder(d = 4, h = 5);
                translate([0, 0, 5])
                    cylinder(d = 6, h = 2);
            }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  ENSAMBLAJE FINAL
// ═══════════════════════════════════════════════════════════════════════════════

module reactor_assembly() {
    // Explosión vertical para vista
    explode_z = exploded_view ? 10 : 0;

    // Cuerpo principal
    reactor_body();

    // Tapa superior
    translate([0, 0, explode_z])
        top_cover();

    // Electrodos
    translate([0, 0, wall_thickness])
        electrodes_system();

    // Conectores de fluido
    translate([inlet_pos[0], inlet_pos[1], -connector_length])
        connector(inlet_diameter, connector_length, "inlet");

    translate([outlet_pos[0], outlet_pos[1], -connector_length])
        connector(outlet_diameter, connector_length, "outlet");

    // Conector de gas
    translate([-connector_length, total_depth/2, total_height/2])
        rotate([0, 90, 0])
            connector(gas_inlet_diameter, connector_length, "gas");

    // Sistema de enfriamiento
    cooling_system();

    // Conectores eléctricos
    electrical_connectors();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  RENDERIZADO
// ═══════════════════════════════════════════════════════════════════════════════

// Renderizar el ensamblaje completo
reactor_assembly();

// Información del modelo
echo("═══════════════════════════════════════════════════════════════");
echo("  REACTOR DBD PARA CQDs - Dimensiones");
echo("═══════════════════════════════════════════════════════════════");
echo(str("  Ancho total: ", total_width, " mm"));
echo(str("  Profundidad total: ", total_depth, " mm"));
echo(str("  Altura total: ", total_height + wall_thickness, " mm"));
echo(str("  Volumen canal: ", channel_width * channel_height * channel_length / 1000, " mL"));
echo(str("  Área plasma-líquido: ", channel_width * channel_length / 100, " cm²"));
echo("═══════════════════════════════════════════════════════════════");
