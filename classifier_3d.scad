/*
 * =====================================================================================
 *  CLASIFICADOR DE QUANTUM DOTS POR FLOTABILIDAD OPTICA
 *  Modelo 3D parametrico - Separacion selectiva por excitacion LED
 * =====================================================================================
 *
 *  Clasificador con zonas de excitacion optica. Cada zona contiene:
 *    - Array de LEDs en la parte superior (excitacion selectiva)
 *    - Puerto de coleccion superior (QDots excitados que flotan)
 *    - Puerto de coleccion inferior (debris que sedimenta por gravedad)
 *    - Barreras verticales (malla de poro grande) entre zonas
 *
 *  Disposicion HORIZONTAL con separacion VERTICAL:
 *
 *         LED 520nm        LED 405nm        LED 365nm
 *           vvvv              vvvv              vvvv
 *    ┌──────────────┬──────────────┬──────────────┐
 *    │  ^ QDots R   │  ^ QDots B   │  ^ QDots UV  │  <- Puerto ARRIBA
 *  IN→              barrera                        → WASTE
 *    │  v debris    │  v debris    │  v debris    │  <- Puerto ABAJO
 *    └──────────────┴──────────────┴──────────────┘
 *
 *  Para generar STL: openscad -o classifier.stl classifier_3d.scad
 *  Para visualizar:  openscad classifier_3d.scad
 */

// =====================================================================================
//  PARAMETROS DEL CLASIFICADOR
// =====================================================================================

// Numero de zonas de excitacion
n_zones = 3;

// Dimensiones de cada zona
zone_length = 40.0;          // mm - longitud (eje X)
zone_height = 15.0;          // mm - altura (eje Z, critica para separacion)
zone_width = 10.0;           // mm - profundidad (eje Y)

// Canal y paredes
channel_height = 2.0;        // mm - canal de entrada
wall_thickness = 2.0;        // mm - espesor de paredes

// Barrera entre zonas (malla de poro grande)
barrier_thickness = 1.0;     // mm - espesor de la ranura para malla
barrier_slot_tolerance = 0.2; // mm - tolerancia de insercion

// LEDs (parte superior)
led_diameter = 5.0;           // mm
led_mount_depth = 3.0;        // mm
led_count_per_zone = 4;
led_spacing = 8.0;            // mm - distancia entre LEDs

// Puertos de coleccion
top_port_diameter = 2.0;      // mm - QDots excitados (arriba)
bottom_port_diameter = 2.0;   // mm - debris (abajo)

// Conexiones
inlet_diameter = 2.0;         // mm
waste_diameter = 2.0;         // mm
connector_length = 8.0;       // mm

// Ventana de observacion lateral
window_diameter = 8.0;        // mm
window_seat_depth = 0.3;      // mm
show_windows = true;

// Tornillos tapa (M3)
screw_diameter = 3.0;         // mm
screw_head_diameter = 5.5;    // mm

// Visualizacion
$fn = 50;
exploded_view = false;
cross_section = false;
show_leds = true;
show_barriers = true;

// =====================================================================================
//  CALCULOS DERIVADOS
// =====================================================================================

// Espacio para LEDs arriba
led_space = led_mount_depth + 2.0;

// Longitud del cuerpo (sin conectores)
body_length = n_zones * zone_length + (n_zones - 1) * barrier_thickness +
              2 * wall_thickness;
total_length = body_length + connector_length * 2;

// Ancho total
total_width = zone_width + 2 * wall_thickness;

// Altura total
total_height = wall_thickness + zone_height + wall_thickness + led_space;

// Alturas de referencia
zone_base_z = wall_thickness;
zone_top_z = wall_thickness + zone_height;
led_base_z = zone_top_z + wall_thickness;

// Centro Y
center_y = total_width / 2;

// =====================================================================================
//  FUNCIONES DE POSICION
// =====================================================================================

// Posicion X del inicio de una zona
function zone_x_start(i) = wall_thickness + i * (zone_length + barrier_thickness);

// Posicion X del centro de una zona
function zone_x_center(i) = zone_x_start(i) + zone_length / 2;

// Posicion X de una barrera (entre zona i e i+1)
function barrier_x(i) = wall_thickness + (i + 1) * zone_length + i * barrier_thickness +
                         barrier_thickness / 2;

// =====================================================================================
//  MODULOS
// =====================================================================================

// -------------------------------------------------------------------------------------
//  Cavidad de zona - espacio rectangular donde ocurre la separacion
// -------------------------------------------------------------------------------------
module zone_cavity(zone_idx) {
    translate([zone_x_start(zone_idx), wall_thickness, zone_base_z])
        cube([zone_length, zone_width, zone_height]);
}

// -------------------------------------------------------------------------------------
//  Ranura para barrera/malla entre zonas
// -------------------------------------------------------------------------------------
module barrier_slot(barrier_idx) {
    bx = barrier_x(barrier_idx);
    translate([bx - (barrier_thickness + barrier_slot_tolerance) / 2,
               wall_thickness + 0.5,
               zone_base_z])
        cube([barrier_thickness + barrier_slot_tolerance,
              zone_width - 1.0,  // ligeramente menor para asentar
              zone_height]);
}

// -------------------------------------------------------------------------------------
//  Alojamientos de LEDs en la parte superior de una zona
// -------------------------------------------------------------------------------------
module led_bores(zone_idx) {
    xc = zone_x_center(zone_idx);
    xs = zone_x_start(zone_idx);

    for (i = [0 : led_count_per_zone - 1]) {
        offset_x = (i - (led_count_per_zone - 1) / 2) * led_spacing;
        led_x = xc + offset_x;

        // Solo si el LED cae dentro de la zona
        if (led_x > xs + 3.0 && led_x < xs + zone_length - 3.0) {
            translate([led_x, center_y, total_height - led_mount_depth])
                cylinder(d = led_diameter, h = led_mount_depth + 0.01);
        }
    }
}

// -------------------------------------------------------------------------------------
//  Puerto de coleccion superior (QDots excitados)
// -------------------------------------------------------------------------------------
module top_collection_port(zone_idx) {
    xc = zone_x_center(zone_idx) - zone_length / 4;

    // Puerto vertical desde techo de zona hacia arriba
    translate([xc, center_y, zone_top_z - 0.01])
        cylinder(d = top_port_diameter, h = wall_thickness + led_space + 0.02);
}

// -------------------------------------------------------------------------------------
//  Puerto de coleccion inferior (debris)
// -------------------------------------------------------------------------------------
module bottom_collection_port(zone_idx) {
    xc = zone_x_center(zone_idx) + zone_length / 4;

    // Puerto vertical desde base de zona hacia abajo
    translate([xc, center_y, -0.01])
        cylinder(d = bottom_port_diameter, h = wall_thickness + 0.02);
}

// -------------------------------------------------------------------------------------
//  Ventana lateral de observacion
// -------------------------------------------------------------------------------------
module observation_window(zone_idx) {
    if (show_windows) {
        xc = zone_x_center(zone_idx);
        zone_center_z = zone_base_z + zone_height / 2;

        // Ventana en la cara frontal (Y = 0)
        translate([xc, -0.01, zone_center_z])
            rotate([-90, 0, 0]) {
                // Orificio pasante
                cylinder(d = window_diameter - 2.0, h = wall_thickness + 0.02);
                // Asiento para disco de cuarzo
                translate([0, 0, -window_seat_depth])
                    cylinder(d = window_diameter, h = window_seat_depth + 0.02);
            }
    }
}

// -------------------------------------------------------------------------------------
//  Conector de entrada (inlet)
// -------------------------------------------------------------------------------------
module inlet_connector() {
    zone_center_z = zone_base_z + zone_height / 2;
    translate([wall_thickness / 2, center_y, zone_center_z])
        rotate([0, -90, 0])
            cylinder(d = inlet_diameter, h = connector_length + wall_thickness);
}

// -------------------------------------------------------------------------------------
//  Conector de residuo (waste)
// -------------------------------------------------------------------------------------
module waste_connector() {
    zone_center_z = zone_base_z + zone_height / 2;
    translate([body_length - wall_thickness / 2, center_y, zone_center_z])
        rotate([0, 90, 0])
            cylinder(d = waste_diameter, h = connector_length + wall_thickness);
}

// =====================================================================================
//  CUERPO PRINCIPAL DEL CLASIFICADOR
// =====================================================================================

module classifier_body() {
    difference() {
        // Bloque solido exterior
        cube([body_length, total_width, total_height]);

        // Cavidades de zona
        for (i = [0 : n_zones - 1]) {
            zone_cavity(i);
        }

        // Ranuras de barrera
        for (i = [0 : n_zones - 2]) {
            barrier_slot(i);
        }

        // Elementos por zona
        for (i = [0 : n_zones - 1]) {
            led_bores(i);
            top_collection_port(i);
            bottom_collection_port(i);
            observation_window(i);
        }

        // Conectores
        inlet_connector();
        waste_connector();

        // Corte transversal para visualizacion
        if (cross_section) {
            translate([-1, center_y, -1])
                cube([body_length + 2, total_width, total_height + 2]);
        }
    }
}

// =====================================================================================
//  COMPONENTES DE VISUALIZACION
// =====================================================================================

// Indicadores de LEDs
module led_indicators() {
    if (show_leds) {
        led_colors = ["Green", "Purple", "Violet"];
        for (i = [0 : n_zones - 1]) {
            xc = zone_x_center(i);
            xs = zone_x_start(i);
            color_idx = i < 3 ? i : 0;

            color(led_colors[color_idx], 0.7)
                for (j = [0 : led_count_per_zone - 1]) {
                    offset_x = (j - (led_count_per_zone - 1) / 2) * led_spacing;
                    led_x = xc + offset_x;
                    if (led_x > xs + 3.0 && led_x < xs + zone_length - 3.0) {
                        translate([led_x, center_y,
                                   total_height - led_mount_depth + 0.5])
                            cylinder(d = led_diameter - 0.5,
                                     h = led_mount_depth - 1.0);
                    }
                }
        }
    }
}

// Indicadores de barreras/mallas
module barrier_indicators() {
    if (show_barriers) {
        for (i = [0 : n_zones - 2]) {
            bx = barrier_x(i);
            color("SteelBlue", 0.4)
                translate([bx - barrier_thickness / 2,
                           wall_thickness + 1.0,
                           zone_base_z])
                    cube([barrier_thickness, zone_width - 2.0, zone_height]);
        }
    }
}

// Conectores exteriores (visuales)
module inlet_connector_visual() {
    zone_center_z = zone_base_z + zone_height / 2;
    color("SteelBlue", 0.6)
        translate([-connector_length, center_y, zone_center_z])
            rotate([0, 90, 0])
                difference() {
                    cylinder(d = inlet_diameter + wall_thickness,
                             h = connector_length);
                    translate([0, 0, -0.1])
                        cylinder(d = inlet_diameter, h = connector_length + 0.2);
                }
}

module waste_connector_visual() {
    zone_center_z = zone_base_z + zone_height / 2;
    color("OrangeRed", 0.6)
        translate([body_length, center_y, zone_center_z])
            rotate([0, 90, 0])
                difference() {
                    cylinder(d = waste_diameter + wall_thickness,
                             h = connector_length);
                    translate([0, 0, -0.1])
                        cylinder(d = waste_diameter, h = connector_length + 0.2);
                }
}

// Conectores de coleccion (visuales)
module collection_port_connectors() {
    for (i = [0 : n_zones - 1]) {
        // Puerto superior (QDots)
        color("LimeGreen", 0.6) {
            xc = zone_x_center(i) - zone_length / 4;
            translate([xc, center_y, total_height])
                difference() {
                    cylinder(d = top_port_diameter + wall_thickness,
                             h = connector_length / 2);
                    translate([0, 0, -0.1])
                        cylinder(d = top_port_diameter,
                                 h = connector_length / 2 + 0.2);
                }
        }

        // Puerto inferior (debris)
        color("OrangeRed", 0.4) {
            xc = zone_x_center(i) + zone_length / 4;
            translate([xc, center_y, -connector_length / 2])
                difference() {
                    cylinder(d = bottom_port_diameter + wall_thickness,
                             h = connector_length / 2);
                    translate([0, 0, -0.1])
                        cylinder(d = bottom_port_diameter,
                                 h = connector_length / 2 + 0.2);
                }
        }
    }
}

// Ventanas de cuarzo (visuales)
module window_indicators() {
    if (show_windows) {
        for (i = [0 : n_zones - 1]) {
            xc = zone_x_center(i);
            zone_center_z = zone_base_z + zone_height / 2;
            color("Cyan", 0.3)
                translate([xc, -window_seat_depth, zone_center_z])
                    rotate([-90, 0, 0])
                        cylinder(d = window_diameter - 0.5, h = 1.0);
        }
    }
}

// =====================================================================================
//  ENSAMBLAJE FINAL
// =====================================================================================

module classifier_assembly() {
    explode_z = exploded_view ? 10 : 0;

    // Cuerpo principal
    color("LightGray", 0.9)
        classifier_body();

    // Componentes visuales
    translate([0, 0, explode_z]) {
        led_indicators();
    }

    barrier_indicators();
    window_indicators();

    // Conectores exteriores
    inlet_connector_visual();
    waste_connector_visual();
    collection_port_connectors();
}

// =====================================================================================
//  RENDERIZADO
// =====================================================================================

classifier_assembly();

// =====================================================================================
//  INFORMACION DEL MODELO
// =====================================================================================

echo("=====================================================================================");
echo("  CLASIFICADOR POR FLOTABILIDAD OPTICA - Separacion selectiva de QDots");
echo("=====================================================================================");
echo(str("  Numero de zonas:        ", n_zones));
echo(str("  Longitud del cuerpo:    ", body_length, " mm"));
echo(str("  Ancho total:            ", total_width, " mm"));
echo(str("  Altura total:           ", total_height, " mm"));
echo(str("  Zona (L x W x H):      ", zone_length, " x ", zone_width, " x ", zone_height, " mm"));
echo("-------------------------------------------------------------------------------------");
echo("  Zonas de excitacion:");
zone_leds = ["520 nm (verde)", "405 nm (UV-azul)", "365 nm (UV)"];
zone_targets = ["QDots rojos 3.5-5.0 nm", "QDots azules 2.5-3.5 nm", "QDots UV 1.5-2.5 nm"];
for (i = [0 : n_zones - 1]) {
    echo(str("    Zona ", i + 1, ": LED ", zone_leds[i], " -> ", zone_targets[i]));
}
echo("-------------------------------------------------------------------------------------");
echo(str("  LEDs: ", led_count_per_zone, " x ", led_diameter, " mm por zona"));
echo(str("  Barrera: malla ~50 um (sin caida de presion)"));
echo(str("  Puerto superior: diam ", top_port_diameter, " mm (coleccion QDots)"));
echo(str("  Puerto inferior: diam ", bottom_port_diameter, " mm (coleccion debris)"));
echo(str("  Ventana observacion: diam ", window_diameter, " mm"));
echo("=====================================================================================");
