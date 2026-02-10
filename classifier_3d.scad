/*
 * =====================================================================================
 *  CLASIFICADOR DE QUANTUM DOTS POR TAMANIO - LABERINTO ESCALONADO
 *  Modelo 3D parametrico para modulo post-reactor
 * =====================================================================================
 *
 *  Modulo de clasificacion con 5 etapas en escalera lineal.
 *  Cada etapa contiene:
 *    - Canal principal de flujo (camino del laberinto)
 *    - Soporte circular para membrana (13 mm Nuclepore/Anodisc)
 *    - Camara de coleccion debajo de la membrana
 *    - Puertos opticos: ventana de cuarzo, LED y detector a 90 grados
 *    - Puerto de salida individual
 *
 *  Disposicion LINEAL (escalera) para acceso facil a membranas:
 *
 *    INLET --[canal]--|membrana|--[canal]--|membrana|-- ... --WASTE
 *                        |                    |
 *                    [camara1]            [camara2]
 *                   (optica)              (optica)
 *                        |                    |
 *                    [salida1]            [salida2]
 *
 *  Para generar STL: openscad -o classifier.stl classifier_3d.scad
 *  Para visualizar:  openscad classifier_3d.scad
 */

// =====================================================================================
//  PARAMETROS DEL CLASIFICADOR
// =====================================================================================

// Numero de etapas
n_stages = 5;

// Dimensiones del canal principal
channel_width = 2.0;          // mm - ancho del canal de flujo
channel_height = 1.0;         // mm - altura del canal de flujo
wall_thickness = 2.0;         // mm - espesor de paredes

// Pitch entre etapas
stage_pitch = 25.0;           // mm - distancia centro a centro entre etapas

// Membrana
membrane_diameter = 13.0;     // mm - Whatman 13 mm
membrane_seat_depth = 0.5;    // mm - profundidad del asiento
membrane_clamp_thickness = 1.0; // mm - grosor de la zona de sujecion

// Camara de coleccion
chamber_width = 10.0;         // mm
chamber_length = 15.0;        // mm
chamber_depth = 5.0;          // mm

// Puertos opticos
window_diameter = 8.0;        // mm - ventana de cuarzo
window_thickness = 1.0;       // mm - grosor del disco de cuarzo
window_seat_depth = 0.3;      // mm - rebaje para asentar la ventana
led_diameter = 5.0;           // mm - diametro del LED
led_mount_depth = 3.0;        // mm - profundidad del alojamiento LED
detector_diameter = 5.0;      // mm - diametro del fotodetector
detector_mount_depth = 3.0;   // mm - profundidad del alojamiento detector

// Conexiones
inlet_diameter = 2.0;         // mm
outlet_diameter = 1.5;        // mm
waste_diameter = 2.0;         // mm
connector_length = 8.0;       // mm

// Tornillos de la tapa
screw_diameter = 3.0;         // mm - M3
screw_head_diameter = 5.5;    // mm - cabeza M3
screw_positions_per_stage = 4;

// Visualizacion
$fn = 50;                     // Resolucion de superficies curvas
exploded_view = false;        // Vista explosionada
cross_section = false;        // Vista en corte (plano Y)
show_membranes = true;        // Mostrar discos de membrana
show_optics = true;           // Mostrar componentes opticos

// =====================================================================================
//  CALCULOS DERIVADOS
// =====================================================================================

// Dimensiones totales del cuerpo
// Largo: desde conector de entrada hasta conector de residuo
body_length = (n_stages - 1) * stage_pitch + membrane_diameter + wall_thickness * 4;
total_length = body_length + connector_length * 2;

// Ancho: canal arriba + camara abajo + paredes
total_width = max(chamber_length, membrane_diameter) + wall_thickness * 4;

// Altura: canal + membrana + camara + pared inferior + pared superior
channel_zone_height = channel_height + wall_thickness;
membrane_zone_height = membrane_seat_depth + membrane_clamp_thickness;
chamber_zone_height = chamber_depth;
total_height = wall_thickness + channel_zone_height + membrane_zone_height +
               chamber_zone_height + wall_thickness;

// Tapa superior
top_plate_thickness = wall_thickness + 1.0;

// Posicion Y del centro del canal principal (eje del flujo)
channel_center_y = total_width / 2;

// Altura Z de la base del canal principal
channel_base_z = total_height - wall_thickness - channel_height;

// Altura Z del asiento de membrana (justo debajo del canal)
membrane_seat_z = channel_base_z - membrane_seat_depth;

// Altura Z del techo de la camara de coleccion
chamber_top_z = membrane_seat_z - membrane_clamp_thickness;

// Altura Z de la base de la camara
chamber_base_z = chamber_top_z - chamber_depth;

// Centro Z de la camara (para puertos opticos)
chamber_center_z = chamber_top_z - chamber_depth / 2;

// =====================================================================================
//  FUNCION: posicion X del centro de cada etapa
// =====================================================================================

function stage_x(i) = wall_thickness * 2 + membrane_diameter / 2 + i * stage_pitch;

// =====================================================================================
//  MODULOS
// =====================================================================================

// -------------------------------------------------------------------------------------
//  Canal principal recto - recorre toda la longitud del cuerpo
// -------------------------------------------------------------------------------------
module main_channel() {
    // Canal recto a lo largo del eje X
    translate([wall_thickness, channel_center_y - channel_width / 2, channel_base_z])
        cube([body_length - wall_thickness * 2, channel_width, channel_height]);
}

// -------------------------------------------------------------------------------------
//  Asiento de membrana - rebaje circular para disco de filtro
// -------------------------------------------------------------------------------------
module membrane_slot(stage_idx) {
    cx = stage_x(stage_idx);

    // Asiento circular para la membrana
    translate([cx, channel_center_y, membrane_seat_z])
        cylinder(d = membrane_diameter, h = membrane_seat_depth + 0.01);

    // Orificio pasante para flujo (ligeramente menor que la membrana)
    translate([cx, channel_center_y, membrane_seat_z - membrane_clamp_thickness - 0.01])
        cylinder(d = membrane_diameter - 2.0, h = membrane_clamp_thickness + 0.02);
}

// -------------------------------------------------------------------------------------
//  Camara de coleccion - rectangular, debajo de la membrana
// -------------------------------------------------------------------------------------
module collection_chamber(stage_idx) {
    cx = stage_x(stage_idx);

    translate([cx - chamber_width / 2,
               channel_center_y - chamber_length / 2,
               chamber_base_z])
        cube([chamber_width, chamber_length, chamber_depth]);
}

// -------------------------------------------------------------------------------------
//  Asiento para ventana optica - bolsillo circular en la pared lateral de la camara
// -------------------------------------------------------------------------------------
module optical_window_seat(stage_idx) {
    cx = stage_x(stage_idx);

    // Ventana en la pared frontal de la camara (eje -Y)
    translate([cx, -0.01, chamber_center_z])
        rotate([-90, 0, 0]) {
            // Orificio pasante para la luz
            cylinder(d = window_diameter - 2.0,
                     h = channel_center_y - chamber_length / 2 + 0.02);
            // Asiento rebajado para disco de cuarzo (en la cara exterior)
            translate([0, 0, -window_seat_depth])
                cylinder(d = window_diameter, h = window_seat_depth + 0.02);
        }
}

// -------------------------------------------------------------------------------------
//  Alojamiento para LED - orificio cilindrico en pared lateral
// -------------------------------------------------------------------------------------
module led_bore(stage_idx) {
    cx = stage_x(stage_idx);

    // LED entra desde la pared frontal (Y = 0), perpendicular a la camara
    translate([cx, -0.01, chamber_center_z])
        rotate([-90, 0, 0])
            cylinder(d = led_diameter, h = led_mount_depth + 0.01);
}

// -------------------------------------------------------------------------------------
//  Alojamiento para fotodetector - orificio a 90 grados del LED
// -------------------------------------------------------------------------------------
module detector_bore(stage_idx) {
    cx = stage_x(stage_idx);

    // Detector entra desde la pared posterior (Y = total_width), a 90 grados del LED
    translate([cx, total_width + 0.01, chamber_center_z])
        rotate([90, 0, 0])
            cylinder(d = detector_diameter, h = detector_mount_depth + 0.01);
}

// -------------------------------------------------------------------------------------
//  Puerto de salida de cada etapa - orificio vertical en la base de la camara
// -------------------------------------------------------------------------------------
module stage_outlet(stage_idx) {
    cx = stage_x(stage_idx);

    // Salida vertical hacia abajo
    translate([cx, channel_center_y, -0.01])
        cylinder(d = outlet_diameter, h = chamber_base_z + 0.02);
}

// -------------------------------------------------------------------------------------
//  Agujeros para tornillos M3 alrededor de cada membrana
// -------------------------------------------------------------------------------------
module screw_holes(stage_idx) {
    cx = stage_x(stage_idx);
    screw_ring_radius = membrane_diameter / 2 + wall_thickness;

    for (j = [0:screw_positions_per_stage - 1]) {
        angle = j * 360 / screw_positions_per_stage + 45;
        sx = cx + screw_ring_radius * cos(angle);
        sy = channel_center_y + screw_ring_radius * sin(angle);

        // Orificio pasante completo (para el tornillo)
        translate([sx, sy, -0.01])
            cylinder(d = screw_diameter, h = total_height + top_plate_thickness + 0.02);
    }
}

// -------------------------------------------------------------------------------------
//  Conector de entrada (tubular)
// -------------------------------------------------------------------------------------
module inlet_connector() {
    // Orificio que conecta el exterior con el inicio del canal
    translate([wall_thickness / 2, channel_center_y, channel_base_z + channel_height / 2])
        rotate([0, -90, 0])
            cylinder(d = inlet_diameter, h = connector_length + wall_thickness);
}

// -------------------------------------------------------------------------------------
//  Conector de residuo (tubular, extremo opuesto)
// -------------------------------------------------------------------------------------
module waste_connector() {
    // Orificio que conecta el final del canal con el exterior
    translate([body_length - wall_thickness / 2, channel_center_y,
               channel_base_z + channel_height / 2])
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

        // Sustraccion del canal principal
        main_channel();

        // Sustracciones por etapa
        for (i = [0 : n_stages - 1]) {
            membrane_slot(i);
            collection_chamber(i);
            optical_window_seat(i);
            led_bore(i);
            detector_bore(i);
            stage_outlet(i);
            screw_holes(i);
        }

        // Conectores de entrada y residuo
        inlet_connector();
        waste_connector();

        // Corte transversal para visualizacion
        if (cross_section) {
            translate([-1, channel_center_y, -1])
                cube([body_length + 2, total_width, total_height + top_plate_thickness + 2]);
        }
    }
}

// =====================================================================================
//  TAPA SUPERIOR REMOVIBLE
// =====================================================================================

module top_plate() {
    difference() {
        // Placa solida
        translate([0, 0, total_height])
            cube([body_length, total_width, top_plate_thickness]);

        // Aberturas circulares para acceso a membranas
        for (i = [0 : n_stages - 1]) {
            cx = stage_x(i);
            translate([cx, channel_center_y, total_height - 0.01])
                cylinder(d = membrane_diameter + 0.5, h = top_plate_thickness + 0.02);
        }

        // Agujeros para tornillos (avellanados en la tapa)
        for (i = [0 : n_stages - 1]) {
            cx = stage_x(i);
            screw_ring_radius = membrane_diameter / 2 + wall_thickness;

            for (j = [0 : screw_positions_per_stage - 1]) {
                angle = j * 360 / screw_positions_per_stage + 45;
                sx = cx + screw_ring_radius * cos(angle);
                sy = channel_center_y + screw_ring_radius * sin(angle);

                // Pasante del tornillo
                translate([sx, sy, total_height - 0.01])
                    cylinder(d = screw_diameter, h = top_plate_thickness + 0.02);

                // Avellanado para cabeza del tornillo
                translate([sx, sy, total_height + top_plate_thickness - wall_thickness])
                    cylinder(d = screw_head_diameter, h = wall_thickness + 0.01);
            }
        }

        // Corte transversal (si esta activo)
        if (cross_section) {
            translate([-1, channel_center_y, -1])
                cube([body_length + 2, total_width, total_height + top_plate_thickness + 2]);
        }
    }
}

// =====================================================================================
//  COMPONENTES DE VISUALIZACION (membranas, optica)
// =====================================================================================

// Discos de membrana (indicadores visuales)
module membrane_indicators() {
    if (show_membranes) {
        for (i = [0 : n_stages - 1]) {
            cx = stage_x(i);
            color("RoyalBlue", 0.5)
                translate([cx, channel_center_y, membrane_seat_z])
                    cylinder(d = membrane_diameter - 0.5, h = 0.2);
        }
    }
}

// LEDs (indicadores visuales)
module led_indicators() {
    if (show_optics) {
        for (i = [0 : n_stages - 1]) {
            cx = stage_x(i);
            color("Purple", 0.7)
                translate([cx, 0.5, chamber_center_z])
                    rotate([-90, 0, 0])
                        cylinder(d = led_diameter - 0.5, h = led_mount_depth - 0.5);
        }
    }
}

// Detectores (indicadores visuales)
module detector_indicators() {
    if (show_optics) {
        for (i = [0 : n_stages - 1]) {
            cx = stage_x(i);
            color("Green", 0.7)
                translate([cx, total_width - 0.5, chamber_center_z])
                    rotate([90, 0, 0])
                        cylinder(d = detector_diameter - 0.5,
                                 h = detector_mount_depth - 0.5);
        }
    }
}

// Ventanas de cuarzo (indicadores visuales)
module window_indicators() {
    if (show_optics) {
        for (i = [0 : n_stages - 1]) {
            cx = stage_x(i);
            color("Cyan", 0.3)
                translate([cx, -window_seat_depth, chamber_center_z])
                    rotate([-90, 0, 0])
                        cylinder(d = window_diameter - 0.5, h = window_thickness);
        }
    }
}

// Conectores tubulares exteriores (visuales)
module inlet_connector_visual() {
    color("SteelBlue", 0.6)
        translate([-connector_length, channel_center_y,
                   channel_base_z + channel_height / 2])
            rotate([0, 90, 0])
                difference() {
                    cylinder(d = inlet_diameter + wall_thickness, h = connector_length);
                    translate([0, 0, -0.1])
                        cylinder(d = inlet_diameter, h = connector_length + 0.2);
                    // Collar
                    translate([0, 0, connector_length - 2])
                        difference() {
                            cylinder(d = inlet_diameter + wall_thickness * 2, h = 2);
                            translate([0, 0, -0.1])
                                cylinder(d = inlet_diameter, h = 2.2);
                        }
                }
}

module waste_connector_visual() {
    color("OrangeRed", 0.6)
        translate([body_length, channel_center_y,
                   channel_base_z + channel_height / 2])
            rotate([0, 90, 0])
                difference() {
                    cylinder(d = waste_diameter + wall_thickness, h = connector_length);
                    translate([0, 0, -0.1])
                        cylinder(d = waste_diameter, h = connector_length + 0.2);
                    // Collar
                    translate([0, 0, connector_length - 2])
                        difference() {
                            cylinder(d = waste_diameter + wall_thickness * 2, h = 2);
                            translate([0, 0, -0.1])
                                cylinder(d = waste_diameter, h = 2.2);
                        }
                }
}

// Conectores de salida por etapa (visuales)
module outlet_connectors_visual() {
    color("LimeGreen", 0.6)
        for (i = [0 : n_stages - 1]) {
            cx = stage_x(i);
            translate([cx, channel_center_y, -connector_length])
                difference() {
                    cylinder(d = outlet_diameter + wall_thickness, h = connector_length);
                    translate([0, 0, -0.1])
                        cylinder(d = outlet_diameter, h = connector_length + 0.2);
                }
        }
}

// =====================================================================================
//  ENSAMBLAJE FINAL
// =====================================================================================

module classifier_assembly() {
    // Desplazamiento para vista explosionada
    explode_z = exploded_view ? 15 : 0;

    // Cuerpo principal
    color("LightGray", 0.9)
        classifier_body();

    // Tapa superior removible
    color("Silver", 0.8)
        translate([0, 0, explode_z])
            top_plate();

    // Indicadores de componentes
    translate([0, 0, explode_z > 0 ? -5 : 0]) {
        membrane_indicators();
    }

    led_indicators();
    detector_indicators();
    window_indicators();

    // Conectores exteriores
    inlet_connector_visual();
    waste_connector_visual();
    outlet_connectors_visual();
}

// =====================================================================================
//  RENDERIZADO
// =====================================================================================

// Renderizar el ensamblaje completo
classifier_assembly();

// =====================================================================================
//  INFORMACION DEL MODELO
// =====================================================================================

echo("=====================================================================================");
echo("  CLASIFICADOR DE QUANTUM DOTS - Laberinto Escalonado");
echo("=====================================================================================");
echo(str("  Numero de etapas:       ", n_stages));
echo(str("  Longitud del cuerpo:    ", body_length, " mm"));
echo(str("  Ancho total:            ", total_width, " mm"));
echo(str("  Altura total:           ", total_height + top_plate_thickness, " mm"));
echo(str("  Pitch entre etapas:     ", stage_pitch, " mm"));
echo(str("  Diametro membrana:      ", membrane_diameter, " mm"));
echo(str("  Profundidad camara:     ", chamber_depth, " mm"));
echo("-------------------------------------------------------------------------------------");
echo("  Posiciones de etapas (X centro):");
for (i = [0 : n_stages - 1]) {
    echo(str("    Etapa ", i + 1, ": X = ", stage_x(i), " mm"));
}
echo("-------------------------------------------------------------------------------------");
echo(str("  Canal: ", channel_width, " x ", channel_height, " mm"));
echo(str("  Ventana optica: diam ", window_diameter, " mm"));
echo(str("  LED: diam ", led_diameter, " mm,  Detector: diam ", detector_diameter, " mm"));
echo(str("  Tornillos: M", screw_diameter, " x ", screw_positions_per_stage, " por etapa"));
echo("=====================================================================================");
