#!/usr/bin/env python3
"""
===============================================================================
  SISTEMA DE PRODUCCION CONTINUA: REACTOR DBD + CLASIFICADOR EN CASCADA
  Integracion inline para operacion al mismo flujo
===============================================================================

  Problema:
    El reactor DBD produce ~50 mg/h de CQDs a 5 mL/min, pero el clasificador
    batch procesa ~0.019 mL/min efectivos (263x mas lento). No existe conexion
    de codigo entre ambos.

  Solucion:
    En lugar de recircular N veces por 1 camara (acumula calor, degrada Soret),
    usar N camaras fisicas en serie. Cada camara tiene estado termico
    independiente (fresco) -> sin degradacion acumulada.

  Arquitectura:

    REACTOR -> [PF] -> [S1] -> [S2] -> [S3] -> [R1] -> [R2] -> [R3] -> PRODUCTO
                         |        |        |
                         v        v        v
                       waste    waste    waste -> [WR1] -> [WR2] -> merge

  Variables de optimizacion:
    - flow_ml_min: caudal compartido reactor-clasificador
    - stage_volume_uL: volumen de cada camara de clasificacion

  USO:
    python continuous_production.py                           # Default: optimizar
    python continuous_production.py --optimize                # Grid search completo
    python continuous_production.py --flow 0.1 --stage-vol 5  # Evaluar punto
    python continuous_production.py --solvent chloroform      # Cambiar solvente
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import argparse
import sys
import os

# Importar desde modulos existentes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reactor_design import (
    ReactorDesigner, ReactorParameters, MATERIALS
)
from classifier_design import (
    MultiStageRecirculationSystem, ClassifierParameters,
    RecirculationStageConfig, RecirculationStageType,
    ChamberEnergyState, SolventProperties,
    SOLVENTS, SOLVENT_WATER, SOLVENT_CHLOROFORM,
    DENSITY_CQD, BOLTZMANN,
)


# =============================================================================
#  CONSTANTES DE CONVERSION
# =============================================================================

# CQD tipico: d=3nm, rho=1800 kg/m3
# Volumen = (4/3)*pi*(1.5e-9)^3 = 1.414e-26 m^3
# Masa = 1800 * 1.414e-26 = 2.545e-23 kg = 2.545e-20 g
CQD_DIAMETER_M = 3e-9
CQD_VOLUME_M3 = (4.0 / 3.0) * np.pi * (CQD_DIAMETER_M / 2.0) ** 3
CQD_MASS_KG = DENSITY_CQD * CQD_VOLUME_M3
CQD_MASS_G = CQD_MASS_KG * 1e3
PARTICLES_PER_MG = 1e-3 / CQD_MASS_KG  # ~3.93e16


@dataclass
class CascadeResult:
    """Resultado de la cascada de clasificacion"""
    # Particulas
    n_qdots_in: float
    n_nonqdots_in: float
    n_qdots_out: float
    n_nonqdots_out: float
    # Metricas
    recovery: float          # QDots salida / QDots entrada
    purity: float            # QDots / total en salida
    # Energia y termica
    total_energy_J: float
    max_dT_K: float
    max_T_C: float
    # Detalle por etapa
    stage_details: list
    # Waste recapture
    waste_recovered_qdots: float
    waste_recovered_nonqdots: float


@dataclass
class ProductionResult:
    """Resultado completo de produccion continua"""
    # Parametros de entrada
    flow_ml_min: float
    stage_volume_uL: float
    solvent_name: str
    # Reactor
    reactor_production_mg_h: float
    reactor_concentration_mg_ml: float
    reactor_wavelength_nm: float
    reactor_size_nm: float
    reactor_residence_time_s: float
    # Cascada
    cascade_recovery: float
    cascade_purity: float
    cascade_energy_J: float
    cascade_max_T_C: float
    cascade_stages: int
    # Produccion neta
    purified_mg_h: float
    purified_particles_per_s: float
    energy_per_mg_J: float
    # Figure of Merit
    figure_of_merit: float
    # Comparacion con batch
    batch_recovery: float
    batch_purity: float
    cascade_advantage: float  # cascade_recovery / batch_recovery


class ContinuousProductionSystem:
    """
    Sistema integrado de produccion continua: reactor DBD + cascada inline.

    En lugar de recircular N veces por 1 camara (batch, acumula calor),
    usa N camaras fisicas en serie. Cada camara tiene estado termico
    independiente -> sin degradacion por calor acumulado.
    """

    def __init__(self, solvent_name: str = 'water',
                 sep_stages: int = 3, ref_stages: int = 3,
                 waste_recapture_stages: int = 2):
        self.solvent_name = solvent_name
        self.solvent = SOLVENTS.get(solvent_name, SOLVENT_WATER)
        self.sep_stages = sep_stages
        self.ref_stages = ref_stages
        self.waste_recapture_stages = waste_recapture_stages

    # =========================================================================
    #  REACTOR
    # =========================================================================

    def run_reactor_at_flow(self, flow_ml_min: float) -> Dict:
        """Evalua reactor DBD a flujo dado via ReactorDesigner"""
        params = ReactorParameters(liquid_flow_ml_min=flow_ml_min)
        designer = ReactorDesigner(params)

        geometry = designer.calculate_geometry()
        flow = designer.calculate_flow_dynamics(geometry)
        electrical = designer.calculate_electrical(geometry)
        production = designer.calculate_production(geometry, flow, electrical)
        thermal = designer.calculate_thermal(electrical)

        return {
            'production_mg_h': production['production_mg_h'],
            'concentration_mg_ml': production['concentration_mg_ml'],
            'wavelength_nm': production['estimated_wavelength_nm'],
            'size_nm': production['estimated_size_nm'],
            'monodispersity': production['monodispersity_index'],
            'residence_time_s': flow['residence_time_s'],
            'energy_density_j_ml': electrical['energy_density_j_ml'],
            'power_w': electrical['power_w'],
            'temp_rise_c': thermal['estimated_temp_rise_c'],
            'in_spec': production['in_spec'],
        }

    # =========================================================================
    #  CASCADA INLINE (una camara = una pasada con estado termico fresco)
    # =========================================================================

    def _make_classifier_system(self, stage_volume_uL: float,
                                 flow_ml_min: float
                                 ) -> MultiStageRecirculationSystem:
        """
        Crea un MultiStageRecirculationSystem configurado para una camara
        individual de la cascada, con volumen y flujo fijos.
        """
        params = ClassifierParameters(
            flow_rate_ml_min=flow_ml_min,
            laser_power_mw=500.0,
        )
        system = MultiStageRecirculationSystem(
            base_params=params,
            solvent=self.solvent,
            microchannel_volume_uL=stage_volume_uL,
        )
        # Override el auto-adjust del flow rate: en cascada el flujo es fijo
        system.base_params.flow_rate_ml_min = flow_ml_min
        return system

    def run_inline_stage(self, stage_type: str, power_mw: float,
                          n_qdots: float, n_nonqdots: float,
                          stage_volume_uL: float, flow_ml_min: float) -> Dict:
        """
        Una camara en serie con estado termico FRESCO.

        A diferencia del batch donde el fluido se recalienta en cada pasada,
        aqui cada camara empieza a T_ref -> maximo coeficiente de Soret.
        """
        system = self._make_classifier_system(stage_volume_uL, flow_ml_min)

        # Potencia segura para este solvente y volumen
        safe_power = system._calculate_safe_power(power_mw)

        # Estado termico fresco (T = T_ref)
        fresh_state = ChamberEnergyState(
            T_fluid_K=system.T_REF,
            T_fluid_initial_K=system.T_REF,
        )

        # Ejecutar modelo de separacion para 1 pasada
        sep = system._calculate_single_separation_pass(
            safe_power, fresh_state, 405.0)

        P_qdot = sep['P_qdot_capture']
        P_nonqdot = sep['P_nonqdot_drag']

        # Para refinamiento: poder reducido -> menor captura pero mas selectivo
        if stage_type == 'refinement':
            # El refinamiento aplica selectividad por tamano adicional
            # QDots del tamano objetivo: alta captura
            # QDots fuera de tamano: menor captura (factor 0.6)
            P_nonqdot *= 0.5  # Aun menos arrastre

        # Aplicar separacion
        qdots_collected = n_qdots * P_qdot
        nonqdots_collected = n_nonqdots * P_nonqdot
        qdots_waste = n_qdots * (1.0 - P_qdot)
        nonqdots_waste = n_nonqdots * (1.0 - P_nonqdot)

        # Energia de esta pasada
        geometry = system._base_designer.calculate_zone_geometry()
        t_res = system._get_residence_time_s(geometry['zone_residence_time_s'])
        energy_J = system._calculate_energy_per_pass(safe_power, t_res)

        # Calentamiento de esta pasada (desde fresco)
        fluid_vol_ml = system._get_effective_volume_ml(geometry['total_volume_ml'])
        dT_K = system._calculate_fluid_heating(energy_J, fluid_vol_ml)

        return {
            'qdots_collected': qdots_collected,
            'nonqdots_collected': nonqdots_collected,
            'qdots_waste': qdots_waste,
            'nonqdots_waste': nonqdots_waste,
            'P_qdot': P_qdot,
            'P_nonqdot': P_nonqdot,
            'power_mw': safe_power,
            'energy_J': energy_J,
            'dT_K': dT_K,
            'T_max_C': 25.0 + dT_K,
            'residence_time_s': t_res,
            'stage_type': stage_type,
            'degradation': sep['degradation_factor'],
        }

    def run_cascade(self, flow_ml_min: float, stage_volume_uL: float,
                     n_qdots_in: float, n_nonqdots_in: float) -> CascadeResult:
        """
        Cascada completa: PF + N_sep + waste recapture + N_ref

        Cada camara tiene estado termico independiente (fresco).
        """
        stage_details = []
        total_energy = 0.0
        max_dT = 0.0

        current_qdots = n_qdots_in
        current_nonqdots = n_nonqdots_in
        initial_qdots = n_qdots_in

        # ===================== PRE-FILTRO =====================
        # Modelo simple: retiene particulas en rango 1-6 nm
        pf_efficiency = 0.95
        pf_leak = 0.005  # 0.5% de fuera-de-rango se cuela

        # Fracciones entrantes (del reactor)
        # QDots: estan en rango, pasan con alta eficiencia
        # No-QDots en rango de tamano: ~10% del total no-QDot
        # No-QDots fuera de rango: 90% del total no-QDot
        nonqdot_in_range_frac = 0.10
        n_nonqdots_in_range = current_nonqdots * nonqdot_in_range_frac
        n_nonqdots_out_range = current_nonqdots * (1.0 - nonqdot_in_range_frac)

        pf_qdots_out = current_qdots * pf_efficiency
        pf_nonqdots_out = n_nonqdots_in_range * pf_efficiency + n_nonqdots_out_range * pf_leak

        stage_details.append({
            'name': 'PRE-FILTRO',
            'qdots_in': current_qdots,
            'nonqdots_in': current_nonqdots,
            'qdots_out': pf_qdots_out,
            'nonqdots_out': pf_nonqdots_out,
            'energy_J': 0.0,
            'dT_K': 0.0,
        })

        current_qdots = pf_qdots_out
        current_nonqdots = pf_nonqdots_out

        # ===================== SEPARACION (N camaras en serie) =====================
        waste_qdots_total = 0.0
        waste_nonqdots_total = 0.0

        for i in range(self.sep_stages):
            result = self.run_inline_stage(
                'separation', 500.0,
                current_qdots, current_nonqdots,
                stage_volume_uL, flow_ml_min)

            waste_qdots_total += result['qdots_waste']
            waste_nonqdots_total += result['nonqdots_waste']
            total_energy += result['energy_J']
            max_dT = max(max_dT, result['dT_K'])

            stage_details.append({
                'name': f'SEP-{i+1}',
                'qdots_in': current_qdots,
                'nonqdots_in': current_nonqdots,
                'qdots_out': result['qdots_collected'],
                'nonqdots_out': result['nonqdots_collected'],
                'P_qdot': result['P_qdot'],
                'P_nonqdot': result['P_nonqdot'],
                'power_mw': result['power_mw'],
                'energy_J': result['energy_J'],
                'dT_K': result['dT_K'],
            })

            current_qdots = result['qdots_collected']
            current_nonqdots = result['nonqdots_collected']

        # ===================== WASTE RECAPTURE =====================
        waste_recovered_qdots = 0.0
        waste_recovered_nonqdots = 0.0

        if self.waste_recapture_stages > 0 and waste_qdots_total > 0:
            wr_qdots = waste_qdots_total
            wr_nonqdots = waste_nonqdots_total

            for i in range(self.waste_recapture_stages):
                # Potencia reducida (0.6x), decreciente por ronda
                wr_power = 500.0 * 0.6 * (0.85 ** i)
                wr_result = self.run_inline_stage(
                    'separation', wr_power,
                    wr_qdots, wr_nonqdots,
                    stage_volume_uL, flow_ml_min)

                waste_recovered_qdots += wr_result['qdots_collected']
                waste_recovered_nonqdots += wr_result['nonqdots_collected']
                total_energy += wr_result['energy_J']
                max_dT = max(max_dT, wr_result['dT_K'])

                stage_details.append({
                    'name': f'WR-{i+1}',
                    'qdots_in': wr_qdots,
                    'nonqdots_in': wr_nonqdots,
                    'qdots_out': wr_result['qdots_collected'],
                    'nonqdots_out': wr_result['nonqdots_collected'],
                    'power_mw': wr_result['power_mw'],
                    'energy_J': wr_result['energy_J'],
                    'dT_K': wr_result['dT_K'],
                })

                wr_qdots = wr_result['qdots_waste']
                wr_nonqdots = wr_result['nonqdots_waste']

            # Merge con stream principal
            current_qdots += waste_recovered_qdots
            current_nonqdots += waste_recovered_nonqdots

        # ===================== REFINAMIENTO (M camaras en serie) =====================
        for i in range(self.ref_stages):
            # Potencia reducida (0.25x) para selectividad por tamano
            result = self.run_inline_stage(
                'refinement', 500.0 * 0.25,
                current_qdots, current_nonqdots,
                stage_volume_uL, flow_ml_min)

            total_energy += result['energy_J']
            max_dT = max(max_dT, result['dT_K'])

            stage_details.append({
                'name': f'REF-{i+1}',
                'qdots_in': current_qdots,
                'nonqdots_in': current_nonqdots,
                'qdots_out': result['qdots_collected'],
                'nonqdots_out': result['nonqdots_collected'],
                'P_qdot': result['P_qdot'],
                'P_nonqdot': result['P_nonqdot'],
                'power_mw': result['power_mw'],
                'energy_J': result['energy_J'],
                'dT_K': result['dT_K'],
            })

            current_qdots = result['qdots_collected']
            current_nonqdots = result['nonqdots_collected']

        # ===================== METRICAS FINALES =====================
        total_out = current_qdots + current_nonqdots
        purity = current_qdots / max(total_out, 1e-10)
        recovery = current_qdots / max(initial_qdots, 1e-10)

        return CascadeResult(
            n_qdots_in=n_qdots_in,
            n_nonqdots_in=n_nonqdots_in,
            n_qdots_out=current_qdots,
            n_nonqdots_out=current_nonqdots,
            recovery=recovery,
            purity=purity,
            total_energy_J=total_energy,
            max_dT_K=max_dT,
            max_T_C=25.0 + max_dT,
            stage_details=stage_details,
            waste_recovered_qdots=waste_recovered_qdots,
            waste_recovered_nonqdots=waste_recovered_nonqdots,
        )

    # =========================================================================
    #  BATCH REFERENCE (para comparacion)
    # =========================================================================

    def run_batch_reference(self, stage_volume_uL: float,
                             n_qdots_in: float, n_nonqdots_in: float
                             ) -> Dict:
        """
        Ejecuta el clasificador batch (recirculacion por 1 camara) como
        referencia para comparar con la cascada inline.
        """
        # Batch: microcanal con auto-ajuste de flujo
        params = ClassifierParameters(laser_power_mw=500.0)
        system = MultiStageRecirculationSystem(
            base_params=params,
            solvent=self.solvent,
            microchannel_volume_uL=stage_volume_uL,
        )

        # Usar las N pasadas equivalentes a la cascada
        total_passes = self.sep_stages + self.ref_stages
        n_total = n_qdots_in + n_nonqdots_in
        qdot_frac = n_qdots_in / max(n_total, 1e-10)

        sep_config = RecirculationStageConfig(
            stage_type=RecirculationStageType.QDOT_SEPARATION,
            n_passes=total_passes,
            initial_laser_power_mw=500.0,
            cooling_time_s=5.0,
            cooling_efficiency=0.7,
            target_wavelength_nm=405.0,
            target_purity=0.99,
        )

        results = system.run_qdot_separation(sep_config, n_qdots_in, n_nonqdots_in)
        if results:
            last = results[-1]
            total_out = last.n_qdots_collected + last.n_nonqdots_collected
            return {
                'recovery': last.n_qdots_collected / max(n_qdots_in, 1e-10),
                'purity': last.n_qdots_collected / max(total_out, 1e-10),
                'max_T_K': last.energy_state.T_fluid_K,
                'dT_accumulated_K': last.energy_state.dT_accumulated_K,
                'soret_factor': last.energy_state.soret_factor,
                'degradation': system._calculate_qdot_degradation(
                    last.energy_state.T_fluid_K),
            }

        return {'recovery': 0.0, 'purity': 0.0, 'max_T_K': 298.15,
                'dT_accumulated_K': 0.0, 'soret_factor': 1.0, 'degradation': 1.0}

    # =========================================================================
    #  EVALUACION COMPLETA
    # =========================================================================

    def evaluate(self, flow_ml_min: float, stage_volume_uL: float) -> ProductionResult:
        """Punto completo: reactor + cascada a (flow, volume)"""
        # 1. Reactor
        reactor = self.run_reactor_at_flow(flow_ml_min)

        # 2. Convertir concentracion a particulas/s
        # production_mg_h -> particulas/s
        production_particles_per_s = (reactor['production_mg_h'] / 3600.0 *
                                       1e-3 / CQD_MASS_KG)

        # Fraccion QDot en solucion post-reactor
        # Concentracion tipica: ~0.1-0.5 mg/mL de CQDs
        # Total particulas en solucion: mucho mas (debris, precursor no reaccionado)
        # Ratio QDot/total ~ concentracion / concentracion_total_solidos
        # Asumimos concentracion total solidos ~ 2 mg/mL (precursor + productos)
        total_solids_mg_ml = 2.0
        qdot_fraction = reactor['concentration_mg_ml'] / total_solids_mg_ml
        qdot_fraction = max(0.01, min(0.5, qdot_fraction))

        # Particulas por segundo entrando al clasificador
        total_particles_per_s = production_particles_per_s / qdot_fraction
        n_qdots_per_s = production_particles_per_s
        n_nonqdots_per_s = total_particles_per_s - n_qdots_per_s

        # Normalizar a 10000 particulas para la simulacion
        scale = 10000.0 / max(total_particles_per_s, 1e-10)
        n_qdots_sim = n_qdots_per_s * scale
        n_nonqdots_sim = n_nonqdots_per_s * scale

        # 3. Cascada
        cascade = self.run_cascade(
            flow_ml_min, stage_volume_uL, n_qdots_sim, n_nonqdots_sim)

        # 4. Batch reference
        batch = self.run_batch_reference(stage_volume_uL, n_qdots_sim, n_nonqdots_sim)

        # 5. Produccion neta
        purified_mg_h = reactor['production_mg_h'] * cascade.recovery
        purified_particles_per_s = production_particles_per_s * cascade.recovery

        # Energia por mg purificado
        # En operacion continua, cada camara consume P_laser continuamente.
        # cascade.total_energy_J es la energia depositada en el fluido por
        # un volumen de residencia. En flujo continuo la energia por hora es:
        #   E_h = sum(P_stage) * 3600
        # Estimamos P_stage desde E_stage / t_res de cada camara.
        if purified_mg_h > 0:
            total_power_W = 0.0
            for sd in cascade.stage_details:
                e = sd.get('energy_J', 0.0)
                # E = P * t_res * absorptance * fluid_fraction, pero E ya
                # esta calculado. Para la potencia continua simplemente:
                # En estado estacionario, un volumen de fluido tarda t_res
                # en atravesar cada camara, y absorbe E joules.
                # Potencia termica continua = E / t_res
                # (ya incluye absorptancia y fraccion)
                if e > 0:
                    # Usar potencia del laser como proxy directa
                    p_mw = sd.get('power_mw', 0.0)
                    total_power_W += p_mw * 1e-3
            energy_per_hour_J = total_power_W * 3600.0
            energy_per_mg = energy_per_hour_J / purified_mg_h
        else:
            energy_per_mg = float('inf')

        # 6. Figure of Merit
        # FoM = purified_mg_h * purity * safety * integrity
        safety = 1.0
        boiling_C = self.solvent.boiling_point_C
        if cascade.max_T_C > boiling_C - 10:
            safety = 0.0
        elif cascade.max_T_C > boiling_C - 20:
            safety = (boiling_C - 10 - cascade.max_T_C) / 10.0

        integrity = 1.0 if reactor['in_spec'] else 0.5
        cascade_advantage = cascade.recovery / max(batch['recovery'], 1e-10)

        fom = purified_mg_h * cascade.purity * safety * integrity

        return ProductionResult(
            flow_ml_min=flow_ml_min,
            stage_volume_uL=stage_volume_uL,
            solvent_name=self.solvent.name,
            reactor_production_mg_h=reactor['production_mg_h'],
            reactor_concentration_mg_ml=reactor['concentration_mg_ml'],
            reactor_wavelength_nm=reactor['wavelength_nm'],
            reactor_size_nm=reactor['size_nm'],
            reactor_residence_time_s=reactor['residence_time_s'],
            cascade_recovery=cascade.recovery,
            cascade_purity=cascade.purity,
            cascade_energy_J=cascade.total_energy_J,
            cascade_max_T_C=cascade.max_T_C,
            cascade_stages=(self.sep_stages + self.ref_stages +
                            self.waste_recapture_stages + 1),  # +1 for PF
            purified_mg_h=purified_mg_h,
            purified_particles_per_s=purified_particles_per_s,
            energy_per_mg_J=energy_per_mg,
            figure_of_merit=fom,
            batch_recovery=batch['recovery'],
            batch_purity=batch['purity'],
            cascade_advantage=cascade_advantage,
        )

    # =========================================================================
    #  OPTIMIZACION
    # =========================================================================

    def optimize(self) -> Dict:
        """Grid search 2D: flow_ml_min x stage_volume_uL"""
        flows = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        volumes = [1, 2, 3, 5, 10, 20, 50]

        results = []
        best = None
        best_fom = -1

        print(f"\n  Optimizando: {len(flows)} flujos x {len(volumes)} volumenes "
              f"= {len(flows)*len(volumes)} puntos")
        print(f"  Solvente: {self.solvent.name}")
        print(f"  Cascada: {self.sep_stages}S + {self.waste_recapture_stages}WR "
              f"+ {self.ref_stages}R = "
              f"{self.sep_stages + self.waste_recapture_stages + self.ref_stages + 1} camaras")
        print()

        for flow in flows:
            for vol in volumes:
                try:
                    r = self.evaluate(flow, vol)
                    results.append(r)
                    if r.figure_of_merit > best_fom:
                        best_fom = r.figure_of_merit
                        best = r
                except Exception as e:
                    print(f"  [WARN] flow={flow}, vol={vol}: {e}")

        # Ordenar por FoM descendente
        results.sort(key=lambda r: r.figure_of_merit, reverse=True)

        return {
            'all_results': results,
            'best': best,
            'top5': results[:5],
        }

    # =========================================================================
    #  REPORTES
    # =========================================================================

    def print_report(self, result: ProductionResult):
        """Reporte detallado: reactor | cascada | produccion neta"""
        r = result

        print("\n" + "=" * 90)
        print("  SISTEMA DE PRODUCCION CONTINUA: REACTOR DBD + CASCADA INLINE")
        print("=" * 90)

        # Parametros
        print(f"\n  {'PARAMETROS':=<88}")
        print(f"  Solvente:          {r.solvent_name}")
        print(f"  Flujo compartido:  {r.flow_ml_min:.3f} mL/min")
        print(f"  Vol. por camara:   {r.stage_volume_uL:.1f} uL")
        print(f"  Camaras totales:   {r.cascade_stages}")
        print(f"  Punto ebullicion:  {self.solvent.boiling_point_C:.0f} C")

        # Reactor
        print(f"\n  {'REACTOR DBD':=<88}")
        print(f"  Produccion bruta:  {r.reactor_production_mg_h:.2f} mg/h")
        print(f"  Concentracion:     {r.reactor_concentration_mg_ml:.4f} mg/mL")
        print(f"  Longitud de onda:  {r.reactor_wavelength_nm:.1f} nm "
              f"{'[EN SPEC]' if abs(r.reactor_wavelength_nm - 450) < 20 else '[FUERA SPEC]'}")
        print(f"  Tamano estimado:   {r.reactor_size_nm:.2f} nm")
        print(f"  Tiempo residencia: {r.reactor_residence_time_s:.2f} s")

        # Cascada
        print(f"\n  {'CASCADA INLINE':=<88}")
        print(f"  Recovery:          {r.cascade_recovery*100:.1f}%")
        print(f"  Pureza:            {r.cascade_purity*100:.1f}%")
        print(f"  T maxima:          {r.cascade_max_T_C:.1f} C "
              f"({'OK' if r.cascade_max_T_C < self.solvent.boiling_point_C - 10 else 'RIESGO'})")
        print(f"  Energia total:     {r.cascade_energy_J*1e3:.2f} mJ (por lote sim.)")

        # Comparacion batch vs cascade
        print(f"\n  {'BATCH vs CASCADE':=<88}")
        print(f"  Batch recovery:    {r.batch_recovery*100:.1f}%")
        print(f"  Batch purity:      {r.batch_purity*100:.1f}%")
        print(f"  Cascade recovery:  {r.cascade_recovery*100:.1f}%")
        print(f"  Cascade purity:    {r.cascade_purity*100:.1f}%")
        print(f"  Ventaja cascade:   {r.cascade_advantage:.2f}x recovery")
        if r.cascade_advantage > 1:
            print(f"  -> Cascade MEJOR:  sin degradacion termica acumulada")
        else:
            print(f"  -> Similar rendimiento (bajo calor acumulado en ambos)")

        # Produccion neta
        print(f"\n  {'PRODUCCION NETA':=<88}")
        print(f"  Purificado:        {r.purified_mg_h:.2f} mg/h")
        print(f"  Particulas/s:      {r.purified_particles_per_s:.2e}")
        print(f"  Energia/mg:        {r.energy_per_mg_J:.1f} J/mg")
        print(f"  Figure of Merit:   {r.figure_of_merit:.2f}")

        print("\n" + "=" * 90)

    def print_optimization_report(self, opt_results: Dict):
        """Reporte de optimizacion con top 5 y detalle del mejor"""
        top5 = opt_results['top5']
        best = opt_results['best']

        print("\n" + "=" * 90)
        print("  RESULTADOS DE OPTIMIZACION - TOP 5 CONFIGURACIONES")
        print("=" * 90)

        header = (f"  {'#':<4} {'Flow':<10} {'Vol(uL)':<10} {'Prod(mg/h)':<12} "
                  f"{'Purif(mg/h)':<13} {'Purity':<10} {'Recovery':<10} "
                  f"{'T_max(C)':<10} {'FoM':<10}")
        print(header)
        print(f"  {'-'*86}")

        for i, r in enumerate(top5):
            line = (f"  {i+1:<4} {r.flow_ml_min:<10.3f} {r.stage_volume_uL:<10.1f} "
                    f"{r.reactor_production_mg_h:<12.2f} {r.purified_mg_h:<13.2f} "
                    f"{r.cascade_purity*100:<9.1f}% {r.cascade_recovery*100:<9.1f}% "
                    f"{r.cascade_max_T_C:<10.1f} {r.figure_of_merit:<10.2f}")
            print(line)

        print(f"\n  {'='*86}")
        print(f"  MEJOR: flow={best.flow_ml_min:.3f} mL/min, "
              f"vol={best.stage_volume_uL:.1f} uL")
        print(f"  {'='*86}")

        # Reporte detallado del mejor
        self.print_report(best)

        # Heatmap ASCII
        self._print_heatmap(opt_results['all_results'])

    def _print_heatmap(self, results):
        """Heatmap ASCII de FoM vs flow y volume"""
        print(f"\n  {'HEATMAP: Figure of Merit':=<88}")

        # Agrupar por (flow, vol)
        flows = sorted(set(r.flow_ml_min for r in results))
        volumes = sorted(set(r.stage_volume_uL for r in results))
        grid = {}
        max_fom = max(r.figure_of_merit for r in results) if results else 1

        for r in results:
            grid[(r.flow_ml_min, r.stage_volume_uL)] = r.figure_of_merit

        # Header
        vol_header = "  Flow\\Vol  " + "".join(f"{v:>8.0f}" for v in volumes)
        print(vol_header)
        print(f"  {'-'*(11 + 8*len(volumes))}")

        chars = " ░▒▓█"
        for flow in flows:
            row = f"  {flow:>8.3f}  "
            for vol in volumes:
                fom = grid.get((flow, vol), 0)
                level = int(fom / max(max_fom, 1e-10) * (len(chars) - 1))
                level = min(level, len(chars) - 1)
                val = f"{fom:>7.1f}" if fom > 0 else "      -"
                row += f"{val} "
            print(row)

    def print_flow_diagram(self):
        """Diagrama ASCII del sistema integrado"""
        n_s = self.sep_stages
        n_r = self.ref_stages
        n_w = self.waste_recapture_stages

        print("\n" + "=" * 90)
        print("  DIAGRAMA DE FLUJO: PRODUCCION CONTINUA")
        print("=" * 90)

        # Linea principal
        stages_sep = " -> ".join(f"[S{i+1}]" for i in range(n_s))
        stages_ref = " -> ".join(f"[R{i+1}]" for i in range(n_r))
        stages_wr = " -> ".join(f"[WR{i+1}]" for i in range(n_w))

        print(f"\n  REACTOR -> [PF] -> {stages_sep} -> {stages_ref} -> PRODUCTO")

        # Waste lines
        if n_w > 0:
            # Calcular posiciones para waste arrows
            pf_width = 10  # "  [PF] -> "
            sep_width = 8 * n_s  # "[Sx] -> " per stage
            waste_start = 14  # offset

            waste_line1 = " " * waste_start
            waste_line2 = " " * waste_start
            for i in range(n_s):
                pos = 8 * i + 4
                waste_line1 += " " * (pos - len(waste_line1) + waste_start) + "|"
                waste_line2 += " " * (pos - len(waste_line2) + waste_start) + "v"

            print(f"  {'':>12}" + "".join(
                f"{'|':>8}" if i < n_s else "" for i in range(n_s)))
            print(f"  {'':>12}" + "".join(
                f"{'v':>8}" if i < n_s else "" for i in range(n_s)))
            print(f"  {'':>10}" + "".join(
                f"{'waste':>8}" if i < n_s else "" for i in range(n_s))
                  + f" -> {stages_wr} -> merge ^")

        print(f"""
  Leyenda:
    [PF]  = Pre-Filtro (DLD/membrana, sin energia)
    [Sx]  = Camara de Separacion #{'{x}'} (laser optotermico, estado fresco)
    [Rx]  = Camara de Refinamiento #{'{x}'} (laser 0.25x, selectividad tamano)
    [WRx] = Waste Recapture #{'{x}'} (laser 0.6x sobre waste combinado)

  Ventaja clave: cada camara opera con estado termico FRESCO (T=T_ref)
    -> Maximo coeficiente de Soret
    -> Sin degradacion termica de QDots
    -> Potencia no limitada por calor acumulado
""")
        print("=" * 90)


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Produccion Continua: Reactor DBD + Clasificador en Cascada')
    parser.add_argument('--optimize', action='store_true',
                        help='Grid search completo (default si no se especifica --flow)')
    parser.add_argument('--flow', type=float, default=None,
                        help='Flujo en mL/min (evaluar punto especifico)')
    parser.add_argument('--stage-vol', type=float, default=5.0,
                        help='Volumen por camara en uL (default: 5)')
    parser.add_argument('--solvent', type=str, default='water',
                        choices=list(SOLVENTS.keys()),
                        help='Solvente (default: water)')
    parser.add_argument('--sep-stages', type=int, default=3,
                        help='Camaras de separacion (default: 3)')
    parser.add_argument('--ref-stages', type=int, default=3,
                        help='Camaras de refinamiento (default: 3)')
    parser.add_argument('--wr-stages', type=int, default=2,
                        help='Camaras de waste recapture (default: 2)')
    parser.add_argument('--diagram', action='store_true',
                        help='Mostrar diagrama de flujo')

    args = parser.parse_args()

    system = ContinuousProductionSystem(
        solvent_name=args.solvent,
        sep_stages=args.sep_stages,
        ref_stages=args.ref_stages,
        waste_recapture_stages=args.wr_stages,
    )

    # Diagrama
    if args.diagram:
        system.print_flow_diagram()

    if args.flow is not None:
        # Evaluar punto especifico
        result = system.evaluate(args.flow, args.stage_vol)
        system.print_report(result)
        system.print_flow_diagram()
    else:
        # Optimizar (default)
        opt = system.optimize()
        system.print_optimization_report(opt)
        system.print_flow_diagram()


if __name__ == '__main__':
    main()
