#!/usr/bin/env python3
"""
===============================================================================
  CLASIFICADORES PARALELOS + RECOVERY MEJORADA + RECIRCULACION WASTE
  Integracion milireactor DBD + N clasificadores opticos + waste loop
===============================================================================

  Problema:
    El milireactor optimizado (multi-channel 8x300mm, TiO2) produce
    ~505 mg/h a 5 mL/min. Pero el clasificador optico necesita 0.1 mL/min
    para funcionar (52% recovery). A 5 mL/min el recovery es 0%.

  Tres problemas resueltos:
    1. THROUGHPUT MISMATCH: reactor 50x mas rapido que clasificador
       -> numbering-up de N clasificadores en paralelo
    2. RECOVERY BAJA: 52% -> mejorar con camaras mas grandes, mas potencia
       -> stage_vol 20 uL, laser 800 mW, 4 sep + 2 ref + 3 WR
    3. WASTE PERDIDO: 48% QDots se pierden
       -> recircular 80% del waste al inlet del reactor

  Arquitectura:

    REACTOR (MC 8ch, TiO2, 5 mL/min)
       |
       v
    MANIFOLD DISTRIBUIDOR
       |
       +---> [Clasificador 1] (0.1 mL/min) ---> PRODUCTO
       +---> [Clasificador 2] (0.1 mL/min) ---> PRODUCTO
       +---> ...
       +---> [Clasificador N] (0.1 mL/min) ---> PRODUCTO
       |                                    |
       |         WASTE MERGE <--------------+
       |              |
       |         80% recirculado
       |              |
       +<-------------+
       |
       20% -> fertilizante/descarte

  Estado estacionario con recirculacion:
    bruto = P_base / (1 - f_recirc * (1 - recovery))
    net   = recovery * bruto

  USO:
    python parallel_classifier.py                                    # Default
    python parallel_classifier.py --reactor-flow 5 --topology multi_channel
    python parallel_classifier.py --n-classifiers 50 --classifier-flow 0.1
    python parallel_classifier.py --optimize
    python parallel_classifier.py --stage-vol 20 --laser-power 800
    python parallel_classifier.py --no-recirculation
    python parallel_classifier.py --buffer
    python parallel_classifier.py --sweep-recovery
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from math import ceil
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reactor_scaleup import (
    MillimetricReactorDesigner, ScaledReactorParameters, ScaleTopology,
    ScaleupResult,
)
from continuous_production import (
    ContinuousProductionSystem, CascadeResult,
    CQD_MASS_KG, PARTICLES_PER_MG,
)
from classifier_design import SOLVENTS, SOLVENT_WATER


# =============================================================================
#  CONSTANTES DEL SISTEMA PARALELO
# =============================================================================

DEFAULT_CLASSIFIER_FLOW_ML_MIN = 0.1     # Flujo optimo por clasificador
DEFAULT_STAGE_VOLUME_UL = 20.0           # Camaras mas grandes (vs 5 uL original)
DEFAULT_LASER_POWER_MW = 800.0           # Mayor potencia (vs 500 mW)
DEFAULT_SEP_STAGES = 4                   # 4 etapas separacion (vs 3)
DEFAULT_REF_STAGES = 2                   # 2 refinamiento (vs 3 -> menos perdida)
DEFAULT_WR_STAGES = 3                    # 3 waste recapture (vs 2)
BUFFER_TANK_ML = 1000.0                  # Tanque buffer 1L entre reactor y clasificadores
MANIFOLD_DEAD_VOLUME_UL = 50.0           # Volumen muerto del manifold distribuidor

# Economicos
QDOT_PRICE_USD_PER_MG = 50.0            # Precio de mercado CQDs purificados
ENERGY_COST_USD_PER_KWH = 0.12          # Costo energia (promedio)
CLASSIFIER_UNIT_COST_USD = 200.0        # Costo estimado por clasificador (chip + optica)
REACTOR_BASE_COST_USD = 5000.0          # Costo base del milireactor


# =============================================================================
#  DATACLASSES
# =============================================================================

@dataclass
class ParallelClassifierConfig:
    """Configuracion del sistema de clasificadores en paralelo"""
    # Reactor
    reactor_topology: ScaleTopology = ScaleTopology.MULTI_CHANNEL
    reactor_flow_ml_min: float = 5.0
    reactor_n_channels: int = 8
    reactor_channel_length_mm: float = 300.0
    catalyst_type: Optional[str] = 'tio2_anatase'
    tio2_barrier: bool = True
    pulse_width_ns: float = 100.0
    # Clasificadores
    n_classifiers: Optional[int] = None   # None = auto-calcular
    classifier_flow_ml_min: float = DEFAULT_CLASSIFIER_FLOW_ML_MIN
    stage_volume_uL: float = DEFAULT_STAGE_VOLUME_UL
    laser_power_mw: float = DEFAULT_LASER_POWER_MW
    sep_stages: int = DEFAULT_SEP_STAGES
    ref_stages: int = DEFAULT_REF_STAGES
    wr_stages: int = DEFAULT_WR_STAGES
    solvent: str = 'water'
    # Buffer
    use_buffer: bool = False
    buffer_volume_ml: float = BUFFER_TANK_ML
    # Waste recirculation
    waste_recirculation: bool = True
    waste_recirculation_fraction: float = 0.80  # 80% del waste vuelve al reactor


@dataclass
class ParallelSystemResult:
    """Resultado completo del sistema paralelo"""
    # Config
    n_classifiers: int
    classifier_flow_ml_min: float
    stage_volume_uL: float
    laser_power_mw: float
    # Reactor
    reactor_production_mg_h: float
    reactor_wavelength_nm: float
    reactor_in_spec: bool
    reactor_power_w: float
    reactor_topology: str
    # Clasificacion
    gross_recovery: float              # Sin recirculacion
    waste_fraction: float              # Lo que se pierde
    cascade_purity: float              # Pureza por clasificador
    # Waste recirculation
    recirculated_mg_h: float           # Waste que vuelve al reactor
    recirculation_boost_factor: float  # Multiplicador por recirculacion
    waste_recirculation_active: bool
    # Neto
    net_production_mg_h: float         # Producto purificado final
    net_production_g_day: float
    purity: float
    # Energia
    classifier_power_w: float          # Total todos los clasificadores
    total_power_w: float
    energy_per_mg_purified_J: float
    kwh_per_g: float
    # Economico
    classifier_cost_usd: float
    reactor_cost_est_usd: float
    total_hw_cost_usd: float
    daily_product_value_usd: float     # A $50/mg
    daily_energy_cost_usd: float
    payback_hours: float
    # Buffer
    buffer_fill_time_min: float
    buffer_capacity_doses: float
    # Detalle cascada
    cascade_stages_total: int
    cascade_max_T_C: float


# =============================================================================
#  SISTEMA DE CLASIFICACION PARALELA
# =============================================================================

class ParallelClassificationSystem:
    """
    Sistema integrado: milireactor DBD + N clasificadores opticos en paralelo
    + recirculacion de waste al inlet del reactor.

    Resuelve el mismatch de flujo entre reactor (mL/min) y clasificador
    (0.1 mL/min) mediante numbering-up de clasificadores.

    La recirculacion de waste crea un loop cerrado donde los QDots no
    capturados vuelven al reactor, aumentando la produccion neta en
    estado estacionario.
    """

    def __init__(self, config: Optional[ParallelClassifierConfig] = None):
        self.config = config or ParallelClassifierConfig()
        self._n_classifiers = self._auto_calculate_n_classifiers()

    # =========================================================================
    #  CALCULO DE N CLASIFICADORES
    # =========================================================================

    def _auto_calculate_n_classifiers(self) -> int:
        """
        Calcula cuantos clasificadores se necesitan para absorber
        el flujo total del reactor.

        N = ceil(reactor_flow / classifier_flow)
        """
        cfg = self.config
        if cfg.n_classifiers is not None:
            return cfg.n_classifiers
        return int(ceil(cfg.reactor_flow_ml_min / cfg.classifier_flow_ml_min))

    # =========================================================================
    #  REACTOR
    # =========================================================================

    def _run_reactor(self) -> ScaleupResult:
        """
        Evalua el milireactor con la topologia y parametros configurados.
        Retorna ScaleupResult con produccion, potencia, wavelength, etc.
        """
        cfg = self.config

        params = ScaledReactorParameters(
            topology=cfg.reactor_topology,
            n_channels=cfg.reactor_n_channels,
            channel_length_mm=cfg.reactor_channel_length_mm,
            liquid_flow_ml_min=cfg.reactor_flow_ml_min,
            catalyst_type=cfg.catalyst_type,
            tio2_barrier=cfg.tio2_barrier,
            pulse_width_ns=cfg.pulse_width_ns,
        )

        designer = MillimetricReactorDesigner(params)
        return designer.evaluate()

    # =========================================================================
    #  CLASIFICADOR INDIVIDUAL (con parametros mejorados)
    # =========================================================================

    def _run_single_classifier(self, n_qdots: float,
                                n_nonqdots: float) -> CascadeResult:
        """
        Ejecuta un clasificador individual con parametros mejorados.

        Crea un ContinuousProductionSystem y ejecuta la cascada con:
        - stage_volume_uL mayor (20 vs 5) -> mas tiempo de residencia
        - sep_stages, ref_stages, wr_stages ajustados
        - laser_power_mw mayor (800 vs 500) -> mas fuerza optica

        Para usar la potencia mejorada, llamamos run_inline_stage
        directamente en lugar de run_cascade (que hardcodea 500 mW).
        """
        cfg = self.config

        system = ContinuousProductionSystem(
            solvent_name=cfg.solvent,
            sep_stages=cfg.sep_stages,
            ref_stages=cfg.ref_stages,
            waste_recapture_stages=cfg.wr_stages,
        )

        return self._run_enhanced_cascade(
            system, cfg.classifier_flow_ml_min, cfg.stage_volume_uL,
            cfg.laser_power_mw, n_qdots, n_nonqdots)

    def _run_enhanced_cascade(self, system: ContinuousProductionSystem,
                               flow_ml_min: float, stage_volume_uL: float,
                               laser_power_mw: float,
                               n_qdots_in: float,
                               n_nonqdots_in: float) -> CascadeResult:
        """
        Cascada mejorada: misma estructura que ContinuousProductionSystem.run_cascade
        pero con potencia laser configurable.

        PF -> N_sep (laser_power) -> waste_recapture -> N_ref (laser*0.25)
        """
        stage_details = []
        total_energy = 0.0
        max_dT = 0.0

        current_qdots = n_qdots_in
        current_nonqdots = n_nonqdots_in
        initial_qdots = n_qdots_in

        # ===================== PRE-FILTRO =====================
        pf_efficiency = 0.95
        pf_leak = 0.005

        nonqdot_in_range_frac = 0.10
        n_nonqdots_in_range = current_nonqdots * nonqdot_in_range_frac
        n_nonqdots_out_range = current_nonqdots * (1.0 - nonqdot_in_range_frac)

        pf_qdots_out = current_qdots * pf_efficiency
        pf_nonqdots_out = (n_nonqdots_in_range * pf_efficiency +
                           n_nonqdots_out_range * pf_leak)

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

        # ===================== SEPARACION (N camaras, potencia mejorada) ====
        waste_qdots_total = 0.0
        waste_nonqdots_total = 0.0

        for i in range(system.sep_stages):
            result = system.run_inline_stage(
                'separation', laser_power_mw,
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

        if system.waste_recapture_stages > 0 and waste_qdots_total > 0:
            wr_qdots = waste_qdots_total
            wr_nonqdots = waste_nonqdots_total

            for i in range(system.waste_recapture_stages):
                wr_power = laser_power_mw * 0.6 * (0.85 ** i)
                wr_result = system.run_inline_stage(
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

            current_qdots += waste_recovered_qdots
            current_nonqdots += waste_recovered_nonqdots

        # ===================== REFINAMIENTO (potencia reducida) =====================
        for i in range(system.ref_stages):
            result = system.run_inline_stage(
                'refinement', laser_power_mw * 0.25,
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
    #  CLASIFICADORES EN PARALELO
    # =========================================================================

    def _run_parallel_classifiers(self, n_qdots_total: float,
                                   n_nonqdots_total: float
                                   ) -> Dict:
        """
        Divide particulas entre N clasificadores, ejecuta cada uno,
        merge outputs.

        Cada clasificador recibe 1/N de las particulas y opera al
        flujo optimo (0.1 mL/min).
        """
        n = self._n_classifiers
        n_qdots_per = n_qdots_total / n
        n_nonqdots_per = n_nonqdots_total / n

        # Ejecutar un clasificador representativo (todos son identicos)
        single_result = self._run_single_classifier(n_qdots_per, n_nonqdots_per)

        # Escalar a N clasificadores
        total_qdots_out = single_result.n_qdots_out * n
        total_nonqdots_out = single_result.n_nonqdots_out * n
        total_energy = single_result.total_energy_J * n

        # Waste: lo que no salio como producto
        waste_qdots = n_qdots_total - total_qdots_out
        waste_nonqdots = n_nonqdots_total - total_nonqdots_out

        return {
            'n_classifiers': n,
            'single_result': single_result,
            'total_qdots_out': total_qdots_out,
            'total_nonqdots_out': total_nonqdots_out,
            'total_energy_J': total_energy,
            'recovery': single_result.recovery,
            'purity': single_result.purity,
            'waste_qdots': max(0, waste_qdots),
            'waste_nonqdots': max(0, waste_nonqdots),
            'max_T_C': single_result.max_T_C,
            'cascade_stages': len(single_result.stage_details),
        }

    # =========================================================================
    #  RECIRCULACION DE WASTE
    # =========================================================================

    def _calculate_waste_recirculation(self, recovery: float,
                                        base_production_mg_h: float
                                        ) -> Dict:
        """
        Modelo de recirculacion en estado estacionario.

        El waste del clasificador contiene QDots no capturados + non-QDots.
        Una fraccion f del waste se recircula al inlet del reactor.
        En estado estacionario:

          bruto = P_base / (1 - f * (1 - recovery))
          net   = recovery * bruto
          boost = 1 / (1 - f * (1 - recovery))

        Esto funciona porque los QDots recirculados pasan por el reactor
        (no se degradan — son estables termicamente) y vuelven a entrar
        al clasificador.
        """
        cfg = self.config

        if not cfg.waste_recirculation:
            return {
                'boost_factor': 1.0,
                'bruto_mg_h': base_production_mg_h,
                'net_mg_h': base_production_mg_h * recovery,
                'recirculated_mg_h': 0.0,
                'waste_discarded_mg_h': base_production_mg_h * (1 - recovery),
                'active': False,
            }

        f = cfg.waste_recirculation_fraction
        denominator = 1.0 - f * (1.0 - recovery)
        # Proteger contra denominador <= 0 (imposible fisicamente
        # pero defensivo numericamente)
        denominator = max(denominator, 0.01)

        boost = 1.0 / denominator
        bruto = base_production_mg_h * boost
        net = recovery * bruto
        recirculated = f * (1.0 - recovery) * bruto
        discarded = (1.0 - f) * (1.0 - recovery) * bruto

        return {
            'boost_factor': boost,
            'bruto_mg_h': bruto,
            'net_mg_h': net,
            'recirculated_mg_h': recirculated,
            'waste_discarded_mg_h': discarded,
            'active': True,
        }

    # =========================================================================
    #  EVALUACION COMPLETA
    # =========================================================================

    def evaluate(self) -> ParallelSystemResult:
        """
        Pipeline completo:
          reactor -> split(N) -> N x classifier -> merge
                  -> waste_recirculation_loop -> result
        """
        cfg = self.config

        # 1. Reactor
        reactor = self._run_reactor()

        # 2. Convertir produccion a particulas para simulacion
        production_mg_h = reactor.production_mg_h
        production_particles_per_s = (production_mg_h / 3600.0 *
                                       1e-3 / CQD_MASS_KG)

        # Fraccion QDot en solucion post-reactor
        total_solids_mg_ml = 2.0
        qdot_fraction = reactor.concentration_mg_ml / total_solids_mg_ml
        qdot_fraction = max(0.01, min(0.5, qdot_fraction))

        total_particles_per_s = production_particles_per_s / qdot_fraction
        n_qdots_per_s = production_particles_per_s
        n_nonqdots_per_s = total_particles_per_s - n_qdots_per_s

        # Normalizar a 10000 particulas para simulacion
        scale = 10000.0 / max(total_particles_per_s, 1e-10)
        n_qdots_sim = n_qdots_per_s * scale
        n_nonqdots_sim = n_nonqdots_per_s * scale

        # 3. Clasificadores en paralelo
        parallel = self._run_parallel_classifiers(n_qdots_sim, n_nonqdots_sim)
        gross_recovery = parallel['recovery']
        cascade_purity = parallel['purity']

        # 4. Recirculacion de waste
        recirc = self._calculate_waste_recirculation(
            gross_recovery, production_mg_h)

        net_mg_h = recirc['net_mg_h']
        net_g_day = net_mg_h * 24.0 / 1000.0

        # 5. Energia
        # Potencia clasificadores: potencia continua de todos los lasers
        classifier_power_per = 0.0
        for sd in parallel['single_result'].stage_details:
            p_mw = sd.get('power_mw', 0.0)
            if p_mw > 0:
                classifier_power_per += p_mw * 1e-3  # W
        classifier_power_total = classifier_power_per * self._n_classifiers
        total_power = reactor.power_w + classifier_power_total

        if net_mg_h > 0:
            energy_per_mg = total_power * 3600.0 / net_mg_h  # J/mg
            kwh_per_g = total_power / 1000.0 / (net_g_day / 24.0) if net_g_day > 0 else float('inf')
        else:
            energy_per_mg = float('inf')
            kwh_per_g = float('inf')

        # 6. Economia
        classifier_cost = self._n_classifiers * CLASSIFIER_UNIT_COST_USD
        reactor_cost = REACTOR_BASE_COST_USD
        total_hw = classifier_cost + reactor_cost
        daily_value = net_g_day * 1000.0 * QDOT_PRICE_USD_PER_MG  # g -> mg -> USD
        daily_energy = total_power / 1000.0 * 24.0 * ENERGY_COST_USD_PER_KWH

        if daily_value - daily_energy > 0:
            payback_h = total_hw / ((daily_value - daily_energy) / 24.0)
        else:
            payback_h = float('inf')

        # 7. Buffer
        if cfg.use_buffer:
            buffer_fill_min = cfg.buffer_volume_ml / cfg.reactor_flow_ml_min
            buffer_doses = cfg.buffer_volume_ml / cfg.classifier_flow_ml_min
        else:
            buffer_fill_min = 0.0
            buffer_doses = 0.0

        return ParallelSystemResult(
            n_classifiers=self._n_classifiers,
            classifier_flow_ml_min=cfg.classifier_flow_ml_min,
            stage_volume_uL=cfg.stage_volume_uL,
            laser_power_mw=cfg.laser_power_mw,
            reactor_production_mg_h=production_mg_h,
            reactor_wavelength_nm=reactor.wavelength_nm,
            reactor_in_spec=reactor.in_spec,
            reactor_power_w=reactor.power_w,
            reactor_topology=reactor.topology,
            gross_recovery=gross_recovery,
            waste_fraction=1.0 - gross_recovery,
            cascade_purity=cascade_purity,
            recirculated_mg_h=recirc['recirculated_mg_h'],
            recirculation_boost_factor=recirc['boost_factor'],
            waste_recirculation_active=recirc['active'],
            net_production_mg_h=net_mg_h,
            net_production_g_day=net_g_day,
            purity=cascade_purity,
            classifier_power_w=classifier_power_total,
            total_power_w=total_power,
            energy_per_mg_purified_J=energy_per_mg,
            kwh_per_g=kwh_per_g,
            classifier_cost_usd=classifier_cost,
            reactor_cost_est_usd=reactor_cost,
            total_hw_cost_usd=classifier_cost + reactor_cost,
            daily_product_value_usd=daily_value,
            daily_energy_cost_usd=daily_energy,
            payback_hours=payback_h,
            buffer_fill_time_min=buffer_fill_min,
            buffer_capacity_doses=buffer_doses,
            cascade_stages_total=parallel['cascade_stages'],
            cascade_max_T_C=parallel['max_T_C'],
        )

    # =========================================================================
    #  OPTIMIZACION
    # =========================================================================

    def optimize(self) -> Dict:
        """
        Grid search sobre: n_classifiers, stage_volume_uL,
        laser_power_mw, classifier_flow_ml_min.
        """
        cfg = self.config

        n_classifiers_range = [25, 50, 75, 100]
        stage_vols = [5, 10, 20, 50]
        laser_powers = [500, 800, 1000]
        flows = [0.05, 0.1, 0.2]

        results = []
        best = None
        best_score = -1

        total = len(n_classifiers_range) * len(stage_vols) * len(laser_powers) * len(flows)
        print(f"\n  Optimizando sistema paralelo: {total} configuraciones...")
        print(f"  Reactor: {cfg.reactor_topology.value}, "
              f"{cfg.reactor_flow_ml_min} mL/min")
        print(f"  Recirculacion: {'SI' if cfg.waste_recirculation else 'NO'} "
              f"({cfg.waste_recirculation_fraction*100:.0f}%)")
        print()

        count = 0
        for n_cl in n_classifiers_range:
            for sv in stage_vols:
                for lp in laser_powers:
                    for fl in flows:
                        count += 1
                        try:
                            test_cfg = ParallelClassifierConfig(
                                reactor_topology=cfg.reactor_topology,
                                reactor_flow_ml_min=cfg.reactor_flow_ml_min,
                                reactor_n_channels=cfg.reactor_n_channels,
                                reactor_channel_length_mm=cfg.reactor_channel_length_mm,
                                catalyst_type=cfg.catalyst_type,
                                tio2_barrier=cfg.tio2_barrier,
                                pulse_width_ns=cfg.pulse_width_ns,
                                n_classifiers=n_cl,
                                classifier_flow_ml_min=fl,
                                stage_volume_uL=sv,
                                laser_power_mw=lp,
                                sep_stages=cfg.sep_stages,
                                ref_stages=cfg.ref_stages,
                                wr_stages=cfg.wr_stages,
                                solvent=cfg.solvent,
                                waste_recirculation=cfg.waste_recirculation,
                                waste_recirculation_fraction=cfg.waste_recirculation_fraction,
                            )
                            sys_test = ParallelClassificationSystem(test_cfg)
                            r = sys_test.evaluate()
                            results.append(r)

                            # Score: maximizar produccion neta * pureza,
                            # penalizar payback largo
                            payback_penalty = min(1.0, 10.0 / max(r.payback_hours, 0.01))
                            score = (r.net_production_mg_h *
                                     r.purity *
                                     payback_penalty)

                            if score > best_score:
                                best_score = score
                                best = r
                        except Exception as e:
                            if count <= 5:
                                print(f"  [WARN] n={n_cl}, sv={sv}, "
                                      f"lp={lp}, fl={fl}: {e}")

        results.sort(key=lambda r: r.net_production_mg_h * r.purity,
                      reverse=True)

        return {
            'all_results': results,
            'best': best,
            'top5': results[:5],
            'total_evaluated': count,
        }

    # =========================================================================
    #  SWEEP RECOVERY
    # =========================================================================

    def sweep_recovery(self) -> List[Dict]:
        """
        Sweep de recovery vs parametros de clasificador.
        Compara stage_volume_uL y laser_power_mw.
        """
        cfg = self.config
        results = []

        combos = [
            ('Original (5uL, 500mW, 3S/3R/2WR)', 5.0, 500.0, 3, 3, 2),
            ('Vol mejorado (20uL, 500mW, 3S/3R/2WR)', 20.0, 500.0, 3, 3, 2),
            ('Power mejorado (5uL, 800mW, 3S/3R/2WR)', 5.0, 800.0, 3, 3, 2),
            ('Stages mejorado (5uL, 500mW, 4S/2R/3WR)', 5.0, 500.0, 4, 2, 3),
            ('Todo mejorado (20uL, 800mW, 4S/2R/3WR)', 20.0, 800.0, 4, 2, 3),
        ]

        for label, sv, lp, ss, rs, ws in combos:
            test_cfg = ParallelClassifierConfig(
                reactor_topology=cfg.reactor_topology,
                reactor_flow_ml_min=cfg.reactor_flow_ml_min,
                reactor_n_channels=cfg.reactor_n_channels,
                reactor_channel_length_mm=cfg.reactor_channel_length_mm,
                catalyst_type=cfg.catalyst_type,
                tio2_barrier=cfg.tio2_barrier,
                n_classifiers=self._n_classifiers,
                classifier_flow_ml_min=cfg.classifier_flow_ml_min,
                stage_volume_uL=sv,
                laser_power_mw=lp,
                sep_stages=ss,
                ref_stages=rs,
                wr_stages=ws,
                solvent=cfg.solvent,
                waste_recirculation=cfg.waste_recirculation,
                waste_recirculation_fraction=cfg.waste_recirculation_fraction,
            )
            sys_test = ParallelClassificationSystem(test_cfg)
            r = sys_test.evaluate()
            results.append({
                'label': label,
                'stage_vol': sv,
                'laser_power': lp,
                'sep': ss,
                'ref': rs,
                'wr': ws,
                'recovery': r.gross_recovery,
                'purity': r.purity,
                'net_mg_h': r.net_production_mg_h,
                'boost': r.recirculation_boost_factor,
            })

        return results

    # =========================================================================
    #  REPORTES
    # =========================================================================

    def print_architecture_diagram(self):
        """Diagrama ASCII del sistema completo"""
        n = self._n_classifiers
        cfg = self.config

        print("\n" + "=" * 90)
        print("  ARQUITECTURA: MILIREACTOR + CLASIFICADORES PARALELOS + WASTE LOOP")
        print("=" * 90)

        print(f"""
  REACTOR DBD ({cfg.reactor_topology.value}, {cfg.reactor_flow_ml_min} mL/min)
  [TiO2 anatase barrier, plasma frio {cfg.pulse_width_ns}ns]
       |
       |  {cfg.reactor_flow_ml_min:.1f} mL/min bruto
       v""")

        if cfg.use_buffer:
            print(f"""  [BUFFER TANK {cfg.buffer_volume_ml:.0f} mL]
       |""")

        print(f"""  MANIFOLD DISTRIBUIDOR (dead vol: {MANIFOLD_DEAD_VOLUME_UL:.0f} uL)
       |
       +---> [Clasificador  1] ({cfg.classifier_flow_ml_min} mL/min, {cfg.laser_power_mw:.0f}mW) --+
       +---> [Clasificador  2] ({cfg.classifier_flow_ml_min} mL/min, {cfg.laser_power_mw:.0f}mW) --+
       +---> [Clasificador  3]                                          --+""")

        if n > 6:
            print(f"       +---> [    ...     ]  ({n-4} clasificadores mas)             --+")
        elif n > 4:
            for i in range(3, n - 1):
                print(f"       +---> [Clasificador {i+1:2d}]"
                      f"                                          --+")

        print(f"       +---> [Clasificador {n:2d}]"
              f"                                          --+")

        print(f"""       |                                                           |
       |                                                           v
       |                                                     MERGE PRODUCTO
       |                                                           |
       |                                                     WASTE MERGE""")

        if cfg.waste_recirculation:
            pct = cfg.waste_recirculation_fraction * 100
            print(f"""       |                                                      /         \\
       |                                              {pct:.0f}% recirc    {100-pct:.0f}% descarte
       |                                                    |            |
       +<---------------------------------------------------+     fertilizante
       |
       v  (loop estacionario)""")
        else:
            print(f"""       |                                                           |
       |                                                      100% descarte""")

        # Detalle del clasificador individual
        n_s = cfg.sep_stages
        n_r = cfg.ref_stages
        n_w = cfg.wr_stages
        total_stages = 1 + n_s + n_w + n_r  # PF + sep + wr + ref

        stages_sep = " -> ".join(f"[S{i+1}]" for i in range(n_s))
        stages_ref = " -> ".join(f"[R{i+1}]" for i in range(n_r))
        stages_wr = " -> ".join(f"[WR{i+1}]" for i in range(n_w))

        print(f"""
  DETALLE DE CADA CLASIFICADOR ({total_stages} camaras):

    [PF] -> {stages_sep} -> {stages_ref} -> PRODUCTO
                    {'|  ' * n_s}
                    {'v  ' * n_s}
                   {'waste ' * min(n_s, 3)}-> {stages_wr} -> merge ^

    PF:  Pre-filtro (DLD/membrana)           | 0 mW
    S:   Separacion ({cfg.laser_power_mw:.0f} mW, {cfg.stage_volume_uL:.0f} uL)    | t_res 4x mayor
    WR:  Waste Recapture ({cfg.laser_power_mw*0.6:.0f} mW)       | recupera QDots del waste
    R:   Refinamiento ({cfg.laser_power_mw*0.25:.0f} mW)          | selectividad por tamano
""")
        print("=" * 90)

    def print_report(self, result: ParallelSystemResult):
        """Reporte completo del sistema paralelo"""
        r = result
        cfg = self.config

        print("\n" + "=" * 90)
        print("  SISTEMA PARALELO: MILIREACTOR + N CLASIFICADORES + WASTE RECIRCULATION")
        print("=" * 90)

        # Reactor
        print(f"\n  {'REACTOR':=<88}")
        print(f"  Topologia:         {r.reactor_topology}")
        print(f"  Flujo reactor:     {cfg.reactor_flow_ml_min:.1f} mL/min")
        print(f"  Produccion bruta:  {r.reactor_production_mg_h:.1f} mg/h")
        print(f"  Longitud de onda:  {r.reactor_wavelength_nm:.1f} nm "
              f"{'[EN SPEC]' if r.reactor_in_spec else '[FUERA SPEC]'}")
        print(f"  Potencia reactor:  {r.reactor_power_w:.1f} W")
        print(f"  Catalizador:       {cfg.catalyst_type or 'ninguno'}"
              f"{' (barrera TiO2)' if cfg.tio2_barrier else ''}")
        print(f"  Plasma:            frio (pulso {cfg.pulse_width_ns:.0f} ns)")

        # Clasificadores
        print(f"\n  {'CLASIFICADORES PARALELOS':=<88}")
        print(f"  N clasificadores:  {r.n_classifiers}")
        print(f"  Flujo/clasificador:{r.classifier_flow_ml_min:.3f} mL/min")
        print(f"  Flujo total:       {r.n_classifiers * r.classifier_flow_ml_min:.1f} mL/min "
              f"(reactor: {cfg.reactor_flow_ml_min:.1f} mL/min)")
        print(f"  Vol. camara:       {r.stage_volume_uL:.0f} uL")
        print(f"  Laser potencia:    {r.laser_power_mw:.0f} mW")
        print(f"  Cascada:           {cfg.sep_stages}S + {cfg.wr_stages}WR + "
              f"{cfg.ref_stages}R = {r.cascade_stages_total} camaras/clasificador")
        print(f"  T maxima camara:   {r.cascade_max_T_C:.1f} C")

        # Recovery
        print(f"\n  {'RECOVERY & PUREZA':=<88}")
        print(f"  Gross recovery:    {r.gross_recovery*100:.1f}%")
        print(f"  Pureza producto:   {r.purity*100:.2f}%")
        print(f"  Waste fraccion:    {r.waste_fraction*100:.1f}%")

        # Waste recirculation
        print(f"\n  {'WASTE RECIRCULATION':=<88}")
        if r.waste_recirculation_active:
            print(f"  Estado:            ACTIVO")
            print(f"  Fraccion recirc:   {cfg.waste_recirculation_fraction*100:.0f}%")
            print(f"  Boost factor:      {r.recirculation_boost_factor:.3f}x")
            print(f"  Recirculado:       {r.recirculated_mg_h:.1f} mg/h")
            print(f"  Formula:")
            print(f"    boost = 1 / (1 - {cfg.waste_recirculation_fraction:.2f} "
                  f"* (1 - {r.gross_recovery:.3f})) = "
                  f"{r.recirculation_boost_factor:.3f}")
            print(f"    net = {r.reactor_production_mg_h:.1f} * {r.gross_recovery:.3f} "
                  f"* {r.recirculation_boost_factor:.3f} = "
                  f"{r.net_production_mg_h:.1f} mg/h")
        else:
            print(f"  Estado:            DESACTIVADO")
            print(f"  Net = bruto * recovery = "
                  f"{r.reactor_production_mg_h:.1f} * {r.gross_recovery:.3f} = "
                  f"{r.net_production_mg_h:.1f} mg/h")

        # Produccion neta
        print(f"\n  {'PRODUCCION NETA':=<88}")
        print(f"  Purificado:        {r.net_production_mg_h:.1f} mg/h")
        print(f"  Diario:            {r.net_production_g_day:.2f} g/dia")
        print(f"  Pureza:            {r.purity*100:.2f}%")

        # Energia
        print(f"\n  {'ENERGIA':=<88}")
        print(f"  Reactor:           {r.reactor_power_w:.1f} W")
        print(f"  Clasificadores:    {r.classifier_power_w:.1f} W "
              f"({r.n_classifiers} x {r.classifier_power_w/max(r.n_classifiers,1):.2f} W)")
        print(f"  Total:             {r.total_power_w:.1f} W")
        print(f"  Energia/mg:        {r.energy_per_mg_purified_J:.1f} J/mg")
        print(f"  kWh/g:             {r.kwh_per_g:.2f}")

        # Economia
        print(f"\n  {'ECONOMIA':=<88}")
        print(f"  Costo clasificadores: ${r.classifier_cost_usd:,.0f} "
              f"({r.n_classifiers} x ${CLASSIFIER_UNIT_COST_USD:.0f})")
        print(f"  Costo reactor:     ${r.reactor_cost_est_usd:,.0f}")
        print(f"  Total HW:          ${r.total_hw_cost_usd:,.0f}")
        print(f"  Valor producto/dia:${r.daily_product_value_usd:,.0f} "
              f"(@${QDOT_PRICE_USD_PER_MG}/mg)")
        print(f"  Costo energia/dia: ${r.daily_energy_cost_usd:.2f}")
        print(f"  Margen neto/dia:   ${r.daily_product_value_usd - r.daily_energy_cost_usd:,.0f}")
        if r.payback_hours < float('inf'):
            print(f"  Payback:           {r.payback_hours:.1f} h "
                  f"({r.payback_hours/24:.1f} dias)")
        else:
            print(f"  Payback:           N/A (sin produccion neta)")

        # Buffer
        if cfg.use_buffer:
            print(f"\n  {'BUFFER':=<88}")
            print(f"  Volumen:           {cfg.buffer_volume_ml:.0f} mL")
            print(f"  Tiempo llenado:    {r.buffer_fill_time_min:.1f} min")
            print(f"  Capacidad:         {r.buffer_capacity_doses:.0f} dosis de clasificador")

        print("\n" + "=" * 90)

    def print_optimization_report(self, opt_results: Dict):
        """Reporte de optimizacion con top 5"""
        top5 = opt_results['top5']
        best = opt_results['best']

        print("\n" + "=" * 90)
        print("  OPTIMIZACION - TOP 5 CONFIGURACIONES")
        print("=" * 90)

        header = (f"  {'#':<3} {'N_cl':<6} {'Vol':<6} {'Laser':<7} "
                  f"{'Flow':<7} {'Recov':<8} {'Purity':<8} "
                  f"{'Net mg/h':<10} {'g/dia':<8} {'Payback':<10}")
        print(header)
        print(f"  {'-'*88}")

        for i, r in enumerate(top5):
            line = (f"  {i+1:<3} {r.n_classifiers:<6} "
                    f"{r.stage_volume_uL:<6.0f} "
                    f"{r.laser_power_mw:<7.0f} "
                    f"{r.classifier_flow_ml_min:<7.3f} "
                    f"{r.gross_recovery*100:<7.1f}% "
                    f"{r.purity*100:<7.1f}% "
                    f"{r.net_production_mg_h:<10.1f} "
                    f"{r.net_production_g_day:<8.2f} "
                    f"{r.payback_hours:<10.1f}h")
            print(line)

        if best:
            print(f"\n  MEJOR: {best.n_classifiers} clasificadores, "
                  f"{best.stage_volume_uL:.0f} uL, "
                  f"{best.laser_power_mw:.0f} mW, "
                  f"{best.classifier_flow_ml_min:.3f} mL/min")
            print(f"  -> {best.net_production_mg_h:.1f} mg/h purificado, "
                  f"{best.net_production_g_day:.2f} g/dia, "
                  f"payback {best.payback_hours:.1f}h")

        print("\n" + "=" * 90)

    def print_sweep_report(self, sweep: List[Dict]):
        """Reporte del sweep de recovery"""
        print("\n" + "=" * 90)
        print("  SWEEP RECOVERY: EFECTO DE PARAMETROS MEJORADOS")
        print("=" * 90)

        header = (f"  {'Config':<48} {'Recov':<8} {'Purity':<8} "
                  f"{'Net mg/h':<10} {'Boost':<8}")
        print(header)
        print(f"  {'-'*88}")

        for s in sweep:
            line = (f"  {s['label']:<48} "
                    f"{s['recovery']*100:<7.1f}% "
                    f"{s['purity']*100:<7.1f}% "
                    f"{s['net_mg_h']:<10.1f} "
                    f"{s['boost']:<8.3f}x")
            print(line)

        if len(sweep) >= 2:
            base = sweep[0]
            best = max(sweep, key=lambda s: s['net_mg_h'])
            if base['net_mg_h'] > 0:
                improvement = best['net_mg_h'] / base['net_mg_h']
                print(f"\n  Mejora total: {improvement:.2f}x "
                      f"({base['net_mg_h']:.1f} -> {best['net_mg_h']:.1f} mg/h)")

        print("\n" + "=" * 90)


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sistema Paralelo: Milireactor + N Clasificadores + Waste Recirculation')

    # Reactor
    parser.add_argument('--reactor-flow', type=float, default=5.0,
                        help='Flujo del reactor en mL/min (default: 5)')
    parser.add_argument('--topology', type=str, default='multi_channel',
                        choices=['falling_film', 'multi_channel', 'annular', 'bubble_column'],
                        help='Topologia del reactor (default: multi_channel)')
    parser.add_argument('--n-channels', type=int, default=8,
                        help='Canales del reactor multi-channel (default: 8)')
    parser.add_argument('--channel-length', type=float, default=300.0,
                        help='Longitud de canal en mm (default: 300)')

    # Clasificadores
    parser.add_argument('--n-classifiers', type=int, default=None,
                        help='Numero de clasificadores (default: auto)')
    parser.add_argument('--classifier-flow', type=float,
                        default=DEFAULT_CLASSIFIER_FLOW_ML_MIN,
                        help=f'Flujo por clasificador mL/min (default: {DEFAULT_CLASSIFIER_FLOW_ML_MIN})')
    parser.add_argument('--stage-vol', type=float,
                        default=DEFAULT_STAGE_VOLUME_UL,
                        help=f'Volumen por camara uL (default: {DEFAULT_STAGE_VOLUME_UL})')
    parser.add_argument('--laser-power', type=float,
                        default=DEFAULT_LASER_POWER_MW,
                        help=f'Potencia laser mW (default: {DEFAULT_LASER_POWER_MW})')
    parser.add_argument('--sep-stages', type=int, default=DEFAULT_SEP_STAGES,
                        help=f'Etapas de separacion (default: {DEFAULT_SEP_STAGES})')
    parser.add_argument('--ref-stages', type=int, default=DEFAULT_REF_STAGES,
                        help=f'Etapas de refinamiento (default: {DEFAULT_REF_STAGES})')
    parser.add_argument('--wr-stages', type=int, default=DEFAULT_WR_STAGES,
                        help=f'Etapas waste recapture (default: {DEFAULT_WR_STAGES})')
    parser.add_argument('--solvent', type=str, default='water',
                        choices=list(SOLVENTS.keys()),
                        help='Solvente (default: water)')

    # Buffer
    parser.add_argument('--buffer', action='store_true',
                        help='Usar tanque buffer entre reactor y clasificadores')
    parser.add_argument('--buffer-vol', type=float, default=BUFFER_TANK_ML,
                        help=f'Volumen buffer en mL (default: {BUFFER_TANK_ML})')

    # Recirculacion
    parser.add_argument('--no-recirculation', action='store_true',
                        help='Desactivar recirculacion de waste')
    parser.add_argument('--recirc-fraction', type=float, default=0.80,
                        help='Fraccion de waste recirculado (default: 0.80)')

    # Modos
    parser.add_argument('--optimize', action='store_true',
                        help='Grid search de parametros')
    parser.add_argument('--sweep-recovery', action='store_true',
                        help='Sweep de recovery vs parametros mejorados')
    parser.add_argument('--diagram', action='store_true',
                        help='Solo mostrar diagrama de arquitectura')

    args = parser.parse_args()

    # Mapear topologia
    topo_map = {
        'falling_film': ScaleTopology.FALLING_FILM,
        'multi_channel': ScaleTopology.MULTI_CHANNEL,
        'annular': ScaleTopology.ANNULAR,
        'bubble_column': ScaleTopology.BUBBLE_COLUMN,
    }

    config = ParallelClassifierConfig(
        reactor_topology=topo_map[args.topology],
        reactor_flow_ml_min=args.reactor_flow,
        reactor_n_channels=args.n_channels,
        reactor_channel_length_mm=args.channel_length,
        n_classifiers=args.n_classifiers,
        classifier_flow_ml_min=args.classifier_flow,
        stage_volume_uL=args.stage_vol,
        laser_power_mw=args.laser_power,
        sep_stages=args.sep_stages,
        ref_stages=args.ref_stages,
        wr_stages=args.wr_stages,
        solvent=args.solvent,
        use_buffer=args.buffer,
        buffer_volume_ml=args.buffer_vol,
        waste_recirculation=not args.no_recirculation,
        waste_recirculation_fraction=args.recirc_fraction,
    )

    system = ParallelClassificationSystem(config)

    # Diagrama
    if args.diagram:
        system.print_architecture_diagram()
        return

    # Sweep recovery
    if args.sweep_recovery:
        sweep = system.sweep_recovery()
        system.print_sweep_report(sweep)
        return

    # Optimizar
    if args.optimize:
        opt = system.optimize()
        system.print_optimization_report(opt)
        if opt['best']:
            # Crear sistema con la mejor config para reporte completo
            best_r = opt['best']
            best_cfg = ParallelClassifierConfig(
                reactor_topology=config.reactor_topology,
                reactor_flow_ml_min=config.reactor_flow_ml_min,
                reactor_n_channels=config.reactor_n_channels,
                reactor_channel_length_mm=config.reactor_channel_length_mm,
                n_classifiers=best_r.n_classifiers,
                classifier_flow_ml_min=best_r.classifier_flow_ml_min,
                stage_volume_uL=best_r.stage_volume_uL,
                laser_power_mw=best_r.laser_power_mw,
                sep_stages=config.sep_stages,
                ref_stages=config.ref_stages,
                wr_stages=config.wr_stages,
                solvent=config.solvent,
                waste_recirculation=config.waste_recirculation,
                waste_recirculation_fraction=config.waste_recirculation_fraction,
            )
            best_system = ParallelClassificationSystem(best_cfg)
            best_system.print_architecture_diagram()
        return

    # Default: evaluar y reportar
    result = system.evaluate()
    system.print_report(result)
    system.print_architecture_diagram()


if __name__ == '__main__':
    main()
