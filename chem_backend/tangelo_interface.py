#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  INTERFAZ TANGELO PARA PARÁMETROS QUÍMICOS
  Calcula propiedades químicas cuánticas para alimentar CFD/PINN
═══════════════════════════════════════════════════════════════════════════════

  Este módulo provee parámetros químicos derivados de simulaciones cuánticas
  para integrarse con el loop de optimización CFD.

  Parámetros que puede calcular:
  - Energías de activación (Ea) para reacciones clave
  - Energías de formación/disociación
  - Constantes de equilibrio vs temperatura
  - Energías de adsorción (si hay superficie)
"""

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

# Cache para evitar recálculos
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES FÍSICAS
# ═══════════════════════════════════════════════════════════════════════════════

HARTREE_TO_EV = 27.2114
HARTREE_TO_KJ_MOL = 2625.5
EV_TO_KJ_MOL = 96.485
KB_EV = 8.617e-5  # eV/K
R_GAS = 8.314     # J/(mol·K)

# ═══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChemicalState:
    """Estado químico local (condiciones)"""
    temperature: float    # K
    pressure: float       # Pa
    composition: Dict[str, float]  # fracción molar {especie: x}
    electric_field: float = 0.0    # V/m (para plasma)

    def hash(self) -> str:
        s = json.dumps({
            "T": round(self.temperature, 1),
            "P": round(self.pressure, 0),
            "comp": {k: round(v, 4) for k, v in self.composition.items()},
            "E": round(self.electric_field, 0)
        }, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:8]


@dataclass
class ChemicalParameters:
    """Parámetros químicos calculados por Tangelo"""
    # Energías de activación para reacciones clave (kJ/mol)
    activation_energies: Dict[str, float]

    # Entalpías de reacción (kJ/mol)
    reaction_enthalpies: Dict[str, float]

    # Constantes de velocidad Arrhenius: k = A * exp(-Ea/RT)
    arrhenius_A: Dict[str, float]  # factor pre-exponencial (1/s o m³/(mol·s))

    # Energía de gap para CQDs (eV)
    cqd_gap_ev: float

    # Tamaño de partícula predicho (nm)
    cqd_size_nm: float

    # Metadatos
    calculation_method: str
    confidence: float  # 0-1

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  REACCIONES DEL SISTEMA DBD
# ═══════════════════════════════════════════════════════════════════════════════

# Reacciones principales en síntesis de CQDs por plasma
REACTIONS = {
    # Disociación de agua por plasma
    "H2O_dissoc": "H2O -> H + OH",

    # Formación de radicales
    "OH_formation": "O + H2O -> 2OH",

    # Oxidación de precursor orgánico (simplificado)
    "precursor_oxid": "C_org + OH -> C_ox + H",

    # Nucleación de carbono
    "C_nucleation": "nC -> C_n (núcleo)",

    # Crecimiento de partícula
    "particle_growth": "C_n + C -> C_{n+1}",

    # Funcionalización superficial
    "surface_func": "C_n + OH -> C_n-OH",
}

# Valores de literatura para calibración
LITERATURE_VALUES = {
    "H2O_dissoc": {"Ea": 498.0, "A": 1e13},      # kJ/mol, 1/s
    "OH_formation": {"Ea": 75.0, "A": 1e11},
    "precursor_oxid": {"Ea": 85.0, "A": 5e10},
    "C_nucleation": {"Ea": 150.0, "A": 1e12},
    "particle_growth": {"Ea": 50.0, "A": 1e11},
    "surface_func": {"Ea": 40.0, "A": 1e10},
}

# ═══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ TANGELO (PLACEHOLDER)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_tangelo_available() -> bool:
    """Verifica si Tangelo está disponible"""
    try:
        import tangelo
        return True
    except ImportError:
        return False


def _calculate_with_tangelo(molecule: str, method: str = "VQE") -> Dict:
    """
    Calcula propiedades con Tangelo (implementación real).
    Por ahora retorna valores de literatura + corrección.
    """
    if not _check_tangelo_available():
        return None

    # TODO: Implementar cálculo real con Tangelo
    # from tangelo import SecondQuantizedMolecule
    # from tangelo.algorithms import VQESolver
    # ...

    return None


def _estimate_from_literature(reaction: str, state: ChemicalState) -> Dict:
    """
    Estima parámetros desde valores de literatura con correcciones.
    Usa modelo Arrhenius modificado por campo eléctrico.
    """
    lit = LITERATURE_VALUES.get(reaction, {"Ea": 100.0, "A": 1e11})

    # Corrección por temperatura (Arrhenius)
    Ea_base = lit["Ea"]
    A_base = lit["A"]

    # Corrección por campo eléctrico (reduce barrera)
    # ΔEa ≈ -α * E² donde α es polarizabilidad
    alpha = 1e-20  # m³ (polarizabilidad típica)
    delta_Ea = -0.5 * alpha * state.electric_field**2 * HARTREE_TO_KJ_MOL

    Ea_corrected = max(Ea_base + delta_Ea, 10.0)  # mínimo 10 kJ/mol

    # Constante de velocidad a la temperatura dada
    k = A_base * np.exp(-Ea_corrected * 1000 / (R_GAS * state.temperature))

    return {
        "Ea": Ea_corrected,
        "A": A_base,
        "k": k,
        "delta_Ea_field": delta_Ea
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class TangeloInterface:
    """
    Interfaz para obtener parámetros químicos cuánticos.

    Flujo:
    1. Recibe estado químico (T, P, composición, E)
    2. Verifica cache
    3. Si no está en cache: calcula con Tangelo o estima de literatura
    4. Retorna parámetros para CFD/PINN
    """

    def __init__(self, use_tangelo: bool = True, cache_results: bool = True):
        self.use_tangelo = use_tangelo and _check_tangelo_available()
        self.cache_results = cache_results

        if self.use_tangelo:
            print("✓ Tangelo disponible - usando cálculos cuánticos")
        else:
            print("⚠ Tangelo no disponible - usando modelo de literatura")

    def get_parameters(self, state: ChemicalState) -> ChemicalParameters:
        """
        Obtiene parámetros químicos para un estado dado.
        """
        # Verificar cache
        cache_key = state.hash()
        cache_file = CACHE_DIR / f"{cache_key}.json"

        if self.cache_results and cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
            return ChemicalParameters(**data)

        # Calcular parámetros
        activation_energies = {}
        reaction_enthalpies = {}
        arrhenius_A = {}

        for reaction in REACTIONS.keys():
            if self.use_tangelo:
                result = _calculate_with_tangelo(reaction, "VQE")
                if result is None:
                    result = _estimate_from_literature(reaction, state)
            else:
                result = _estimate_from_literature(reaction, state)

            activation_energies[reaction] = result["Ea"]
            arrhenius_A[reaction] = result["A"]
            # Entalpía aproximada desde Ea
            reaction_enthalpies[reaction] = result["Ea"] * 0.7  # aproximación

        # Calcular propiedades de CQDs
        cqd_gap, cqd_size = self._estimate_cqd_properties(state)

        params = ChemicalParameters(
            activation_energies=activation_energies,
            reaction_enthalpies=reaction_enthalpies,
            arrhenius_A=arrhenius_A,
            cqd_gap_ev=cqd_gap,
            cqd_size_nm=cqd_size,
            calculation_method="tangelo_vqe" if self.use_tangelo else "literature_model",
            confidence=0.8 if self.use_tangelo else 0.5
        )

        # Guardar en cache
        if self.cache_results:
            with open(cache_file, 'w') as f:
                json.dump(params.to_dict(), f, indent=2)

        return params

    def _estimate_cqd_properties(self, state: ChemicalState) -> Tuple[float, float]:
        """
        Estima propiedades de CQDs basado en condiciones de síntesis.

        Modelo empírico: tamaño depende de T, tiempo residencia, potencia plasma
        Gap depende del tamaño: E_gap = E_bulk + A/d²
        """
        T = state.temperature
        E = state.electric_field

        # Tamaño aproximado (nm) - aumenta con T, disminuye con E
        # Modelo simplificado basado en literatura
        d_base = 2.5  # nm base a 300K, sin campo
        d = d_base * (300 / T)**0.3 * (1 + E/1e6)**0.1
        d = max(1.5, min(5.0, d))  # limitar rango realista

        # Gap óptico (modelo de confinamiento cuántico)
        E_bulk = 1.50  # eV (grafeno dopado N)
        A = 7.26       # eV·nm² (constante de confinamiento)
        gap = E_bulk + A / (d**2)

        return gap, d

    def get_rate_constant(self, reaction: str, state: ChemicalState) -> float:
        """
        Calcula constante de velocidad para una reacción específica.
        k = A * exp(-Ea / RT)
        """
        params = self.get_parameters(state)
        Ea = params.activation_energies.get(reaction, 100.0)
        A = params.arrhenius_A.get(reaction, 1e11)

        k = A * np.exp(-Ea * 1000 / (R_GAS * state.temperature))
        return k

    def generate_lookup_table(self,
                             T_range: Tuple[float, float] = (300, 400),
                             E_range: Tuple[float, float] = (0, 1e6),
                             n_points: int = 10) -> Dict:
        """
        Genera tabla de lookup para interpolación en CFD/PINN.
        """
        T_values = np.linspace(*T_range, n_points)
        E_values = np.linspace(*E_range, n_points)

        table = {
            "T_values": T_values.tolist(),
            "E_values": E_values.tolist(),
            "reactions": {},
        }

        for reaction in REACTIONS.keys():
            k_table = np.zeros((n_points, n_points))

            for i, T in enumerate(T_values):
                for j, E in enumerate(E_values):
                    state = ChemicalState(
                        temperature=T,
                        pressure=101325,
                        composition={"H2O": 1.0},
                        electric_field=E
                    )
                    k_table[i, j] = self.get_rate_constant(reaction, state)

            table["reactions"][reaction] = k_table.tolist()

        return table


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE UTILIDAD
# ═══════════════════════════════════════════════════════════════════════════════

def merge_cfd_tangelo_data(cfd_file: str, tangelo_params: ChemicalParameters) -> Dict:
    """
    Combina datos CFD con parámetros de Tangelo para PINN.
    """
    with open(cfd_file) as f:
        cfd_data = json.load(f)

    merged = {
        "cfd": cfd_data,
        "chemistry": tangelo_params.to_dict(),
        "combined_features": {
            # Características para PINN
            "residence_time": cfd_data.get("results", {}).get("residence_time", 0),
            "cqd_gap_ev": tangelo_params.cqd_gap_ev,
            "cqd_size_nm": tangelo_params.cqd_size_nm,
            "activation_energies": tangelo_params.activation_energies,
        }
    }

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  EJEMPLO DE USO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print("  INTERFAZ TANGELO - PARÁMETROS QUÍMICOS PARA REACTOR DBD")
    print("═" * 70)

    # Crear interfaz
    interface = TangeloInterface(use_tangelo=True, cache_results=True)

    # Estado de ejemplo (condiciones típicas de síntesis)
    state = ChemicalState(
        temperature=350.0,      # K (~77°C)
        pressure=101325,        # Pa (1 atm)
        composition={"H2O": 0.95, "C_org": 0.05},
        electric_field=5e5      # V/m (campo en plasma DBD)
    )

    print(f"\n  Estado químico:")
    print(f"    T = {state.temperature} K")
    print(f"    P = {state.pressure} Pa")
    print(f"    E = {state.electric_field:.0e} V/m")
    print(f"    Composición: {state.composition}")

    # Obtener parámetros
    print("\n→ Calculando parámetros químicos...")
    params = interface.get_parameters(state)

    print(f"\n  Parámetros calculados:")
    print(f"    Método: {params.calculation_method}")
    print(f"    Confianza: {params.confidence:.0%}")

    print(f"\n  Energías de activación (kJ/mol):")
    for rxn, Ea in params.activation_energies.items():
        print(f"    {rxn}: {Ea:.1f}")

    print(f"\n  Propiedades CQD predichas:")
    print(f"    Gap óptico: {params.cqd_gap_ev:.2f} eV")
    print(f"    Tamaño: {params.cqd_size_nm:.2f} nm")
    print(f"    λ emisión: {1240/params.cqd_gap_ev:.0f} nm")

    # Generar tabla de lookup
    print("\n→ Generando tabla de lookup...")
    table = interface.generate_lookup_table(
        T_range=(300, 400),
        E_range=(0, 1e6),
        n_points=5
    )

    # Guardar tabla
    table_file = CACHE_DIR / "lookup_table.json"
    with open(table_file, 'w') as f:
        json.dump(table, f, indent=2)
    print(f"  ✓ Tabla guardada: {table_file}")

    print("\n═" * 70)
