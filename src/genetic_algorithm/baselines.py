"""
Algoritmos baseline para comparação com o Algoritmo Genético VRP.

Dois baselines são implementados:

    1. Rota aleatória (random): distribui as paradas sem nenhuma
       inteligência — representa o pior caso esperado.

    2. Vizinho mais próximo (greedy nearest-neighbor): a cada passo,
       visita a parada ainda não atendida mais próxima da posição atual.
       É o algoritmo guloso clássico para TSP/VRP — rápido, determinístico
       e fácil de entender, mas longe do ótimo global.

Ambos retornam um ResultadoBaseline com a mesma estrutura de ResultadoGA
(rotas por veículo + fitness) para facilitar a comparação direta.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fitness import (
    PESOS_PADRAO,
    VEICULO_PADRAO,
    PesosFitness,
    ParametrosVeiculo,
    calcular_fitness,
)
from genetic_algorithm.vrp import cromossomo_para_rotas, fitness_vrp


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class ResultadoBaseline:
    """Resultado de um algoritmo baseline, comparável a ResultadoGA."""
    nome:           str
    rotas:          List[List[int]]
    fitness_total:  float


# ---------------------------------------------------------------------------
# Baseline 1 — Rota aleatória
# ---------------------------------------------------------------------------

def rota_aleatoria(
    df: pd.DataFrame,
    n_veiculos: int = 3,
    veiculo: ParametrosVeiculo = VEICULO_PADRAO,
    pesos: PesosFitness = PESOS_PADRAO,
    seed: Optional[int] = 42,
) -> ResultadoBaseline:
    """
    Distribui as paradas em ordem aleatória entre os veículos.
    Representa o pior caso — nenhuma otimização, puro acaso.
    Usado como limite inferior de qualidade (o GA deve superar isso por larga margem).
    """
    ids = df[df["tipo"] != "deposito"]["id"].tolist()
    rng = random.Random(seed)
    cromossomo = ids[:]
    rng.shuffle(cromossomo)

    rotas = cromossomo_para_rotas(cromossomo, df, n_veiculos, veiculo)
    fitness = fitness_vrp(cromossomo, df, n_veiculos, veiculo, pesos)

    return ResultadoBaseline(nome="Aleatório", rotas=rotas, fitness_total=fitness)


# ---------------------------------------------------------------------------
# Baseline 2 — Vizinho mais próximo (greedy)
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def vizinho_mais_proximo(
    df: pd.DataFrame,
    n_veiculos: int = 3,
    veiculo: ParametrosVeiculo = VEICULO_PADRAO,
    pesos: PesosFitness = PESOS_PADRAO,
) -> ResultadoBaseline:
    """
    Algoritmo guloso do vizinho mais próximo para VRP.

    Constrói a rota de cada veículo iterativamente: partindo do depósito,
    sempre visita a parada não atendida mais próxima da posição atual,
    respeitando capacidade e autonomia. Quando um veículo esgota sua
    capacidade ou autonomia, o próximo assume do depósito.

    É determinístico e rápido, mas tende a produzir rotas ~20-25% piores
    que o ótimo global — o GA deve superá-lo claramente.
    """
    idx = df.set_index("id")
    ids_restantes = set(df[df["tipo"] != "deposito"]["id"].tolist())

    rotas: List[List[int]] = [[] for _ in range(n_veiculos)]
    v = 0
    carga = 0.0
    dist  = 0.0
    lat_atual = float(idx.loc[0, "latitude"])
    lon_atual = float(idx.loc[0, "longitude"])

    while ids_restantes and v < n_veiculos:
        # Encontra o vizinho mais próximo que ainda cabe no veículo atual
        melhor_id   = None
        melhor_dist = float("inf")

        for pid in ids_restantes:
            demanda  = float(idx.loc[pid, "demanda_kg"])
            lat_p    = float(idx.loc[pid, "latitude"])
            lon_p    = float(idx.loc[pid, "longitude"])
            d        = _haversine(lat_atual, lon_atual, lat_p, lon_p)

            cabe_cap = carga + demanda <= veiculo.capacidade_max_kg
            cabe_aut = dist  + d       <= veiculo.autonomia_max_km

            if cabe_cap and cabe_aut and d < melhor_dist:
                melhor_dist = d
                melhor_id   = pid

        if melhor_id is not None:
            # Visita o vizinho mais próximo
            rotas[v].append(melhor_id)
            carga    += float(idx.loc[melhor_id, "demanda_kg"])
            dist     += melhor_dist
            lat_atual = float(idx.loc[melhor_id, "latitude"])
            lon_atual = float(idx.loc[melhor_id, "longitude"])
            ids_restantes.remove(melhor_id)
        else:
            # Nenhuma parada cabe neste veículo — passa para o próximo
            v += 1
            if v >= n_veiculos:
                break
            carga     = 0.0
            dist      = 0.0
            lat_atual = float(idx.loc[0, "latitude"])
            lon_atual = float(idx.loc[0, "longitude"])

    # Paradas que sobram (se todos os veículos esgotaram) ficam no último
    for pid in ids_restantes:
        rotas[-1].append(pid)

    cromossomo = [p for r in rotas for p in r]
    fitness = fitness_vrp(cromossomo, df, n_veiculos, veiculo, pesos)

    return ResultadoBaseline(
        nome="Vizinho mais próximo (greedy)",
        rotas=rotas,
        fitness_total=fitness,
    )
