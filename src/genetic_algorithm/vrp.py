"""
Algoritmo Genético para VRP (Vehicle Routing Problem) médico.

Adaptado do código-base TSP de sergiopolimante/genetic_algorithm_tsp.
Estende TSP (uma rota, minimizar distância) para VRP (múltiplos veículos,
múltiplas restrições: capacidade, autonomia, janelas de horário, prioridade).

Representação do cromossomo:
    Permutação de IDs de paradas (sem depósito 0).
    O decodificador (cromossomo_para_rotas) percorre a lista da esquerda
    para a direita e distribui as paradas entre veículos respeitando
    capacidade e autonomia. Quando um limite é excedido, inicia-se um
    novo veículo.

Operadores genéticos:
    - Order Crossover (OX): mesmo algoritmo do repositório TSP base,
      adaptado para IDs inteiros em vez de coordenadas.
    - Mutação por troca (swap): troca dois índices aleatórios.
    - Mutação por inversão: inverte um segmento — explora vizinhança maior.
    - Seleção por torneio: mais robusta que roleta para espaços de busca
      com grandes penalidades.
    - Elitismo: melhores N indivíduos passam direto para a próxima geração.
"""

import copy
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional

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


# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------

Cromossomo = List[int]   # permutação de IDs de paradas (sem depósito 0)
Populacao  = List[Cromossomo]


# ---------------------------------------------------------------------------
# Configuração do GA
# ---------------------------------------------------------------------------

@dataclass
class ConfigGA:
    """
    Parâmetros do algoritmo genético — variar esses valores nos experimentos.

    Experimento 1: variar tamanho_populacao (50, 100, 200)
    Experimento 2: variar taxa_mutacao_swap e taxa_mutacao_inversao
    Experimento 3: variar n_veiculos / usar pesos de fitness diferentes
    """
    tamanho_populacao: int   = 100
    n_geracoes:        int   = 200
    taxa_mutacao_swap: float = 0.30
    taxa_mutacao_inversao: float = 0.10
    tamanho_torneio:   int   = 5
    n_elite:           int   = 2     # indivíduos que passam direto (elitismo)
    n_veiculos:        int   = 3     # frota disponível
    seed:              Optional[int] = 42


CONFIG_PADRAO = ConfigGA()


# ---------------------------------------------------------------------------
# Inicialização da população
# ---------------------------------------------------------------------------

def gerar_populacao(
    ids_pontos: List[int],
    tamanho: int,
    seed: Optional[int] = 42,
) -> Populacao:
    """
    Cria população inicial com permutações aleatórias dos IDs.
    Adaptado de generate_random_population() do repositório TSP base.
    """
    rng = random.Random(seed)
    return [
        rng.sample(ids_pontos, len(ids_pontos))
        for _ in range(tamanho)
    ]


# ---------------------------------------------------------------------------
# Decodificação: cromossomo → sub-rotas por veículo
# ---------------------------------------------------------------------------

def cromossomo_para_rotas(
    cromossomo: Cromossomo,
    df: pd.DataFrame,
    n_veiculos: int,
    veiculo: ParametrosVeiculo = VEICULO_PADRAO,
) -> List[List[int]]:
    """
    Decodifica um cromossomo em N sub-rotas para os veículos disponíveis.

    Estratégia gulosa: percorre a permutação da esquerda para a direita;
    quando o próximo ponto excede a capacidade ou autonomia do veículo
    atual, inicia um novo veículo. Paradas excedentes ficam no último
    veículo (com penalidade natural pela violação de restrição).

    Returns:
        Lista de N listas de IDs; cada sublista é a rota de um veículo.
    """
    idx = df.set_index("id")
    rotas: List[List[int]] = [[] for _ in range(n_veiculos)]
    v = 0
    carga = 0.0
    dist  = 0.0
    lat_prev = float(idx.loc[0, "latitude"])
    lon_prev = float(idx.loc[0, "longitude"])

    for pid in cromossomo:
        demanda   = float(idx.loc[pid, "demanda_kg"])
        lat_atual = float(idx.loc[pid, "latitude"])
        lon_atual = float(idx.loc[pid, "longitude"])
        dist_leg  = _haversine(lat_prev, lon_prev, lat_atual, lon_atual)

        excede_cap = carga + demanda  > veiculo.capacidade_max_kg
        excede_aut = dist  + dist_leg > veiculo.autonomia_max_km

        if (excede_cap or excede_aut) and v < n_veiculos - 1:
            v += 1
            carga = 0.0
            dist  = 0.0
            lat_prev = float(idx.loc[0, "latitude"])
            lon_prev = float(idx.loc[0, "longitude"])
            dist_leg = _haversine(lat_prev, lon_prev, lat_atual, lon_atual)

        rotas[v].append(pid)
        carga    += demanda
        dist     += dist_leg
        lat_prev  = lat_atual
        lon_prev  = lon_atual

    return rotas


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Fitness VRP
# ---------------------------------------------------------------------------

def fitness_vrp(
    cromossomo: Cromossomo,
    df: pd.DataFrame,
    n_veiculos: int,
    veiculo: ParametrosVeiculo = VEICULO_PADRAO,
    pesos: PesosFitness = PESOS_PADRAO,
) -> float:
    """
    Fitness total para VRP: soma dos fitness de cada sub-rota.
    Sub-rotas vazias (veículos ociosos) não contribuem.
    """
    rotas = cromossomo_para_rotas(cromossomo, df, n_veiculos, veiculo)
    return sum(
        calcular_fitness(r, df, veiculo, pesos)
        for r in rotas if r
    )


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------

def crossover_ox(parent1: Cromossomo, parent2: Cromossomo) -> Cromossomo:
    """
    Order Crossover (OX) — adaptado de order_crossover() do TSP base.

    Copia um segmento de parent1 (entre dois pontos de corte aleatórios)
    para o filho e preenche as posições restantes com os genes de parent2
    na ordem em que aparecem, preservando a validade da permutação.
    """
    n = len(parent1)
    start = random.randint(0, n - 1)
    end   = random.randint(start + 1, n)

    filho: List[Optional[int]] = [None] * n
    filho[start:end] = parent1[start:end]

    livres = [i for i in range(n) if filho[i] is None]
    genes  = [g for g in parent2 if g not in filho]

    for pos, gene in zip(livres, genes):
        filho[pos] = gene

    return filho  # type: ignore[return-value]


def mutacao_swap(cromossomo: Cromossomo, taxa: float) -> Cromossomo:
    """
    Mutação por troca: seleciona dois índices aleatórios e os troca.
    Adaptado de mutate() do TSP base (swap aleatório em vez de adjacente
    para maior diversidade no VRP com muitas paradas).
    """
    mutado = cromossomo[:]
    if random.random() < taxa and len(mutado) >= 2:
        i, j = random.sample(range(len(mutado)), 2)
        mutado[i], mutado[j] = mutado[j], mutado[i]
    return mutado


def mutacao_inversao(cromossomo: Cromossomo, taxa: float) -> Cromossomo:
    """
    Mutação por inversão: inverte um segmento aleatório.
    Explora vizinhança estruturalmente diferente da troca — especialmente
    útil para melhorar sub-rotas após crossover.
    """
    mutado = cromossomo[:]
    if random.random() < taxa and len(mutado) >= 2:
        i, j = sorted(random.sample(range(len(mutado)), 2))
        mutado[i : j + 1] = mutado[i : j + 1][::-1]
    return mutado


# ---------------------------------------------------------------------------
# Seleção
# ---------------------------------------------------------------------------

def selecao_torneio(
    populacao: Populacao,
    fitnesses: List[float],
    tamanho: int,
) -> Cromossomo:
    """
    Seleção por torneio: escolhe `tamanho` candidatos aleatórios e retorna
    o melhor. Mais robusto que roleta quando há penalidades com grande
    disparidade de escala (como capacidade e prioridade).
    """
    idxs = random.sample(range(len(populacao)), k=tamanho)
    melhor = min(idxs, key=lambda i: fitnesses[i])
    return populacao[melhor][:]


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class ResultadoGA:
    """Encapsula a solução e o histórico de convergência da evolução."""
    melhor_cromossomo: Cromossomo
    melhor_fitness:    float
    rotas:             List[List[int]]   # sub-rotas decodificadas por veículo
    historico_fitness: List[float]       # melhor fitness por geração
    historico_media:   List[float]       # fitness médio da população por geração
    n_geracoes:        int


# ---------------------------------------------------------------------------
# Loop de evolução
# ---------------------------------------------------------------------------

def evoluir(
    df: pd.DataFrame,
    config: ConfigGA = CONFIG_PADRAO,
    veiculo: ParametrosVeiculo = VEICULO_PADRAO,
    pesos: PesosFitness = PESOS_PADRAO,
    callback: Optional[Callable[[int, float], None]] = None,
) -> ResultadoGA:
    """
    Executa o algoritmo genético para VRP médico.

    Args:
        df:       DataFrame de pontos (gerado por gerar_dados.py)
        config:   Hiperparâmetros do GA
        veiculo:  Parâmetros do veículo (capacidade, autonomia, velocidade)
        pesos:    Pesos da função de fitness (para experimentos)
        callback: Chamado ao fim de cada geração com (geração, melhor_fitness)
                  Útil para barras de progresso no Streamlit.

    Returns:
        ResultadoGA com a melhor solução encontrada e histórico de convergência.
    """
    if config.seed is not None:
        random.seed(config.seed)

    ids = df[df["tipo"] != "deposito"]["id"].tolist()
    pop = gerar_populacao(ids, config.tamanho_populacao, config.seed)

    historico_fitness: List[float] = []
    historico_media:   List[float] = []
    melhor_cromo = pop[0][:]
    melhor_fit   = float("inf")

    for geracao in range(config.n_geracoes):
        fits = [
            fitness_vrp(ind, df, config.n_veiculos, veiculo, pesos)
            for ind in pop
        ]

        # Ordena população: menor fitness (melhor) primeiro
        pares = sorted(zip(fits, pop), key=lambda x: x[0])
        fits  = [p[0] for p in pares]
        pop   = [p[1] for p in pares]

        if fits[0] < melhor_fit:
            melhor_fit   = fits[0]
            melhor_cromo = pop[0][:]

        historico_fitness.append(melhor_fit)
        historico_media.append(sum(fits) / len(fits))

        if callback:
            callback(geracao, melhor_fit)

        # Elitismo + seleção + crossover + mutação
        nova_pop = [pop[i][:] for i in range(config.n_elite)]

        while len(nova_pop) < config.tamanho_populacao:
            p1 = selecao_torneio(pop, fits, config.tamanho_torneio)
            p2 = selecao_torneio(pop, fits, config.tamanho_torneio)
            filho = crossover_ox(p1, p2)
            filho = mutacao_swap(filho, config.taxa_mutacao_swap)
            filho = mutacao_inversao(filho, config.taxa_mutacao_inversao)
            nova_pop.append(filho)

        pop = nova_pop

    rotas = cromossomo_para_rotas(melhor_cromo, df, config.n_veiculos, veiculo)

    return ResultadoGA(
        melhor_cromossomo=melhor_cromo,
        melhor_fitness=melhor_fit,
        rotas=rotas,
        historico_fitness=historico_fitness,
        historico_media=historico_media,
        n_geracoes=config.n_geracoes,
    )
