"""
Função de fitness para o sistema de otimização de rotas médicas (VRP).

A fitness avalia a qualidade de uma rota — quanto menor o valor, melhor a rota.
É composta por quatro componentes com pesos configuráveis:

    fitness = distancia_total
            + peso_prioridade  × penalidade_prioridade
            + peso_janela      × penalidade_janela_horario
            + peso_capacidade  × penalidade_capacidade

Cada componente captura uma restrição real do problema hospitalar:
    - Distância       → custo operacional (combustível, tempo)
    - Prioridade      → emergências atendidas primeiro (obrigatório no enunciado)
    - Janela horário  → respeitar horários seguros (ex.: protocolo discreto)
    - Capacidade      → veículo não pode ultrapassar carga máxima
"""

import math
import pandas as pd
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# Parâmetros do veículo
# ---------------------------------------------------------------------------

@dataclass
class ParametrosVeiculo:
    """Configuração de um veículo da frota."""
    capacidade_max_kg: float = 20.0    # carga máxima em kg
    autonomia_max_km: float = 150.0    # distância máxima por rota
    velocidade_kmh: float = 40.0       # velocidade média urbana


# ---------------------------------------------------------------------------
# Pesos da função de fitness (configuráveis para os experimentos)
# ---------------------------------------------------------------------------

@dataclass
class PesosFitness:
    """
    Pesos de cada componente da fitness.
    Alterar esses valores gera experimentos diferentes — conforme exigido
    pelo enunciado (mínimo 3 experimentos com configurações distintas).
    """
    prioridade: float = 120.0   # penalidade por postergar emergências
                                # (peso alto o bastante para colocar emergências
                                #  no início da rota, superando o termo de distância)
    janela: float = 30.0        # penalidade por violar janela de horário
    capacidade: float = 100.0   # penalidade por exceder capacidade
    autonomia: float = 80.0     # penalidade por exceder autonomia do veículo


# Configuração padrão — usada quando nenhuma é passada explicitamente
PESOS_PADRAO = PesosFitness()
VEICULO_PADRAO = ParametrosVeiculo()


# ---------------------------------------------------------------------------
# Distância geográfica
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula a distância em km entre dois pontos geográficos (fórmula de Haversine).
    Mais preciso que distância euclidiana para coordenadas lat/lon reais.
    """
    R = 6371.0  # raio médio da Terra em km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Componentes individuais da fitness
# ---------------------------------------------------------------------------

def calcular_distancia_rota(rota: List[int], df: pd.DataFrame) -> float:
    """
    Distância total (km) percorrida em uma rota, incluindo retorno ao depósito.

    Args:
        rota: lista de IDs de pontos na ordem de visita (sem o depósito 0)
        df:   DataFrame com colunas latitude e longitude indexado por id
    """
    if not rota:
        return 0.0

    idx = df.set_index("id")
    rota_completa = [0] + list(rota) + [0]  # parte e volta ao depósito
    distancia = 0.0
    for i in range(len(rota_completa) - 1):
        a, b = rota_completa[i], rota_completa[i + 1]
        distancia += haversine(
            idx.loc[a, "latitude"], idx.loc[a, "longitude"],
            idx.loc[b, "latitude"], idx.loc[b, "longitude"],
        )
    return distancia


def calcular_penalidade_prioridade(rota: List[int], df: pd.DataFrame) -> float:
    """
    Penaliza rotas que postergam pontos de alta prioridade.

    Lógica: quanto mais tarde na rota um ponto urgente é atendido,
    maior a penalidade. Pontos de prioridade 1 (emergência obstétrica)
    têm peso máximo; prioridade 5 (preventivo) tem peso mínimo.

    Retorna um valor adimensional proporcional ao atraso acumulado.
    """
    if not rota:
        return 0.0

    idx = df.set_index("id")
    penalidade = 0.0
    n = len(rota)
    for posicao, ponto_id in enumerate(rota):
        prioridade = idx.loc[ponto_id, "prioridade"]
        if prioridade == 0:  # depósito — ignora
            continue
        # Quanto maior a prioridade (menor número), maior o peso
        # Quanto mais tarde na rota (posicao alto), maior a penalidade
        peso_urgencia = (6 - prioridade)  # P1 → 5, P5 → 1
        fator_posicao = posicao / n       # 0.0 (início) a 1.0 (fim)
        penalidade += peso_urgencia * fator_posicao
    return penalidade


def calcular_penalidade_janela(
    rota: List[int],
    df: pd.DataFrame,
    veiculo: ParametrosVeiculo = VEICULO_PADRAO,
) -> float:
    """
    Penaliza chegadas fora da janela de horário do ponto.

    Simula o horário de chegada acumulando tempo de deslocamento
    e tempo de serviço em cada parada. Penaliza quando o veículo
    chega antes do horario_inicio ou depois do horario_fim.

    Inclui o protocolo_discreto (violência doméstica): chegada
    fora do horário seguro recebe penalidade dobrada.
    """
    if not rota:
        return 0.0

    idx = df.set_index("id")
    rota_completa = [0] + list(rota)
    hora_atual = 8.0  # início do dia operacional (8h)
    penalidade = 0.0

    for i in range(1, len(rota_completa)):
        anterior, atual = rota_completa[i - 1], rota_completa[i]
        dist_km = haversine(
            idx.loc[anterior, "latitude"], idx.loc[anterior, "longitude"],
            idx.loc[atual, "latitude"],    idx.loc[atual, "longitude"],
        )
        hora_atual += dist_km / veiculo.velocidade_kmh  # tempo de deslocamento (h)

        inicio = idx.loc[atual, "horario_inicio"]
        fim = idx.loc[atual, "horario_fim"]
        protocolo = idx.loc[atual, "protocolo"]

        # Chegou antes da janela: espera até o início (sem penalidade de espera,
        # mas pode atrasar os próximos pontos)
        if hora_atual < inicio:
            hora_atual = float(inicio)

        # Chegou depois do fim da janela: penalidade pelo atraso
        if hora_atual > fim:
            atraso = hora_atual - fim
            multiplicador = 2.0 if protocolo == "protocolo_discreto" else 1.0
            penalidade += atraso * multiplicador

        # Adiciona tempo de serviço no ponto
        hora_atual += idx.loc[atual, "tempo_servico_min"] / 60.0

    return penalidade


def calcular_penalidade_capacidade(rota: List[int], df: pd.DataFrame,
                                   veiculo: ParametrosVeiculo = VEICULO_PADRAO) -> float:
    """
    Penaliza rotas que excedem a capacidade de carga do veículo.

    Retorna o excesso de kg acima da capacidade máxima (0 se dentro do limite).
    """
    if not rota:
        return 0.0

    idx = df.set_index("id")
    carga_total = sum(idx.loc[ponto_id, "demanda_kg"] for ponto_id in rota)
    excesso = max(0.0, carga_total - veiculo.capacidade_max_kg)
    return excesso


def calcular_penalidade_autonomia(rota: List[int], df: pd.DataFrame,
                                  veiculo: ParametrosVeiculo = VEICULO_PADRAO) -> float:
    """
    Penaliza rotas que excedem a autonomia (distância máxima) do veículo.

    Retorna o excesso de km acima da autonomia máxima (0 se dentro do limite).
    """
    distancia = calcular_distancia_rota(rota, df)
    excesso = max(0.0, distancia - veiculo.autonomia_max_km)
    return excesso


# ---------------------------------------------------------------------------
# Função de fitness principal
# ---------------------------------------------------------------------------

def calcular_fitness(
    rota: List[int],
    df: pd.DataFrame,
    veiculo: ParametrosVeiculo = VEICULO_PADRAO,
    pesos: PesosFitness = PESOS_PADRAO,
) -> float:
    """
    Calcula o fitness total de uma rota. Menor valor = melhor rota.

    Args:
        rota:    lista ordenada de IDs de pontos a visitar (sem depósito)
        df:      DataFrame gerado por gerar_dados.py
        veiculo: parâmetros do veículo (capacidade, autonomia, velocidade)
        pesos:   pesos de cada componente da penalidade

    Returns:
        float: valor de fitness (sempre >= 0; quanto menor, melhor)
    """
    if not rota:
        return float("inf")

    distancia = calcular_distancia_rota(rota, df)
    pen_prioridade = calcular_penalidade_prioridade(rota, df)
    pen_janela = calcular_penalidade_janela(rota, df, veiculo)
    pen_capacidade = calcular_penalidade_capacidade(rota, df, veiculo)
    pen_autonomia = calcular_penalidade_autonomia(rota, df, veiculo)

    return (
        distancia
        + pesos.prioridade * pen_prioridade
        + pesos.janela     * pen_janela
        + pesos.capacidade * pen_capacidade
        + pesos.autonomia  * pen_autonomia
    )


def detalhar_fitness(
    rota: List[int],
    df: pd.DataFrame,
    veiculo: ParametrosVeiculo = VEICULO_PADRAO,
    pesos: PesosFitness = PESOS_PADRAO,
) -> dict:
    """
    Retorna o detalhamento de cada componente do fitness.
    Útil para análise nos notebooks de experimentos e no relatório técnico.
    """
    distancia = calcular_distancia_rota(rota, df)
    pen_prioridade = calcular_penalidade_prioridade(rota, df)
    pen_janela = calcular_penalidade_janela(rota, df, veiculo)
    pen_capacidade = calcular_penalidade_capacidade(rota, df, veiculo)
    pen_autonomia = calcular_penalidade_autonomia(rota, df, veiculo)

    return {
        "fitness_total": round(
            distancia
            + pesos.prioridade * pen_prioridade
            + pesos.janela     * pen_janela
            + pesos.capacidade * pen_capacidade
            + pesos.autonomia  * pen_autonomia, 4
        ),
        "distancia_km":          round(distancia, 4),
        "pen_prioridade":        round(pen_prioridade, 4),
        "pen_janela_h":          round(pen_janela, 4),
        "pen_capacidade_kg":     round(pen_capacidade, 4),
        "pen_autonomia_km":      round(pen_autonomia, 4),
        "carga_total_kg":        round(sum(df.set_index("id").loc[p, "demanda_kg"] for p in rota), 4),
        "n_pontos":              len(rota),
    }
