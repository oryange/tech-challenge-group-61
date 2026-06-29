"""
Testes automatizados para src/genetic_algorithm/baselines.py.

Cobre: reprodutibilidade, validade das rotas (permutação completa),
determinismo do greedy e consistência do ResultadoBaseline.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gerar_dados import gerar_pontos
from fitness import ParametrosVeiculo, PesosFitness
from genetic_algorithm.baselines import (
    ResultadoBaseline,
    rota_aleatoria,
    vizinho_mais_proximo,
)


@pytest.fixture(scope="module")
def df():
    return gerar_pontos(n=20)


@pytest.fixture(scope="module")
def ids(df):
    return df[df["tipo"] != "deposito"]["id"].tolist()


# ---------------------------------------------------------------------------
# ResultadoBaseline
# ---------------------------------------------------------------------------

def test_rota_aleatoria_retorna_resultado_baseline(df):
    res = rota_aleatoria(df)
    assert isinstance(res, ResultadoBaseline)


def test_vizinho_mais_proximo_retorna_resultado_baseline(df):
    res = vizinho_mais_proximo(df)
    assert isinstance(res, ResultadoBaseline)


def test_resultado_tem_nome(df):
    assert rota_aleatoria(df).nome != ""
    assert vizinho_mais_proximo(df).nome != ""


# ---------------------------------------------------------------------------
# Rota aleatória — reprodutibilidade e validade
# ---------------------------------------------------------------------------

def test_rota_aleatoria_reprodutivel_com_seed(df, ids):
    res1 = rota_aleatoria(df, seed=7)
    res2 = rota_aleatoria(df, seed=7)
    pontos1 = sorted(p for r in res1.rotas for p in r)
    pontos2 = sorted(p for r in res2.rotas for p in r)
    assert pontos1 == pontos2
    assert res1.fitness_total == res2.fitness_total


def test_rota_aleatoria_seeds_diferentes_geram_ordens_diferentes(df):
    res1 = rota_aleatoria(df, seed=1)
    res2 = rota_aleatoria(df, seed=99)
    ordem1 = [p for r in res1.rotas for p in r]
    ordem2 = [p for r in res2.rotas for p in r]
    assert ordem1 != ordem2


def test_rota_aleatoria_cobre_todos_os_pontos(df, ids):
    res = rota_aleatoria(df)
    pontos_cobertos = sorted(p for r in res.rotas for p in r)
    assert pontos_cobertos == sorted(ids)


def test_rota_aleatoria_n_veiculos_correto(df):
    res = rota_aleatoria(df, n_veiculos=3)
    assert len(res.rotas) == 3


def test_rota_aleatoria_fitness_positivo(df):
    assert rota_aleatoria(df).fitness_total > 0


# ---------------------------------------------------------------------------
# Vizinho mais próximo — determinismo e validade
# ---------------------------------------------------------------------------

def test_vizinho_mais_proximo_deterministico(df, ids):
    res1 = vizinho_mais_proximo(df)
    res2 = vizinho_mais_proximo(df)
    assert res1.fitness_total == res2.fitness_total
    assert [p for r in res1.rotas for p in r] == [p for r in res2.rotas for p in r]


def test_vizinho_mais_proximo_cobre_todos_os_pontos(df, ids):
    res = vizinho_mais_proximo(df)
    pontos_cobertos = sorted(p for r in res.rotas for p in r)
    assert pontos_cobertos == sorted(ids)


def test_vizinho_mais_proximo_n_veiculos_correto(df):
    res = vizinho_mais_proximo(df, n_veiculos=3)
    assert len(res.rotas) == 3


def test_vizinho_mais_proximo_fitness_positivo(df):
    assert vizinho_mais_proximo(df).fitness_total > 0


# ---------------------------------------------------------------------------
# Comparativo entre abordagens
# ---------------------------------------------------------------------------

def test_ga_supera_aleatório(df):
    """GA (mesmo com poucas gerações) deve superar a rota aleatória."""
    from genetic_algorithm.vrp import ConfigGA, evoluir
    res_alea = rota_aleatoria(df, seed=42)
    res_ga = evoluir(df, config=ConfigGA(tamanho_populacao=20, n_geracoes=30, seed=42))
    assert res_ga.melhor_fitness < res_alea.fitness_total


def test_vizinho_supera_aleatorio(df):
    """Greedy deve superar rota aleatória — é uma heurística, não acaso."""
    res_alea   = rota_aleatoria(df, seed=42)
    res_greedy = vizinho_mais_proximo(df)
    assert res_greedy.fitness_total < res_alea.fitness_total


def test_veiculo_grande_nao_divide_rotas(df, ids):
    """Com capacidade ilimitada, greedy coloca tudo no veículo 0."""
    veiculo_grande = ParametrosVeiculo(capacidade_max_kg=9999.0, autonomia_max_km=9999.0)
    res = vizinho_mais_proximo(df, n_veiculos=3, veiculo=veiculo_grande)
    assert len(res.rotas[0]) == len(ids)
    assert res.rotas[1] == []
    assert res.rotas[2] == []
