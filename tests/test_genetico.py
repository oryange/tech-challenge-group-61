"""
Testes automatizados para src/genetic_algorithm/vrp.py.

Cobre: inicialização da população, operadores genéticos (OX, swap, inversão),
seleção por torneio, decodificação de cromossomo em rotas e loop de evolução.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gerar_dados import gerar_pontos
from fitness import ParametrosVeiculo, PesosFitness
from genetic_algorithm.vrp import (
    ConfigGA,
    crossover_ox,
    cromossomo_para_rotas,
    evoluir,
    fitness_vrp,
    gerar_populacao,
    mutacao_inversao,
    mutacao_swap,
    selecao_torneio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df():
    return gerar_pontos(n=20)


@pytest.fixture(scope="module")
def ids(df):
    return df[df["tipo"] != "deposito"]["id"].tolist()


@pytest.fixture(scope="module")
def populacao(ids):
    return gerar_populacao(ids, tamanho=20, seed=42)


# ---------------------------------------------------------------------------
# Inicialização da população
# ---------------------------------------------------------------------------

def test_populacao_tamanho_correto(ids):
    pop = gerar_populacao(ids, tamanho=30, seed=0)
    assert len(pop) == 30


def test_individuo_e_permutacao_valida(ids):
    pop = gerar_populacao(ids, tamanho=10, seed=0)
    for ind in pop:
        assert sorted(ind) == sorted(ids), "Cromossomo deve conter cada ID exatamente uma vez"


def test_populacao_tem_diversidade(ids):
    pop = gerar_populacao(ids, tamanho=10, seed=7)
    assert len({tuple(ind) for ind in pop}) > 1, "População inicial deve ter diversidade"


def test_seed_garante_reproducibilidade(ids):
    pop1 = gerar_populacao(ids, tamanho=10, seed=99)
    pop2 = gerar_populacao(ids, tamanho=10, seed=99)
    assert pop1 == pop2


# ---------------------------------------------------------------------------
# Crossover OX
# ---------------------------------------------------------------------------

def test_crossover_ox_tamanho_correto(ids):
    pop = gerar_populacao(ids, 5, seed=1)
    filho = crossover_ox(pop[0], pop[1])
    assert len(filho) == len(ids)


def test_crossover_ox_e_permutacao_valida(ids):
    pop = gerar_populacao(ids, 10, seed=2)
    for i in range(len(pop) - 1):
        filho = crossover_ox(pop[i], pop[i + 1])
        assert sorted(filho) == sorted(ids)


def test_crossover_ox_herda_segmento_de_parent1(ids):
    pop = gerar_populacao(ids, 5, seed=3)
    p1, p2 = pop[0], pop[1]
    # executa 20 vezes para cobrir casos com vários pontos de corte
    for _ in range(20):
        filho = crossover_ox(p1, p2)
        # todo gene do filho veio de p1 ou p2
        assert set(filho) == set(p1)


# ---------------------------------------------------------------------------
# Mutação por troca (swap)
# ---------------------------------------------------------------------------

def test_mutacao_swap_mantem_permutacao(ids):
    pop = gerar_populacao(ids, 5, seed=4)
    for ind in pop:
        mutado = mutacao_swap(ind, taxa=1.0)
        assert sorted(mutado) == sorted(ids)


def test_mutacao_swap_taxa_zero_nao_muda(ids):
    pop = gerar_populacao(ids, 3, seed=5)
    for ind in pop:
        mutado = mutacao_swap(ind, taxa=0.0)
        assert mutado == ind


def test_mutacao_swap_taxa_um_muda_algo(ids):
    pop = gerar_populacao(ids, 20, seed=6)
    houve_mudanca = any(mutacao_swap(ind, taxa=1.0) != ind for ind in pop)
    assert houve_mudanca


# ---------------------------------------------------------------------------
# Mutação por inversão
# ---------------------------------------------------------------------------

def test_mutacao_inversao_mantem_permutacao(ids):
    pop = gerar_populacao(ids, 5, seed=7)
    for ind in pop:
        mutado = mutacao_inversao(ind, taxa=1.0)
        assert sorted(mutado) == sorted(ids)


def test_mutacao_inversao_taxa_zero_nao_muda(ids):
    pop = gerar_populacao(ids, 3, seed=8)
    for ind in pop:
        mutado = mutacao_inversao(ind, taxa=0.0)
        assert mutado == ind


# ---------------------------------------------------------------------------
# Seleção por torneio
# ---------------------------------------------------------------------------

def test_selecao_torneio_retorna_cromossomo_valido(ids, populacao):
    fits = [float(i) for i in range(len(populacao))]
    selecionado = selecao_torneio(populacao, fits, tamanho=3)
    assert sorted(selecionado) == sorted(ids)


def test_selecao_torneio_favorece_menor_fitness(ids, populacao):
    # fitness crescente → índice 0 tem menor fitness
    fits = list(range(len(populacao)))
    contagem_idx0 = sum(
        selecao_torneio(populacao, fits, tamanho=len(populacao)) == populacao[0]
        for _ in range(50)
    )
    # com torneio de tamanho total, o melhor sempre vence
    assert contagem_idx0 == 50


# ---------------------------------------------------------------------------
# Decodificação de cromossomo
# ---------------------------------------------------------------------------

def test_cromossomo_para_rotas_cobre_todos_pontos(df, ids):
    rotas = cromossomo_para_rotas(ids, df, n_veiculos=3)
    pontos_cobertos = [p for r in rotas for p in r]
    assert sorted(pontos_cobertos) == sorted(ids)


def test_cromossomo_para_rotas_n_veiculos_correto(df, ids):
    rotas = cromossomo_para_rotas(ids, df, n_veiculos=3)
    assert len(rotas) == 3


def test_cromossomo_para_rotas_com_veiculo_grande(df, ids):
    veiculo_grande = ParametrosVeiculo(capacidade_max_kg=9999.0, autonomia_max_km=9999.0)
    rotas = cromossomo_para_rotas(ids, df, n_veiculos=3, veiculo=veiculo_grande)
    pontos = [p for r in rotas for p in r]
    assert sorted(pontos) == sorted(ids)
    # com capacidade ilimitada, só o veículo 0 deve ser usado
    assert len(rotas[0]) == len(ids)
    assert rotas[1] == []
    assert rotas[2] == []


# ---------------------------------------------------------------------------
# Fitness VRP
# ---------------------------------------------------------------------------

def test_fitness_vrp_positivo(df, ids):
    assert fitness_vrp(ids, df, n_veiculos=3) > 0


def test_fitness_vrp_aumenta_com_pesos_maiores(df, ids):
    pesos_baixos = PesosFitness(prioridade=1.0, janela=1.0, capacidade=1.0, autonomia=1.0)
    pesos_altos  = PesosFitness(prioridade=999.0, janela=999.0, capacidade=999.0, autonomia=999.0)
    f_baixo = fitness_vrp(ids, df, n_veiculos=3, pesos=pesos_baixos)
    f_alto  = fitness_vrp(ids, df, n_veiculos=3, pesos=pesos_altos)
    assert f_alto >= f_baixo


# ---------------------------------------------------------------------------
# Loop de evolução
# ---------------------------------------------------------------------------

def test_evoluir_retorna_resultado_valido(df, ids):
    config = ConfigGA(tamanho_populacao=10, n_geracoes=5, seed=0)
    res = evoluir(df, config=config)
    assert sorted(res.melhor_cromossomo) == sorted(ids)
    assert res.melhor_fitness > 0
    assert len(res.historico_fitness) == 5
    assert len(res.historico_media) == 5
    assert len(res.rotas) == config.n_veiculos


def test_evoluir_fitness_nao_piora(df):
    config = ConfigGA(tamanho_populacao=20, n_geracoes=20, seed=1)
    res = evoluir(df, config=config)
    for i in range(1, len(res.historico_fitness)):
        assert res.historico_fitness[i] <= res.historico_fitness[i - 1] + 1e-9, (
            f"Fitness piorou na geração {i}: "
            f"{res.historico_fitness[i-1]} → {res.historico_fitness[i]}"
        )


def test_evoluir_reprodutivel_com_seed(df):
    config = ConfigGA(tamanho_populacao=10, n_geracoes=5, seed=77)
    res1 = evoluir(df, config=config)
    res2 = evoluir(df, config=config)
    assert res1.melhor_fitness == res2.melhor_fitness
    assert res1.melhor_cromossomo == res2.melhor_cromossomo


def test_evoluir_callback_chamado(df):
    geracoes_vistas = []
    def cb(g, f):
        geracoes_vistas.append(g)

    config = ConfigGA(tamanho_populacao=10, n_geracoes=8, seed=2)
    evoluir(df, config=config, callback=cb)
    assert geracoes_vistas == list(range(8))


def test_evoluir_rotas_cobrem_todos_pontos(df):
    config = ConfigGA(tamanho_populacao=10, n_geracoes=5, seed=3)
    res = evoluir(df, config=config)
    ids_cobertos = sorted(p for r in res.rotas for p in r)
    ids_esperados = sorted(df[df["tipo"] != "deposito"]["id"].tolist())
    assert ids_cobertos == ids_esperados
