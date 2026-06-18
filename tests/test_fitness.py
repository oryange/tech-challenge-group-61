"""
Testes automatizados para src/fitness.py.

Valida cada componente da função de fitness e o comportamento
esperado do sistema de pontuação de rotas.
"""

import math
import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fitness import (
    haversine,
    calcular_distancia_rota,
    calcular_penalidade_prioridade,
    calcular_penalidade_janela,
    calcular_penalidade_capacidade,
    calcular_penalidade_autonomia,
    calcular_fitness,
    detalhar_fitness,
    ParametrosVeiculo,
    PesosFitness,
)
from gerar_dados import gerar_pontos


@pytest.fixture(scope="module")
def df():
    return gerar_pontos(n=30)


@pytest.fixture(scope="module")
def rota_todos(df):
    """Rota com todos os pontos de atendimento (sem depósito)."""
    return df[df["tipo"] != "deposito"]["id"].tolist()


@pytest.fixture(scope="module")
def rota_emergencia(df):
    """Rota apenas com pontos de emergência obstétrica (prioridade 1)."""
    return df[df["tipo"] == "emergencia_obstetrica"]["id"].tolist()


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def test_haversine_mesmo_ponto():
    """Distância de um ponto a si mesmo é zero."""
    assert haversine(-23.55, -46.63, -23.55, -46.63) == 0.0


def test_haversine_distancia_positiva():
    """Distância entre dois pontos distintos é sempre positiva."""
    d = haversine(-23.55, -46.63, -23.60, -46.70)
    assert d > 0


def test_haversine_simetria():
    """Distância A→B é igual a B→A."""
    d1 = haversine(-23.55, -46.63, -23.60, -46.70)
    d2 = haversine(-23.60, -46.70, -23.55, -46.63)
    assert math.isclose(d1, d2, rel_tol=1e-9)


def test_haversine_valor_aproximado():
    """SP → RJ ≈ 357 km (referência conhecida)."""
    d = haversine(-23.5505, -46.6333, -22.9068, -43.1729)
    assert 340 < d < 380


# ---------------------------------------------------------------------------
# Distância da rota
# ---------------------------------------------------------------------------

def test_distancia_rota_vazia(df):
    assert calcular_distancia_rota([], df) == 0.0


def test_distancia_rota_positiva(df, rota_todos):
    assert calcular_distancia_rota(rota_todos, df) > 0


def test_distancia_rota_inclui_retorno_deposito(df):
    """Rota com 1 ponto: ida + volta ao depósito."""
    ponto = df[df["tipo"] != "deposito"]["id"].iloc[0]
    d_um_ponto = calcular_distancia_rota([ponto], df)
    assert d_um_ponto > 0


# ---------------------------------------------------------------------------
# Penalidade de prioridade
# ---------------------------------------------------------------------------

def test_penalidade_prioridade_rota_vazia(df):
    assert calcular_penalidade_prioridade([], df) == 0.0


def test_emergencia_no_inicio_menor_penalidade(df):
    """Emergência no início da rota gera menos penalidade que no fim."""
    emergencias = df[df["tipo"] == "emergencia_obstetrica"]["id"].tolist()
    outros = df[(df["tipo"] != "deposito") & (df["tipo"] != "emergencia_obstetrica")]["id"].tolist()

    rota_emergencia_primeiro = emergencias + outros
    rota_emergencia_ultimo = outros + emergencias

    pen_primeiro = calcular_penalidade_prioridade(rota_emergencia_primeiro, df)
    pen_ultimo = calcular_penalidade_prioridade(rota_emergencia_ultimo, df)

    assert pen_primeiro < pen_ultimo


# ---------------------------------------------------------------------------
# Penalidade de janela
# ---------------------------------------------------------------------------

def test_penalidade_janela_rota_vazia(df):
    assert calcular_penalidade_janela([], df) == 0.0


def test_penalidade_janela_nao_negativa(df, rota_todos):
    assert calcular_penalidade_janela(rota_todos, df) >= 0.0


def test_veiculo_mais_rapido_menos_penalidade_janela(df, rota_todos):
    """Veículo mais rápido chega dentro das janelas com mais facilidade."""
    veiculo_lento = ParametrosVeiculo(velocidade_kmh=20.0)
    veiculo_rapido = ParametrosVeiculo(velocidade_kmh=80.0)

    pen_lento = calcular_penalidade_janela(rota_todos, df, veiculo_lento)
    pen_rapido = calcular_penalidade_janela(rota_todos, df, veiculo_rapido)

    assert pen_rapido <= pen_lento


# ---------------------------------------------------------------------------
# Penalidade de capacidade
# ---------------------------------------------------------------------------

def test_penalidade_capacidade_rota_vazia(df):
    assert calcular_penalidade_capacidade([], df) == 0.0


def test_sem_penalidade_capacidade_com_veiculo_grande(df, rota_todos):
    """Veículo com capacidade enorme não deve ter penalidade."""
    veiculo_grande = ParametrosVeiculo(capacidade_max_kg=9999.0)
    assert calcular_penalidade_capacidade(rota_todos, df, veiculo_grande) == 0.0


def test_penalidade_capacidade_com_veiculo_pequeno(df, rota_todos):
    """Veículo com capacidade mínima deve ter penalidade."""
    veiculo_pequeno = ParametrosVeiculo(capacidade_max_kg=0.1)
    assert calcular_penalidade_capacidade(rota_todos, df, veiculo_pequeno) > 0.0


# ---------------------------------------------------------------------------
# Penalidade de autonomia
# ---------------------------------------------------------------------------

def test_sem_penalidade_autonomia_com_veiculo_de_longa_autonomia(df, rota_todos):
    veiculo_longo = ParametrosVeiculo(autonomia_max_km=9999.0)
    assert calcular_penalidade_autonomia(rota_todos, df, veiculo_longo) == 0.0


def test_penalidade_autonomia_com_veiculo_de_curta_autonomia(df, rota_todos):
    veiculo_curto = ParametrosVeiculo(autonomia_max_km=1.0)
    assert calcular_penalidade_autonomia(rota_todos, df, veiculo_curto) > 0.0


# ---------------------------------------------------------------------------
# Fitness total
# ---------------------------------------------------------------------------

def test_fitness_rota_vazia(df):
    assert calcular_fitness([], df) == float("inf")


def test_fitness_sempre_positivo(df, rota_todos):
    assert calcular_fitness(rota_todos, df) > 0


def test_fitness_menor_com_menos_penalidades(df):
    """Rota só com 1 ponto tem fitness menor que rota com todos os pontos."""
    ponto = df[df["tipo"] != "deposito"]["id"].iloc[0]
    f_um = calcular_fitness([ponto], df)
    rota_todos = df[df["tipo"] != "deposito"]["id"].tolist()
    f_todos = calcular_fitness(rota_todos, df)
    assert f_um < f_todos


def test_pesos_maiores_aumentam_fitness(df, rota_todos):
    """Aumentar pesos de penalidade deve aumentar (ou manter) o fitness."""
    pesos_baixos = PesosFitness(prioridade=1.0, janela=1.0, capacidade=1.0, autonomia=1.0)
    pesos_altos = PesosFitness(prioridade=999.0, janela=999.0, capacidade=999.0, autonomia=999.0)
    f_baixo = calcular_fitness(rota_todos, df, pesos=pesos_baixos)
    f_alto = calcular_fitness(rota_todos, df, pesos=pesos_altos)
    assert f_alto >= f_baixo


# ---------------------------------------------------------------------------
# Detalhar fitness
# ---------------------------------------------------------------------------

def test_detalhar_fitness_retorna_todas_as_chaves(df, rota_todos):
    chaves_esperadas = {
        "fitness_total", "distancia_km", "pen_prioridade",
        "pen_janela_h", "pen_capacidade_kg", "pen_autonomia_km",
        "carga_total_kg", "n_pontos",
    }
    detalhe = detalhar_fitness(rota_todos, df)
    assert chaves_esperadas == set(detalhe.keys())


def test_detalhar_fitness_n_pontos_correto(df, rota_todos):
    detalhe = detalhar_fitness(rota_todos, df)
    assert detalhe["n_pontos"] == len(rota_todos)


def test_detalhar_fitness_total_coerente(df, rota_todos):
    """fitness_total do detalhe deve bater com calcular_fitness."""
    f1 = calcular_fitness(rota_todos, df)
    f2 = detalhar_fitness(rota_todos, df)["fitness_total"]
    assert math.isclose(f1, f2, rel_tol=1e-4)
