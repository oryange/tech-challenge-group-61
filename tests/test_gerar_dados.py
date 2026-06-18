"""
Testes automatizados para src/gerar_dados.py.

Valida que o gerador de dados sintéticos produz saída consistente,
completa e compatível com os requisitos do projeto (enunciado pág. 6-7).
"""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gerar_dados import gerar_pontos, salvar_csv, TIPOS, PROTOCOLOS, SEED


COLUNAS_OBRIGATORIAS = [
    "id", "nome", "tipo", "prioridade",
    "latitude", "longitude", "demanda_kg",
    "janela_horas", "horario_inicio", "horario_fim",
    "protocolo", "tempo_servico_min",
]


@pytest.fixture(scope="module")
def df():
    return gerar_pontos(n=30)


def test_numero_total_de_pontos(df):
    """31 = 30 pontos de atendimento + 1 depósito."""
    assert len(df) == 31


def test_colunas_obrigatorias_presentes(df):
    """Todas as colunas necessárias para fitness, mapa e LLM devem existir."""
    for col in COLUNAS_OBRIGATORIAS:
        assert col in df.columns, f"Coluna '{col}' ausente"


def test_deposito_presente(df):
    """Ponto id=0 (Hospital_Base) deve existir — é a origem de todos os veículos."""
    deposito = df[df["id"] == 0]
    assert len(deposito) == 1
    assert deposito.iloc[0]["nome"] == "Hospital_Base"
    assert deposito.iloc[0]["tipo"] == "deposito"


def test_prioridade_1_presente(df):
    """Emergência obstétrica (prioridade máxima) é requisito obrigatório do enunciado."""
    emergencias = df[df["tipo"] == "emergencia_obstetrica"]
    assert len(emergencias) >= 1


def test_todos_os_tipos_de_atendimento_cobertos(df):
    """Os 4 tipos prioritários do enunciado (pág. 7) devem estar presentes."""
    tipos_esperados = set(TIPOS.keys())
    tipos_presentes = set(df[df["tipo"] != "deposito"]["tipo"].unique())
    assert tipos_esperados.issubset(tipos_presentes), (
        f"Tipos ausentes: {tipos_esperados - tipos_presentes}"
    )


def test_prioridades_validas(df):
    """Prioridade deve estar entre 0 (depósito) e 5 (preventivo)."""
    assert df["prioridade"].between(0, 5).all()


def test_sem_valores_nulos(df):
    """Nenhum campo pode ser nulo — quebraria fitness e visualização."""
    assert df.isnull().sum().sum() == 0


def test_coordenadas_dentro_da_area_de_cobertura(df):
    """Coordenadas devem estar dentro de ~16 km do centro de São Paulo."""
    assert df["latitude"].between(-23.75, -23.35).all()
    assert df["longitude"].between(-46.85, -46.45).all()


def test_janela_de_horario_coerente(df):
    """horario_fim deve ser sempre maior que horario_inicio."""
    pontos = df[df["tipo"] != "deposito"]
    assert (pontos["horario_fim"] > pontos["horario_inicio"]).all()


def test_demanda_positiva(df):
    """Demanda de suprimentos deve ser >= 0 para todos os pontos."""
    assert (df["demanda_kg"] >= 0).all()


def test_tempo_servico_positivo(df):
    """Tempo de serviço deve ser >= 0 para todos os pontos."""
    assert (df["tempo_servico_min"] >= 0).all()


def test_protocolos_validos(df):
    """Cada ponto deve ter um protocolo reconhecido."""
    protocolos_validos = set(PROTOCOLOS.values()) | {"base"}
    for protocolo in df["protocolo"]:
        assert protocolo in protocolos_validos, f"Protocolo desconhecido: '{protocolo}'"


def test_reproducibilidade_com_seed(df):
    """Mesma seed deve gerar exatamente os mesmos dados — essencial para experimentos."""
    df2 = gerar_pontos(n=30)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2.reset_index(drop=True))


def test_salvar_csv(tmp_path, df):
    """CSV deve ser salvo corretamente e legível."""
    caminho = tmp_path / "pontos_teste.csv"
    path = salvar_csv(df, str(caminho))
    assert path.exists()
    df_lido = pd.read_csv(path)
    assert len(df_lido) == len(df)
    assert list(df_lido.columns) == list(df.columns)
