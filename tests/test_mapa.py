"""
Testes automatizados para src/visualization/mapa.py.

Valida a criação do mapa, a presença dos marcadores, o desenho de rotas
e a função de salvamento em HTML.
"""

import sys
from pathlib import Path

import folium
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gerar_dados import gerar_pontos
from visualization.mapa import (
    criar_mapa,
    salvar_mapa,
    _popup_html,
    CORES_TIPO,
    ROTULOS_TIPO,
)


@pytest.fixture(scope="module")
def df():
    return gerar_pontos(n=30)


def test_criar_mapa_retorna_folium_map(df):
    mapa = criar_mapa(df)
    assert isinstance(mapa, folium.Map)


def test_mapa_centralizado_no_deposito(df):
    """O mapa deve ser centralizado nas coordenadas do depósito."""
    deposito = df[df["tipo"] == "deposito"].iloc[0]
    mapa = criar_mapa(df)
    assert mapa.location == [deposito["latitude"], deposito["longitude"]]


def test_mapa_sem_rota_nao_tem_polyline(df):
    """Sem rota fornecida, não deve haver linha desenhada."""
    mapa = criar_mapa(df)
    tem_polyline = any(
        isinstance(child, folium.PolyLine)
        for child in mapa._children.values()
    )
    assert not tem_polyline


def test_mapa_com_rota_tem_polyline(df):
    """Com rota fornecida, deve haver uma linha desenhada."""
    rota = df[df["tipo"] != "deposito"]["id"].tolist()[:5]
    mapa = criar_mapa(df, rota=rota)
    tem_polyline = any(
        isinstance(child, folium.PolyLine)
        for child in mapa._children.values()
    )
    assert tem_polyline


def test_todos_os_tipos_tem_cor_definida(df):
    """Cada tipo presente nos dados deve ter uma cor mapeada."""
    for tipo in df["tipo"].unique():
        assert tipo in CORES_TIPO, f"Tipo '{tipo}' sem cor definida"


def test_todos_os_tipos_tem_rotulo(df):
    """Cada tipo presente nos dados deve ter um rótulo legível."""
    for tipo in df["tipo"].unique():
        assert tipo in ROTULOS_TIPO, f"Tipo '{tipo}' sem rótulo definido"


def test_popup_contem_dados_do_ponto(df):
    """O popup deve conter o nome e a prioridade do ponto."""
    linha = df[df["tipo"] != "deposito"].iloc[0]
    popup = _popup_html(linha)
    assert linha["nome"] in popup
    assert str(int(linha["prioridade"])) in popup


def test_popup_escapa_html():
    """Conteúdo do popup deve ser escapado (proteção contra injeção)."""
    import pandas as pd
    linha_maliciosa = pd.Series({
        "nome": "<script>alert(1)</script>",
        "tipo": "consulta_preventiva",
        "prioridade": 5,
        "horario_inicio": 8,
        "horario_fim": 14,
        "demanda_kg": 1.0,
        "protocolo": "padrao",
    })
    popup = _popup_html(linha_maliciosa)
    assert "<script>" not in popup
    assert "&lt;script&gt;" in popup


def test_salvar_mapa_cria_html(tmp_path, df):
    mapa = criar_mapa(df)
    caminho = tmp_path / "mapa_teste.html"
    path = salvar_mapa(mapa, str(caminho))
    assert path.exists()
    conteudo = path.read_text(encoding="utf-8")
    assert "<html" in conteudo.lower()
    assert "leaflet" in conteudo.lower()  # folium usa Leaflet.js
