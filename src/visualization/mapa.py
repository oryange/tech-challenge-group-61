"""
Visualização de pontos e rotas de atendimento médico em um mapa interativo.

Usa folium (Leaflet.js) para renderizar:
    - Os pontos de atendimento, coloridos por tipo (codificação obrigatória
      pelo enunciado);
    - O depósito/hospital-base com destaque;
    - Opcionalmente, uma rota otimizada desenhada como linha sobre o mapa.

O resultado é um arquivo HTML interativo, ideal para a demonstração em vídeo
e para a interface Streamlit.

Nota de segurança: os popups exibem dados sintéticos gerados pelo próprio
projeto (sem entrada de usuário). Ainda assim, o conteúdo textual é escapado
com html.escape antes de ir para o HTML, evitando qualquer injeção acidental.
"""

import html
from pathlib import Path
from typing import List, Optional

import folium
import pandas as pd


# Cores por tipo de atendimento (codificação visual obrigatória)
CORES_TIPO = {
    "emergencia_obstetrica":    "red",        # prioridade máxima
    "violencia_domestica":      "purple",     # protocolo discreto
    "medicamento_hormonal":     "blue",       # cadeia de frio
    "acompanhamento_pos_parto": "green",      # agendado
    "consulta_preventiva":      "lightgray",  # preventivo
    "deposito":                 "black",      # hospital-base
}

# Rótulos legíveis para a legenda
ROTULOS_TIPO = {
    "emergencia_obstetrica":    "Emergência obstétrica",
    "violencia_domestica":      "Violência doméstica",
    "medicamento_hormonal":     "Medicamento hormonal",
    "acompanhamento_pos_parto": "Acompanhamento pós-parto",
    "consulta_preventiva":      "Consulta preventiva",
    "deposito":                 "Hospital-base (depósito)",
}


def _popup_html(linha: pd.Series) -> str:
    """Monta o conteúdo HTML (escapado) do popup de um ponto."""
    nome = html.escape(str(linha["nome"]))
    tipo = html.escape(ROTULOS_TIPO.get(linha["tipo"], str(linha["tipo"])))
    protocolo = html.escape(str(linha["protocolo"]))
    return (
        f"<b>{nome}</b><br>"
        f"Tipo: {tipo}<br>"
        f"Prioridade: {int(linha['prioridade'])}<br>"
        f"Janela: {int(linha['horario_inicio'])}h–{int(linha['horario_fim'])}h<br>"
        f"Demanda: {linha['demanda_kg']} kg<br>"
        f"Protocolo: {protocolo}"
    )


def _adicionar_legenda(mapa: folium.Map) -> None:
    """Adiciona uma legenda fixa de cores por tipo no canto do mapa."""
    itens = "".join(
        f'<div style="margin:2px 0;">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{CORES_TIPO[tipo]};border:1px solid #555;margin-right:6px;"></span>'
        f'{html.escape(rotulo)}</div>'
        for tipo, rotulo in ROTULOS_TIPO.items()
    )
    legenda = (
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;'
        'background:white;padding:10px 14px;border:1px solid #999;'
        'border-radius:6px;font-size:12px;font-family:sans-serif;'
        'box-shadow:0 1px 4px rgba(0,0,0,0.3);">'
        '<b>Tipo de atendimento</b>'
        f'{itens}</div>'
    )
    mapa.get_root().html.add_child(folium.Element(legenda))


# Paleta para diferenciar as rotas de cada veículo (uma cor por veículo)
_CORES_VEICULO = [
    "#2c3e50", "#e67e22", "#16a085", "#8e44ad", "#c0392b",
    "#2980b9", "#d35400", "#27ae60", "#7f8c8d", "#f39c12",
]


def criar_mapa(
    df: pd.DataFrame,
    rota: Optional[List[int]] = None,
    rotas: Optional[List[List[int]]] = None,
    zoom: int = 12,
) -> folium.Map:
    """
    Cria um mapa interativo com os pontos de atendimento.

    Args:
        df:    DataFrame gerado por gerar_dados.py
        rota:  lista opcional de IDs na ordem de visita (uma única rota).
               Se fornecida, desenha a rota como linha (depósito → pontos → depósito).
        rotas: lista opcional de sub-rotas por veículo. Se fornecida, desenha
               cada veículo com uma cor distinta (depósito → pontos → depósito).
               Tem precedência sobre `rota`.
        zoom:  nível de zoom inicial do mapa.

    Returns:
        folium.Map pronto para salvar ou exibir.
    """
    deposito = df[df["tipo"] == "deposito"].iloc[0]
    centro = [deposito["latitude"], deposito["longitude"]]

    mapa = folium.Map(location=centro, zoom_start=zoom, tiles="OpenStreetMap")

    # Marcadores dos pontos
    for _, linha in df.iterrows():
        cor = CORES_TIPO.get(linha["tipo"], "gray")
        is_deposito = linha["tipo"] == "deposito"
        folium.Marker(
            location=[linha["latitude"], linha["longitude"]],
            popup=folium.Popup(_popup_html(linha), max_width=250),
            tooltip=html.escape(str(linha["nome"])),
            icon=folium.Icon(
                color=cor if cor in _CORES_FOLIUM else "gray",
                icon="plus-sign" if is_deposito else "info-sign",
            ),
        ).add_to(mapa)

    idx = df.set_index("id")

    def _desenhar_rota(seq_ids: List[int], cor: str, rotulo: str) -> None:
        sequencia = [0] + list(seq_ids) + [0]  # parte e volta ao depósito
        coords = [
            [idx.loc[pid, "latitude"], idx.loc[pid, "longitude"]]
            for pid in sequencia
        ]
        folium.PolyLine(
            coords, color=cor, weight=3, opacity=0.8, tooltip=rotulo,
        ).add_to(mapa)

    # Múltiplas rotas (uma por veículo) têm precedência
    if rotas:
        for i, sub in enumerate(rotas):
            if sub:
                cor = _CORES_VEICULO[i % len(_CORES_VEICULO)]
                _desenhar_rota(sub, cor, f"Veículo {i + 1}")
    elif rota:
        _desenhar_rota(rota, "#2c3e50", "Rota otimizada")

    _adicionar_legenda(mapa)
    return mapa


# Cores suportadas nativamente pelo folium.Icon (as demais caem em "gray")
_CORES_FOLIUM = {
    "red", "blue", "green", "purple", "orange", "darkred", "lightred",
    "beige", "darkblue", "darkgreen", "cadetblue", "darkpurple", "white",
    "pink", "lightblue", "lightgreen", "gray", "black", "lightgray",
}


def salvar_mapa(mapa: folium.Map, caminho: str = "data/mapa.html") -> Path:
    """Salva o mapa em arquivo HTML. Cria o diretório se necessário."""
    path = Path(caminho)
    path.parent.mkdir(parents=True, exist_ok=True)
    mapa.save(str(path))
    return path


if __name__ == "__main__":
    df = pd.read_csv("data/pontos.csv")
    mapa = criar_mapa(df)
    path = salvar_mapa(mapa)
    print(f"Mapa salvo em: {path.resolve()}")
    print(f"Pontos plotados: {len(df)} (incluindo depósito)")
    print("Abra o arquivo HTML no navegador para visualizar.")
