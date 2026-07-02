"""
Interface Streamlit — Otimização de Rotas Médicas (VRP).

Permite:
  1. Visualizar os pontos de atendimento no mapa;
  2. Configurar e executar o algoritmo genético;
  3. Visualizar as rotas otimizadas por veículo;
  4. Analisar o histórico de convergência;
  5. Interagir com a LLM local para obter o roteiro em linguagem natural.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fitness import ParametrosVeiculo, PesosFitness
from gerar_dados import gerar_pontos, salvar_csv
from genetic_algorithm.vrp import ConfigGA, evoluir
from visualization.mapa import criar_mapa, salvar_mapa, CORES_TIPO, ROTULOS_TIPO

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Rotas Médicas — GA Optimizer",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Otimização de Rotas de Atendimento à Saúde da Mulher")
st.caption(
    "Algoritmo Genético (VRP) · São Paulo · Dados 100% sintéticos · "
    "Nenhuma informação real de paciente é utilizada."
)

# ---------------------------------------------------------------------------
# Sidebar — configuração
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("Dados")
    n_pontos = st.slider("Número de pontos de atendimento", 10, 50, 30, step=5)

    st.subheader("Frota")
    n_veiculos = st.slider("Veículos disponíveis", 1, 5, 3)
    capacidade = st.number_input("Capacidade máx. por veículo (kg)", 5.0, 100.0, 20.0, step=5.0)
    autonomia  = st.number_input("Autonomia máx. por veículo (km)", 20.0, 500.0, 150.0, step=10.0)
    velocidade = st.number_input("Velocidade média (km/h)", 10.0, 100.0, 40.0, step=5.0)

    st.subheader("Algoritmo Genético")
    pop_size   = st.slider("Tamanho da população", 20, 300, 100, step=20)
    n_geracoes = st.slider("Número de gerações", 20, 500, 150, step=10)
    tx_swap    = st.slider("Taxa de mutação swap", 0.0, 1.0, 0.30, step=0.05)
    tx_inv     = st.slider("Taxa de mutação inversão", 0.0, 1.0, 0.10, step=0.05)
    seed       = st.number_input("Seed (reprodutibilidade)", 0, 9999, 42)

    st.subheader("Pesos da Fitness")
    w_prior = st.slider("Peso prioridade", 0.0, 300.0, 120.0, step=10.0)
    w_jan   = st.slider("Peso janela de horário", 0.0, 200.0, 30.0, step=10.0)
    w_cap   = st.slider("Peso capacidade", 0.0, 500.0, 100.0, step=20.0)
    w_aut   = st.slider("Peso autonomia", 0.0, 500.0, 80.0, step=20.0)

    executar = st.button("🚀 Otimizar Rotas", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Carregar / gerar dados
# ---------------------------------------------------------------------------

@st.cache_data
def carregar_dados(n: int) -> pd.DataFrame:
    return gerar_pontos(n=n)


df = carregar_dados(n_pontos)

# ---------------------------------------------------------------------------
# Layout em abas
# ---------------------------------------------------------------------------

aba_mapa, aba_resultado, aba_convergencia, aba_llm = st.tabs(
    ["🗺️ Mapa", "📋 Resultado", "📈 Convergência", "🤖 Assistente LLM"]
)

# ---------------------------------------------------------------------------
# Aba Mapa — estado inicial
# ---------------------------------------------------------------------------

with aba_mapa:
    col_mapa, col_legenda = st.columns([3, 1])

    with col_legenda:
        st.markdown("**Legenda**")
        for tipo, rotulo in ROTULOS_TIPO.items():
            cor_css = {
                "red": "#e74c3c", "purple": "#9b59b6", "blue": "#3498db",
                "green": "#27ae60", "lightgray": "#aaaaaa", "black": "#2c3e50",
            }.get(CORES_TIPO[tipo], "#888")
            st.markdown(
                f'<span style="display:inline-block;width:14px;height:14px;'
                f'background:{cor_css};border-radius:2px;margin-right:6px;'
                f'vertical-align:middle;"></span>{rotulo}',
                unsafe_allow_html=True,
            )

        st.divider()
        st.metric("Total de pontos", len(df) - 1)
        contagem = df[df["tipo"] != "deposito"]["tipo"].value_counts()
        for tipo, qtd in contagem.items():
            st.caption(f"P{df[df['tipo']==tipo]['prioridade'].iloc[0]} | {ROTULOS_TIPO.get(tipo, tipo)}: **{qtd}**")

    with col_mapa:
        if "rotas_resultado" not in st.session_state:
            # Mostra mapa sem rota (apenas pontos)
            mapa_inicial = criar_mapa(df)
            st.components.v1.html(mapa_inicial._repr_html_(), height=500, scrolling=False)
        else:
            # Mostra rotas otimizadas — cada veículo com uma cor distinta
            rotas = st.session_state["rotas_resultado"]
            mapa_otim = criar_mapa(df, rotas=rotas)
            st.components.v1.html(mapa_otim._repr_html_(), height=500, scrolling=False)
            n_ativos = sum(1 for r in rotas if r)
            st.caption(f"Rotas exibidas: {n_ativos} veículo(s), cada um com uma cor distinta.")

# ---------------------------------------------------------------------------
# Executar GA
# ---------------------------------------------------------------------------

if executar:
    config = ConfigGA(
        tamanho_populacao=pop_size,
        n_geracoes=n_geracoes,
        taxa_mutacao_swap=tx_swap,
        taxa_mutacao_inversao=tx_inv,
        n_veiculos=n_veiculos,
        seed=int(seed),
    )
    veiculo = ParametrosVeiculo(
        capacidade_max_kg=capacidade,
        autonomia_max_km=autonomia,
        velocidade_kmh=velocidade,
    )
    pesos = PesosFitness(
        prioridade=w_prior,
        janela=w_jan,
        capacidade=w_cap,
        autonomia=w_aut,
    )

    barra = st.progress(0, text="Iniciando evolução...")
    status_txt = st.empty()

    def atualizar_progresso(geracao: int, melhor_fit: float):
        pct = int((geracao + 1) / n_geracoes * 100)
        barra.progress(pct, text=f"Geração {geracao + 1}/{n_geracoes} — melhor fitness: {melhor_fit:.2f}")
        status_txt.caption(f"Evolução em andamento… geração {geracao + 1}")

    with st.spinner("Algoritmo genético em execução…"):
        resultado = evoluir(df, config=config, veiculo=veiculo, pesos=pesos,
                            callback=atualizar_progresso)

    barra.progress(100, text="Concluído!")
    status_txt.empty()

    st.session_state["rotas_resultado"]    = resultado.rotas
    st.session_state["historico_fitness"]  = resultado.historico_fitness
    st.session_state["historico_media"]    = resultado.historico_media
    st.session_state["melhor_fitness"]     = resultado.melhor_fitness
    st.session_state["df"]                 = df
    st.session_state["n_veiculos"]         = n_veiculos

    st.success(f"✅ Otimização concluída! Melhor fitness: **{resultado.melhor_fitness:.2f}**")
    st.rerun()

# ---------------------------------------------------------------------------
# Aba Resultado
# ---------------------------------------------------------------------------

with aba_resultado:
    if "rotas_resultado" not in st.session_state:
        st.info("Execute a otimização para ver os resultados.")
    else:
        rotas     = st.session_state["rotas_resultado"]
        df_dados  = st.session_state["df"]
        idx       = df_dados.set_index("id")

        st.metric("Melhor fitness total", f"{st.session_state['melhor_fitness']:.2f}")
        st.divider()

        for i, rota in enumerate(rotas):
            if not rota:
                st.markdown(f"**Veículo {i+1}:** ocioso")
                continue

            carga = sum(idx.loc[p, "demanda_kg"] for p in rota)
            n_emerg = sum(1 for p in rota if idx.loc[p, "prioridade"] == 1)

            with st.expander(
                f"🚐 Veículo {i+1} — {len(rota)} paradas | {carga:.1f} kg | {n_emerg} emergências",
                expanded=(i == 0),
            ):
                linhas = []
                for ordem, pid in enumerate(rota, start=1):
                    row = idx.loc[pid]
                    linhas.append({
                        "Ordem": ordem,
                        "Nome": row["nome"],
                        "Tipo": ROTULOS_TIPO.get(row["tipo"], row["tipo"]),
                        "Prioridade": int(row["prioridade"]),
                        "Janela": f"{int(row['horario_inicio'])}h–{int(row['horario_fim'])}h",
                        "Demanda (kg)": row["demanda_kg"],
                        "Protocolo": row["protocolo"],
                    })
                st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Aba Convergência
# ---------------------------------------------------------------------------

with aba_convergencia:
    if "historico_fitness" not in st.session_state:
        st.info("Execute a otimização para ver o gráfico de convergência.")
    else:
        import plotly.graph_objects as go

        hist_best = st.session_state["historico_fitness"]
        hist_avg  = st.session_state["historico_media"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(hist_best) + 1)),
            y=hist_best,
            mode="lines",
            name="Melhor fitness",
            line=dict(color="#2c3e50", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(hist_avg) + 1)),
            y=hist_avg,
            mode="lines",
            name="Fitness médio",
            line=dict(color="#95a5a6", width=1, dash="dot"),
        ))
        fig.update_layout(
            title="Convergência do Algoritmo Genético",
            xaxis_title="Geração",
            yaxis_title="Fitness (menor = melhor)",
            legend=dict(orientation="h"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Fitness inicial", f"{hist_best[0]:.2f}")
        col2.metric("Fitness final",   f"{hist_best[-1]:.2f}")
        melhora = (hist_best[0] - hist_best[-1]) / hist_best[0] * 100
        col3.metric("Melhora", f"{melhora:.1f}%", delta=f"-{melhora:.1f}%")

# ---------------------------------------------------------------------------
# Aba LLM
# ---------------------------------------------------------------------------

with aba_llm:
    st.markdown(
        "**Assistente de rota** — gera roteiro em linguagem natural e "
        "responde perguntas sobre a rota otimizada.\n\n"
        "> ℹ️ O modelo LLM roda **100% local** (Hugging Face). "
        "Nenhum dado sai da máquina — especialmente importante para "
        "pontos de violência doméstica (protocolo discreto)."
    )

    if "rotas_resultado" not in st.session_state:
        st.info("Execute a otimização primeiro para usar o assistente.")
    else:
        try:
            from llm.gerador import (
                gerar_roteiro,
                gerar_relatorio_resumo,
                responder_pergunta,
            )
            llm_disponivel = True
        except ImportError:
            llm_disponivel = False
            st.warning(
                "Módulo LLM não encontrado em `src/llm/gerador.py`. "
                "Faça o merge do PR de integração LLM para habilitar esta funcionalidade."
            )

        if llm_disponivel:
            rotas    = st.session_state["rotas_resultado"]
            df_dados = st.session_state["df"]
            rota_v1  = rotas[0] if rotas else []

            # As funções da LLM esperam um DataFrame com as LINHAS da rota
            # (na ordem de visita), não a lista de IDs. Converte aqui.
            idx_dados = df_dados.set_index("id")
            df_rota_v1 = (
                idx_dados.loc[rota_v1].reset_index()
                if rota_v1 else df_dados.iloc[0:0]
            )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📋 Gerar roteiro do dia"):
                    with st.spinner("Gerando roteiro…"):
                        try:
                            roteiro = gerar_roteiro(df_rota_v1)
                            st.markdown("**Roteiro — Veículo 1**")
                            st.text(roteiro)
                        except Exception as e:
                            st.error(f"Erro ao gerar roteiro: {e}")

            with col_btn2:
                if st.button("📊 Gerar relatório resumo"):
                    with st.spinner("Gerando relatório…"):
                        try:
                            relatorio = gerar_relatorio_resumo(
                                df_rota_v1,
                                fitness=st.session_state.get("melhor_fitness"),
                            )
                            st.markdown("**Relatório**")
                            st.text(relatorio)
                        except Exception as e:
                            st.error(f"Erro ao gerar relatório: {e}")

            st.divider()
            st.markdown("**Faça uma pergunta sobre a rota:**")
            pergunta = st.text_input(
                "Ex.: Qual é a próxima parada urgente? Há algum protocolo especial?",
                key="pergunta_llm",
            )
            if st.button("Enviar pergunta") and pergunta:
                with st.spinner("Consultando LLM…"):
                    try:
                        resposta = responder_pergunta(pergunta, df_rota_v1)
                        st.markdown(f"**Resposta:** {resposta}")
                    except Exception as e:
                        st.error(f"Erro: {e}")
