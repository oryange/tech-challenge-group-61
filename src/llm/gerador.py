"""
Integração com LLM local (Hugging Face / flan-t5) para geração de
instruções e relatórios de rota.

Gera três tipos de output a partir de uma rota de pontos de atendimento:
    1. Manual de instruções — documento prático para a equipe de transporte,
       com orientações específicas por tipo de atendimento;
    2. Roteiro detalhado — sequência legível de visitas com tempos estimados;
    3. Chat em linguagem natural — responde perguntas sobre a rota.

Decisão de privacidade: LLM roda 100% local (sem API externa).
Nenhum dado de localização de paciente sai da máquina — especialmente
relevante para pontos de violência doméstica (protocolo discreto).

Nota sobre prompt injection: inputs do usuário (perguntas no chat)
são sanitizados antes de compor o prompt — caracteres de controle e
sequências que tentam "escapar" do template são removidos.
"""

import re
import os
from functools import lru_cache
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Configuração do modelo
# ---------------------------------------------------------------------------

# Modelo padrão: flan-t5-base equilibra qualidade e velocidade.
# Pode ser sobrescrito via variável de ambiente HF_MODEL (configurável no .env).
# flan-t5-small → mais leve; flan-t5-large → melhor qualidade (mais RAM).
_MODELO_PADRAO = os.getenv("HF_MODEL", "google/flan-t5-large")

# Rótulos legíveis para os tipos de atendimento
_ROTULOS = {
    "emergencia_obstetrica":    "Emergência obstétrica",
    "violencia_domestica":      "Violência doméstica",
    "medicamento_hormonal":     "Medicamento hormonal",
    "acompanhamento_pos_parto": "Acompanhamento pós-parto",
    "consulta_preventiva":      "Consulta preventiva",
    "deposito":                 "Hospital-base",
}

# Instruções fixas de segurança por protocolo (não geradas pela LLM —
# evita que o modelo invente protocolos médicos incorretos)
_INSTRUCOES_PROTOCOLO = {
    "atendimento_imediato": (
        "URGENTE: atendimento imediato. Acione a equipe médica ao chegar. "
        "Não aguarde confirmação da paciente."
    ),
    "protocolo_discreto": (
        "PROTOCOLO DISCRETO: veículo sem identificação hospitalar. "
        "Entrar em contato apenas pelo número seguro previamente combinado. "
        "Não revelar natureza do atendimento a terceiros."
    ),
    "cadeia_frio": (
        "CADEIA DE FRIO: medicamento deve permanecer refrigerado até a entrega. "
        "Verificar temperatura da embalagem antes de entregar."
    ),
    "agendado": (
        "Visita agendada. Confirmar presença da paciente antes de partir."
    ),
    "padrao": (
        "Atendimento preventivo de rotina."
    ),
    "base": (
        "Ponto de origem/retorno — hospital-base."
    ),
}


# ---------------------------------------------------------------------------
# Carregamento lazy do modelo (apenas na primeira chamada)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _carregar_modelo(nome_modelo: str):
    """
    Carrega tokenizador e modelo uma única vez (cache em memória).
    Na primeira execução baixa o modelo do Hugging Face Hub e cacheia em
    ~/.cache/huggingface — execuções subsequentes usam o cache local.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(nome_modelo)
        model = AutoModelForSeq2SeqLM.from_pretrained(nome_modelo)
        return tok, model
    except Exception as e:
        raise RuntimeError(
            f"Falha ao carregar '{nome_modelo}'. "
            f"Na primeira execução é necessário conexão com a internet.\n"
            f"Detalhe: {e}"
        ) from e


def _gerar_texto(prompt: str, max_tokens: int = 120,
                 modelo: str = _MODELO_PADRAO) -> str:
    """
    Chama o modelo local e retorna o texto gerado.
    Centraliza a chamada para facilitar substituição futura do modelo.
    """
    tok, model = _carregar_modelo(modelo)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    saida = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=4,           # beam search melhora coerência
        early_stopping=True,
        repetition_penalty=1.3,  # evita loops de repetição sem distorcer o output
        no_repeat_ngram_size=3,  # bloqueia repetição de trigramas
    )
    return tok.decode(saida[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Sanitização de input do usuário (proteção contra prompt injection)
# ---------------------------------------------------------------------------

def _sanitizar_pergunta(texto: str) -> str:
    """
    Remove padrões que tentam manipular o template do prompt:
    - Sequências de nova linha excessivas (tentativa de "quebrar" o prompt)
    - Prefixos de instrução (Ignore previous, System:, etc.)
    - Caracteres de controle
    Retorna texto limitado a 200 caracteres.
    """
    # Remove caracteres de controle e múltiplas quebras de linha
    texto = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", texto)
    texto = re.sub(r"\n{2,}", " ", texto)
    # Remove tentativas comuns de prompt injection
    texto = re.sub(
        r"(?i)(ignore\s+(previous|all)|system\s*:|assistant\s*:|<\|.*?\|>)",
        "",
        texto,
    )
    return texto.strip()[:200]


# ---------------------------------------------------------------------------
# Funções principais
# ---------------------------------------------------------------------------

# Cache de dicas por (rotulo, protocolo) — evita N chamadas LLM para o mesmo tipo
_dica_cache: dict = {}


def _dica_para_tipo(rotulo: str, protocolo: str, modelo: str) -> str:
    chave = (rotulo, protocolo, modelo)
    if chave not in _dica_cache:
        prompt = (
            f"Generate a one-sentence practical tip in Portuguese for a "
            f"medical delivery team visiting a {rotulo} patient "
            f"({protocolo} protocol):"
        )
        _dica_cache[chave] = _gerar_texto(prompt, max_tokens=60, modelo=modelo)
    return _dica_cache[chave]


def gerar_roteiro(df_rota: pd.DataFrame,
                  velocidade_kmh: float = 40.0,
                  hora_inicio: float = 8.0,
                  modelo: str = _MODELO_PADRAO) -> str:
    """
    Gera o roteiro detalhado da rota em linguagem natural.

    A estrutura (ordem, horários estimados, tipo de cada parada) é montada
    em Python — o modelo LLM enriquece cada parada com uma observação
    contextual curta.

    Args:
        df_rota:        DataFrame dos pontos na ordem de visita (sem depósito).
        velocidade_kmh: velocidade média para estimar tempo de deslocamento.
        hora_inicio:    horário de início do expediente (padrão: 8.0 = 08h00).
        modelo:         modelo Hugging Face a usar.

    Returns:
        String com o roteiro completo.
    """
    if df_rota.empty:
        return "Nenhum ponto de atendimento na rota."

    from src.utils import haversine

    linhas = ["=== ROTEIRO DO DIA ===\n"]
    hora = hora_inicio
    lat_ant, lon_ant = None, None

    for posicao, (_, ponto) in enumerate(df_rota.iterrows(), start=1):
        tipo = ponto.get("tipo", "")
        rotulo = _ROTULOS.get(tipo, tipo)
        protocolo = ponto.get("protocolo", "padrao")

        # Tempo de deslocamento
        if lat_ant is not None:
            dist = haversine(lat_ant, lon_ant,
                             ponto["latitude"], ponto["longitude"])
            hora += dist / velocidade_kmh
        lat_ant, lon_ant = ponto["latitude"], ponto["longitude"]

        # Aguarda janela se chegou cedo
        if hora < ponto.get("horario_inicio", 0):
            hora = float(ponto["horario_inicio"])

        hora_str = f"{int(hora):02d}h{int((hora % 1) * 60):02d}"

        dica = _dica_para_tipo(rotulo, protocolo, modelo)

        linhas.append(
            f"Parada {posicao:02d} — {hora_str} | {rotulo}\n"
            f"  Ponto:     {ponto['nome']}\n"
            f"  Protocolo: {_INSTRUCOES_PROTOCOLO.get(protocolo, '')}\n"
            f"  Dica:      {dica}\n"
        )
        hora += ponto.get("tempo_servico_min", 15) / 60.0

    linhas.append(f"\nTotal de paradas: {len(df_rota)}")
    return "\n".join(linhas)


def gerar_manual(df_rota: pd.DataFrame,
                 modelo: str = _MODELO_PADRAO) -> str:
    """
    Gera o manual de instruções para a equipe de transporte.

    O manual consolida os protocolos de segurança de todos os tipos de
    atendimento presentes na rota, com contexto gerado pela LLM para
    tornar as instruções mais claras e práticas.

    Args:
        df_rota: DataFrame dos pontos da rota (sem depósito).
        modelo:  modelo Hugging Face a usar.

    Returns:
        String com o manual completo.
    """
    if df_rota.empty:
        return "Nenhum ponto de atendimento na rota."

    tipos_presentes = df_rota["tipo"].unique()
    emergencias = len(df_rota[df_rota["tipo"] == "emergencia_obstetrica"])
    violencia = len(df_rota[df_rota["tipo"] == "violencia_domestica"])
    total = len(df_rota)
    carga = df_rota["demanda_kg"].sum()

    # Resumo gerado pelo LLM
    prompt_resumo = (
        f"Write a brief safety briefing in Portuguese for a medical delivery "
        f"team with {total} stops, including {emergencias} obstetric "
        f"emergencies and {violencia} domestic violence cases. "
        f"Total cargo: {carga:.1f} kg. Be concise and practical:"
    )
    resumo = _gerar_texto(prompt_resumo, max_tokens=100, modelo=modelo)

    secoes = [
        "=== MANUAL DE INSTRUÇÕES — EQUIPE DE TRANSPORTE ===\n",
        f"Briefing do dia:\n{resumo}\n",
        "--- PROTOCOLOS POR TIPO DE ATENDIMENTO ---\n",
    ]

    for tipo in tipos_presentes:
        if tipo == "deposito":
            continue
        rotulo = _ROTULOS.get(tipo, tipo)
        qtd = len(df_rota[df_rota["tipo"] == tipo])
        protocolo_chave = df_rota[df_rota["tipo"] == tipo]["protocolo"].iloc[0]
        instrucao_fixa = _INSTRUCOES_PROTOCOLO.get(protocolo_chave, "")

        prompt_tipo = (
            f"In Portuguese, write one practical sentence advising a delivery "
            f"team about {rotulo} visits in a hospital context:"
        )
        conselho = _gerar_texto(prompt_tipo, max_tokens=60, modelo=modelo)

        secoes.append(
            f"{rotulo} ({qtd} parada{'s' if qtd > 1 else ''})\n"
            f"  {instrucao_fixa}\n"
            f"  {conselho}\n"
        )

    return "\n".join(secoes)


def responder_pergunta(pergunta: str,
                       df_rota: pd.DataFrame,
                       modelo: str = _MODELO_PADRAO) -> str:
    """
    Responde a uma pergunta em linguagem natural sobre a rota.

    O contexto da rota é extraído estruturalmente (contagens, tipos, horários)
    e combinado com a pergunta para gerar uma resposta via LLM.
    O input do usuário é sanitizado antes de compor o prompt.

    Args:
        pergunta: pergunta em português sobre a rota.
        df_rota:  DataFrame dos pontos da rota (sem depósito).
        modelo:   modelo Hugging Face a usar.

    Returns:
        Resposta em texto gerada pelo modelo.
    """
    pergunta_limpa = _sanitizar_pergunta(pergunta)
    if not pergunta_limpa:
        return "Por favor, faça uma pergunta sobre a rota."

    if df_rota.empty:
        return "Não há rota carregada para responder perguntas."

    # Contexto estruturado da rota (gerado por Python, não pelo usuário)
    emergencias = df_rota[df_rota["tipo"] == "emergencia_obstetrica"]
    primeira_emerg = (
        emergencias.iloc[0]["nome"] if not emergencias.empty else "nenhuma"
    )
    contagens = df_rota.groupby("tipo").size().to_dict()
    resumo_contagens = ", ".join(
        f"{_ROTULOS.get(t, t)}: {n}" for t, n in contagens.items()
    )

    contexto = (
        f"Route summary: {len(df_rota)} stops. "
        f"Counts: {resumo_contagens}. "
        f"First emergency: {primeira_emerg}. "
        f"Total cargo: {df_rota['demanda_kg'].sum():.1f} kg."
    )

    prompt = (
        f"Context: {contexto}\n"
        f"Question in Portuguese: {pergunta_limpa}\n"
        f"Answer in Portuguese:"
    )

    return _gerar_texto(prompt, max_tokens=80, modelo=modelo)


def gerar_relatorio_resumo(df_rota: pd.DataFrame,
                           fitness: Optional[float] = None,
                           modelo: str = _MODELO_PADRAO) -> str:
    """
    Gera um relatório de eficiência da rota para uso no relatório técnico
    e no app Streamlit.

    Args:
        df_rota:  DataFrame dos pontos da rota.
        fitness:  valor de fitness da rota (opcional — se disponível do AG).
        modelo:   modelo Hugging Face a usar.

    Returns:
        String com o relatório de eficiência.
    """
    if df_rota.empty:
        return "Nenhum dado disponível para o relatório."

    total = len(df_rota)
    emergencias = len(df_rota[df_rota["tipo"] == "emergencia_obstetrica"])
    carga = df_rota["demanda_kg"].sum()
    fitness_str = f"{fitness:.2f}" if fitness is not None else "não calculado"

    prompt = (
        f"Write a brief efficiency report in Portuguese for a medical route "
        f"with {total} stops, {emergencias} emergencies, "
        f"{carga:.1f} kg cargo, fitness score {fitness_str}. "
        f"Include time savings and patient safety impact:"
    )
    analise = _gerar_texto(prompt, max_tokens=100, modelo=modelo)

    contagens = df_rota.groupby("tipo").size()

    linhas = [
        "=== RELATÓRIO DE EFICIÊNCIA DA ROTA ===\n",
        f"Total de paradas:   {total}",
        f"Emergências:        {emergencias}",
        f"Carga total:        {carga:.1f} kg",
        f"Score de fitness:   {fitness_str}",
        "\nDistribuição de atendimentos:",
    ]
    for tipo, qtd in contagens.items():
        linhas.append(f"  {_ROTULOS.get(tipo, tipo):<35} {qtd}")

    linhas += ["\nAnálise:", analise]
    return "\n".join(linhas)


if __name__ == "__main__":
    df = pd.read_csv("data/pontos.csv")
    rota = df[df["tipo"] != "deposito"].head(5)

    print("Carregando modelo (aguarde na primeira vez)...\n")

    print("=" * 60)
    print("1. ROTEIRO DETALHADO")
    print("=" * 60)
    print(gerar_roteiro(rota))

    print("\n" + "=" * 60)
    print("2. MANUAL DE INSTRUÇÕES")
    print("=" * 60)
    print(gerar_manual(rota))

    print("\n" + "=" * 60)
    print("3. CHAT")
    print("=" * 60)
    for pergunta in [
        "Quantas emergências temos hoje?",
        "Qual é o primeiro atendimento urgente?",
        "Quantas paradas de violência doméstica?",
    ]:
        print(f"P: {pergunta}")
        print(f"R: {responder_pergunta(pergunta, rota)}\n")

    print("=" * 60)
    print("4. RELATÓRIO DE EFICIÊNCIA")
    print("=" * 60)
    print(gerar_relatorio_resumo(rota, fitness=1250.5))
