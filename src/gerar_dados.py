"""
Gerador de dados sintéticos para o sistema de otimização de rotas médicas.

Simula pontos de atendimento domiciliar especializado à saúde da mulher em
São Paulo. Os dados são 100% sintéticos — nenhuma informação real de paciente
é utilizada, garantindo privacidade e conformidade ética.

Contexto geográfico: coordenadas próximas ao centro de São Paulo (lat -23.55,
lon -46.63), com dispersão de ~0.15 graus (~16 km), simulando a área de
cobertura de uma rede hospitalar urbana.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Seed fixo garante reprodutibilidade total entre execuções.
# Não usar random.random() — sem seed, resultados mudam a cada run e
# impossibilita comparação de experimentos.
SEED = 42

# Ponto central: referência geográfica fictícia em São Paulo
LAT_BASE = -23.5505
LON_BASE = -46.6333
DISPERSAO = 0.15  # graus (~16 km de raio)

# Tipos de atendimento com prioridade (1 = máxima urgência)
# Ordem definida pelo enunciado oficial (pág. 7 do doc da Secretaria)
TIPOS = {
    "emergencia_obstetrica":  {"prioridade": 1, "demanda_kg": 2.0,  "janela_h": 1},
    "violencia_domestica":    {"prioridade": 2, "demanda_kg": 1.5,  "janela_h": 2},
    "medicamento_hormonal":   {"prioridade": 3, "demanda_kg": 0.5,  "janela_h": 3},
    "acompanhamento_pos_parto": {"prioridade": 4, "demanda_kg": 3.0, "janela_h": 4},
    "consulta_preventiva":    {"prioridade": 5, "demanda_kg": 1.0,  "janela_h": 6},
}

# Protocolos de segurança por tipo de atendimento
# Violência doméstica: entrega discreta em horário seguro (sem identificação externa)
PROTOCOLOS = {
    "emergencia_obstetrica":    "atendimento_imediato",
    "violencia_domestica":      "protocolo_discreto",
    "medicamento_hormonal":     "cadeia_frio",
    "acompanhamento_pos_parto": "agendado",
    "consulta_preventiva":      "padrao",
}

# Horários de entrega mais seguros por tipo (hora de início da janela)
# Hora de início da janela de atendimento (formato 24h).
# O motivo específico de cada tipo está em PROTOCOLOS.
HORARIO_INICIO_BASE = {
    "emergencia_obstetrica":    0,   # qualquer hora
    "violencia_domestica":      8,   # horário comercial
    "medicamento_hormonal":     7,   # cedo — hormonais geralmente tomados em jejum/manhã
    "acompanhamento_pos_parto": 8,   # horário comercial
    "consulta_preventiva":      8,   # horário comercial
}


def _gerar_ponto_base(rng: np.random.Generator) -> dict:
    """Gera coordenadas aleatórias dentro da área de cobertura."""
    lat = LAT_BASE + rng.uniform(-DISPERSAO, DISPERSAO)
    lon = LON_BASE + rng.uniform(-DISPERSAO, DISPERSAO)
    return {"latitude": round(lat, 6), "longitude": round(lon, 6)}


def gerar_pontos(n: int = 30) -> pd.DataFrame:
    """
    Gera n pontos de atendimento sintéticos + 1 ponto-base (depósito/hospital).

    Colunas do DataFrame resultante:
        id              : identificador único (int, 0 = depósito)
        nome            : nome descritivo do ponto
        tipo            : categoria de atendimento
        prioridade      : 1 (máxima urgência) a 5 (preventivo)
        latitude        : coordenada geográfica (fictícia)
        longitude       : coordenada geográfica (fictícia)
        demanda_kg      : peso/volume de suprimentos necessários
        janela_horas    : duração máxima da janela de atendimento (em horas)
        horario_inicio  : hora de início da janela de atendimento (0-23)
        horario_fim     : hora de fim da janela de atendimento (0-23)
        protocolo       : protocolo de segurança aplicável
        tempo_servico_min: tempo estimado de atendimento no local (minutos)
    """
    # rng criado aqui — cada chamada reinicia do SEED, garantindo reprodutibilidade
    rng = np.random.default_rng(SEED)

    tipos_lista = list(TIPOS.keys())

    # Distribuição de frequência reflete realidade clínica:
    # preventivo é mais comum que emergência
    pesos = [0.10, 0.15, 0.25, 0.25, 0.25]

    tipos_escolhidos = rng.choice(tipos_lista, size=n, p=pesos)

    registros = []
    for i, tipo in enumerate(tipos_escolhidos):
        meta = TIPOS[tipo]
        inicio = HORARIO_INICIO_BASE[tipo]
        fim = min(inicio + meta["janela_h"], 22)

        # Variação realista no tempo de serviço por tipo
        tempo_base = {"emergencia_obstetrica": 45, "violencia_domestica": 60,
                      "medicamento_hormonal": 15, "acompanhamento_pos_parto": 30,
                      "consulta_preventiva": 20}
        tempo = int(rng.integers(
            tempo_base[tipo] - 5,
            tempo_base[tipo] + 15
        ))

        ponto = _gerar_ponto_base(rng)
        registros.append({
            "id": i + 1,
            "nome": f"Ponto_{i + 1:02d}_{tipo[:4].upper()}",
            "tipo": tipo,
            "prioridade": meta["prioridade"],
            "latitude": ponto["latitude"],
            "longitude": ponto["longitude"],
            "demanda_kg": round(meta["demanda_kg"] + rng.uniform(-0.2, 0.2), 2),
            "janela_horas": meta["janela_h"],
            "horario_inicio": inicio,
            "horario_fim": fim,
            "protocolo": PROTOCOLOS[tipo],
            "tempo_servico_min": tempo,
        })

    # Ponto 0: depósito/hospital-base (ponto de partida e chegada de todos os veículos)
    deposito = {
        "id": 0,
        "nome": "Hospital_Base",
        "tipo": "deposito",
        "prioridade": 0,
        "latitude": LAT_BASE,
        "longitude": LON_BASE,
        "demanda_kg": 0.0,
        "janela_horas": 24,
        "horario_inicio": 0,
        "horario_fim": 23,
        "protocolo": "base",
        "tempo_servico_min": 0,
    }

    df = pd.DataFrame([deposito] + registros)
    df = df.sort_values(["prioridade", "id"]).reset_index(drop=True)
    return df


def salvar_csv(df: pd.DataFrame, caminho: str = "data/pontos.csv") -> Path:
    """Salva o DataFrame em CSV. Cria o diretório se não existir."""
    path = Path(caminho)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def resumo(df: pd.DataFrame) -> None:
    """Imprime um resumo dos pontos gerados."""
    print(f"\n{'='*50}")
    print(f"Pontos gerados: {len(df)} (incluindo depósito)")
    print(f"\nDistribuição por tipo:")
    contagem = df[df["tipo"] != "deposito"]["tipo"].value_counts()
    for tipo, qtd in contagem.items():
        prioridade = TIPOS[tipo]["prioridade"]
        print(f"  P{prioridade} | {tipo:<30} {qtd:>3} pontos")
    print(f"\nDemanda total de suprimentos: {df['demanda_kg'].sum():.1f} kg")
    print(f"Área de cobertura: ~{DISPERSAO * 111:.0f} km de raio")
    print(f"Seed usado (reprodutibilidade): {SEED}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    df = gerar_pontos(n=30)
    path = salvar_csv(df)
    resumo(df)
    print(f"Arquivo salvo em: {path.resolve()}")
    print(df[["id", "nome", "tipo", "prioridade",
              "horario_inicio", "horario_fim", "protocolo"]].to_string(index=False))
