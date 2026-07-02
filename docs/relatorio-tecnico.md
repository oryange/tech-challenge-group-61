# Relatório Técnico — Tech Challenge Fase 2
## Otimização de Rotas de Atendimento à Saúde da Mulher (VRP)

**Grupo 61** | PosTech FIAP — Inteligência Artificial para Devs  
**Projeto:** Projeto 2 — Otimização de Rotas Médicas  
**Repositório:** https://github.com/oryange/tech-challenge-group-61

---

## 1. Visão Geral do Sistema

O projeto implementa um sistema de otimização de rotas diárias para atendimento especializado à saúde da mulher em São Paulo. Uma frota de veículos precisa cobrir pontos de atendimento domiciliar com diferentes graus de urgência — de emergências obstétricas (prioridade máxima) a consultas preventivas — respeitando restrições de capacidade, autonomia e janelas de horário.

O núcleo da solução é um **Algoritmo Genético para VRP** (Vehicle Routing Problem), adaptado do código-base TSP fornecido pela disciplina. Uma **LLM local** (Hugging Face `flan-t5-large`) transforma as rotas otimizadas em linguagem natural para as equipes de campo.

Todos os dados são **100% sintéticos** — nenhuma informação real de paciente é utilizada. Esta decisão é especialmente importante dado que o domínio inclui pontos de violência doméstica, cujo protocolo de segurança exige discrição total.

---

## 2. Arquitetura da Solução

> Os diagramas em formato Mermaid (componentes, fluxo de dados e dependências
> entre módulos) estão em [`docs/arquitetura.md`](arquitetura.md). O diagrama
> abaixo resume o fluxo principal.

```
┌─────────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit)                        │
│   Mapa interativo │ Resultado por veículo │ Convergência │ Chat  │
└────────────┬──────────────────┬───────────────────┬─────────────┘
             │                  │                   │
    ┌────────▼───────┐  ┌───────▼──────┐  ┌────────▼────────┐
    │ gerar_dados.py │  │ genetic_     │  │  llm/gerador.py  │
    │                │  │ algorithm/   │  │                  │
    │ 30 pontos      │  │ vrp.py       │  │ flan-t5-large    │
    │ sintéticos     │  │ baselines.py │  │ (local, offline) │
    │ (São Paulo)    │  └───────┬──────┘  └─────────────────-┘
    └────────┬───────┘          │
             │         ┌────────▼────────┐
             └────────►│  fitness.py     │
                       │                 │
                       │ distância       │
                       │ + prioridade    │
                       │ + janela        │
                       │ + capacidade    │
                       │ + autonomia     │
                       └─────────────────┘
                                │
                       ┌────────▼────────┐
                       │ visualization/  │
                       │ mapa.py (Folium)│
                       └─────────────────┘
```

**Fluxo de execução:**
1. `gerar_dados.py` cria os pontos sintéticos e salva em `data/pontos.csv`
2. O usuário configura frota e parâmetros do GA via interface Streamlit
3. `vrp.py` executa a evolução e retorna a melhor solução
4. `mapa.py` renderiza as rotas no mapa interativo
5. `llm/gerador.py` gera o roteiro e responde perguntas em linguagem natural

---

## 3. Dados Sintéticos

O gerador (`src/gerar_dados.py`) cria pontos de atendimento com as seguintes características:

| Tipo | Prioridade | Protocolo | Janela | Frequência |
|---|---|---|---|---|
| Emergência obstétrica | 1 (máxima) | Atendimento imediato | Qualquer hora | 10% |
| Violência doméstica | 2 | Protocolo discreto | Horário comercial | 15% |
| Medicamento hormonal | 3 | Cadeia de frio | A partir das 7h | 25% |
| Acompanhamento pós-parto | 4 | Agendado | Horário comercial | 25% |
| Consulta preventiva | 5 | Padrão | Horário comercial | 25% |

**Contexto geográfico:** coordenadas fictícias próximas ao centro de São Paulo (lat -23.55, lon -46.63), com dispersão de ~16 km de raio. Seed fixo (42) garante reprodutibilidade total entre experimentos.

---

## 4. Algoritmo Genético — Implementação

### 4.1 Ponto de partida: repositório TSP base

O código base fornecido (`sergiopolimante/genetic_algorithm_tsp`) resolve o TSP clássico: uma rota, um veículo, minimizar distância euclidiana. Nossa implementação parte diretamente desse código e o estende para VRP.

| Componente | TSP base | Nossa implementação VRP |
|---|---|---|
| Cromossomo | `List[Tuple[float, float]]` (coordenadas) | `List[int]` (IDs de paradas) |
| `generate_random_population` | → | `gerar_populacao()` — mesma lógica |
| `order_crossover` (OX) | → | `crossover_ox()` — algoritmo idêntico, tipo do gene diferente |
| `mutate` (swap adjacente) | → | `mutacao_swap()` — swap aleatório (maior diversidade) |
| — | novo | `mutacao_inversao()` — inverte segmento |
| Roleta | → | `selecao_torneio()` — mais robusto com penalidades |
| 1 veículo | → | `cromossomo_para_rotas()` — decodifica 1 permutação em N rotas |
| Distância | → | `fitness_vrp()` — soma 4 componentes ponderados |

### 4.2 Representação do cromossomo

Um cromossomo é uma **permutação dos IDs** de todos os pontos de atendimento (excluindo o depósito 0). Exemplo com 5 pontos:

```
Cromossomo: [3, 1, 5, 2, 4]
```

O decodificador `cromossomo_para_rotas()` percorre essa lista da esquerda para a direita e distribui as paradas entre os veículos: quando adicionar o próximo ponto excederia a capacidade ou autonomia do veículo atual, inicia-se um novo veículo a partir do depósito.

Esta representação garante que **cada ponto é visitado exatamente uma vez** (invariante de permutação), simplificando o crossover e preservando a validade das soluções.

### 4.3 Operadores genéticos

**Order Crossover (OX):** copia um segmento aleatório do parent1 para o filho, preenche as posições restantes com os genes do parent2 na ordem em que aparecem. Preserva posições relativas sem duplicar genes.

```
Parent1: [3, 1 | 5, 2 | 4]      ← segmento [5, 2] copiado
Parent2: [1, 5 | 2, 3 | 4]
Filho:   [1, 3 | 5, 2 | 4]      ← posições livres preenchidas pela ordem do P2
```

**Mutação por troca (swap):** seleciona dois índices aleatórios e troca seus valores. Exploração local eficiente.

**Mutação por inversão:** reverte um segmento aleatório. Explora uma vizinhança estruturalmente diferente — especialmente útil para reorganizar sub-rotas após crossover.

**Seleção por torneio:** escolhe `k` candidatos aleatórios e retorna o melhor. Mais robusto que roleta quando as penalidades têm escalas muito diferentes (capacidade em kg vs. distância em km).

**Elitismo:** os 2 melhores indivíduos passam direto para a próxima geração, garantindo que a melhor solução encontrada nunca se perde.

### 4.4 Função de fitness

```
fitness = distância_total_km
        + w_prioridade  × penalidade_prioridade
        + w_janela      × penalidade_janela
        + w_capacidade  × penalidade_capacidade
        + w_autonomia   × penalidade_autonomia
```

| Componente | O que mede | Peso padrão |
|---|---|---|
| Distância | Km total percorrido (Haversine) | 1× |
| Prioridade | Posição relativa de pontos urgentes na rota | 120× |
| Janela de horário | Horas de atraso fora da janela (dobrado para protocolo discreto) | 30× |
| Capacidade | Excesso de kg acima da capacidade do veículo | 100× |
| Autonomia | Excesso de km acima da autonomia do veículo | 80× |

Os pesos são configuráveis via `PesosFitness`, permitindo os experimentos comparativos.

---

## 5. Restrições Implementadas

O enunciado exige pelo menos 2 restrições além da distância. Implementamos **4 restrições adicionais**:

1. **Ordem de prioridade** (obrigatória pelo enunciado): emergência obstétrica > violência doméstica > medicamento hormonal > pós-parto. Implementada via penalidade proporcional à posição na rota.

2. **Capacidade de carga**: cada veículo tem limite de kg. Exceder gera penalidade e, na decodificação, inicia um novo veículo.

3. **Autonomia (distância máxima)**: cada veículo tem limite de km por rota. Implementado tanto na função de fitness quanto no decodificador.

4. **Janelas de horário**: cada ponto tem horário de início e fim. Chegadas fora da janela geram penalidade proporcional ao atraso. Pontos com `protocolo_discreto` (violência doméstica) têm penalidade dobrada — chegada fora do horário comercial é especialmente prejudicial.

5. **Múltiplos veículos (VRP)**: extensão direta do TSP base. A frota é configurável (padrão: 3 veículos).

### 5.1 Protocolos de segurança implementados

Cada tipo de atendimento carrega um protocolo de segurança próprio, aplicado tanto na modelagem quanto nas instruções geradas para a equipe de campo:

| Tipo | Protocolo | Como é aplicado no sistema |
|---|---|---|
| Emergência obstétrica | `atendimento_imediato` | Prioridade máxima na fitness — atendida no início da rota |
| Violência doméstica | `protocolo_discreto` | Penalidade de janela **dobrada** na fitness + instrução fixa (veículo sem identificação hospitalar, contato só por número seguro, sigilo perante terceiros) |
| Medicamento hormonal | `cadeia_frio` | Janela a partir das 7h + instrução fixa de refrigeração/verificação de temperatura |
| Acompanhamento pós-parto | `agendado` | Janela de horário comercial (confirmação de presença antes do deslocamento) |

As instruções operacionais desses protocolos são **fixas no código** (`_INSTRUCOES_PROTOCOLO` em `src/llm/gerador.py`), nunca geradas pela LLM — garantindo que procedimentos sensíveis (especialmente o de violência doméstica) não sejam alterados por alucinação do modelo.

---

## 6. Integração com LLM

O módulo `src/llm/gerador.py` usa `google/flan-t5-large` (Hugging Face) rodando **100% localmente**. Nenhum dado sai da máquina — decisão crítica dado o protocolo de sigilo para pontos de violência doméstica.

**Decisão de privacidade:** protocolos de segurança (instruções para situações de violência doméstica, cadeia de frio) são hardcoded no código Python, não gerados pela LLM — evita que o modelo invente procedimentos médicos incorretos.

**Sanitização de prompts:** entradas do usuário (perguntas no chat) passam por sanitização antes de compor o prompt, removendo caracteres de controle e sequências de escape.

Funções disponíveis:

| Função | Output |
|---|---|
| `gerar_roteiro(rota)` | Sequência de paradas com horários estimados e protocolos |
| `gerar_manual(rota)` | Manual de instruções por tipo de atendimento para a equipe |
| `responder_pergunta(pergunta, rota)` | Chat em linguagem natural sobre a rota |
| `gerar_relatorio_resumo(rota)` | Relatório de eficiência com análise da LLM |

---

## 7. Comparativo de Desempenho

Comparamos o GA com dois baselines clássicos usando os mesmos dados (30 pontos, 3 veículos, seed 42). Resultados extraídos do notebook executado (`notebooks/experimentos.ipynb`):

| Abordagem | Fitness | Melhora vs GA | Tempo |
|---|---|---|---|
| Rota aleatória | 8.156 | GA 56% melhor | < 0,01s |
| Vizinho mais próximo (greedy) | 6.081 | GA 42% melhor | < 0,01s |
| **Algoritmo Genético** (pop=100, 150 ger.) | **3.556** | referência | ~91s |

O **vizinho mais próximo** é a abordagem gulosa clássica para TSP/VRP: rápido e determinístico, mas toma decisões localmente ótimas que podem ser globalmente ruins (ex: visitar um ponto próximo de baixa prioridade antes de uma emergência distante). O GA, ao explorar o espaço global de soluções via evolução, encontra ordenações que o greedy não consegue.

O gráfico de convergência (notebook `experimentos.ipynb`) mostra o GA partindo de um fitness próximo ao aleatório e superando o greedy ainda nas primeiras gerações.

---

## 8. Experimentos

Realizamos 3 experimentos variando configurações do GA. Resultados completos com gráficos no notebook `notebooks/experimentos.ipynb`.

**Experimento 1 — Tamanho da população (50 / 100 / 200):**

| Configuração | Fitness final | Melhora | Tempo |
|---|---|---|---|
| Pop = 50  | 3948.70 | 27.5% | 44s  |
| Pop = 100 | 3555.60 | 34.7% | 91s  |
| Pop = 200 | 3454.19 | 36.5% | 179s |

Pop 200 produz a melhor solução (+1.8 pp sobre Pop 100) ao custo de 2× mais tempo. Para uso interativo no app Streamlit, Pop 100 oferece o melhor equilíbrio qualidade × tempo. Pop 50 converge mais rápido mas fica preso em ótimos locais com mais frequência (melhora de apenas 27.5%).

**Experimento 2 — Taxas de mutação:**

| Configuração | Fitness final | Melhora | Tempo |
|---|---|---|---|
| Swap=0.05 Inv=0.00 (baixa)       | 3941.57 | 27.6% | 91s |
| Swap=0.30 Inv=0.10 (padrão)      | 3555.60 | 34.7% | 90s |
| Swap=0.60 Inv=0.30 (alta)        | 3562.34 | 34.6% | 91s |
| Swap=0.30 Inv=0.30 (inv. forte)  | **3538.81** | **35.0%** | 89s |

O resultado mais nítido é o da **taxa baixa** (swap=0.05, sem inversão): fitness 3941.57, claramente o pior, confirmando a hipótese de convergência prematura — pouca diversidade prende a população em ótimos locais. As três configurações com maior diversidade (padrão, alta e inversão forte) ficam **praticamente empatadas** (3538–3562, spread < 1%), com a inversão forte marginalmente à frente. A leitura é que, acima de um limiar mínimo de diversidade, o ganho satura para este espaço de busca de 30 pontos: o que importa é *ter* diversidade suficiente, não maximizá-la. A configuração padrão (swap=0.30, inv=0.10) foi mantida por oferecer resultado equivalente ao melhor com menor risco de instabilidade.

**Experimento 3 — Restrições de negócio:**

| Cenário | Fitness final | Melhora | Tempo |
|---|---|---|---|
| A: Baseline (3v, 40km/h, w_prior=120)  | 3555.60 | 34.7% | 91s |
| B: Prioridade máx. (w_prior=250)       | 6342.96 | 27.2% | 89s |
| C: Frota reduzida (2 veículos)         | 4893.30 | 65.6% | 81s |
| D: Veículo lento (20km/h, w_jan=150)   | 7298.33 | 53.2% | 92s |

Análise da posição média das emergências obstétricas na rota principal:

| Cenário | Posição média emergências |
|---|---|
| A: Baseline         | 3.0 / 10 paradas — início da rota |
| B: Prioridade máx.  | 0.0 / 10 paradas — sempre primeiro |
| C: Frota reduzida   | sem emergências na rota principal — distribuídas entre veículos |
| D: Veículo lento    | 9.0 / 11 paradas — postergadas pelo peso alto de janela |

Com o peso de prioridade padrão (120×), o cenário A já coloca as emergências no início da rota (posição média 3.0/10); elevar o peso para 250× (cenário B) leva-as à posição 0 absoluta, ao custo de fitness maior (as penalidades de posição valem mais). O cenário C mostra o efeito da restrição de frota: com apenas 2 veículos, o decodificador distribui as emergências entre os veículos e elas não se concentram na rota principal. O cenário D é o mais revelador do trade-off entre restrições: com veículo lento (20 km/h) **e** peso de janela alto (150×), a conformidade de janela passa a dominar a fitness e empurra as emergências para o fim da rota principal (posição 9.0/11) — evidenciando que, quando o veículo é lento, priorizar urgência e cumprir janelas de horário entram em conflito direto, e a escolha dos pesos define qual objetivo prevalece.

---

## 8.1 Análise de Impacto — Tempo de Resposta em Emergências

O indicador operacional mais crítico do domínio não é o fitness abstrato, e sim **quanto tempo a equipe leva para chegar a uma emergência obstétrica**. Simulamos o horário de chegada a partir da saída do hospital-base às 8h (deslocamento via Haversine ÷ velocidade + tempo de serviço em cada parada), medindo cada emergência dentro da rota do seu veículo. Comparativo com as mesmas rotas otimizadas da seção 7:

| Abordagem | 1ª emergência | Tempo médio | Última emergência |
|---|---|---|---|
| Rota aleatória | 269,4 min | 358,9 min | 466,6 min |
| Vizinho mais próximo (greedy) | 7,8 min | 154,1 min | 296,4 min |
| **Algoritmo Genético** | **16,1 min** | 156,8 min | **262,4 min** |

O GA atende a primeira emergência em ~16 minutos, contra **~4,5 horas** da rota aleatória — uma redução de mais de 90% no tempo de resposta, com impacto direto na segurança da paciente. O greedy chega à *primeira* emergência ligeiramente mais rápido (7,8 min), mas por acaso geográfico: ele não prioriza urgência, apenas segue o ponto mais próximo. Isso se reflete em (a) fitness global muito pior (6.081 vs 3.556, ver §7), pois ignora janelas, capacidade e a prioridade dos demais pontos; e (b) tempo maior até a *última* emergência (296 vs 262 min). O GA equilibra atender emergências cedo **e** manter a rota globalmente eficiente — exatamente o comportamento desejado quando há múltiplas urgências concorrendo com outras restrições.

---

## 9. Considerações Éticas e de Privacidade

O domínio é sensível por natureza — envolve localização de pacientes em situação de vulnerabilidade. As decisões tomadas:

**Dados sintéticos:** nenhuma informação real de paciente é utilizada. O gerador cria coordenadas fictícias dentro de uma área geográfica arbitrária de São Paulo.

**LLM local:** o modelo roda offline na máquina do operador. Nenhum dado de rota, localização ou protocolo é enviado a APIs externas (OpenAI, Anthropic, etc.).

**Protocolo discreto (violência doméstica):** pontos deste tipo recebem penalidade dobrada por violação de janela horária, incentivando o GA a respeitá-la. As instruções operacionais (veículo sem identificação hospitalar, contato apenas por número seguro) são fixas no código — não geradas pela LLM — para garantir que nunca sejam alteradas por alucinação do modelo.

**Ausência de PII:** o CSV gerado não contém nome, CPF, endereço real ou qualquer dado identificável. Os campos `nome` e `latitude/longitude` são completamente fictícios.

**Mitigação de bias:** a distribuição de frequência dos tipos de atendimento (pesos no gerador) reflete dados clínicos aproximados, evitando sub-representar tipos de alta vulnerabilidade como violência doméstica.

---

## 10. Organização do Projeto

```
tech-challenge-group-61/
├── src/
│   ├── gerar_dados.py              # Gerador de dados sintéticos
│   ├── fitness.py                  # Função de fitness VRP
│   ├── genetic_algorithm/
│   │   ├── vrp.py                  # Algoritmo Genético principal
│   │   └── baselines.py            # Baselines para comparação
│   ├── visualization/
│   │   └── mapa.py                 # Visualização Folium
│   └── llm/
│       └── gerador.py              # Integração LLM local
├── tests/                          # 71+ testes automatizados
├── notebooks/
│   └── experimentos.ipynb          # 3 experimentos + comparativo
├── data/
│   └── pontos.csv                  # Dados sintéticos gerados
├── docs/
│   ├── relatorio-tecnico.md        # Este documento
│   └── arquitetura.md              # Diagrama de arquitetura
├── app.py                          # Interface Streamlit
└── requirements.txt
```

**Ambiente:** Python 3.13, venv, dependências em `requirements.txt`.  
**Testes:** `pytest tests/ -v` — 110 testes cobrindo todos os módulos.  
**App:** `streamlit run app.py` (instalar streamlit do PyPI público).
