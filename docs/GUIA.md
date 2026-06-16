# Tech Challenge — Fase 2 — Guia do Projeto

> Documento de orientação para a dupla. Resume **o que é o projeto escolhido, o passo a passo, onde fazer cada coisa e as dúvidas práticas** (banco de dados, repositório em dupla).

---

## Decisão tomada

- **Versão oficial do enunciado:** a da *Secretaria* (13 páginas, foco em **saúde e segurança da mulher**). É ela que vale. O outro PDF é rascunho/contexto.
- **Projeto escolhido: Projeto 2 — Otimização de Rotas (VRP) para atendimento especializado à mulher.**
- **Motivo:** queremos *fazer algo novo e aprender mais* (sem reaproveitar o modelo da Fase 1). No Projeto 2 o **algoritmo genético é o protagonista** (a gente escreve os operadores à mão), os **dados são sintéticos** (zero dependência do projeto anterior) e o resultado rende uma **demo visual forte** (mapa + app + LLM).

---

## O projeto em português (a história)

Uma rede de hospitais de saúde da mulher precisa, todo dia, fazer uma **lista de paradas** pela cidade:

- 🚨 Emergência obstétrica (prioridade máxima)
- 🏠 Acompanhamento pós-parto (tem hora marcada)
- 💊 Medicamento hormonal com temperatura controlada (não pode demorar)
- 🔒 Atendimento de violência doméstica (protocolo especial, horário seguro)
- ...e mais várias paradas espalhadas.

Há **alguns veículos** disponíveis (van, moto, drone), cada um com capacidade, autonomia e custo diferentes.

**Pergunta central do projeto:** _qual a melhor forma de distribuir as paradas entre os veículos e em que ordem visitá-las?_ — atendendo urgências primeiro, respeitando horários, sem estourar capacidade e rodando o mínimo possível.

### Por que entra "algoritmo genético"
O número de combinações possíveis é astronômico — força bruta é impossível. O algoritmo genético imita a evolução natural:

1. Cria **várias rotas aleatórias** (uma "população", todas ruins no começo).
2. Dá uma **nota** (*fitness*) a cada rota — boa rota = atende urgência cedo, respeita horários, roda pouco.
3. As **melhores "se reproduzem"** (crossover combina pedaços de duas boas rotas; mutação muda algo aleatório).
4. Repete por centenas de gerações → a população vai melhorando.
5. No fim, sobra uma rota **excelente**.

Você não programa "a solução", você programa o **processo de evolução**.

### Onde entra a LLM (Hugging Face local)
> Decisão da dupla: usamos uma **LLM local open-source (Hugging Face / `transformers`)**, não API paga (Claude/GPT). Roda na máquina, sem chave nem custo — ver seção *Ferramentas*.

O algoritmo cospe algo cru: `[Parada 4 → Parada 1 → Parada 7 → ...]`. A LLM transforma isso em algo usável por humanos:
- **Roteiro do dia** legível para a equipe de transporte.
- **Relatório** (nº de paradas, emergências, economia de tempo).
- **Respostas em linguagem natural** ("qual minha próxima parada urgente?").

### Foto final que se entrega
Um **app simples (Streamlit)** com: mapa da cidade + rotas coloridas por tipo de atendimento + botão "otimizar" + caixa de conversa com a LLM.

**Em uma frase:** um *"Waze inteligente" para entregas médicas*, onde uma IA evolutiva descobre as melhores rotas e uma IA de linguagem explica tudo para a equipe.

---

## Dúvidas práticas

### Preciso de banco de dados (como no Projeto 1)?
**Não.** No Projeto 1 não havia banco de verdade — eram **arquivos** na pasta `data/` (CSV/Excel) baixados da internet.

No Projeto 2 é mais simples: o enunciado manda **criar dados sintéticos**. Ou seja:
- ❌ Não baixa dataset.
- ❌ Não usa banco (Postgres/MySQL/etc.).
- ✅ Um script Python **inventa** os pontos de atendimento (lat, lon, tipo, prioridade, janela de horário) e salva em `data/pontos.csv`.

Esse CSV gerado é o "banco". Mesma ideia da pasta `data/` da Fase 1, só que **nós geramos o arquivo**.

### Posso fazer tudo em um repositório só, já que é em dupla?
**Sim — é o recomendado.** Um repo no GitHub com as duas como colaboradoras.

Para trabalhar sem conflito:
- Cada uma cria uma **branch** para sua parte (ex.: `feat/algoritmo-genetico`, `feat/interface`) e abre **Pull Request** para `main`.
- Dividir por **pastas/módulos diferentes** evita pisar uma na outra.
- A `main` fica sempre funcionando.

---

## Estrutura do repositório

```
tech-challenge-fase2/
├── data/
│   └── pontos.csv            ← gerado por código (não baixado)
├── src/
│   ├── gerar_dados.py        ← inventa os pontos de atendimento
│   ├── genetico.py           ← o algoritmo genético (o coração)
│   ├── fitness.py            ← a "nota" de cada rota
│   ├── mapa.py               ← desenha as rotas no mapa
│   └── llm.py                ← conversa com a IA (gera roteiro/relatório)
├── notebooks/
│   └── experimentos.ipynb    ← os 3 experimentos exigidos + comparações
├── app.py                    ← interface Streamlit (o app final)
├── tests/                    ← testes automatizados (exigidos)
├── docs/
│   └── relatorio-tecnico.md  ← o relatório que vale nota
├── README.md
└── pyproject.toml            ← Poetry (ou requirements.txt como na Fase 1)
```

---

## Passo a passo: o quê e onde

> **Status:** ⬜ pendente · 🟡 em andamento · ✅ feito — edite o emoji conforme avançarem.

| # | Status | Passo | Onde | O que entrega |
|---|---|---|---|---|
| **1** | ✅ | Criar o repo no GitHub e clonar; configurar ambiente Python | GitHub + máquina | Repo + `pyproject.toml`/`requirements.txt` |
| **2** | ✅ | **Gerar dados sintéticos**: ~20–30 pontos com tipo, prioridade, janela de horário, coordenadas | `src/gerar_dados.py` → `data/pontos.csv` | O "banco" |
| **3** | ✅ | **Função de fitness**: dada uma rota, calcular a nota (distância + prioridade + janelas + capacidade) | `src/fitness.py` | A regra do jogo |
| **4** | ⬜ | **Algoritmo genético**: população, seleção, crossover, mutação, gerações | `src/genetico.py` | O coração do projeto |
| **5** | ⬜ | **3 experimentos**: variar população/taxa de mutação e comparar | `notebooks/experimentos.ipynb` | Recomendado (é exigência do Projeto 1; fortalece o relatório) |
| **6** | ⬜ | **Comparativo vs. outra abordagem**: GA contra baseline (vizinho mais próximo e/ou OR-tools) | `notebooks/experimentos.ipynb` | **Exigência obrigatória do relatório** |
| **7** | ✅ | **Visualização no mapa**: desenhar a melhor rota colorida por tipo | `src/visualization/mapa.py` | Exigência obrigatória |
| **8** | ⬜ | **Integração LLM**: rota → LLM gera manual + roteiro + responde perguntas | `src/llm.py` | Exigência obrigatória |
| **9** | ⬜ | **App Streamlit**: mapa + botão otimizar + chat | `app.py` | Recurso extra (recomendado) |
| **10** | ⬜ | **Testes** automatizados | `tests/` | Exigência |
| **11** | ⬜ | **Relatório técnico + vídeo** (≤ 10 min) | `docs/` + YouTube/Vimeo | Entregáveis finais |

**Divisão sugerida na dupla:** uma pessoa nos passos **2–5** (dados + algoritmo genético, parte mais pesada); a outra nos passos **6–8** (mapa + LLM + app). Passos **9–10** as duas juntas no fim.

---

## Melhorias pendentes (aguardando código base de TSP)

Decidimos esperar o código base da disciplina antes de finalizar a fitness e o AG,
para alinhar a estrutura e evitar retrabalho. Pendências registradas:

- [ ] **VRP múltiplos veículos** — a fitness atual (`src/fitness.py`) pontua **1 rota / 1 veículo**. O enunciado pede frota (vários veículos). Reestruturar para pontuar um *conjunto* de rotas. *Obs.: demanda total (~41,6 kg) já excede a capacidade do veículo padrão (20 kg), então a frota é necessária.*
- [ ] **Bug da janela da emergência** — emergência obstétrica tem janela `0h–1h` nos dados, mas a `calcular_penalidade_janela` começa o relógio às 8h → toda emergência é punida como "atrasada" sempre. Isentar emergências da penalidade de janela (são "atendimento imediato, qualquer hora") ou recodificar a janela como o dia inteiro.
- [ ] **Matriz de distâncias pré-computada** — o AG chamará a fitness milhões de vezes; hoje cada chamada refaz `set_index` e recalcula Haversine repetidamente. Pré-computar matriz N×N uma vez (ganho grande de performance — critério de avaliação).

---

## Requisitos obrigatórios do Projeto 2 (checklist)

- [ ] Partir do **código base de TSP** fornecido no GitHub da disciplina e evoluir para **VRP**.
- [ ] Otimizar rotas de entrega de medicamentos/atendimento específicos para saúde da mulher.
- [ ] Representação genética adequada para rotas + operadores especializados (seleção, crossover, mutação).
- [ ] **Restrição obrigatória — ordem de prioridade:** emergência obstétrica (máxima) > violência doméstica (protocolo especial) > medicamentos hormonais (temperatura controlada) > pós-parto (janelas de tempo).
- [ ] **Pelo menos 2 restrições adicionais** (capacidade, autonomia, janelas de horário, custo por distância, frota, tipos de veículo). *A criatividade aqui vale nota de inovação.*
- [ ] **Protocolos de segurança implementados** (item explícito do relatório) — ex.: horário seguro e discrição para entregas em casos de violência doméstica. Ligado ao critério de ética.
- [ ] **Comparativo de desempenho com outras abordagens de roteamento** (item obrigatório do relatório) — GA vs. baseline (vizinho mais próximo) e/ou solver dedicado (OR-tools). *Não confundir com os 3 experimentos, que comparam o GA com ele mesmo.*
- [ ] **Análise de impacto** — medir explicitamente o **tempo de resposta em emergências** (tempo até a 1ª emergência obstétrica) e a segurança da paciente.
- [ ] **Visualização das rotas em mapa** com codificação por tipo de atendimento (obrigatório).
- [ ] **LLM** gerando manual de instruções + roteiro detalhado + respondendo perguntas em linguagem natural.
- [ ] Projeto Python estruturado (Poetry/Pipenv/venv), bom README, testes automatizados, diagramas de arquitetura.

---

## Critérios de avaliação (peso igual — 33,3% cada)

1. **Impacto social e ética** — privacidade, bias e equidade na saúde feminina.
2. **Qualidade técnica** — algoritmo genético correto, integração efetiva com LLM, qualidade do código.
3. **Aplicabilidade prática e inovação** — viabilidade real + criatividade nas restrições.

### Ponto de atenção: ética/privacidade vale 1/3 da nota
O domínio é sensível (violência doméstica, localização de pacientes, diagnóstico). Reservar uma seção do relatório para:
- Uso de **dados sintéticos** (nada de PII real).
- **LLM local (Hugging Face):** mitigação concreta de privacidade — nenhum dado de localização de paciente (incluindo vítimas de violência doméstica) é enviado a APIs de terceiros. Tudo roda na máquina.
- **Protocolos de segurança:** horário seguro e discrição para entregas em casos sensíveis (violência doméstica).
- Cuidado com o que vai no **prompt da LLM**, mesmo sendo local.
- Mitigação de **bias** e considerações de **equidade** entre grupos demográficos de mulheres.

É critério explícito de avaliação, não enfeite.

---

## Entregáveis finais

- **Repositório Git** — código completo, scripts/notebooks de demonstração, bom README.
- **Relatório técnico** — deve cobrir explicitamente (itens da pág. 10 do oficial):
  - Implementação do GA a partir do código base fornecido (codificação, fitness, crossover, mutação).
  - Estratégias para lidar com as restrições específicas da saúde da mulher.
  - **Protocolos de segurança implementados.**
  - Integração com LLMs para geração de instruções e relatórios especializados.
  - **Comparativo de desempenho com outras abordagens de roteamento.**
  - **Análise de impacto: tempo de resposta em emergências e segurança da paciente.**
  - Visualizações e análises das rotas otimizadas por tipo de atendimento.
  - Considerações éticas: privacidade de localização e segurança da paciente.
  - (Se nuvem) Arquitetura da solução.
- **Vídeo (≤ 10 min)** — foco em demonstrar o sistema em execução (não precisa explicar o que é GA/LLM; o professor já conhece os fundamentos).

---

## Ferramentas

- **Linguagem:** Python.
- **Ambiente:** Poetry (recomendado) ou venv/requirements.txt (como na Fase 1).
- **LLM:** **decisão tomada — modelo local open-source (Hugging Face / `transformers`).** O enunciado permite qualquer LLM (API paga ou open-source) e não exige latência. Escolhemos local por dois motivos a justificar no relatório: (1) **reprodutibilidade** — qualquer avaliador clona o repo e roda tudo sem chave de API nem custo; (2) **privacidade** — nenhum dado de localização de paciente sai da máquina para terceiros (ver seção de ética).
- **Interface:** Streamlit (altamente recomendado pelo enunciado).
- **Mapa/visualização:** ex. `folium` ou `plotly`.
