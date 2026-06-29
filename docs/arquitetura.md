# Diagrama de Arquitetura — VRP Médico

## Visão geral dos componentes

```mermaid
graph TD
    subgraph Entrada
        A[gerar_dados.py\nDados sintéticos] --> B[(data/pontos.csv)]
    end

    subgraph Core
        B --> C[fitness.py\nDistância + 4 penalidades]
        C --> D[genetic_algorithm/vrp.py\nGA: população → evolução → melhor rota]
        C --> E[genetic_algorithm/baselines.py\nAleatório + Vizinho mais próximo]
    end

    subgraph Saída
        D --> F[visualization/mapa.py\nMapa Folium interativo]
        D --> G[llm/gerador.py\nflan-t5-large local\nRoteiro + Manual + Chat]
        E --> H[notebooks/experimentos.ipynb\nComparativo + 3 experimentos]
    end

    subgraph Interface
        F --> I[app.py\nStreamlit]
        G --> I
        H --> I
    end
```

---

## Fluxo de dados

```mermaid
sequenceDiagram
    participant U as Usuário
    participant App as Streamlit (app.py)
    participant GA as vrp.py
    participant F as fitness.py
    participant M as mapa.py
    participant L as llm/gerador.py

    U->>App: Configura frota e parâmetros
    U->>App: Clica "Otimizar Rotas"
    App->>GA: evoluir(df, config, veiculo, pesos)
    loop N gerações
        GA->>F: fitness_vrp(cromossomo)
        F-->>GA: fitness total
        GA->>GA: seleção + crossover + mutação
    end
    GA-->>App: ResultadoGA (rotas, histórico)
    App->>M: criar_mapa(df, rota)
    M-->>App: folium.Map HTML
    App-->>U: Mapa + gráfico de convergência
    U->>App: Pergunta ao assistente
    App->>L: responder_pergunta(pergunta, rota)
    L-->>App: Resposta em linguagem natural
    App-->>U: Resposta
```

---

## Estrutura de módulos

```mermaid
graph LR
    app.py --> gerar_dados.py
    app.py --> vrp.py
    app.py --> mapa.py
    app.py --> gerador.py

    vrp.py --> fitness.py
    baselines.py --> fitness.py
    baselines.py --> vrp.py

    experimentos.ipynb --> vrp.py
    experimentos.ipynb --> baselines.py
    experimentos.ipynb --> fitness.py
    experimentos.ipynb --> gerar_dados.py
```

---

## Componentes e responsabilidades

| Módulo | Responsabilidade | Tecnologia |
|---|---|---|
| `gerar_dados.py` | Geração de pontos sintéticos reprodutíveis | NumPy, Pandas |
| `fitness.py` | Avaliação de qualidade de rotas (5 componentes) | Python puro |
| `genetic_algorithm/vrp.py` | Algoritmo Genético completo (TSP → VRP) | Python puro |
| `genetic_algorithm/baselines.py` | Baselines para comparação (aleatório, greedy) | Python puro |
| `visualization/mapa.py` | Renderização de rotas em mapa interativo | Folium (Leaflet.js) |
| `llm/gerador.py` | Geração de linguagem natural a partir das rotas | Hugging Face Transformers |
| `app.py` | Interface web completa | Streamlit |
| `notebooks/experimentos.ipynb` | Análise experimental e comparativos | Jupyter, Matplotlib |
