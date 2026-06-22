# Tech Challenge - Fase 2 | Grupo 61
## Projeto 2: Otimização de Rotas para Distribuição de Medicamentos e Insumos

Solução para otimização logística hospitalar usando **algoritmos genéticos** (TSP/VRP) e **LLMs** para geração de relatórios e instruções de entrega.

> **LLM 100% local e gratuita** (Hugging Face / `transformers`). Não requer chave de API nem custo — qualquer pessoa clona o repositório e roda o projeto inteiro.

---

## Estrutura do Projeto

```
tech-challenge-group-61/
├── data/                        # Dados de localização das unidades hospitalares
├── notebooks/
│   ├── 01_eda_rotas.ipynb       # Análise exploratória dos pontos de entrega
│   ├── 02_algoritmo_genetico.ipynb  # Implementação e experimentos do AG
│   ├── 03_llm_integracao.ipynb  # Integração com LLM para relatórios
│   └── relatorio-tecnico.ipynb  # Relatório técnico completo
├── src/
│   ├── genetic_algorithm/       # Implementação do algoritmo genético
│   ├── llm/                     # Integração com LLM local (Hugging Face)
│   └── visualization/           # Visualização de rotas em mapas
└── tests/                       # Testes automatizados
```

---

## Setup

> **Pré-requisito:** Python 3.13 instalado.

### 1. Criar e ativar o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```

> ⚠️ **Importante:** confirme que o ambiente está ativo — o prompt do terminal
> deve começar com `(.venv)`. Todos os comandos seguintes precisam ser rodados
> com o venv ativo, caso contrário caem no Python global do sistema.

### 2. Instalar as dependências

```bash
python -m pip install -r requirements.txt
```

> Use `python -m pip` (e não apenas `pip`) para garantir que a instalação vá
> para o Python do `.venv`. O `torch` é grande (~88 MB) e a instalação pode
> levar alguns minutos — **não interrompa** o processo.

### 3. Validar a instalação

```bash
python test_setup.py
```

Deve listar todas as dependências com suas versões e terminar com
`Ambiente OK — todas as dependencias instaladas.`

### 4. (opcional) Testar a LLM local

A LLM é local (Hugging Face) — não requer chave de API. Na primeira execução o
modelo é baixado automaticamente e fica em cache (`~/.cache/huggingface`). Para
validar que ela roda de verdade:

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained('google/flan-t5-small')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
inputs = tok('Translate to Portuguese: deliver medicine to the hospital', return_tensors='pt')
print(tok.decode(model.generate(**inputs, max_new_tokens=40)[0], skip_special_tokens=True))
"
```

Se imprimir uma frase de saída, a LLM local está funcional. O texto pode sair
imperfeito — `flan-t5-small` é um modelo mínimo, usado só para o smoke test.

### 5. Iniciar o Jupyter

```bash
jupyter notebook
```

---

### Docker (alternativo)

```bash
docker build -t tech-challenge-grupo61 .
docker run -p 8888:8888 -v $(pwd):/app tech-challenge-grupo61
```

---

## Uso

Os comandos abaixo devem ser executados a partir da raiz do projeto, com o
ambiente virtual ativo (`source .venv/bin/activate`).

### Gerar os dados sintéticos

Cria os pontos de atendimento (depósito + 30 pontos) em `data/pontos.csv`:

```bash
python src/gerar_dados.py
```

### Gerar o mapa das rotas

Gera `data/mapa.html` — um mapa interativo com os pontos coloridos por tipo de
atendimento. Abra o arquivo no navegador para visualizar:

```bash
python src/visualization/mapa.py
```

> O `data/mapa.html` já vem versionado no repositório, então é possível abri-lo
> direto sem gerar novamente.

### Gerar instruções e relatórios com a LLM

Roda a LLM local (Hugging Face) e exibe roteiro, manual, chat e relatório
de eficiência para uma rota de exemplo:

```bash
python src/llm/gerador.py
```

> Na primeira execução o modelo `flan-t5-large` (~3 GB) é baixado
> automaticamente e fica em cache (`~/.cache/huggingface`). As execuções
> seguintes carregam do cache local — sem internet e sem custo.

### Rodar os testes

```bash
python -m pytest tests/ -v
```

---

## Troubleshooting

| Sintoma | Causa provável | Solução |
| --- | --- | --- |
| `pip` aponta para `/opt/homebrew/...` no traceback | venv não está ativo | Rode `source .venv/bin/activate` antes de instalar |
| `test_setup.py` diz `NAO ENCONTRADO` | dependência não instalada no venv | Reative o venv e rode `python -m pip install -r requirements.txt` |
| `python-dotenv` aparece como `n/a` | a lib não expõe `__version__` | Normal, não é erro |
| Instalação muito lenta / travada | `torch` é grande | Aguarde — não use Ctrl+C |

---

## Time

Larissa Nunes da Silva, Oryange Strifezze
