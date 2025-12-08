# Classificação Zero-Shot de Sentimento em Reviews do IMDb com Modelos Abertos (Vanilla LLM + RAG)

Projeto da disciplina **Modelos de Linguagem e Generativos** – Universidade Presbiteriana Mackenzie.

Este trabalho implementa uma solução de **Text Zero-Shot Classification** usando reviews de filmes do **IMDb** e compara:

1. Um modelo **Vanilla Zero-Shot** (sem ajuste específico), e  
2. Uma abordagem baseada em **RAG (Retrieval-Augmented Generation)**.

Toda a solução usa **apenas modelos e recursos abertos** e é executável em um único **notebook Colab**.

---

## 1. Introdução

Plataformas como IMDb, lojas virtuais e serviços de streaming acumulam milhares de comentários de usuários diariamente. Ler e interpretar manualmente todos esses textos é impraticável, o que torna importante o uso de técnicas de **Processamento de Linguagem Natural (PLN)** para automatizar tarefas como **análise de sentimento** – isto é, identificar se um texto expressa uma opinião mais positiva ou mais negativa.

Com o avanço dos **modelos de linguagem de grande porte (LLMs)**, tornou-se possível resolver tarefas de PLN sem treinar um modelo do zero para cada problema. Em especial, técnicas de **zero-shot learning** permitem utilizar um modelo pré-treinado para classificar textos em novas categorias apenas por meio de instruções em linguagem natural e definição de rótulos, sem exigir um grande conjunto rotulado para fine-tuning. Em paralelo, arquiteturas de **RAG (Retrieval-Augmented Generation)** combinam um módulo de busca semântica com um LLM gerativo, enriquecendo o contexto do modelo com exemplos ou documentos relevantes antes da geração da resposta.

Neste projeto, implementamos uma solução de **classificação de sentimento em reviews de filmes do IMDb** utilizando apenas modelos abertos. Comparamos duas abordagens: (i) um modelo “vanilla” de **Zero-Shot Text Classification**, baseado em um LLM pré-treinado aplicado diretamente, e (ii) uma solução com **RAG**, na qual o modelo recebe, junto com o review, exemplos similares já rotulados para apoiar a decisão. O objetivo é avaliar, de forma prática, em que medida a inclusão de recuperação de exemplos melhora (ou não) o desempenho em relação à solução zero-shot básica.

---

## 2. Referencial Teórico

### 2.1 Modelos de Linguagem (LLMs)

Modelos de linguagem de grande porte são redes neurais treinadas em grandes quantidades de texto, capazes de prever a próxima palavra, gerar respostas, resumir documentos e executar tarefas de classificação por meio de **prompting**. Exemplos incluem BART, T5, LLaMA, Mistral, entre outros.

### 2.2 Zero-Shot Text Classification

Na **classificação zero-shot**, um modelo pré-treinado é utilizado para classificar textos em rótulos que não fizeram parte do treinamento supervisionado original. Em vez de treinar um novo classificador, descrevemos as possíveis classes (por exemplo, `"positive"` e `"negative"`) e pedimos ao modelo que escolha o rótulo mais compatível com o texto. Modelos como **BART-MNLI** são amplamente usados nessa tarefa por meio de pipelines prontos de `zero-shot-classification`.

### 2.3 Embeddings de Sentenças

Modelos de **sentence embeddings** (como `all-MiniLM-L6-v2`) transformam textos em vetores densos em um espaço contínuo, de modo que textos semanticamente similares fiquem próximos. Esses vetores podem ser usados para busca semântica, clustering ou como base para arquiteturas de RAG.

### 2.4 Retrieval-Augmented Generation (RAG)

RAG combina:

- um **retriever** (que busca documentos ou exemplos relevantes usando embeddings), e  
- um **gerador** (LLM), que recebe o texto da consulta junto com os documentos recuperados e produz a resposta.

A ideia é que o LLM não dependa apenas da “memória” treinada nos pesos, mas também de informações trazidas dinamicamente no momento da inferência. No contexto deste trabalho, usamos RAG em um cenário de classificação de sentimento: antes de pedir a classificação de um review, fornecemos ao LLM exemplos semelhantes já rotulados.

---

## 3. Metodologia

### 3.1 Dados

- **Fonte:** Dataset público do IMDb disponibilizado via `datasets` (`stanfordnlp/imdb`), contendo reviews de filmes em inglês e um rótulo binário:
  - `0` → review **negative**
  - `1` → review **positive**
- **Divisão:** Aproveitamos os splits padrão do dataset (`train` e `test`) e, por questões de custo computacional em Colab, usamos subconjuntos:
  - `N_TRAIN = 2000` exemplos para base de treino/embeddings,
  - `N_TEST = 200` exemplos para avaliação.
- O objetivo é classificar cada review como **“positive”** ou **“negative”** e comparar as saídas dos dois modelos com os rótulos verdadeiros.

### 3.2 Modelos Utilizados

Todos os modelos são **abertos**:

1. **Baseline – Zero-Shot com BART-MNLI**
   - Modelo: `facebook/bart-large-mnli`
   - Uso via pipeline `zero-shot-classification` da biblioteca `transformers`.
   - Rótulos candidatos: `["positive", "negative"]`.
   - Para cada review, o modelo retorna um score para cada label e escolhemos o rótulo com maior probabilidade.

2. **Retriever – Embeddings com MiniLM**
   - Modelo: `sentence-transformers/all-MiniLM-L6-v2`.
   - Usado para gerar embeddings (vetores) dos reviews rotulados no conjunto de treino (`N_BASE = 1000`–`2000`).
   - Similaridade calculada via produto interno para encontrar reviews mais parecidos.

3. **LLM Gerativo – TinyLlama**
   - Modelo: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
   - Recebe um **prompt few-shot**, contendo:
     - uma instrução em linguagem natural,
     - alguns reviews semelhantes já rotulados,
     - e o review-alvo.
   - O modelo é instruído a responder apenas com `"positive"` ou `"negative"`.

### 3.3 Pipeline Baseline (Zero-Shot)

1. Carregamos o dataset IMDb e selecionamos subconjuntos de treino e teste.
2. Inicializamos o pipeline `zero-shot-classification` com o modelo `facebook/bart-large-mnli`.
3. Para cada review do conjunto de teste:
   - Chamamos o pipeline passando o texto e as labels `["positive", "negative"]`.
   - Registramos o rótulo previsto (label com maior score).
4. Comparamos as previsões com os rótulos verdadeiros e calculamos:
   - **acurácia**,
   - **precision, recall e F1-score (macro)**,
   - **matriz de confusão**.

### 3.4 Pipeline RAG (Retriever + TinyLlama)

1. A partir do conjunto de treino, selecionamos `N_BASE` exemplos rotulados para compor a **base de exemplos**.
2. Geramos embeddings desses textos com o modelo `all-MiniLM-L6-v2` e guardamos em uma matriz.
3. Para cada review do conjunto de teste:
   - Geramos o embedding do review,
   - Recuperamos os `k` exemplos mais semelhantes (ex.: `k = 3`),
   - Montamos um **prompt few-shot**, incluindo:
     - instruções de que o modelo deve classificar o sentimento,
     - os exemplos recuperados com seus labels,
     - o review a ser classificado.
   - Chamamos o TinyLlama para gerar a saída a partir desse prompt.
   - A resposta gerada é pós-processada e mapeada para `"positive"` ou `"negative"`.
4. Calculamos as mesmas métricas utilizadas no baseline (acurácia, precision, recall, F1 e matriz de confusão).

---

## 4. Resultados

A tabela abaixo resume as métricas obtidas nos 200 exemplos de teste:

| Modelo                         | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-------------------------------|----------|-------------------|----------------|------------|
| BART Zero-Shot (baseline)     | 0.8850   | 0.8855            | 0.8842         | 0.8847     |
| TinyLlama + RAG (k = 3)       | 0.4800   | 0.2400            | 0.5000         | 0.3243     |

### 4.1 Análise do Baseline (BART Zero-Shot)

O modelo `facebook/bart-large-mnli`, aplicado em modo zero-shot, apresentou um desempenho **alto e equilibrado**:

- Acurácia de aproximadamente **88,5%**,
- F1-score de **0,8910** para a classe **negative** (104 exemplos),
- F1-score de **0,8783** para a classe **positive** (96 exemplos).

A matriz de confusão para o baseline foi:

```text
[[94 10]
 [13 83]]
```

### 9. Autores

- Guilherme Vinicius Sennes Domingues – 10751468
- Matheus Klein – 00000
- Wendell Lima – 0000

