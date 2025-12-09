# Classificação de Sentimento em Reviews do IMDb com Modelos Abertos (Zero-Shot vs RAG)

Projeto desenvolvido para a disciplina de **Modelos de Linguagem e Generativos**, comparando uma abordagem de **Text Zero-Shot Classification** com uma solução baseada em **Retrieval-Augmented Generation (RAG)**, utilizando exclusivamente modelos e recursos abertos e executáveis em Google Colab.

---

## 1. Introdução

O objetivo deste trabalho é classificar automaticamente o **sentimento** de reviews de filmes do IMDb como **positivo** ou **negativo**, empregando modelos de linguagem abertos.

São comparadas duas abordagens:

1. Um modelo **Zero-Shot** usando `facebook/bart-large-mnli`, que já vem pré-treinado e consegue fazer classificação textual sem fine-tuning específico no dataset.
2. Uma solução com **RAG (Retrieval-Augmented Generation)**, combinando um modelo de embeddings (`all-MiniLM-L6-v2`) com um modelo gerativo (`TinyLlama-1.1B-Chat`), em que exemplos recuperados do próprio dataset são usados para orientar a decisão do LLM.

O foco é analisar se a arquitetura RAG melhora o desempenho em relação ao modelo Zero-Shot “vanilla” e discutir as limitações práticas de cada abordagem.

---

## 2. Referencial Teórico

### 2.1 Modelos de Linguagem (LLMs)

Modelos de Linguagem de Grande Porte (LLMs) são redes neurais treinadas em grandes quantidades de texto para aprender padrões estatísticos da língua, sendo capazes de gerar texto, responder perguntas, realizar tradução e executar tarefas de compreensão textual.

### 2.2 Zero-Shot Text Classification

Na configuração **Zero-Shot Text Classification**, um LLM pré-treinado recebe:

- um texto de entrada, e  
- uma lista de rótulos candidatos (por exemplo, `["positive", "negative"]`),

e estima qual rótulo é mais compatível com o texto, mesmo sem ter sido ajustado especificamente naquele dataset. Modelos como `facebook/bart-large-mnli` são amplamente utilizados para essa tarefa, explorando a formulação de inferência textual (NLI) para realizar classificação zero-shot.

### 2.3 Retrieval-Augmented Generation (RAG)

**RAG (Retrieval-Augmented Generation)** é uma arquitetura que combina:

- um módulo de **retrieval** (busca de documentos ou exemplos relevantes), e  
- um modelo de linguagem gerativo, que produz a resposta a partir do contexto recuperado.

Em vez de depender apenas do conhecimento armazenado nos pesos do modelo, RAG permite que o LLM consulte documentos externos em tempo de execução, enriquecendo o contexto e potencialmente melhorando a qualidade das respostas, principalmente em tarefas dependentes de conhecimento específico (como bulário eletrônico, artigos científicos etc.).

Neste projeto, RAG é adaptado para uma tarefa de **classificação binária de sentimento**, usando como “repositório” os próprios reviews de treino do IMDb.

---

## 3. Metodologia

### 3.1 Dados: IMDb

Os experimentos utilizam o dataset público do IMDb disponibilizado em:

- `stanfordnlp/imdb` no Hugging Face.

Cada exemplo contém:

- `text`: o review em inglês;  
- `label`: rótulo binário (`0 = negative`, `1 = positive`).

Para tornar os notebooks executáveis em um ambiente de Colab, foram usados subconjuntos:

- **2.000 exemplos** para treino;  
- **200 exemplos** para teste.

### 3.2 Abordagem 1 – Zero-Shot com BART-MNLI (Baseline)

A primeira abordagem utiliza o modelo:

- `facebook/bart-large-mnli`,  
- via pipeline `zero-shot-classification` da biblioteca `transformers`.

Para cada review:

1. O texto é passado para o pipeline junto com as labels `["positive", "negative"]`.  
2. O modelo calcula scores de compatibilidade para cada label.  
3. É escolhido o rótulo com maior score como previsão.  
4. As previsões são comparadas aos rótulos reais, gerando acurácia, precision, recall, F1 e matriz de confusão.

Essa abordagem serve como **baseline** “vanilla LLM” de Zero-Shot.

### 3.3 Abordagem 2 – RAG com MiniLM + TinyLlama

Na segunda abordagem, foi implementada uma versão de **RAG** adaptada à tarefa de classificação de sentimento:

1. **Base de exemplos rotulados**  
   - Um subconjunto de reviews de treino é selecionado.  
   - Cada review possui `text` e `label` (`negative` ou `positive`).

2. **Embeddings semânticos (Retriever)**  
   - É utilizado o modelo `sentence-transformers/all-MiniLM-L6-v2` para gerar **embeddings** dos textos de treino.  
   - Esses embeddings formam uma base vetorial de exemplos rotulados (negative/positive).

3. **Recuperação de exemplos (Retrieval)**  
   - Para cada review do conjunto de teste, gera-se o embedding correspondente.  
   - São recuperados os *k* exemplos mais similares na base, usando produto interno nos vetores.  

4. **Construção do prompt (Augmented Generation)**  
   - Os reviews recuperados e seus rótulos são inseridos em um **prompt few-shot** para o modelo gerativo `TinyLlama-1.1B-Chat`, juntamente com:
     - uma instrução clara para atuar como classificador de sentimento;  
     - o review alvo, que deve ser rotulado.
   - O modelo é instruído a responder **apenas** com uma palavra: `positive` ou `negative`.

5. **Controle de contexto e truncamento**  
   - Para respeitar o limite de contexto (~2048 tokens) do TinyLlama, são aplicados:
     - truncamento de tokens nos exemplos (limite máximo por exemplo);  
     - truncamento no review alvo;  
     - redução do número de exemplos *k* no prompt.

6. **Parsing da resposta**  
   - A saída do TinyLlama é pós-processada para extrair apenas `positive` ou `negative`, que são convertidos em rótulos de classe.  
   - Com esses rótulos, são calculadas as mesmas métricas de classificação usadas no baseline.

Importante: nessa arquitetura, **o TinyLlama não é treinado** no dataset IMDb; ele apenas utiliza os exemplos recuperados como contexto no prompt (few-shot), mantendo os pesos do modelo inalterados.

---

## 4. Resultados

### 4.1 Métricas por abordagem

#### Baseline – Zero-Shot com BART-MNLI

No conjunto de teste com 200 reviews, o modelo Zero-Shot apresentou:

- **Acurácia:** 0.8850

**Métricas por classe:**

| Classe    | Precision | Recall | F1-score | Suporte |
|----------|-----------|--------|----------|---------|
| negative | 0.8785    | 0.9038 | 0.8910   | 104     |
| positive | 0.8925    | 0.8646 | 0.8783   | 96      |

**Matriz de confusão (true × predicted):**

```text
[[94 10]
 [13 83]]
```

### 9. URL do Youtube

https://www.youtube.com/watch?v=6m6jaZE9oho

### 10. Autores

- Guilherme Vinicius Sennes Domingues – 10751468@mackenzista.com.br
- Mateus Klein Lourenço – 10388729@mackenzista.com.br
- Wendell de lima - 10746314@mackenzista.com.br

