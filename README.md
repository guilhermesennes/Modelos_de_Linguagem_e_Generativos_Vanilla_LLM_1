# Classificação de Sentimento em Reviews do IMDb com Modelos Abertos (Zero-Shot vs RAG)

Projeto desenvolvido para a disciplina de **Modelos de Linguagem e Generativos**, comparando uma abordagem de **Text Zero-Shot Classification** com uma solução baseada em **Retrieval-Augmented Generation (RAG)**, utilizando exclusivamente modelos e recursos abertos e executáveis em Google Colab.

---

## 1. Introdução

O objetivo deste trabalho é classificar automaticamente o **sentimento** de reviews de filmes do IMDb como **positivo** ou **negativo**, empregando modelos de linguagem abertos.

São comparadas duas abordagens:

1. Um modelo **Zero-Shot** usando `facebook/bart-large-mnli`, que já vem pré-treinado e consegue fazer classificação textual sem fine-tuning específico no dataset.
2. Uma solução com **RAG (Retrieval-Augmented Generation)**, combinando um modelo de embeddings (`all-MiniLM-L6-v2`) com um modelo gerativo (`google/flan-t5-base`), em que exemplos recuperados do próprio dataset são usados para orientar a decisão do LLM.

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
- **200 exemplos** para teste (amostra do split de teste original).

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

### 3.3 Abordagem 2 – RAG com MiniLM + FLAN-T5

Na segunda abordagem, foi implementada uma versão de **RAG** adaptada à tarefa de classificação de sentimento:

1. **Base de exemplos rotulados**  
   - Um subconjunto de reviews de treino é selecionado.  
   - Cada review possui `text` e `label` (`negative` ou `positive`).

2. **Embeddings semânticos (Retriever)**  
   - É utilizado o modelo `sentence-transformers/all-MiniLM-L6-v2` para gerar **embeddings** dos textos de treino.  
   - Esses embeddings formam uma base vetorial de exemplos rotulados (negative/positive), indexada para busca vetorial.

3. **Recuperação de exemplos (Retrieval)**  
   - Para cada review do conjunto de teste, gera-se o embedding correspondente.  
   - São recuperados os *k* exemplos mais similares na base, usando produto interno ou similaridade de cosseno entre vetores.  

4. **Construção do prompt (Augmented Generation)**  
   - Os reviews recuperados e seus rótulos são inseridos em um **prompt few-shot** para o modelo gerativo `google/flan-t5-base`, utilizando o pipeline de `text2text-generation` da biblioteca `transformers`.  
   - O prompt inclui:
     - uma instrução clara para atuar como classificador de sentimento;  
     - alguns exemplos reais com seus rótulos;  
     - o review alvo, que deve ser rotulado.  
   - O modelo é instruído a responder **apenas** com uma palavra: `positive` ou `negative`.

5. **Controle de tamanho de contexto**  
   - Para manter o prompt compacto e o processamento eficiente, são aplicados:
     - truncamento de tokens nos exemplos (limite máximo por exemplo);  
     - truncamento no review alvo;  
     - ajuste do número de exemplos *k* no prompt.

6. **Parsing da resposta**  
   - A saída do FLAN-T5 é pós-processada para extrair apenas `positive` ou `negative`, que são convertidos em rótulos de classe.  

### 3.4 Metodologia de Mensuração dos Resultados

A mensuração dos resultados segue uma metodologia padronizada de avaliação de modelos de classificação supervisionada, utilizando as funções da biblioteca `scikit-learn`:

1. **Predição sobre o conjunto de teste**  
   - Para cada uma das duas abordagens (Zero-Shot e RAG), o modelo gera uma previsão de rótulo (`positive` ou `negative`) para **todos os 200 exemplos** do conjunto de teste.  
   - No caso do Zero-Shot, a previsão é o rótulo com maior score retornado pelo pipeline de `zero-shot-classification`.  
   - No caso do RAG, a previsão é obtida a partir da resposta textual do FLAN-T5, pós-processada para mapear para `positive` ou `negative`.

2. **Comparação com os rótulos verdadeiros**  
   - As previsões são comparadas diretamente com o campo `label` do dataset (`0 = negative`, `1 = positive`), convertendo os rótulos textuais para valores numéricos quando necessário.

3. **Cálculo das métricas**  
   - São utilizadas as funções:
     - `confusion_matrix(y_true, y_pred)`  
     - `accuracy_score(y_true, y_pred)`  
     - `precision_score(y_true, y_pred, average='binary' ou 'macro')`  
     - `recall_score(y_true, y_pred, average='binary' ou 'macro')`  
     - `f1_score(y_true, y_pred, average='binary' ou 'macro')`  
     - `classification_report(y_true, y_pred)`  
   - A partir dessas funções, são obtidas:
     - **Matriz de confusão**, detalhando acertos e erros por classe;  
     - **Acurácia**, que mede a proporção total de acertos;  
     - **Precision**, **Recall** e **F1-score** por classe (`negative` e `positive`), além de médias macro e ponderadas.  

4. **Comparação entre modelos**  
   - As métricas são calculadas separadamente para cada abordagem.  
   - A comparação é feita principalmente em termos de **acurácia** e **F1 macro**, além da análise qualitativa da matriz de confusão (equilíbrio entre as classes).

Essa metodologia garante uma comparação justa entre Zero-Shot e RAG, pois ambos são avaliados exatamente no mesmo conjunto de teste e com o mesmo conjunto de métricas.

---

## 4. Resultados

### 4.1 Baseline – Zero-Shot com BART-MNLI

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
````

O modelo demonstra desempenho **elevado e equilibrado** nas duas classes, funcionando como um bom classificador de sentimento sem qualquer fine-tuning específico no IMDb.

### 4.2 RAG – MiniLM + FLAN-T5

Para a abordagem RAG com MiniLM e FLAN-T5, a matriz de confusão obtida foi:

```text
[[85 19]
 [44 52]]
```

* 85 reviews negativos corretamente classificados como `negative`;
* 19 negativos classificados como `positive`;
* 52 reviews positivos corretamente classificados como `positive`;
* 44 positivos classificados como `negative`.

**Métricas agregadas:**

* **Accuracy:** 0.685
* **Precision global:** ≈ 0.73
* **Recall global:** ≈ 0.54
* **F1-score global:** ≈ 0.62

**Por classe (classification report):**

| Classe   | Precision | Recall | F1-score | Suporte |
| -------- | --------- | ------ | -------- | ------- |
| negative | 0.66      | 0.82   | 0.73     | 104     |
| positive | 0.73      | 0.54   | 0.62     | 96      |

O modelo com RAG apresenta bom recall para a classe negativa (0,82), indicando que consegue recuperar a maioria dos reviews negativos, mas desempenho mais fraco na classe positiva (recall de 0,54), errando uma parte relevante dos reviews positivos.

---

## 5. Discussão e Conclusão

Os resultados permitem destacar alguns pontos:

1. O modelo **Zero-Shot com BART-MNLI** se mostrou **muito eficiente** para a tarefa de classificação de sentimento em reviews do IMDb, mesmo sem qualquer fine-tuning. A acurácia de 88,5% e o F1 equilibrado entre as classes indicam que este modelo é altamente adequado como classificador de base.

2. A abordagem com **RAG (MiniLM + FLAN-T5)** atingiu desempenho **intermediário**, com acurácia em torno de 68,5% e F1 global em torno de 0,62. O modelo passou a reconhecer a maioria dos reviews negativos (recall 0,82), mas ainda apresenta dificuldade importante em recuperar corretamente todos os positivos (recall 0,54).

3. Um ponto que gerou discussão dentro da equipe foi justamente o fato de o **RAG não ter sido superior** ao modelo Zero-Shot. A expectativa inicial era que a combinação de retrieval com exemplos few-shot no prompt melhorasse o desempenho. Em pesquisas e leituras complementares, foi observado que, de forma geral, RAG costuma trazer mais benefício em tarefas que dependem de **conhecimento externo complexo** e contexto longo (por exemplo, perguntas e respostas sobre documentos, bulário ou artigos científicos), e **não é necessariamente a solução mais eficiente para problemas de classificação binária ou multiclasse elementares**, nos quais já existem modelos especializados e bem treinados.

4. Em tarefas binárias relativamente simples, com dados bem representados no domínio de treinamento de modelos como BART-MNLI, a solução Zero-Shot pode ser mais eficaz, robusta e simples de operar do que arquiteturas mais complexas com LLMs gerativos e RAG.

5. Por outro lado, a experiência com RAG neste projeto é valiosa do ponto de vista didático: evidencia a sensibilidade da abordagem a limites de contexto, seleção de exemplos e design do prompt, e abre espaço para aplicações futuras em cenários em que a recuperação de documentos longos e específicos seja realmente determinante.

Em síntese, o trabalho atende ao enunciado proposto: implementa uma solução de **Text Zero-Shot Classification** com dados do IMDb, compara com uma solução baseada em **RAG**, apresenta a metodologia de mensuração e as métricas de cada abordagem, discute criticamente os resultados e reflete sobre em quais tipos de tarefa cada estratégia tende a ser mais adequada.

---

## 6. Como Executar (Colab)

O projeto é composto por dois notebooks principais:

* `ModeloZeroShot.ipynb` – baseline Zero-Shot com BART-MNLI;
* `ModeloRAG.ipynb` – abordagem RAG com MiniLM + FLAN-T5.

### Passos gerais

1. Abrir o notebook desejado em **Google Colab**.
2. Em *Runtime > Change runtime type*, selecionar **GPU**.
3. Executar as células na ordem em que aparecem:

   * Instalação das dependências;
   * Carregamento do dataset IMDb;
   * Execução do modelo (Zero-Shot ou RAG);
   * Cálculo e exibição das métricas de avaliação.
4. Ao final, o notebook mostra as métricas (accuracy, precision, recall, F1) e a matriz de confusão.

As dependências principais estão listadas em `requirements.txt`.

---

## 7. Mini Plano de Negócios – Aplicação Real

Uma aplicação prática desse projeto é a criação de uma **API de análise de sentimento** para empresas que recebem grande volume de comentários de usuários, como:

* Plataformas de streaming de vídeo e música;
* Lojas virtuais e marketplaces;
* Aplicativos de serviços (delivery, transporte, hospedagem).

### Proposta de valor

* Classificação automática de reviews como **positivos** ou **negativos**;
* Geração de métricas agregadas por filme, produto ou loja;
* Monitoramento contínuo de satisfação do cliente e detecção rápida de problemas.

Um exemplo próximo no mundo real é a **seção de opiniões de produtos em marketplaces como o Mercado Livre**, onde sistemas baseados em IA conseguem resumir rapidamente os principais pontos das avaliações dos compradores (o que os usuários mais elogiam e criticam). Uma API construída com base neste projeto poderia alimentar funcionalidades semelhantes, oferecendo um resumo quantitativo e, eventualmente, qualitativo das opiniões dos usuários.

### Modelo de negócio

* Oferta como **SaaS** (Software as a Service), com planos baseados em volume de textos analisados por mês;
* Possibilidade de implantação **on-premise** para empresas com requisitos rígidos de privacidade e compliance, já que os modelos utilizados são **abertos** e podem rodar em infraestrutura própria.

### Diferenciais

* Uso exclusivo de **modelos abertos**, evitando dependência de APIs proprietárias;
* Flexibilidade para ajuste fino, substituição de modelos ou adaptação para outros idiomas e domínios;
* Potencial para estender a solução para tasks mais complexas, como classificação multi-rótulo, sumarização de feedbacks e detecção de tópicos.

---

## 8. Tecnologias Utilizadas

* Python 3.x
* `transformers`
* `datasets`
* `sentence-transformers`
* `scikit-learn`
* Google Colab (GPU)

As versões específicas estão definidas em `requirements.txt`.

---

## 9. URL do Youtube

[https://www.youtube.com/watch?v=6m6jaZE9oho](https://www.youtube.com/watch?v=6m6jaZE9oho)

---

## 10. Autores

* Guilherme Vinicius Sennes Domingues – [10751468@mackenzista.com.br](mailto:10751468@mackenzista.com.br)
* Mateus Klein Lourenço – [10388729@mackenzista.com.br](mailto:10388729@mackenzista.com.br)
* Wendell de Lima – [10746314@mackenzista.com.br](mailto:10746314@mackenzista.com.br)

