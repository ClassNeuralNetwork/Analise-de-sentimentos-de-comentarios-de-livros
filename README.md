# Sistema de Avaliação de Sentimentos de Livros

O sistema de análise de sentimentos foi feito usando como base os comentários para a avaliação de livros na Amazon, utilizando o dataset `Amazon Book Reviews 2023`.

É possível enxergar este problema sob uma ótica de **aprendizado supervisionado**, onde a partir do *dataset* dos comentários das resenhas de livros da Amazon, iremos extrair algumas amostras e categorizar as análises como positivas, neutras ou negativas. Em seguida, um novo *dataset* com menos amostras e traduzidas será salvo, contendo além de uma coluna de comentários, uma coluna constando qual é a classe do comentário. 

Evidenciando um problema clássico de **classificação por multi-classe**, a tarefa do modelo pré-treinado exportado será processar o conjunto de treino que representa parte do *dataset* traduzido e devolver os valores crus das probabilidades, os quais serão passados por uma função de ativação *softmax* a qual será responsável por entregar um vetor contendo as probabilidades de cada uma das classes, dentre elas, a que tiver a maior probabilidade irá representar, consequentemente, a confiança do modelo em avaliar o sentimento daquele comentário. 

**ACESSE O ANALISADOR DE SENTIMENTOS AQUI**: https://analisadores-de-sentimentos.streamlit.app/

---

# Sumário 

- [Como usar](#como-usar)
- [Parte 1 : Tratamento do Dataset](#parte-1--tratamento-do-dataset)
- [Parte 2 : Modelo](#parte-2--modelo)
- [Parte 3 : App](#parte-3--app)
- [Parte 4 : Deploy](#parte-4--deploy)

---

## Como usar

Primeiramente, é necessário instalar as dependências. Para isso, basta executar o comando abaixo:

```bash
pip install -r requirements.txt
```

Em seguida, é preciso ativar o ambiente virtual:

```bash
source venv/bin/activate
```

Como o modelo final já está disponível no repositório (via Git LFS), você não precisa rodar o notebook para realizar o treinamento. Basta executar o app diretamente:

```bash
streamlit run app.py
```

Caso queira ver o desenvolvimento do sistema mais afundo, confira o arquivo `sistema_de_avaliacao_de_sentimento.ipynb`. Nele você pode conferir todos os passos do desenvolvimento do sistema de análise de sentimentos, desde a coleta de dados até testes rápidos do modelo.

## Parte 1 : Tratamento do Dataset

### Carregamento do dataset 

O *dataset* que será aplicado será o `Amazon Book Reviews 2023`. Ele não foi utilizado de forma completa. Nesse sentido, o conjunto de dados foi carregado no modo *streaming* para que ao invés de baixar tudo, baixar apenas uma quantidade escolhida de amostras (i.e. comentários), determinadas por `SAMPLE_SIZE`. Isso será necessário por que o tamanho original do conjunto de dados é de 750 GB.

### Mapeando as notas para as 3 classes 

Para mapear as notas das avaliações em "Positivo", "Neutro" e "Negativo", será necessário a existência de 3 calsses para as notas:

- Negativo: 1 e 2 estrelas
- Neutro: 3 estrelas
- Positivo: 4 e 5 estrelas 

Dessa forma, em cada linha do *dataset*, será pego as estrelas dadas pelo comentário e serão convertidas em uma das 3 classes. 

### Traduzindo os comentários contidos no dataset para pt-BR

Essa etapa será responsável por traduzir os comentários em inglês extraídos do dataset e salvar o conjunto de dados traduzido. Isso será necessário pois, como o modelo que será aplicado é em pt-BR, o conjunto de dados que será utilizado para o treino precisa, obrigatoriamente, estar em português. 

Caso contrário, o modelo vai aprender associações erradas entre as palavras que ele nunca viu durante o pré-treino, impactando negativamente a predição do sentimento.

---

## Parte 2 : Modelo 

### Carregando o modelo  

Agora iremos carregar o modelo `bert-base-portuguese-cased` juntamente com o tokenizador.

- `bert-base-portuguese-cased`

Trata-se de um modelo treinado em corpus brasileiro em partes (utilizando Wikipedia PT). Sendo "*cased*" pois ele é capaz de diferenciar letras maiúsculas das minúsculas. Falando de forma mais geral sobre a sua arquitetura, o modelo possui 12 camadas *Transformer*, com aproximadamente 110 milhões de parâmetros da rede. 

Como mencionado anteriormente, é daí que vem a necessidade de utilizar um dataset de comentários traduzidos em PT-BR, pois como o *bert-base-portuguese-cased* foi treinado em português, o conjunto de dados utilizado para treino precisa estar em português para que o *fine-tuning* faça sentido com as palavras em português que já foram treinadas durante a fase de pré-treino do modelo. 

- Tokenizador (*tokenizer*)

O tokenizador é a entidade responsável por realizar a etapa de tokenização do texto, quebrando as palavras e convertendo em vetores de ID's numéricos para que, quando inseridos no modelo *bert*, passe pelas etapas de *embedding* posicional e no bloco de *encoder*.

### Dividindo o conjunto de dados

Com o conjunto de dados já tratado, ele será dividido em uma parte de treino e outra parte de validação. Ou seja, uma parte será utilizada para o aprendizado do modelo e a outra parte será utilizada como forma de avaliar o modelo.

### Re-treinamento do modelo 

Pelo fato do `bert-base-portuguese-cased` ser um modelo que já foi treinado anteriormente em grandes volumes de texto, ele já aprendeu a "entender" a lingua portuguesa de forma geral.

Nesse sentido, essa etapa de re-treinamento será responsável por pegar o modelo já treinado e especializa-lo na seguinte tarefa: classificar os sentimentos dos comentários a respeito dos livros. 

#### Configurações de fine tuning

Para esse re-treino, serão *setadas* as configurações de ajuste-fino (*fine-tuning*) para que o modelo possa se tornar um especialista em livros. Logo abaixo estão listados os principais parâmetros utilizados para o ajuste fino, juntamente com a explicação deles.

- `num_train_epochs` indica a quantidade de épocas a serem treinadas.

Em uma época, o modelo vê todo o *dataset* de treino uma vez. Com algumas épocas ele passa pelo *dataset*, ajustando os pesos a cada passagem.

É preciso ter equilíbrio neste parâmetro pois com poucas épocas, modelo não aprende o suficiente. Já com muitas épocas, pode ocasionar probleamas de sobre ajuste (*overfitting*) pois ele acaba decorando o treino, indo mal em dados novos.

Neste caso, como eu realizei esse processo localmente na minha máquina, eu apliquei apenas 3 épocas.

- `per_device_train_batch_size` e `per_device_eval_batch_size` indicam o tamanho do lote de amostras (comentários) que serão processadas para o conjunto de dados de treino e de validação.

Em vez de aprender uma frase por vez, o modelo processa um lote com quantidades maiores de frases simultaneamente. Entretanto, é preciso ter um equilíbrio para a quantidade de amostras a serem processadas por lote, pois quanto maior for, maior será o custo computacional para realizar a operação. 

Para ambos os conjuntos de treino e validação, eu apliquei em lotes de 8.

- `eval_strategy = "epoch"` indica a estratégia de validação utilizada para o re-treinamento. 

Nesse caso, a estratégia foi adotada por época, indicando que no final de cada uma das épocas, o modelo é avaliado no conjunto de validação. Permitindo acompanhar se ele está melhorando ou entrando em problemas de sobreajuste, por exemplo.

- `load_best_model_at_end = True`

Ao final do treino, ele carrega o *checkpoint* que indica o estado onde houve o melhor desempenho na validação, o qual pode não ser necessáriamente na última época.

- *Data Collator*

Sabemos que em cada comentário, as frases possuem tamanhos diferentes mas por conta das especificações em lotes exigidas em `per_device_train_batch_size` e `per_device_eval_batch_size`, todas as entradas de um lote precisam ter o mesmo tamanho. Caso o contrário, o modelo não conseguiria processar as amostras pois cada uma teriam dimensões diferentes. 

A fim de evitar esse problema, é aplicado o `DataCollatorWithPadding` o qual irá preencher o comprimento dos vetores em tamanhos fixos. Vale mencionar que o tamanho do vetor se baseia em um valor máximo estabelecido. No exemplo abaixo, coloquei como se fosse para 7. Para o projeto, coloquei um máximo de 128 tokens.

```bash
Frase 1: [101, 5789, 4321,  102,    0,    0,   0] > 7 tokens (3 zero PADs)
Frase 2: [101, 8823,  991, 2341, 7765, 3312, 102] > 7 tokens
Frase 3: [101, 4421,  102,    0,    0,    0,   0] > 7 tokens (4 zero PADs)
```

Perceba como *data collator* aplica o padding: pegando o lote, identificando o que falta para o valor máximo e completando com zeros, para preservar a dimensão do vetor de tokens.

Posteriormente dentro do modelo, esses vetores passarão por uma camada de *attention mask* que identificará o que é token e o que é apenas PAD. 

```bash
input_ids       : [101, 5789, 4321, 102,  0,  0,  0]
attention_mask  : [  1,    1,    1,   1,  0,  0,  0]
```

#### Métricas de avaliação

Ao final de cada época de treinamento, foram calculadas duas métricas: acurácia e *f1-score*.

- Acurácia (*accuracy*)

Será a razão entre as predições que acertadas (i.e. a predição do sentimento feita pelo modelo é a mesma prevista pelo rótulo) e o total de exemplos (i.e. o total de comentários pegos).

É uma métrica simples e poderosa, contudo, ela pode enganar quando partirmos do contexto em que as classes do conjunto de dados que usaremos para o treino e validação estão desbalanceadas. 

| Classe (Rótulo)       | Quantidade de Amostras   |
|-----------------------|------------------------- |
|  Positivo             |  1741                    |
|  Negativo             |  162                     |
|  Neutro               |  97                      | 
| **TOTAL DE AMOSTRAS** |  2000                    |

Como podemos visualizar, se levarmos apenas a métrica de acurácia em consideração, a classe majoritária irá enviezar a acurácia, de forma que se o modelo realizar predições apenas da classe *Positivo*, ele terá uma acurácia alta, mesmo que o modelo tenha dificuldades severas em realizar a predição das classes minoritárias *Negativo* e *Neutro*.

- *F1-Score*

É uma métrica que serve para complementar a acurácia do modelo, pois combina as métricas de precisão e *recall* em uma única métrica, levando em consideração o desbalanceamento das classes.

Logo, para cada classe o F1 calcula a precisão (dos que eu classifiquei como neutro, quantos eram realmente neutro ?) e o *recall* (dos que realmente eram neutros, quantos encontrei ?). Depois disso, F1 é definido pela média harmônica entre os dois:

```
F1_Score = 2 x (Precisão x Recall) / (Precisão + Recall)
```

Pelo fato de F1 ser um média ponderada de cada classe, ele terá este formato:

```
F1_Score_weighted = (F1_pos × 1741 + F1_neg × 162 + F1_neu × 97) / 2000
```

Note que por mais que *Negativo* e *Neutro* sejam classes minoritárias, seu desempenho é contabilizado de forma proporcional à classe majoritária *Positivo*. Dessa forma, se o modelo for péssimo em qualquer uma das classes, *F1_Score_weighted* será a métrica que refletirá isso.

#### Testando o modelo

Após o treinamento, a função `classificar_comentario()` executa 4 passos:

1) Tokeniza o comentário de entrada, transformando em vetores de IDs numéricos que serão reconhecíveis na entrada do modelo *bert*.

2) Os IDs numéricos passam pela estrutura encoder do modelo, gerando um vetor contendo os 3 *logits* de cada classe. Vale mencionar que os *logits* representam os valores brutos que saem do modelo.

3) No final do modelo, é aplicada uma função de ativação *softmax* que converte o vetor de logits em um vetor contendo a probabilidade de cada classe para aquela respectiva amostra.

4) No final, a função *Argmax* é aplicada para extrair a classe com a maior probabilidade e, dessa forma, o modelo consegue identificar qual seria a sua predição definitiva.

Vale mencionar que, nesse contexto, a confiança representa a probabilidade da classe dominante predita após a *softmax*. Quanto mais próximo dos 100%, mais "garantia" o modelo tem na predição daquele comentário.

---

## Parte 3 : App 

Logo após a fase de testes do modelo, o mesmo foi posto em uma aplicação web feita com *streamlit*, que permite ao usuário testar o modelo com frases da sua escolha. Sendo possível conferir as métricas do modelo no *app*.

![Image text](/assets/exemploanalisador.png)

---

## Parte 4 : Deploy 

A fase de **deploy** do modelo consistiu em disponibilizar o modelo de análise de sentimentos para uso em produção, permitindo que usuários finais possam interagir com ele. Para isso, foi utilizado a plataforma **Streamlit Cloud Community**, que permite o deploy de aplicações web feitas com *streamlit*. 

---

## Licença

Este projeto está licenciado sob a Licença **MIT**. Você tem permissão para usar, copiar, modificar e distribuir o software, desde que mantenha os avisos de direitos autorais originais.