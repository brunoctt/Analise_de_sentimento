# Análise de Sentimento
Análise de sentimento, para classificar se uma frase representa felicidade, tristeza ou raiva.

Utilizando frameworks de PNL é possível realizar Análise de Sentimento para determinar se uma frase é positiva, neutra ou negativa. No entanto, para classificar a emoção da frase, esta abordagem não é suficiente.
Para tal objetivo é necessário utilizar uma abordagem com um algoritmo para classificar as frases de entrada.
O dataset utilizado para treinar o algoritmo foi o [Twitter Emotion Classification](https://www.kaggle.com/code/shtrausslearning/twitter-emotion-classification/notebook), com vários tweets em inglês classificados como (tristeza, alegria, amor, raiva, medo, surpresa).

O pipeline definido para o projeto segue:
1. **Pré-processar o DataSet**: Remover tweets com emoções não desejadas ou valores nulos e traduzir o texto para português utilizando [deep-translator](https://deep-translator.readthedocs.io/en/latest/index.html);
2. **Definição de Modelo**: Definição do modelo que será utilizado;
3. **Treinamento do Modelo**: Treinamento utilizando o DataSet em português;
4. **Validação do Modelo**: Validação se o modelo alcança o objetivo proposto
