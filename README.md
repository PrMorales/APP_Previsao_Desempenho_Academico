Previsão de Desempenho Acadêmico

Este projeto foi desenvolvido como parte da conclusão do Bootcamp de Análise e Ciência de Dados da WoMakersCode.
O objetivo é prever o desempenho de estudantes em exames finais com base em hábitos de estudo, estilo de vida e fatores socioeconômicos.

-> Tecnologias e Ferramentas Utilizadas

Linguagem: Python

Ambiente: Google Colab

Framework de Deploy: Streamlit

-> Bibliotecas Principais:

pandas e numpy → manipulação e análise de dados

scikit-learn → pré-processamento, regressão linear e avaliação de modelo

joblib → salvar e carregar modelos treinados

streamlit → criação de aplicação interativa



📂 Estrutura do Projeto

train_model.py → script para pré-processamento, treino e salvamento do modelo

app.py → aplicação em Streamlit para realizar previsões de notas acadêmicas

model_artifacts/ → diretório com artefatos do modelo (modelo treinado, encoders, scaler e colunas)


🔎 Metodologia

1. Pré-processamento dos dados:

Codificação de variáveis categóricas

Normalização com StandardScaler



2. Treinamento do modelo:

Algoritmo utilizado: Regressão Linear

Avaliação por R² e Mean Squared Error (MSE)



3. Deploy com Streamlit:

Interface interativa para entrada de dados

Exibição da nota prevista e interpretação do desempenho

-> Acesse:
https://previsaodesempenhoacademico.streamlit.app/


📌 Palavras-chave

Ciência de Dados, Machine Learning, Regressão Linear, Streamlit,WoMakersCode, Python, Colab, Análise de Dados
