import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Configuração da página
st.set_page_config(page_title="Previsão de Notas Acadêmicas", page_icon="📚", layout="wide")

# Verificar se os arquivos do modelo existem
def check_model_files():
    required_files = [
        'model_artifacts/modelo_treinado.joblib',
        'model_artifacts/features.joblib',
        'model_artifacts/label_encoders.joblib',
        'model_artifacts/scaler.joblib'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

# Função para carregar os artefatos do modelo
@st.cache_resource
def load_model():
    missing_files = check_model_files()
    if missing_files:
        st.error(f"Arquivos do modelo não encontrados: {', '.join(missing_files)}")
        st.error("Por favor, execute primeiro o script train_model.py para gerar os arquivos do modelo.")
        st.stop()
    
    try:
        model = joblib.load('model_artifacts/modelo_treinado.joblib')
        features = joblib.load('model_artifacts/features.joblib')
        label_encoders = joblib.load('model_artifacts/label_encoders.joblib')
        scaler = joblib.load('model_artifacts/scaler.joblib')
        return model, features, label_encoders, scaler
    except Exception as e:
        st.error(f"Erro ao carregar os artefatos do modelo: {str(e)}")
        st.stop()

# Carregar os artefatos
model, features, label_encoders, scaler = load_model()

# Mapeamentos para a interface
GENDER_MAP = {'Masculino': 'Male', 'Feminino': 'Female', 'Outro': 'Other'}
JOB_MAP = {'Sim': 'Yes', 'Não': 'No'}
DIET_MAP = {'Pobre': 'Poor', 'Regular': 'Fair', 'Boa': 'Good'}
EDUCATION_MAP = {
    'Nenhum': 'None',
    'Ensino Médio': 'High School',
    'Bacharelado': 'Bachelor',
    'Mestrado': 'Master'
}
INTERNET_MAP = {'Pobre': 'Poor', 'Regular': 'Average', 'Boa': 'Good'}
EXTRACURRICULAR_MAP = {'Sim': 'Yes', 'Não': 'No'}

# Interface do usuário
st.title("📊 Previsão de Desempenho Acadêmico")
st.markdown("""
Este aplicativo prevê a nota de um estudante com base em seus hábitos e características pessoais.
Preencha os campos abaixo e clique em **Prever Nota** para ver o resultado.
""")

# Layout do formulário
col1, col2 = st.columns(2)

with col1:
    st.header("Informações Pessoais")
    age = st.slider("Idade", 17, 24, 20)
    gender_display = st.selectbox("Gênero", list(GENDER_MAP.keys()))
    part_time_job_display = st.selectbox("Trabalho de meio período", list(JOB_MAP.keys()))
    
    st.header("Hábitos de Estudo")
    study_hours = st.slider("Horas de estudo por dia", 0.0, 10.0, 3.0, 0.1)
    attendance = st.slider("Frequência escolar (%)", 50, 100, 85)
    extracurricular_display = st.selectbox("Participa de atividades extracurriculares?", list(EXTRACURRICULAR_MAP.keys()))

with col2:
    st.header("Hábitos de Vida")
    sleep_hours = st.slider("Horas de sono por noite", 3.0, 10.0, 7.0, 0.1)
    diet_quality_display = st.selectbox("Qualidade da dieta", list(DIET_MAP.keys()))
    exercise_freq = st.slider("Frequência de exercícios por semana", 0, 7, 3)
    
    st.header("Uso de Tecnologia")
    social_media = st.slider("Horas em redes sociais por dia", 0.0, 10.0, 2.0, 0.1)
    netflix_hours = st.slider("Horas em Netflix/streaming por dia", 0.0, 10.0, 1.0, 0.1)
    
    st.header("Outros Fatores")
    parental_education_display = st.selectbox("Nível educacional dos pais", list(EDUCATION_MAP.keys()))
    internet_quality_display = st.selectbox("Qualidade da internet", list(INTERNET_MAP.keys()))
    mental_health = st.slider("Autoavaliação de saúde mental (1-10)", 1, 10, 7)

# Processamento da previsão
if st.button("Prever Nota"):
    # Preparar os dados de entrada
    input_data = {
        'age': age,
        'gender': GENDER_MAP[gender_display],
        'study_hours_per_day': study_hours,
        'social_media_hours': social_media,
        'netflix_hours': netflix_hours,
        'part_time_job': JOB_MAP[part_time_job_display],
        'attendance_percentage': attendance,
        'sleep_hours': sleep_hours,
        'diet_quality': DIET_MAP[diet_quality_display],
        'exercise_frequency': exercise_freq,
        'parental_education_level': EDUCATION_MAP[parental_education_display],
        'internet_quality': INTERNET_MAP[internet_quality_display],
        'mental_health_rating': mental_health,
        'extracurricular_participation': EXTRACURRICULAR_MAP[extracurricular_display]
    }
    
    try:
        # Converter para DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Codificar variáveis categóricas
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))
        
        # Reordenar colunas
        input_df = input_df[features]
        
        # Normalizar os dados
        input_scaled = scaler.transform(input_df)
        
        # Fazer a previsão
        prediction = model.predict(input_scaled)[0]
        
        # Exibir resultados
        st.success(f"### Nota prevista: {prediction:.1f}/100")
        
        # Interpretação
        st.markdown("---")
        st.subheader("Interpretação")
        if prediction >= 90:
            st.success("Excelente desempenho! Continue assim.")
        elif prediction >= 70:
            st.info("Bom desempenho. Pode melhorar ainda mais.")
        elif prediction >= 50:
            st.warning("Desempenho médio. Recomendamos ajustes.")
        else:
            st.error("Desempenho abaixo da média. Necessário rever hábitos.")
            
    except Exception as e:
        st.error(f"Erro durante a previsão: {str(e)}")

# Rodapé
st.markdown("---")
st.caption("Aplicativo desenvolvido para previsão de desempenho acadêmico")
