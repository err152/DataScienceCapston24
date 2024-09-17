import streamlit as st
import joblib
import pandas as pd

# Cargar el modelo guardado con joblib
def load_model():
    try:
        model = joblib.load('C:Users/Eduardo/Desktop/Master/Capstone/DataScienceCapston24/data/best_random_forest_model_compressed.pkl')
        st.write("Modelo cargado con éxito.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Cargar el modelo
model = load_model()

if model:
    # Título de la aplicación
    st.title('Aplicación de Predicción de Riesgo de padecer una Enfermedad Cardiovascular')

    st.write("Esta aplicación utiliza un modelo de predicción Random Forest con SMOTE cargado desde un archivo `.pkl` en tu sistema local.")

    # Entradas del usuario
    st.header('Introduce los datos para realizar la predicción')

    # Entradas para variables numéricas
    age = st.number_input('Edad', value=50.0)
    cigsPerDay = st.number_input('Cigarrillos por Día', value=5.0)
    totChol = st.number_input('Colesterol Total', value=250.0)
    sysBP = st.number_input('Presión Arterial Sistólica', value=120.0)
    diaBP = st.number_input('Presión Arterial Diastólica', value=80.0)
    BMI = st.number_input('Índice de Masa Corporal (BMI)', value=0.0)
    heartRate = st.number_input('Frecuencia Cardíaca', value=0.0)
    glucose = st.number_input('Glucosa en Plasma', value=100.0)

    # Entradas para variables categóricas tratadas como numéricas
    gender = st.number_input('Género (0=Hombre, 1=Mujer)', min_value=0, max_value=1, value=0)
    education = st.number_input('Educación (numérico)', min_value=0, max_value=3, value=0)
    currentSmoker = st.number_input('Fumador Actual (0=No, 1=Sí)', min_value=0, max_value=1, value=0)
    BPMeds = st.number_input('Medicamentos para la Presión (0=No, 1=Sí)', min_value=0, max_value=1, value=0)
    prevalentStroke = st.number_input('Accidente Cerebrovascular Prevalente (0=No, 1=Sí)', min_value=0, max_value=1, value=0)
    prevalentHyp = st.number_input('Hipertensión Prevalente (0=No, 1=Sí)', min_value=0, max_value=1, value=0)
    diabetes = st.number_input('Diabetes (0=No, 1=Sí)', min_value=0, max_value=1, value=0)

    # Botón para predecir
    if st.button('Predecir'):
        # Crear un DataFrame con las entradas
        input_data = pd.DataFrame([[age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose,
                                    gender, education, currentSmoker, BPMeds, prevalentStroke, prevalentHyp, diabetes]],
                                  columns=["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
                                           "gender", "education", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"])

        # Realizar predicción con el modelo
        prediction = model.predict(input_data)
        
        # Mostrar el resultado
        st.subheader(f'Predicción del modelo: {prediction[0]}')
