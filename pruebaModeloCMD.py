import joblib
import pandas as pd
from colorama import init, Fore, Style

# Inicializar colorama para habilitar los colores
init(autoreset=True)

print(Fore.CYAN + "----- Predictor de Enfermedades Cardiovasculares -----")
print(Fore.YELLOW + ">> Cargando modelo...")

# Ruta al archivo del modelo
model_path = 'data/best_random_forest_model_compressed.pkl'

# Cargar el modelo
model = joblib.load(model_path)
print(Fore.GREEN + ">> OK.")
print(Fore.GREEN + ">> Modelo cargado con éxito.")

# Definir las características de entrada
numeric_features = ["gender", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", 
                     "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "age_groups", "heart_rate_groups"]

# Valores predeterminados
default_values = {
    "gender": 0,
    "age": 50,
    "education": 1,
    "currentSmoker": 1,
    "cigsPerDay": 30,
    "BPMeds": 0,
    "prevalentStroke": 0,
    "prevalentHyp": 0,
    "diabetes": 0,
    "totChol": 203,
    "sysBP": 128.5,
    "diaBP": 82,
    "BMI": 18.99,
    "heartRate": 55,
    "glucose": 84,
    "age_groups": 1,
    "heart_rate_groups": 0
}

print(Fore.YELLOW + ">> Ejecución del modelo...")
print(Fore.YELLOW + ">> Introduzca los valores de los parámetros correspondientes:")

# Función para pedir valores al usuario con un valor predeterminado
def get_input(prompt, default):
    try:
        return float(input(f"{prompt} (default {default}): ") or default)
    except ValueError:
        print(Fore.RED + f"Entrada inválida. Usando valor predeterminado {default}.")
        return default

# Obtener entradas del usuario
input_data = pd.DataFrame([{
    "gender": int(get_input("Género (0=Hombre, 1=Mujer)", default_values["gender"])),
    "age": get_input("Edad", default_values["age"]),
    "education": int(get_input("Educación (numérico)", default_values["education"])),
    "currentSmoker": int(get_input("Fumador Actual (0=No, 1=Sí)", default_values["currentSmoker"])),
    "cigsPerDay": get_input("Cigarrillos por Día", default_values["cigsPerDay"]),
    "BPMeds": int(get_input("Medicamentos para la tensión (0=No, 1=Sí)", default_values["BPMeds"])),
    "prevalentStroke": int(get_input("Accidente Cerebrovascular Prevalente (0=No, 1=Sí)", default_values["prevalentStroke"])),
    "prevalentHyp": int(get_input("Hipertensión Prevalente (0=No, 1=Sí)", default_values["prevalentHyp"])),
    "diabetes": int(get_input("Diabetes (0=No, 1=Sí)", default_values["diabetes"])),
    "totChol": get_input("Colesterol Total", default_values["totChol"]),
    "sysBP": get_input("Presión Arterial Sistólica", default_values["sysBP"]),
    "diaBP": get_input("Presión Arterial Diastólica", default_values["diaBP"]),
    "BMI": get_input("Índice de Masa Corporal (BMI)", default_values["BMI"]),
    "heartRate": get_input("Frecuencia Cardíaca", default_values["heartRate"]),
    "glucose": get_input("Glucosa en Plasma", default_values["glucose"]),
    "age_groups": int(get_input("Grupos de Edad (numérico)", default_values["age_groups"])),
    "heart_rate_groups": int(get_input("Grupos de Frecuencia Cardíaca (numérico)", default_values["heart_rate_groups"]))
}], columns=numeric_features)

# Realizar predicción
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)[:, 1]

print(Fore.CYAN + ">>...")

# Mostrar los resultados
print(Fore.MAGENTA + f'>> Predicción: {prediction[0]}')
print(Fore.MAGENTA + f'>> Confianza del modelo en la predicción: {prediction_proba[0]:.2f} (0 a 1)')

print(Fore.CYAN + "----- Fin del programa ------")

# Ejemplo Positivo: 0,62,1,0,0,0,0,0,0,266,124,69,22.9,66,82,2,1
# Ejemplo Negativo: 0,50,1,1,30,0,0,0,0,203,128.5,82,18.99,55,84,1,0
