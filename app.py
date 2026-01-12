import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Configuración de página 
st.set_page_config(page_title="Predicción de Fuga de Clientes", page_icon="📉", layout="wide")

# Cargar modelo
@st.cache_resource
def load_model():
    return joblib.load('churn_pipeline.pkl') #esto carga tanto el modelo como el preprocesamiento hecho 

try:
    pipeline = load_model()
except FileNotFoundError:
    st.error("Error: no se encuentra el archivo")
    st.stop()
    
# Inputs del modelo
st.sidebar.title("Perfil del Cliente")
st.sidebar.markdown("Define las características del cliente para evaluar su riesgo.")

def user_input_features():
    # 1. Datos del contrato
    st.sidebar.subheader("1. Contrato y costes")
    contract = st.sidebar.selectbox('Tipo de contrato', ['Month-to-month', 'One year', 'Two year'])
    tenure = st.sidebar.slider('Antigüedad en meses', 0, 72, 12)
    monthly_charges = st.sidebar.slider('Cuota mensual', 5.0, 120.0, 20.0)
    
    # Calcular total_charges de forma automatica
    total_charges = tenure * monthly_charges
    st.sidebar.info(f" Importe total del cliente: ${total_charges:.2f}")

    # 2. Servicios
    st.sidebar.subheader("2. Servicios contratados")
    phone_service = st.sidebar.radio("Servicio telefónico", ["Yes", "No"], horizontal=True)
    # Si no tiene servicio móvil se rellenan automaticamente las que dependan de ello
    if phone_service == "Yes":
        multiple_lines = st.sidebar.selectbox("Líneas múltiples", ["No", "Yes", "No phone service"])
    else:
        multiple_lines = "No phone service"
        
    internet_service = st.sidebar.selectbox('Servicio de internet', ['DSL', 'Fiber optic', 'No'])
    
    # Si no tiene Internet, automaticamente se rellenan las columnas que dependan de ello
    if internet_service != 'No':
        online_security = st.sidebar.selectbox('Seguridad online', ['Yes', 'No'])
        tech_support = st.sidebar.selectbox('Soporte técnico', ['Yes', 'No'])
        online_backup = st.sidebar.selectbox('Copia de seguridad', ['Yes', 'No'])
        device_protection = st.sidebar.selectbox("Protección de dispositivo", ["No", "Yes", "No internet service"])
        tech_support = st.sidebar.selectbox("Soporte técnico", ["No", "Yes", "No internet service"])
        streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.sidebar.selectbox("Streaming películas", ["No", "Yes", "No internet service"])
    else:
        online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No internet service"

    # 3. Datos demográficos
    st.sidebar.subheader("3. Otros Datos")
    gender = st.sidebar.radio("Género", ["Male", "Female"], horizontal=True)
    payment_method = st.sidebar.selectbox('Método de pago', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless_billing = st.sidebar.radio("Factura sin papel", ['Yes', 'No'], horizontal=True)
    partner = st.sidebar.checkbox("Tiene pareja", value=False)
    dependents = st.sidebar.checkbox("Tiene personas a su cargo", value=False)
    senior_citizen = st.sidebar.checkbox("Tiene más de 65 años", value=False)

    # Convertir valores de Streamlit de True/False a Yes/No o 0/1.
    partner_str = 'Yes' if partner else 'No'
    dependents_str = 'Yes' if dependents else 'No'
    senior_val = 1 if senior_citizen else 0

    # Crear el DataFrame
    data = {
        'gender': gender, 
        'SeniorCitizen': senior_val,
        'Partner': partner_str,
        'Dependents': dependents_str,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Panel principal
st.title("Predictor de Fuga de Clientes (Telco Churn). Made by Iván Benito")
st.markdown("""
Esta aplicación utiliza un modelo de **Machine Learning (XGBoost)** para predecir la probabilidad de que un cliente cancele su servicio.
""")
st.divider()

# Resultados
    
# Botón para ejecutar
if st.button('Calcular Riesgo de abandono de cliente', type="primary"):
    with st.spinner('Analizando perfil del cliente...'):
        # Predicción
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1] 

        # Visualización
        if probability > 0.5:
            st.error(f"⚠️ **Probabilidad elevada de abandono**")
            st.markdown(f"Este cliente tiene un **{probability:.1%}** de probabilidad de abandonar la compañia.")
            st.progress(float(probability))
        else:
            st.success(f"✅ **Probabilidad baja de abandono**")
            st.markdown(f"Este cliente tiene un **{probability:.1%}** de probabilidad de abandonar la compañia.")
            st.progress(float(probability))

    # Explicabilidad

    st.subheader("Influencia de las Variables")
    st.write("Que variables estan afectando a la probabilidad de abandono del cliente (Usando los valores SHAP)")

    # Valores shap
    # 1. Extraer los componentes del pipeline para aplicarlos a los datos
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['classifier']

    # B. Transformar los datos de entrada con el preprocessor
    X_transformed = preprocessor.transform(input_df)

    # C. Obtener los nombres de las columnas después del encoding que hizo el preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # D. Crear DataFrame
    X_processed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # E. Calcular valores shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_processed_df)

    # F. Crear el gráfico Waterfall
    # El gráfico muestra cómo pasamos de la probabilidad base a la final
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], max_display=8, show=False)

    plt.title("Impacto por variable")
    st.pyplot(fig)
    st.markdown("""
        **¿Cómo leer este gráfico?**
        * **Eje horizontal (E(f(x))):** Es la probabilidad base. El modelo parte de una media y cada factor la desplaza.
        * **Bloques Rojos (Derecha):** Factores que **aumentan** la probabilidad de abandono.
        * **Bloques Azules (Izquierda):** Factores que **disminuyen** la probabilidad de abandono. 
        * **Valores numéricos:** Indican cuánto aporta  cada característica a la probabilidad final.
        """)