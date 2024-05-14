import streamlit as st
import base64
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.write('Powered By')
st.image('DOTLOGO.png', use_column_width=True)
st.title('Herramienta para la predicción')
st.write("""
Esta aplicación realiza un pronóstico utilizando el modelo SARIMA (Seasonal Autoregressive Integrated Moving Average) sobre un conjunto de datos proporcionados por el usuario. Selecciona un archivo Excel con tus datos y elige el número de meses para el pronóstico, luego haz clic en el botón para calcular el pronóstico y ver las gráficas.
""")
st.image('Pasos.png', use_column_width=True)
# Paso 1: Cargar datos
st.header("1. Cargar datos")
st.write("Sube un archivo Excel con datos históricos, donde la primera columna sea fechas y las siguientes columnas contengan variables como ventas o ingresos.")
archivo_excel = st.file_uploader("Cargar archivo Excel", type=["xlsx"])


if archivo_excel is not None:
    # Paso 2: Elegir pronóstico
    st.header("2. Elegir pronóstico")
    # Solicitar al usuario que elija el número de meses para el pronóstico
    meses_pronostico = st.number_input('Elige el número de meses para el pronóstico:', min_value=1, max_value=36, value=12, step=1)
    
    # Paso 3: Calcular pronóstico
    st.header("3. Calcular pronóstico")
    if st.button('Calcular pronóstico'):  # Si se presiona el botón 'Calcular pronóstico'
        datos = pd.read_excel(archivo_excel)  # Leer datos desde el archivo Excel
        datos['fecha'] = pd.to_datetime(datos['fecha'])  # Convertir la columna 'fecha' a formato de fecha y hora
        datos.set_index('fecha', inplace=True)  # Establecer la columna 'fecha' como el índice
        
        ultima_fecha = datos.index[-1]  # Obtener la última fecha en los datos

        resultados = {}  # Crear un diccionario para almacenar los resultados
        fig, axs = plt.subplots(len(datos.columns), 1, figsize=(10, 5 * len(datos.columns)))  # Crear subgráficos para cada columna de datos

        if len(datos.columns) == 1:  # Si solo hay una columna de datos
            axs = [axs]  # Convertir axs en una lista para evitar problemas de iteración

        progress_bar = st.progress(0)  # Barra de progreso inicializada en 0%

        for i, columna in enumerate(datos.columns):  # Iterar sobre cada columna de datos
            # Obtener datos históricos de la columna actual y su última fecha
            datos_historicos = datos[columna].dropna()
            ultima_fecha_historico = datos_historicos.index[-1]
            
            # Generar modelo SARIMA y realizar el pronóstico
            modelo = SARIMAX(datos_historicos, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            resultado = modelo.fit()
            pronostico = resultado.get_forecast(steps=meses_pronostico)  # Pronosticar los siguientes meses
            pronostico_index = pd.date_range(start=ultima_fecha_historico, periods=meses_pronostico + 1, freq='M')  # Crear índice de fechas para el pronóstico
            
            # Pronóstico para la última fecha
            pronostico_last_date = resultado.get_forecast(steps=1)
            pronostico_last_value = pronostico_last_date.predicted_mean.values[0]
            
            # Asegurar que los valores predichos sean enteros y no negativos
            pronostico_mean_int = pronostico.predicted_mean.astype(int).clip(lower=0)
            pronostico_last_value_int = int(pronostico_last_value) if pronostico_last_value >= 0 else 0
            
            # Agregar el pronóstico para la última fecha al resultado
            resultados[columna] = list(pronostico_mean_int.values) + [pronostico_last_value_int]
            
            # Plotear datos históricos
            axs[i].plot(datos_historicos.index, datos_historicos, label='Histórico', color='green')
            
            # Plotear el pronóstico y el intervalo de confianza
            axs[i].plot(pronostico_index, list(pronostico_mean_int) + [pronostico_last_value_int], color='red')
            
            # Agregar puntos de previsión como puntos rojos con los valores escritos
            axs[i].scatter(pronostico_index, list(pronostico_mean_int) + [pronostico_last_value_int], color='red', label='Pronóstico')
            for x, y in zip(pronostico_index, list(pronostico_mean_int) + [pronostico_last_value_int]):
                axs[i].text(x, y, f'{y}', ha='left', va='bottom')  # Agregar el valor al lado del punto

            axs[i].set_title(columna)  # Establecer título de la subgráfica
            axs[i].legend()  # Mostrar leyenda
            axs[i].grid(True)  # Activar la cuadrícula en la subgráfica
            
            # Formatear el eje horizontal con el formato 'Mes-Año'
            axs[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b-%Y'))
            
            # Actualizar barra de progreso
            progress_bar.progress((i + 1) / len(datos.columns))

        # Crear DataFrame con los resultados
        df_resultados = pd.DataFrame(resultados, index=pronostico_index)

        # Guardar los resultados en un archivo Excel
        fecha_actual = datetime.now().strftime("%Y%m%d")
        nombre_archivo = f'Predicción_{meses_pronostico}_meses_{fecha_actual}.xlsx'
        df_resultados.to_excel(nombre_archivo)
        st.header("4. Resultados")
        st.write("Visualiza gráficos que comparan los datos históricos con las predicciones para cada variable. También puedes descargar un archivo Excel con los resultados.")
        # Mostrar gráficas
        st.pyplot(fig)
        
        # Mostrar mensaje de descarga
        st.success(f'Se ha generado el archivo de previsión: [{nombre_archivo}](data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{base64.b64encode(open(nombre_archivo,"rb").read()).decode()})')
