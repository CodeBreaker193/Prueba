import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from datetime import timedelta

# Configuración inicial de Streamlit
st.set_page_config(
    page_title="Predicción de Indicadores Clave",
    layout="wide",
)

# Variables globales
modelo_stock = None
modelo_servicio = None
datos_filtrados_global = pd.DataFrame()

# Carga de modelos entrenados
modelo_stock = load("ModeloStockSeguridadV2.joblib") 
modelo_servicio = load("ModeloNiveldeServicioV2.joblib")
scaler = load("ScalerStockAjustado.joblib")

# Títulos y descripción
st.title("Predicción de Indicadores Clave")
st.write("Aplicación para predecir stock de seguridad y nivel de servicio con modelos entrenados.")


def calcular_fecha_limite(datos, predicciones_stock):
    """
    Calcula la fecha límite recomendada para pedir más stock.
    """
    fechas_limite = []
    for idx, stock_predicho in enumerate(predicciones_stock):
        tiempo_entrega_promedio = int(datos.iloc[idx]['Tiempo de Entrega Promedio']) 
        fecha_pedido = datos.iloc[idx]['Fecha de Pedido']
        fechas_limite.append(fecha_pedido + timedelta(days=tiempo_entrega_promedio))
    return fechas_limite

def preprocesar_datos_combinados(datos, scaler):
    """
    Preprocesa los datos para preparar las entradas necesarias para ambos modelos.
    """
    try:
        # Asegurarse de que las fechas estén en formato datetime
        datos['Fecha de Pedido'] = pd.to_datetime(datos['Fecha de Pedido'], format='%Y-%m-%d')
        datos['Fecha de Recepción'] = pd.to_datetime(datos['Fecha de Recepción'], format='%Y-%m-%d')

        # Calcular el Tiempo de Entrega (en días)
        datos['Tiempo de Entrega'] = (datos['Fecha de Recepción'] - datos['Fecha de Pedido']).dt.days

        # Calcular Tiempo de Entrega Promedio por Producto
        tiempo_entrega_promedio = datos.groupby('Producto')['Tiempo de Entrega'].mean().reset_index()
        tiempo_entrega_promedio.rename(columns={'Tiempo de Entrega': 'Tiempo de Entrega Promedio'}, inplace=True)

        # Fusionar Tiempo de Entrega Promedio con los datos originales
        datos = datos.merge(tiempo_entrega_promedio, on='Producto', how='left')

        # Calcular Retrasos en Entrega
        datos['Retrasos en Entrega'] = (datos['Cantidad Pedida'] > datos['Cantidad Entregada']).astype(int)

        # Escalar Stock Actual
        datos['Stock Actual Escalado'] = scaler.transform(datos[['Stock Actual']])
        datos.rename(columns={'Stock Actual Escalado': 'Stock Actual'}, inplace=True)

        # Definir el orden exacto de las características
        columnas_requeridas_stock = ['Cantidad Pedida', 'Cantidad Entregada', 'Tiempo de Entrega Promedio', 'Stock Actual']
        columnas_requeridas_servicio = ['Cantidad Pedida', 'Cantidad Entregada', 'Tiempo de Entrega Promedio', 'Retrasos en Entrega']

        # Validar si las columnas requeridas existen
        for col in columnas_requeridas_stock + columnas_requeridas_servicio:
            if col not in datos.columns:
                st.error(f"Falta la columna requerida: {col}")
                raise ValueError(f"Falta la columna requerida: {col}")

        # Reordenar columnas para cada modelo
        datos_stock = datos[columnas_requeridas_stock].loc[:, ~datos[columnas_requeridas_stock].columns.duplicated()]
        datos_servicio = datos[columnas_requeridas_servicio]

        # Mostrar mensajes en Streamlit
        st.write("Columnas para modelo Stock de Seguridad:", datos_stock.columns.tolist())
        st.write("Columnas para modelo Nivel de Servicio:", datos_servicio.columns.tolist())

        return datos_stock, datos_servicio

    except Exception as e:
        st.error(f"Error en preprocesar_datos_combinados: {e}")
        return None, None

def aplicar_filtros(datos, producto):
    """
    Aplica el filtro por Producto al DataFrame.
    """
    try:
        if producto != "Todos":
            datos = datos[datos['Producto'] == producto]
        return datos
    except Exception as e:
        st.error(f"Error al aplicar el filtro: {e}")
        return pd.DataFrame()  # Retornar un DataFrame vacío en caso de error


def realizar_predicciones(archivo, producto):
    """
    Realiza predicciones con los modelos entrenados y actualiza las gráficas y la tabla de resultados.
    """
    try:
        # Cargar datos desde el archivo
        datos = pd.read_csv(archivo)

        # Preprocesar los datos
        datos_stock, datos_servicio = preprocesar_datos_combinados(datos, scaler)

        # Validar si el preprocesamiento devolvió valores
        if datos_stock is None or datos_servicio is None:
            st.error("El preprocesamiento no devolvió datos válidos.")
            return None

        # Realizar predicciones
        predicciones_stock = modelo_stock.predict(datos_stock)
        predicciones_servicio = modelo_servicio.predict(datos_servicio)

        # **Ajuste 1: Escalar las predicciones del Stock de Seguridad**
        factor_ajuste_stock = datos['Stock Actual'].mean() / predicciones_stock.mean()
        predicciones_stock_ajustadas = predicciones_stock * factor_ajuste_stock

        # **Ajuste 2: Suavizado para reducir fluctuaciones extremas**
        predicciones_stock_ajustadas = pd.Series(predicciones_stock_ajustadas).rolling(window=3, min_periods=1).mean()

        # **Ajuste 3: Limitar predicciones dentro de un rango razonable**
        limite_superior_stock = datos['Stock Actual'].quantile(0.95)  # 95% del valor real
        limite_inferior_stock = datos['Stock Actual'].quantile(0.05)  # 5% del valor real
        predicciones_stock_ajustadas = predicciones_stock_ajustadas.clip(lower=limite_inferior_stock, upper=limite_superior_stock)

        # Agregar predicciones ajustadas al DataFrame original
        datos['Predicción Stock de Seguridad'] = predicciones_stock_ajustadas
        datos['Predicción Nivel de Servicio'] = predicciones_servicio

        # Ajustar y limitar el Nivel de Servicio
        datos['Predicción Nivel de Servicio'] = (datos['Predicción Nivel de Servicio'] * 1.5).clip(upper=100)

        # Clasificar Nivel de Servicio
        datos['Predicción Nivel de Servicio'] = pd.to_numeric(datos['Predicción Nivel de Servicio'], errors='coerce')
        datos['Predicción Nivel de Servicio'] = datos['Predicción Nivel de Servicio'].fillna(0)
        datos['Clasificación Nivel de Servicio'] = datos['Predicción Nivel de Servicio'].apply(clasificar_nivel_servicio)

        # Aplicar filtro por producto si es necesario
        if producto != "Todos":
            datos = datos[datos['Producto'] == producto]

        # Mostrar los resultados en Streamlit
        st.subheader("Tabla de Resultados")
        st.write(datos)

        # Generar gráficos
        st.subheader("Gráficos")
        st.line_chart(datos[['Fecha de Pedido', 'Stock Actual', 'Predicción Stock de Seguridad']].set_index('Fecha de Pedido'))
        st.line_chart(datos[['Fecha de Pedido', 'Cantidad Entregada', 'Predicción Nivel de Servicio']].set_index('Fecha de Pedido'))

        # Guardar resultados
        datos.to_csv("resultados_predicciones.csv", index=False)
        st.success("Predicciones guardadas en: resultados_predicciones.csv")

        return datos  # Devolver datos procesados y predicciones

    except ValueError as ve:
        st.error(f"Error de validación: {ve}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return None

def clasificar_nivel_servicio(nivel):
    """
    Clasifica el nivel de servicio según su porcentaje.
    """
    if nivel >= 90:
        return "Excelente"
    elif nivel >= 75:
        return "Bueno"
    elif nivel >= 50:
        return "Aceptable"
    else:
        return "Deficiente"

def actualizar_grafico_proveedor_producto(datos):
    """
    Actualiza el gráfico de relación Proveedor-Producto para ser utilizado en Streamlit.
    """
    try:
        import matplotlib.pyplot as plt

        # Abreviar nombres de proveedores
        datos['Proveedor Corto'] = datos['Proveedor'].apply(lambda x: x.split()[0])  # Usa solo la primera palabra

        # Crear la figura y el eje
        fig, ax = plt.subplots(figsize=(10, 6))  # Ajustar el tamaño del gráfico

        # Crear gráfico de barras apiladas
        datos.groupby(['Proveedor Corto', 'Producto']).size().unstack().plot(kind='bar', stacked=True, ax=ax)

        # Títulos y etiquetas
        ax.set_title("Relación Proveedor-Producto")
        ax.set_xlabel("Proveedor")
        ax.set_ylabel("Cantidad de Productos")

        # Ajustar la leyenda
        ax.legend(title="Producto", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        # Mostrar la figura en Streamlit
        import streamlit as st
        st.pyplot(fig)

    except Exception as e:
        import streamlit as st
        st.error(f"Error en actualizar_grafico_proveedor_producto: {e}")

def actualizar_graficas(datos, titulo, etiquetas=None):
    """
    Actualiza las gráficas con datos predichos, usando etiquetas más representativas si están disponibles.
    Compatible con Streamlit.
    """
    import matplotlib.pyplot as plt
    import streamlit as st

    if len(datos) == 0:
        st.warning(f"No hay datos para generar la gráfica: {titulo}")
        return

    # Crear la figura
    fig, ax = plt.subplots(figsize=(8, 5))  # Ajustar tamaño de la gráfica

    # Generar la gráfica
    ax.plot(range(len(datos)), datos, marker="o")

    # Configurar etiquetas si están disponibles
    if etiquetas is not None and len(etiquetas) == len(datos):
        ax.set_xticks(range(len(etiquetas)))
        ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)  # Ajustar rotación y tamaño de fuente
        ax.set_xlabel("Producto")  # Etiqueta más descriptiva para el eje X
    else:
        ax.set_xlabel("Índice")

    # Configurar títulos y etiquetas
    ax.set_title(titulo)
    ax.set_ylabel("Valor")

    # Ajustar márgenes y espaciamiento
    fig.tight_layout()

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)

def actualizar_graficas_principal(archivo, producto, modo):
    """
    Actualiza las gráficas dependiendo del modo seleccionado para Streamlit.
    """
    # Realizar las predicciones y obtener los datos procesados
    datos = realizar_predicciones(archivo, producto)
    
    if datos is not None:
        if modo == "Comparativo":
            # Generar gráficas comparativas
            st.subheader("Gráfica Comparativa: Actual vs Predicción")
            col1, col2 = st.columns(2)  # Dividir en dos columnas para mostrar ambas gráficas lado a lado
            with col1:
                graficar_comparativo(datos, tipo="Stock de Seguridad")
            with col2:
                graficar_comparativo(datos, tipo="Nivel de Servicio")
        elif modo == "Individual":
            # Mostrar mensaje sobre modo individual (gráficas ya renderizadas en realizar_predicciones)
            st.subheader("Gráficas Individuales: Predicción de Indicadores")
            actualizar_graficas(
                datos['Predicción Stock de Seguridad'].tolist(),
                titulo="Predicción de Stock de Seguridad",
                etiquetas=datos['Producto'].tolist()
            )
            actualizar_graficas(
                datos['Predicción Nivel de Servicio'].tolist(),
                titulo="Predicción de Nivel de Servicio",
                etiquetas=datos['Producto'].tolist()
            )


def graficar_comparativo(datos, tipo):
    """
    Genera gráficas comparativas de datos históricos vs predicción con dimensiones ajustadas.
    """
    try:
        # Verificar columnas necesarias para el gráfico
        columnas_requeridas = ['Fecha de Pedido', 'Stock Actual', 'Predicción Stock de Seguridad', 
                               'Cantidad Entregada', 'Cantidad Pedida', 'Predicción Nivel de Servicio']
        for columna in columnas_requeridas:
            if columna not in datos.columns:
                st.error(f"Error: Falta la columna requerida '{columna}' en los datos.")
                return

        # Ordenar datos por fecha
        datos = datos.sort_values(by="Fecha de Pedido")

        # Formatear fechas para simplificar etiquetas
        datos['Fecha Formateada'] = datos['Fecha de Pedido'].dt.strftime('%m-%d')

        # Selección del gráfico según el tipo
        if tipo == "Stock de Seguridad":
            fig, ax = plt.subplots(figsize=(6, 5))  # Ajustar proporciones: menos ancho y menos alto
            ax.plot(datos['Fecha Formateada'], datos['Stock Actual'], label="Actual", color="blue")
            ax.plot(datos['Fecha Formateada'], datos['Predicción Stock de Seguridad'], label="Predicción", linestyle="--", color="orange")
            ax.set_title("Stock de Seguridad: Actual vs Predicción")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Cantidad")
            ax.legend(loc="upper right")  # Posición ajustada de la leyenda

        elif tipo == "Nivel de Servicio":
            fig, ax = plt.subplots(figsize=(6, 5))  # Ajustar proporciones: menos ancho y menos alto
            nivel_actual = (datos['Cantidad Entregada'] / datos['Cantidad Pedida']) * 100  # Calcular nivel de servicio actual
            ax.plot(datos['Fecha Formateada'], nivel_actual, label="Actual", color="blue")
            ax.plot(datos['Fecha Formateada'], datos['Predicción Nivel de Servicio'], label="Predicción", linestyle="--", color="orange")
            ax.set_title("Nivel de Servicio: Actual vs Predicción")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Porcentaje")
            ax.legend(loc="upper right")  # Posición ajustada de la leyenda

        # Ajustar densidad de etiquetas en el eje X
        step = max(1, len(datos) // 10)
        ax.set_xticks(datos['Fecha Formateada'][::step])
        ax.tick_params(axis='x', rotation=45, labelsize=8)  # Rotar y reducir tamaño de las etiquetas
        fig.tight_layout()

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error en graficar_comparativo: {e}")


def mostrar_tabla(datos):
    """
    Muestra los datos procesados como una tabla en Streamlit.
    """
    try:
        # Preparar los datos para la tabla
        tabla = datos[['Producto', 
                       'Predicción Stock de Seguridad', 
                       'Predicción Nivel de Servicio', 
                       'Fecha de Pedido', 
                       'Clasificación Nivel de Servicio']].copy()

        # Formatear las columnas
        tabla['Predicción Stock de Seguridad'] = tabla['Predicción Stock de Seguridad'].apply(lambda x: f"{x:.2f}")
        tabla['Predicción Nivel de Servicio'] = tabla['Predicción Nivel de Servicio'].apply(lambda x: f"{x:.2f}%")
        tabla['Fecha de Pedido'] = tabla['Fecha de Pedido'].dt.strftime('%Y-%m-%d')

        # Mostrar la tabla en Streamlit
        st.dataframe(tabla, use_container_width=True)

    except Exception as e:
        st.error(f"Error al mostrar la tabla: {e}")


def actualizar_tabla_streamlit(datos):
    """
    Actualiza y muestra los datos procesados como tabla en Streamlit.
    """
    try:
        # Validar datos
        columnas_necesarias = ['Producto', 
                               'Predicción Stock de Seguridad', 
                               'Predicción Nivel de Servicio', 
                               'Fecha Límite Pedido', 
                               'Clasificación Nivel de Servicio']
        for columna in columnas_necesarias:
            if columna not in datos.columns:
                st.error(f"Error: La columna necesaria '{columna}' no está disponible en los datos.")
                return

        # Preparar datos para la tabla
        datos_tabla = datos.copy()
        datos_tabla['Predicción Nivel de Servicio'] = datos_tabla['Predicción Nivel de Servicio'].apply(lambda x: f"{x:.2f}%")
        datos_tabla['Fecha Límite Pedido'] = datos_tabla['Fecha Límite Pedido'].dt.strftime('%Y-%m-%d')

        # Mostrar tabla en Streamlit
        st.dataframe(datos_tabla[columnas_necesarias], use_container_width=True)

    except Exception as e:
        st.error(f"Error al actualizar la tabla: {e}")


def seleccionar_archivo_streamlit(canvas1, canvas2, canvas_proveedor_producto, tabla):
    """
    Permite al usuario cargar un archivo y ejecuta las predicciones.
    """
    st.subheader("Carga de Archivo para Predicciones")
    archivo = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            # Leer el archivo cargado
            datos = pd.read_csv(archivo)

            # Ejecutar las predicciones
            st.success("Archivo cargado con éxito. Realizando predicciones...")
            realizar_predicciones(datos, "Todos", canvas1, canvas2, canvas_proveedor_producto, tabla)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.warning("Por favor, selecciona un archivo CSV.")

import streamlit as st
import pandas as pd

def exportar_datos_streamlit(datos_filtrados_global):
    """
    Exporta los datos filtrados en un archivo CSV desde Streamlit.
    """
    if datos_filtrados_global.empty:
        st.warning("No hay datos filtrados para exportar.")
        return

    st.subheader("Exportar Datos Filtrados")
    st.download_button(
        label="Exportar Datos a CSV",
        data=datos_filtrados_global.to_csv(index=False),
        file_name="datos_filtrados.csv",
        mime="text/csv"
    )


def aplicar_y_actualizar_streamlit(archivo, producto, proveedor, datos_filtrados_global, modelo_stock, modelo_servicio, scaler):
    """
    Aplica los filtros seleccionados, genera las predicciones si es necesario,
    y actualiza las gráficas y la tabla de resultados en Streamlit.
    """
    try:
        # Verificar si se cargó un archivo
        if archivo is None:
            st.warning("No se ha cargado ningún archivo.")
            return datos_filtrados_global

        # Cargar los datos desde el archivo
        datos = pd.read_csv(archivo)

        # Verificar si las columnas necesarias para predicciones están presentes
        columnas_requeridas = ['Producto', 'Cantidad Pedida', 'Cantidad Entregada',
                               'Fecha de Pedido', 'Fecha de Recepción', 'Stock Actual', 'Proveedor']
        for col in columnas_requeridas:
            if col not in datos.columns:
                st.error(f"El archivo cargado no contiene la columna requerida: {col}")
                return datos_filtrados_global

        # Generar predicciones si no existen
        if 'Predicción Stock de Seguridad' not in datos.columns or 'Predicción Nivel de Servicio' not in datos.columns:
            datos_stock, datos_servicio = preprocesar_datos_combinados(datos, scaler)
            datos['Predicción Stock de Seguridad'] = modelo_stock.predict(datos_stock)
            datos['Predicción Nivel de Servicio'] = modelo_servicio.predict(datos_servicio)

        # Filtrar los datos según los criterios seleccionados
        datos_filtrados = datos.copy()
        if producto != "Todos":
            datos_filtrados = datos_filtrados[datos_filtrados['Producto'] == producto]
        if proveedor != "Todos":
            datos_filtrados = datos_filtrados[datos_filtrados['Proveedor'] == proveedor]

        # Verificar si hay datos después de aplicar los filtros
        if datos_filtrados.empty:
            st.warning("No hay datos que coincidan con los filtros seleccionados.")
            return pd.DataFrame()  # Retornar un DataFrame vacío

        # Actualizar la tabla en Streamlit
        st.subheader("Resultados Filtrados")
        st.dataframe(datos_filtrados)

        # Actualizar gráficas en Streamlit
        st.subheader("Gráficas")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Stock de Seguridad: Actual vs Predicción**")
            fig1, ax1 = plt.subplots()
            ax1.plot(datos_filtrados['Fecha de Pedido'], datos_filtrados['Stock Actual'], label="Actual")
            ax1.plot(datos_filtrados['Fecha de Pedido'], datos_filtrados['Predicción Stock de Seguridad'], label="Predicción", linestyle="--")
            ax1.legend()
            st.pyplot(fig1)

        with col2:
            st.write("**Nivel de Servicio: Actual vs Predicción**")
            fig2, ax2 = plt.subplots()
            nivel_actual = (datos_filtrados['Cantidad Entregada'] / datos_filtrados['Cantidad Pedida']) * 100
            ax2.plot(datos_filtrados['Fecha de Pedido'], nivel_actual, label="Actual")
            ax2.plot(datos_filtrados['Fecha de Pedido'], datos_filtrados['Predicción Nivel de Servicio'], label="Predicción", linestyle="--")
            ax2.legend()
            st.pyplot(fig2)

        # Retornar los datos filtrados para exportar
        return datos_filtrados

    except FileNotFoundError:
        st.error("El archivo especificado no existe.")
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return pd.DataFrame()

def crear_interfaz_streamlit():
    """
    Crea la interfaz gráfica con filtros, gráficas y resultados usando Streamlit.
    """
    st.title("Predicción de Indicadores Clave")

    # Banner con el logo de la empresa
    st.image("1d.png", use_column_width=True)

    # Cargar archivo
    st.sidebar.header("Cargar archivo")
    archivo_cargado = st.sidebar.file_uploader("Selecciona un archivo CSV o Excel", type=["csv", "xlsx"])

    if archivo_cargado:
        try:
            # Leer archivo cargado
            if archivo_cargado.name.endswith(".csv"):
                datos = pd.read_csv(archivo_cargado)
            else:
                datos = pd.read_excel(archivo_cargado)

            st.sidebar.success(f"Archivo cargado: {archivo_cargado.name}")
        except Exception as e:
            st.sidebar.error(f"Error al cargar el archivo: {e}")
            return
    else:
        st.sidebar.warning("Por favor, carga un archivo para continuar.")
        return

    # Filtros
    st.sidebar.header("Filtros")
    productos = ["Todos"] + datos["Producto"].unique().tolist()
    producto_seleccionado = st.sidebar.selectbox("Selecciona un producto", productos)

    modos = ["Individual", "Comparativo"]
    modo_seleccionado = st.sidebar.selectbox("Selecciona el modo de visualización", modos)

    # Aplicar predicciones y visualizar
    if st.sidebar.button("Aplicar Predicciones"):
        try:
            # Preprocesar datos y realizar predicciones
            datos_filtrados = datos.copy()
            if producto_seleccionado != "Todos":
                datos_filtrados = datos_filtrados[datos_filtrados['Producto'] == producto_seleccionado]

            datos_filtrados = realizar_predicciones_streamlit(datos_filtrados)

            # Mostrar gráficas según el modo seleccionado
            if modo_seleccionado == "Comparativo":
                st.subheader("Gráficas Comparativas")
                graficar_comparativo_streamlit(datos_filtrados)
            else:
                st.subheader("Gráficas Individuales")
                mostrar_graficas_individuales(datos_filtrados)

            # Mostrar tabla
            st.subheader("Tabla de Resultados")
            st.dataframe(datos_filtrados)

        except Exception as e:
            st.error(f"Error al aplicar predicciones: {e}")

def realizar_predicciones_streamlit(datos):
    """
    Realiza predicciones y ajusta los datos con los modelos entrenados.
    """
    datos_stock, datos_servicio = preprocesar_datos_combinados(datos, scaler)

    datos['Predicción Stock de Seguridad'] = modelo_stock.predict(datos_stock)
    datos['Predicción Nivel de Servicio'] = modelo_servicio.predict(datos_servicio)

    # Ajustar y limitar valores
    datos['Predicción Nivel de Servicio'] = (datos['Predicción Nivel de Servicio'] * 1.5).clip(upper=100)

    return datos

def graficar_comparativo_streamlit(datos):
    """
    Genera gráficos comparativos de datos históricos vs predicción.
    """
    fig1, ax1 = plt.subplots()
    ax1.plot(datos['Fecha de Pedido'], datos['Stock Actual'], label="Actual")
    ax1.plot(datos['Fecha de Pedido'], datos['Predicción Stock de Seguridad'], label="Predicción", linestyle="--")
    ax1.set_title("Stock de Seguridad: Actual vs Predicción")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    nivel_actual = (datos['Cantidad Entregada'] / datos['Cantidad Pedida']) * 100
    ax2.plot(datos['Fecha de Pedido'], nivel_actual, label="Actual")
    ax2.plot(datos['Fecha de Pedido'], datos['Predicción Nivel de Servicio'], label="Predicción", linestyle="--")
    ax2.set_title("Nivel de Servicio: Actual vs Predicción")
    ax2.legend()
    st.pyplot(fig2)

def mostrar_graficas_individuales(datos):
    """
    Muestra las gráficas individuales de predicción.
    """
    fig1, ax1 = plt.subplots()
    ax1.plot(range(len(datos)), datos['Predicción Stock de Seguridad'], marker="o")
    ax1.set_title("Predicción de Stock de Seguridad")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(datos)), datos['Predicción Nivel de Servicio'], marker="o")
    ax2.set_title("Predicción de Nivel de Servicio")
    st.pyplot(fig2)

# Ejecutar en Streamlit
if __name__ == "__main__":
    crear_interfaz_streamlit()