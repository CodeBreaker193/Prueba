import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from joblib import load
from datetime import timedelta
from PIL import Image, ImageTk


# Variables globales
modelo_stock = None
modelo_servicio = None
datos_filtrados_global = pd.DataFrame()

# Carga de modelos entrenados
modelo_stock = load("ModeloStockSeguridadV2.joblib") 
modelo_servicio = load("ModeloNiveldeServicioV2.joblib")
scaler = load("ScalerStockAjustado.joblib")

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
                raise ValueError(f"Falta la columna requerida: {col}")

        # Reordenar columnas para cada modelo
        datos_stock = datos[columnas_requeridas_stock].loc[:, ~datos[columnas_requeridas_stock].columns.duplicated()]
        datos_servicio = datos[columnas_requeridas_servicio]

        # Confirmar el orden de las características
        print("Columnas para modelo Stock de Seguridad:", datos_stock.columns.tolist())
        print("Columnas para modelo Nivel de Servicio:", datos_servicio.columns.tolist())

        return datos_stock, datos_servicio

    except Exception as e:
        print(f"Error en preprocesar_datos_combinados: {e}")
        return None, None

def aplicar_filtros(datos, producto):
    """
    Aplica el filtro por Producto al DataFrame.
    """
    if producto != "Todos":
        datos = datos[datos['Producto'] == producto]
    return datos



    try:
        # Cargar datos desde el archivo
        datos = pd.read_csv(archivo)

        # Preprocesar los datos
        datos_stock, datos_servicio = preprocesar_datos_combinados(datos, scaler)

        # Validar si el preprocesamiento devolvió valores
        if datos_stock is None or datos_servicio is None:
            raise ValueError("El preprocesamiento no devolvió datos válidos.")

        # Realizar predicciones
        predicciones_stock = modelo_stock.predict(datos_stock)
        predicciones_servicio = modelo_servicio.predict(datos_servicio)

        # Agregar predicciones al DataFrame original
        datos['Predicción Stock de Seguridad'] = predicciones_stock
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

        # Llenar la tabla con datos agrupados
        datos_agrupados = datos.groupby('Producto', as_index=False).first()

        # Actualizar gráficas con datos filtrados
        actualizar_graficas(
            canvas1,
            datos_agrupados['Predicción Stock de Seguridad'].tolist(),
            "Predicción de Stock de Seguridad",
            etiquetas=datos_agrupados['Producto'].tolist()  # Etiquetas con nombres de productos
        )
        actualizar_graficas(
            canvas2,
            datos_agrupados['Predicción Nivel de Servicio'].tolist(),
            "Predicción de Nivel de Servicio",
            etiquetas=datos_agrupados['Producto'].tolist()  # Etiquetas con nombres de productos
        )
        actualizar_grafico_proveedor_producto(canvas_proveedor_producto, datos)

        # Llenar tabla con datos únicos por producto
        llenar_tabla(tabla, datos_agrupados)

        # Guardar resultados
        datos.to_csv("resultados_predicciones.csv", index=False)
        print("Predicciones guardadas en: resultados_predicciones.csv")

    except ValueError as ve:
        print(f"Error de validación: {ve}")
    except Exception as e:
        print(f"Error inesperado: {e}")

def realizar_predicciones(archivo, producto, canvas1, canvas2, canvas_proveedor_producto, tabla):
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
            raise ValueError("El preprocesamiento no devolvió datos válidos.")

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

        # Llenar la tabla con datos agrupados
        datos_agrupados = datos.groupby('Producto', as_index=False).first()

        # Actualizar gráficas con datos filtrados
        actualizar_graficas(
            canvas1,
            datos_agrupados['Predicción Stock de Seguridad'].tolist(),
            "Predicción de Stock de Seguridad",
            etiquetas=datos_agrupados['Producto'].tolist()  # Etiquetas con nombres de productos
        )
        actualizar_graficas(
            canvas2,
            datos_agrupados['Predicción Nivel de Servicio'].tolist(),
            "Predicción de Nivel de Servicio",
            etiquetas=datos_agrupados['Producto'].tolist()  # Etiquetas con nombres de productos
        )
        actualizar_grafico_proveedor_producto(canvas_proveedor_producto, datos)

        # Llenar tabla con datos únicos por producto
        llenar_tabla(tabla, datos_agrupados)

        # Guardar resultados
        datos.to_csv("resultados_predicciones.csv", index=False)
        print("Predicciones guardadas en: resultados_predicciones.csv")

        return datos  # Devolver datos procesados y predicciones

    except ValueError as ve:
        print(f"Error de validación: {ve}")
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None


def clasificar_nivel_servicio(nivel):
    if nivel >= 90:
        return "Excelente"
    elif nivel >= 75:
        return "Bueno"
    elif nivel >= 50:
        return "Aceptable"
    else:
        return "Deficiente"

def actualizar_grafico_proveedor_producto(canvas, datos):
    """
    Actualiza el gráfico de relación Proveedor-Producto.
    """
    try:
        # Abreviar nombres de proveedores
        datos['Proveedor Corto'] = datos['Proveedor'].apply(lambda x: x.split()[0])  # Usa solo la primera palabra

        # Crear la figura y el eje
        fig, ax = plt.subplots()

        # Crear gráfico de barras apiladas
        datos.groupby(['Proveedor Corto', 'Producto']).size().unstack().plot(kind='bar', stacked=True, ax=ax)

        # Títulos y etiquetas
        ax.set_title("Relación Proveedor-Producto")
        ax.set_xlabel("Proveedor")
        ax.set_ylabel("Cantidad de Productos")

        # Ajustar la leyenda
        ax.legend(title="Producto", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        # Actualizar el canvas con la figura
        canvas.figure = fig
        canvas.draw()

    except Exception as e:
        print(f"Error en actualizar_grafico_proveedor_producto: {e}")

def actualizar_graficas(canvas, datos, titulo, etiquetas=None):
    """
    Actualiza las gráficas con datos predichos, usando etiquetas más representativas si están disponibles.
    """
    if len(datos) == 0:
        print(f"No hay datos para generar la gráfica: {titulo}")
        return

    # Crear la figura
    fig, ax = plt.subplots()

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
    plt.subplots_adjust(bottom=0.3)  # Aumentar el espacio inferior para las etiquetas rotadas

    # Dibujar en el canvas
    canvas.figure = fig
    canvas.draw()

def actualizar_graficas_principal(archivo, producto, modo, canvas1, canvas2, canvas_proveedor_producto, tabla):
    """
    Actualiza las gráficas dependiendo del modo seleccionado.
    """
    datos = realizar_predicciones(archivo, producto, canvas1, canvas2, canvas_proveedor_producto, tabla)
    if datos is not None:
        if modo == "Comparativo":
            graficar_comparativo(datos, canvas1, canvas2)
        elif modo == "Individual":
            print("Modo individual ya manejado en realizar_predicciones")

def graficar_comparativo(datos, canvas1, canvas2):
    """
    Genera gráficas comparativas de datos históricos vs predicción con dimensiones ajustadas.
    """
    try:
        # Verificar columnas necesarias para el gráfico
        columnas_requeridas = ['Fecha de Pedido', 'Stock Actual', 'Predicción Stock de Seguridad', 
                               'Cantidad Entregada', 'Cantidad Pedida', 'Predicción Nivel de Servicio']
        for columna in columnas_requeridas:
            if columna not in datos.columns:
                print(f"Error: Falta la columna requerida '{columna}' en los datos.")
                return

        # Ordenar datos por fecha
        datos = datos.sort_values(by="Fecha de Pedido")

        # Formatear fechas para simplificar etiquetas
        datos['Fecha Formateada'] = datos['Fecha de Pedido'].dt.strftime('%m-%d')

        # Gráfico de Stock de Seguridad
        fig1, ax1 = plt.subplots(figsize=(6, 5))  # Ajustar proporciones: menos ancho y menos alto
        ax1.plot(datos['Fecha Formateada'], datos['Stock Actual'], label="Actual", color="blue")
        ax1.plot(datos['Fecha Formateada'], datos['Predicción Stock de Seguridad'], label="Predicción", linestyle="--", color="orange")
        ax1.set_title("Stock de Seguridad: Actual vs Predicción")
        ax1.set_xlabel("Fecha")
        ax1.set_ylabel("Cantidad")
        ax1.legend(loc="upper right")  # Posición ajustada de la leyenda

        # Ajustar densidad de etiquetas en el eje X
        step = max(1, len(datos) // 10)
        ax1.set_xticks(datos['Fecha Formateada'][::step])
        ax1.tick_params(axis='x', rotation=45, labelsize=8)  # Rotar y reducir tamaño de las etiquetas
        fig1.tight_layout()

        canvas1.figure = fig1
        canvas1.draw()

        # Gráfico de Nivel de Servicio
        fig2, ax2 = plt.subplots(figsize=(6, 5))  # Ajustar proporciones: menos ancho y menos alto
        nivel_actual = (datos['Cantidad Entregada'] / datos['Cantidad Pedida']) * 100  # Calcular nivel de servicio actual
        ax2.plot(datos['Fecha Formateada'], nivel_actual, label="Actual", color="blue")
        ax2.plot(datos['Fecha Formateada'], datos['Predicción Nivel de Servicio'], label="Predicción", linestyle="--", color="orange")
        ax2.set_title("Nivel de Servicio: Actual vs Predicción")
        ax2.set_xlabel("Fecha")
        ax2.set_ylabel("Porcentaje")
        ax2.legend(loc="upper right")  # Posición ajustada de la leyenda

        # Ajustar densidad de etiquetas en el eje X
        ax2.set_xticks(datos['Fecha Formateada'][::step])
        ax2.tick_params(axis='x', rotation=45, labelsize=8)  # Rotar y reducir tamaño de las etiquetas
        fig2.tight_layout()

        canvas2.figure = fig2
        canvas2.draw()

    except Exception as e:
        print(f"Error en graficar_comparativo: {e}")


def llenar_tabla(tabla, datos):
    """
    Llena la tabla con los datos procesados.
    """
    for row in tabla.get_children():
        tabla.delete(row)

    for _, row in datos.iterrows():
        tabla.insert("", "end", values=(
            row['Producto'],
            f"{row['Predicción Stock de Seguridad']:.2f}",
            f"{row['Predicción Nivel de Servicio']:.2f}%",
            row['Fecha de Pedido'].strftime('%Y-%m-%d'),
            row['Clasificación Nivel de Servicio']
        ))

def actualizar_tabla(tabla, datos):
    """
    Actualiza la tabla con los datos procesados.
    """
    # Limpia la tabla existente
    for item in tabla.get_children():
        tabla.delete(item)

    # Validar datos
    if 'Producto' not in datos.columns or 'Predicción Stock de Seguridad' not in datos.columns:
        print("Error: Las columnas necesarias no están disponibles en los datos.")
        return

    # Llenar la tabla con los datos
    for _, row in datos.iterrows():
        try:
            tabla.insert("", "end", values=(
                row['Producto'],
                row['Predicción Stock de Seguridad'],
                f"{row['Predicción Nivel de Servicio']:.2f}%",
                row['Fecha Límite Pedido'].strftime("%Y-%m-%d"),
                row['Clasificación Nivel de Servicio']
            ))
        except Exception as e:
            print(f"Error al agregar fila a la tabla: {e}")

def seleccionar_archivo(entry, canvas1, canvas2):
    """
    Permite al usuario seleccionar un archivo y ejecuta las predicciones.
    """
    archivo = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
    if archivo:
        entry.delete(0, tk.END)
        entry.insert(0, archivo)
        realizar_predicciones(archivo, canvas1, canvas2)
    else:
        print("No se seleccionó ningún archivo.")

def exportar_datos():
    """
    Exporta los datos filtrados en un archivo CSV.
    """
    global datos_filtrados_global
    if datos_filtrados_global.empty:
        print("No hay datos filtrados para exportar.")
        return

    archivo = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("Archivos CSV", "*.csv")])
    if archivo:
        datos_filtrados_global.to_csv(archivo, index=False)
        print(f"Datos exportados exitosamente a {archivo}")

def aplicar_y_actualizar(archivo, producto, proveedor, canvas1, canvas2, tabla):
    """
    Aplica los filtros seleccionados, genera las predicciones si es necesario,
    y actualiza las gráficas y la tabla de resultados.
    """
    global datos_filtrados_global

    try:
        # Verificar si se cargó un archivo
        if not archivo:
            raise ValueError("No se ha seleccionado ningún archivo.")

        # Cargar los datos desde el archivo
        datos = pd.read_csv(archivo)

        # Verificar si las columnas necesarias para predicciones están presentes
        columnas_requeridas = ['Producto', 'Cantidad Pedida', 'Cantidad Entregada',
                               'Fecha de Pedido', 'Fecha de Recepción', 'Stock Actual', 'Proveedor']
        for col in columnas_requeridas:
            if col not in datos.columns:
                raise ValueError(f"El archivo cargado no contiene la columna requerida: {col}")

        # Generar predicciones si no existen
        if 'Predicción Stock de Seguridad' not in datos.columns or 'Predicción Nivel de Servicio' not in datos.columns:
            datos_stock, datos_servicio = preprocesar_datos_combinados(datos)
            datos['Predicción Stock de Seguridad'] = modelo_stock.predict(datos_stock)
            datos['Predicción Nivel de Servicio'] = modelo_servicio.predict(datos_servicio)

        # Filtrar los datos según los criterios seleccionados
        datos_filtrados = aplicar_filtros(datos, producto, proveedor)

        # Verificar si hay datos después de aplicar los filtros
        if datos_filtrados.empty:
            print("No hay datos que coincidan con los filtros seleccionados.")
            datos_filtrados_global = pd.DataFrame()  # Resetear los datos filtrados
            actualizar_tabla(tabla, datos_filtrados_global)
            return

        # Guardar los datos filtrados para exportar
        datos_filtrados_global = datos_filtrados

        # Actualizar gráficas
        actualizar_graficas(canvas1, canvas2, datos_filtrados)

        # Actualizar la tabla
        actualizar_tabla(tabla, datos_filtrados)

    except ValueError as ve:
        print(f"Error: {ve}")
    except FileNotFoundError:
        print("Error: El archivo especificado no existe.")
    except Exception as e:
        print(f"Error inesperado: {e}")

def crear_interfaz():
    """
    Crea la interfaz gráfica con filtros, gráficas y resultados.
    """
    root = tk.Tk()
    root.title("Predicción de Indicadores Clave")
    root.geometry("1400x850")  # Ajuste de altura para dar más espacio a las gráficas

    # Espacio para el banner de la empresa
    banner_frame = tk.Frame(root, bg="black")
    banner_frame.pack(fill="x", pady=5)

    # Logo de la empresa
    logo_image = Image.open("1d.png")  # Asegúrate de que el archivo esté en el mismo directorio
    logo_image = logo_image.resize((1400, 80), Image.LANCZOS)  # Ajustar tamaño completo
    logo = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(banner_frame, image=logo, bg="black")
    logo_label.image = logo  # Guardar referencia
    logo_label.pack(side="left", fill="both", expand=True)

    # Título del banner
    banner_label = tk.Label(banner_frame, text="Predicción de Indicadores Clave", bg="black", fg="white", font=("Arial", 16))
    banner_label.pack(side="left", padx=10)

    # Selección de archivo
    archivo_frame = tk.Frame(root)
    archivo_frame.pack(pady=10)

    tk.Label(archivo_frame, text="Cargar archivo (Excel o CSV):").pack(side="left", padx=5)
    archivo_entry = tk.Entry(archivo_frame, width=50)
    archivo_entry.pack(side="left", padx=5)

    def seleccionar_archivo():
        archivo = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv"), ("Archivos Excel", "*.xlsx")])
        if archivo:
            archivo_entry.delete(0, tk.END)
            archivo_entry.insert(0, archivo)

    archivo_boton = tk.Button(archivo_frame, text="Seleccionar Archivo", command=seleccionar_archivo)
    archivo_boton.pack(side="left", padx=5)

    # Filtro de producto
    filtros_frame = tk.Frame(root)
    filtros_frame.pack(pady=10)

    tk.Label(filtros_frame, text="Producto:").grid(row=0, column=0, padx=5)
    producto_combo = ttk.Combobox(filtros_frame, values=["Todos", "Caoba", "Cedro", "Tornillo", "Nogal", "Abeto", "Roble", "Arce", "Pino"], width=20)
    producto_combo.grid(row=0, column=1, padx=5)
    producto_combo.set("Todos")

    # Combobox para seleccionar tipo de gráfico
    tk.Label(filtros_frame, text="Visualización:").grid(row=0, column=2, padx=5)
    modo_combo = ttk.Combobox(filtros_frame, values=["Individual", "Comparativo"], width=20)
    modo_combo.grid(row=0, column=3, padx=5)
    modo_combo.set("Individual")

    # Botón para aplicar predicciones
    aplicar_boton = tk.Button(
        filtros_frame,
        text="Aplicar Predicciones",
        command=lambda: actualizar_graficas_principal(
            archivo_entry.get(),
            producto_combo.get(),
            modo_combo.get(),
            canvas1,
            canvas2,
            canvas_proveedor_producto,
            tabla
        ),
    )
    aplicar_boton.grid(row=0, column=4, padx=5)

    # Gráficas
    graficas_frame = tk.Frame(root)
    graficas_frame.pack(fill="both", expand=True, pady=10)

    canvas1 = FigureCanvasTkAgg(plt.figure(), master=graficas_frame)
    canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

    canvas2 = FigureCanvasTkAgg(plt.figure(), master=graficas_frame)
    canvas2.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

    canvas_proveedor_producto = FigureCanvasTkAgg(plt.figure(), master=graficas_frame)
    canvas_proveedor_producto.get_tk_widget().grid(row=0, column=2, padx=10, pady=10, rowspan=2)

    # Tabla de resultados
    tabla_frame = tk.Frame(root)
    tabla_frame.pack(fill="x", pady=5)

    columnas = ["Producto", "Predicción Stock de Seguridad", "Predicción Nivel de Servicio",
                "Fecha Límite Pedido", "Clasificación Nivel de Servicio"]
    tabla = ttk.Treeview(tabla_frame, columns=columnas, show="headings")
    for col in columnas:
        tabla.heading(col, text=col)
        tabla.column(col, width=150)
    tabla.pack(fill="x", padx=10, pady=5)

    root.mainloop()

# Ejecutar la interfaz
if __name__ == "__main__":
    crear_interfaz()
