import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import joblib
import pandas as pd

# Cargar el modelo entrenado y las columnas del conjunto de entrenamiento
model = joblib.load('endometriosis_model.pkl')
columns = joblib.load('endometriosis_columns.pkl')


# Definir la función de predicción
def predecir_riesgo(paciente):
    # Convertir los datos del paciente en un DataFrame
    paciente_df = pd.DataFrame([paciente])

    # Convertir características categóricas a numéricas (dummy variables)
    paciente_df = pd.get_dummies(paciente_df, columns=['Intensidad de dolor'])

    # Asegurarse de que tenga las mismas columnas que los datos de entrenamiento
    for col in columns:
        if col not in paciente_df.columns:
            paciente_df[col] = 0

    # Ordenar las columnas para que coincidan con las del conjunto de entrenamiento
    paciente_df = paciente_df[columns]

    # Predecir el riesgo
    prediccion = model.predict(paciente_df)

    # Obtener la probabilidad de la predicción
    probabilidad = model.predict_proba(paciente_df).max()

    # Definir los rangos de riesgo
    if probabilidad <= 0.2:
        nivel_riesgo = "0%-20%: nivel muy bajo"
    elif probabilidad <= 0.4:
        nivel_riesgo = "20%-40%: nivel bajo"
    elif probabilidad <= 0.6:
        nivel_riesgo = "40%-60%: nivel intermedio"
    elif probabilidad <= 0.8:
        nivel_riesgo = "60%-80%: nivel alto"
    else:
        nivel_riesgo = "80%-100%: nivel muy alto"

    return nivel_riesgo, probabilidad


# Función para manejar la predicción cuando se presiona el botón
def realizar_prediccion():
    paciente = {
        'Edad': int(entry_edad.get()),
        'Duración ciclo menstrual': int(entry_duracion_ciclo.get()),
        'Alargue de duración de ciclo menstrual': int(entry_alargue_ciclo.get()),
        'Aumento de sangrado': bool(var_aumento_sangrado.get()),
        'Intensidad de dolor': combobox_dolor.get(),
        'Dolor durante relaciones sexuales': bool(var_dolor_relaciones.get()),
        'Parientes cercanos con endometriosis': bool(var_parientes.get()),
        'Dificultad para embarazo': bool(var_dificultad_embarazo.get())
    }

    nivel_riesgo, probabilidad = predecir_riesgo(paciente)
    messagebox.showinfo("Predicción de Riesgo", f"Nivel de riesgo: {nivel_riesgo}\nProbabilidad: {probabilidad:.2f}")


# Crear la ventana principal
root = tk.Tk()
root.title("Sistema de Detección de Endometriosis")

# Crear y colocar los widgets en la ventana
tk.Label(root, text="Nombres del paciente").grid(row=0, column=0, padx=10, pady=5)
entry_nombre = tk.Entry(root)
entry_nombre.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Edad").grid(row=1, column=0, padx=10, pady=5)
entry_edad = tk.Entry(root)
entry_edad.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Duración ciclo menstrual (días)").grid(row=2, column=0, padx=10, pady=5)
entry_duracion_ciclo = tk.Entry(root)
entry_duracion_ciclo.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Alargue de duración de ciclo menstrual (días)").grid(row=3, column=0, padx=10, pady=5)
entry_alargue_ciclo = tk.Entry(root)
entry_alargue_ciclo.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Aumento de sangrado").grid(row=4, column=0, padx=10, pady=5)
var_aumento_sangrado = tk.IntVar()
tk.Checkbutton(root, variable=var_aumento_sangrado).grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Intensidad de dolor").grid(row=5, column=0, padx=10, pady=5)
combobox_dolor = ttk.Combobox(root, values=["ninguno", "leve", "moderado", "intenso"])
combobox_dolor.grid(row=5, column=1, padx=10, pady=5)
combobox_dolor.current(0)

tk.Label(root, text="Dolor durante relaciones sexuales").grid(row=6, column=0, padx=10, pady=5)
var_dolor_relaciones = tk.IntVar()
tk.Checkbutton(root, variable=var_dolor_relaciones).grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Parientes cercanos con endometriosis").grid(row=7, column=0, padx=10, pady=5)
var_parientes = tk.IntVar()
tk.Checkbutton(root, variable=var_parientes).grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="Dificultad para embarazo").grid(row=8, column=0, padx=10, pady=5)
var_dificultad_embarazo = tk.IntVar()
tk.Checkbutton(root, variable=var_dificultad_embarazo).grid(row=8, column=1, padx=10, pady=5)

tk.Button(root, text="Realizar Predicción", command=realizar_prediccion).grid(row=9, column=0, columnspan=2, pady=10)

# Iniciar el bucle principal de la interfaz gráfica
root.mainloop()
