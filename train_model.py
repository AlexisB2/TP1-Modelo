import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Cargar los datos con el delimitador correcto y la codificación 'latin1'
data = pd.read_csv('endometriosis_data.csv', delimiter=';', encoding='latin1')

# Convertir la columna 'Intensidad de dolor' en variables dummy
data = pd.get_dummies(data, columns=['Intensidad de dolor'])

# Definir las características (features) y la etiqueta (label)
features = ['Edad', 'Duración ciclo menstrual', 'Alargue de duración de ciclo menstrual',
            'Aumento de sangrado', 'Dolor durante relaciones sexuales',
            'Parientes cercanos con endometriosis', 'Dificultad para embarazo',
            'Intensidad de dolor_intenso', 'Intensidad de dolor_leve',
            'Intensidad de dolor_moderado', 'Intensidad de dolor_ninguno']

X = data[features]
y = data['Nivel de riesgo']

# Dividir los datos en conjunto de entrenamiento y prueba usando StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Guardar los nombres de las columnas del conjunto de entrenamiento
columns = X_train.columns

# Verificar la distribución de las etiquetas
print("Distribución de etiquetas en el conjunto de entrenamiento:")
print(y_train.value_counts())

print("Distribución de etiquetas en el conjunto de prueba:")
print(y_test.value_counts())

# Crear y entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo entrenado y las columnas del conjunto de entrenamiento
joblib.dump(model, 'endometriosis_model.pkl')
joblib.dump(columns, 'endometriosis_columns.pkl')

# Evaluar el modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
