import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

estudiantes = {
    "horas_estudio": [2, 4, 5, 6, 8, 10, 3, 7, 9, 1],
    "asistencia": [60, 70, 75, 80, 90, 95, 65, 85, 92, 50],
    "aprobo": [0, 0, 1, 1, 1, 1, 0, 1, 1, 0],
}

df = pd.DataFrame(estudiantes)
print("Estadísticas:\n", df.describe())
df["promedio_semanal"] = df["horas_estudio"] / 7
print("\nDataFrame con promedio semanal:\n", df)

aprobados_menos_de_6_horas = df[
    (df["aprobo"] == 1) & (df["horas_estudio"] < 6)
]
print(
    "\nEstudiantes que aprobaron con menos de 6 horas de estudio:\n",
    aprobados_menos_de_6_horas,
)

promedio_asistencia = np.mean(df["asistencia"])
desviacion_asistencia = np.std(df["asistencia"])
print(f"\nPromedio de asistencia: {promedio_asistencia:.2f}%")
print(f"Desviación estándar de asistencia: {desviacion_asistencia:.2f}")

X = df[["horas_estudio", "asistencia"]]
y = df["aprobo"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
modelo = LogisticRegression()
modelo.fit(X_train, y_train)
precision = accuracy_score(y_test, modelo.predict(X_test))
print(f"\nPrecisión del modelo: {precision * 100:.1f}%")

nuevo_aprendiz = pd.DataFrame({"horas_estudio": [5], "asistencia": [78]})
for prediccion in modelo.predict(nuevo_aprendiz):
    print("Aprobará" if prediccion == 1 else "No aprobará")