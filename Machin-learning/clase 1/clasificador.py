from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
# Cargar el conjunto de datos Iris
iris = load_iris()#carga de datos
# Crear un DataFrame a partir de los datos
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)#me devuelve la ldata y le asigna nombre
#df['especie'] = iris.target
df['especie'] = [iris.target_names[i] for i in iris.target]
print(df.head())
#SEPARAR x(PREGUNTAS) E Y(ETIQUETAS)
X = iris.data
y = iris.target
#paso 3 dividir para entrenamiento y examen 
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#paso 4 crear el modelo y entrenarlo
modelo = KNeighborsClassifier(n_neighbors=3) 
modelo.fit(x_train, y_train)

#paso 5 hacer predicciones
prediciones=modelo.predict(x_test)
precision=accuracy_score(y_test, prediciones)
print(f"Precisión del modelo: {precision*100:.2f}%")
#inventamos una Flor:[sepalo largo, sepalo ancho, petalo largo, petalo ancho]
mi_flor=[[5.1, 3.5, 1.4, 0.2]]
prediccion_flor=modelo.predict(mi_flor)
nombre_flor=iris.target_names[prediccion_flor][0]
print(f"La flor es: {prediccion_flor[0]}")#me devuelve el numero de la flor
print(f'la maquina dice que es una : {nombre_flor}')
#