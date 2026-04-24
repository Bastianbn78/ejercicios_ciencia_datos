import pandas as pd
from sklearn.linear_model import    LinearRegression
import matplotlib.pyplot as plt 
#variable independiente = tamaño casa
#variable dependiente = precio casa
#cargar datos
datos={
    'tamaño_casa':[50, 60, 80, 100, 120,150],
    'precio_casa':[150, 180, 220, 260, 300,360]
}
df=pd.DataFrame(datos)
print(df)
#separar x e y
X=df[['tamaño_casa']]
y=df['precio_casa']
#crear Y entrenar el modelo

modelo=LinearRegression()
modelo.fit(X,y)
#realizar prediciones
predicciones=modelo.predict([[90]])#el doble corquete es importante por que la maquina lo recibe como data frame 
print(f"El precio estimado para una casa de 90 metros cuadrados es: {predicciones[0]:.2f}")


#visualizar resultados
plt.scatter(X,y)
plt.plot(X, modelo.predict(X), color='red')
plt.xlabel('Tamaño de la casa (m²)')
plt.ylabel('Precio de la casa (miles de dólares)')
plt.title('Regresión Lineal: Precio vs Tamaño de la Casa')
plt.show()

#saber el tamaño por el precio 