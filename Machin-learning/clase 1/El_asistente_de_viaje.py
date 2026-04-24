import pandas as pd
from sklearn.linear_model import    LinearRegression
import matplotlib.pyplot as plt 
#variable independiente = kilometros
#variable dependiente = litros gastados
#cargar data:
data = { "kilometros": [10, 50, 100, 200, 350, 500],
"litros_gastados": [1.2, 5.5, 10.2, 21.0, 36.5, 52.0]}
df = pd.DataFrame(data)
print(df)

#separar x e y
x = df[["kilometros"]]
y = df["litros_gastados"]
#crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(x, y)
#realizar predicciones
x_input = float(input("Ingrese la cantidad de kilómetros que desea recorrer: "))
predicciones = modelo.predict([[x_input]])  # El doble corchete es importante porque la máquina lo recibe como un DataFrame
print(f"Los litros estimados para recorrer {x_input} kilómetros son: {predicciones[0]:.2f} litros")
print(f"Los litros estimados para recorrer {x_input *1000} mil centimetros es de  son: {predicciones[0]:.2f} litros")
#analisis

print("el analisis teorico corresponde a la formula de regresion lineal ")


n = 5
for i in range(1, n + 1):
    espacios = ' ' * (n - i)
    asteriscos = '*' * (2 * i - 1)
    print(espacios + asteriscos)
print(" "*4 +( "|"))
print(" "*4 +( "|"))
#visualizar resultados
plt.scatter(x, y)
plt.plot(x, modelo.predict(x), color='red')
plt.xlabel('Kilómetros')
plt.ylabel('Litros Gastados')
plt.title('Regresión Lineal: Litros Gastados vs Kilómetros')
plt.show()