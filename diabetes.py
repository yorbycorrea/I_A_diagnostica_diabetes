#Importamos librerias necesarias
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_diabetes # Esto es una base de datos y aquí podemos encontrar todas las variables necesarias
from sklearn.model_selection import train_test_split
from sklearn import metrics

#datos de entrenamiento
diabetes = load_diabetes()

#separamos las variables de entrada (x) de la variable objetivo (y)
X = diabetes.data
y = diabetes.target

#dividimos el conjunto de entrenamiento y el conjunto de prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#clasificamos el arbol de decisiones y entramos los datos
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)

#pedimos al usuario los datos del paciente
nombre = input("Ingresa el nombre del paciente: ")
edad = int(input("Ingresa la edad del paciente: "))
masa_corporal = float(input("Ingresa la masa corporal del paciente: "))
presion_sanguinea = float(input("Ingresa la presión sanguínea del paciente: "))
colesterol = float(input("Ingresa el nivel de colesterol del paciente: "))
nivel_azucar = float(input("Ingresa el nivel de azúcar del paciente: "))
nivel_trigliceridos = float(input("Ingresa el nivel de triglicéridos del paciente: "))

#realizamos la predicción para los datos ingresados por el usuario
prediccion = clf.predict([[nivel_azucar, nivel_trigliceridos, masa_corporal, presion_sanguinea, colesterol, 0, 0, 0, 0, 0]])

#Mostamos el resultado de la prediccion al usuario
if prediccion == 0:
   print(f"El paciente {nombre} no tiene diabetes.")

else:
   print(f"El paciente {nombre} tiene diabetes.")






#dentro de los datos que nos pediras son: ID, nivel de azucar, nivel de trigliceridos, .