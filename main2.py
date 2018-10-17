# Libreria para plotear graficos
import matplotlib.pyplot as plt
# Librerias para importar el dataset, las maquians de soporte vectorial y las metricas para los reportes.
from sklearn import datasets, svm, metrics
import seaborn as sns; sns.set()

import seaborn as sn
import pandas as pd


# Declaramos una variable numero, donde se almanecera la informacion del dataset
numeros = datasets.load_digits()


images_labels = list(zip(numeros.images, numeros.target)) #Creamos una lista la cual va contener la imagen y su respectiva etiqueta

for index, (image, label) in enumerate(images_labels[:4]):
    """
    Recorremos la lista creada anteriormente, y ploteamos la imagen con su respectiva etiqueta utilizando interpolation = 'nearest'
    simplemente muestra una imagen sin intentar interpolar entre píxeles si la resolución de la pantalla no es la misma que la resolución
    de la imagen (lo que suele ser el caso). El resultado será una imagen en la que los píxeles se muestran como un cuadrado de varios píxeles.
    
    Y realizamos un ploteo de la informacion de entrenamiento en la primera mitad del plot
    """
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Train: %i' % label)


"""
Procedemos a convertir los datos en una matriz, donde estaran (muestra, caracteristicas) de cada una de las imagenes, para el quel clasificador pueda aprender de las imagenes
Ya que todo clasificador aprende de numeros.
"""
n_samples = len(numeros.images)
data = numeros.images.reshape((n_samples, -1))

"""
Definimos nuestro clasificador de Maquinas de Soporte Vectorial, con un gamma de 0.001 y nuestra variable C, donde C es el parámetro para la función de coste de margen suave,
que controla la influencia de cada vector de soporte individual; Este proceso implica el comercio de la pena de error para la estabilidad.
"""
classifier = svm.SVC(C=100, gamma=0.001)

# Tomamos la mitad de los datos para entrenar nuestro modelo
classifier.fit(data[:n_samples // 2], numeros.target[:n_samples // 2])

# Ahora predice el valor del dígito y procedemos a pintarlo en la segunda mitad del plot
expected = numeros.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Informe de clasificación para clasificador %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Matriz de confuccion:\n%s" % metrics.confusion_matrix(expected, predicted))

"""
definimos parametros para nuestra matriz de confusion
"""
cnf_matrix = metrics.confusion_matrix(expected, predicted)
class_names = numeros.target

"""
graficamos nuestra matriz de confusion
"""
df_cm = pd.DataFrame(cnf_matrix, range(10),
                  range(10))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size

"""
procedemos a pintar en el plot nuestros resultados de predicion
"""

images_and_predictions = list(zip(numeros.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediccion: %i' % prediction)

plt.show()

"""
Luego de realizar el proceso de entrenamiento, de validacion cruzada, y clasificacion, procedemos a verificar cual es nuestro precision_score dando como resultado el 97% de presicion
"""
print("Score: %",metrics.accuracy_score(expected, predicted))