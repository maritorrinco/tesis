########## imports ##########
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import mahotas
import csv
import sys
from sklearn.preprocessing import MinMaxScaler

########## parámetros ##########
BASE_DATOS_PARAM = "Outex_TC_00013"
if len(sys.argv) > 1:
  BASE_DATOS_PARAM = sys.argv[1] # nombre de la base de datos a usar

CLASIFICADOR = "svm"
if len(sys.argv) > 2:
  CLASIFICADOR = sys.argv[2] # clasficador a usar ("knn", "svm", "ovsr", "rf")


########## constantes ##########
PATH_BASE = "../../databases/{}/".format(BASE_DATOS_PARAM)
CSV_PRUEBA = "../../resultados/haralick/vector_prueba_{}.csv".format(BASE_DATOS_PARAM)
CSV_ENTRENAMIENTO =  "../../resultados/haralick/vector_entrenamiento_{}.csv".format(BASE_DATOS_PARAM)
CSV_EXACTITUD = "../../resultados/haralick/exactitud_{}.csv".format(BASE_DATOS_PARAM)

########## Funciones ##########
def leer_imagenes(txt): # txt: archivo.txt
  labels = np.array([])

  # Lectura de nombres de archivos de entrenamiento
  f = open(PATH_BASE + "000/" + txt,"r")
  lineas = f.readlines()

  imagenes = []
  for i in range(1, len(lineas)):
    nombreArchivo = lineas[i].split()[0]
    labels = np.append(labels, lineas[i].split()[1])
    path = PATH_BASE + "images/" + nombreArchivo
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imagenes.append(gray_img)
  return imagenes, labels # retorna un array de imágenes y un array de nombre de clases

def extraer_haralick_features(imagenes):
  features_imagenes = []

  for imagen in imagenes:
    haralick = mahotas.features.haralick(imagen).mean(axis=0) # haralick devuelve un vector de 14 valores
    features_imagenes.append(haralick) 
  
  return features_imagenes

def entrenamiento(imagenes_entrenamiento, labels_entrenamiento):
  # Extracción de features de las imágenes de entrenamiento
  features_imagenes = extraer_haralick_features(imagenes_entrenamiento)

  # creación archivo de vector de caracteristicas
  with open(CSV_ENTRENAMIENTO,'w') as f2:
    w = csv.writer(f2)
    titulos = ["CLASE"] + list(range(13))
    w.writerow(titulos)
    for i in range(len(features_imagenes)):
      fila = []
      fila.append(labels_entrenamiento[i])
      fila = fila + features_imagenes[i].tolist()
      w.writerow(fila)

  if CLASIFICADOR == "svm":
    classifier = SVC(kernel = 'linear')
  elif CLASIFICADOR == "knn":
    classifier = KNeighborsClassifier(n_neighbors = 1)
  elif CLASIFICADOR == "ovsr":
    classifier = OneVsRestClassifier(SVC(kernel = 'linear'))
  elif CLASIFICADOR == "rf":
    classifier = RandomForestClassifier(random_state = 0)
  elif CLASIFICADOR == "nb":
    classifier = MultinomialNB()
  elif CLASIFICADOR == 'mnb':
    classifier = MultinomialNB()
  elif CLASIFICADOR == 'mnbs':
    scaler = MinMaxScaler()
    classifier = MultinomialNB()
    features_imagenes = scaler.fit_transform(features_imagenes)
  elif CLASIFICADOR == "mlp":
    classifier = MLPClassifier(random_state=0, max_iter=500)

  # Entrenar clasificador
  classifier.fit(features_imagenes, labels_entrenamiento)
    
  return classifier

def prueba(imagenes_prueba, labels_prueba, classifier):
  # Extracción de features de las imágenes de prueba
  features_imagenes = extraer_haralick_features(imagenes_prueba)

  # creación archivo de vector de caracteristicas
  with open(CSV_PRUEBA,'w') as f2:
    w = csv.writer(f2)
    titulos = ["CLASE"] + list(range(14))
    w.writerow(titulos)
    for i in range(len(features_imagenes)):
      fila = []
      fila.append(labels_prueba[i])
      fila = fila + features_imagenes[i].tolist()
      w.writerow(fila)

  # Predicciones utilizando SVM
  predicciones = classifier.predict(features_imagenes)

  # Resultados
  exactitud = accuracy_score(labels_prueba, predicciones)
  print ('Exactitud: %0.3f' % exactitud)
  return exactitud

def main(imagenes_entrenamiento, labels_entrenamiento, imagenes_prueba, labels_prueba):
  print("Entrenamiento...")
  classifier = entrenamiento(imagenes_entrenamiento, labels_entrenamiento)

  print("Prueba...")
  exactitud = prueba(imagenes_prueba, labels_prueba, classifier)
  return exactitud

#Leer imágenes de entrenamiento (una sola vez)
print("Lectura de imágenes de entrenamiento....")
imagenes_entrenamiento, labels_entrenamiento =  leer_imagenes("train.txt")
print("Cantidad de imágenes de entrenamiento:",  len(imagenes_entrenamiento))

# Lectura de imágenes de prueba (una sola vez)
print("Lectura de imágenes de prueba....")
imagenes_prueba, labels_prueba =  leer_imagenes("test.txt")
print("Cantidad de imágenes de prueba:",  len(imagenes_prueba))

exactitud = main(imagenes_entrenamiento, labels_entrenamiento, imagenes_prueba, labels_prueba)

print(exactitud)

with open(CSV_EXACTITUD,'a') as f2:
  w = csv.writer(f2)
  titulos = ["Descriptor", "Escala", "Base de datos", "Clasificador", "Exactitud"]
  w.writerow(titulos)
  w.writerow(["Haralick", "escala de grises", BASE_DATOS_PARAM, CLASIFICADOR, exactitud])