import cv2 
import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import sys
from sklearn.preprocessing import MinMaxScaler

##################################### Parámetros ############################################
BASE_DATOS_PARAM = "Outex_TC_00013"
if len(sys.argv) > 1:
  BASE_DATOS_PARAM = sys.argv[1] # nombre de la base de datos a usar

# 0: grayscale
# 1: rgb marginal
# 2: rgb vectorial
COVARIANCE_TYPE = 1
if len(sys.argv) > 2:
  COVARIANCE_TYPE = int(sys.argv[2]) 

CLASIFICADOR = "svc"
if len(sys.argv) > 3:
  CLASIFICADOR = sys.argv[3] # clasficador a usar ("knn", "svc" o "ovsr")


####################################### constantes #############################################
PATH_BASE = "../../databases/{}/".format(BASE_DATOS_PARAM)
CSV_PRUEBA = "../../resultados/covarianza/vector_prueba_{}.csv".format(BASE_DATOS_PARAM)
CSV_ENTRENAMIENTO =  "../../resultados/covarianza/vector_entrenamiento_{}.csv".format(BASE_DATOS_PARAM)
CSV_EXACTITUD = "../../resultados/covarianza/exactitud_{}.csv".format(BASE_DATOS_PARAM)

DISTANCES = [1, 5, 9, 13]
#DISTANCES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#DISTANCES = [1]
ANGLES = [0, 45, 90, 135]

MEASURE = "VOLUME"
#MEASURE = "ENERGY"

def leer_imagenes(txt): # txt: archivo.txt
  labels = np.array([])

  # Lectura de nombres de archivos de entrenamiento
  path_base = PATH_BASE
  f = open(path_base + "000/" + txt,"r")
  lineas = f.readlines()

  imagenes = []
  for i in range(1, len(lineas)):
    nombreArchivo = lineas[i].split()[0]
    labels = np.append(labels, lineas[i].split()[1])
    path = path_base + "images/" + nombreArchivo
    img = cv2.imread(path)
    if COVARIANCE_TYPE == 0:
      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      imagenes.append(gray_img)
    else:
      imagenes.append(img)
  return imagenes, labels # retorna un array de imágenes y un array de nombre de clases

def volumen(img):
  return np.sum(img)

def energia(img):
  nrg = 0
  width, height = img.shape
  
  for v in range(height):
    for u in range(width):
      nrg += img[u][v]**2
    
  return nrg

def get_structuring_element(distance, angle):  
  if angle == 0:
    point_2 = [distance*1, 0]
  elif angle == 45:
    point_2 = [distance*1, distance*1]
  elif angle == 90:
    point_2 = [0, distance*1]
  elif angle == 135:
    point_2 = [distance*(-1), distance*1]

  se = { 'p1': [0, 0], 'p2': point_2 }

  return se
  
def get_list_structuring_element(distances, angles):
  list_se = []
  for d in distances:
    for a in angles:
      list_se.append(get_structuring_element(d,a))

  return list_se

def mininum_single_channel(image, se): #se: elemento estructural
  width, height = image.shape

  img_processed = image

  for v in range(height):
    for u in range(width):
      pval_min = 255

      for p in se:
        px = se[p][0] + u
        py = se[p][1] + v

        if px >= width or py >= height or px < 0 or py < 0:
          continue
        
        pval = img_processed[px][py]
                    
        #busqueda del minimo  
        if pval_min > pval:
          pval_min = pval	
                    
      img_processed[u][v] = pval_min
          
  return img_processed
  
def mininum_single_channel_vectorial(image, se): #se: elemento estructural
  width, height, channels = image.shape

  img_processed = image

  for v in range(height):
    for u in range(width):
      distance_min = distance.euclidean((255, 255, 255), (0, 0, 0))
      concrete_min = 0

      for p in se:
        px = se[p][0] + u
        py = se[p][1] + v

        if px >= width or py >= height or px < 0 or py < 0:
          continue
          
        pval = image[px][py]

        distance_val = distance.euclidean(pval, (0, 0, 0))

                      
        #busqueda del minimo  
        if distance_min > distance_val:
          distance_min = distance_val
          concrete_min = pval;	

                      
      img_processed[u][v] = concrete_min
          
  return img_processed

def covariance_grayscale(imagenes, list_of_se, medida):
  features = []

  for i in range(len(imagenes)):
    image = imagenes[i]
    vol_img_original = medida(image)

    res_covarianza = []

    for se in list_of_se:
      erosion = mininum_single_channel(image, se)
      vol_img_closing = medida(erosion)
      division_volumenes = vol_img_closing / vol_img_original
      res_covarianza.append(division_volumenes)
    features.append(res_covarianza)

  return features

def covariance_marginal(imagenes, list_of_se, medida):
  features = []
  
  for i in range(len(imagenes)):
    image = imagenes[i]
    try:
      b,g,r = cv2.split(image)

      vol_img_original = [medida(r), medida(g), medida(b)]

      res_covarianza = []
      for se in list_of_se:
        for index, channel in enumerate([r,g,b]):
          erosion = mininum_single_channel(channel, se)
          vol_img_closing = medida(erosion)
          division_volumenes = vol_img_closing / vol_img_original[index]
          res_covarianza.append(division_volumenes)
      features.append(res_covarianza)
    except Exception as e:
      print(e)
  return features

def covariance_vectorial(imagenes, list_of_se, medida):
  features = []
  
  for i in range(len(imagenes)):
    image = imagenes[i]
    try:
      b,g,r = cv2.split(image)

      vol_img_original = [medida(r), medida(g), medida(b)]

      res_covarianza = []
      for se in list_of_se:
        erosion = mininum_single_channel_vectorial(image, se)
        erosion_b, erosion_g, erosion_r = cv2.split(erosion)
        erosions = [erosion_r, erosion_g, erosion_b]
        for index, channel in enumerate([r,g,b]):
          vol_img_closing = medida(erosions[index])
          division_volumenes = vol_img_closing / vol_img_original[index]
          res_covarianza.append(division_volumenes)
      features.append(res_covarianza)
    except Exception as e:
      print(e)

  return features

def entrenamiento(imagenes_entrenamiento, labels_entrenamiento, list_of_se):
  medida = volumen
  if MEASURE == "ENERGY":
    medida = energia
  if COVARIANCE_TYPE == 0:
    features_imagenes = covariance_grayscale(imagenes_entrenamiento, list_of_se, medida)
  elif COVARIANCE_TYPE == 1:
    features_imagenes = covariance_marginal(imagenes_entrenamiento, list_of_se, medida)
  elif COVARIANCE_TYPE == 2:
    features_imagenes = covariance_vectorial(imagenes_prueba, list_of_se, medida)

  # creación archivo de vector de caracteristicas
  len_hist_entrenamiento = range(len(features_imagenes))
  with open(CSV_ENTRENAMIENTO,'w') as f2:
    w = csv.writer(f2)
    titulos = ["CLASE"] + list(len_hist_entrenamiento)
    w.writerow(titulos)
    for i in len_hist_entrenamiento:
      fila = []
      fila.append(labels_entrenamiento[i])
      fila = fila + features_imagenes[i]
      w.writerow(fila)

  if CLASIFICADOR == "svc":
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
    classifier = MLPClassifier(random_state=0, max_iter=1000, learning_rate = 'adaptive', learning_rate_init = 1)
     
  # Entrenar clasificador
  classifier.fit(features_imagenes, labels_entrenamiento)
  
  return classifier

def prueba(imagenes_prueba, labels_prueba, classifier, list_of_se):
  medida = volumen
  if MEASURE == "ENERGY":
    medida = energia
  # Extracción de features de las imágenes de prueba
  if COVARIANCE_TYPE == 0:
    features_imagenes = covariance_grayscale(imagenes_prueba, list_of_se, medida)
  elif COVARIANCE_TYPE == 1:
    features_imagenes = covariance_marginal(imagenes_prueba, list_of_se, medida)
  elif COVARIANCE_TYPE == 2:
    features_imagenes = covariance_vectorial(imagenes_prueba, list_of_se, medida)
  
  # creación archivo de vector de caracteristicas
  len_hist_prueba = range(len(features_imagenes))
  with open(CSV_PRUEBA,'w') as f2:
    w = csv.writer(f2)
    titulos = ["CLASE"] + list(len_hist_prueba)
    w.writerow(titulos)
    for i in len_hist_prueba:
      fila = []
      fila.append(labels_prueba[i])
      fila = fila + features_imagenes[i]
      w.writerow(fila)

  # Predicciones
  predicciones = classifier.predict(features_imagenes)

  # Resultados
  exactitud = accuracy_score(labels_prueba, predicciones)

  print ('Exactitud: %0.3f' % exactitud)
  return exactitud

def main(imagenes_entrenamiento, labels_entrenamiento, imagenes_prueba, labels_prueba):
  list_of_structuring_elements = get_list_structuring_element(DISTANCES, ANGLES)

  print("Entrenamiento...")
  classifier = entrenamiento(imagenes_entrenamiento, labels_entrenamiento, list_of_structuring_elements)

  print("Prueba...")
  exactitud = prueba(imagenes_prueba, labels_prueba, classifier, list_of_structuring_elements)
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
  titulos = ["Descriptor", "Escala", "Base de datos", "Clasificador", "Exactitud", "distances", "max_iter"]
  w.writerow(titulos)
  w.writerow(["Covarianza", COVARIANCE_TYPE, BASE_DATOS_PARAM, CLASIFICADOR, exactitud, DISTANCES, 1000])

