import cv2
import csv
import sys
import numpy as np
from math import ceil
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

##################################### Parámetros ############################################
BASE_DATOS_PARAM = "Outex_TC_00013"
if len(sys.argv) > 1:
  BASE_DATOS_PARAM = sys.argv[1] # nombre de la base de datos a usar

CLASIFICADOR = "svc"
if len(sys.argv) > 2:
  CLASIFICADOR = sys.argv[2] # clasficador a usar ("knn", "svc", "ovsr", "rf")


####################################### Constantes #############################################
PATH_BASE = "../../databases/{}/".format(BASE_DATOS_PARAM)
CSV_PRUEBA = "../../resultados/color_layout_descriptor/vector_prueba_{}.csv".format(BASE_DATOS_PARAM)
CSV_ENTRENAMIENTO =  "../../resultados/color_layout_descriptor/vector_entrenamiento_{}.csv".format(BASE_DATOS_PARAM)
CSV_EXACTITUD = "../../resultados/color_layout_descriptor/exactitud_{}.csv".format(BASE_DATOS_PARAM)
ROWS = 8
COLUMNS = 8

###################################### Funciones ###############################################
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
    imagenes.append(img)
  return imagenes, labels # retorna un array de imágenes y un array de nombre de clases

def findCLD(img):
	averages = np.zeros((ROWS, COLUMNS, 3)) #(8, 8, 3)
	M, N, _ = img.shape
	
	for row in range(ROWS):
		for col in range(COLUMNS):
			"""
				Step 1 - Image partitioning: Input picture [M x N]. Input picture divided into 64 blocks [M/8xN/8]
				slice_ represents one of the 64 blocks in each iteration.
    	"""
			slice_ = img[ceil(M/ROWS) * row: ceil(M/ROWS) * (row+1), ceil(N/COLUMNS)*col : ceil(N/COLUMNS)*(col+1)]
			"""
				Step 2 - Representative color selection: After the image partitioning stage, a single representative 
				color is selected from each block. The average of the pixel colors in a block is used as the corresponding 
				representative color. The selection results in a tiny image icon of size 8x8.
  		"""
			average_color = np.mean(np.mean(slice_, axis=0), axis=0) # the mean of each block
			#the image of size 8x8 where each pixel is the mean.
			averages[row][col][0] = average_color[0]
			averages[row][col][1] = average_color[1]
			averages[row][col][2] = average_color[2]
  # Color space conversion between RGB and YCbCr is applied. rgb to YCbCr.
	icon = cv2.cvtColor(np.array(averages, dtype=np.uint8), cv2.COLOR_BGR2YCR_CB)
	y, cr, cb = cv2.split(icon)
	"""
		Step 3 - DCT transformation: In the fourth stage, the luminance (Y) and the blue and red chrominance (Cb and Cr)
		are transformed by 8x8 DCT, so three sets of 64 DCT coefficients are obtained. 
	"""
	dct_y = cv2.dct(np.float32(y))
	dct_cb = cv2.dct(np.float32(cb))
	dct_cr = cv2.dct(np.float32(cr))
	"""
		Step 4 - Zigzag scanning: A zigzag scanning is performed with these three sets of 64 DCT coefficients.
  	3 zigzag scanned matrix (DY, DCb, DCr).
  """
	dct_y_zigzag = []
	dct_cb_zigzag = []
	dct_cr_zigzag = []
	flip = True
	flipped_dct_y = np.fliplr(dct_y)
	flipped_dct_cb = np.fliplr(dct_cb)
	flipped_dct_cr = np.fliplr(dct_cr)
	for i in range(ROWS + COLUMNS -1):
		k_diag = ROWS - 1 - i
		diag_y = np.diag(flipped_dct_y, k=k_diag)
		diag_cb = np.diag(flipped_dct_cb, k=k_diag)
		diag_cr = np.diag(flipped_dct_cr, k=k_diag)
		if flip:
			diag_y = diag_y[::-1]
			diag_cb = diag_cb[::-1]
			diag_cr = diag_cr[::-1]
		dct_y_zigzag.append(diag_y)
		dct_cb_zigzag.append(diag_cb)
		dct_cr_zigzag.append(diag_cr)
		flip = not flip

	return np.concatenate([np.concatenate(dct_y_zigzag), np.concatenate(dct_cb_zigzag), np.concatenate(dct_cr_zigzag)])

def extraer_CLD(images):
  features = []
  
  for img in images:
    features.append(findCLD(img)) 
  
  return features

def entrenamiento(imagenes_entrenamiento, labels_entrenamiento):
  # Extracción de features de las imágenes de entrenamiento
  features_imagenes = extraer_CLD(imagenes_entrenamiento)

  # creación archivo de vector de caracteristicas
  len_hist_entrenamiento = range(len(features_imagenes))
  with open(CSV_ENTRENAMIENTO,'w') as f2:
    w = csv.writer(f2)
    titulos = ["CLASE"] + list(range(192))
    w.writerow(titulos)
    for i in len_hist_entrenamiento:
      fila = []
      fila.append(labels_entrenamiento[i])
      fila = fila + features_imagenes[i].tolist()
      w.writerow(fila)

  if CLASIFICADOR == "svc":
    classifier = SVC(kernel = 'linear')
  elif CLASIFICADOR == "knn":
    classifier = KNeighborsClassifier(n_neighbors = 1)
  elif CLASIFICADOR == "ovsr":
    classifier = OneVsRestClassifier(SVC(kernel = 'linear'))
  elif CLASIFICADOR == "rf":
    classifier = RandomForestClassifier(random_state = 0)
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
  features_imagenes = extraer_CLD(imagenes_prueba)

  # creación archivo de vector de caracteristicas
  len_hist_prueba = range(len(features_imagenes))
  with open(CSV_PRUEBA,'w') as f2:
    w = csv.writer(f2)
    titulos = ["CLASE"] + list(range(192))
    w.writerow(titulos)
    for i in len_hist_prueba:
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
  w.writerow(["Color Layout Descriptor", "RGB", BASE_DATOS_PARAM, CLASIFICADOR, exactitud])



