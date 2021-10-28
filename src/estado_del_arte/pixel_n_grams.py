"""
Texture Image Classification using Pixel N-grams
Estado del Arte
"""

"""
Cómo ejecutar desde /tesis/src:
python3 -u pixel_n_grams_final.py Outex_TC_00013 H 4 True svc npz |& tee -a ../resultados/pixel_n_grams/[bd]_[clasificador].txt &
python3 -u pixel_n_grams_final.py Brodatz_dividido H 4 True knn npz |& tee -a ../resultados/pixel_n_grams/[Brodatz_dividido]_[knn].txt &
"""

from skimage.util import view_as_windows
import cv2
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv
import sys
import scipy.sparse
from sklearn.preprocessing import minmax_scale


########################################## Parámetros ##########################################
print("\n\nGeneración de parámetros...")

BASE_DATOS_PARAM = "Outex_TC_00013"
if len(sys.argv) > 1: # Primer parámetro: Base de Datos
	BASE_DATOS_PARAM = sys.argv[1] # Nombre de la base de datos a usar
 
DIRECCION = "H"
if len(sys.argv) > 2: # Segundo parámetro: Dirección: H o V (Horizontal o Vertical)
	DIRECCION = sys.argv[2]
 
NGRAMA = 4
if len(sys.argv) > 3: # Tercer parámetro: n para n-grama
	NGRAMA = int(sys.argv[3])

NORMALIZAR = True # Posible valores: True o False
if len(sys.argv) > 4:
	NORMALIZAR = sys.argv[4] == "True"

CLASIFICADOR = 'svc' # Posibles valores: svc; knn; rf
if len(sys.argv) > 5:
	CLASIFICADOR = sys.argv[5]

FORMAT_VECT_CAR = 'npz' # Posibles valores: csv; npz
if len(sys.argv) > 6:
	FORMAT_VECT_CAR = sys.argv[6]

print('Nombre del Archivo python ejecutandose:', sys.argv[0])
print('Base de datos:', BASE_DATOS_PARAM)
print('Dirección:', DIRECCION)
print('N-Grama:', NGRAMA)
print('Normalizar:', NORMALIZAR)
print('Clasificador:', CLASIFICADOR)
print('Formato de Vector de Características:', FORMAT_VECT_CAR)


########################################## Constantes ##########################################
print("\nCreación de constantes...")

LETRAS = '12345678'
RANGO = 32
STEP = 1
if DIRECCION == 'H':
	V1 = 1
	V2 = NGRAMA
else:
	V1 = NGRAMA
	V2 = 1
VENTANA = (V1, V2)
PATH_BASE = "../databases/{}/".format(BASE_DATOS_PARAM)
"""
CSV_PRUEBA = "../resultados/vector_prueba_{bd}.{extension}".format(bd=BASE_DATOS_PARAM, extension=FORMAT_VECT_CAR)
CSV_ENTRENAMIENTO =  "../resultados/vector_entrenamiento_{bd}.{extension}".format(bd=BASE_DATOS_PARAM, extension=FORMAT_VECT_CAR)
"""
print("LETRAS:", LETRAS) 
print("RANGO:", RANGO)
print("STEP:", STEP)
print("VENTANA:", VENTANA)
print("CARPETA DE IMÁGENES:", PATH_BASE)


########################################## Funciones ##########################################
print("\nCreación de funciones...")

def getLetra(n):
	# n es el valor del pixel
	return LETRAS[int(n // RANGO)]

def reduccion_scala(image):
	return np.vectorize(getLetra)(image)

def extract_patches(image, patch_size, step):
	view = view_as_windows(image, patch_size, step) # crea ventanas cuadradas 

	return view.reshape(-1,patch_size[0]*patch_size[1]) # convierte cada ventana en un vector

def leer_imagenes(txt): # txt: archivo.txt
	# Lectura de nombres de archivos de entrenamiento
	f = open(PATH_BASE + "000/" + txt,"r")
	lineas = f.readlines()

	imagenes = []
	labels = []

	for i in range(1, len(lineas)):
		labels.append(lineas[i].split()[1])
		path = PATH_BASE + "images/" + lineas[i].split()[0]
		img = cv2.imread(path)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		imagenes.append(reduccion_scala(img_gray))

	return imagenes, labels # retorna un array de imágenes y un array de nombre de clases

def extraer_palabras(imagenes):
	palabras_por_imagen = []
	
	for imagen in imagenes:
		ventanas = extract_patches(imagen, VENTANA, STEP) # se extraen las ventanas de la imagen
		palabras_por_imagen.append(' '.join([''.join(v) for v in ventanas]))

	return palabras_por_imagen

def entrenamiento(imagenes_entrenamiento, labels_entrenamiento):
	print("\nEntrenamiento...")
	# Extracción de palabras de las imágenes de entrenamiento
	palabras_por_imagen_entrenamiento = extraer_palabras(imagenes_entrenamiento)
	
	# Generar diccionario
	vectorizer = CountVectorizer(token_pattern=r"(?u)[" + LETRAS + "]+", lowercase=False) # Nos indica con qué caracteres están conformadas las palabras.
	# Obtener los histogramas
	X = vectorizer.fit_transform(palabras_por_imagen_entrenamiento)

	diccionario = vectorizer.get_feature_names() # diccionario
	tam_diccionario = len(diccionario) # Tamaño del diccionario / alfabeto

	# creación archivo de vector de caracteristicas - entrenamiento
	'''
	if FORMAT_VECT_CAR == 'csv':
		with open(CSV_ENTRENAMIENTO,'w') as f2:
			w = csv.writer(f2)
			titulos = ["CLASE"] + vectorizer.get_feature_names()
			w.writerow(titulos)
			for i in range(len(X.toarray())):
				fila = []
				fila.append(labels_entrenamiento[i])
				fila = fila + X[i].toarray().tolist()[0]
				w.writerow(fila)
	elif FORMAT_VECT_CAR == 'npz':
		scipy.sparse.save_npz(CSV_ENTRENAMIENTO, X)
	'''
	
	if CLASIFICADOR == "svc":
		classifier = SVC(kernel = "linear")
	elif CLASIFICADOR == "knn":
		classifier = KNeighborsClassifier(n_neighbors = 1)
	elif CLASIFICADOR == "rf":
		classifier = RandomForestClassifier(random_state = 0)
		
	# Normalizar si es necesario
	if NORMALIZAR:
		hists_normalizados = [minmax_scale(h) for h in X.toarray()]
	else:
		hists_normalizados = X.toarray()
  
	# Entrenar clasificador
	classifier.fit(hists_normalizados, labels_entrenamiento)
	
	return classifier, vectorizer, tam_diccionario

def prueba(imagenes_prueba, labels_prueba, classifier, vectorizer):
	print("Prueba...")
	# Extracción de palabras de las imágenes de prueba
	palabras_por_imagen_prueba = extraer_palabras(imagenes_prueba)

	# Generación de histogramas de las imágenes de prueba
	histogramas_img_prueba = vectorizer.transform(palabras_por_imagen_prueba)
	
	# guardar vector de características - prueba
	'''
	if FORMAT_VECT_CAR == 'csv':
		with open(CSV_PRUEBA,'w') as f2:
			w = csv.writer(f2)
			titulos = ["CLASE"] + vectorizer.get_feature_names()
			w.writerow(titulos)
			for i in range(len(histogramas_img_prueba.toarray())):
				fila = []
				fila.append(labels_prueba[i])
				fila = fila + histogramas_img_prueba[i].toarray().tolist()[0]
				w.writerow(fila)
	elif FORMAT_VECT_CAR == 'npz':
		scipy.sparse.save_npz(CSV_PRUEBA, histogramas_img_prueba)
	'''

	# Normalizar si es necesario
	if NORMALIZAR:
		hists_normalizados = [minmax_scale(h) for h in histogramas_img_prueba.toarray()]
	else:
		hists_normalizados = histogramas_img_prueba.toarray()

	# Predicción
	predicciones = classifier.predict(hists_normalizados)

	# Resultados
	exactitud = accuracy_score(labels_prueba, predicciones)
	return exactitud

def main(imagenes_entrenamiento, labels_entrenamiento, imagenes_prueba, labels_prueba):
	classifier, vectorizer, tam_diccionario = entrenamiento(imagenes_entrenamiento, labels_entrenamiento)
	exactitud = prueba(imagenes_prueba, labels_prueba, classifier, vectorizer)

	return exactitud, tam_diccionario


########################################## EJECUTAR ##########################################
# Lectura de imágenes
print("\nLectura de imágenes y Reducción de escala...")
imagenes_entrenamiento, labels_entrenamiento = leer_imagenes("train.txt")
print("Cantidad de imágenes de entrenamiento:",  len(imagenes_entrenamiento))
imagenes_prueba, labels_prueba = leer_imagenes("test.txt")
print("Cantidad de imágenes de prueba:",  len(imagenes_prueba))

# Ejecución
exactitud, tam_diccionario = main(imagenes_entrenamiento, labels_entrenamiento, imagenes_prueba, labels_prueba)
print('\nResultados...')
print('Tamaño del diccionario:', tam_diccionario)
print('Exactitud:', exactitud)