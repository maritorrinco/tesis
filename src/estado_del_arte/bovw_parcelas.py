
"""
Implementación del paper "Lung Texture Classification Using Bag of Visual Words"
"""

"""
Cómo ejecutar desde /tesis/src/estado_del_arte:
python3 -W ignore -u bovw.py Outex_TC_00013 scv |& tee -a ../../resultados/bowv_2/[bd]_[clasificador].txt &
"""

import numpy as np
import cv2
from skimage.util import view_as_windows
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import euclidean_distances
import sys
import csv


########################################## Parámetros ##########################################
print("\n\nGeneración de parámetros...")

BASE_DATOS_PARAM = "Outex_TC_00013" # Nombre de la base de datos a usar
if len(sys.argv) > 1:
  BASE_DATOS_PARAM = sys.argv[1] 

CLASIFICADOR = "svc" # Posibles valores: svc; knn; rf
if len(sys.argv) > 2:
  CLASIFICADOR = sys.argv[2] 

FORMAT_VECT_CAR = 'npz' # Posibles valores: csv; npz
if len(sys.argv) > 3:
	FORMAT_VECT_CAR = sys.argv[3]

print('Nombre del Archivo python ejecutandose:', sys.argv[0])
print('Base de datos:', BASE_DATOS_PARAM)
print('Clasificador:', CLASIFICADOR)
print('Formato de Vector de Características:', FORMAT_VECT_CAR)


########################################## Constantes ##########################################
print("\nCreación de constantes...")

PATCH_SIZE = (7,7) # tamaño de parche
PATCH_STEP = 3
NUMBER_COMPONENTS = 10 # numero de pca componentes
NUMBER_CLUSTER = 50 # cada sub-diccionario (de cada clase) tendrá NUMBER_CLUSTER palabras
PATH_BASE = "../../databases/{}/".format(BASE_DATOS_PARAM)
"""
CSV_PRUEBA = "../../resultados/lung_BoVW/vector_prueba_{bd}.{extension}".format(bd=BASE_DATOS_PARAM, extension=FORMAT_VECT_CAR)
CSV_ENTRENAMIENTO =  "../../resultados/lung_BoVW/vector_entrenamiento_{bd}.{extension}".format(bd=BASE_DATOS_PARAM, extension=FORMAT_VECT_CAR)
"""
print("TAMAÑO DE PARCHE/VENTANA:", PATCH_SIZE)
print("PATCH STEP:", PATCH_STEP)
print("LONGITUD DE CADA PALABRA:", NUMBER_COMPONENTS)
print("NÚMERO DE CLUSTER:", NUMBER_CLUSTER)
print("CARPETA DE IMÁGENES:", PATH_BASE)


########################################## Funciones ##########################################
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
		imagenes.append(img_gray)

	return imagenes, labels # retorna un array de imágenes y un array de nombre de clases

def extract_patches(image, patch_size, step):
	view = view_as_windows(image, patch_size, step) # crea ventanas cuadradas 

	return view.reshape(-1,patch_size[0]*patch_size[1]) # convierte cada ventana en un vector

def extraer_features(imagenes, labels):
	number_images = len(imagenes)
	number_features_per_image = [0]*number_images
	number_features_per_class = dict.fromkeys(set(labels), 0)

	for index in range(number_images):
		patches = extract_patches(imagenes[index], PATCH_SIZE, PATCH_STEP) # obtiene los feature vectors
		number_features_per_image[index] = patches.shape[0]
		number_features_per_class[labels[index]] += patches.shape[0]
		features = patches if index == 0 else np.vstack((features, patches))

	return features, number_features_per_image, number_features_per_class

def rearrange_features(all_features, number_of_features_per_element): # number_of_features_per_element (element could be class or image)
	r_features = []
	index = 0

	if type(number_of_features_per_element) is dict: # number_of_features_per_class
		for key, number in number_of_features_per_element.items():
			r_features.append(all_features[index : index + number])
			index += number
	else: # number_of_features_per_image
		for number in number_of_features_per_element:
			r_features.append(all_features[index : index + number])
			index += number
   
	return r_features

def cluster_descriptors(descriptors, no_clusters):
	kmeans = KMeans(n_clusters = no_clusters, n_jobs=-1, random_state=1).fit(descriptors)

	return kmeans

def clusterization(features_per_class):
	dictionary = []

	for i in range(NUMBER_CLASSES):
		print("clusterización número:", i)
		kmeans = cluster_descriptors(features_per_class[i], NUMBER_CLUSTER)
		dictionary.append(kmeans.cluster_centers_)

	GLOBAL_DICTIONARY_LENGTH = NUMBER_CLASSES*NUMBER_CLUSTER 
	global_dictionary = np.array(dictionary).reshape(GLOBAL_DICTIONARY_LENGTH, NUMBER_COMPONENTS)

	return global_dictionary

def create_histogram_per_image(global_dictionary, features_per_image):
	number_images = len(features_per_image)
	histogram_per_image = np.zeros((number_images, len(global_dictionary)))

	for i in range(number_images): #  por cada imagen
		for j in range(len(features_per_image[i])): # cada feature de la imagen i
			feature = features_per_image[i][j]
			distances = euclidean_distances([feature], global_dictionary) # encuentra la palabra más cercana en el diccionario de palabras
			index_min = np.argmin(distances) # obtiene su indice
			histogram_per_image[i][index_min] += 1 # crea histograma por imagen
		histogram_per_image[i] = histogram_per_image[i]/sum(histogram_per_image[i]) # normalizar histograma

	return histogram_per_image

def get_classifier(features, train_labels):
	if CLASIFICADOR == "svc":
		classifier = SVC(kernel = "linear")
	elif CLASIFICADOR == "knn":
		classifier = KNeighborsClassifier(n_neighbors = 1)
	elif CLASIFICADOR == "rf":
		classifier = RandomForestClassifier(random_state = 0)

	# Entrenar clasificador
	classifier.fit(features, train_labels)

	return classifier

def entrenamiento(imagenes_entrenamiento, labels_entrenamiento):
	print("\nEntrenamiento...")
	
	# Step 1: extract patches
	all_features, number_features_per_image, number_features_per_class = extraer_features(imagenes_entrenamiento, labels_entrenamiento)

	# Step 2: scaling and PCA - entrenamiento
	scaler = StandardScaler()
	scaler.fit(all_features)
	features_scaled = scaler.transform(all_features)
	pca = PCA(n_components = NUMBER_COMPONENTS)
	pca.fit(features_scaled)
	features_pca = pca.transform(features_scaled)
 
	# Step 3: clustering using K-means
	features_per_class = rearrange_features(features_pca, number_features_per_class)
	global_dictionary = clusterization(features_per_class)
 
	# Step 4 and 5: Quantization: Cuantificación utilizando el diccionario de textura - Histogram of each image
	features_per_image = rearrange_features(features_pca, number_features_per_image)
	histogram_per_image = create_histogram_per_image(global_dictionary, features_per_image)

	# Step 6: Classification
	classifier = get_classifier(histogram_per_image, labels_entrenamiento)

	return scaler, pca, global_dictionary, classifier

def prueba(imagenes_prueba, labels_prueba, scaler, pca, global_dictionary, classifier):
	print("Prueba...")
	# Step 1: extract patches
	all_features, number_features_per_image, number_features_per_class = extraer_features(imagenes_prueba, labels_prueba)
 
	# Step 2: scaling and PCA - prueba
	features_scaled = scaler.transform(all_features)
	features_pca = pca.transform(features_scaled)
 
	# Step 3 and 4: Quantization: Cuantificación utilizando el diccionario de textura - Histogram of each image
	features_per_image = rearrange_features(features_pca, number_features_per_image)
	histogram_per_image = create_histogram_per_image(global_dictionary, features_per_image)
 
	# Step 5: Classification - predicción
	predicciones = classifier.predict(histogram_per_image)
	
	# Resultados
	exactitud = accuracy_score(labels_prueba, predicciones)
	return exactitud

def main(imagenes_entrenamiento, labels_entrenamiento, imagenes_prueba, labels_prueba):
	scaler, pca, global_dictionary, classifier = entrenamiento(imagenes_entrenamiento, labels_entrenamiento)
	exactitud = prueba(imagenes_prueba, labels_prueba, scaler, pca, global_dictionary, classifier)

	return exactitud, len(global_dictionary)


########################################## EJECUTAR ##########################################
# Lectura de imágenes
print("\nLectura de imágenes...")
imagenes_entrenamiento, labels_entrenamiento = leer_imagenes("train.txt")
print("Cantidad de imágenes de entrenamiento:",  len(imagenes_entrenamiento))
imagenes_prueba, labels_prueba = leer_imagenes("test.txt")
print("Cantidad de imágenes de prueba:",  len(imagenes_prueba))
NUMBER_CLASSES = len(set(labels_entrenamiento))
print('NÚMERO DE CLASES:', NUMBER_CLASSES)

#Ejecución
exactitud, tam_diccionario = main(imagenes_entrenamiento, labels_entrenamiento, imagenes_prueba, labels_prueba)
print('\nResultados...')
print('Tamaño del diccionario:', tam_diccionario)
print('Exactitud:', exactitud)