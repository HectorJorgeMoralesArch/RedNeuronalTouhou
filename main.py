import requests
import os
import json
import csv
import random
import shutil
import concurrent.futures
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time

def JSON2SCV(item, character, count):
# Definir la ruta y nombre de archivo
    filename = 'archivo.csv'
    # Comprobar si el archivo CSV ya existe
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Escribir los encabezados
            writer.writerow(['character', 'count','image', 'width', 'height'])

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        # Obtener los valores de cada campo
        #image_id = item['id']
        image = item['image']
        #tags = item['tags']
        width = item['width']
        height = item['height']
        # Reemplazar los espacios en blanco en la cadena de tags con comas
        #tags = tags.replace(' ', ', ')
        # Escribir la fila en el archivo CSV
        writer.writerow([character, count, image, width, height])
    # Cerrar el archivo
    f.close()

# Variable global para controlar la ejecución de los hilos
stop_execution = False


def descargar(url, character):
    global stop_execution
    
    # Definir los parámetros de búsqueda
    limit = 100    # el número máximo de resultados por página
    pages = 50    # calcular el número total de páginas a descargar
    
    # Inicializar el contador de imagen
    image_count = 1
    
    # Iterar a través de las páginas y descargar las imágenes
    start_time = time.time()  # Tomar tiempo de inicio
    
    while image_count < 5000 and not stop_execution:
        # Realizar la solicitud a la API de Safebooru para la página actual
        response = requests.get(f'{url}&limit={limit}&pid={pages}&json=1')
        data = json.loads(response.content)
        # Descargar cada imagen en la página actual
        for item in data:
            # Obtener la extensión del archivo de imagen
            image_extension = os.path.splitext(item["image"])[1].lower()

            # Verificar si la extensión es válida (.jpg o .png)
            if image_extension not in ['.jpg', '.png']:
                continue
            if item['sample']:
                continue
            # Guardar la respuesta JSON en un archivo
            # with open(f'{character}_{image_count}.json', 'w') as f:
            #    json.dump(item, f)
            # Obtener la información de la imagen
            id = item['id']
            image_url = f'https://safebooru.org//images/{item["directory"]}/{item["image"]}?{item["id"]}'
            # owner = item['owner'] # el nombre del artista
            # tagsP = item['tags']# las etiquetas de la imagen
            # Descargar la imagen
            try:
                response = requests.get(image_url, stream=True)
                response.raise_for_status()  # Verificar si hubo errores en la respuesta HTTP
                with open(f'{item["image"]}', 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
            except requests.exceptions.RequestException as e:
                print(f"Error al realizar la solicitud: {e}")
                continue
            except IOError as e:
                print(f"Error al guardar la imagen: {e}")
                continue
            # Imprimir la información de la imagen
            print(f'Character\t{character}\tImagen {image_count}')
            # print(f'Artista: {owner}')
            # print(f'Etiquetas: {tagsP}')
            # print(f'URL: {image_url}\n')
            JSON2SCV(item, character, image_count)  # Incrementar el contador de imagen
            image_count += 1
        pages -= 1



def RedNeuronal():
    # Cargar el archivo CSV
    df = pd.read_csv("archivo.csv")

    # Filtrar por cantidad mínima de imágenes
    min_count = 100
    character_counts = df['character'].value_counts()
    filtered_characters = character_counts[character_counts >= min_count].index.tolist()

    # Filtrar por categorías específicas
    df_filtered = df[df['character'].isin(filtered_characters)]

    # Imprimir los resultados
    print(df_filtered['character'])
    df=df_filtered
    # Dividir el conjunto de datos en entrenamiento, prueba y validación
    train_ratio = 0.6
    validation_ratio = 0.2
    test_ratio = 0.2

    # Obtener la lista de rutas de imágenes
    image_paths = df['image'].tolist()
    labels = df['character'].tolist()

    # Codificar las etiquetas
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Dividir las rutas de imágenes y las etiquetas en conjuntos de entrenamiento, prueba y validación
    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels_encoded, test_size=test_ratio, stratify=labels_encoded, random_state=42)
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(train_paths, train_labels, test_size=validation_ratio/(train_ratio+validation_ratio), stratify=train_labels, random_state=42)

    # Definir el generador de datos para el conjunto de entrenamiento
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    # Cargar y preparar las imágenes de entrenamiento
    train_images = []
    for image_path in train_paths:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = train_datagen.random_transform(image)
        train_images.append(image)

    # Convertir las listas de imágenes y etiquetas en matrices NumPy
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Definir el generador de datos para el conjunto de prueba
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    # Cargar y preparar las imágenes de prueba
    test_images = []
    for image_path in test_paths:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = test_datagen.random_transform(image)
        test_images.append(image)

    # Convertir las listas de imágenes y etiquetas en matrices NumPy
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Definir el generador de datos para el conjunto de validación
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    # Cargar y preparar las imágenes de validación
    validation_images = []
    for image_path in validation_paths:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = validation_datagen.random_transform(image)
        validation_images.append(image)

    # Convertir las listas de imágenes y etiquetas en matrices NumPy
    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)

    # Crear un modelo de red neuronal
    num_classes = 10
    model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=True, weights=None, classes=num_classes)

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'f1-score'])

    # Entrenar el modelo
    model.fit(train_images, tf.keras.utils.to_categorical(train_labels, num_classes=num_classes),
              validation_data=(validation_images, tf.keras.utils.to_categorical(validation_labels, num_classes=num_classes)),
              epochs=10)

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy, precision, recall, f1_score = model.evaluate(test_images, tf.keras.utils.to_categorical(test_labels, num_classes=num_classes))
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1_score}')

    # Obtener las predicciones del modelo en el conjunto de prueba
    predictions = model.predict(test_images)

    # Obtener las etiquetas predichas
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_labels)

    # Crear una matriz de confusión
    cm = confusion_matrix(test_labels, predicted_labels)

    # Imprimir la matriz de confusión
    print('Confusion Matrix:')
    print(cm)

    # Imprimir el informe de clasificación
    print('Classification Report:')
    print(classification_report(test_labels, predicted_labels))

    # Calcular la curva ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels, predictions[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Graficar la curva ROC
    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

    # Retornar el modelo entrenado
    return model

def main():
    global stop_execution
    
    # Inicio del programa principal
    start_time_main = time.time()
    
    # Definir la lista de personajes
    # Lista de personajes y sus respectivos enlaces
    characters = {
        'Hakurei Reimu': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=hakurei_reimu+1girl+touhou',
        'Kirisame Marisa': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=kirisame_marisa+1girl+touhou',
        'Cirno': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=cirno+1girl+touhou',
        'Komeiji Koishi': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=komeiji_koishi+1girl+touhou',
        'Komeiji Satori': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=komeiji_satori+1girl+touhou',
        'Yakumo Yukari': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=yakumo_yukari+1girl+touhou',
        'Shameimaru Aya': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=shameimaru_aya+1girl+touhou',
        'Fujiwara No Mokou': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=fujiwara_no_mokou+1girl+touhou',
        'Houraisan Kaguya': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=houraisan_kaguya+1girl+touhou',
        'Rumia': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=rumia+1girl+touhou'
    }

    # Lista para almacenar los hilos
    thread_list = []
    
    # Inicio del tiempo A
    start_time_A = time.time()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for character, url in characters.items():
            thread = executor.submit(descargar, url, character)
            thread_list.append(thread)

        # Esperar a que todas las descargas se completen
        concurrent.futures.wait(thread_list)
    
    # Fin del tiempo A
    end_time_A = time.time()
    elapsed_time_A = end_time_A - start_time_A

    print("Todas las descargas de imágenes han sido completadas.")
    
    # Cargar el archivo CSV
    df = pd.read_csv("archivo.csv")

    # Contar la cantidad de personajes
    character_counts = df['character'].value_counts()

    # Imprimir los resultados
    print(character_counts)
    
    # Tiempo B
    for thread in thread_list:
        thread_result = thread.result()
        print(f"El hilo del personaje {thread_result[0]} ha terminado en {thread_result[1]} segundos.")
    
    # Tiempo C
    end_time_main = time.time()
    elapsed_time_main = end_time_main - start_time_main
    difference_BC = elapsed_time_main - elapsed_time_A
    print(f"La diferencia entre el tiempo B y C fue de {difference_BC} segundos.")
    print(f"El tiempo total para descargar todas las imágenes fue de {elapsed_time_main} segundos.")
    
    # Inicio de la red neuronal
    start_time_D = time.time()
    
    model = RedNeuronal()
    model.save(f'Touhou_model.h5')
    
    # Fin de la red neuronal
    end_time_D = time.time()
    elapsed_time_D = end_time_D - start_time_D
    
    print(f"La red neuronal ha sido entrenada en {elapsed_time_D} segundos.")
    
    # Tiempo E
    end_time_E = time.time()
    elapsed_time_E = end_time_E - start_time_main
    difference_DE = elapsed_time_E - elapsed_time_D
    print(f"La diferencia entre el tiempo D y E fue de {difference_DE} segundos.")
    print(f"El tiempo total del programa fue de {elapsed_time_E} segundos.")


if __name__ == '__main__':
    main()