import requests
import os
import json
import csv
import random
import shutil
import concurrent.futures
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

def JSON2SCV(data, character):
    # Definir la ruta y nombre de archivo
    filename = 'archivo.csv'
    # Comprobar si el archivo CSV ya existe
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Escribir los encabezados
            writer.writerow(['id', 'image', 'tags', 'width', 'height', 'character'])

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for item in data:
            # Obtener los valores de cada campo
            id = item['id']
            image = item['image']
            tags = item['tags']
            width = item['width']
            height = item['height']
            character = character

            # Reemplazar los espacios en blanco en la cadena de tags con comas
            tags = tags.replace(' ', ', ')
            # Escribir la fila en el archivo CSV
            writer.writerow([id, image, tags, width, height, character])
    # Cerrar el archivo
    f.close()

def download_json(url, character):
    # Realizar la solicitud a la API de Safebooru para obtener los datos JSON
    response = requests.get(url)
    # Analizar la respuesta JSON
    data = json.loads(response.content)
    # Guardar los datos JSON en un archivo
    with open(f'{character}.json', 'w') as f:
        json.dump(data, f)

    # Agregar los datos al archivo CSV
    JSON2SCV(data, character)

def download_image(image_data):
    image_url, image_count = image_data

    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Verificar si hubo errores en la respuesta HTTP
        with open(f'image_{image_count}.jpg', 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
    except requests.exceptions.RequestException as e:
        print(f"Error al realizar la solicitud: {e}")
    except IOError as e:
        print(f"Error al guardar la imagen: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

def download_images(urls):
    image_count = 1

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for url in urls:
            future = executor.submit(download_image, (url, image_count))
            futures.append(future)
            image_count += 1

        # Esperar a que todas las tareas de descarga se completen
        concurrent.futures.wait(futures)
def RedNeuronal(csv_file):
    # Cargar el archivo CSV
    df = pd.read_csv(csv_file)
    image_paths = df['path'].tolist()

    # Dividir el conjunto de datos en entrenamiento, prueba y validación
    train_ratio = 0.6
    validation_ratio = 0.2
    test_ratio = 0.2

    # Obtener la lista de rutas de imágenes
    random.shuffle(image_paths)

    # Calcular la cantidad de imágenes para cada conjunto
    total_images = len(image_paths)
    num_train = int(total_images * train_ratio)
    num_validation = int(total_images * validation_ratio)
    num_test = total_images - num_train - num_validation

    # Dividir las rutas de imágenes en conjuntos de entrenamiento, prueba y validación
    train_paths = image_paths[:num_train]
    validation_paths = image_paths[num_train:num_train + num_validation]
    test_paths = image_paths[num_train + num_validation:]

    # Definir el generador de datos para el conjunto de entrenamiento
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Cargar y preparar las imágenes de entrenamiento
    train_images = []
    train_labels = []
    for image_path in train_paths:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = train_datagen.random_transform(image)
        train_images.append(image)
        train_labels.append(os.path.dirname(image_path))

    # Convertir las listas de imágenes y etiquetas en matrices NumPy
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Definir el generador de datos para el conjunto de prueba
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Cargar y preparar las imágenes de prueba
    test_images = []
    test_labels = []
    for image_path in test_paths:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = test_datagen.random_transform(image)
        test_images.append(image)
        test_labels.append(os.path.dirname(image_path))

    # Convertir las listas de imágenes y etiquetas en matrices NumPy
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Definir el generador de datos para el conjunto de validación
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Cargar y preparar las imágenes de validación
    validation_images = []
    validation_labels = []
    for image_path in validation_paths:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = validation_datagen.random_transform(image)
        validation_images.append(image)
        validation_labels.append(os.path.dirname(image_path))

    # Convertir las listas de imágenes y etiquetas en matrices NumPy
    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)

    # Crear un modelo de red neuronal
    model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=True, weights=None, classes=1)

    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(train_images, train_labels, validation_data=(validation_images, validation_labels), epochs=10)

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # Obtener las predicciones del modelo en el conjunto de prueba
    predictions = model.predict(test_images)

    # Obtener las etiquetas predichas
    predicted_labels = [os.path.dirname(image_path) if prediction >= 0.5 else 'Not ' + os.path.dirname(image_path) for prediction, image_path in zip(predictions, test_paths)]

    # Crear una matriz de confusión
    cm = confusion_matrix(test_labels, predicted_labels)

    # Imprimir la matriz de confusión
    print('Confusion Matrix:')
    print(cm)

    # Imprimir el informe de clasificación
    print('Classification Report:')
    print(classification_report(test_labels, predicted_labels))

    # Guardar el modelo entrenado
    model.save(f'model.h5')

    # Retornar el modelo entrenado
    return model
def main():
    limit = 5000
    page = 1
    characters = {
        'Hakurei Reimu': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=hakurei_reimu+1girl+touhou&limit={limit}&pid={page}&json=1',
        'Kirisame Marisa': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=kirisame_marisa+1girl+touhou&limit={limit}&pid={page}&json=1',
        'Cirno': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=cirno+1girl+touhou&limit={limit}&pid={page}&json=1',
        'Komeiji Koishi': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=komeiji_koishi+1girl+touhou&limit={limit}&pid={page}&json=1',
        'Komeiji Satori': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=komeiji_satori+1girl+touhou&limit={limit}&pid={page}&json=1',
        'Yakumo Yukari': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=yakumo_yukari+1girl+touhou&limit={limit}&pid={page}&json=1',
        'Shameimaru Aya': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=shameimaru_aya+1girl+touhou&limit={limit}&pid={page}&json=1',
        'Fujiwara No Mokou': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=fujiwara_no_mokou+1girl+touhou&limit={limit}&pid={page}&json=1',
        'Houraisan Kaguya': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=houraisan_kaguya+1girl+touhou&limit={limit}&pid={page}&json=1',
        'Rumia': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=rumia+1girl+touhou&limit={limit}&pid={page}&json=1'
    }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        json_futures = []
        for character, url in characters.items():
            print(f'Downloading JSON data of {character}...')
            future = executor.submit(download_json, url, character)
            json_futures.append(future)

        concurrent.futures.wait(json_futures)

        image_urls = []
        for character in characters.keys():
            with open(f'{character}.json', 'r') as f:
                data = json.load(f)
                for item in data:
                    image_urls.append(item['image'])

        download_images(image_urls)

    model = RedNeuronal("archivo.csv")
    model.save(f'Touhou_model.h5')

if __name__ == '__main__':
    main()