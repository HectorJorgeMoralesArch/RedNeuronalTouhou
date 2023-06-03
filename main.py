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

def JSON2CSV(ID, data, character, folder):
    # Definir la ruta y nombre de archivo
    filename = 'archivo.csv'
    # Comprobar si el archivo CSV ya existe
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Escribir los encabezados
            writer.writerow(['ID', 'image id', 'image', 'tags', 'width', 'height', 'character', 'path'])
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for item in data:
            # Obtener los valores de cada campo
            image_id = item['id']
            image = item['image']
            tags = item['tags']
            width = item['width']
            height = item['height']
            character = character
            path = folder + "/" + image
            
            # Reemplazar los espacios en blanco en la cadena de tags con comas
            tags = tags.replace(' ', ', ')
            # Escribir la fila en el archivo CSV
            writer.writerow([ID, image_id, image, tags, width, height, character, path])
    # Cerrar el archivo
    f.close()

def download_image(url, image_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Verificar si hubo errores en la respuesta HTTP
        with open(image_path, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
    except requests.exceptions.RequestException as e:
        print(f"Error al realizar la solicitud: {e}")
    except IOError as e:
        print(f"Error al guardar la imagen: {e}")

def download_images(tags, character, folder):
    # Definir los parámetros de búsqueda
    limit = 100    # el número máximo de resultados por página
    pages = 50    # calcular el número total de páginas a descargar
    image_count = 1   # Inicializar el contador de imagen

    
    # Definir la URL base de la API de Safebooru
    url = 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags='+tags
    # Iterar a través de las páginas y descargar las imágenes
    while image_count < 5000:
        # Realizar la solicitud a la API de Safebooru para la página actual
        response = requests.get(f'{url}&limit={limit}&pid={pages}&json=1')

        # Analizar la respuesta JSON
        data = json.loads(response.content)

        # Descargar cada imagen en la página actual utilizando la programación concurrente
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_image = {
                executor.submit(
                    download_image,
                    f'https://safebooru.org//images/{item["directory"]}/{item["image"]}?{item["id"]}',
                    f'{item["image"]}'
                ): item
                for item in data
            }

            for future in concurrent.futures.as_completed(future_to_image):
                item = future_to_image[future]
                try:
                    future.result()
                    image_count += 1
                    print(f"{character}\t{image_count}")
                except Exception as e:
                    print(f'Error al descargar la imagen {item["id"]}: {e}')

        # Convertir el JSON a CSV
        ID=f"{character}{image_count}"
        JSON2CSV(ID, data, character, character_folder)

        # Decrementar el número de páginas para obtener la siguiente página en la siguiente iteración
        pages -= 1

        # Retraso aleatorio antes de realizar la siguiente solicitud
        time.sleep(random.randint(1, 3))

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
    # Definir los personajes y las carpetas de destino
    characters = {
    'Hakurei Reimu': 'hakurei_reimu+1girl+touhou',
    'Kirisame Marisa': 'kirisame_marisa+1girl+touhou',
    'Cirno': 'cirno+1girl+touhou',
    'Komeiji Koishi': 'komeiji_koishi+1girl+touhou',
    'Komeiji Satori': 'komeiji_satori+1girl+touhou',
    'Yakumo Yukari': 'yakumo_yukari+1girl+touhou',
    'Shameimaru Aya': 'shameimaru_aya+1girl+touhou',
    'Fujiwara No Mokou': 'fujiwara_no_mokou+1girl+touhou',
    'Houraisan Kaguya': 'houraisan_kaguya+1girl+touhou',
    'Rumia': 'rumia+1girl+touhou'
    }
    for character, url in characters.items():
        character_folder = character.replace(' ', '_')
        os.makedirs(character_folder, exist_ok=True)
    # Descargar las imágenes para cada personaje
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_character = {
            executor.submit(
                download_images,
                f'{tags}',
                character,
                f'{character.replace(" ", "_")}'
            ): (character, character.replace(" ", "_"))
            for character, tags in characters.items()
        }

        for future in concurrent.futures.as_completed(future_to_character):
            character, folder = future_to_character[future]
            try:
                future.result()
                print(f"Descarga de imágenes para {character} completada.")
            except Exception as e:
                print(f"Error al descargar imágenes para {character}: {e}")

    print("Todas las descargas de imágenes han sido completadas.")
    model = RedNeuronal("archivo.csv")
    model.save(f'Touhou_model.h5')

if __name__ == '__main__':
    main()