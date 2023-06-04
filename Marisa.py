import requests
import shutil
import json
import csv
import os
import pandas as pd

def JSON2SCV(data, character):
    # Definir la ruta y nombre de archivo
    filename = 'archivo.csv'
    # Comprobar si el archivo CSV ya existe
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Escribir los encabezados
            writer.writerow(['image', 'width', 'height', 'character'])

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for item in data:
            # Obtener los valores de cada campo
            #image_id = item['id']
            image = item['image']
            #tags = item['tags']
            width = item['width']
            height = item['height']

            # Reemplazar los espacios en blanco en la cadena de tags con comas
            #tags = tags.replace(' ', ', ')
            # Escribir la fila en el archivo CSV
            writer.writerow([image, width, height, character])
    # Cerrar el archivo
    f.close()
def descargar(url, character):
    # Definir los parámetros de búsqueda
    limit = 100
    pages = 100

    # Inicializar el contador de imagen
    image_count = 1
    # Iterar a través de las páginas y descargar las imágenes
    while image_count<5000:
        # Realizar la solicitud a la API de Safebooru para la página actual
        response = requests.get(url)
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
            #with open(f'{character}_{image_count}.json', 'w') as f:
            #    json.dump(item, f)
            # Obtener la información de la imagen
            id=item['id']
            image_url = f'https://safebooru.org//images/{item["directory"]}/{item["image"]}?{item["id"]}'
            #owner = item['owner'] # el nombre del artista
            #tagsP = item['tags']# las etiquetas de la imagen
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
            print(f'Imagen {image_count}:')
            #print(f'Artista: {owner}')
            #print(f'Etiquetas: {tagsP}')
            #print(f'URL: {image_url}\n')
            JSON2SCV(data, character)# Incrementar el contador de imagen
            image_count += 1
            
def RedNeuronal():
    # Cargar el archivo CSV
    df = pd.read_csv("archivo.csv")
    image_paths = df['image'].tolist()

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
    # Definir los parámetros de búsqueda
    limit = 100
    pages = 100
    # Definir los personajes y las carpetas de destino
    characters = {
        'Hakurei Reimu': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=hakurei_reimu+1girl+touhou&limit={limit}&pid={pages}&json=1',
        'Kirisame Marisa': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=kirisame_marisa+1girl+touhou&limit={limit}&pid={pages}&json=1',
        'Cirno': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=cirno+1girl+touhou&limit={limit}&pid={pages}&json=1',
        'Komeiji Koishi': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=komeiji_koishi+1girl+touhou&limit={limit}&pid={pages}&json=1',
        'Komeiji Satori': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=komeiji_satori+1girl+touhou&limit={limit}&pid={pages}&json=1',
        'Yakumo Yukari': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=yakumo_yukari+1girl+touhou&limit={limit}&pid={pages}&json=1',
        'Shameimaru Aya': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=shameimaru_aya+1girl+touhou&limit={limit}&pid={pages}&json=1',
        'Fujiwara No Mokou': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=fujiwara_no_mokou+1girl+touhou&limit={limit}&pid={pages}&json=1',
        'Houraisan Kaguya': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=houraisan_kaguya+1girl+touhou&limit={limit}&pid={pages}&json=1',
        'Rumia': 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=rumia+1girl+touhou&limit={limit}&pid={pages}&json=1'
    }
    descargar('https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=kirisame_marisa+1girl+touhou&limit={limit}&pid={pages}&json=1',"Kirisame Marisa")

if _name_ == '_main_':
    main()