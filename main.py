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

def JSON2SCV(data, Character, folder):
    # Definir la ruta y nombre de archivo
    filename = 'archivo.csv'
    # Comprobar si el archivo CSV ya existe
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Escribir los encabezados
            writer.writerow(['id', 'image', 'tags', 'width', 'height', 'character', 'path'])
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for item in data:
            # Obtener los valores de cada campo
            id = item['id']
            image = item['image']
            tags = item['tags']
            width = item['width']
            height = item['height']
            character = Character
            path = folder + "/"+image
            
            # Reemplazar los espacios en blanco en la cadena de tags con comas
            tags = tags.replace(' ', ', ')
            # Escribir la fila en el archivo CSV
            writer.writerow([id, image, tags, width, height, character, path])
    # Cerrar el archivo
    f.close()

def descargar(url, character, folder):
    # Definir los parámetros de búsqueda
    limit = 100    # el número máximo de resultados por página
    pages = 50    # calcular el número total de páginas a descargar
    
    # Inicializar el contador de imagen
    image_count = 1
    
    # Iterar a través de las páginas y descargar las imágenes
    while(image_count<5000):
        # Realizar la solicitud a la API de Safebooru para la página actual
        response = requests.get(f'{url}&limit={limit}&pid={pages}&json=1')
        
        # Analizar la respuesta JSON
        data = json.loads(response.content)
        
        # Descargar cada imagen en la página actual
        for item in data:
            # Obtener la extensión del archivo de imagen
            image_extension = os.path.splitext(item["image"])[1].lower()
            
            # Verificar si la extensión es válida (.jpg o .png)
            if image_extension not in ['.jpg', '.png']:
                continue
            
            # Guardar la respuesta JSON en un archivo
            #with open(f'safebooru_{image_count}.json', 'w') as f:
            #    json.dump(item, f)
            
            # Obtener la información de la imagen
            id=item['id']
            image_url = f'https://safebooru.org//images/{item["directory"]}/{item["image"]}?{item["id"]}'
            #owner = item['owner'] # el nombre del artista
            #tagsP = item['tags']# las etiquetas de la imagen
            # Descargar la imagen
            response = requests.get(image_url, stream=True)
            with open(f'{folder}_{image_count}', 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            # Imprimir la información de la imagen
            #print(f'Artista: {owner}')
            #print(f'Etiquetas: {tagsP}')
            #print(f'URL: {image_url}\n')
            # Imprimir la información de la imagen
            print(f'Character\t{character}\t\t\t\tImagen\t{image_count}:')
            #print(f'URL: {image_url}\n')
            image_count += 1
            # Agregar los datos al archivo CSV
            JSON2SCV(data, character, folder)


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


def randomize_csv():
    # Definir la ruta y nombre de archivo
    filename = 'archivo.csv'
    randomfilename = 'random.csv'
    # Comprobar si el archivo CSV ya existe
    if not os.path.isfile(randomfilename):
        with open(randomfilename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Escribir los encabezados
            writer.writerow(['Path'])
    Path=folder+"/"+Character
    # Leer el archivo CSV de entrada
    with open(filename, 'r') as file:
        reader = csv.reader(file["Path"])
        data = list(reader)
    # Randomizar los datos
    random.shuffle(data)
    # Escribir los datos randomizados en el archivo CSV de salida
    with open(randomfilename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("Archivo randomizado generado con éxito.")



def descargar_wrapper(url, images, character, folder):
    descargar(url, images, character, folder)

def main():

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

    # Crear una carpeta para cada personaje y descargar las imágenes
    for character, url in characters.items():
        print(f'Downloading images of {character}...')
        character_folder = character.replace(' ', '_')
        os.makedirs(character_folder, exist_ok=True)
        descargar(url, character, character_folder)
    model = RedNeuronal("archivo.csv")
    model.save(f'Touhou_model.h5')
    
if __name__ == '__main__':
    main()

