import requests
import os
import json
import csv
import shutil
from json.decoder import JSONDecodeError
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def JSON2CSV(data, character):
    # Definir la ruta y nombre de archivo
    filename = 'archivo.csv'
    # Comprobar si el archivo CSV ya existe
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Escribir los encabezados
            writer.writerow(['directory', 'hash', 'height', 'id', 'image', 'change', 'owner', 'parent_id', 'rating', 'sample', 'sample_height', 'sample_width', 'score', 'tags', 'width', 'character'])
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for item in data:
            # Obtener los valores de cada campo
            directory = item['directory']
            hash_value = item['hash']
            height = item['height']
            image_id = item['id']
            image = item['image']
            change = item['change']
            owner = item['owner']
            parent_id = item['parent_id']
            rating = item['rating']
            sample = item['sample']
            sample_height = item['sample_height']
            sample_width = item['sample_width']
            score = item['score']
            tags = item['tags']
            width = item['width']
            character = character

            # Reemplazar los espacios en blanco en la cadena de tags con comas
            tags = tags.replace(' ', ', ')
            # Escribir la fila en el archivo CSV
            writer.writerow([directory, hash_value, height, image_id, image, change, owner, parent_id, rating, sample, sample_height, sample_width, score, tags, width, character])
    # Cerrar el archivo
    f.close()


def download_images(url, count, character_name):
    # Crear la carpeta si no existe
    folder_name = character_name.replace(' ', '_')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Definir los parámetros de búsqueda
    limit = 100  # el número máximo de resultados por página
    pages = (count + limit - 1) // limit  # calcular el número total de páginas a descargar

    # Inicializar el contador de imagen
    image_count = 1

    # Iterar a través de las páginas y descargar las imágenes
    for page in range(1, pages + 1):
        # Realizar la solicitud a la API de Safebooru para la página actual
        response = requests.get(f'{url}&limit={limit}&pid={page}&json=1')

        # Analizar la respuesta JSON
        try:
            data = json.loads(response.content)
        except JSONDecodeError:
            print(f'Error decoding JSON response for {character_name}.')
            return

        # Descargar cada imagen en la página actual
        for item in data:
            # Obtener la información de la imagen
            image_url = f'https://safebooru.org//images/{item["directory"]}/{item["image"]}'
            image_id = item['id']

            # Construir la ruta de archivo completa
            file_path = os.path.join(folder_name, f'{image_id}.jpg')

            # Descargar la imagen y guardarla en el archivo
            response = requests.get(image_url, stream=True)
            with open(file_path, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)

            # Mostrar el progreso de descarga
            print(f'Downloading image {image_count}/{count} for {character_name}...')

            # Incrementar el contador de imagen
            image_count += 1

            # Comprobar si se ha alcanzado el número máximo de imágenes
            if image_count > count:
                return

    print(f'Downloaded {image_count-1} images for {character_name}.')


def RedNeuronal(data_folder):
    # Obtener la lista de carpetas (personajes) dentro de la carpeta de datos
    character_folders = os.listdir(data_folder)
    num_characters = len(character_folders)

    # Crear listas para almacenar las imágenes y las etiquetas correspondientes
    images = []
    labels = []

    # Leer las imágenes y las etiquetas de cada carpeta de personaje
    for character_index, character_folder in enumerate(character_folders):
        character_path = os.path.join(data_folder, character_folder)
        image_files = os.listdir(character_path)
        for image_file in image_files:
            image_path = os.path.join(character_path, image_file)
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image)
            labels.append(character_index)

    # Convertir las listas de imágenes y etiquetas en arreglos numpy
    images = np.array(images)
    labels = np.array(labels)

    # Dividir los datos en conjuntos de entrenamiento, prueba y validación
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Preprocesamiento de datos y aumento de datos para el conjunto de entrenamiento
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

    # Preprocesamiento de datos para los conjuntos de prueba y validación
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=32)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

    # Definir la arquitectura de la red neuronal
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_characters, activation='softmax')
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(train_generator, steps_per_epoch=len(X_train) // 32, epochs=10, validation_data=val_generator, validation_steps=len(X_val) // 32)

    # Evaluar el modelo con el conjunto de prueba
    test_loss, test_acc = model.evaluate(test_generator, verbose=2)

    # Imprimir la precisión de la clasificación en el conjunto de prueba
    print(f'Test accuracy: {test_acc}')

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(test_generator)
    y_pred = np.argmax(y_pred, axis=1)

    # Crear la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Mostrar el informe de clasificación
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Visualizar la precisión de entrenamiento y validación a lo largo de las épocas
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Visualizar la pérdida de entrenamiento y validación a lo largo de las épocas
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    return model


# Lista de personajes y sus respectivos enlaces
characters = {
    'Hakurei Reimu': 'https://safebooru.org/index.php?page=post&s=list&tags=hakurei_reimu+1girl+touhou',
    'Kirisame Marisa': 'https://safebooru.org/index.php?page=post&s=list&tags=kirisame_marisa+1girl+touhou',
    'Cirno': 'https://safebooru.org/index.php?page=post&s=list&tags=cirno+1girl+touhou',
    'Komeiji Koishi': 'https://safebooru.org/index.php?page=post&s=list&tags=komeiji_koishi+1girl+touhou',
    'Komeiji Satori': 'https://safebooru.org/index.php?page=post&s=list&tags=komeiji_satori+1girl+touhou',
    'Yakumo Yukari': 'https://safebooru.org/index.php?page=post&s=list&tags=yakumo_yukari+1girl+touhou',
    'Shameimaru Aya': 'https://safebooru.org/index.php?page=post&s=list&tags=shameimaru_aya+1girl+touhou',
    'Fujiwara No Mokou': 'https://safebooru.org/index.php?page=post&s=list&tags=fujiwara_no_mokou+1girl+touhou',
    'Houraisan Kaguya': 'https://safebooru.org/index.php?page=post&s=list&tags=houraisan_kaguya+1girl+touhou',
    'Rumia': 'https://safebooru.org/index.php?page=post&s=list&tags=rumia+1girl+touhou'
}

# Descargar las imágenes y generar los archivos CSV para cada personaje
for character, url in characters.items():
    print(f'Downloading images of {character}...')
    download_images(url, 5000, character)
    print()

# Convertir los datos JSON descargados a formato CSV
JSON2CSV(data, character_name)

# Ruta de la carpeta de datos
data_folder = './data'

# Entrenar una red neuronal con los datos descargados
model = RedNeuronal(data_folder)

# Distribución de las etiquetas en el conjunto de prueba
unique_labels, label_counts = np.unique(y_test, return_counts=True)
plt.bar(unique_labels, label_counts)
plt.title('Label Distribution in Test Set')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Correlación entre las etiquetas reales y las predicciones en el conjunto de prueba
correlation_matrix = np.corrcoef(y_test, y_pred)
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.colorbar()
plt.show()

# Descomposición de la serie de tiempo de precisión en el conjunto de prueba
time_series = history.history['accuracy']
plt.plot(time_series)
plt.title('Accuracy Time Series')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# División del conjunto de datos en entrenamiento, prueba y validación
num_characters = len(os.listdir(data_folder))
print(f'Number of characters: {num_characters}')
print(f'Total number of images: {image_count * num_characters}')
print(f'Number of images in training set: {len(X_train)}')
print(f'Number of images in test set: {len(X_test)}')
print(f'Number of images in validation set: {len(X_val)}')

# Sesgo vs Varianza
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, train_acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Alineación de hiperparámetros
model.summary()

# Selección de variables
features = ['height', 'width', 'rating']
X = df[features].values
y = df['character'].values

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='coolwarm')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Indicadores
accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_score}')

# Conclusión y Recomendación
print('Conclusion:')
print('The trained neural network achieved a test accuracy of', test_acc)
print('The model showed good performance in classifying the characters.')
print('Further improvements can be made by collecting more data and fine-tuning the hyperparameters.')

print('Recommendation:')
print('To improve the accuracy of the model, it is recommended to collect more diverse images of the characters.')
print('Additionally, exploring other neural network architectures and hyperparameter configurations may also lead to better results.')
