import requests
import os
import json
import csv
from json.decoder import JSONDecodeError
def JSON2SCV(data,Character):
    # Definir la ruta y nombre de archivo
    filename = 'archivo.csv'
    # Comprobar si el archivo SCV ya existe
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Escribir los encabezados
            writer.writerow(['directory', 'hash', 'height', 'id', 'image', 'change', 'owner', 'parent_id', 'rating', 'sample', 'sample_height', 'sample_width', 'score', 'tags', 'width','character'])
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for item in data:
            # Obtener los valores de cada campo
            directory = item['directory']
            hash = item['hash']
            height = item['height']
            id = item['id']
            image = item['image']
            change = item['change']
            owner = item['owner']
            parent_id = item['parent_id']
            rating = item['rating']
            sample = item['sample']
            sample_height = item['sample_height']
            sample_width = item['sample_width']
            score = item['score']
            tag = item['tags']
            width = item['width']
            character = Character
            
            #print(directory, "\n", hash, "\n", height, "\n", id, "\n", image, "\n", change, "\n", owner, "\n", parent_id, "\n", rating, "\n", sample, "\n", sample_height, "\n", sample_width, "\n", score, "\n", tag, "\n", width, "\n")
            # Reemplazar los espacios en blanco en la cadena de tags con comas
            tag = tag.replace(' ', ', ')
            # Escribir la fila en el archivo SCV
            writer.writerow([directory, hash, height, id, image, change, owner, parent_id, rating, sample, sample_height, sample_width, score, tag, width])
    # Cerrar el archivo
    f.close()
def descargar(url, images, character):
    # Definir los parámetros de búsqueda
    limit = 100    # el número máximo de resultados por página
    pages = 50    # el número total de páginas a descargar

    # Inicializar el contador de imagen
    image_count = 1
    # Iterar a través de las páginas y descargar las imágenes
    for page in range(1, pages + 1):
        # Realizar la solicitud a la API de Safebooru para la página actual
        response = requests.get(f'{url}&limit={limit}&pid={page}&json=1')
        # Analizar la respuesta JSON
        data = json.loads(response.content)
        # Descargar cada imagen en la página actual
        for item in data:
            # Guardar la respuesta JSON en un archivo
            with open(f'safebooru_{image_count}.json', 'w') as f:
                json.dump(item, f)
            # Obtener la información de la imagen
            id=item['id']
            image_url = f'https://safebooru.org//images/{item["directory"]}/{item["image"]}?{item["id"]}'
            #owner = item['owner'] # el nombre del artista
            #tagsP = item['tags']# las etiquetas de la imagen
            # Descargar la imagen
            response = requests.get(image_url, stream=True)
            with open(f'{item["image"]}', 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            # Imprimir la información de la imagen
            print(f'Imagen {image_count}:')
            #print(f'Artista: {owner}')
            #print(f'Etiquetas: {tagsP}')
            #print(f'URL: {image_url}\n')
            JSON2SCV(data,character)# Incrementar el contador de imagen
            image_count += 1

def download_images(url, count, character_name):
    params = {
        'page': 'dapi',
        's': 'post',
        'q': 'index',
        'limit': count,
        'json': 1
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        try:
            data = json.loads(response.text)
            directory = f'{character_name}_images'
            os.makedirs(directory, exist_ok=True)

            with open(f'{character_name}.json', 'w') as file:
                json.dump(data, file)

            csv_data = []
            for i, item in enumerate(data, 1):
                image_url = item['file_url']
                image_name = item['image']
                artist = item['artist']
                tags = ', '.join(tag for tag in item['tags'].split() if tag not in [character_name, artist])
                download_image(image_url, directory, image_name)
                csv_data.append([character_name, image_name, artist, tags])

                progress = i / count * 100
                print(f'Progress: {progress:.2f}% ({i}/{count})', end='\r')

            with open(f'{character_name}.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Character', 'Image Name', 'Artist', 'Tags'])
                writer.writerows(csv_data)

            print(f'\n{count} images of {character_name} downloaded and CSV file generated.')

            # Agregar los datos al archivo CSV general
            filename = 'archivo.csv'
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                for item in data:
                    directory = item['directory']
                    hash_value = item['hash']
                    height = item['height']
                    image_id = item['id']
                    image_name = item['image']
                    change = item['change']
                    owner = item['owner']
                    parent_id = item['parent_id']
                    rating = item['rating']
                    sample = item['sample']
                    sample_height = item['sample_height']
                    sample_width = item['sample_width']
                    score = item['score']
                    tags = ', '.join(tag for tag in item['tags'].split() if tag not in [character_name, artist])
                    width = item['width']
                    writer.writerow([directory, hash_value, height, image_id, image_name, change, owner, parent_id, rating, sample, sample_height, sample_width, score, tags, width])

        except JSONDecodeError:
            print(f'Error decoding JSON response for {character_name}.')
    else:
        print(f'Error downloading images of {character_name}.')

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
    descargar(url, 5000, character)
    print()
