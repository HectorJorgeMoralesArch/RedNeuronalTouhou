import requests
import os
import json
import csv
from json.decoder import JSONDecodeError

def download_image(url, directory, filename):
    response = requests.get(url)
    if response.status_code == 200:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'wb') as file:
            file.write(response.content)

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
    download_images(url, 5000, character)
    print()
