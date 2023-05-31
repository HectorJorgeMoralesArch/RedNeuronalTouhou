import requests
import os
import json
import csv

def download_image(url, directory, filename):
    response = requests.get(url)
    if response.status_code == 200:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'wb') as file:
            file.write(response.content)

def download_images(tag, count, character_name):
    url = 'https://safebooru.org/index.php'
    params = {
        'page': 'dapi',
        's': 'post',
        'q': 'index',
        'limit': count,
        'tags': f'{tag} -{character_name}',
        'json': 1
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
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
    else:
        print(f'Error downloading images of {character_name}.')

# Lista de personajes y sus respectivas etiquetas
characters = {
    'Hakurei Reimu': 'reimu_hakurei',
    'Kirisame Marisa': 'kirisame_marisa',
    'Cirno': 'cirno',
    'Komeiji Koishi': 'komeiji_koishi',
    'Komeiji Satori': 'komeiji_satori',
    'Yakumo Yukari': 'yakumo_yukari',
    'Shameimaru Aya': 'shameimaru_aya',
    'Fujiwara No Mokou': 'fujiwara_no_mokou',
    'Houraisan Kaguya': 'houraisan_kaguya',
    'Rumia': 'rumia'
}

# Descargar las imágenes y generar los archivos CSV para cada personaje
for character, tag in characters.items():
    print(f'Downloading images of {character}...')
    download_images(tag, 5000, character)
    print()
