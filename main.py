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
