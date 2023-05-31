import requests
import os
import json
import csv


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
