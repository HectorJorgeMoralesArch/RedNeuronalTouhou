import scv
import request
import os
import json
import pandas as pd

def JSON2SCV(data):
    # Definir la ruta y nombre de archivo
    filename = 'archivo.csv'
    # Comprobar si el archivo CSV ya existe
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Escribir los encabezados
            writer.writerow(['directory', 'hash', 'height', 'id', 'image', 'change', 'owner', 'parent_id', 'rating', 'sample', 'sample_height', 'sample_width', 'score', 'tags', 'width'])
    with open(filename, 'a', newline='') as f:
        writer = scv.writer(f)
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
            #print(directory, "\n", hash, "\n", height, "\n", id, "\n", image, "\n", change, "\n", owner, "\n", parent_id, "\n", rating, "\n", sample, "\n", sample_height, "\n", sample_width, "\n", score, "\n", tag, "\n", width, "\n")
            tag = tag.replace(' ', ', ')
            # Escribir la fila en el archivo CSV
            writer.writerow([directory, hash, height, id, image, change, owner, parent_id, rating, sample, sample_height, sample_width, score, tag, width])
    # Cerrar el archivo
    f.close()
def descargar():
    # Definir los parámetros de búsqueda
    tags = 'Touhou'  # las etiquetas para buscar
    limit = 100
    pages = 100