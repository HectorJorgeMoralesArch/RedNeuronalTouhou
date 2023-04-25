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
        