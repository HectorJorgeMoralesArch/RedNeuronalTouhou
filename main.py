import scv
import request
import os
import json
import pandas as pd

def descargar():
    # Definir los parámetros de búsqueda
    tags = 'Touhou'  # las etiquetas para buscar
    limit = 100
    pages = 100