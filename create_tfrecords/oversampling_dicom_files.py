from os import path, remove, mkdir
import random
import shutil
import pandas as pd
import numpy as np

GAMMA_THRESHOLD = 99.7 # percentage
OVERSAMPLING_FACTOR = 0.7 # percentage
SEED = 12351

dataset_dir = "/hdd/data/radioterapia/ciolaplata"
source_folder = "2019"
dest_folder = "2019.ovs" # create manually
gamma_filename="codex.2019.corregidos.2.csv"
oversampled_gamma_filename = 'codex.2019.oversampled.csv'
try:
    remove(path.join(dataset_dir, oversampled_gamma_filename))
    print(f'Removing directory {path.join(dataset_dir, dest_folder)}...')
    shutil.rmtree((path.join(dataset_dir, dest_folder)))
    print(f'Directory removed.')
except FileNotFoundError:
    pass
df_gamma_in = pd.read_csv(path.join(dataset_dir, gamma_filename), sep=';', delimiter=None, header=0)
df_gamma_in['oversampled'] = 0 # agrego flag indicando si este mapa es una copia por oversampling
N: int = len(df_gamma_in)

# filtrar los mapas con gamma <= GAMMA_THRESHOLD, sea M la cantidad de estos mapas y N la cantidad de mapas restantes
df_gamma = df_gamma_in.copy()
df_gamma['gamma_index'] = df_gamma.apply(lambda row: 100.0 - row['uno menos gamma index'], axis=1)
df_gamma = df_gamma[df_gamma['gamma_index'] <= GAMMA_THRESHOLD]
# se asume que M << N
M: int = len(df_gamma)
N -= M
print(f'Hay {M} mapas con un gamma menor o igual que {GAMMA_THRESHOLD} y {N} con un gamma mayor,'
      f' sobre un total de {len(df_gamma_in)} mapas.')
# generar C = N * OVERSAMPLING_FACTOR - M nuevos mapas al azar tomados de N
C = N * OVERSAMPLING_FACTOR - M
print(f'Se van a hacer {int(C)} copias al azar de mapas con un gamma menor o igual que {GAMMA_THRESHOLD}.')
# no hace falta copiar el directorio source, los mapas no oversampled se toman directamente del directorio source.
# shutil.copytree(path.join(dataset_dir, source_folder), path.join(dataset_dir, dest_folder))
mkdir(path.join(dataset_dir, dest_folder))

# para los mapas M inicializar en cero el contador de copias realizadas
df_gamma['copies'] = 0
# iterar C veces y en cada ciclo:
random.seed(SEED) # Initialize the random number generator.
for i in range(int(C)):
    #    generar un número al azar m de 0 a M-1
    m = random.randrange(M)
    #    incrementar en 1 el contador de copias cm realizada para el mapa m
    año = df_gamma.iloc[m]['año']
    fc = df_gamma.iloc[m]['fluencia calculada']
    fm = df_gamma.iloc[m]['fluencia medida']
    g = df_gamma.iloc[m]['uno menos gamma index']
    row_index = df_gamma[df_gamma['fluencia calculada'] == fc].index[0]
    df_gamma.at[row_index, 'copies'] += 1
    c = df_gamma.at[row_index, 'copies']

    #    hacer una copia del archivo .dcm con el nombre <nombre>.<m>.dcm
    fc_new = f'{fc}-{c}'
    existing = f'{fc}.dcm'
    copy = f'{fc_new}.dcm'
    shutil.copy2(path.join(dataset_dir, source_folder, existing), path.join(dataset_dir, dest_folder, copy))

    #    agregar una fila al archivo de los gamma con el nuevo archivo y el gamma del archivo original
    df_gamma_in = df_gamma_in.append({'año': dest_folder, 'fluencia calculada': fc_new, 'fluencia medida': fm,
                                      'uno menos gamma index': g, 'oversampled': 1}, ignore_index=True)
# guardar el archivo gamma aumentado.
df_gamma_in.to_csv(path.join(dataset_dir, oversampled_gamma_filename), sep=';', index=False, float_format='%.3f')
print(f'{int(C)} files were created at {path.join(dataset_dir, dest_folder)}'
      f' along with {path.join(dataset_dir, oversampled_gamma_filename)} file,'
      f' total of files is {N + M + int(C)}.')




