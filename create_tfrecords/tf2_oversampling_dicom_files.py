import random


def do_oversampling(df_gamma_in, gamma_threshold, oversampling_factor):
    N: int = len(df_gamma_in)

    # filtrar los mapas con gamma <= GAMMA_THRESHOLD, sea M la cantidad de estos mapas y N la cantidad de mapas
    # restantes
    df_gamma = df_gamma_in.copy()
    df_gamma = df_gamma[df_gamma['gamma_index'] <= gamma_threshold]
    # se asume que M << N
    M: int = len(df_gamma)
    N -= M
    print(f'Hay {M} mapas con un gamma menor o igual que {gamma_threshold} y {N} con un gamma mayor,'
          f' sobre un total de {len(df_gamma_in)} mapas.')
    # generar C = N * OVERSAMPLING_FACTOR - M nuevos mapas al azar tomados de N
    C = N * oversampling_factor - M
    print(f'Se van a hacer {int(C)} copias al azar de mapas con un gamma menor o igual que {gamma_threshold}.')

    # iterar C veces y en cada ciclo:
    for i in range(int(C)):
        # generar un número al azar m de 0 a M-1
        m = random.randrange(M)
        # incrementar en 1 el contador de copias cm realizada para el mapa m
        dicom_full_filepath = df_gamma.iloc[m]['dicom_full_filepath']
        gamma_index = df_gamma.iloc[m]['gamma_index']

        # agregar una fila al archivo de los gamma con el nuevo archivo y el gamma del archivo original
        df_gamma_in = df_gamma_in.append({'dicom_full_filepath': dicom_full_filepath,
                                          'gamma_index': gamma_index},
                                         ignore_index=True)

    print(f'Ovesampling completado, total de mapas {len(df_gamma_in)}.')
    return df_gamma_in




