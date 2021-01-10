import random


def do_oversampling(df_gamma_in, gamma_threshold, oversampling_factor):
    total_maps: int = len(df_gamma_in)

    # filtrar los mapas con gamma <= GAMMA_THRESHOLD, sea count_low_gamma la cantidad de estos mapas
    df_low_gamma = df_gamma_in.copy()
    df_low_gamma = df_low_gamma[df_low_gamma['gamma_index'] <= gamma_threshold]
    count_low_gamma: int = len(df_low_gamma)
    print(f'Hay {count_low_gamma} mapas con un gamma menor o igual que {gamma_threshold} y {total_maps-count_low_gamma} con un gamma mayor,'
          f' sobre un total de {total_maps} mapas.')
    count_new_maps = count_low_gamma * oversampling_factor
    print(f'Oversampling factor es {oversampling_factor:.2f}, se van a hacer {int(count_new_maps)} copias al azar de mapas con un gamma menor o igual que {gamma_threshold}.')

    for i in range(int(count_new_maps)):
        # generar un número al azar m en [0, count_low_gamma-1] para elegir el mapa a duplicar
        m = random.randrange(count_low_gamma)
        # obtengo el file name y el gamma index del mapa a 'copiar'
        dicom_full_filepath = df_low_gamma.iloc[m]['dicom_full_filepath']
        gamma_index = df_low_gamma.iloc[m]['gamma_index']

        # agregar una fila al archivo de los gamma con el nuevo file name y el gamma index del archivo original
        df_gamma_in = df_gamma_in.append({'dicom_full_filepath': dicom_full_filepath,
                                          'gamma_index': gamma_index},
                                         ignore_index=True)

    print(f'Ovesampling completado, total de mapas despues del oversampling: {len(df_gamma_in)}.')
    return df_gamma_in




