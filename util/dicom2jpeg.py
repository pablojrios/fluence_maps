import numpy as np
import os, pydicom
from PIL import Image # https://pillow.readthedocs.io/en/stable/index.html
import cv2


source_folder = '/home/pablo/datasets/data/radioterapia/ciolaplata/2017'
# Requiere tener creado el directorio de salida, 2019-jpeg en este caso
destination_folder = '/home/pablo/datasets/data/radioterapia/ciolaplata/2017-opencv-jpeg'

USE_OPENCV = True # default: True
if USE_OPENCV:
    print("Image files will be written with OpenCV version {}.".format(cv2.__version__))
else:
    print("Image files will be written with Pillow.")


def dicom2jpeg(source_folder, output_folder):
    """Convierte todos los archivos DICOM en un folder a imágenes JPEG.

    Convierte todos los archivos .dcm en formato de archivo DICOM que están en un folder de entrada,
    a archivos .jpeg en formato JPEG a un folder de salida.
    Los archivos DICOM tienen que tener un unica imágen en la parte de los datos.

    Parameters
    ----------
    source_folder : str
        Folder con los archivos .dcm a convertir.
    output_folder : str
        Folder en donde se van a escribir los archivos .jpeg resultantes de la conversión.

    Returns
    -------
        Nothing

    Raises
    ------
        No exception is raised

    """
    list_of_files = os.listdir(source_folder)
    print("Hay {} archivos dicom en {}.".format(len(list_of_files), source_folder))
    for file in list_of_files:
        try:
            ds = pydicom.dcmread(os.path.join(source_folder, file))
            shape = ds.pixel_array.shape  # ds.pixel_array.dtype is dtype('uint16')
            assert len(shape) == 2

            # Convert to float to avoid overflow or underflow losses.
            image_2d = ds.pixel_array.astype(float)

            # To avoid RuntimeWarning: invalid value encountered in true_divide
            # See https://stackoverflow.com/questions/15933741/how-do-i-catch-a-numpy-warning-like-its-an-exception-not-just-for-testing
            with np.errstate(invalid='raise'):
                try:
                    # Rescaling grey scale between 0-255 (0-32767 para JPEG 2000)
                    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
                except FloatingPointError as err:
                    print('Could not convert file {}, error: {}'.format(file, err))
                    # desn't continue with current file conversion
                    continue

            # Convert to uint (convert to uint16 para JPEG 2000)
            image_2d_scaled = np.uint8(image_2d_scaled + 0.5)

            # remuevo .dcm del nombre del archivo
            if file.endswith('.dcm'):
                file = file[:-4]

            # Write the JPEG (.jpeg) or JP2 file (.jp2)
            filename = os.path.join(output_folder, file)+'.jpeg'
            if USE_OPENCV:
                # Save the image using OpenCV
                cv2.imwrite(filename, image_2d_scaled, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            else:
                with open(filename, 'wb') as jpeg_file:
                    # Save the image using Pillow
                    im = Image.fromarray(image_2d_scaled)
                    im.save(jpeg_file)

        except OSError as err:
            print('Could not convert file {}, error: {}'.format(file, err))


dicom2jpeg(source_folder, destination_folder)

# assert que en output_folder tenga la misma cantidad de imagenes que en source folder,
# es decir validar que todos los archivos dicom se hayan convertido.
print("{} archivos dicom se convirtieron a jpeg en {}.".format(len(os.listdir(destination_folder)), destination_folder))
assert len(os.listdir(source_folder)) == len(os.listdir(destination_folder)),\
    "No todos los archivos dicom se pudieron convertir a jpeg."
