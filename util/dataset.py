{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "tf2_evaluate",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyM3TGr4+MgDRgVfSsH1cGxL"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-n_PAVFGnZDx"
      },
      "source": [
        "# Compute predictions on a TF dataset using an stored .h5 model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "A85nWqMEnKPB",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4ce4ac89-bd62-4ec1-d29a-d81ec30e826d"
      },
      "source": [
        "#%cd /content/drive/My\\ Drive/Healthcare/Radioterapia/data/ciolaplata\n",
        "#!unrar x /content/drive/My\\ Drive/Healthcare/Radioterapia/Mapas\\ CIO\\ La\\ Plata/Mapas\\ Calculados/2019.rar\n",
        "#!ls -l 2015/*dcm | wc -l\n",
        "\n",
        "import os\n",
        "\n",
        "%cd -q '/content'\n",
        "if os.path.exists('fluence_maps'):\n",
        "  !rm -fr fluence_maps\n",
        "\n",
        "GIT_USERNAME = \"pablojrios\"\n",
        "GIT_TOKEN = \"1d88a0b85d2b00a03796e4d8b7e5f7b249b12f9b\"\n",
        "!git clone -s https://{GIT_TOKEN}@github.com/{GIT_USERNAME}/fluence_maps.git"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Cloning into 'fluence_maps'...\n",
            "remote: Enumerating objects: 228, done.\u001b[K\n",
            "remote: Counting objects: 100% (228/228), done.\u001b[K\n",
            "remote: Compressing objects: 100% (189/189), done.\u001b[K\n",
            "remote: Total 228 (delta 132), reused 92 (delta 38), pack-reused 0\u001b[K\n",
            "Receiving objects: 100% (228/228), 1.53 MiB | 10.62 MiB/s, done.\n",
            "Resolving deltas: 100% (132/132), done.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4LvLPLr5n_yJ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "99642daa-a34a-49d4-d69e-ecec40dea8fe"
      },
      "source": [
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "%cd '/content/fluence_maps'\n",
        "from util.dataset import _tfrecord_dataset_type_from_folder, _parse_jpeg_image_function\n",
        "from util.preprocess import rescale_0_to_1\n",
        "import os\n",
        "import pandas as pd"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/content/fluence_maps\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LN_LGyd6pWeZ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "12d7b8a0-1b03-43f2-aebe-48e0e920df6c"
      },
      "source": [
        "print('Tensorflow version = {}'.format(tf.__version__))\n",
        "print('Executing eagerly = {}'.format(tf.executing_eagerly()))"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Tensorflow version = 2.4.0\n",
            "Executing eagerly = True\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pzL176Gmq_KA",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "bf934433-19d6-4e32-e645-3151cbea2615"
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "doeibFQqsoHJ",
        "outputId": "24e02697-2847-4535-b8d5-5ae17e6570b7",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "!ls -l '/content/drive/My Drive/Healthcare/Radioterapia/data/ciolaplata/models'"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "total 5628284\n",
            "-rw------- 1 root root 204753152 May  6  2020 1588803713.h5\n",
            "-rw------- 1 root root 204752992 May  8  2020 1588961914.h5\n",
            "-rw------- 1 root root 204752992 Jun  8  2020 1591581915.h5\n",
            "-rw------- 1 root root 204753488 Jan  3 19:04 1609700652.h5\n",
            "-rw------- 1 root root 233090768 Jan  3 19:22 1609701766.h5\n",
            "-rw------- 1 root root 119785520 Jan  3 19:30 1609702231.h5\n",
            "-rw------- 1 root root 119785520 Jan  3 19:39 1609702768.h5\n",
            "-rw------- 1 root root  52148296 Jan  3 20:25 1609705507.h5\n",
            "-rw------- 1 root root  55218296 Jan  3 20:36 1609706214.h5\n",
            "-rw------- 1 root root  44420192 Jan  3 20:49 1609706993.h5\n",
            "-rw------- 1 root root 204753488 Jan  3 21:04 1609707882.h5\n",
            "-rw------- 1 root root 156500328 Jan  3 21:16 1609708579.h5\n",
            "-rw------- 1 root root 204753488 Jan  3 21:32 1609709567.h5\n",
            "-rw------- 1 root root 204754632 Jan  3 21:45 1609710337.h5\n",
            "-rw------- 1 root root 204754632 Jan  3 21:54 1609710864.h5\n",
            "-rw------- 1 root root 204754632 Jan  3 22:01 1609711277.h5\n",
            "-rw------- 1 root root 204753488 Jan  3 22:12 1609711934.h5\n",
            "-rw------- 1 root root 204753488 Jan  3 22:34 1609713262.h5\n",
            "-rw------- 1 root root 204753488 Jan  4 02:21 1609726904.h5\n",
            "-rw------- 1 root root 204753488 Jan  4 02:41 1609728119.h5\n",
            "-rw------- 1 root root 156500040 Jan  6 19:10 1609960235.h5\n",
            "-rw------- 1 root root 132221552 Jan  6 19:22 1609960977.h5\n",
            "-rw------- 1 root root 132221552 Jan  6 23:52 1609977122.h5\n",
            "-rw------- 1 root root 132221552 Jan  9 19:13 1610219616.h5\n",
            "-rw------- 1 root root 132221552 Jan  9 19:38 1610221139.h5\n",
            "-rw------- 1 root root 132221552 Jan  9 21:10 1610226607.h5\n",
            "-rw------- 1 root root 132221848 Jan  9 21:21 1610227264.h5\n",
            "-rw------- 1 root root 132221552 Jan  9 21:31 1610227879.h5\n",
            "-rw------- 1 root root 132221552 Jan  9 21:38 1610228325.h5\n",
            "-rw------- 1 root root 132221552 Jan  9 21:58 1610229513.h5\n",
            "-rw------- 1 root root 132221552 Jan  9 23:08 1610233693.h5\n",
            "-rw------- 1 root root 132221848 Jan  9 23:33 1610235196.h5\n",
            "-rw------- 1 root root 132221552 Jan  9 23:59 1610236790.h5\n",
            "-rw------- 1 root root 132221552 Jan 10 03:09 1610248186.h5\n",
            "-rw------- 1 root root 156500040 Jan 10 03:38 1610249880.h5\n",
            "-rw------- 1 root root 132221552 Jan 10 18:02 1610301746.h5\n",
            "-rw------- 1 root root 156500040 Jan 16 00:07 1610755621.h5\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eWlKtT6Kpbsr"
      },
      "source": [
        "#============================DEFINE YOUR ARGUMENTS==============================\n",
        "# base data directory\n",
        "ARG_DATASET_DIR='/content/drive/My Drive/Healthcare/Radioterapia/data/ciolaplata'\n",
        "# folder under ARG_DATASET_DIR path.\n",
        "ARG_TFDATASET_FOLDER='tfds.2019.localnorm'\n",
        "ARG_MODEL_NAME = '1610755621'\n",
        "# 'train', 'validation', 'test' \n",
        "ARG_PART = 'train'\n",
        "ARG_TRANSFORM_GAMMA=True"
      ],
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "K2-IdOEGqmU1",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "37d6508e-85ea-41aa-8693-9cb383fbbaf7"
      },
      "source": [
        "tfdataset_dir = os.path.join(ARG_DATASET_DIR, ARG_TFDATASET_FOLDER)\n",
        "raw_test = _tfrecord_dataset_type_from_folder(tfdataset_dir, ARG_PART)\n",
        "print(raw_test)"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<TFRecordDatasetV2 shapes: (), types: tf.string>\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tfTOOGlJrYIr",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 262
        },
        "outputId": "cddae96e-7635-4254-9c52-55b4c606c636"
      },
      "source": [
        "# Apply this function to each item in the dataset using the map method:\n",
        "num_workers = 8\n",
        "IMG_SIZE = 256\n",
        "normalization_fn = rescale_0_to_1\n",
        "test = raw_test.map(lambda e: _parse_jpeg_image_function(e, IMG_SIZE, normalization_fn), num_parallel_calls=num_workers, transform_gamma=ARG_TRANSFORM_GAMMA)\n",
        "print(test)"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "error",
          "ename": "TypeError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-14-3d7412794c76>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mIMG_SIZE\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m256\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mnormalization_fn\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mrescale_0_to_1\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m \u001b[0mtest\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mraw_test\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmap\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;32mlambda\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0m_parse_jpeg_image_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mIMG_SIZE\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnormalization_fn\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_parallel_calls\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnum_workers\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtransform_gamma\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mARG_TRANSFORM_GAMMA\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      6\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtest\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mTypeError\u001b[0m: map() got an unexpected keyword argument 'transform_gamma'"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3yG0Vf93rmOw",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6df1a1f8-fcf4-47d0-b1e1-6b4bd40950e2"
      },
      "source": [
        "gamma_values = test.map(lambda image, gamma, filename: gamma)\n",
        "gamma_values = np.array(list(gamma_values.as_numpy_iterator()))\n",
        "BATCH_SIZE = 32 # mae puede variar según batch size.\n",
        "test_batches = test.batch(BATCH_SIZE)\n",
        "print(test_batches)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<BatchDataset shapes: ((None, 256, 256, 3), (None,), (None,)), types: (tf.float32, tf.float32, tf.string)>\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tjrt80mNr0a4",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c7e6f6ec-2db2-45a3-aff8-d68d47f6cc49"
      },
      "source": [
        "# load model\n",
        "dir = os.path.join(ARG_DATASET_DIR, \"models\")\n",
        "saved_model_dir = '{}/{}.h5'.format(dir, ARG_MODEL_NAME)\n",
        "print(f'Loading model {saved_model_dir}...')\n",
        "loaded_model = tf.keras.models.load_model(saved_model_dir)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Loading model /content/drive/My Drive/Healthcare/Radioterapia/data/ciolaplata/models/1610229513.h5...\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZEaSI7qj86Um",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "28f11d24-092a-4586-ad6c-04cf562cc182"
      },
      "source": [
        "# Evaluate dataset with the loaded model to calculate loss (mae) because\n",
        "# metric value could differ from the one reported during training.\n",
        "tmp_test_batches = test_batches.map(lambda image, gamma, filename: (image, gamma))\n",
        "print(tmp_test_batches)\n",
        "loss, mse = loaded_model.evaluate(tmp_test_batches, workers=num_workers, verbose=0)\n",
        "print('\\n\\nLoaded model, test loss: {:5.4f}'.format(loss))\n",
        "print('Loaded model, test mse: {:5.4f}'.format(mse))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<MapDataset shapes: ((None, 256, 256, 3), (None,)), types: (tf.float32, tf.float32)>\n",
            "\n",
            "\n",
            "Loaded model, test loss: 0.8352\n",
            "Loaded model, test mse: 7.9495\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ctN7LhzY9I3n"
      },
      "source": [
        "# Make predictions\n",
        "predictions = loaded_model.predict(tmp_test_batches)\n",
        "# from (1121,1) to (1121,); ie.: ndim = 2 to ndim = 1\n",
        "predictions = predictions.reshape(-1)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NkoSrJyh9jtR",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "04381cb6-7f86-4e11-99f9-8603c0b23fa3"
      },
      "source": [
        "tmp_test_batches = test.map(lambda image, gamma, filename: (filename, gamma))\n",
        "lst = [(filename.numpy().decode('utf-8'), gamma.numpy()) for filename, gamma in tmp_test_batches]\n",
        "lst2 = [(e[0], e[1], p) for e, p in zip(lst, predictions)]\n",
        "\n",
        "# armar un pandas dataframe con el test set completo\n",
        "df = pd.DataFrame(lst2, columns=['filename', 'actual gamma', 'predicted gamma'])\n",
        "dir = os.path.join(ARG_DATASET_DIR, \"predictions\")\n",
        "predictions_file_path = '{}/predicted_gamma_{}.{}.csv'.format(dir, ARG_MODEL_NAME, ARG_PART)\n",
        "df.to_csv(predictions_file_path, index=False)\n",
        "print(f'Predictions saved to {predictions_file_path}.')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Predictions saved to /content/drive/My Drive/Healthcare/Radioterapia/data/ciolaplata/predictions/predicted_gamma_1610229513.train.csv.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1Zey_ms0PZK4",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "764205c8-92ca-4178-cad1-212b2a1e3d46"
      },
      "source": [
        "drive.flush_and_unmount()\n",
        "print('All changes made in this colab session should now be visible in Drive.')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "All changes made in this colab session should now be visible in Drive.\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}