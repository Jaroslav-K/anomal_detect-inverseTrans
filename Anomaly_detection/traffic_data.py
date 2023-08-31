import numpy as np
import pandas as pd
import tensorflow as tf

DATASETS = {
    "HOME": {
        "HEATMAPS": {
            "path": "/home/jaroslavk/coding/Heatmap_Data_Augmentation/data/home_heatmaps.csv",
            "sep": ",",
        },
        "CLUSTERS": {
            "path": "/home/jaroslavk/coding/Heatmap_Data_Augmentation/data/home_clusters.csv",
            "sep": ",",
        },
    },
    "SERVER": {
        "HEATMAPS": {
            "path": "/home/jaroslavk/coding/Heatmap_Data_Augmentation/data/server_heatmaps.csv",
            "sep": ",",
        },
        "CLUSTERS": {
            "path": "/home/jaroslavk/coding/Heatmap_Data_Augmentation/data/server_clusters.csv",
            "sep": ",",
        },
    },
}


def prepare_traffic(traffic_type, batch_size=32):

    if traffic_type not in DATASETS.keys():
        raise Exception("Invalid dataset name. Valid ones:", set(DATASETS.keys()))

    else:
        load_dataset(traffic_type)
        tensor_map = load_heatmaps(traffic_type)
        clusters = load_clusters(traffic_type)
        data = create_data(tensor_map, clusters, batch_size)

    return data, tensor_map, clusters


def load_dataset(traffic_type):
    valid_datasets = {
        "HEATMAPS": load_heatmaps(traffic_type),
        "CLUSTERS": load_clusters(traffic_type),
    }

    # return valid_datasets[name]


def load_heatmaps(traffic_type):
    path = DATASETS[traffic_type]["HEATMAPS"]["path"]
    sep = DATASETS[traffic_type]["HEATMAPS"]["sep"]
    df = pd.read_csv(path, sep=sep)
    np_df = df.to_numpy()

    """ process the .csv raw data 
    (nx2500, where 1row = 1heatmap) to  --> nx50x50 resolution """

    tensor_map = None

    for i in range(np_df.shape[0]):
        heat_map = tf.reshape(np_df[i], (1, 50, 50, 1))
        if tensor_map == None:
            tensor_map = heat_map
        else:
            # tensor_map in nx50x50 format
            tensor_map = tf.concat((tensor_map, heat_map), axis=0)

    tensor_map = tf.cast(tensor_map, "float32")

    assert tensor_map.shape == (tensor_map.shape[0], 50, 50, 1)
    return tensor_map


def load_clusters(traffic_type):
    path = DATASETS[traffic_type]["CLUSTERS"]["path"]
    sep = DATASETS[traffic_type]["CLUSTERS"]["sep"]
    df = pd.read_csv(path, sep=sep)
    clusters = df.to_numpy()

    clusters = tf.cast(clusters, "float32")

    assert clusters.shape == (clusters.shape[0], 1)
    return clusters


def create_data(data, labels, batch_size):
    return tf.data.Dataset.from_tensor_slices((data, labels)).batch(
        batch_size, drop_remainder=False
    )
