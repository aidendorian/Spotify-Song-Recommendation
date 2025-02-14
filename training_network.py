import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda
from tensorflow.keras.saving import register_keras_serializable
import pandas as pd

# Clustering separately saves time
df =  pd.read_csv("data/normalized_clustering.csv")

# Siamese Network
def build_network(input_shape):
    input_layer = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    return Model(input_layer, x)

# Generates pairs for siamese model training
def pair_generator(df, feature_columns, batch_size=1024, max_negative_samples=5):
    def generator():
        song1_list, song2_list, labels = [], [], []

        while True:
            for cluster_id in df["cluster"].unique():
                cluster_songs = df[df["cluster"] == cluster_id]
                non_cluster_songs = df[df["cluster"] != cluster_id]

                cluster_features = cluster_songs[feature_columns].values
                non_cluster_features = non_cluster_songs[feature_columns].values

                if len(cluster_songs) > 1:
                    for i in range(len(cluster_songs) - 1):
                        song1_list.append(cluster_features[i])
                        song2_list.append(cluster_features[i + 1])
                        labels.append(1)

                num_non_cluster_samples = min(max_negative_samples, len(non_cluster_songs))
                non_cluster_indices = np.random.choice(len(non_cluster_songs), num_non_cluster_samples, replace=False)

                for i in non_cluster_indices:
                    song1_list.append(cluster_features[0])
                    song2_list.append(non_cluster_features[i])
                    labels.append(0)

                if len(song1_list) >= batch_size:
                    song1_array = np.array(song1_list[:batch_size], dtype=np.float32)
                    song2_array = np.array(song2_list[:batch_size], dtype=np.float32)
                    label_array = np.array(labels[:batch_size], dtype=np.int8)

                    yield (song1_array, song2_array), label_array


                    song1_list, song2_list, labels = [], [], []

    output_signature = (
        (tf.TensorSpec(shape=(None, len(feature_columns)), dtype=tf.float32),
         tf.TensorSpec(shape=(None, len(feature_columns)), dtype=tf.float32)),
        tf.TensorSpec(shape=(None,), dtype=tf.int8)
    )

    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

# defining manually
feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'artist_id', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_10', 'key_11', 'mode_1', 'explicit_1']


train_generator = pair_generator(df = df,feature_columns = feature_columns)

# Parts defined here so I can use the embdedding layer separately
input_shape = (len(feature_columns))
network = build_network(input_shape)

input_1 = Input(shape=(input_shape,))
input_2 = Input(shape=(input_shape,))

embedding_1 = network(input_1)
embedding_2 = network(input_2)

# These two functions need to serialized
@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    return tf.math.sqrt(tf.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True) + 1e-10)

@register_keras_serializable()
def euclidean_distance_shape(input_shapes):
    shape1, shape2 = input_shapes
    return (shape1[0], 1)

distance = Lambda(euclidean_distance, output_shape=euclidean_distance_shape)([embedding_1, embedding_2])

siamese_model = Model(inputs=[input_1, input_2], outputs=distance)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)

# Binary cross-entropy worked better
siamese_model.compile(loss="binary_crossentropy", optimizer = optimizer)

# Training the model
siamese_model.fit(train_generator, steps_per_epoch=100, epochs=116)

siamese_model.save("models/siamese_model.keras")

val_generator = pair_generator(df, feature_columns, batch_size=1024)
siamese_model.evaluate(val_generator, steps=20)