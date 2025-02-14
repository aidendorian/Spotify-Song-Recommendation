import tensorflow as tf
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Lambda
from keras.layers import Dense, Input, Dropout, Lambda
from tensorflow.keras.saving import register_keras_serializable
import pandas as pd

# Defining all these here again so that this recommendation model can work anywhere readily
def build_network(input_shape):
    input_layer = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    return Model(input_layer, x)

# routine stuff
df = pd.read_csv("data/normalized_clustered.csv", index_col =[0])
feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'artist_id', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_10', 'key_11', 'mode_1', 'explicit_1']

# again
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

train_generator = pair_generator(df = df,feature_columns = feature_columns)

input_shape = (len(feature_columns))
network = build_network(input_shape)

input_1 = Input(shape=(input_shape,))
input_2 = Input(shape=(input_shape,))

embedding_1 = network(input_1)
embedding_2 = network(input_2)

# Serializing didn't work, but doing it regardless
@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    return tf.math.sqrt(tf.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True) + 1e-10)

@register_keras_serializable()
def euclidean_distance_shape(input_shapes):
    shape1, shape2 = input_shapes
    return (shape1[0], 1)

distance = Lambda(euclidean_distance, output_shape=euclidean_distance_shape)([embedding_1, embedding_2])

# custom objects could not be serialized so had to define them again 
siamese_model = load_model("models/siamese_model.keras", custom_objects={
    "euclidean_distance": euclidean_distance,
    "euclidean_distance_shape": euclidean_distance_shape
})

# Example song IDs, add or remove as per your need
liked_song_ids = ["60a0Rd6pjrkxjPbaKzXjfq", "7oVEtyuv9NBmnytsCIsY5I", "3Ofmpyhv5UAQ70mENzB277", "7qiZfU4dY1lWllzX7mPBI3",
                  "1rfofaqEpACxVEHIZBJe6W", "3UYiU57SMiAS5LqolhHJw1", "5GorCbAP4aL0EJ16frG2hd", "6SRWhUJcD2YKahCwHavz3X",
                  "7GVUmCP00eSsqc4tzj1sDD", "58zsLZPvfflaiIbNWoA22O"]

# Recommendor Function
def recommend_songs(user_songs, all_songs, embedding_model, top_n=20):
    user_embeddings = embedding_model.predict(user_songs)
    all_embeddings = embedding_model.predict(all_songs)

    all_distances = []
    for user_embedding in user_embeddings:
        distances = np.linalg.norm(all_embeddings - user_embedding, axis=1)
        all_distances.append(distances)

    average_distances = np.mean(all_distances, axis=0)

    recommended_indices = np.argsort(average_distances)[:top_n]

    return list(df.iloc[recommended_indices]["track_name"])

# Using only the embedding layer of the Siamese Network
embedding_model = Model(network.input, network.output)

# Features of the User-Liked songs
user_songs = df.loc[df["track_id"].isin(liked_song_ids)][feature_columns].values

recommendations = recommend_songs(user_songs, df[~df["track_id"].isin(liked_song_ids)][feature_columns].values, embedding_model)
print("Recommended Songs:")
print(*recommendations, sep='\n')