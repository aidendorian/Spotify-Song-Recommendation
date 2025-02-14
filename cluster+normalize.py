import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import hdbscan

# Data from Spotify, can't be extracted anymore because of API changes
# Change the database as you see fit but the features should be same
df = pd.read_csv('data/raw_data.csv')

# scaling different parameters separately
numerical_features_standard = ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']
numerical_features_robust = ["loudness", "tempo"]
categorical_features = ["key", "mode", "explicit"]

oneHot_encoder = OneHotEncoder(sparse_output=False, drop="first")
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()
label_encoder = LabelEncoder()

df[numerical_features_standard] = standard_scaler.fit_transform(df[numerical_features_standard])
df[numerical_features_robust] = robust_scaler.fit_transform(df[numerical_features_robust])
encoded_array = oneHot_encoder.fit_transform(df[categorical_features])
df["artist_id"] = label_encoder.fit_transform(df["artists"])

encoded_df = pd.DataFrame(encoded_array, columns=oneHot_encoder.get_feature_names_out(categorical_features))
df = df.drop(columns=categorical_features)
df = pd.concat([df, encoded_df], axis=1)

# this will used throughout the project
feature_columns = numerical_features_standard + numerical_features_robust + list(encoded_df.columns) + ["artist_id"]

#  Hierarchical Density-Based Spatial Clustering of Applications with Noise
# "leaf" performing better than "eom"
hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=64,
        metric="euclidean",
        min_samples=10,
        gen_min_span_tree=True,
        cluster_selection_method="leaf",
        cluster_selection_epsilon = 0.5
    )

df["cluster"] = hdbscan_clusterer.fit_predict(df[feature_columns])

# Check for Outliers and the number of clusters
print(f"Number of clusters: {len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)}")
print(f"Cluster sizes:\n{df['cluster'].value_counts()}")

# Was getting a substantial number of outliers so using this to assign outliers to clusters
# Outliers Marked as -1
def assign_outliers_to_clusters(df, feature_columns):
    outliers = df[df["cluster"] == -1]
    clusters = df[df["cluster"] != -1]

    knn = NearestNeighbors(n_neighbors=1).fit(clusters[feature_columns])
    distances, indices = knn.kneighbors(outliers[feature_columns])

    closest_clusters = clusters.iloc[indices.flatten()]["cluster"].values
    df.loc[df["cluster"] == -1, "cluster"] = closest_clusters

assign_outliers_to_clusters(df, feature_columns)

# No outliers after this
print(f"Number of clusters: {len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)}")
print(f"Cluster sizes:\n{df['cluster'].value_counts()}")

# Evaluate the quality of Clustering
# Make changes as need to the hdbscan parameters if not satisfied
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

sil_score = silhouette_score(df[feature_columns], df["cluster"])
print(f"📊 Silhouette Score: {sil_score:.4f}")
db_score = davies_bouldin_score(df[feature_columns], df["cluster"])
print(f"📉 Davies-Bouldin Score: {db_score:.4f}")

# 📊 Silhouette Score: 0.5100
# 📉 Davies-Bouldin Score: 0.5300

df.to_csv("normalized_clustering.csv")