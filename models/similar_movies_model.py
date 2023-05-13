"""
  Deep Learning-Based Content-Based Recommendation System 

    - recommend movies similar to a given movie.

"""

# Import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import pickle


# Read Movies Data
movies_df = pd.read_csv('movies.csv')
movies_df = movies_df.dropna()

movies = movies_df.drop('id', axis=1)
movies = movies.drop('original_title', axis=1)


# OneHotEncoding Genre
def encodeGenre(pdGenre):
    genres = [eval(x) for x in pdGenre]
    mlb = MultiLabelBinarizer()
    one_hot_encoded_cast = pd.DataFrame(mlb.fit_transform(genres))
    return one_hot_encoded_cast


genreVector = encodeGenre(movies['genre'])
movies = movies.drop('genre', axis=1)
movies = pd.concat([movies, genreVector], axis=1)


# Vectorization of Cast Using CountVectorizer
def encodeCast(pdCast):
    vectorizer = CountVectorizer(max_features=5000)
    vectorizer.fit(pdCast)
    matrix = pd.DataFrame(vectorizer.transform(pdCast).toarray())
    return matrix


castVector = encodeCast(movies['cast'])
movies = movies.drop('cast', axis=1)
movies = pd.concat([movies, castVector], axis=1)


# Dimensionality Reduction Of Cast Vector Using PCA
# pca = PCA(n_components=matrix.shape[0])
# X_pca = pd.DataFrame(pca.fit_transform(matrix))
# movies = movies.drop('cast', axis=1)
# movies = pd.concat([movies, X_pca], axis=1)


# Vectorization of Title Using BERT
def bertEncoder(pdText):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    # Tokenize the movie titles and convert them to input IDs and attention masks
    tokenized = tokenizer.batch_encode_plus(pdText.tolist(),
                                            padding=True,
                                            truncation=True,
                                            max_length=128,
                                            return_tensors="pt")
    # Pass the input IDs and attention masks through the BERT model to get the embeddings
    with torch.no_grad():
        embeddings = model(**tokenized).last_hidden_state.mean(dim=1)
    # Convert the embeddings to a NumPy array
    embeddings_np = embeddings.numpy()
    embeddings = pd.DataFrame(embeddings_np)
    # embeddings = embeddings.reshape(embeddings.shape[0], -1)
    return embeddings


titleVector = bertEncoder(movies['title'])
movies = movies.drop('title', axis=1)
movies = pd.concat([movies, titleVector], axis=1)


# Vectorization of Overview using BERT
overviewVector = bertEncoder(movies['overview'])
movies = movies.drop('overview', axis=1)
movies = pd.concat([movies, overviewVector], axis=1)


# Export Movie Vector to CSV File
movieVector = pd.concat([movies_df['id'], movies], axis=1)
movieVector.to_csv('movies_vec.csv', index=False)


# Movies Neural Network
num_outputs = 32
tf.random.set_seed(1)

item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='linear'),
])

# Input layer
input_item_m = tf.keras.layers.Input(shape=2285)
# Use the trained item_NN
vm_m = item_NN(input_item_m)
# Incorporate normalization
vm_m = tf.linalg.l2_normalize(vm_m, axis=1)

model_m = tf.keras.Model(input_item_m, vm_m)
vms = model_m.predict(movies)


# Compute similarity between movies
similarity = cosine_similarity(vms)


# Export Model
pickle.dump(similarity, open('similar_movies_model.pkl', 'wb'))


# Recommend Similar Movies
def recommend(movie_id):
    movie_df = movies_df[movies_df['id'] == movie_id]
    if movie_df.empty:
        return []
    movie_index = movie_df.index[0]
    distances = similarity[movie_index]
    moviesList = sorted(list(enumerate(distances)),
                        reverse=True, key=lambda x: x[1])[1:11]
    data = []
    for i in moviesList:
        data.append(movies_df.iloc[i[0]].original_title)
    return data
