"""

  Deep Learning-Based Recommendation System 

    - recommend movies based on user taste.

"""

# Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle


# Read user, movie, and rating data
users_df = pd.read_csv('users.csv')
movies_df = pd.read_csv('movies_vec.csv')
ratings_df = pd.read_csv('ratings.csv')


# Merge user and movie data into rating data
merged_df = ratings_df.merge(users_df, left_on='user_id', right_on='id').merge(
    movies_df, left_on='movie_id', right_on='id')


# Separate rating, user, and movie from merged_df
ratings = merged_df.iloc[:, 2:3]
users = merged_df.iloc[:, 4:users_df.shape[1]+3]
movies = merged_df.iloc[:, users_df.shape[1]+4:]


# Feature Scaling

# Scale user data
scalerUser = StandardScaler()
users = scalerUser.fit_transform(users)

# Scale movie data
scalerMovie = StandardScaler()
movies = scalerMovie.fit_transform(movies)

# Scale rating data
scalerRating = MinMaxScaler((-1, 1))
ratings = scalerRating.fit_transform(ratings)


# Train-Test split
user_train, user_test = train_test_split(
    users, train_size=0.80, shuffle=True, random_state=1)
movie_train, movie_test = train_test_split(
    movies, train_size=0.80, shuffle=True, random_state=1)
rating_train, rating_test = train_test_split(
    ratings, train_size=0.80, shuffle=True, random_state=1)


# Neural Network Model

num_outputs = 64

# User neural network
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs, activation='linear'),
])

# Movie neural network
movie_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs, activation='linear'),
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(user_train.shape[1]))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_movie = tf.keras.layers.Input(shape=(movie_train.shape[1]))
vm = movie_NN(input_movie)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_movie], output)

model.summary()

# Using a mean squared error loss and an Adam optimizer
tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)

# Training the model
tf.random.set_seed(1)
model.fit([user_train, movie_train], rating_train, epochs=30)

# Evaluate the loss of the model
model.evaluate([user_test, movie_test], rating_test)


# Recommend movies to the user
def recommend(user_id, movie_ids):
    # Create a new dataframe for selected movies
    selected_movies_df = pd.DataFrame({'id': movie_ids})
    # Merge selected movies dataframe with movie dataframe to get details for selected movies
    selected_movies_df = pd.merge(
        selected_movies_df, movies_df, on='id', how='left')
    # Drop column id
    selected_movies_df = selected_movies_df.drop('id', axis=1)

    # Get user data
    selected_user_df = users_df[users_df['id'] == user_id]
    # Copy user data same number of times as the number of movies
    selected_user_df = pd.concat(
        [selected_user_df]*len(movie_ids), ignore_index=True)
    # Drop id column
    selected_user_df = selected_user_df.drop('id', axis=1)

    # scale our user and movie vectors
    suser_vecs = scalerUser.transform(selected_user_df)
    smovie_vecs = scalerMovie.transform(selected_movies_df)

    # make a prediction
    rating_p = model.predict([suser_vecs, smovie_vecs])

    # unscale y prediction
    rating_p = scalerRating.inverse_transform(rating_p)


# Export Model
model.save('user_movies_model.h5')
pickle.dump([scalerUser, scalerMovie, scalerRating], open('scalers.pkl', 'wb'))
