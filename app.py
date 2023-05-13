from flask import Flask, request, jsonify
import pandas as pd
import sklearn
import tensorflow as tf
import pickle


def get_similar_movies(movie_id):
    try:
        movies_df = pd.read_csv('movies.csv')
        similar_movies_model = pickle.load(
            open('./models/similar_movies_model.pkl', 'rb'))

        movie_df = movies_df[movies_df['id'] == movie_id]
        movie_index = movie_df.index[0]
        distances = similar_movies_model[movie_index]
        moviesList = sorted(list(enumerate(distances)),
                            reverse=True, key=lambda x: x[1])[1:11]

        data = []
        for i in moviesList:
            data.append({'title': movies_df.iloc[i[0]].original_title, 'id': str(
                movies_df.iloc[i[0]].id)})

        return data

    except Exception as e:
        print(e)
        return []


def get_user_movies(user_id, movie_ids):
    try:
        users_df = pd.read_csv('users.csv')
        movies_df = pd.read_csv('movies_vec.csv')
        user_movies_model = tf.keras.models.load_model(
            './models/user_movies_model.h5')
        scalerUser, scalerMovie, scalerRating = pickle.load(
            open('./scalers.pkl', 'rb'))

        # Create movie details dataframe
        selected_movies_df = pd.DataFrame({'id': movie_ids})
        selected_movies_df = pd.merge(
            selected_movies_df, movies_df, on='id', how='left')
        selected_movies_df = selected_movies_df.drop('id', axis=1)

        # Create user details dataframe
        selected_user_df = users_df[users_df['id'] == user_id]
        selected_user_df = pd.concat(
            [selected_user_df]*len(movie_ids), ignore_index=True)
        selected_user_df = selected_user_df.drop('id', axis=1)

        # scale user and movie vectors
        suser_vecs = scalerUser.transform(selected_user_df)
        smovie_vecs = scalerMovie.transform(selected_movies_df)

        # make predictions
        rating_p = user_movies_model.predict([suser_vecs, smovie_vecs])

        # unscale y prediction
        rating_p = scalerRating.inverse_transform(rating_p)
        rating_p = [rating[0] for rating in rating_p]

        # Get highest rated movies
        movie_ratings = list(zip(movie_ids, rating_p))
        movie_ratings = sorted(movie_ratings, key=lambda x: x[1], reverse=True)

        movies_df2 = pd.read_csv('movies.csv')

        data = []
        for i in movie_ratings:
            data.append({'title': movies_df2.loc[movies_df2['id'] == i[0], 'original_title'].iloc[0], 'id': str(
                i[0])})

        return data

    except Exception as e:
        print(e)
        return []


app = Flask(__name__)


@app.route('/')
def index():
    return "Welcome to the Deep Learning-Based Recommendation System API"


@app.route('/recommend/movie')
def recommend_similar_movies():
    movie_id = int(request.args.get('id'))
    data = get_similar_movies(movie_id)
    return jsonify(data)


@app.route('/recommend/user', methods=['POST'])
def recommend_user_movies():
    req_body = request.get_json()
    movie_ids = [int(id) for id in req_body['movies']]
    if 'recent' in req_body:
        recent_ids = [int(id) for id in req_body['recent']]
        for id in recent_ids:
            similar_movies = get_similar_movies(id)
            for movie in similar_movies:
                movie_ids.append(int(movie['id']))
    user_id = int(req_body['user_id'])
    data = get_user_movies(user_id, movie_ids)
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
