import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
    ratings = pd.read_csv("data/ratings.csv")  # userId, movieId, rating
    movies = pd.read_csv("data/movies.csv")    # movieId, title

    # Encode userId and movieId
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    ratings["user"] = user_encoder.fit_transform(ratings["userId"])
    ratings["movie"] = movie_encoder.fit_transform(ratings["movieId"])

    num_users = ratings["user"].nunique()
    num_movies = ratings["movie"].nunique()

    return ratings[["user", "movie", "rating"]], user_encoder, movie_encoder, movies, num_users, num_movies
