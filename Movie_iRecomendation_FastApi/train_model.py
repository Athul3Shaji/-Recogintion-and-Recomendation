from data_loader import load_data
from model import get_model
import tensorflow as tf

ratings, user_enc, movie_enc, movies_df, num_users, num_movies = load_data()

model = get_model(num_users, num_movies)

model.fit([ratings["user"], ratings["movie"]], ratings["rating"], epochs=5, batch_size=64)

model.save("recommender_model.h5")
