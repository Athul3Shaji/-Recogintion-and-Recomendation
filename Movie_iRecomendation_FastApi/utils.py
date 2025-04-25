import numpy as np
import tensorflow as tf

def recommend_movies(model, user_id, ratings_df, movie_df, user_encoder, movie_encoder, top_n=5):
    encoded_user = user_encoder.transform([user_id])[0]
    all_movie_ids = ratings_df["movie"].unique()

    user_array = np.full_like(all_movie_ids, encoded_user)
    print(user_array,encoded_user)
    predictions = model.predict([user_array, all_movie_ids], verbose=0).flatten()
    print("predictions",predictions)

    top_indices = predictions.argsort()[-top_n:][::-1]
    print(top_indices)
    top_movies_encoded = all_movie_ids[top_indices]

    top_movie_ids = movie_encoder.inverse_transform(top_movies_encoded)
    print(top_movie_ids,top_movies_encoded)
    top_movies = movie_df[movie_df["movieId"].isin(top_movie_ids)]["title"].values.tolist()
    return top_movies
