# model.py

import tensorflow as tf
from tensorflow.keras import layers, Model

def get_model(num_users, num_movies, embedding_size=50):
    user_input = layers.Input(shape=(1,), name='user')
    movie_input = layers.Input(shape=(1,), name='movie')

    user_embedding = layers.Embedding(num_users, embedding_size)(user_input)
    movie_embedding = layers.Embedding(num_movies, embedding_size)(movie_input)

    # Optional: remove extra dimension
    user_vec = layers.Reshape((embedding_size,))(user_embedding)
    movie_vec = layers.Reshape((embedding_size,))(movie_embedding)

    # Dot product of embeddings
    dot = layers.Dot(axes=1)([user_vec, movie_vec])

    output = layers.Dense(1, activation='linear')(dot)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
