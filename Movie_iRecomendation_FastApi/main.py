from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from recommender import get_recommendations
import pandas as pd
import tensorflow as tf

from data_loader import load_data
from utils import recommend_movies



app = FastAPI()


ratings_df, user_enc, movie_enc, movie_df, _, _ = load_data()
model = tf.keras.models.load_model("recommender_model.h5")

class UserInput(BaseModel):
    preferred_genres: List[str]

class MovieOutput(BaseModel):
    movie_title: str
    similarity_score: float


    
@app.post("/recommend", response_model=List[MovieOutput])
def recommend(input: UserInput):
    return get_recommendations(input.preferred_genres)



class RequestModel(BaseModel):
    user_id: int
    num_recommendations: int = 5

class ResponseModel(BaseModel):
    recommended_movies: list

@app.post("/recommend-cf", response_model=ResponseModel)
def get_recommendation(req: RequestModel):
    try:
        recs = recommend_movies(
            model, req.user_id, ratings_df, movie_df, user_enc, movie_enc, req.num_recommendations
        )
        return {"recommended_movies": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
