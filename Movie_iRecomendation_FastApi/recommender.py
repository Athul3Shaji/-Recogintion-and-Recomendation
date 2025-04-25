import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv("data/movies.csv")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim = cosine_similarity(tfidf_matrix)

print(cosine_sim)


def get_recommendations(preferred_genres, top_n=5):
    # Convert preferred genres into a string
    input_str = " ".join(preferred_genres)
    input_vec = tfidf.transform([input_str])
    print(input_vec)
    similarity_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
    print("Similarity",similarity_scores)

    top_indices = similarity_scores.argsort()[::-1][:top_n]
    print(top_indices)
    results = []

    for i in top_indices:
        results.append({
            "movie_title": movies.iloc[i]["title"],
            "similarity_score": float(similarity_scores[i])
        })

    return results