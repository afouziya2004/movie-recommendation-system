from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
movies = [
    {"title": "Inception", "description": "sci-fi thriller dream hacking"},
    {"title": "Interstellar", "description": "space exploration sci-fi drama"},
    {"title": "The Dark Knight", "description": "batman action crime thriller"},
    {"title": "Avengers", "description": "superheroes action sci-fi"},
    {"title": "Titanic", "description": "romantic drama love story"}
]

# Extract descriptions
descriptions = [movie["description"] for movie in movies]

# Convert text to vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(descriptions)

def recommend_movie(movie_title):
    titles = [movie["title"] for movie in movies]

    if movie_title not in titles:
        print("Movie not found")
        return

    index = titles.index(movie_title)
    similarity_scores = cosine_similarity(vectors[index:index+1], vectors).flatten()

    recommendations = sorted(
        list(enumerate(similarity_scores)),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nMovies similar to {movie_title}:\n")
    for i, score in recommendations[1:4]:
        print(f"{movies[i]['title']} (Similarity: {round(score, 2)})")

# Test recommendation
recommend_movie("Inception")
