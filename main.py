# main.py

import joblib
from src.preprocess import load_data
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def train_model():
    df = load_data()

    # Filter movies with more than 100 ratings
    movie_counts = df['title'].value_counts()
    df = df[df['title'].isin(movie_counts[movie_counts > 100].index)]

    # Filter users who rated more than 50 movies
    user_counts = df['userId'].value_counts()
    df = df[df['userId'].isin(user_counts[user_counts > 50].index)]

    print(f"🟡 Filtered to {df['title'].nunique()} movies and {df['userId'].nunique()} users")

    # Create pivot table
    # Pre-pivot deduplication
    df = df.groupby(['title', 'userId'])['rating'].mean().reset_index()

    # Now pivot safely
    movie_user_matrix = df.pivot(index='title', columns='userId', values='rating').fillna(0)


    # Convert to sparse matrix
    movie_sparse = csr_matrix(movie_user_matrix.values)

    # Calculate similarity
    similarity = cosine_similarity(movie_sparse)

    # Save artifacts
    joblib.dump(movie_user_matrix, 'models/movie_data.pkl')
    joblib.dump(similarity, 'models/similarity.pkl')

    print("✅ Model saved:", movie_user_matrix.shape)

if __name__ == '__main__':
    train_model()
