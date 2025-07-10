#  Movie Recommendation System

> RISE Internship Project 5 – Tamizhan Skills  
> Built with Pandas, Scikit-Learn, Cosine Similarity, and Streamlit

A machine learning-powered recommender that suggests movies similar to a selected title using collaborative filtering and similarity scoring. This is the fifth project in the **Machine Learning & AI** track of the RISE Internship by Tamizhan Skills.

---

##  Project Objective

To build a movie recommendation engine that:
- Loads user rating data and movie metadata from CSV files
- Constructs a user-item interaction matrix
- Computes **cosine similarity** between movies using collaborative filtering
- Provides an interactive **Streamlit UI** to suggest top 5 similar movies

---

##  Tech Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-learn (Cosine Similarity)**
- **SciPy (Sparse Matrix Optimization)**
- **Joblib** (for model persistence)
- **Streamlit** (for frontend UI)

---

##  Project Structure

```bash
movie-recommender/
├── app.py                     # Streamlit frontend for movie suggestions
├── main.py                    # Model training + similarity matrix generation
├── requirements.txt           # Required packages
├── data/
│   ├── movies.csv             # Movie metadata (movieId, title, genres)
│   └── ratings.csv            # User rating data (userId, movieId, rating)
├── models/
│   ├── movie_data.pkl         # Pivot table: movie × user ratings
│   └── similarity.pkl         # Cosine similarity matrix between movies
├── src/
│   └── preprocess.py          # Data loader and cleaner
└── README.md                  # This file
```

---

## Dataset

- Source: MovieLens 20M Dataset(https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv)
- movies.csv: Contains movie titles and genres
- ratings.csv: Contains user ratings for movies
- Merged to build a pivot matrix of movie ratings by users
  
---

## How to Run

- Step 1: Install Dependencies
  
```bash
  pip install -r requirements.txt
```

- Step 2: Train the Model
  
```bash
  python main.py
```

- Step 3: Launch the Web App
  
```bash
  streamlit run app.py
```

  ---

## Model Performance

✅ Collaborative Filtering via user-movie pivot
✅ Cosine similarity to score movie similarity
✅ Filtered users & movies to avoid memory overload
✅ Sparse matrix computation for performance

---

## Highlights

- Cleaned and filtered the dataset to avoid sparsity & memory errors
- Uses cosine similarity to find the most similar movies
- Efficient with large datasets using sparse matrix optimization
- Top-5 recommendations delivered in real-time via Streamlit
- Modular, scalable design for future API or poster integration

---

## Acknowledgements

Thanks to Tamizhan Skills for the opportunity to build real-world AI/ML projects under the RISE Internship.

Built by @ShaikJasmin11
