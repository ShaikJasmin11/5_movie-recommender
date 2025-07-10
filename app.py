# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="🎬 Movie Recommender")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie you like and get similar suggestions!")

# Load model
movie_pivot = joblib.load("models/movie_data.pkl")
similarity = joblib.load("models/similarity.pkl")

# Dropdown to select movie
movie_list = movie_pivot.index.tolist()
print("Sample movies:", movie_list[:5])
selected_movie = st.selectbox("Select a Movie", movie_list)

# Recommend
if st.button("Recommend"):
    movie_idx = movie_pivot.index.get_loc(selected_movie)
    distances = similarity[movie_idx]
    top_5_idx = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    st.subheader("🎥 Recommended Movies")
    for i, (idx, score) in enumerate(top_5_idx, 1):
        st.write(f"{i}. {movie_pivot.index[idx]}")

if st.button("Reset"):
    st.rerun()

