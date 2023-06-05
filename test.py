from flask import Flask, jsonify, request

app = Flask(_name_)

# Your code for model training, loading, and prediction will go here

if _name_ == '_main_':
    app.run()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib

# Load the TF-IDF vectorizer and k-NN model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('knn_model.pkl')
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the user query from the request
    user_query = request.json['query']

    # Transform the user query using the loaded vectorizer
    user_query_embedding = vectorizer.transform([user_query]).toarray()

    # Find similar items using the k-NN model
    distances, indices = model.kneighbors(user_query_embedding)

    # Get the top recommended items from the indices
    top_recommended_items = food_data.iloc[indices[0]]

    # Convert the results to JSON and return
    return jsonify(top_recommended_items.to_dict(orient='records'))