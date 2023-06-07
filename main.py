import pymysql
from app import app
from config import mysql
from flask import jsonify
from flask import flash, request
# deployment ml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import joblib

tfidf_model = joblib.load('tfidf_model.pkl')
model = joblib.load('knn_model.pkl')

@app.route('/create', methods=['POST'])
def recommend():
    try:
        _json = request.json
        _namaMakanan = _json['data']
        print(_namaMakanan)
        # running ml 1
        df = pd.read_csv('recommend.csv')
        user_query_embedding = tfidf_model.transform(_namaMakanan).toarray()
        distances, indices = model.kneighbors(user_query_embedding)
        top_recommended_items = df.iloc[indices[0]]
        response = { "data": top_recommended_items.to_dict(orient='records')}
        return jsonify(response)
    except Exception as e:
        print(e)
        return "a string"

@app.errorhandler(404)
def showMessage(error=None):
    message = {
        'status': 404,
        'message': 'Record not found: ' + request.url,
    }
    respone = jsonify(message)
    respone.status_code = 404
    return respone
        
if __name__ == "__main__":
    app.run()