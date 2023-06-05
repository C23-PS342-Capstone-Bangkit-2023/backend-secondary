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
        _namaMakanan = _json['name']
        conn = mysql.connect()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT meal_id, meal_name, meal_image, calories, carb, protein, fat, tag FROM meals_data")
        empRows = cursor.fetchall()
        changeJson = jsonify(empRows)
        respone = changeJson
        # running ml 1
        # df = pd.DataFrame(empRows, columns = ['meal_id', 'meal_name', 'meal_image', 'calories', 'carb', 'protein', 'fat', 'tag'])
        # user_query_embedding = tfidf_model.transform([_namaMakanan]).toarray()
        # distances, indices = model.kneighbors(user_query_embedding)
        # top_recommended_items = df.iloc[indices[0]]
        # print(df)
        # print(user_query_embedding)
        # print(distances)
        # print(indices)
        # print(top_recommended_items)

        # running ml 2
        # df = pd.DataFrame(empRows, columns = ['tag'])
        # tf_idf_tag = tfidf_model.fit_transform(df)
        # to_array = [v.toarray() for v in tf_idf_tag]
        # tf_idf_tag_array = np.concatenate(to_array, axis=0)

        # model = NearestNeighbors(n_neighbors=4)
        # model.fit(tf_idf_tag_array)

        # user_input = ['vegan', 'karbohidrat', 'protein']
        # user_input_tf_idf = tfidf_model.transform(user_input).toarray()
        # distances, indices = model.kneighbors(user_input_tf_idf)
        # top_10_food = df.iloc[indices[0]]
        tmpCall = pd.read_json(changeJson)
        print(tmpCall)
        respone.status_code = 200
        return respone
        # return jsonify(top_recommended_items.to_dict(orient='records'))
    except Exception as e:
        print(e)
    finally:
        cursor.close()
        conn.close()

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