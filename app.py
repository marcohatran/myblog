from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import numpy as np
from sklearn.externals import joblib
import nltk

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    model = joblib.load('model.pkl')
    if request.method == 'POST':
        message = request.form['message']
        data = [message]

        my_prediction = model.predict(data)
    return render_template('results.html', prediction=my_prediction)





if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)