from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json


app = Flask(__name__)


@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    vector = tfidf.transform(data)
    prediction = np.array2string(model.predict(vector))

    return jsonify(prediction)

if __name__ == '__main__':
    modelfile = 'models/TP7EX3_prediction.pickle'
    tfidfFile = 'models/tfidf.pickle'
    model = p.load(open(modelfile, 'rb'))
    tfidf = p.load(open(tfidfFile, 'rb'))
    app.run(debug=True, host='0.0.0.0')