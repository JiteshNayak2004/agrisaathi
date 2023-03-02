from flask import Flask, request,render_template,jsonify
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

app = Flask(__name__)
model=pickle.load(open('crop_recommender.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/data',methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    # print ("printing"+request.values.get("N"))
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    # print(prediction)
    output=prediction[0]
    print("Crop we recommend you is "+output)
    # return render_template("index.html',prediction_text='Crop we recommend you to grow {}".format(output))
    # print("Crop we recommend you is "+output)
    return render_template('index.html', prediction_text='Crop we recommend is {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)