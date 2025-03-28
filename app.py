from flask import Flask,jsonify,render_template,request
import numpy as np 
import pandas as pd 
import pickle 
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')
app=Flask(__name__)
iris=load_iris()
target_names=iris.target_names
model=pickle.load(open("model.pkl","rb"))
flower_images = {
    "setosa": "setosa.jpg",
    "versicolor": "versicolor.jpg",
    "virginica": "virginica.jpg"
}
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    float_features=[float(i) for i in request.form.values()]
    features=[np.array(float_features)]
    prediction=model.predict(features)
    prs=target_names[prediction][0]
    image_file = flower_images.get(prs, None)
    return render_template("index.html",prediction_text=f"This flower is : {prs}",image_file=image_file)


if __name__=="__main__":
    app.run(debug=True)