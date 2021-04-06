from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
from preprocess import Preprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return flask.render_template('first.html')


@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['data_file']
    
    if not f:
        return "No file"
    standardSaler=joblib.load('./models/train_scaler.pkl')
    
   
    
    df = pd.read_csv(request.files.get('data_file'))
    final_data=Preprocess(df).forward_final()
    prediction=loaded_model.predict(standardSaler.transform(final_data).reshape(1,-1))
    yard=np.argmax(prediction)-99
    prob=np.cumsum(prediction)[yard+99]
    plt.plot([i for i in range(-99,100)],np.cumsum(prediction)) 
    plt.xlabel('Yards')
    plt.ylabel('CDF Probablity')
    plt.title('CDF of Yard Gained')
    plt.savefig('./static/images/new_plot.png')
    return flask.render_template('result_and_inference.html', name = 'CDF_plot', url ='./static/images/new_plot.png',prob=prob,yard=yard)
    
    

if __name__ == '__main__':

    # opening and store file in a variable

    json_file = open('./models/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()

    # use Keras model_from_json to make a loaded model

    loaded_model = tf.keras.models.model_from_json(loaded_model_json)

    # load weights into new model

    loaded_model.load_weights("./models/model.h5")
    app.run(debug=True)
