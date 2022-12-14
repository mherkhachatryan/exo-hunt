#from hunter.preprocessing import ExoHuntDataset

from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os
import yaml

#configs_path = os.path.join(os.getcwd(), "hunter", "configs.yaml")

#with open(configs_path, "r") as f:
 #   _configs = yaml.load(f, yaml.FullLoader)

#configs = RunConfigs(_configs)

app = Flask("exo_hunter")
with open('/home/edmon/DataspellProjects/exo-hunt/exo-hunt/logs/RandomForest/RandomForest--2022-12-13 19:11:51.194401.pkl', "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


#@app.route('/prediction')
#def predict():
    # recive the data sent by client
   # data = request.get_json(force=True)
  #  dataset = ExoHuntDataset(configs.data_path, configs.random_state)

    # X_test, y_test = dataset.load_test_data()
 #   prediction = model.predict(np.array(data['input']).reshape(1, -1))
#
  #  return str(prediction[0])


if __name__ == '__main__':
    app.run(debug=True)