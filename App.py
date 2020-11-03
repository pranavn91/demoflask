import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#run_with_ngrok(app)   #starts ngrok when the app is run
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('Sample.html')

@app.route('/predict',methods=['POST','GET'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('Sample.html', prediction_text='CO2EMISSIONS should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)





if __name__ == "__main__":
    app.run()
