import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
model = pickle.load(open('heart_disease_model.pkl', 'rb'))
scaler = pickle.load(open('normalizer.pkl', 'rb'))


#scaler = MinMaxScaler(feature_range=(0,1))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features = scaler.fit_transform(final_features)    
    prediction = model.predict(final_features)
    print("final features",final_features)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    print(output)

    if output == 0:
        return render_template('index.html', prediction_text='THE PATIENT DOES NOT HAVE A HEART DISEASE')
    else:
         return render_template('index.html', prediction_text='THE PATIENT HAS A HEART DISEASE')
        
@app.route('/predict_api',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)