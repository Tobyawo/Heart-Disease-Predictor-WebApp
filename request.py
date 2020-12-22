import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age': 40, 'sex':1, 'cp':1, 'trestbps':125, 'chol':212, 'fbs':0, 'restecg':1, 'thalach':168,'exang':0, 'oldpeak':1, 'slope':2, 'ca':2, 'thal':3})

print(r.json())