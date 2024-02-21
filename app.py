import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify

#create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    #text = ""
    #if request.method == "POST":
    #text = request.form.get("content")
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    features = [(x) for x in request.form.values()]
    features = [np.array(features)]
    prediction = model.predict(features)

    #sigara durum kontrolü
    prediction = 1 if prediction == 1 else 0
    #return predict
    return render_template("index.html", prediction_text = "Sigara içiyor mu ? {}".format(prediction))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

   
