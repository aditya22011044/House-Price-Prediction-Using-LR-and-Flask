from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        val1 = float(request.form.get('bedrooms', 0))
        val2 = float(request.form.get('bathrooms', 0))
        val3 = float(request.form.get('floors', 1))
        val4 = float(request.form.get('yr_built', 2000))
        val5 = float(request.form.get('sqft_living', 1000))  # Default reasonable value

        arr = np.array([val1, val2, val3, val4, val5]).reshape(1, -1)
        pred = model.predict(arr)

        return render_template('index.html', data=int(pred[0]))

    except Exception as e:
        return render_template('index.html', data="Error in Prediction: " + str(e))


if __name__ == '__main__':
    app.run(debug=True)
