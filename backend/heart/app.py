import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('heart_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('heart.html')


@app.route('/predict', methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(int, feature_list))
    final_features = np.array(feature_list).reshape(1, 12)

    print(final_features)

    prediction = model.predict(feature_list)

    print(prediction)

    output = int(prediction[0])
    if output == 1:
        text = "diseases iku"
    else:
        text = "disaese ila"

    return render_template('result.html', prediction_text='Employee Income is {}'.format(text))

# def ValuePredictor(to_predict_list):
#     to_predict = np.array(to_predict_list).reshape(1, 12)
#     loaded_model = pickle.load(
#         open("heart_model.pkl", "rb"))
#     result = loaded_model.predict(to_predict)
#     return result[0]

    return render_template('result.html', prediction_text=prediction)

    # to_predict_list = request.form.to_dict()
    # to_predict_list = list(to_predict_list.values())
    # to_predict_list = list(map(float, to_predict_list))

    # if(int(result) == 1):
    #     prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    # else:
    #     prediction = "No need to fear. You have no dangerous symptoms of the disease"
    # return(render_template("result.html", prediction_text=prediction))


if __name__ == "__main__":
    app.run(debug=True)
