import json
import pickle

from flask import Flask, jsonify, request

def load(input_file):
    with open(input_file, 'rb') as f_in:
        return pickle.load(f_in)


dv = load(f'dv.pkl')
model = load(f'model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    print(f'model.pkl loaded successfully')
    student = request.get_json()
    print(student)

    student_X = dv.transform([student])
    y_pred_proba = model.predict_proba(student_X)[0, 1]
    # print(y_pred_proba)
    graduate = (y_pred_proba >= 0.5)
    # print(graduate)

    # Convert the predicted output to a JSON string.
    result = {
        'graduate_probability': float(y_pred_proba),
        'graduate': bool(graduate)
    }
    return result


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
