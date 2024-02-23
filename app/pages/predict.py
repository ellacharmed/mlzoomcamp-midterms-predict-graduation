import pickle
import urllib
from pathlib import Path

import numpy as np
import streamlit as st

# needs to  be outside, *.bin are globally used
# Set the path to the current file
current_file_path = Path().resolve()
bin_path = current_file_path / 'pages'
# if bin_path.exists():
#     st.write(f'{bin_path = } ')
#     st.write(f"{bin_path} / 'dv.bin' FOUND!  ")
# else:
#     st.write(f" {bin_path} *.bin files not found!!! ")
# Load the pickled scaler and model
dv = pickle.load(open(bin_path / 'dv.bin', 'rb'))
# st.write('dv is loaded')
model = pickle.load(open(bin_path / 'model.bin', 'rb'))
# st.write('model is loaded')


def main():
    # Create a Streamlit app
    st.title('Graduation Prediction App')

    predictor_component()

# function to make prediction when 'predict' button pressed


def predict(student):
    # Transform the student data using the scaler
    student_X = dv.transform([student])

    # Make a prediction using the model
    y_pred_proba = model.predict_proba(np.array(student_X))[0, 1]
    # st.write('y_pred_proba:', y_pred_proba)
    y_pred = model.predict(student_X)[0]
    # st.write('y_pred:', y_pred)

    # Determine the predicted graduation outcome
    graduate = (y_pred_proba >= 0.188)
    st.write('graduate:', graduate)

    # Return the prediction results
    return {
        'graduate_probability': float(y_pred_proba),
        'graduate': bool(graduate)
    }


def predictor_component():
    """## Predictor Component

    A user can input some of the features of the dataset and see the predicted graduation status
    """
    st.markdown("## Predict Student Graduation Rate in 5 years")

    # Create input fields for the student data
    student_data = {
        'sat_total_score': st.slider("SAT Score", min_value=400, max_value=2400, step=1),
        'parental_level_of_education': st.radio("Parent's Education",
                                                ['some high school',
                                                 'high school',
                                                 'associates degree',
                                                 'some college',
                                                 'bachelors degree',
                                                 'masters degree'
                                                 ]),
        'parental_income': st.slider("Parent's Income"),
        'college_gpa': st.slider("College GPA", min_value=2.0, max_value=4.0, step=0.01)
    }

    if st.button("Predict"):
        # print(f'in predictor_component-if')
        st.write(student_data)
        prediction_results = predict(student_data)

        # Display the prediction results
        st.write('Graduate Probability:',
                 prediction_results['graduate_probability'])
        st.write('Predicted Graduation:', prediction_results['graduate'])

        st.success(
            f"The student is predicted to graduate within 5 years: {prediction_results['graduate']}")


main()
