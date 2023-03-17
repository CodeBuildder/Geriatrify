import streamlit as st
import pickle
import numpy as np
import os

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False


def load_model():
    if not os.path.isfile('heart_model.pkl'):
        raise FileNotFoundError("Pickle file not found.")

    with open('heart_model.pkl', 'rb') as file:
        model = pickle.load(file)

    return {"model": model}


try:
    data = load_model()
    ml = data["model"]
except Exception as e:
    print(e)


def show_predict_page():
    st.title("Super Cool Awesome Healthhhh Apppoooooo")

    st.write("""### We need some information to predict the your stupid health""")

    cp = st.selectbox(
        "Chest Pain Type?",
        ("Typical Angina", "Atypical Angina", "Non-Anginal Pain, Asymptomatic"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )
    anaemia = st.radio(
        "Do you have anaemia?",
        ('Yes', 'No'))
    if anaemia == 'Yes':
        val = '1'
        anaemia += val
        anaemia = anaemia[3:]
    else:
        val = '0'
        anaemia += val
        anaemia = anaemia[2:]

    creatinine_phosphokinase = st.text_input('Creatinine Phosphokinase')

    diabetes = st.radio(
        "Do you have diabetes?",
        ('Yes', 'No'))
    if diabetes == 'Yes':
        # print('works')
        val = '1'
        diabetes += val
        diabetes = diabetes[3:]
    else:
        val = '0'
        diabetes += val
        diabetes = diabetes[2:]

    ejection_fraction = st.text_input('Ejection Fraction')

    high_blood_pressure = st.radio(
        "Do you have high BP?",
        ('Yes', 'No'))
    if high_blood_pressure == 'Yes':
        # print('works')
        val = '1'
        high_blood_pressure += val
        high_blood_pressure = high_blood_pressure[3:]
    else:
        val = '0'
        high_blood_pressure += val
        high_blood_pressure = high_blood_pressure[2:]

    platelets = st.text_input('Platelets')

    serum_creatinine = st.text_input('Serum Creatinine')

    serum_sodium = st.text_input('Serum Sodium')

    sex = st.radio(
        "Gender?",
        ('Male', 'Female'))

    if sex == 'Male':
        val = '1'
        sex += val
        sex = sex[4:]
    else:
        val = '0'
        sex += val
        sex = sex[6:]

    smoking = st.radio(
        "Do you smoke?",
        ('Yes', 'No'))

    if smoking == 'Yes':
        # print('works')
        val = '1'
        smoking += val
        smoking = smoking[3:]
    else:
        val = '0'
        smoking += val
        smoking = smoking[2:]

    time = st.text_input('time')

    # X = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
    #              high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]])

    # print(X)

    # health = ml.predict(X)

    # st.subheader(health)

    # st.subheader(f"ayo you ")
