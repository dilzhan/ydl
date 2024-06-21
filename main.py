import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

model = pickle.load(open('model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

@st.cache_data
def load_data():
    df = pd.read_csv("bodyfat.csv")
    return df


df = load_data()

st.sidebar.title("BodyFat Predictor Bot!")
st.sidebar.write("Hello, please, select the options suitable for you")

age = st.sidebar.slider("age", 1, 100, 21)
weight = st.sidebar.slider('weight(kg)', 0, 150, 83) * 2.2
height = st.sidebar.slider('height(cm)', 0, 250, 178) * 0.3937
neck = st.sidebar.slider('Neck Circumference(cm)', 0, 100, 38)
chest = st.sidebar.slider('Chest Circumference(cm)', 0, 150, 99)
abdomen = st.sidebar.slider('Abdomen Circumference(cm)', 0, 150, 91)
hip = st.sidebar.slider('Hip Circumference(cm)', 0, 150, 101)
thigh = st.sidebar.slider('Thigh Circumference(cm)', 0, 100, 58)
knee = st.sidebar.slider('Knee Circumference(cm)', 0, 70, 38)
ankle = st.sidebar.slider('Ankle Circumference(cm)', 0, 50, 21)
biceps = st.sidebar.slider('Biceps Circumference(cm, extended)', 0, 50, 31)
forearm = st.sidebar.slider('Forearm Circumference(cm)', 0, 50, 28)
wrist = st.sidebar.slider('Wrist Circumference(cm)', 0, 50, 16)

bmi = weight * 703.0 / height**2
bf_bmi = bmi * 1.39 + age * 0.16 - 19.34
bai = hip / (height * 0.0254 * np.sqrt(height * 0.0254)) - 18
navy = 86.010 * np.log10(abdomen - neck) - 70.041 * np.log10(height) + 36.76
    
x = [[age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist, bmi, bf_bmi, bai, navy]]
x = scaler.transform(x)
pred = model.predict(x)[0].round(2)
ratio = int(round(df[df['BodyFat'] > pred].shape[0] / df.shape[0], 2) * 100)

st.markdown(f'The predicted BF is: {pred}')
st.markdown(f'You are leaner than {ratio}% of men')

st.write("The Distribution of the bodyfat over men population:")
fig = plt.figure()
plt.title("Bodyfat Percentage")
sns.histplot(df["BodyFat"], kde = True, stat = 'density')
st.pyplot(fig)