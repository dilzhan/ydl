import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

st.sidebar.title("My pet project")
st.sidebar.write("Hello, please, select the bot you want to use")

bot = st.sidebar.selectbox('Bot: ', list(['Salary Predictor Bot', 'BodyFat Percentage Predictor Bot']))

if bot == 'BodyFat Percentage Predictor Bot':
    model = pickle.load(open('model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
    df = pd.read_csv("bodyfat.csv")

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
    g = sns.histplot(df["BodyFat"], kde = True, stat = 'density')
    g.axvline(pred, color = 'red')
    st.pyplot(fig)

else:
    model = pickle.load(open('model_salaries.sav', 'rb'))
    df = pd.read_csv('_salary_train.csv')

    algebra = st.sidebar.slider("algebra", 0, 100, 50)
    programming = st.sidebar.slider("programming", 0, 100, 50)
    data_science = st.sidebar.slider("data science", 0, 100, 50)
    robotics = st.sidebar.slider("robotics", 0, 100, 50)
    economics = st.sidebar.slider("economics", 0, 100, 50)
    finance = st.sidebar.slider("finance", 0, 100, 50)
    
    job_titles = ['data scientist', 'developer', 'economist', 'finance director', 'junior developer', 'logistics manager', 'robotics engineer', 'senior developer']
    job = st.sidebar.selectbox('job', list(job_titles))
    job_arr = np.zeros(8)
    job_arr[job_titles.index(job)] = 1

    x = np.append(np.array([algebra, programming, data_science, robotics, economics, finance]), job_arr)
    x = [x]
    pred = round(model.predict(x)[0]) * 10000
    st.write(f"Your expected salary is {pred}")
    stats = []

    df = df[df['job'] == job]
    cols = ['algebra', 'programming', 'data science', 'robotics', 'economics', 'finance']

    st.write(f'Here is how does the labour market for {job}s look like:')
    fig = plt.figure()
    sns.boxplot(df[cols])
    st.pyplot(fig)

    for i in range(len(cols)):
        col = cols[i]
        stats.append(round(df[df[col] < x[0][i]][col].count() / df.shape[0] * 100))
    
    for i in range(len(cols)):
        fig = plt.figure()
        g = sns.histplot(df[cols[i]])
        g.axvline(x[0][i], color = 'red')
        st.pyplot(fig)
        st.write(f'In {cols[i]}, you are better than {stats[i]}% of people')