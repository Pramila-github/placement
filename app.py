import streamlit as st
import numpy as np
import pandas as pd
import pickle


models = pickle.load(open('model.pkl', 'rb'))
def home():
    return "welcome"


def predict(gender,comSkill1, ssc_p,ssc_b, hsc_p,hsc_b,hsc_s, degree_p,degree_t, No_certi, internships1,etest_p, placetest_p,sports1):
    prediction = models.predict([[gender,comSkill1, ssc_p,ssc_b, hsc_p,hsc_b,hsc_s, degree_p,degree_t, No_certi, internships1,etest_p, placetest_p,sports1]])
    print(prediction)
    return prediction


def convert_to_int(word):
    word_dict = {'Good': 1, 'Bad': 0, 'yes': 1, 'no': 0, 'completed': 1, 'none': 0,'Male':0,'Female':1,'Central':0, 'Other':1,'Science':0, 'Arts':1, 'biology':1, 'Others':0,'Computer science Engineering':0, ' B.Tech IT':1, 'Electronics':2}
    return word_dict[word]


def main():
    st.title("STUDENT PLACEMENT PREDICTION⭐")
    st.sidebar.header('STUDENT DETAILS💻')
    name = st.sidebar.text_input('Name 📋')
    sl_no = st.sidebar.text_input('Register Number 📋')
    gender = st.sidebar.radio("Gender 👫", ('Male', 'Female'))
    gender1=convert_to_int(gender)
    comSkill = st.sidebar.radio("Communication Skills 💬🎤", ('Good', 'Bad'))
    comSkill1 = convert_to_int(comSkill)
    ssc_p = st.sidebar.slider('SSC percentage', 0, 1, 100)
    ssc_b = st.sidebar.radio("SSC Board 📚", ('Central', 'Other'))
    ssc_b = convert_to_int(ssc_b)
    hsc_p = st.sidebar.slider('HSC percentage', 0, 1, 100)
    hsc_b = st.sidebar.radio("HSC Board 📚", ('Central', 'Other'))
    hsc_b = convert_to_int(hsc_b)
    hsc_s = st.sidebar.radio("HSC Group", ('Science', 'Arts', 'biology', 'Others'))
    hsc_s = convert_to_int(hsc_s)
    degree_t = st.radio("Degree(Engineering) 🎓", ('Computer science Engineering', ' B.Tech IT', 'Electronics','Others'))
    degree_t =convert_to_int(degree_t)
    degree_p = st.slider('Degree percentage', 0, 1, 100)
    No_certi = st.text_input('Number of certifications 🏆📜')
    internships = st.radio("Internships 👨‍💻", ('completed', 'none'))
    internships1 = convert_to_int(internships)
    etest_p = st.slider('E-tests(aptitude)  🏁', 0, 1, 100)
    sports = st.radio("Through Sports ⚽️🏏🏃‍♂️ ", ('yes', 'no'))
    sports1 = convert_to_int(sports)
    placetest_p = st.slider('Placement Percentage 🎓🎓 ', 0, 1, 100)
    st.radio("Interest about placement ✔️", ('Interested', 'Not Interested'))
    

    result = ""
    if st.button("submit"):
        result = predict(gender1,comSkill1, ssc_p,ssc_b, hsc_p,hsc_b,hsc_s, degree_p,degree_t, No_certi, internships1,etest_p, placetest_p,sports1)
        print(result)

        st.subheader("Entered Details")
        st.write("Name 📋 : ", name)
        st.write("Register Number 📋 :", sl_no)
        st.write("SSC percentage 📚 :", ssc_p)
        st.write("HSC percentage 📚 :", hsc_p)
        st.write("Degree percentage 🎓 :", degree_p)
        st.write("E-Test(Aptitude) 🏁 :", etest_p)
        st.write("Placement Percentage 🎓  :", placetest_p)
        st.write("Communication Skills 💬🎤 :", comSkill)
        st.write("Internship status 👨‍💻 :", internships)
        st.write("Number of certifications 🏆📜 :", No_certi)
        st.write("Through sports ⚽️🏏🏃‍♂️  :", sports)

        st.success('Predicted Result is {}'.format(result))


if __name__ == "__main__":
    main()
