"""
Unified Streamlit launcher for the four tasks.
Run:
    streamlit run streamlit_app.py
"""
import streamlit as st
from importlib import import_module

st.title('Multi-Task ML Demo')
app = st.sidebar.selectbox('Choose task', ['Task 1: CNN','Task 2: Chatbot','Task 3: Object Detection','Task 4: Recommender'])

if app=='Task 1: CNN':
    st.write('Open a terminal and run:')
    st.code('python src/task1_cnn.py --epochs 2 --batch-size 128')
elif app=='Task 2: Chatbot':
    import src.task2_chatbot as t2
elif app=='Task 3: Object Detection':
    import src.task3_object_detection as t3
elif app=='Task 4: Recommender':
    st.write('Open a terminal and run:')
    st.code('python src/task4_recommender.py --download True')
