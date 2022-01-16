import os
import pandas as pd
import matplotlib.pyplot as plt 
import io 
import requests 
import openpyxl
import streamlit as st 
import random
import numpy as np
import music21 as m2
from utils import *

from pages.introduction import introduction_page
from pages.melody import melody_page
from pages.development import development_page 
from pages.structure import structure_page 
from pages.harmony import harmony_page 
from pages.arrangement import arrangement_page 


# global parameters about the music
meta_input = st.beta_container()

playground = st.beta_container()
author_credits = st.beta_container()



with meta_input:
    st.session_state.task = st.sidebar.selectbox('Choose Task', ['Introduction', 'Main melody', 'Development melody', 'Structure', 'Harmony', 'Instrumentation'])

    st.sidebar.header('Parameter Selection') 

    time_signature = st.sidebar.selectbox('Choose Time Signature', ['2/4', '3/4', '4/4'],
                                key="time_signature",
                                on_change=modify_mc)
    key_signature = st.sidebar.selectbox('Choose Key Signature', ['C Major', 'G Major', 'D Major', 'A Major', 'E Major'],
                                key="key_signature",
                                on_change=modify_mc)
    st.session_state.n_measures = st.sidebar.slider('Num. of bars in main melody ', min_value = 4, max_value= 12, value= 8, step=2)
    st.sidebar.slider('Tempo ', min_value = 60, max_value= 180, value= 120, step=4,
                                key='tempo', on_change=modify_tempo)

    # np.random.seed(88)

    prepare_state()


with playground:
    if st.session_state.task == "Introduction":
        introduction_page()
    elif st.session_state.task == 'Development melody':
        development_page()
    elif st.session_state.task == 'Structure':
        structure_page()
    elif st.session_state.task == 'Harmony':
        harmony_page()
    elif st.session_state.task == 'Instrumentation':
        arrangement_page()
    else:
        melody_page()
        


