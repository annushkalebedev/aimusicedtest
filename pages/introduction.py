import os, time
import sys
sys.path.append("..")
import streamlit as st 
import random
import numpy as np
import crash
from utils import *
from params import *

from pages.melody import gen_melody
from pages.development import gen_alldev 
from pages.structure import gen_structure
from pages.harmony import gen_harmony


def introduction_page():

    st.header('AI Music Education System')

    st.markdown('''
        From the list on the left, you can perform different tasks in music composition: **Generate a main melody**, **Generate development melodies**, **Generate song structure**, **Generate harmonies**. Each of the task is equipped with different algorithms, waiting for you to explore. 
        Please start from the task **Generate a main melody**, and begin composing your own music. 
        You can also choose to compose all parts at once using the "Compose all at once" button.
        ''')


    if st.button('Compose all at once!'):
        with st.spinner(text='In progress'):
            gen_melody()
            gen_alldev()
            gen_structure()
            gen_harmony()

        if st.session_state.pure_chords:
            audio_file = open(f'{write_dir}/score.wav', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')





