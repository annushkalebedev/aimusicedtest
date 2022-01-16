import os, time
import sys
sys.path.append("..")
import pandas as pd
import matplotlib.pyplot as plt 
import openpyxl
import streamlit as st 
import random
import numpy as np
import music21 as m2
import crash
from utils import *
from params import *



def change_instrument():

    if isinstance(st.session_state.full_melody[0], m2.instrument.Instrument):
        st.session_state.full_melody.pop(0)
    st.session_state.full_melody.insert(0, instr_trans[st.session_state.melody_instr])

    if isinstance(st.session_state.harmony[0], m2.instrument.Instrument):
        st.session_state.harmony.pop(0)
    st.session_state.harmony.insert(0, instr_trans[st.session_state.harmony_instr])
    
    st.session_state.score = m2.stream.Stream()
    st.session_state.score.insert(0, st.session_state.full_melody)
    st.session_state.score.insert(0, st.session_state.harmony)

    # st.session_state.score.show('text')

    return 

def change_drum():

    print("change")
    if len(st.session_state.score) >=3:
        st.session_state.score = st.session_state.score[:-1]

    if st.session_state.drum_pattern == "No drum track":
        return 

    init_drum()
    dp = drum_patterns[st.session_state.drum_pattern]

    length = int(len(st.session_state.full_melody) / 2)

    for i in range(0, length, 16):
        for drum_pitch, pattern in dp.items():

            for idx, p in enumerate(pattern):
                st.session_state.drum.insert(idx + i,
                    m2.note.Note(drum_pitch) if p else m2.note.Rest() 
                )

    st.session_state.score.insert(0, st.session_state.drum)

    return 

def arrangement_page():
    st.header('Instrumentation')

    col1, col2 = st.beta_columns(2)
    instruments = list(instr_trans.keys())
    with col1:
        col1.selectbox('Melodic Instrument', instruments[:int(len(instruments)/2)],
            key='melody_instr',
            on_change=change_instrument)
        synthaudio(st.session_state.full_melody, "full_melody")
        synthaudio(st.session_state.score, "score")
    with col2:
        col2.selectbox('Harmony Instrument', instruments[int(len(instruments)/2):],
            key='harmony_instr',
            on_change=change_instrument)
        synthaudio(st.session_state.harmony, "harmony")
        synthaudio(st.session_state.score, "score")

    st.text('Melody')
    audio_file = open(f'{write_dir}/full_melody.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

    st.text('Harmony')
    audio_file = open(f'{write_dir}/harmony.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

    st.text('Full Song')
    audio_file = open(f'{write_dir}/score.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

    st.markdown('---')

    col1, col2 = st.beta_columns(2)
    with col1:
        col1.selectbox('Drum pattern', ['No drum tack'] + list(drum_patterns.keys()),
            key='drum_pattern',
            on_change=change_drum)
        synthaudio(st.session_state.score, "score")

    st.text('Full song + drum')
    audio_file = open(f'{write_dir}/score.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

