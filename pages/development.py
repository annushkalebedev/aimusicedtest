import os, time
import sys
sys.path.append("..")
import streamlit as st 
import random
import numpy as np
import music21 as m2
import crash
from utils import * 
from params import *
from evolution import *

# write the generated solution as music21 stream
def write_stream(solution, name, melody="M2"):
    if melody == "M2":
        stream = st.session_state.melody_m2
        note_list = st.session_state.melody_m2_note_list
    elif melody == "M3":
        stream = st.session_state.melody_m3
        note_list = st.session_state.melody_m3_note_list
    elif melody == "B1":
        stream = st.session_state.melody_b1
        note_list = st.session_state.melody_b1_note_list
    else:
        stream = st.session_state.melody_b2
        note_list = st.session_state.melody_b2_note_list
    for token in solution:
        if (token < 24) or (token > 96):
            stream.append(m2.note.Rest(quarterLength=0.5))
            note_list.append("R")
        else:
            note = m2.note.Note(token, quarterLength=0.5)
            stream.append(note)
            if "-" in note.name:
                note_list.append(note.pitch.getEnharmonic().name)
            else:
                note_list.append(note.name)

    return 


def gen_alldev():

    init_ga()

    init_melody_m2()
    solution = generate_development(st.session_state.ga)
    write_stream(solution, "melody_m2")
    synthaudio(st.session_state.melody_m2, "melody_m2")

    init_melody_b1()
    solution = generate_development(st.session_state.ga,
        bridge=True)
    write_stream(solution, "melody_b1", melody="B1")
    synthaudio(st.session_state.melody_b1, "melody_b1")

    init_melody_m3()
    solution = generate_development(st.session_state.ga)
    write_stream(solution, "melody_m3", melody="M3")
    synthaudio(st.session_state.melody_m3, "melody_m3")

    init_melody_b2()
    solution = generate_development(st.session_state.ga,
        bridge=True)
    write_stream(solution, "melody_b2", melody="B2")
    synthaudio(st.session_state.melody_b2, "melody_b2")


    return 


def development_page():
    st.header('Development melody')

    with st.beta_expander('Intro: Development melody'):
        st.markdown('''
        After the main melody, we hope to enrich our song with more music materials. Here, we use Genetic Algorithm to generate development melodies. 
        Each melody can be represented with a vector of MIDI pitch number, such as \[64, 62, 0, 64, 64, 65, 67, 0\]. In genetic algorithm, we use such vectors to simulate a population, where different vectors combine and mutate according to our musical rules. After a number of iterations, the randomly initialized vectors will evolute into satisfying melodies. 
        In this task, we are going to generate four music lines: M2, M3, B1, B2.
        ''')

    st.subheader('Existing main melody M1：')
    st.subheader(format_sequence(st.session_state.melody_note_list))

    audio_file = open(f'{write_dir}/melody.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

    n_measures = st.session_state.n_measures
    n_per_measure = time_sigs[st.session_state.time_signature]

    init_ga()

    st.markdown("""---""")

    cola, colb, colc = st.beta_columns(3)
    with cola:
        st.slider('Iterations', min_value=20, max_value=256, 
            value= 100, step=2, key="num_gen", on_change=init_ga,
            help="Amount of iterations to run，between 20 to 256.")

    with colb:
        st.slider('Population size', min_value=64, max_value=256, 
            value= 64, step=4, key="sol_per_pop", on_change=init_ga,
            help="Amount of population to filter in each iteration.")

    with colc:
        st.slider('Mating size', min_value=16, max_value=64, 
            value= 32, step=2, key="num_parents_mating", on_change=init_ga,
            help="Amount of population to mate in each iteration.")

    st.markdown("""---""")


    col1, col2 = st.beta_columns(2)

    with col1:

        st.slider('Key weight', min_value=0.0, max_value=1.0, 
            value= 0.5, step=0.05, key="key_weight", on_change=init_ga,
            help="Higher key weight indicates more notes in the key。")
        st.slider('Smoothing weight', min_value=0.0, max_value=1.0, 
            value= 0.5, step=0.05, key="smoothing_weight", on_change=init_ga,
            help="Higher smoothing weight indicates a smoother melody contour.")

        st.markdown("""---""")

        if col1.button('Generate melody M2！'):
            init_melody_m2()
            with st.spinner(text='In progress'):
                solution = generate_development(st.session_state.ga)
                write_stream(solution, "melody_m2")
                synthaudio(st.session_state.melody_m2, "melody_m2")
        
        st.markdown(format_sequence(st.session_state.melody_m2_note_list))

        if st.session_state.melody_m2_note_list:
            audio_file = open(f'{write_dir}/melody_m2.wav', 'rb')
            audio_bytes = audio_file.read()
            col1.audio(audio_bytes, format='audio/wav')

        if col1.button('Generate transition B1！'):
            init_melody_b1()
            with st.spinner(text='In progress'):
                solution = generate_development(st.session_state.ga,
                    bridge=True)
                write_stream(solution, "melody_b1", melody="B1")
                synthaudio(st.session_state.melody_b1, "melody_b1")

        st.markdown(format_sequence(st.session_state.melody_b1_note_list))

        if st.session_state.melody_b1_note_list:
            audio_file = open(f'{write_dir}/melody_b1.wav', 'rb')
            audio_bytes = audio_file.read()
            col1.audio(audio_bytes, format='audio/wav')


    with col2:
        st.slider('Similarity weight', min_value=0.0, max_value=1.0, 
            value= 0.5, step=0.05, key="similarity_weight", on_change=init_ga,
            help="The similarity between generated melody and main melody.")
        st.slider('Rhythm weight', min_value=0.0, max_value=1.0, 
            value= 0.5, step=0.05, key="rhythm_weight", on_change=init_ga,
            help="The higher rhythm weight is, the beats are more regular.")

        st.markdown("""---""")


        if col2.button('Generate melody M3！'):
            init_melody_m3()
            with st.spinner(text='In progress'):
                solution = generate_development(st.session_state.ga)
                write_stream(solution, "melody_m3", melody="M3")
                synthaudio(st.session_state.melody_m3, "melody_m3")
        
        st.markdown(format_sequence(st.session_state.melody_m3_note_list))

        if st.session_state.melody_m3_note_list:
            audio_file = open(f'{write_dir}/melody_m3.wav', 'rb')
            audio_bytes = audio_file.read()
            col2.audio(audio_bytes, format='audio/wav')

        if col2.button('Generate transition B2！'):
            init_melody_b2()
            with st.spinner(text='In progress'):
                solution = generate_development(st.session_state.ga,
                    bridge=True)
                write_stream(solution, "melody_b2", melody="B2")
                synthaudio(st.session_state.melody_b2, "melody_b2")
        
        st.markdown(format_sequence(st.session_state.melody_b2_note_list))

        if st.session_state.melody_b2_note_list:
            audio_file = open(f'{write_dir}/melody_b2.wav', 'rb')
            audio_bytes = audio_file.read()
            col2.audio(audio_bytes, format='audio/wav')


