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
import pretty_midi
import crash
from markov import MarkovChain
from utils import *
from params import *



'''
pitches: list of generated pitches
melody_note_list: list of note string token
melody: music21 stream 
'''
def write_stream(pitches):
    # fit the pitch into the rhythmic pattern
    for i, n in enumerate(st.session_state.melody_note_list):
        if n == "":
            st.session_state.melody_note_list[i] = pitches.pop(0)

    print(st.session_state.melody_note_list)
    prev_midi = m2.note.Note(st.session_state.melody_note_list[0]).pitch.midi
    for note in st.session_state.melody_note_list:
        if note == "R":
            st.session_state.melody.append(m2.note.Rest(note, quarterLength=EIGHTH))
        elif note == "-": 
            last_note = st.session_state.melody.pop(-1)
            last_note.quarterLength = last_note.quarterLength + EIGHTH
            st.session_state.melody.append(last_note)
        else:
            midi_num = m2.note.Note(note).pitch.midi
            if midi_num - prev_midi > 6:
                midi_num -= 12
            elif prev_midi - midi_num > 6:
                midi_num += 12
            else:
                pass
            st.session_state.melody.append(m2.note.Note(midi_num, quarterLength=EIGHTH))
            prev_midi = midi_num
    return 

def make_patterns():
    for i in range(st.session_state.n_measures):
        st.session_state.melody_note_list.extend(
            random.choice(rhythmic_patterns[st.session_state.time_signature]))
    
    return len([n for n in st.session_state.melody_note_list if n == ""])


def gen_melody():
    init_melody()
    pitch_slots = make_patterns()
    pitches = st.session_state.mc.generate(PITCHES.index(st.session_state.start_note), pitch_slots)
    
    # force an ending
    pitches[-1] = key_sigs[st.session_state.key_signature].tonic.name

    write_stream(pitches)
    synthaudio(st.session_state.melody, "melody")
    plot_piano_roll("melody")
    return 

def melody_page():

    st.header('Generate main melody')

    with st.beta_expander('Intro: Generate main melody'):
        st.markdown('''
        First, we want to generate a simple melody. In this task, we use Markov Chains to develop a pitch sequence similar to the style of a transition matrix. 
        909, hooktheory and weimarjazz are three datasets of different music style. We've extracted their melody lines in advance and computed the transition matrix. 
        ''')

    col1, col2, col3 = st.beta_columns(3)

    with col1:
        st.session_state.rank = col1.selectbox('Choose Markov order', [1, 2, 3], 
                                on_change=modify_mc)
    with col2:
        st.session_state.style = col2.selectbox('Choose transition matrix of different style:', ['909', 'hooktheory', 'weimarjazz'],
                                on_change=modify_mc) 

    with col3:
        start_note = col3.selectbox('Starting note', options=PITCHES[:12],
                                key="start_note", on_change=modify_mc)
        if st.session_state.mc.cur == None: 
            st.session_state.mc.cur = PITCHES.index(start_note)

    # if st.button('一个一个来！'):
    #     note = st.session_state.mc.get_next_note()
    #     write_stream(note)
    #     synthaudio(st.session_state.melody, "melody")

    if st.button('Generate at once！'):
        # init_melody()
        # pitch_slots = make_patterns()
        # pitches = st.session_state.mc.generate(PITCHES.index(start_note), pitch_slots)
        
        # # force an ending
        # pitches[-1] = key_sigs[st.session_state.key_signature].tonic.name

        # write_stream(pitches)
        # synthaudio(st.session_state.melody, "melody")
        # plot_piano_roll("melody")
        gen_melody()


    # df = pd.read_csv(f"{assets_dir}/{st.session_state.style}_pitch_markov_{st.session_state.rank}.csv", index_col=0)
    st.dataframe(st.session_state.df.style.format("{:7,.2f}") )


    st.subheader(format_sequence(st.session_state.melody_note_list))


    if st.session_state.melody_note_list:
        st.image(f"{write_dir}/melody.png")

        audio_file = open(f'{write_dir}/melody.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')





