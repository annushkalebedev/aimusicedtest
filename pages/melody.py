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


def melody_page():

    st.header('旋律生成')

    with st.beta_expander('介绍：主旋律生成'):
        st.markdown('''
        首先，我们希望生成一条简单的旋律。在这个任务中，我们将使用马尔科夫链来生成前后相关的音高序列，即通过一个转接概率矩阵(transition matrix)，得到基于前N个音后，下一个音的概率。  
        909, hooktheory和weimarjazz是三个不同风格的数据集。我们已经预先提取了数据中的旋律并且得到其中的转接概率矩阵，不同的风格中，得到音的概率也不一样。
        在生成音高序列之外，我们的节奏型是从模板而得到的。
        ''')

    col1, col2, col3 = st.beta_columns(3)

    with col1:
        st.session_state.rank = col1.selectbox('选择马尔科夫阶数', [1, 2, 3], 
                                on_change=modify_mc)
    with col2:
        st.session_state.style = col2.selectbox('选择不同风格的的概率矩阵:', ['909', 'hooktheory', 'weimarjazz'],
                                on_change=modify_mc) 

    with col3:
        start_note = col3.selectbox('初始音高', options=PITCHES[:12],
                                key="start_note", on_change=modify_mc)
        if st.session_state.mc.cur == None: 
            st.session_state.mc.cur = PITCHES.index(start_note)

    # if st.button('一个一个来！'):
    #     note = st.session_state.mc.get_next_note()
    #     write_stream(note)
    #     synthaudio(st.session_state.melody, "melody")

    if st.button('一次生成！'):
        init_melody()
        pitch_slots = make_patterns()
        pitches = st.session_state.mc.generate(PITCHES.index(start_note), pitch_slots)
        
        # force an ending
        pitches[-1] = key_sigs[st.session_state.key_signature].tonic.name

        write_stream(pitches)
        synthaudio(st.session_state.melody, "melody")
        plot_piano_roll("melody")


    # df = pd.read_csv(f"assets/{st.session_state.style}_pitch_markov_{st.session_state.rank}.csv", index_col=0)
    st.dataframe(st.session_state.df.style.format("{:7,.2f}") )


    st.subheader(format_sequence(st.session_state.melody_note_list))


    if st.session_state.melody_note_list:
        st.image("assets/melody.png")

        audio_file = open('assets/melody.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')





