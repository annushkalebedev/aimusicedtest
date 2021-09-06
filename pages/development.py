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

def init_ga():
    weights = {"key_weight": st.session_state.key_weight,
            "smoothing_weight": st.session_state.smoothing_weight,
            "similarity_weight": st.session_state.similarity_weight,
            "rhythm_weight": st.session_state.rhythm_weight}
    
    melody = [m2.note.Note(n).pitch.midi if (n not in ['-', 'R']) else 0 
            for n in st.session_state.melody_note_list]

    st.session_state.ga = GA(
        st.session_state.n_measures, 
        time_sigs[st.session_state.time_signature], 
        weights, 
        melody)
    return 

def development_page():
    st.header('发展旋律')

    with st.beta_expander('介绍：发展旋律'):
        st.markdown('''
        有了主旋律之后，我们希望创造出更多的旋律素材。在这里，我们使用基因算法(Genetic Algorithm)来生成旋律序列。  
        每一条旋律都可以表示成一个由MIDI音高值构成的向量，例如\[64, 62, 0, 64, 64, 65, 67, 0\]。基因算法中，向量可以模拟一个种群，在种群中，不同的向量会进行结合与变异。每一轮的“进化”会根据我们期望的规则(音高或者节奏)，选出种群中比较动听的向量(旋律)。经过反复的迭代，随机初始化的种群中会演变出令人满意的旋律。  
        在这一个任务中，我们生成四条新的旋律素材：M2, M3, B1, B2。
        ''')

    st.subheader('现有主旋律M1：')
    st.subheader(format_sequence(st.session_state.melody_note_list))

    audio_file = open('assets/melody.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

    n_measures = st.session_state.n_measures
    n_per_measure = time_sigs[st.session_state.time_signature]

    init_ga()

    st.markdown("""---""")

    NUM_GEN = 100
    sol_per_pop = 64 # mating pool size
    num_parents_mating = 32 # population size

    cola, colb, colc = st.beta_columns(3)
    with cola:
        st.slider('进化轮数', min_value=20, max_value=256, 
            value= 100, step=2, key="num_gen", on_change=init_ga,
            help="选择遗传算法需要演化的轮数，20到256之间。")

    with colb:
        st.slider('种群大小', min_value=64, max_value=256, 
            value= 64, step=4, key="sol_per_pop", on_change=init_ga,
            help="每一轮用于筛选的序列数量，64到256之间。")

    with colc:
        st.slider('生存大小', min_value=16, max_value=64, 
            value= 32, step=2, key="num_parents_mating", on_change=init_ga,
            help="每一轮筛选后生存的序列数量，32到64之间。")

    st.markdown("""---""")


    col1, col2 = st.beta_columns(2)

    with col1:

        st.slider('调性权重', min_value=0.0, max_value=1.0, 
            value= 0.5, step=0.05, key="key_weight", on_change=init_ga,
            help="根据序列中的音高是否在调内增加惩罚。调性权重越高，越容易生成出符合调性的旋律。")
        st.slider('平滑度权重', min_value=0.0, max_value=1.0, 
            value= 0.5, step=0.05, key="smoothing_weight", on_change=init_ga,
            help="旋律的平滑度，指音和音之间距离(音程)的大小。平滑度权重越高，旋律越平缓(好唱)。")

        st.markdown("""---""")

        if col1.button('生成旋律M2！'):
            init_melody_m2()
            with st.spinner(text='In progress'):
                solution = generate_development(st.session_state.ga)
                write_stream(solution, "melody_m2")
                synthaudio(st.session_state.melody_m2, "melody_m2")
        
        st.markdown(format_sequence(st.session_state.melody_m2_note_list))

        if st.session_state.melody_m2_note_list:
            audio_file = open('assets/melody_m2.wav', 'rb')
            audio_bytes = audio_file.read()
            col1.audio(audio_bytes, format='audio/wav')

        if col1.button('生成过渡B1！'):
            init_melody_b1()
            with st.spinner(text='In progress'):
                solution = generate_development(st.session_state.ga,
                    bridge=True)
                write_stream(solution, "melody_b1", melody="B1")
                synthaudio(st.session_state.melody_b1, "melody_b1")

        st.markdown(format_sequence(st.session_state.melody_b1_note_list))

        if st.session_state.melody_b1_note_list:
            audio_file = open('assets/melody_b1.wav', 'rb')
            audio_bytes = audio_file.read()
            col1.audio(audio_bytes, format='audio/wav')


    with col2:
        st.slider('相似度权重', min_value=0.0, max_value=1.0, 
            value= 0.5, step=0.05, key="similarity_weight", on_change=init_ga,
            help="相似度指的是和主旋律的相似度，权重越高，生成的旋律和主旋律越像。")
        st.slider('节奏权重', min_value=0.0, max_value=1.0, 
            value= 0.5, step=0.05, key="rhythm_weight", on_change=init_ga,
            help="节奏权重指的是拍点和重拍上的音。权重越高，生成的旋律会更有节奏感。")

        st.markdown("""---""")


        if col2.button('生成旋律M3！'):
            init_melody_m3()
            with st.spinner(text='In progress'):
                solution = generate_development(st.session_state.ga)
                write_stream(solution, "melody_m3", melody="M3")
                synthaudio(st.session_state.melody_m3, "melody_m3")
        
        st.markdown(format_sequence(st.session_state.melody_m3_note_list))

        if st.session_state.melody_m3_note_list:
            audio_file = open('assets/melody_m3.wav', 'rb')
            audio_bytes = audio_file.read()
            col2.audio(audio_bytes, format='audio/wav')

        if col2.button('生成过渡B2！'):
            init_melody_b2()
            with st.spinner(text='In progress'):
                solution = generate_development(st.session_state.ga,
                    bridge=True)
                write_stream(solution, "melody_b2", melody="B2")
                synthaudio(st.session_state.melody_b2, "melody_b2")
        
        st.markdown(format_sequence(st.session_state.melody_b2_note_list))

        if st.session_state.melody_b2_note_list:
            audio_file = open('assets/melody_b2.wav', 'rb')
            audio_bytes = audio_file.read()
            col2.audio(audio_bytes, format='audio/wav')


