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

    st.header('AI作曲教学系统')

    st.markdown('''
        从左边的任务列表中，你可以选择不同的任务：**主旋律生成**，**发展旋律生成**，**曲子结构生成**，**和弦生成**。其中每一个乐曲，都对应着不同的算法，等待你去探索。   
        请从**主旋律生成**任务开始，创作你的音乐吧。  
        当然，你可以选择下面的一键生成，完成曲子所有部分的创作(进入每个单独任务界面查看结果)。
        ''')


    if st.button('一键生成！'):
        with st.spinner(text='In progress'):
            gen_melody()
            gen_alldev()
            gen_structure()
            gen_harmony()

        if st.session_state.pure_chords:
            audio_file = open(f'{write_dir}/score.wav', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')





