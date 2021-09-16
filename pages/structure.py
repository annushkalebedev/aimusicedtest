import os, time
import sys
sys.path.append("..")
import streamlit as st 
import copy
import random
import numpy as np
import music21 as m2
from nltk.parse.generate import generate
from nltk.grammar import Nonterminal
from nltk import CFG
import pydot
import crash
from utils import * 
from params import *

# write full_melody into music21 stream
def write_stream(sentence):

    melody_map = {'M1':st.session_state.melody, 
                'M2':st.session_state.melody_m2, 
                'M3':st.session_state.melody_m3, 
                'B1':st.session_state.melody_b1, 
                'B2':st.session_state.melody_b2}

    melody_map_note_list = {'M1':st.session_state.melody_note_list, 
                'M2':st.session_state.melody_m2_note_list, 
                'M3':st.session_state.melody_m3_note_list, 
                'B1':st.session_state.melody_b1_note_list, 
                'B2':st.session_state.melody_b2_note_list}

    for phrase in sentence:
        print(phrase)
        melody = copy.deepcopy(melody_map[phrase])
        for n in list(melody.notesAndRests):
            st.session_state.full_melody.append(n)
        st.session_state.full_melody_note_list.extend(
            melody_map_note_list[phrase])

    return

def plot_tree(trace):
    print(trace)
    graph = pydot.Dot("my_graph", graph_type="graph")

    for i, production in enumerate(trace):
        l_node = pydot.Node(str(i), label=production.lhs().symbol(), shape="plaintext")
        graph.add_node(l_node)
        for j, item in enumerate(production.rhs()):
            r_node = pydot.Node(str(i)+str(j), label=item.symbol(), shape="plaintext")
            graph.add_node(r_node)
            graph.add_edge(pydot.Edge(l_node, r_node))

    graph.write_png(f'{write_dir}/graph.png')

    return

# parse the grammar user input using nltk
def parse_grammar(user_input):
    grammar = CFG.fromstring(user_input)
    return grammar

# generate the full melody using given grammar, recursive
def generate_full_melody(grammar, graph, node=None, idx="0_0_0", items=["S"]):
    frags = []
    level, key, _ = idx.split("_")
    if len(items) == 1:
        if isinstance(items[0], str) and items[0] not in TERMINALS:
            expansions = [prod for prod in grammar.productions() if prod.lhs().symbol() == items[0]]
            expansion = random.choice(expansions)
            if items == ["S"]:
                l_node = pydot.Node(idx, label=expansion.lhs().symbol(), shape="plaintext")
                graph.add_node(l_node)
            else:
                l_node = node
            nodes = []
            for j, item in enumerate(expansion.rhs()):
                r_node = pydot.Node(f"{str(int(level)+1)}_{key}_{j}", label=item.symbol(), shape="plaintext")
                graph.add_node(r_node)
                graph.add_edge(pydot.Edge(l_node, r_node))
                nodes.append(r_node)

            subfrags = generate_full_melody(grammar, graph, 
                node=nodes, idx=f"{str(int(level)+1)}_{key}_?", items=expansion.rhs())
            frags.extend(subfrags)
        else:
            frags.append(items[0] if isinstance(items[0], str) else items[0].symbol())
    else:
        for key, term in enumerate(items):
            subfrags = generate_full_melody(grammar, graph, 
                node=node[key], idx=f"{level}_{str(key)}_-1",
                items=[term if isinstance(term, str) else term.symbol()])
            frags.extend(subfrags)
    return frags

def gen_structure():
    init_full_melody()
    try:
        grammar = parse_grammar(st.session_state.grammar_input)
    except:
        st.error('输入的格式有误。请按照[左] -> [右]来定义你的语法。')
        st.stop()
    graph = pydot.Dot("my_graph", graph_type="graph", bgcolor="#00000000")
    sentence = generate_full_melody(grammar, graph)
    graph.write_png(f'{assets_dir}/graph.png')
    write_stream(sentence)
    synthaudio(st.session_state.full_melody, "full_melody")


    return 

def structure_page():
    st.header('结构')

    with st.beta_expander('介绍'):
        st.markdown('''
        我们将用生成语法(Generative Grammar)来定义整首曲子的结构。  
        生成语法总体上认为，应该假设一套规则来描写语言能力。例如用语类规则来描述句子、词组等各种语类的内部结构，用 S → NP + VP来表示一个句子，句子(S)由一个名词词组(NP)加上一个动词词组(VP)构成。这种精确的，形式化的语法称为生成语法。  
        对于音乐的结构而言，我们定义：  
        S：歌曲  
        V：主歌  
        C：副歌  
        M1：乐句1  
        M2：乐句2  
        M3：乐句3  
        B1：衔接1  
        B2：衔接2  
        S -> V C C 则表示 整首歌 可以展开为 主歌 副歌 副歌。主歌和副歌则可以按照其他的规则展开成我们在前一部分生成的乐句。  
        在下面的输入框中，可以按照[左] -> [右]来定义你自己的音乐结构。一个例子如下：''')

    st.session_state.grammar_input = st.text_area('定义你的生成语法！',
        height=200, 
        value='''S -> V C C
S -> V C V C 
V -> M1 M2
V -> B1 M2
C -> M3 B2
C -> M1 M3 M3 B2''')

    if st.button('解析+生成！'):
        init_full_melody()
        try:
            grammar = parse_grammar(st.session_state.grammar_input)
        except:
            st.error('输入的格式有误。请按照[左] -> [右]来定义你的语法。')
            st.stop()
        graph = pydot.Dot("my_graph", graph_type="graph", bgcolor="#00000000")
        sentence = generate_full_melody(grammar, graph)
        graph.write_png(f'{write_dir}/graph.png')
        # plot_tree(trace)
        st.text(sentence)
        with st.spinner(text='In progress'):
            write_stream(sentence)
            synthaudio(st.session_state.full_melody, "full_melody")
    
    if st.session_state.full_melody_note_list:
        st.image(f"{write_dir}/graph.png")
        with st.beta_expander('全曲'):
            st.text(format_sequence(st.session_state.full_melody_note_list))


        audio_file = open(f'{write_dir}/full_melody.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

