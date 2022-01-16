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
        st.error('Error in input. Please strictly follow the syntax of [left] -> [right] to build your grammar. ')
        st.stop()
    graph = pydot.Dot("my_graph", graph_type="graph", bgcolor="#00000000")
    sentence = generate_full_melody(grammar, graph)
    graph.write_png(f'{write_dir}/graph.png')
    write_stream(sentence)
    synthaudio(st.session_state.full_melody, "full_melody")


    return 

def structure_page():
    st.header('Structure')

    with st.beta_expander('Intro: Structure'):
        st.markdown('''
        We are going to use Generative Grammar to find out the structure and repeats of the song.
        Generative Grammar, in summary, is the set of rules to characterize language syntax. For example, S → NP + VP implies that a Sentence(S) is built up by a Noun Phrase (NP) and a Verb Phrase (VP).
        Similarly, for the syntax of music, we define:  
        S：Song  
        V：Verse  
        C：Chorus  
        M1：Melody 1  
        M2：Melody 2  
        M3：Melody 3  
        B1：Bridge 1  
        B2：Bridge 2  
        For example, S -> V C C represents that a full song can be expanded as Verse + Chorus + Chorus. Verse and Chorus, in turn, can be expanded similarly from other rules into the melodies we generated before.  
        In the input box below, you can use [left] -> [right] to define your rules of expansion. We provide an example here: ''')

    st.session_state.grammar_input = st.text_area('Define your generative grammar！',
        height=200, 
        value='''S -> V C C
S -> V C V C 
V -> M1 M2
V -> B1 M2
C -> M3 B2
C -> M1 M3 M3 B2''')

    if st.button('Parse + Generate!'):
        init_full_melody()
        try:
            grammar = parse_grammar(st.session_state.grammar_input)
        except:
            st.error('Error in input. Please strictly follow the syntax of [left] -> [right] to build your grammar. ')
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
        with st.beta_expander('Full song'):
            st.text(format_sequence(st.session_state.full_melody_note_list))


        audio_file = open(f'{write_dir}/full_melody.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

