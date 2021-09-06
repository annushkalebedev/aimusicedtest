import os, sys
import random, json
import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st 
import numpy as np
import music21 as m2
import pretty_midi
import matplotlib.pyplot as plt
import librosa.display
from hmmlearn import hmm
from midi2audio import FluidSynth

from markov import *
import crash
from params import *
import pickle
import warnings


def warn(*args, **kwargs):
    pass
warnings.warn = warn

def init_melody():
    st.session_state.melody = m2.stream.Part()
    st.session_state.melody.append([
        st.session_state.melody_instr,
        m2.tempo.MetronomeMark(number=st.session_state.tempo)]) 
    st.session_state.melody_note_list = []     
    return 

def init_harmony():
    st.session_state.harmony = m2.stream.Part()
    st.session_state.harmony.append([
        st.session_state.harmony_instr,
        m2.tempo.MetronomeMark(number=st.session_state.tempo)])   
    return 

def init_melody_m2():
    st.session_state.melody_m2 = m2.stream.Part()
    st.session_state.melody_m2.append([
        st.session_state.melody_instr,
        m2.tempo.MetronomeMark(number=st.session_state.tempo)]) 
    st.session_state.melody_m2_note_list = []  
    return 

def init_melody_m3():
    st.session_state.melody_m3 = m2.stream.Part()
    st.session_state.melody_m3.append([
        st.session_state.melody_instr,
        m2.tempo.MetronomeMark(number=st.session_state.tempo)]) 
    st.session_state.melody_m3_note_list = []  
    return 

def init_melody_b1():
    st.session_state.melody_b1 = m2.stream.Part()
    st.session_state.melody_b1.append([
        st.session_state.melody_instr,
        m2.tempo.MetronomeMark(number=st.session_state.tempo)]) 
    st.session_state.melody_b1_note_list = []  
    return 

def init_melody_b2():
    st.session_state.melody_b2 = m2.stream.Part()
    st.session_state.melody_b2.append([
        st.session_state.melody_instr,
        m2.tempo.MetronomeMark(number=st.session_state.tempo)]) 
    st.session_state.melody_b2_note_list = []  
    return 

def init_full_melody():
    st.session_state.full_melody = m2.stream.Part()
    st.session_state.full_melody.append([
        st.session_state.melody_instr,
        m2.tempo.MetronomeMark(number=st.session_state.tempo)]) 
    st.session_state.full_melody_note_list = []  
    return 

def init_drum():
    st.session_state.drum = m2.stream.Part()
    return 

def init_score():
    st.session_state.score = m2.stream.Stream()
    return 

# Synth a music21 stream to midi then to audio 
def synthaudio(stream, name):
    mf = m2.midi.translate.streamToMidiFile(stream)
    # make the last track drum channel
    if len(mf.tracks) >= 3:
        for e in mf.tracks[-1].events:
            e.channel = 10
    mf.open(f'assets/{name}.mid', 'wb')
    mf.write()
    mf.close()
    fs = FluidSynth()
    fs.midi_to_audio(f'assets/{name}.mid', f'assets/{name}.wav')
    return 

# input: pretty_midi data
def get_low_high(midi_data):
    low, high = float("inf"), float("-inf")
    for ins in midi_data.instruments:
        pitches = [note.pitch for note in ins.notes]
        low, high = min(min(pitches), low), max(max(pitches), high)
    return low, high

# save the pianoroll image
def plot_piano_roll(name):
    midi_data = pretty_midi.PrettyMIDI(f'assets/{name}.mid')
    low, high = get_low_high(midi_data)

    fig, ax = plt.subplots(figsize=(12,5))
    piano_roll = midi_data.get_piano_roll()[low-1:high+1]
    np.set_printoptions(threshold=sys.maxsize)
    x_coords = np.arange(piano_roll.shape[1])
    librosa.display.specshow(piano_roll, ax=ax,
                         hop_length=1,  
                         y_axis='cqt_note',
                         # x_coords=x_coords,
                         fmin=pretty_midi.note_number_to_hz(low-1),
                         cmap="Greens")
    ax.set_xticklabels([])
    fig.savefig(f'assets/{name}.png', transparent=True)
    return 

# plot both the melody and melody_b piano roll together
def contrast_piano_roll():

    return 

# input a sequence of notes and format it into measures 
# return str
def format_sequence(sequence, n_per_measure=None):
    if not n_per_measure:
        n_per_measure = time_sigs[st.session_state.time_signature]
    measures = [" ".join(sequence[i:i+n_per_measure]) for i in range(
        0, len(sequence), n_per_measure)]
    formatted = [" | ".join(measures[i:i+8]) + " |\n" for i in range(
        0, len(measures), 8)]

    return "".join(formatted)


def prepare_state():

    if 'melody_instr' not in st.session_state:
        st.session_state.melody_instr = m2.instrument.Saxophone()
    if 'harmony_instr' not in st.session_state:
        st.session_state.harmony_instr = m2.instrument.AcousticGuitar()

    if 'start_note' not in st.session_state:
        st.session_state.start_note = "C"
    if 'melody' not in st.session_state:
        init_melody()
    if 'harmony' not in st.session_state:
        init_harmony()
    if 'melody_m2' not in st.session_state:
        init_melody_m2()
    if 'melody_m3' not in st.session_state:
        init_melody_m3()
    if 'melody_b1' not in st.session_state:
        init_melody_b1()
    if 'melody_b2' not in st.session_state:
        init_melody_b2()
    if 'full_melody' not in st.session_state:
        init_full_melody()
    if 'drum' not in st.session_state:
        init_drum()
    if 'score' not in st.session_state:
        init_score()

    if 'melody_note_list' not in st.session_state:
        st.session_state.melody_note_list = []

    if 'melody_m2_note_list' not in st.session_state:
        st.session_state.melody_m2_note_list = []
    if 'melody_m3_note_list' not in st.session_state:
        st.session_state.melody_m3_note_list = []
    if 'melody_b1_note_list' not in st.session_state:
        st.session_state.melody_b1_note_list = []
    if 'melody_b2_note_list' not in st.session_state:
        st.session_state.melody_b2_note_list = []
    if 'full_melody_note_list' not in st.session_state:
        st.session_state.full_melody_note_list = []
    if 'pure_chords' not in st.session_state:
        st.session_state.pure_chords = []
    if 'rank' not in st.session_state:
        st.session_state.rank = 1
    if 'style' not in st.session_state:
        st.session_state.style = '909'
    if 'key_signature' not in st.session_state:
        st.session_state.style = 'C大调'
    if 'mc' not in st.session_state:
        df = pd.read_csv(f"assets/{st.session_state.style}_pitch_markov_{st.session_state.rank}.csv", index_col=0)
        st.session_state.mc = MarkovChain(np.array(df))
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv(f"assets/{st.session_state.style}_pitch_markov_{st.session_state.rank}.csv", index_col=0)
    if 'ga' not in st.session_state:
        st.session_state.ga = None
    if 'key_weight' not in st.session_state:
        st.session_state.key_weight = 0.5
    if 'smoothing_weight' not in st.session_state:
        st.session_state.smoothing_weight =  0.5   
    if 'similarity_weight' not in st.session_state:
        st.session_state.similarity_weight = 0.5
    if 'rhythm_weight' not in st.session_state:
        st.session_state.rhythm_weight = 0.5 

# When signature changes, clear the markov chain and start again
def modify_mc():
    df = pd.read_csv(f"assets/{st.session_state.style}_pitch_markov_{st.session_state.rank}.csv", index_col=0)
    
    key_idx = PITCHES.index(st.session_state.key_signature[0])
    key_pitches = PITCHES[key_idx:] + PITCHES[:key_idx] # scale of that key
    df.columns = key_pitches
    df.index = key_pitches
    df = df[PITCHES]
    df = df.reindex(PITCHES)

    st.session_state.df = df
    st.session_state.mc = MarkovChain(np.array(df))
    print("mc reassigned")

    st.session_state.melody_note_list = []
    init_melody()

    return 


# When tempo changes, resynthesize the audio
def modify_tempo():

    if isinstance(st.session_state.melody[1], m2.tempo.MetronomeMark):
        st.session_state.melody.pop(1)
    st.session_state.melody.insert(1, m2.tempo.MetronomeMark(
                                number=st.session_state.tempo))

    if isinstance(st.session_state.harmony[1], m2.tempo.MetronomeMark):
        st.session_state.harmony.pop(1)
    st.session_state.harmony.insert(1, m2.tempo.MetronomeMark(
                                number=st.session_state.tempo))
    
    st.session_state.score = m2.stream.Stream()
    st.session_state.score.insert(0, st.session_state.full_melody)
    st.session_state.score.insert(0, st.session_state.harmony)

    synthaudio(st.session_state.melody, "melody")
    synthaudio(st.session_state.score, "score")


    return 




