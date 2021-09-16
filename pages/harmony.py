import os, time
import sys
sys.path.append("..")
import pickle
import torch
import streamlit as st 
import random
import numpy as np
import music21 as m2
import crash
from hmmlearn import hmm
from utils import * 
from params import *
from RNN import *


# write the harmony into the music21 stream
def write_stream(chords):
    if len(st.session_state.harmony) > 2:
        st.session_state.harmony = m2.stream.Part()
        st.session_state.harmony.insert(m2.instrument.AcousticGuitar())
    for chord in chords:

        if chord == "R":
            # st.session_state.harmony.append(m2.note.Rest(quarterLength=EIGHTH))
            pass
        else:
            root, kind = chord.split("-")
            c = m2.chord.Chord(m2.harmony.ChordSymbol(root=root, kind=kind).pitches,
                    quarterLength=EIGHTH * time_sigs[st.session_state.time_signature])
            c.volume.velocity = 50
            st.session_state.harmony.append(c)

    init_score()
    st.session_state.score.insert(0, st.session_state.full_melody)
    st.session_state.score.insert(0, st.session_state.harmony)
    return 


# return: pitches as a list of indexes
def get_downbeat_pitch():
    pitches, indexes, prev = [], [], "C"
    for i, note in enumerate(st.session_state.full_melody_note_list):
        if note in ["R", "-"] and st.session_state.full_melody_note_list[i-1] not in ["R", "-"]:
            prev = st.session_state.full_melody_note_list[i-1]
        if (not (i % time_sigs[st.session_state.time_signature])):
            pitches.append([PITCHES.index(note if (note not in ["R", "-"]) else prev)])
            indexes.append(i)

    return indexes, pitches


def generate_chords_hmm(seed):

    with open(f"{assets_dir}/state_chord_pair.pkl", "rb") as file:
        all_pairs = pickle.load(file)
    with open(f"{assets_dir}/hmm.pkl", "rb") as file: 
        remodel = pickle.load(file)

    all_pairs = list(all_pairs)
    random.Random(seed).shuffle(all_pairs)
    chord_labels = dict()
    for idx, chord in (all_pairs):
        if idx not in chord_labels and chord not in chord_labels.values():
            if "E--" in chord:
                chord = chord.replace("E-", "D#")
            if "B--" in chord:
                chord = chord.replace("B-", "A#")
            chord_labels[idx] = chord
    # chord_labels = dict(all_pairs)
    print(chord_labels)

    indexes, pitches = get_downbeat_pitch()
    preds = remodel.predict(pitches)
    chords = [chord_labels[pred] for pred in preds]
    pure_chords = chords
    print(chords)

    # add rests
    chords = [chords[indexes.index(i)] if i in indexes else 'R' for i in range(
        len(st.session_state.full_melody_note_list))]

    return pure_chords, chords

def generate_chords_rnn(temperature):

    model = Seq2Seq(Encoder(), Decoder())
    model.load_state_dict(torch.load(f"{assets_dir}/rnnmodel.pt")['model_state_dict'])
    model.eval()

    indexes, pitches = get_downbeat_pitch()

    melody = F.one_hot(torch.tensor(pitches).squeeze(1), num_classes=INPUT_DIM).float()
    outputs = model(melody.unsqueeze(0)) # outputs: (1, seq_len, out_dim)
    outputs = F.softmax(outputs.squeeze(0) / temperature, dim=1)
    outputs = torch.multinomial(outputs, 1).squeeze(1)
    # print(outputs.shape)

    chords = [CHORDS[pred] for pred in outputs]

    # transpose to the key
    new_chords = []
    for c in chords:
        idx = PITCHES.index(c.split("-")[0])
        semitones = list(key_sigs.keys()).index(st.session_state.key_signature) * 7 % 12
        new_root = PITCHES[(idx + semitones) % 12]
        new_chords.append("{}-{}".format(new_root, c.split("-")[1]))

    chords = new_chords
    pure_chords = chords
    print(chords)

    # add rests
    chords = [chords[indexes.index(i)] if i in indexes else 'R' for i in range(
        len(st.session_state.full_melody_note_list))]

    return pure_chords, chords

def gen_harmony():
    st.session_state.pure_chords, chords = generate_chords_rnn(st.session_state.temperature)
    write_stream(chords)
    synthaudio(st.session_state.harmony, "harmony")
    synthaudio(st.session_state.score, "score")
    plot_piano_roll("score")

    return 

def harmony_page():
    st.header('和弦')

    with st.beta_expander('介绍：和弦页'):
        st.markdown('''
        根据刚刚生成的旋律们，我们为其配上和弦，形成一首完整的曲目。 
        在这里，我们将旋律视作一个序列，而和弦则是与之相对应的新序列。从序列到序列的任务被称作Seq2Seq，并且在不少机器学习任务中得到应用。 
        我们的 旋律->和弦 任务将由不同的序列模型来实现：HMM, RNN 和 Transformer。
        ''')


    # st.subheader('全曲：')
    # st.subheader(" ".join(st.session_state.full_melody_note_list))

    with st.beta_expander('全曲'):
        # print(format_sequence(st.session_state.full_melody_note_list))
        st.text(format_sequence(st.session_state.full_melody_note_list))

    audio_file = open(f'{write_dir}/full_melody.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

    col1, col2 = st.beta_columns(2)

    with col1:

        seed = st.slider("随机种子", min_value=0, max_value=100, value= 42, step=1,
                help="HMM决定隐藏状态后，将隐藏状态和和弦随机对应起来。")

        if st.button('HMM'):
            st.session_state.pure_chords, chords = generate_chords_hmm(seed)
            write_stream(chords)
            synthaudio(st.session_state.harmony, "harmony")
            synthaudio(st.session_state.score, "score")
            plot_piano_roll("score")

    with col2:

        temperature = st.slider("采样温度", min_value=0.05, max_value=1.0, value= 0.7, step=0.05,
                        key="temperature",
                        help="采样温度, temperature sampling，指的是将预测的概率分布除以一个温度。当温度更高时选择更加发散，温度更低时分布更加保守集中。")

        if st.button('RNN'):
            st.session_state.pure_chords, chords = generate_chords_rnn(temperature)
            write_stream(chords)
            synthaudio(st.session_state.harmony, "harmony")
            synthaudio(st.session_state.score, "score")
            plot_piano_roll("score")

    if st.session_state.pure_chords:
        st.text(format_sequence(st.session_state.pure_chords, n_per_measure=1))
        st.image(f"{write_dir}/score.png")

        audio_file = open(f'{write_dir}/score.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')






