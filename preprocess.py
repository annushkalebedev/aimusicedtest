import os
import random, json
import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st 
import numpy as np
import music21 as m2
from tqdm import tqdm
import torch.nn as nn
from hmmlearn import hmm
from midi2audio import FluidSynth

from markov import *
import crash
from params import *
import pickle
import warnings


# BASE_DIR = "../POP909-Dataset-master/POP909"
# BASE_DIR = "../Hooktheory"
# BASE_DIR = "../htmini"
BASE_DIR = "../WeimarJazz"


# read pop909 datasets into token
def read_melody_token(dataset="909"):
    with open("assets/token_lists_weimarjazz_mini.txt", "w") as f:
        for dir_name in tqdm(os.listdir(BASE_DIR)[:100]):
            if ".xlsx" in dir_name:
                continue
            token_list = []
            if dataset == "909":
                midi_path = "{}/{}/{}.mid".format(BASE_DIR, dir_name, dir_name)                
            else:
                midi_path = "{}/{}".format(BASE_DIR, dir_name)

            if midi_path == '../WeimarJazz/JohnColtrane_Impressions_1961_FINAL.mid':
                continue
            try:
                score = m2.converter.parse(midi_path)
            except Exception as e:
                print(e)
                continue
            melody = score.parts[0]
            key = score.analyze('key').tonic

            if "minor" in key.name:
                continue
            if "-" in key.name:
                key = key.getEnharmonic()
            key_idx = PITCHES.index(key.name)

            # transpose all to C major
            tmelody = melody.transpose(-key_idx)
            for event in tmelody:
                if isinstance(event, m2.note.Note) or isinstance(event, m2.note.Rest):
                    name = (event.pitch.getEnharmonic().name 
                        if "-" in event.name else event.name)
                    token_list.append("R" if name == 'rest' else name)
                    token_list.extend(["-"] * int(event.duration.quarterLength / 0.5 - 1))
            f.write(" ".join(token_list))
            f.write("\n")
            # except:
            #     pass

    f.close()


# read melody-chords sequence into txt
def read_chords_sequence(dataset="hooktheory"):
    with open("assets/chords_lists_hooktheory.txt", "w") as f:
        for dir_name in tqdm(os.listdir(BASE_DIR)):
            if ".xlsx" in dir_name:
                continue
            print(dir_name)
            token_list, chord_list = [], []
            midi_path = "{}/{}".format(BASE_DIR, dir_name)

            try:
                score = m2.converter.parse(midi_path)
            except:
                print("parse failed")
                continue
            key = score.analyze('key').tonic
            if "-" in key.name:
                key = key.getEnharmonic()
            key_idx = PITCHES.index(key.name)

            # transpose all to C major
            tscore = score.transpose(-key_idx)

            if len(tscore.parts) < 2:
                print("not enough parts")
                continue

            tmelody, tharmony = tscore.parts
            tchords = [event for event in tharmony if isinstance(event, m2.chord.Chord)]

            offsets = [event.offset for event in tchords]
            for event in tmelody:
                if isinstance(event, m2.note.Note) and not (event.offset % 1.0):
                    if (event.offset in offsets):
                        chord_idx = offsets.index(event.offset)  
                        c = tchords[chord_idx]
                        if c.quality == "other":
                            if 'major' in c.pitchedCommonName:
                                quality = 'major'
                            elif 'minor' in c.pitchedCommonName:
                                quality = 'minor'
                            else:
                                continue
                        else:
                            quality = c.quality
                        cname = f"{c.root().name}-{quality}"
                        chord_list.append(cname)
                    else:
                        continue
                    name = (event.pitch.getEnharmonic().name 
                        if "-" in event.name else event.name)
                    token_list.append("R" if name == 'rest' else name)                        
            assert(len(token_list) == len(chord_list))
            f.write(" ".join(token_list) + "|" + " ".join(chord_list))
            f.write("\n")

    f.close()

# generate markov chain matrix from sequences
def gen_matrix(order=1):
    N = 12 ** order
    if order == 1:
        indexes = PITCHES
    elif order == 2:
        indexes = [f"{p1}{p2}" for p1 in PITCHES for p2 in PITCHES]
    else:
        indexes = [f"{p1}{p2}{p3}" for p1 in PITCHES for p2 in PITCHES for p3 in PITCHES]
    matrix = np.zeros((N, 12))
    with open("assets/token_lists_weimarjazz_mini.txt", "r") as f:
        for tokens in f.read().splitlines():
            token_list = tokens.split(" ")
            token_list = [t for t in token_list if (t not in ["-", "R"])]
            if order == 1:
                pairs = zip(token_list, token_list[1:])
            elif order == 2:
                pairs = zip(zip(token_list, token_list[1:]), token_list[2:])
            else:
                pairs = zip(zip(zip(token_list, token_list[1:]), token_list[2:]), token_list[3:])
            for x, y in pairs:
                x = "".join(x if order == 2 else [k for xx in x for k in xx]) if order != 1 else x
                # crash()
                matrix[indexes.index(x), PITCHES.index(y)] += 1
    f.close()

    denom = np.sum(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, denom, where=denom!=0)
    df = pd.DataFrame(matrix, columns=PITCHES, index=indexes)
    df.to_csv(f"assets/weimarjazz_pitch_markov_{order}.csv")
    return


'''
train the hmm with melody sequences
match states to 24 chords (2 triads * 12 scale degrees)
return the trained model
'''
def train_hmm():
    chord_labels = dict()

    with open("assets/chords_lists_hooktheory.txt", "r") as f:
        all_melody, all_length, all_harmony = [], [], []
        for line in f.read().splitlines():
            melody, harmony = line.split("|")

            # [[0], [7], [4]]
            melody = [[PITCHES.index(m)] for m in melody.split()]
            if len(melody) > 5:
                all_melody.append(melody)
                all_length.append(len(melody))
                # ["Am", "C", "Am"]
                all_harmony.append((harmony.split(" ")))

    cutoff = 5000

    # remodel = hmm.GaussianHMM(n_components=12, covariance_type="full", n_iter=100)
    # remodel.fit([m for x in all_melody[:cutoff] for m in x], lengths=all_length[:cutoff])

    with open("assets/hmm.pkl", "rb") as file: 
        remodel = pickle.load(file)

    all_pairs = [zip(remodel.predict(melody), all_harmony[i]) for i, melody in enumerate(all_melody[:cutoff])]
    all_pairs = set([pair for seq in all_pairs for pair in seq if (
        pair[1].split("-")[-1] in ["major", "minor"])])

    chord_labels = dict()
    for idx, chord in tqdm(all_pairs):
        if idx not in chord_labels and chord not in chord_labels.values():
            chord_labels[idx] = chord
    # chord_labels = dict(all_pairs)
    print(chord_labels)
    # crash()

    with open("assets/states.pkl", "wb") as file:
        pickle.dump(chord_labels, file)
    with open("assets/state_chord_pair.pkl", "wb") as file:
        pickle.dump(all_pairs, file)
    with open("assets/hmm.pkl", "wb") as file: 
        pickle.dump(remodel, file)

    return remodel


if __name__ == '__main__':

    # extracts to txt
    # read_melody_token("weimarjazz")
    # read_chords_sequence()

    # gen_matrix(order=3)
    
    # with open("assets/states.pkl", "rb") as file:
    #     chord_labels = pickle.load(file)
    #     crash()

    train_hmm()
