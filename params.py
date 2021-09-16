import pandas as pd
import streamlit as st 
import random
import numpy as np
import music21 as m2

assets_dir = "assets"
write_dir = "/tmp"


PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORDS = [f"{pitch}-{quality}" for pitch in PITCHES[:12] for quality in ('major', 'minor')]
CHORDS2 = [f"{pitch}{quality}" for pitch in PITCHES[:12] for quality in ('maj', 'min')]

EIGHTH = 0.5

time_sigs = {"2/4": 4, "3/4": 6, "4/4": 8, "6/8": 6, "9/8": 9, "12/8": 12}

key_sigs = {"C大调": m2.key.Key("C"), "G大调": m2.key.Key("G"),
			"D大调": m2.key.Key("D"), "A大调": m2.key.Key("A"),
			"E大调": m2.key.Key("E")}

instr_trans = {
	'萨克斯': m2.instrument.Saxophone(),
	'长笛': m2.instrument.Flute(),
	'双簧管': m2.instrument.Oboe(),
	'小提琴': m2.instrument.Violin(),
	'中提琴': m2.instrument.Viola(),
	'人声': m2.instrument.Vocalist(),
	'电吉他': m2.instrument.ElectricGuitar(),

	'木吉他': m2.instrument.AcousticGuitar(),
	'钢琴': m2.instrument.Piano(),
	'大键琴': m2.instrument.Clavichord(),
	'口琴': m2.instrument.Harmonica(),
	'竖琴': m2.instrument.Harp(),
	'马林巴': m2.instrument.Marimba(),
	'弦乐': m2.instrument.StringInstrument()
}


# Evolution algorithm

# Generative Grammar
TERMINALS = ['M1', 'M2', 'M3', 'B1', 'B2']


# Sequential Models
INPUT_DIM = 12
HIDDEN_DIM = 24
OUTPUT_DIM = 24
N_LAYERS = 2
BATCH_SIZE = 32
SEQ_LEN = 15
N_EPOCHS = 100

# rhythmic patterns
r24_1 = ["", "-", "", "-"]
r24_2 = ["", "", "", ""]
r24_3 = ["", "-", "-", ""]
r24_4 = ["", "R", "", ""]
r24_5 = ["", "", "", "-"]

r34_1 = ["", "-", "", "-", "", "-"]
r34_2 = ["", "", "", "", "", ""]
r34_3 = ["", "-", "-", "", "", ""]
r34_4 = ["", "-", "", "", "-", ""]
r34_5 = ["", "", "", "", "-", "-"]
r34_6 = ["", "-", "", "-", "", ""]

r44_1 = ["", "-", "", "-", "", "", "", ""]
r44_2 = ["", "", "", "", "", "-", "-", ""]
r44_3 = ["", "-", "-", "", "", "-", "", "-"]
r44_4 = ["", "R", "", "", "R", "", "R", ""]
r44_5 = ["", "", "", "-", "", "", "", "-"]

rhythmic_patterns = {
	'2/4': [r24_1, r24_2, r24_3, r24_4, r24_5],
	'3/4': [r34_1, r34_2, r34_3, r34_4, r34_5, r34_6],
	'4/4': [r44_1, r44_2, r44_3, r44_4, r44_5],
}


# drum patterns
# kick drum: 36, snare drum: 38, closed hh: 42, open hh: 46
d24_1 = {36: ["x", "", "x", "", "", "", "x", "", "", "", "x", "", "", "x", "", ""],
		38: ["", "", "", "", "x", "", "", "x", "", "x", "", "x", "x", "", "", "x"],
		42: ["x", "x", "x", "x", "x", "x", "x", "", "x", "x", "x", "x", "x", "", "x", "x"],
		46: ["", "", "", "", "", "", "", "x", "", "", "", "", "", "x", "", ""]
		}
d24_2 = {36: ["x", "", "x", "", "", "", "x", "", "", "", "x", "", "", "x", "", ""],
		38: ["", "", "", "", "x", "", "", "x", "", "x", "", "x", "x", "", "", "x"],
		42: ["x", "", "x", "", "x", "", "x", "x", "x", "", "", "", "x", "", "x", ""],
		46: ["", "", "", "", "", "", "", "", "", "", "x", "", "", "", "", ""]
		}
d24_3 = {36: ["x", "x", "", "", "", "", "", "x", "", "", "x", "x", "", "", "", ""],
		38: ["", "", "", "", "x", "", "", "", "", "", "", "", "x", "", "", ""],
		42: ["x", "", "x", "", "x", "", "x", "", "x", "", "x", "", "x", "", "x", ""]
		}


drum_patterns = {
	"The Funcky Drummer": d24_1, 
	"Impeach the President": d24_2, 
	"When the Levee Breaks": d24_3
}
