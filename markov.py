import os, time
import pandas as pd
import matplotlib.pyplot as plt 
import io 
import requests 
import openpyxl
import streamlit as st 
import random
import numpy as np
import music21 as m2
from params import *
import crash


class MarkovChain(object):
    """
    matrix: numpy 12 x 12 matrix (first order)
    """
    def __init__(self, matrix):
        # print(matrix)
        self.matrix = matrix
        self.N = len(matrix)
        self.cur = None

    # test if the markov chain matrix is valid
    def is_valid(self):
        return np.isclose(np.sum(self.matrix, axis=1), np.ones((self.N))).all()

    # current: int, index of the last-generated element
    def get_next_note(self, start=None):
        current = start or self.cur
        next_idx = np.random.choice(np.arange(self.matrix.shape[1]), p=self.matrix[current])
        self.cur = next_idx        
        return PITCHES[next_idx]

    '''
    start: int, index of the start element
    n: int, number of elements to generate
    return: 1 x n list
    '''
    def generate(self, start, n):
        # if self.is_valid():
            self.cur = start
            gen_list = [PITCHES[self.cur]]
            for _ in range(1, n):
                gen = self.get_next_note()
                gen_list.append(gen)
                self.cur = PITCHES.index(gen)
            return gen_list
        # else:
        #     raise RuntimeError("matrix not valid")
        #     return None




