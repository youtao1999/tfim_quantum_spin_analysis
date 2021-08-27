#!/usr/bin/env python

""""TFIMED.py
    Tao You
    01/26/2021
    --Build the building block matrices for each order of perturbation theory
    --Requires: numpy, scipy.sparse, scipy.linalg, progressbar
"""
import tfim
import tfim_perturbation
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy import linalg
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy import optimize
import progressbar
import argparse
import os

###############################################################################

# For generalizing matrix approach to perturbation theory

def PVP(basis, GS_indices, N):
    # PVP matrix
    PVP = np.zeros((len(GS_indices), len(GS_indices)))
    
    for column, ket in enumerate(GS_indices):
        state = basis.state(ket)
        for i in range(N):
            basis.flip(state,i)
            bra = basis.index(state)
            subspace_index = np.argwhere(GS_indices == bra)
            if len(subspace_index) > 0:
                row = subspace_index[0][0]
                PVP[row, column] += 1
            basis.flip(state,i)
    return PVP

def PVQ_1(basis, Jij, GS_indices, ES_1_indices, N, GS_energy):
    # Construct PVQ matrix
    PVQ = np.zeros((len(GS_indices), len(ES_1_indices)))
    for column, ES_ket_index in enumerate(ES_1_indices):
        state = basis.state(ES_ket_index)
        for i in range(N):
            basis.flip(state, i)
            state_flipped_index = basis.index(state)
            if state_flipped_index in GS_indices:
                GS_bra_index = np.argwhere(np.array(GS_indices) == state_flipped_index)
                if len(GS_bra_index) > 0:
                    row = GS_bra_index[0][0]
                    PVQ[row, column] += 1
            basis.flip(state, i)
    return PVQ

def Q_1VQ_1(basis, ES_1_indices, GS_indices, N):
    # QVQ matrix
    QVQ = np.zeros((len(ES_1_indices), len(ES_1_indices)))
    for column, ket in enumerate(ES_1_indices):
        state = basis.state(ket)
        for i in range(N):
            basis.flip(state, i)
            bra = basis.index(state)
            subspace_index = np.argwhere(ES_1_indices == bra)
            if len(subspace_index) > 0 and ES_1_indices[subspace_index] not in GS_indices:
                row = subspace_index[0][0]
                QVQ[row, column] += 1
            basis.flip(state,i)
    return QVQ

# a function that take a state index as input and returns all the indices of 
# excited states that are one hamming distance away from that state

def Q_1VQ_2(basis, ES_2_indices, ES_1_indices, GS_indices, N):
    #ES_2_indices denotes the indices of all the states that are one Hamming distance away from ES_1_indices
    # QVQ matrix
    QVQ = np.zeros((len(ES_1_indices), len(ES_2_indices)))
    for column, ket in enumerate(ES_2_indices):
        state = basis.state(ket)
        for i in range(N):
            basis.flip(state, i)
            bra = basis.index(state)
            subspace_index = np.argwhere(ES_1_indices == bra)
            if len(subspace_index) > 0 and ES_1_indices[subspace_index] not in GS_indices:
                row = subspace_index[0][0]
                QVQ[row, column] += 1
            basis.flip(state,i)
    return QVQ

def Hamming_set(basis, input_state_indices, N, GS_indices):
    Hamming_set = []
    for i, state_index in enumerate(input_state_indices):
        state = basis.state(state_index)
        for j in range(N):
            basis.flip(state, j)
            state_flipped_index = basis.index(state)
            if state_flipped_index not in GS_indices:
                Hamming_set.append(state_flipped_index)
            basis.flip(state, j)
    Hamming_set = np.array(Hamming_set)
    np.sort(Hamming_set)
    return np.unique(Hamming_set)

def energy_gap(basis, Jij, input_state_indices, N, GS_energy, exponent):
    # Construct energy gap as a diagonal matrix
    # Input_state_indices is a 1D NumPy array of state indices that are a certain number of Hamming distances away from GS_indices: Q_1, Q_2... etc
    energy_gap_matrix = np.zeros((len(input_state_indices), len(input_state_indices)))
    for i,  state_index in enumerate(input_state_indices):
        energy_gap = GS_energy - tfim_perturbation.state_energy(basis, Jij, state_index)
        energy_gap_matrix[i, i] += 1/(energy_gap**exponent)
    return energy_gap_matrix

def hc(matrix):
    # helper function for summation between the original matrix and its hermitian conjugate
    return matrix + np.transpose(matrix)