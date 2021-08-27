#!/usr/bin/env python

""""tfim_rdm.py
    Chris Herdman
    10.21.2018
    --Functions related to the reduced density matrix of the TFIM
    --Requires: tfim.py, numpy, scipy.sparse, scipy.linalg, progressbar
"""

import tfim
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy import linalg
import progressbar
import argparse


###############################################################################
def svd(basis, A, B, v, Compute_UV=True):
    """Compute the singular value decomposition of vector v
        ---A, B are the lists of sites in each bipartition"""   
 
    # Setup bases for A & B
    A_lattice = tfim.Lattice([len(A)])
    B_lattice = tfim.Lattice([len(B)])

    A_basis = tfim.IsingBasis(A_lattice)
    B_basis = tfim.IsingBasis(B_lattice)

    
    # Build psi matrix
    psiMat = np.zeros([A_basis.M, B_basis.M])
    for index in range(basis.M):
        state = basis.state(index)
        a_state = state[A]
        b_state = state[B]
        a_index = A_basis.index(a_state)
        b_index = B_basis.index(b_state)
        psiMat[a_index, b_index] = v[index]

    # Perform SVD
    if Compute_UV:
        U, S, V = linalg.svd(psiMat, compute_uv=True)
        return S, U, V
    else:
        S = linalg.svd(psiMat, compute_uv=False)
        return S


###############################################################################
def entropy(S):
    
    return -np.sum( S**2*np.log(S**2) )
