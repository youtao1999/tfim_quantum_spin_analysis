#!/usr/bin/env python

""""TFIMED.py
    Tao You
    07/02/2020
    --Build first and second order approximated matrices using perturbation theory
    --Requires: numpy, scipy.sparse, scipy.linalg, progressbar
"""
import tfim_perturbation.tfim_matrices as tfim_matrices
import numpy as np
from scipy.linalg import eigh
###############################################################################

# To be used in tfim_search:

def Hamming_distance(state_1, state_2):
    # Define Hamming distance
    return len(np.nonzero(state_1 - state_2)[0])

def Hamming_array(GS_indices, basis):
    # Calculate Hamming distance array
    Hamming = np.zeros((len(GS_indices), len(GS_indices)-1))
    for i, n in enumerate(GS_indices):
        basis_n = basis.state(n)
        for j, m in enumerate(np.delete(GS_indices, i)):
            basis_m = basis.state(m)
            Hamming[i, j] = Hamming_distance(basis_n, basis_m)
    return Hamming

def judge(order, array, N):
    if array[np.nonzero(array <= N/2.0)] != []:
        return np.max(array[np.nonzero(array <= N/2.0)]) <= order

def minOrder(order, array, N):
    # if this function returns true, then this instance is a threshold instance for which our perturbation expansion is non-trivial
    if array[np.nonzero(array <= N/2.0)] != []:
        return np.min(array[np.nonzero(array <= N/2.0)]) == order
    
###############################################################################

def state_energy(basis,J,state_index):
    """ Computes specific state energy"""
    
    shift_state = np.zeros(basis.N,dtype=int)
    state = basis.spin_state(state_index)
    energy = 0
    for shift in range(1,int(basis.N/2+1)):
        shift_state[shift:] = state[:-shift]
        shift_state[:shift] = state[-shift:]
        if (basis.N%2 == 0) and (shift == basis.N/2):
            energy = energy + 0.5*np.dot(J[shift-1,:]*shift_state,state)
        else:
            energy = energy + np.dot(J[shift-1,:]*shift_state,state)
    energy = energy*(-1)
    return energy

def GS(Energies):
    GS_energy = np.min(Energies)
    GS_indices = np.nonzero(Energies == GS_energy)[0]
    return GS_energy, GS_indices
    
def H_app_0(GS_energy, GS_indices):
    return GS_energy*np.identity(len(GS_indices))

def H_app_1(basis, GS_indices, N):
    
    # First-Order term in perturbation theory
    V = np.zeros((len(GS_indices), len(GS_indices)))
    
    for column, ket in enumerate(GS_indices):
        state = basis.state(ket)
        for i in range(N):
            basis.flip(state,i)
            bra = basis.index(state)
            subspace_index = np.argwhere(GS_indices == bra)
            if len(subspace_index) > 0:
                row = subspace_index[0][0]
                V[row, column] += 1
            basis.flip(state,i)
    return V

def H_app_2(basis, Jij, GS_indices, N, GS_energy):
    # Second-Order term in perturbation theory
    H_app_2 = np.zeros((len(GS_indices), len(GS_indices)))
    
    for column, GS_ket_1 in enumerate(GS_indices):
        state_1 = basis.state(GS_ket_1)
        for i in range(N):
            basis.flip(state_1, i)
            state_1_flipped_index = basis.index(state_1)
            if state_1_flipped_index not in GS_indices:
                energy_gap = state_energy(basis, Jij, state_1_flipped_index) - GS_energy
                for j in range(N):
                    basis.flip(state_1,j)
                    ES_2_flipped_index = basis.index(state_1)
                    GS_2_index = np.argwhere(np.array(GS_indices) == ES_2_flipped_index)
                    if len(GS_2_index) > 0:
                        row = GS_2_index[0][0]
                        H_app_2[row, column] -= 1./energy_gap
                    basis.flip(state_1, j)
            basis.flip(state_1, i)
    return H_app_2

def H_app_3(basis, Jij, GS_indices, N, GS_energy):
    # 3rd order approximation term
    H_app_3 = np.zeros((len(GS_indices), len(GS_indices)))
    
    for column, GS_bra_1 in enumerate(GS_indices):
        state_0 = basis.state(GS_bra_1)
        for i in range(N):
            basis.flip(state_0, i)
            state_1_index = basis.index(state_0)
            if state_1_index in GS_indices:
                for j in range(N):
                    basis.flip(state_0, j)
                    state_2_index = basis.index(state_0)
                    if state_2_index not in GS_indices:
                        energy_gap = GS_energy - state_energy(basis, Jij, state_2_index)
                        for k in range(N):
                            basis.flip(state_0, k)
                            state_3_index = basis.index(state_0)
                            GS_3_index = np.argwhere(np.array(GS_indices) == state_3_index)
                            if len(GS_3_index) > 0:
                                row = GS_3_index[0][0]
                                H_app_3[row, column] += -0.5/(energy_gap**2)
                            basis.flip(state_0, k)
                    basis.flip(state_0, j)
            basis.flip(state_0, i)
    term_2 = np.transpose(H_app_3)
    H_app_3 += term_2
    
#     for column, GS_bra_1 in enumerate(GS_indices):        
#         state_0 = basis.state(GS_bra_1)
#         for i in range(N):
#             basis.flip(state_0, i)
#             state_1_index = basis.index(state_0)
#             if state_1_index not in GS_indices:
#                 energy_gap = state_energy(basis, Jij, state_1_index) - GS_energy
#                 for j in range(N):
#                     basis.flip(state_0, j)
#                     state_2_index = basis.index(state_0)
#                     if state_2_index in GS_indices:
#                         for k in range(N):
#                             basis.flip(state_0, k)
#                             state_3_index = basis.index(state_0)
#                             GS_3_index = np.argwhere(np.array(GS_indices) == state_3_index)
#                             if len(GS_3_index) > 0:
#                                 row = GS_3_index[0][0]
#                                 H_app_3[row, column] += -0.5/(energy_gap**2)
#                             basis.flip(state_0, k)
#                     basis.flip(state_0, j)
#             basis.flip(state_0, i)

    for column, GS_bra_1 in enumerate(GS_indices):
        state_0 = basis.state(GS_bra_1)
        for i in range(N):
            basis.flip(state_0, i)
            state_1_index = basis.index(state_0)
            if state_1_index not in GS_indices:
                energy_gap_1 = GS_energy - state_energy(basis, Jij, state_1_index)
                for j in range(N):
                    basis.flip(state_0, j)
                    state_2_index = basis.index(state_0)
                    if state_2_index not in GS_indices:
                        energy_gap_2 = GS_energy - state_energy(basis, Jij, state_2_index)
                        for k in range(N):
                            basis.flip(state_0, k)
                            state_3_index = basis.index(state_0)
                            GS_3_index = np.argwhere(np.array(GS_indices) == state_3_index)
                            if len(GS_3_index) > 0:
                                row = GS_3_index[0][0]
                                H_app_3[row, column] += 1.0/(energy_gap_1*energy_gap_2)
                            basis.flip(state_0, k)
                    basis.flip(state_0, j)
            basis.flip(state_0, i)
    return H_app_3

def H_app_1st(h_x, H_0, V, J):
    # Calculate final 1st order
    return H_0 - h_x*V

def H_app_2nd(h_x, H_0, V, H_2, J):
    # Calculate final 2nd order approximated matrix
    c_2 = h_x**2
    return H_0 - h_x*V + H_2*c_2

def H_app_3rd(h_x, H_0, V, H_2, H_3, J):
    # Calculate final 3rd approximated matrix
    c_2 = h_x**2
    c_3 = h_x**3
    return H_0 - h_x*V + H_2*c_2 - H_3*c_3

def H_app_4th(h_x, H_0, V, H_2, H_3, H_4, J):
    # Calculate final 3rd approximated matrix
    c_2 = h_x**2
    c_3 = h_x**3
    c_4 = h_x**4
    return H_0 - h_x*V + H_2*c_2 - H_3*c_3 + H_4*c_4

def V_exact(basis, lattice):
    V_exact = np.zeros((basis.M, basis.M))
    for ket in range(basis.M):
        state = basis.state(ket)
        for i in range(lattice.N):
            basis.flip(state,i)
            bra = basis.index(state)
            V_exact[bra, ket] += 1
            basis.flip(state,i)
    return V_exact

def H_0_exact(Energies):
    return np.diag(Energies)

def H_exact(h_x, V_exact, H_0_exact):
    # Calculate exact matrix
    return H_0_exact - h_x*V_exact

###############################################################################

def app_1_eigensystem(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))

    H_0 = H_app_0(GS_energy, GS_indices)
    V = H_app_1(basis, GS_indices, N)

    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = np.linalg.eigh(H_app_1st(h_x, H_0, V, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, V

def app_2_eigensystem(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    
    H_0 = H_app_0(GS_energy, GS_indices)
    V = H_app_1(basis, GS_indices, N)
    H_2 = H_app_2(basis, Jij, GS_indices, N, GS_energy)

    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = np.linalg.eigh(H_app_2nd(h_x, H_0, V, H_2, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, H_2

def app_2_eigensystem_general_matrices(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    
    ES_1_indices = tfim_matrices.Hamming_set(basis, GS_indices, N, GS_indices)
    PVP = tfim_matrices.PVP(basis, GS_indices, N)
    PVQ = tfim_matrices.PVQ_1(basis, Jij, GS_indices, ES_1_indices, N, GS_energy)
    QVP = np.transpose(PVQ)
    energy_gap_matrix = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 1)
    
    # Build 0th order approximated matrix
    H_0 = H_app_0(GS_energy, GS_indices)

    # Start building the 1st order Hamiltonian
    H_app_1 = PVP

    # Start building the 2nd Hamiltonian
    H_app_2 = PVQ @ energy_gap_matrix @ QVP
    
    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = np.linalg.eigh(H_app_2nd(h_x, H_0, H_app_1, H_app_2, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, H_app_2

def app_3_eigensystem(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    
    H_0 = H_app_0(GS_energy, GS_indices)
    V = H_app_1(basis, GS_indices, N)
    H_2 = H_app_2(basis, Jij, GS_indices, N, GS_energy)
    H_3 = H_app_3(basis, Jij, GS_indices, N, GS_energy)
    
    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = eigh(H_app_3rd(h_x, H_0, V, H_2, H_3, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, H_3

def app_3_eigensystem_general_matrices(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    
        # Building blocks matrices
    ES_1_indices = tfim_matrices.Hamming_set(basis, GS_indices, N, GS_indices)
    PVP = tfim_matrices.PVP(basis, GS_indices, N)
    PVQ1 = tfim_matrices.PVQ_1(basis, Jij, GS_indices, ES_1_indices, N, GS_energy)
    Q1VP = np.transpose(PVQ1)
    Q1VQ1 = tfim_matrices.Q_1VQ_1(basis, ES_1_indices, GS_indices, N)

    # energy_gap_matrix_12 (EGM) denotes 1/(E_0-QH_0Q)^2 from Q1 to Q1
    EGM_12 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 2)
    EGM_13 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 3)
    EGM_11 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 1)

    # Start building Hamiltonians
    H_0 = H_app_0(GS_energy, GS_indices)

    H_app_1 = PVP

    H_app_2 = PVQ1 @ EGM_11 @ Q1VP

    H_app_3 = -0.5*(PVP @ PVQ1 @ EGM_12 @ Q1VP + np.transpose(PVP @ PVQ1 @ EGM_12 @ Q1VP)) + PVQ1 @ EGM_11 @ Q1VQ1 @ EGM_11 @ Q1VP
    
    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = eigh(H_app_3rd(h_x, H_0, H_app_1, H_app_2, H_app_3, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, H_app_3

def app_4_eigensystem_general_matrices(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    
    # Building blocks matrices
    ES_1_indices = tfim_matrices.Hamming_set(basis, GS_indices, N, GS_indices)
    ES_2_indices = tfim_matrices.Hamming_set(basis, ES_1_indices, N, GS_indices)
    
    # Building blocks matrices
    PVP = tfim_matrices.PVP(basis, GS_indices, N)
    PVQ1 = tfim_matrices.PVQ_1(basis, Jij, GS_indices, ES_1_indices, N, GS_energy)
    Q1VP = np.transpose(PVQ1)
    Q1VQ1 = tfim_matrices.Q_1VQ_1(basis, ES_1_indices, GS_indices, N)
    Q1VQ2 = tfim_matrices.Q_1VQ_2(basis, ES_2_indices, ES_1_indices, GS_indices, N)
    Q2VQ1 = np.transpose(Q1VQ2)

    # energy_gap_matrix_12 (EGM) denotes 1/(E_0-QH_0Q)^2 from Q1 to Q1
    EGM_12 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 2)
    EGM_13 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 3)
    EGM_11 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 1)
    EGM_21 = tfim_matrices.energy_gap(basis, Jij, ES_2_indices, N, GS_energy, 1)
    
    # Start building Hamiltonians

    H_0 = H_app_0(GS_energy, GS_indices)

    H_app_1 = PVP

    H_app_2 = PVQ1 @ EGM_11 @ Q1VP

    H_app_3 = -0.5*(PVP @ PVQ1 @ EGM_12 @ Q1VP + np.transpose(PVP @ PVQ1 @ EGM_12 @ Q1VP)) + PVQ1 @ EGM_11 @ Q1VQ1 @ EGM_11 @ Q1VP
    
    H_app_4 = 0.5*(tfim_matrices.hc(PVQ1 @ EGM_13 @ Q1VP @ PVP @ PVP)) - 0.5*(tfim_matrices.hc(PVQ1 @ EGM_12 @ Q1VP @ PVQ1 @ EGM_11 @ Q1VP)) - 1.*(tfim_matrices.hc(PVQ1 @ EGM_11 @ Q1VQ1 @ EGM_12 @ Q1VP @ PVP)) + 1.*(PVQ1 @ EGM_11 @ Q1VQ2 @ EGM_21 @ Q2VQ1 @ EGM_11 @ Q1VP)

    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = eigh(H_app_4th(h_x, H_0, H_app_1, H_app_2, H_app_3, H_app_4, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, H_app_4

def exc_eigensystem(basis, h_x_range, lattice, Energies):
    # Calculate exact eigenvalues and eigenstates for range(h_x)
    exc_eigenvalues = np.zeros((basis.M, len(h_x_range)))
    exc_eigenstates = np.zeros((len(h_x_range), basis.M, basis.M))
    V_exc = V_exact(basis, lattice)
    H_0_exc = H_0_exact(Energies)
    for j, h_x in enumerate(h_x_range):
        exc_eigenvalue, exc_eigenstate = np.linalg.eigh(H_exact(h_x, V_exc, H_0_exc));
        for i in range(basis.M):
            exc_eigenvalues[i][j] = exc_eigenvalue[i]
            for k in range(basis.M):
                exc_eigenstates[j][i][k] = exc_eigenstate[i][k]
    return exc_eigenvalues, exc_eigenstates

###############################################################################
# For error analysis and curve fit

def poly_5(x, b):
    return b*x**5;

def poly_4(x, b):
    return b*x**4.;

def poly_3(x, b):
    # third order polynomial
    return b*x**3;

def poly_2(x, a):
    # second order polynomial
    return a*x**2;

def prob(eigenstate):
    norm = np.vdot(eigenstate, eigenstate)
    normed_eigenstate = eigenstate/(norm**0.5)
    return np.conjugate(normed_eigenstate)*normed_eigenstate

def prob_app(GS_indices, h_x_range, app_eigenstates):
    # Calculate probabilities for approximated eigenstates
    prob_app = np.zeros((len(GS_indices), len(h_x_range),))
    for j, h_x in enumerate(h_x_range):
        GS_prob_vector = prob(app_eigenstates[j][:, 0])
        for i in range(len(GS_indices)):
            prob_app[i][j] = GS_prob_vector[i]
    return prob_app

def prob_exc(GS_indices, h_x_range, exc_eigenstates, index):
    # Calculate probabilities for exact eigenstates
    prob_exc = np.zeros((len(GS_indices), len(h_x_range)))
    for j, h_x in enumerate(h_x_range):
        GS_prob_vector = prob(exc_eigenstates[j][:, index])
        for i in range(len(GS_indices)):
            prob_exc[i][j] = GS_prob_vector[i]
    return prob_exc

def prob_exc_total(GS_indices, h_x_range, exc_eigenstates):
    # Probability of finding the system to be in each of the ground states
    prob_exc_total = np.zeros((len(GS_indices), len(GS_indices), len(h_x_range),))
    for n in range(len(GS_indices)):
        for j, h_x in enumerate(h_x_range):
            GS_prob_vector = prob(np.transpose(exc_eigenstates[j])[n])
            for k, i in enumerate(GS_indices):
                prob_exc_total[n][k][j] = GS_prob_vector[i]
    return prob_exc_total

def prob_excited_sum(GS_indices, h_x_range, prob_exc_total):
    # Probability of finding the system to be in excited states
    prob_excited_sum = np.zeros((len(GS_indices), len(h_x_range)))
    for n in range(len(GS_indices)):
        for i, h_x in enumerate(h_x_range):
            prob_excited_sum[n][i] = 1 - np.sum(prob_exc_total[n][:, i])
    return prob_excited_sum

def normalize(eigenstate):
    # Normalize
    norm = np.vdot(eigenstate, eigenstate)**0.5
    return eigenstate/norm

def GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates):
    GS_exc_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    for j in range(len(h_x_range)):
        for n, m in enumerate(GS_indices):
            for i, k in enumerate(GS_indices):
                GS_exc_eigenstates[j, n, i] = exc_eigenstates[j, m, i]
    return GS_exc_eigenstates

def norm_GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates):
    # Renormalize
    normalized_GS_exc_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    GS_exc_ES = GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates)
    for j in range(len(h_x_range)):
        for n in range(len(GS_indices)):
            normed_vector = normalize(GS_exc_ES[j, :, n])
            for i in range(len(GS_indices)):
                normalized_GS_exc_eigenstates[j, i, n] = normed_vector[i]
    return normalized_GS_exc_eigenstates

def fidelity(exc_eigenstate, app_eigenstate):
    # Calculate fidelity
    dot = np.vdot(exc_eigenstate, np.conjugate(app_eigenstate))
    return dot*np.conjugate(dot)

def sort(lst):
    # identify degenerate energy level and resort
    epsilon = 1*10**(-12)
    order = [];
    floor = np.array([0]);
    for i in range(1, len(lst)):
        if abs(lst[i-1] - lst[i]) <= epsilon:
            floor = np.append(floor, i)
        else:  
            order.append(floor)
            floor = np.array([i]);
    order.append(floor)
    return list(order) 

def fidelity_array(GS_indices, h_x_range, GS_exc_eigenvalues, app_eigenvalues, exc_eigenstates, app_eigenstates):
    # Produce an array of fidelities between exc and app to be plotted
    fidelity_array = np.zeros((len(GS_indices), len(h_x_range)))
    GS_exc_ES = GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates)
    for i in range(len(h_x_range)):
        sorted_exc_energy_indices = sort(GS_exc_eigenvalues[:, i])
        sorted_app_energy_indices = sort(app_eigenvalues[:, i])
        for l, app_level in enumerate(sorted_app_energy_indices):
            for j in app_level:
                fidel_sum = 0;
                for k in sorted_exc_energy_indices[l]:
                    fidel = fidelity(GS_exc_ES[i, :, k], app_eigenstates[i, :, j])
                    fidel_sum += fidel
                fidelity_array[j, i] = fidel_sum
    return fidelity_array

def infidelity_array(fidelity_array):
    return 1 - fidelity_array