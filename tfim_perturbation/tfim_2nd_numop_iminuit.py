import numpy as np
from iminuit import Minuit
from scipy.linalg import eigh
import tfim
import tfim_perturbation

# Initial system specification
L = [3]
Jij_seed = 19
h_x_range = np.arange(0, 0.001, 0.00002)

PBC = True
J = 1

# Build lattice and basis
###################################
lattice = tfim.Lattice(L, PBC)
N = lattice.N
basis = tfim.IsingBasis(lattice)
###################################

#construct random J matrix
Jij = tfim.Jij_instance(N,J,"bimodal",Jij_seed)

# List out all the spin_states, corresponding indices and energies
Energies = -tfim.JZZ_SK_ME(basis,Jij)
# for index in range(2**N):
#     print(index, basis.state(index), Energies[index])

#construct random J matrix
Jij = tfim.Jij_instance(N,J,"bimodal",Jij_seed)

# Build 2nd order approximated matrix

GS_energy, GS_indices = tfim_perturbation.GS(Energies)

H_app_0 = tfim_perturbation.H_app_0(GS_energy, GS_indices)

H_app_1 = tfim_perturbation.H_app_1(basis, GS_indices, N)

# Parametrize H_app_2

def H_app_2_param(basis, Jij, GS_indices, N, GS_energy, param):
    # Second-Order term in perturbation theory
    H_app_2 = np.zeros((len(GS_indices), len(GS_indices)))

    for column, GS_ket_1 in enumerate(GS_indices):
        state_1 = basis.state(GS_ket_1)
        for i in range(N):
            basis.flip(state_1, i)
            state_1_flipped_index = basis.index(state_1)
            if state_1_flipped_index not in GS_indices:
                energy_gap = tfim_perturbation.state_energy(basis, Jij, state_1_flipped_index) - GS_energy
                for j in range(N):
                    basis.flip(state_1, j)
                    ES_2_flipped_index = basis.index(state_1)
                    GS_2_index = np.argwhere(np.array(GS_indices) == ES_2_flipped_index)
                    if len(GS_2_index) > 0:
                        row = GS_2_index[0][0]
                        H_app_2[row, column] -= param / energy_gap
                    basis.flip(state_1, j)
            basis.flip(state_1, i)
    return H_app_2

# Build exact matrix
V_exc = tfim_perturbation.V_exact(basis, lattice)
H_0_exc = tfim_perturbation.H_0_exact(Energies)

# Define error function to be minimized
def err(par0, par1):
    param = par0
    alpha = par1
    # alpha is the fitting parameter for the 3rd order polynomial
    error_arr = np.zeros(np.shape(h_x_range))
    for i, h_x in enumerate(h_x_range):
        H_app = H_app_0 + h_x*H_app_1 + np.power(h_x, 2.)*H_app_2_param(basis, Jij, GS_indices, N, GS_energy, param)
        # Calculate the energy eigenvalue of the approximated 2nd order matrix
        app_eigenvalues, app_eigenstates = eigh(H_app)
        app_GS_eigenvalue = min(app_eigenvalues)
        # Calculate exact eigenvalues and eigenstates for range(h_x)
        exc_eigenvalues, exc_eigenstates = tfim_perturbation.exc_eigensystem(basis, h_x_range, lattice, Energies)
        exc_GS_eigenvalue = min(exc_eigenvalues[:,i])
        error_arr[i] = abs(abs(app_GS_eigenvalue-exc_GS_eigenvalue) - alpha*np.power(h_x, 3.))
    return np.sqrt(sum(np.power(error_arr, 2.)))

# Perform optimization
m = Minuit(err, par0=0.5, par1=0.2)
m.errors['par0'] = 0.001
m.errors['par1'] = 0.001
m.limits['par0'] = (-1., 1.)
m.limits['par1'] = (0., 1.)
m.errordef = 1
m.migrad()
print('value', m.values)
print('error', m.errors)
print('fval', m.fval)