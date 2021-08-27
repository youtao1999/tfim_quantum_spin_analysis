import tfim
import tfim_perturbation
import numpy as np
from iminuit import Minuit
from scipy.linalg import eigh

# Initial system specification
L = [3]
Jij_seed = 19
h_x_range = np.arange(0, 0.01, 0.0002)

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

# Build 3rd order approximated matrix

GS_energy, GS_indices = tfim_perturbation.GS(Energies)
H_app_0 = tfim_perturbation.H_app_0(GS_energy, GS_indices)
H_app_1 = tfim_perturbation.H_app_1(basis, GS_indices, N)
H_app_2 = tfim_perturbation.H_app_2(basis, Jij, GS_indices, N, GS_energy)

# Parametrize H_app_3

def H_app_3_param(basis, Jij, GS_indices, N, GS_energy, param):
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
                        energy_gap = tfim_perturbation.state_energy(basis, Jij, state_2_index) - GS_energy
                        for k in range(N):
                            basis.flip(state_0, k)
                            state_3_index = basis.index(state_0)
                            GS_3_index = np.argwhere(np.array(GS_indices) == state_3_index)
                            if len(GS_3_index) > 0:
                                row = GS_3_index[0][0]
                                H_app_3[row, column] -= param[0] / (energy_gap ** 2)
                                # H_app_3[row, column] -= 2.5
                            basis.flip(state_0, k)
                    basis.flip(state_0, j)
            basis.flip(state_0, i)

        for i in range(N):
            basis.flip(state_0, i)
            state_1_index = basis.index(state_0)
            if state_1_index not in GS_indices:
                energy_gap = tfim_perturbation.state_energy(basis, Jij, state_1_index) - GS_energy
                for j in range(N):
                    basis.flip(state_0, j)
                    state_2_index = basis.index(state_0)
                    if state_2_index in GS_indices:
                        for k in range(N):
                            basis.flip(state_0, k)
                            state_3_index = basis.index(state_0)
                            GS_3_index = np.argwhere(np.array(GS_indices) == state_3_index)
                            if len(GS_3_index) > 0:
                                row = GS_3_index[0][0]
                                H_app_3[row, column] -= param[1] / (energy_gap ** 2)
                                # H_app_3[row, column] -= 2.5
                            basis.flip(state_0, k)
                    basis.flip(state_0, j)
            basis.flip(state_0, i)

        for i in range(N):
            basis.flip(state_0, i)
            state_1_index = basis.index(state_0)
            if state_1_index not in GS_indices:
                energy_gap_1 = tfim_perturbation.state_energy(basis, Jij, state_1_index) - GS_energy
                for j in range(N):
                    basis.flip(state_0, j)
                    state_2_index = basis.index(state_0)
                    if state_2_index not in GS_indices:
                        energy_gap_2 = tfim_perturbation.state_energy(basis, Jij, state_2_index) - GS_energy
                        for k in range(N):
                            basis.flip(state_0, k)
                            state_3_index = basis.index(state_0)
                            GS_3_index = np.argwhere(np.array(GS_indices) == state_3_index)
                            if len(GS_3_index) > 0:
                                row = GS_3_index[0][0]
                                H_app_3[row, column] -= param[2] / (energy_gap_1 * energy_gap_2)
                                # H_app_3[row, column] -= 1
                            basis.flip(state_0, k)
                    basis.flip(state_0, j)
            basis.flip(state_0, i)
    return H_app_3

# Build exact matrix
V_exc = tfim_perturbation.V_exact(basis, lattice)
H_0_exc = tfim_perturbation.H_0_exact(Energies)


# Define error function to be minimized
def err(par0, par1, par2, par3):
    param = [par0, par1, par2]
    alpha = par3
    # alpha is the fitting parameter for the 3rd order polynomial
    error_arr = np.zeros(np.shape(h_x_range))
    for i, h_x in enumerate(h_x_range):
        H_app = H_app_0 + h_x*H_app_1 + np.power(h_x, 2.)*H_app_2 + np.power(h_x, 3.)*H_app_3_param(basis, Jij, GS_indices, N, GS_energy, param)
        # Calculate the energy eigenvalue of the approximated 2nd order matrix
        app_eigenvalues, app_eigenstates = eigh(H_app)
        app_GS_eigenvalue = min(app_eigenvalues)
        # Calculate exact eigenvalues and eigenstates for range(h_x)
        exc_eigenvalues, exc_eigenstates = tfim_perturbation.exc_eigensystem(basis, h_x_range, lattice, Energies)
        exc_GS_eigenvalue = min(exc_eigenvalues[:,i])
        error_arr[i] = abs(app_GS_eigenvalue-exc_GS_eigenvalue) - alpha*np.power(h_x, 4.)
    return np.sqrt(sum(np.power(error_arr, 2.)))

# Perform optimization

# Perform optimization
m = Minuit(err, par0 = -0.5, par1 = -0.5, par2 = 1., par3 = 14.)
m.errors['par0'] = 0.001
m.errors['par1'] = 0.001
m.errors['par2'] = 0.001
m.errors['par3'] = 0.001
m.limits['par0'] = (-1., 1.)
m.limits['par1'] = (-1., 1.)
m.limits['par2'] = (-1., 1.)
m.limits['par3'] = (0., 100.)
m.errordef = 1
m.migrad()
print('value', m.values)
print('error', m.errors)
print('fval', m.fval)

# # Fixed alpha and plot error function
# x_arr = np.zeros((10,2))
# x_arr[:,0] = np.linspace(0, 1., 10)
# x_arr[:,1] = 0.14*np.ones(10)
# err_arr = np.zeros(np.shape(x_arr[:,0]))
# for i in range(len(x_arr[:,0])):
#     err_arr[i] = err(x_arr[i])
# print(err_arr)
# fig = pl.figure(figsize=(8, 6))
# pl.rcParams['font.size'] = '18'
# pl.plot(x_arr[:,0],err_arr, lw=1.3, ls='-', color="blue")
# pl.ylabel(r'$Error$', fontsize=18)
# pl.xlabel(r'$h_x$', fontsize=18)
# pl.xticks(fontsize=18)
# pl.yticks(fontsize=18)
# pl.tick_params('both', length=7, width=2, which='major')
# pl.tick_params('both', length=5, width=2, which='minor')
# pl.grid(True)
# pl.legend(loc=2, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=2)
# fig.tight_layout(pad=0.5)
# pl.savefig("Error_plot.png")
