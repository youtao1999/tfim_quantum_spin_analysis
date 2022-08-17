'''
This script combines the ground states code written by Ellie Copps, number_components written by Jack Landrigan and
the perturbation code written by Tao You to perform comprehensive analysis of quantum spin glasses

Work flow:

import main() NN_ground.py
    --input denoted parameters
    --save the data to a txt file
    --read the txt file, it is going to be a python dictionary
    --then attach the analysis code, for any seed instance just extract from the python dictionary

import number_components(N, ground_state) where ground_states is a list of indices, exactly the output of Ellie's code
    --input denoted parameters
    --returns dist_comp, which is a dictionary {hamming_distance: number of connected components}, the correct perturbative order
'''

import networkx as nx
from NN_ground import main
import tfim_perturbation as perturbation
import tfim_EE
import numpy as np
import tfim
import os
import shutil

# First obtain all the ground states

# specifying parameters
xwidth = str(4)
yheight = str(4)
num_iter = str(50)
h_x_range = np.linspace(0.01, 2, 100)
J = 1
check_tfim_EE = False
args = [xwidth, yheight, num_iter]

# calculate ground states, store in dictionary
ground_states_dict, N, L, Jij_list, basis = main(args)

# Now that we have obtained the ground states, we calculate the minimum order needed in perturbation theory
# dist_comp, order = number_components(N, ground_states_dict[Jij_seed])
# Search for missing ground states

# Here we specify the partition
A, B = tfim_EE.vertical_bipartition(L)

perturb_entropy_all = np.zeros((int(num_iter), len(h_x_range)))

# open new output directory
output = "fast_entropy_output"

if os.path.isdir(output):
    shutil.rmtree(output)
os.mkdir(output)
os.chdir(output)

for key in ground_states_dict.keys():
    # make sure that GS_indices consists of only integers
    GS_indices = [int(index) for index in ground_states_dict[key]]
    Jij = Jij_list[key]
    GS_energy = perturbation.state_energy(basis, Jij, int(GS_indices[0]))
    # check to see if we have zero energy gap denominator problem
    # function returns false if the problem exists
    num_vanishing_energy_gaps = tfim_EE.energy_denominator_check(basis, Jij, GS_indices, GS_energy, N)
    if num_vanishing_energy_gaps > 0:
        print('seed: ', key, 'num_GS: ', len(GS_indices), 'num_vanishing_energy_gaps: ', len(num_vanishing_energy_gaps), 'GS_energy: ', GS_energy)
        pass
    else: # if none of the energy gaps suffers the zero denominator problem
        adj_matrix, H_0, H_app_1, H_app_2, H_app_3, H_app_4 = perturbation.matrix_terms(GS_indices, GS_energy, N, basis, Jij)
        matrix_terms = [H_0, H_app_1, H_app_2, H_app_3, H_app_4]
        # store the Hamiltonian in txt file
        tfim_EE.H_latex_pdf_generator(H_0, H_app_1, H_app_2, H_app_3, H_app_4, key)
        # check if the ground manifold is splitting into more than two disconnected components
        G = nx.from_numpy_matrix(adj_matrix)
        num_components = nx.number_connected_components(G)
        if num_components <= 2:
            app_eigenvalues, app_eigenstates, app_eigenvalues_all, app_eigenstates_all, app_eigenvalues_connected = perturbation.fourth_order_eigensystem(adj_matrix, matrix_terms, h_x_range, GS_indices, J)
            if check_tfim_EE:
                ###########################################################
                # check perturbation approximation of entanglement entropy against exact diagonalization
                # Build lattice and basis
                Energies = -tfim.JZZ_SK_ME(basis, Jij)
                # Calculate exact eigenvalues and eigenstates for range(h_x)
                exc_eigenvalues, exc_eigenstates = perturbation.exc_eigensystem(basis, h_x_range, N, Energies, lanczos = True, ground_projection = False, reordering = False)
                # compare lanczos and perturbation theory in energy
                tfim_EE.ground_energy_comparison_plot(GS_indices, h_x_range, app_eigenvalues, exc_eigenvalues, key,
                                              num_components)
                outF = open("seed_{seed_num},degeneracy_{degeneracy_num}.txt".format(seed_num=key,degeneracy_num=len(GS_indices)),'w')
                perturb_entropy_arr, exc_entropy_arr = tfim_EE.entropy_analysis(basis, exc_eigenstates, exc_eigenvalues, GS_indices, A, B, app_eigenstates,
                                         h_x_range, key)
                for i in range(len(h_x_range)):
                    outF.write("{index} {h_x_val} {entropy_val_exc} {entropy_val_app} \n".format(index=i, h_x_val=h_x_range[i], entropy_val_exc = exc_entropy_arr[i],
                                                                           entropy_val_app=perturb_entropy_arr[i]))
                print('seed: ', key, 'completed')
                outF.close()
            else:
                # get rid of the constant in energy and extra dependencies on hx
                exponent = 2
                if H_app_1.any():
                    exponent = 1
                E_0_arr = np.ones(len(h_x_range)) * GS_energy
                for i in range(len(app_eigenvalues_connected[0])):
                    app_eigenvalues_connected[:, i] = (app_eigenvalues_connected[:, i] - E_0_arr)/h_x_range**exponent

                outF = open("seed_{seed_num},degeneracy_{degeneracy_num}.txt".format(seed_num=key, degeneracy_num=len(GS_indices)), 'w')
                perturb_entropy_arr = np.zeros(len(h_x_range))
                for i in range(len(h_x_range)):
                    perturb_entropy_arr[i] = tfim_EE.perturb_entropy(basis, GS_indices, A, B, app_eigenstates, i)
                    outF.write("{index} {h_x_val} {entropy_val} \n".format(index=i, h_x_val=h_x_range[i], entropy_val=perturb_entropy_arr[i]))

                    # change the Hamilonian that we are diagonalizing
                    # store all the Hamiltonian, in particular for seed 883

                (plateaus, dF, d2F) = tfim_EE.find_plateaus(perturb_entropy_arr, h_x_range[1]-h_x_range[0], min_length=10, tolerance=0.005, smoothing=10)
                print(plateaus)
                if len(plateaus) != 0.:
                    print(perturb_entropy_arr[plateaus.flatten()[0]: plateaus.flatten()[1]])
                outF.close()
                # energy & entropy plot
                tfim_EE.energy_plot(GS_indices, h_x_range, app_eigenvalues_connected, key, num_components, exponent)
                tfim_EE.entropy_plot(GS_indices, h_x_range, perturb_entropy_arr, key, num_components, exponent)
        else:
            print('for seed: {seed_num} the GS manifold still has {num_disconnected_components} disconnected components at 4th order'.format(seed_num = key, num_disconnected_components = num_components))
os.chdir('../')

# check the spectrum against exact diagonalization 3*3, for 4*4 we have to use lanczos, for the randomly fluctuating
# case we want to check the spectrum to see if the states are nearly degenerate