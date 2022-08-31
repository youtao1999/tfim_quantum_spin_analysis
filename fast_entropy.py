import networkx as nx
from NN_ground import fast_GS
import tfim_perturbation as perturbation
import tfim_EE
import numpy as np
import matplotlib.pyplot as pl
import os

def order(basis, GS_indices):
    weighted_adj_matrix = perturbation.Hamming_array(GS_indices, basis)
    order = 0
    while order <= np.max(weighted_adj_matrix):
        subgraph_adj_matrix = np.zeros(np.shape(weighted_adj_matrix))
        for coord in np.argwhere(weighted_adj_matrix <= order):
            subgraph_adj_matrix[coord[0], coord[1]] = weighted_adj_matrix[coord[0], coord[1]]
        S = nx.from_numpy_matrix(subgraph_adj_matrix)
        if nx.number_connected_components(S) <= 2:
            return order
        else:
            order += 1

def degeneracy_entropy(degeneracy_hist_data, entropy_hist_data, degeneracy):
    '''
    This function puts the entropy into degeneracy bins
    '''
    pos = np.where(degeneracy_hist_data == degeneracy)
    return entropy_hist_data[pos]

def fast_entropy(h_x_range, num_iter, size_list):
    degeneracy_hist_data = np.zeros((num_iter, len(size_list)))
    order_hist_data = np.zeros((num_iter, len(size_list)))
    entropy_hist_data = np.zeros((num_iter, len(size_list)))
    entropy_par_hist_data = []

    for i, size in enumerate(size_list):
        J = 1
        # calculate ground states, store in dictionary
        seed_range = [np.random.randint(1, 1e3) for i in range(num_iter)]
        for j, seed in enumerate(seed_range):
            # make sure that GS_indices consists of only integers
            GS_indices, Jij, basis, typecounts, Jij_type = fast_GS(size[0], size[1], seed)
            N = size[0] * size[1]
            partition_set = tfim_EE.linear_bipartition(size)
            entropy_par_arr = np.zeros((len(partition_set), num_iter))
            degeneracy_hist_data[j, i] = len(GS_indices)
            order_hist_data[j, i] = order(basis, GS_indices)

            # Building blocks matrices
            GS_energy = perturbation.state_energy(basis, Jij, int(GS_indices[0]))
            # check to see if we have zero energy gap denominator problem
            # function returns false if the problem exists
            num_vanishing_energy_gaps = tfim_EE.energy_denominator_check(basis, Jij, GS_indices, GS_energy, N)
            if num_vanishing_energy_gaps > 0:
                print('seed: ', seed, 'num_GS: ', len(GS_indices), 'num_vanishing_energy_gaps: ',
                      len(num_vanishing_energy_gaps), 'GS_energy: ', GS_energy)
                pass

            else: # if none of the energy gaps suffers the zero denominator problem
                adj_matrix, H_0, H_app_1, H_app_2, H_app_3, H_app_4 = perturbation.matrix_terms(GS_indices, GS_energy, N, basis, Jij)
                matrix_terms = [H_0, H_app_1, H_app_2, H_app_3, H_app_4]

                # check if the ground manifold is splitting into more than two disconnected components
                G = nx.from_numpy_matrix(adj_matrix)
                num_components = nx.number_connected_components(G)

                if num_components <= 2:
                    app_eigenvalues, app_eigenstates, app_eigenvalues_all, app_eigenstates_all, app_eigenvalues_connected = perturbation.fourth_order_eigensystem(
                        adj_matrix, matrix_terms, h_x_range, GS_indices, J)
                    entropy_par = np.zeros(len(partition_set))
                    for k in range(len(partition_set)):
                        [A, B] = partition_set[k]
                        entropy = tfim_EE.perturb_entropy(basis, GS_indices, A, B, app_eigenstates, 0)
                        entropy_par[k] = entropy
                        entropy_par_arr[k, j] = entropy
                    entropy_par_hist_data.append(entropy_par_arr)
                    entropy_hist_data[j, i] = np.mean(entropy_par)
                else:
                    print(
                        'for seed: {seed_num} the GS manifold still has {num_disconnected_components} disconnected components at 4th order'.format(
                            seed_num=seed, num_disconnected_components=num_components))
    return degeneracy_hist_data, order_hist_data, entropy_hist_data

def data_store(output, file_name, data):
    # store data
    if os.path.isdir(output):
        os.chdir(output)
    else:
        os.mkdir(output)
        os.chdir(output)

    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name, 'wb') as f:
        for data_subset in data:
            np.save(f, data_subset)
    os.chdir('../')

def data_read(output, file_name):
    os.chdir(output)
    with open(file_name, 'rb') as f:
        degeneracy_hist_data = np.load(f)
        order_hist_data = np.load(f)
        entropy_hist_data = np.load(f)
    return degeneracy_hist_data, order_hist_data, entropy_hist_data

def degeneracy_hist(degeneracy_hist_data, size_list):
    # degeneracy histogram
    nbins = 70
    logbins = np.logspace(np.log(np.min(degeneracy_hist_data)+1), np.log(np.max(degeneracy_hist_data)), nbins)
    size_label = [str(size) for size in size_list]
    fig = pl.figure(figsize=(8, 6))
    # histogram on log scale.
    pl.rcParams['font.size'] = '18'
    n, dbins, patches = pl.hist(degeneracy_hist_data, bins = nbins, label = size_label)
    for item in patches:
        item.set_height(item.get_height()/sum(n))
    pl.ylabel('probability', fontsize=18)
    pl.xlabel('degeneracy', fontsize=18)
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(False)
    pl.legend(loc=5, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)
    pl.ylim(top = 1.)
    title = "degeneracy_vs_size_histogram"
    pl.title(title)
    pl.ioff()
    pl.savefig("degeneracy_vs_size_histogram", bbox_inches='tight')

def order_hist(order_hist_data, size_list):
    # order histogram
    nbins = int(np.max(order_hist_data))
    size_label = [str(size) for size in size_list]
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    n, bins, patches = pl.hist(order_hist_data, bins = nbins, label = size_label)
    for item in patches:
        item.set_height(item.get_height()/sum(n))
    pl.ylabel('probability', fontsize=18)
    pl.xlabel('order', fontsize=18)
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(False)
    pl.ylim(top = 1.)
    pl.legend(loc='upper right', prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)
    title = "order_vs_size_histogram"
    pl.title(title)
    pl.ioff()
    pl.savefig("order_vs_size_histogram", bbox_inches='tight')

def entropy_hist(entropy_hist_data, degeneracy_hist_data, size_list):
    # entropy histogram
    size_label = [str(size) for size in size_list]
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    size_labels = []
    EE_per_degen = []
    for i in range(len(size_list)):
        degen_arr = [4, 8, 16, 32]
        for degen in degen_arr:
            size_labels.append(str(size_label[i]) + ' degen: ' + str(degen))
            EE_per_degen.append(entropy_hist_data[np.where(degeneracy_hist_data == degen)])
    n, bins, patches = pl.hist(EE_per_degen, label=size_labels)
    for i, patch in enumerate(patches):
        for item in patch:
            item.set_height(item.get_height()/sum(n[i]))
    pl.ylabel('probability', fontsize=18)
    pl.xlabel('entropy', fontsize=18)
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(False)
    pl.ylim(top = 1.)
    pl.legend(loc='upper right', prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)
    title = "entropy_histogram"
    pl.title(title)
    pl.ioff()
    pl.savefig("entropy_histogram", bbox_inches='tight')
