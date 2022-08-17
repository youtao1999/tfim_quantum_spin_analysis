import tfim_rdm
import numpy as np
import matplotlib.pyplot as pl
import tfim_matrices
import tfim_perturbation

############################################################################################
# partition related functions

def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif

def merge(lst1, lst2):
    merged_list = []
    for lst in lst1:
        merged_list.append(lst)
    for lst in lst2:
        merged_list.append(lst)
    return merged_list


def vertical_bipartition(L):
    # L[0] is the number of rows, L[1] the number of columns
    N = L[0] * L[1]
    A = []
    mid_line = L[1] // 2
    A_init_lines = range(mid_line)
    #     A = [(mid_line + L[1] * i) for i in range(L[0])]
    for init in A_init_lines:
        A.append([(init + L[1] * i) for i in range(L[0])])
    A = [x for sublist in A for x in sublist]

    complete = [i for i in range(N)]
    B = Diff(complete, A)
    return A, B


def horizontal_bipartition(L):
    # L[0] is the number of rows, L[1] the number of columns
    N = L[0] * L[1]
    A = []
    mid_line = L[0] // 2
    A_init_lines = [i * L[1] for i in range(mid_line)]
    #     A = [(mid_line + L[1] * i) for i in range(L[0])]
    for init in A_init_lines:
        A.append([(init + i) for i in range(L[1])])
    A = [x for sublist in A for x in sublist]

    complete = [i for i in range(N)]
    B = Diff(complete, A)
    return A, B


def h_translation(A, B):
    trans_A = [i + 1 for i in A]
    complete = sorted(A + B)
    trans_B = Diff(complete, trans_A)
    return trans_A, trans_B


def v_translation(A, B, L):
    trans_A = [i + L[1] for i in A]
    complete = sorted(A + B)
    trans_B = Diff(complete, trans_A)
    return trans_A, trans_B


def h_combination(L):
    A_init, B_init = vertical_bipartition(L)
    combination = [[A_init, B_init]]
    for i in range(np.ceil(L[1] / 2.).astype(int)):
        trans_A, trans_B = h_translation(A_init, B_init)
        combination.append([trans_A, trans_B])
        A_init, B_init = trans_A, trans_B
    return combination


def v_combination(L):
    A_init, B_init = horizontal_bipartition(L)
    combination = [[A_init, B_init]]
    for i in range(np.ceil(L[0] / 2.).astype(int)):
        trans_A, trans_B = v_translation(A_init, B_init, L)
        combination.append([trans_A, trans_B])
        A_init, B_init = trans_A, trans_B
    return combination


def linear_bipartition(L):
    return merge(h_combination(L), v_combination(L))

#########################################################################################
def partition_basis(basis, GS_indices, A, B):
    # building initial basis as set since to avoid repeated elements
    A_basis = set()
    B_basis = set()

    for index, GS_index in np.ndenumerate(GS_indices):
        state = basis.state(GS_index)
        A_basis.add(tuple(state[A]))
        B_basis.add(tuple(state[B]))

    def sum_digits(digits):
        return sum(c << i for i, c in enumerate(digits))

        # now we extract the elements from this set and start building the ordered

    # basis

    # reordering basis A
    index_matching_A = {}

    for ele in A_basis:
        ele = np.array(ele)
        index_matching_A[sum_digits(ele)] = np.array(ele)

    A_reordered_basis = np.zeros((len(index_matching_A), len(list(index_matching_A.values())[0])))
    for index, key in enumerate(index_matching_A.keys()):
        A_reordered_basis[index] = index_matching_A[key]

    # reordering basis B
    index_matching_B = {}

    for ele in B_basis:
        ele = np.array(ele)
        index_matching_B[sum_digits(ele)] = np.array(ele)

    B_reordered_basis = np.zeros((len(index_matching_B), len(list(index_matching_B.values())[0])))
    for index, key in enumerate(index_matching_B.keys()):
        B_reordered_basis[index] = index_matching_B[key]

    return A_reordered_basis, B_reordered_basis


def beta_ij(basis, GS_indices, A, B, overall_GS, perturbation_param_index):
    # this function builds beta_ij matrix as a function of the index of the perturbation parameter

    # build A, B basis first
    A_basis, B_basis = partition_basis(basis, GS_indices, A, B)
    s = overall_GS[perturbation_param_index]

    def find(basis, target):
        # this function finds the index of the target state in the basis state, gives the index of 1 in BETA
        for i, state in enumerate(basis):
            if np.array_equiv(state, target):
                return i

    BETA = np.zeros((len(A_basis), len(B_basis)))

    for (probability, GS_index) in zip(s, GS_indices):
        GS_state = basis.state(GS_index)
        i = find(A_basis, GS_state[A])
        j = find(B_basis, GS_state[B])
        BETA[i, j] += probability
    return BETA


def perturb_entropy(basis, GS_indices, A, B, overall_GS, perturbation_param_index):
    BETA = beta_ij(basis, GS_indices, A, B, overall_GS, perturbation_param_index)

    # perform uv decomposition

    u, s, vh = np.linalg.svd(BETA, full_matrices=True)

    s = s[np.where(s != 0.)]
    # add conditional statement to remove all the zero singular values

    entropy = -np.dot(s ** 2, np.log(s ** 2))

    return entropy


def exc_entropy(basis, exc_eigenstates, exc_eigenvalues, A, B, perturbation_param_index):
    if len(np.shape(exc_eigenstates)) > 2:
        psi0 = exc_eigenstates[:, :, np.argmin(exc_eigenvalues[5])]
    else:
        psi0 = exc_eigenstates
    S, U, V = tfim_rdm.svd(basis, A, B, psi0[perturbation_param_index])
    entropy = tfim_rdm.entropy(S)
    return entropy


def entropy_analysis(basis, exc_eigenstates, exc_eigenvalues, GS_indices, A, B, overall_GS, h_x_range, Jij_seed):
    # This function prints out the entropy analysis plots comparing perturbation
    # theory prediction and exact diagonalization methods

    perturb_entropy_arr = np.zeros(len(h_x_range))
    exc_entropy_arr = np.zeros(len(h_x_range))

    for i in range(len(h_x_range)):
        perturb_entropy_arr[i] = perturb_entropy(basis, GS_indices, A, B, overall_GS, i)
        exc_entropy_arr[i] = exc_entropy(basis, exc_eigenstates, exc_eigenvalues, A, B, i)

    # convergence plot
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    pl.plot(h_x_range, exc_entropy_arr, lw=1.3, ls='-', color="blue", label="exact diagonalization")
    pl.plot(h_x_range, perturb_entropy_arr, lw=1.3, ls='-', color="green", label="perturbation theory prediction")
    pl.ylabel('entropy', fontsize=18)
    pl.xlabel('perturbation parameter', fontsize=18)
    # pl.axis([np.power(10, sigma_v)[lowerbound], np.power(10, sigma_v)[upperbound], chisq[lowerbound], chisq[upperbound]])
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(False)
    # pl.xscale('log')
    pl.legend(loc=5, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)
    title = "seed: {seed_num}, \n degeneracy: {degeneracy_num}".format(seed_num=Jij_seed, degeneracy_num=len(GS_indices))
    pl.title(title)
    pl.ioff()
    pl.savefig("entropy_seed_" + str(Jij_seed) + " degeneracy_" + str(len(GS_indices)), bbox_inches='tight')

    # error plot

    # Boundary index for the plot
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    pl.plot(h_x_range, abs(exc_entropy_arr - perturb_entropy_arr), lw=1.3, ls='-', color="blue", label="entropy error")
    pl.ylabel('entropy error', fontsize=18)
    pl.xlabel('perturbation parameter', fontsize=18)
    # pl.axis([np.power(10, sigma_v)[lowerbound], np.power(10, sigma_v)[upperbound], chisq[lowerbound], chisq[upperbound]])
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(False)
    # pl.xscale('log')
    pl.yscale('log')
    pl.legend(loc=4, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)
    title = "seed: {seed_num}, \n degeneracy: {degeneracy_num}".format(seed_num=Jij_seed, degeneracy_num=len(GS_indices))
    pl.title(title)
    pl.ioff()
    pl.savefig("error_seed_" + str(Jij_seed) + " degeneracy_" + str(len(GS_indices)), bbox_inches='tight')

    return perturb_entropy_arr, exc_entropy_arr

def H_latex_pdf_generator(H_0, H_app_1, H_app_2, H_app_3, H_app_4, seed):
    from pylatex import Document, Matrix, Alignat, FlushLeft

    H_app_0 = np.matrix(H_0)
    H_app_1 = np.matrix(H_app_1)
    H_app_2 = np.matrix(H_app_2)
    H_app_3 = np.matrix(H_app_3)
    H_app_4 = np.matrix(H_app_4)

    with open('Hamiltonian_{seed_num}.txt'.format(seed_num = seed), 'w') as f:
        for line in H_app_0:
            np.savetxt(f, line, fmt = '%.2f')
        f.write("\n \n")
        for line in H_app_1:
            np.savetxt(f, line, fmt='%.2f')
        f.write("\n \n")
        for line in H_app_2:
            np.savetxt(f, line, fmt='%.2f')
        f.write("\n \n")
        for line in H_app_3:
            np.savetxt(f, line, fmt='%.2f')
        f.write("\n \n")
        for line in H_app_4:
            np.savetxt(f, line, fmt='%.2f')

    geometry_options = {"tmargin": "0.5cm", "lmargin": "0.5cm"}
    doc = Document(geometry_options=geometry_options)
    with doc.create(FlushLeft()):
        with doc.create(Alignat(numbering=False, escape=False)) as agn:
            agn.append('H_{eff}=&')
            agn.append(Matrix(H_app_0))
            agn.append('\\\\&+')
            agn.append(Matrix(H_app_1))
            agn.append('h_x')
            agn.append('\\\\&+')
            agn.append(Matrix(H_app_2))
            agn.append('h_x^2')
            agn.append('\\\\&+')
            agn.append(Matrix(H_app_3))
            agn.append('h_x^3')
            agn.append('\\\\&+')
            agn.append(Matrix(H_app_4))
            agn.append('h_x^4')
    doc.create(FlushLeft())
    if np.shape(H_0)[0] <= 10:
        doc.generate_pdf(str(seed), clean_tex=True, compiler='pdflatex')
    # else:
        # doc.generate_tex(str(seed))

def energy_plot(GS_indices, h_x_range, app_eigenvalues_all, seed, num_components, exponent):
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    GS_index_upperlim = 4
    low_field_lim = 5
    if np.shape(app_eigenvalues_all)[1] <= GS_index_upperlim:
        GS_index_upperlim = np.shape(app_eigenvalues_all)[1]
    for i in range(GS_index_upperlim):
        pl.plot(h_x_range[:low_field_lim], app_eigenvalues_all[:low_field_lim, i], lw=1.3,
                label=str(GS_indices[i]), alpha=0.7)
    pl.ylabel(r'energy $(E-E_0)/h_x^{expo})'.format(expo = exponent), fontsize=18)
    pl.xlabel('perturbation parameter', fontsize=18)
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(False)
    pl.legend(loc=5, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)
    title = "energy_seed: {seed_num}, \n degeneracy: {degeneracy_num} num_components: {num_comp} first_nontrivial order: {first_nontrivial_order}".format(seed_num=seed,
                                                                                                         degeneracy_num=len(
                                                                                                             GS_indices),
                                                                                                         num_comp=num_components, first_nontrivial_order = exponent)
    pl.title(title)
    pl.ioff()
    pl.savefig("energy_seed_" + str(seed) + " degeneracy_" + str(len(GS_indices)), bbox_inches='tight')

def ground_energy_comparison_plot(GS_indices, h_x_range, app_eigenvalues, exc_eigenvalues, seed, num_components):
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    pl.plot(h_x_range, app_eigenvalues, lw=1.3, label="perturbation", alpha=0.7)
    pl.plot(h_x_range, exc_eigenvalues, lw=1.3, label="lanczos", alpha=0.7)
    pl.ylabel('energy', fontsize=18)
    pl.xlabel('perturbation parameter', fontsize=18)
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(False)
    pl.legend(loc=5, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)
    title = "GS_energy_comparison_seed: {seed_num}, \n degeneracy: {degeneracy_num} num_components: {num_comp}".format(seed_num=seed,
                                                                                                         degeneracy_num=len(
                                                                                                             GS_indices),
                                                                                                         num_comp=num_components)
    pl.title(title)
    pl.ioff()
    pl.savefig("GS_energy_comparison_seed_" + str(seed) + " degeneracy_" + str(len(GS_indices)), bbox_inches='tight')

def entropy_plot(GS_indices, h_x_range, perturb_entropy_arr, seed, num_components, exponent):
    # entropy plot
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    pl.plot(h_x_range, perturb_entropy_arr, lw=1.3, ls='-', color="green",
            label="perturbation theory prediction")
    pl.ylabel('entropy', fontsize=18)
    pl.xlabel('perturbation parameter', fontsize=18)
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(False)
    # pl.xscale('log')
    pl.legend(loc=5, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)
    title = "entropy_seed: {seed_num}, \n degeneracy: {degeneracy_num} num_components: {num_comp}, first_nontrivial_order: {first_nontrivial_order}".format(seed_num=seed,
                                                                               degeneracy_num=len(GS_indices), num_comp = num_components, first_nontrivial_order = exponent)
    pl.title(title)
    print('seed: ', seed, 'completed')
    pl.ioff()
    pl.savefig("entropy_seed_" + str(seed) + " degeneracy_" + str(len(GS_indices)), bbox_inches='tight')

def energy_denominator_check(basis, Jij, GS_indices, GS_energy, N):

    # energy_gap_matrix_12 (EGM) denotes 1/(E_0-QH_0Q)^2 from Q1 to Q1
    ES_1_indices = tfim_matrices.Hamming_set(basis, GS_indices, N, GS_indices)
    EGM_11 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 1)

    test = np.argwhere(EGM_11 == np.inf).T[0]

    return len(test)

def mean_diff(A, axis = 0):
    # assume A is an M by N matrix
    M = np.shape(A)[axis]
    if axis == 1:
        A = A.T
    # axis = 0 means subtracting along the rows
    diff_arr = np.zeros(np.shape(A))
    for i in range(M):
        diff_arr[i] = np.abs(A[i, :] - A[(i+1) % M, :])
    if axis == 1:
        return np.mean(diff_arr.T, axis)
    else:
        return np.mean(diff_arr, axis)


def near_degeneracy_check(app_eigenvalue_connected, k=1, precision = 1e-6):
    app_eigenvalue_connected = app_eigenvalue_connected[:5, :k]
    mean_energy_diff_arr = mean_diff(app_eigenvalue_connected, axis = 1)
    return np.all(mean_energy_diff_arr <= np.ones(np.shape(app_eigenvalue_connected)[0]) * precision)

def near_degeneracy(app_eigenvalue_connected):
    k = 0
    while near_degeneracy_check(app_eigenvalue_connected, k = k):
        k += 1
    return k

def find_plateaus(F, spacing, min_length=200, tolerance=0.75, smoothing=25):
    '''
    Finds plateaus of signal using second derivative of F.
    Parameters
    ----------
    F : Signal.
    min_length: Minimum length of plateau.
    tolerance: Number between 0 and 1 indicating how tolerant
        the requirement of constant slope of the plateau is.
    smoothing: Size of uniform filter 1D applied to F and its derivatives.

    Returns
    -------
    plateaus: array of plateau left and right edges pairs
    dF: (smoothed) derivative of F
    d2F: (smoothed) Second Derivative of F
    '''
    import numpy as np
    from scipy.ndimage.filters import uniform_filter1d

    # calculate smooth gradients
    smoothF = uniform_filter1d(F, size=smoothing)
    dF = uniform_filter1d(np.gradient(smoothF, spacing), size=smoothing)
    d2F = uniform_filter1d(np.gradient(dF, spacing), size=smoothing)

    def zero_runs(x):
        '''
        Helper function for finding sequences of 0s in a signal
        https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array/24892274#24892274
        '''
        iszero = np.concatenate(([0], np.equal(x, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    # Find ranges where second derivative is zero
    # Values under eps are assumed to be zero.
    eps = np.quantile(abs(d2F), tolerance)
    smalld2F = (abs(d2F) <= eps)

    # Find repititions in the mask "smalld2F" (i.e. ranges where d2F is constantly zero)
    p = zero_runs(np.diff(smalld2F))

    # np.diff(p) gives the length of each range found.
    # only accept plateaus of min_length
    plateaus = p[(np.diff(p) > min_length).flatten()]

    return (plateaus, dF, d2F)