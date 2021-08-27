#!/usr/bin/env python

""""tfim_perturbation.py
    Tao You
    06.29.2020
    --BUild frst and second order perturbation theory matrices
    --Requires: tfim.py, numpy, scipy.linalg, progressbar, matplotlib
"""

import tfim
import tfim_perturbation
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy import linalg
import matplotlib.pyplot as plt
from scipy import optimize
import progressbar
import argparse
import os

###############################################################################

def main():
    # Parse command line arguments
    ###################################
    
    parser = argparse.ArgumentParser(description = ("Build approximate matrices using first and second order perturbation theory and return its eigenvalues and eigenstates"))
    
    parser.add_argument('lattice_specifier', help = ("Linear dimensions of the system"))
    
    parser.add_argument('-PBC', type = bool, default = True, help = "Specifying PBC")
    
    parser.add_argument('-J', type=float, default = 1.0, help = 'Nearest neighbor Ising coupling')
    
    parser.add_argument('-seed', type = int, default = 5, help = "Specifying the seed for generating random Jij matrices")
    
    parser.add_argument('--h_min', type=float, default=0.0,
                            help='Minimum value of the transverse field')
    parser.add_argument('--h_max', type=float, default=0.1,
                            help='Maximum value of the transverse field')
    parser.add_argument('--dh', type=float, default=0.01,
                            help='Tranverse fied step size')
    
    parser.add_argument('-o', default='output', help='output filename base')
    
    parser.add_argument('-d', default='Output_file', help='output directory base')
    
    args = parser.parse_args()
    
    ###################################
    # Parameter specification 
    out_filename_base = args.o
    
    # Transverse field
    h_x_range = np.arange(args.h_min,args.h_max+args.dh/2,args.dh)
    
    L = [ int(args.lattice_specifier) ]
    PBC = args.PBC
    J = args.J
    seed = args.seed
    
    # Build lattice and basis
    lattice = tfim.Lattice(L, PBC)
    N = lattice.N
    basis = tfim.IsingBasis(lattice)
        
    # Construct random J matrix
    Jij = tfim.Jij_instance(N,J,"bimodal",seed)
    ###################################
    
    Energies = -tfim.JZZ_SK_ME(basis,Jij)
    GS_energy, GS_indices = tfim_perturbation.GS(Energies)
    
    ###################################
    
    H_0 = tfim_perturbation.H_0(GS_energy, GS_indices)
    H_app_1 = tfim_perturbation.H_app_1(basis, GS_indices, N)
    H_app_2 = tfim_perturbation.H_app_2(basis, Jij, GS_indices, N, GS_energy)
    
    ###################################
    # Diagonalization loop over h_x_range on H_app
    
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(GS_indices), len(GS_indices), len(h_x_range)))
    
    for j, h_x in enumerate(h_x_range):
        H_app = tfim_perturbation.H_app(h_x, H_0, H_app_1, H_app_2, J)
        app_eigenvalue, app_eigenstate = np.linalg.eigh(H_app);
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[i][k][j] = app_eigenstate[i][k]
                
    ###################################
    # Make output directory
    
    Output = args.d
    
    os.mkdir(Output)
    os.chdir(Output)
    
    ###################################
    # Output Eigenvalue file
    
    out_filename_E = out_filename_base + '.dat'
    
    # Quantities to write Eigenvalue ouput file
    
    phys_keys_E = []
    
    for i in range(len(app_eigenvalues)):
        eigenvalue_num = 'Eigenvalue ' + str(i+1)
        phys_keys_E.append(eigenvalue_num)
    
    phys_keys_E.insert(0, 'h_x')
    phys_E = {}  # Dictionary for values
    
    # Setup output Eigenvalue data files
    
    parameter_string = ("L = {}, PBC = {}, J = {}".format(L, PBC, J) )
    
    width = 25
    precision = 16
    header_list = phys_keys_E
    header = ''.join(['{:>{width}}'.format(head,width=width) 
                                            for head in header_list])
    out_eigenvalue_file = open(out_filename_E, 'w')
    print( "\tData will write to {}".format(out_filename_E) )
    out_eigenvalue_file.write( '#\ttfim_diag parameters:\t' + parameter_string + '\n' 
                    + '#' + header[1:] + '\n' )
    
    # Put eigenvalues in phys_E dictionary
    
    for i, h_x in enumerate(h_x_range):
        phys_E['h_x'] = h_x
        for j, key in enumerate(phys_keys_E[1:]):
            phys_E[key] = app_eigenvalues[j, i]
         
    # Write eigenvalues to output files
    
        data_list = [phys_E[key] for key in phys_keys_E]
        data_line = ''.join(['{:{width}.{prec}e}'.format(data,width=width,prec=precision) for data in data_list])
        out_eigenvalue_file.write(data_line+ '\n')
    
    # Close files
    
    out_eigenvalue_file.close()
    
    ###################################
    
    # Output Eigenstate files
    
    for file_num in range(len(app_eigenstates)):
        
        out_filename_V = out_filename_base + '_' + str(file_num) + '.dat'
        
        # Quantities to write Eigenstate output file
        
        phys_keys_V = []
        
        for i in range(len(app_eigenstates)):
            basis = 'Basis ' + str(i+1)
            phys_keys_V.append(basis)
        
        phys_keys_V.insert(0, 'h_x')
        phys_V = {}  # Dictionary for values
        
        # Setup output Eigenstate data files
        
        parameter_string = ("L = {}, PBC = {}, J = {}".format(L, PBC, J) )
        
        width = 25
        precision = 16
        header_list = phys_keys_V
        header = ''.join(['{:>{width}}'.format(head,width=width) 
                                                for head in header_list])
        out_eigenstate_file = open(out_filename_V, 'w')
        print( "\tData will write to {}".format(out_filename_V) )
        out_eigenstate_file.write( '#\ttfim_diag parameters:\t' + parameter_string + '\n' 
                    + '#' + header[1:] + '\n' )
        
        # Put eigenstates in phys_V dictionary
        
        for i, h_x in enumerate(h_x_range):
            phys_V['h_x'] = h_x
            for j, key in enumerate(phys_keys_V[1:]):
                phys_V[key] = app_eigenstates[file_num][j, i]
        
        # Write eigenvalues to output files
        
            data_list = [phys_V[key] for key in phys_keys_V]
            data_line = ''.join(['{:{width}.{prec}e}'.format(data,width=width,prec=precision) for data in data_list])
            out_eigenstate_file.write(data_line+ '\n')
        
        # Close files
        
        out_eigenstate_file.close()
        
    #######################################################
    
    # Exit "Output" directory
    
    os.chdir("../")
    
if __name__ == "__main__":
    main()
