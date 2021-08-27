#!/usr/bin/env python

""""tfim_build.py
    Chris Herdman
    06.07.2017
        --Builds Hamiltonian and observable matrices for 
            transverse field Ising models
        --Requires: tfim.py, numpy, scipy.sparse, scipy.linalg, progressbar
"""

import tfim
import numpy as np
import scipy.sparse
import argparse

###############################################################################
def main():
    
    # Parse command line arguements
    ###################################
    parser = argparse.ArgumentParser(description=(
                            "Builds matrices for "
                            "transverse field Ising Models of the form:\n"
                            "H = -\sum_{ij} J_{ij}\sigma^z_i \sigma^z_j" 
                                                    "- h \sum_i \sigma^x_i"))
    parser.add_argument('L', type=int,help='Linear dimensions of the system')
    parser.add_argument('-D', type=int,default=1,
                                        help='Number of spatial dimensions')
    parser.add_argument('--obc',action='store_true',
                            help='Open boundary condintions (deault is PBC)')            
    parser.add_argument('-J', type=float, default=1.0,
                            help='Nearest neighbor Ising coupling')
    parser.add_argument('-o', default='output', help='output filename base')                                        
    args = parser.parse_args()
    ###################################
    
    # Set calculation Parameters
    ###################################
    out_filename_base = args.o
    D = args.D
    L = [ args.L for d in range(D) ]
    PBC = not args.obc
    J = args.J
    parameter_string = "D = {}, L = {}, PBC = {}, J = {}".format(D, L, PBC, J)
    print('\tStarting tfim_build using parameters:\t' + parameter_string)   
    ###################################
    
    # Set up file formatting
    ##################################
    width = 25
    precision = 16
    header = tfim.build_header(L ,PBC, J)
    ##################################
    
    # Build lattice and basis
    ###################################
    lattice = tfim.Lattice(L, PBC)
    N = lattice.N
    basis = tfim.IsingBasis(lattice)
    ###################################
    
    # Compute diagonal matrix elements
    ###################################
    print( '\tBuilding diagonal matrices...' )
    Mz_ME, Ms_ME = tfim.z_magnetizations_ME(lattice,basis)
    JZZ_ME, ZZ_ME = tfim.z_correlations_NN_ME(lattice,basis,J)
    
    # Write to disk
    columns = ['JZZ', 'ZZ', 'Mz', 'Ms']
    diagonal_arr = np.array([JZZ_ME, ZZ_ME, Mz_ME, Ms_ME]).T
    diag_filename = out_filename_base + tfim.diag_ME_suffix
    col_labels = ''.join( [ '{:>{width}}'.format( tfim.phys_labels[key], 
                                width = (width+1) ) for key in columns ] )[3:]
    print( "\tWriting diagonal matrix elements to {}".format(diag_filename) )
    np.savetxt( diag_filename, diagonal_arr, header= ( header + col_labels ),
                    fmt='%{}.{}e'.format(width,precision-1) )
    ###################################
    
    # Compute off-diagonal matrix elements
    ###################################
    print( '\tBuilding off-diagonal matrices...' )
    Mx = tfim.build_Mx(lattice,basis)
        
    # Write to disk
    Mx_filename = out_filename_base + tfim.Mx_suffix
    print( "\tWriting off-diagonal matrix to {}".format(Mx_filename) )
    tfim.save_sparse_matrix(Mx_filename, Mx)
    ###################################
            
if __name__ == "__main__":
    main()
