#!/usr/bin/env python

""""tfim_search.py
    Tao You
    07.02/2020
    --Search for second order perturbation theory approximable Jij matrices
    --Requires: tfim.py, numpy, scipy.sparse, scipy.linalg, progressbar
"""

import tfim
import tfim_perturbation
import tfim_rdm
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy import linalg
import progressbar
import argparse

###############################################################################
def main():
    
    # Parse command line arguements
    ###################################
    parser = argparse.ArgumentParser(description=("Search for second order perturbation theory approximable Jij matrices") )
    parser.add_argument('lattice_specifier', 
                            help=(  "Either: L (linear dimensions of the system)"
                                    " or the filename base of matrix files") )
    parser.add_argument('seed_limit', help = ("Specifying the range of seeds to search through"))
    
    parser.add_argument('-o', default='output', help='output filename base')
    
    parser.add_argument('-PBC', type = bool, default = True, help = "Specifying PBC")
    
    parser.add_argument('-J', type=float, default = 1.0, help = 'Nearest neighbor Ising coupling')
    
    parser.add_argument('-max_order', type=int, default = 4, help = 'The maximum Hamming distance between degenerate ground states')
    
    args = parser.parse_args()
    ###################################
    
    L = [ int(args.lattice_specifier) ]
    PBC = args.PBC
    J = args.J
    seed_limit = int(args.seed_limit)
    max_order = int(args.max_order)
    
    # Build lattice and basis
    lattice = tfim.Lattice(L, PBC)
    N = lattice.N
    basis = tfim.IsingBasis(lattice)
    
    # Specifying parameters needed
    seed_range = range(seed_limit)
    
    # Begin search
    Jij_array = [];

    for i in seed_range:
        Jij = tfim.Jij_instance(N,J,"bimodal",i) 
        Jij_array.append(Jij)
        
    # Calculate energy array:
    indices_array = []

    for Jij in Jij_array:
        Energies = -tfim.JZZ_SK_ME(basis,Jij)
        GS_energy = np.min(Energies)
        GS_indices = np.nonzero(Energies == GS_energy)[0]
        indices_array.append(GS_indices)
    
    # Search for Hamming distance 2
    seed_list = []
    
    for index, indices in enumerate(indices_array):
        if tfim_perturbation.judge(max_order, tfim_perturbation.Hamming_array(indices, basis), N):
            seed_list.append(index)

    print(seed_list)
    
if __name__ == "__main__":
    main()
