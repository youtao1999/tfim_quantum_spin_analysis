'''
This script combines the ground states code written by Ellie Copps, number_components written by Jack Landrigan and
the perturbation code written by Tao You to perform comprehensive analysis of quantum spin glasses

Work flow:

import main() NN_ground.py
    --input denoted parameters,
    --save the data to a txt file
    --read the txt file, it is going to be a python dictionary
    --then attach the analysis code, for any seed instance just extract from the python dictionary

import number_components(N, ground_state) where ground_states is a list of indices, exactly the output of Ellie's code
    --input denoted parameters
    --returns dist_comp, which is a dictionary {hamming_distance: number of connected components}, the correct perturbative order
'''

from NN_Ground.NN_ground import main
from NN_Ground.connected_components import number_components
from tfim_perturbation.tfim_survey import tfim_analysis

# First obtain all the ground states

# specifying parameters
xwidth = str(2)
yheight = str(2)
initial_seed = str(2)
seed_range = str(1)
args = [xwidth, yheight, initial_seed, seed_range]
Jij_seed = 2

# calculate ground states, store in dictionary
ground_states_dict, N, L = main(args)

# Now that we have obtained the ground states, we calculate the minimum order needed in perturbation theory
dist_comp, order = number_components(N, ground_states_dict[Jij_seed])

print('order needed: ', order)
# After finishing calculating the minimum perturbation order needed, we can perform the analysis
info = tfim_analysis(L, Jij_seed, order)