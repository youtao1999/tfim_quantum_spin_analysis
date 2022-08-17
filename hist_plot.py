import os

import fast_entropy
import numpy as np
import os
# specifying parameters
size_list = [[4,4]]
num_iter = 500
h_x_range = np.linspace(0.01, 4., 1)

# store data
output = 'degeneracy_hist_output'
file_name = 'hist_{s_list}_{iter}_{h_x}'.format(s_list = size_list, iter = num_iter, h_x = h_x_range)

degeneracy_hist_data, order_hist_data, entropy_hist_data = fast_entropy.data_read(output, file_name)
fast_entropy.entropy_hist(entropy_hist_data, degeneracy_hist_data, size_list)

os.chdir('../')