import numpy as np
import fast_entropy

# specifying parameters
size_list = [[4,4]]
num_iter = 500
h_x_range = np.linspace(0.01, 4., 1)

degeneracy_hist_data, order_hist_data, entropy_hist_data = fast_entropy.fast_entropy(h_x_range, num_iter, size_list)

# store data
output = 'degeneracy_hist_output'
file_name = 'hist_{s_list}_{iter}_{h_x}'.format(s_list = size_list, iter = num_iter, h_x = h_x_range)
fast_entropy.data_store(output, file_name, [degeneracy_hist_data, order_hist_data, entropy_hist_data])