import numpy as np
import fast_entropy
import argparse

def main():
    # Parse command line arguments
    ###################################

    parser = argparse.ArgumentParser(description=(
        "Use a rapid-GS-generating algorithm combined with fourth order perturbation theory to calculate the entanglement entropy and present it in histogram form."))
    parser.add_argument('x', nargs='+', type = int, help=("width of the system"))
    parser.add_argument('y', nargs='+', type = int, help=("height of the system"))
    parser.add_argument('num_iter', type = int, help=("Number of J_ij instances to sample"))
    parser.add_argument('-PBC', type=bool, default=True, help="Specifying PBC")
    parser.add_argument('-J', type=float, default=1.0, help='Nearest neighbor Ising coupling')
    parser.add_argument('--h_min', type=float, default=0.01,
                        help='Minimum value of the transverse field')
    # parser.add_argument('--h_max', type=float, default=0.1,
    #                     help='Maximum value of the transverse field')
    # parser.add_argument('--num_h', type=float, default=1,
    #                     help='Number of tranverse field step')
    parser.add_argument('-d', default='degeneracy_hist_output', help='output directory base')
    parser.add_argument('-check', type=bool, default=False, help='check to see if the entanglement entropy calculated matches the exact diagonalization; can only be used on small systems.')
    args = parser.parse_args()

    # specifying parameters
    size_list = [[x, y] for x, y in zip(args.x, args.y)]
    num_iter = args.num_iter
    h_x_range = [args.h_min]

    print(size_list, type(size_list))

    degeneracy_hist_data, order_hist_data, entropy_hist_data = fast_entropy.fast_entropy(h_x_range, num_iter, size_list)

    # store data
    file_name = 'hist_{s_list}_{iter}_{h_x}'.format(s_list = size_list, iter = num_iter, h_x = h_x_range)
    fast_entropy.data_store(args.d, file_name, [degeneracy_hist_data, order_hist_data, entropy_hist_data])

if __name__ == '__main__':
    main()