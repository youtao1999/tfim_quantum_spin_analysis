import networkx as nx
import matplotlib.pyplot as plt
import argparse
import NN_ground
import os
import json
import ast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('xwidth', type=int, help='Width of grid')
    parser.add_argument('yheight', type=int, help='Height of grid')
    parser.add_argument('filename', type=str, help='Ground States File')
    args = parser.parse_args()

    yheight = args.yheight
    xwidth = args.xwidth
    filename = args.filename
    N = xwidth * yheight

    f = open(filename, "r")
    contents = f.read()
    dictionary = ast.literal_eval(contents)

    hamming_dist = []
    print(dictionary)

    for item in dictionary:
        gs = dictionary[item]
        intd = number_components(N, gs)
        dist = intd[1]
        hamming_dist.append(dist)

    f2 = open("hamming_dists.txt", "w+")
    f2.write(json.dumps(hamming_dist))
    f2.close()



def number_components(N, ground_states):
    G = nx.Graph()
    G.add_nodes_from(ground_states)
    #nx.draw(G)
    #plt.show()
    num_comp = 0
    dist = 1
    dist_comp = dict()
    hd_dict = dict()
    N_str = str(N)

    for i in range(len(ground_states)):
        print(len(ground_states))
        str1 = '{0:0' + N_str + 'b}'
        state1 = str1.format(ground_states[i])
        for j in range(i+1, len(ground_states)):
            state2 = str1.format(ground_states[j])
            count = sum(1 for a, b in zip(state1, state2) if a != b)
            hd_dict[(state1, state2)] = count

    while num_comp != 1:
        #print(dist)
        for edge in hd_dict.keys():
            if hd_dict[edge] == dist:
                node1 = int(edge[0], 2)
                node2 = int(edge[1], 2)
                G.add_edge(node1, node2)
        #nx.draw(G, node_size=50)
        #plt.show()
        num_comp = nx.number_connected_components(G)
        dist_comp[dist] = num_comp
        dist += 1

    return dist_comp, dist-1



if __name__ == "__main__":
    main()
