import networkx as nx
from itertools import islice
import pickle
import sys
import json
import random


def generate_graph_manually(num_rows, num_cols, unusable_nodes):
    G = nx.Graph()
    node_ids = []

    node_idx = 0

    # add nodes
    for row in range(num_rows):
        for col in range(num_cols):
            x = num_rows - row
            y = col
            G.add_node(node_idx, pos=(x, y))
            print('Node : {} at position ({},{})'.format(node_idx, x, y))
            node_ids.append(node_idx)
            node_idx += 1

    node_idx = 0
    # add horizontal edges
    for row in range(num_rows):
        for col in range(num_cols - 1):
            from_node = node_idx
            to_node = node_idx + 1
            if from_node in unusable_nodes or to_node in unusable_nodes:
                continue
            G.add_edge(from_node, to_node, weight=0)
            print('Edge from: {} to {}'.format(node_idx, node_idx + 1))
            node_idx += 1
        node_idx += 1

    node_idx = 0
    # add vertical edges
    for row in range(num_rows - 1):
        for col in range(num_cols):
            from_node = node_idx
            to_node = node_idx + num_cols
            if from_node in unusable_nodes or to_node in unusable_nodes:
                continue
            G.add_edge(from_node, to_node, weight=0)
            print('Edge from: {} to {}'.format(node_idx, node_idx + num_cols))
            node_idx += 1

    return G, node_ids

def generate_graph(num_rows, num_cols):
    g = nx.grid_2d_graph(num_rows, num_cols)
    g_int = nx.convert_node_labels_to_integers(g, ordering='sorted', label_attribute='pos')
    return g_int

def print_edge_weights(G):
    for edge in G.edges(data=True):
        print(edge)

def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )

def reassign_weights(G, path):
    print('Updating weights for: {}'.format(path))
    for node_idx in range(len(path) - 1):
        edge_start = path[node_idx]
        edge_end = path[node_idx + 1]
        G[edge_start][edge_end]['weight'] += 1

def reset_weights(G):
    for s,v,d in G.edges(data=True):
        d['weight'] = 0

def get_path_weight(path):
    path_weight = 0

    for node_idx in range(len(path) - 1):
        edge_start = path[node_idx]
        edge_end = path[node_idx + 1]
        path_weight += G[edge_start][edge_end]['weight']
        
    return path_weight

def generate_weighted_routes(G, node_ids):
    routes = {}
    all_routes = {}
    for wavelength in range(wavelengths):
        reset_weights(G)
        random.shuffle(node_ids)
        print('Shuffling done')
        for source in node_ids:
            for destination in node_ids:
                if source != destination:
                    print('Edge weights for: {}-{}'.format(source, destination))
                    print_edge_weights(G)
                    paths = k_shortest_paths(G, source, destination, 1000)
                    if source not in routes:
                        routes[source] = {}
                        all_routes[source] = {}
                    if destination not in routes[source]:
                        routes[source][destination] = []
                    
                    all_routes[source][destination] = paths
                    routes[source][destination].append([wavelength, paths[0]])
                    weights = []
                    for path in paths:
                        print('Weight of path {} is {}'.format(path, get_path_weight(path)))
                        weights.append(get_path_weight(path))
                    
                    path_min_idx = weights.index(min(weights))
                    # path_min_idx = 0
                    print('Picked path: {} from source {} to destination {}'.format(paths[path_min_idx], source, destination))
                    reassign_weights(G, paths[path_min_idx])
                    print('================')

    return routes, all_routes

def find_path_in_list(wavelength, path, picked_paths):
    for picked_path in picked_paths:
        w, nodes_in_picked_path = picked_path
        if w == wavelength:
            for node in path:
                if node in nodes_in_picked_path:
                    return True
    return False

def generate_all_routes(G, node_ids, num_routes, wavelengths, unused_nodes):
    routes = {}

    for source in node_ids:
        for destination in node_ids:
            if source in unused_nodes or destination in unused_nodes:
                continue
            if source != destination:
                paths = k_shortest_paths(G, source, destination, num_routes)
                # print('Num paths between: {} {} is {}'.format(source, destination, len(paths)))
                for path in paths:
                    for w in range(wavelengths):
                        if source not in routes:
                            routes[source] = {}
                        if destination not in routes[source]:
                            routes[source][destination] = []
                        routes[source][destination].append([w, path]) 
    return routes

def generate_routes(G, node_ids, wavelengths):
    routes = {}
    all_routes = {}
    picked_paths = []
    
    for source in node_ids:
        for destination in node_ids:
            found_path = False
            for wavelength in range(wavelengths):
                if found_path:
                    break
                if source != destination:
                    paths = k_shortest_paths(G, source, destination, 1000)
                    print('Paths from {} to {} are {}, paths in use {}'.format(source, destination, paths, picked_paths))
                    for path in paths:
                        found = find_path_in_list(wavelength, path, picked_paths)
                        if not found:
                            if source not in routes:
                                routes[source] = {}
                                all_routes[source] = {}
                            if destination not in routes[source]:
                                routes[source][destination] = []
                                all_routes[source][destination] = paths
                            routes[source][destination].append([wavelength, path])
                            picked_paths.append([wavelength, path])
                            print('Picked path {}, on wavlength {} from {} to {}'.format(path, wavelength, source, destination))
                            found_path = True
                            break
                            
    return routes, all_routes

def save_routes(routes, all_routes, rows, cols, name=''):
    with open('all_routes_{}{}.json'.format(rows * cols, name), 'w') as json_file:
        json_file.write(json.dumps(all_routes))
    
    with open('routes_{}{}.json'.format(rows * cols, name), 'w') as json_file:
        json_file.write(json.dumps(routes))

    with open('routes_{}{}.pkl'.format(rows * cols, name), 'wb') as pickle_file:
        pickle.dump(routes, pickle_file)

def run():
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    num_routes = int(sys.argv[3])

    wavelengths = 6
    G, node_ids = generate_graph(rows, cols)

    routes, all_routes = generate_all_routes(G, node_ids, num_routes, wavelengths)
    save_routes(routes, all_routes, rows, cols)

# run()