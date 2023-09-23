from typing import List
import random
from utils.routing import generate_graph
from parse_comms import get_comm_demands, get_maximum_overlap, log_info
import networkx as nx
from formulate_encoding import formulate_routes
import os
from dnc_utils import run_lightpath_solver
import collective_algorithms
import argparse
import json

parser = argparse.ArgumentParser(description='A program that accepts cli arguments for number of rows, columns and algorithm')
parser.add_argument('--gpus', type=int, required=True, help='Number of GPUs', choices=[4, 8 , 16, 32, 64, 128, 256])
parser.add_argument('--algorithm', type=str, required=True, choices=['lumorph_2', 'lumorph_4', 'ring'], help='Algorithm to use')
parser.add_argument('--model', type=str, required=True, choices=['bert', 'ncf', 'inception'], help='Algorithm to use')

args = parser.parse_args()


def have_common_gpus(gpu_list_1, gpu_list_2):
    set1 = set(gpu_list_1)
    set2 = set(gpu_list_2)
    return len(set1.intersection(set2)) > 0

def add_dependent_edges(dependency_graph: nx.Graph):
    nodes = dependency_graph.nodes()
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            gpu_list_u = dependency_graph.nodes[u]['compute_nodes']
            gpu_list_v = dependency_graph.nodes[v]['compute_nodes']
            if have_common_gpus(gpu_list_u, gpu_list_v):
                dependency_graph.add_edge(u, v)

def extract_nodes(dependency_graph: nx.Graph, node_list):
    extracted_nodes_info = []
    for node in node_list:
        compute_nodes = dependency_graph.nodes[node]['compute_nodes']
        transfer_sizes = dependency_graph.nodes[node]['xfer_size']
        extracted_nodes_info.append((compute_nodes, transfer_sizes))
        dependency_graph.remove_node(node)
    return extracted_nodes_info

def get_cover_set(dependency_graph: nx.Graph):
    sets = []
    while len(dependency_graph.nodes()) > 0:
        try:
            maximal_independent_set_nodes = nx.maximal_independent_set(dependency_graph)
            maximal_independent_set = extract_nodes(dependency_graph, maximal_independent_set_nodes)
            sets.append(maximal_independent_set)
        except Exception as e:
            remaining_nodes = dependency_graph.nodes()
            sets.append(extract_nodes(dependency_graph, remaining_nodes))
    assert(len(dependency_graph.nodes()) == 0)
    return sets

def print_pattern(required_links):
    links_to_print = {}
    for link in required_links:
        u, v = link
        if u not in links_to_print:
            links_to_print[u] = []
        links_to_print[u].append(v)
    for u in links_to_print:
        links_to_print[u].sort()
        print('links are from {}/{}: {}'.format(u, len(links_to_print[u]), links_to_print[u]))

def use_lightpath_encoding(circuits):
    num_lambdas = 4
    routes_formulator = formulate_routes(num_lambdas, G)
    all_routes = routes_formulator.get_routes_for(circuits, G)
    # get pre-condition chunk: list of gpus
    chunk_id = 0
    pre_condition = {}
    post_condition = {}
    for circuit in circuits:
        src, dst = circuit
        pre_condition[chunk_id] = [src]
        post_condition[chunk_id] = [src, dst]
        chunk_id += 1
    
    schedule = run_lightpath_solver(20, 30, all_routes, pre_condition, post_condition, num_lambdas)
    max_rounds = 0

    for ordered_circuit in schedule:
        if ordered_circuit.round_id > max_rounds:
            max_rounds = ordered_circuit.round_id

    print('Max rounds = {}'.format(max_rounds))

def zero_cost_serial_execution(log_file_path, parsed_file_path, G: nx.Graph):
    l2l_comms, ar_comms = get_comm_demands(log_file_path)
    dependency_graph = nx.Graph()
    node_id = 0
    parsed_file_path = parsed_file_path.format('zero')

    l2l_strings, max_len = collective_algorithms.get_zero_l2l_times(l2l_comms, G)
    ar_strings = []

    for ar_key in ar_comms:
        gpus, xfer_sizes = ar_comms[ar_key]
        # gpus, xfer_sizes = [g for g in range(rows * cols)], [4096]
        for xfer_size in xfer_sizes:
            comm_cost = 0
            comm_string = 'AR;{};{}|{}\n'.format(xfer_size, ar_key, comm_cost)
            ar_strings.append(comm_string)
    
    comm_strings_to_store = []
    comm_strings_to_store.extend(l2l_strings)
    comm_strings_to_store.extend(ar_strings)

    collective_algorithms.generate_communication_file_for_simulation(comm_strings_to_store, parsed_file_path)

def generate_mcf_routes(transfers: List[List[collective_algorithms.required_circuit]], G):
    parsed_step_routes = []
    for idx, step_transfers in enumerate(transfers):
        links_in_round = [(circuit.src, circuit.dst) for circuit in step_transfers]
        routes_formulator = formulate_routes(1, G)
        all_routes = routes_formulator.get_routes_for(list(links_in_round), G)
        print('[LOG]: Round: {}, links: {}'.format(idx, len(links_in_round)))
        max_overlap, parsed_routes = get_maximum_overlap(all_routes)
        parsed_step_routes.append({"step": idx, "circuits": parsed_routes})
    return parsed_step_routes

def write_json(file_path, data):
    with open(file_path, 'w') as json_file:
        json_file.write(json.dumps(data, indent='\t'))

def inorder_iteration(log_file_path, parsed_file_path, mcf_file_path, G: nx.Graph, algorithm):
    l2l_comms, ar_comms = get_comm_demands(log_file_path)
    dependency_graph = nx.Graph()
    node_id = 0
    parsed_file_path = parsed_file_path.format(algorithm)
    mcf_strings_to_store = []

    algorithm_fn = collective_algorithms.get_quad_recursive_time
    l2l_function = collective_algorithms.get_layer_to_layer_times
    transfers_fn = collective_algorithms.quad_all_reduce

    if algorithm == 'lumorph_2':
        algorithm_fn = collective_algorithms.get_optimal_time
        transfers_fn = collective_algorithms.recursive_allreduce
    if algorithm == 'ring':
        algorithm_fn = collective_algorithms.get_ring_time
        transfers_fn = None
    
    l2l_strings, l2l_mcf_routes = l2l_function(l2l_comms, G)
    ar_strings = []
    ar_routes = {}

    for ar_key in ar_comms:
        gpus, xfer_sizes = ar_comms[ar_key]
        if len(gpus) < 2:
            continue
        for xfer_size in xfer_sizes:
            comm_cost = algorithm_fn(len(gpus), xfer_size)
            comm_string = 'AR;{};{}|{}\n'.format(xfer_size, ar_key, comm_cost / 1000)
            ar_strings.append(comm_string)
        if transfers_fn != None:
            transfers = transfers_fn(gpus, xfer_size)
            if len(transfers) > 0:
                print('[LOG]: GENERATING MCF ROUTES')
                parsed_routes = generate_mcf_routes(transfers, G)
                ar_routes[ar_key] = parsed_routes
    comm_strings_to_store = []
    comm_strings_to_store.extend(l2l_strings)
    comm_strings_to_store.extend(ar_strings)

    collective_algorithms.generate_communication_file_for_simulation(comm_strings_to_store, parsed_file_path)
    write_json(mcf_file_path, {'AR': ar_routes, 'L2L': l2l_mcf_routes})

def coverset_iteration(log_file_path, parsed_file_path, G: nx.Graph, algorithm):
    l2l_comms, ar_comms = get_comm_demands(log_file_path)
    dependency_graph = nx.Graph()
    node_id = 0
    parsed_file_path = parsed_file_path.format(algorithm)

    algorithm_fn = collective_algorithms.get_quad_recursive_time
    transfers_fn = collective_algorithms.quad_all_reduce
    if algorithm == 'lumorph_2':
        algorithm_fn = collective_algorithms.get_optimal_time
        transfers_fn = None
    if algorithm == 'ring':
        algorithm_fn = collective_algorithms.get_ring_time
        transfers_fn = collective_algorithms.recursive_allreduce

    for ar_key in ar_comms:
        gpus, xfer_sizes = ar_comms[ar_key]
        if len(gpus) > 1:
            dependency_graph.add_node(node_id, compute_nodes=gpus, xfer_size=sum(xfer_sizes))
            node_id += 1
    
    add_dependent_edges(dependency_graph)
    cover_set = get_cover_set(dependency_graph)
    
    l2l_strings, last_layer = collective_algorithms.get_layer_to_layer_times(l2l_comms, G)
    ar_time = collective_algorithms.parse_cover_set(cover_set, G, algorithm=algorithm_fn)
    sliced_entry = 'SE;1|{}\n'.format(ar_time)
    ar_strings = collective_algorithms.get_zeroed_ar_times(ar_comms)

    comm_strings_to_store = []
    comm_strings_to_store.extend(l2l_strings)
    comm_strings_to_store.extend(ar_strings)
    comm_strings_to_store.append(sliced_entry)
    
    collective_algorithms.generate_communication_file_for_simulation(comm_strings_to_store, parsed_file_path)

random.seed(8)
num_gpus_to_grid = {
    4: (2, 2),
    8: (2, 4),
    16: (4, 4),
    32: (4, 8),
    64: (8, 8),
    128: (8, 16),
    256: (16, 16)
}
rows, cols = num_gpus_to_grid[args.gpus]
grid_size = rows, cols
num_gpus = grid_size[0] * grid_size[1]
folder_path = './logs/{}/'.format(args.model)
log_file_path = os.path.join(folder_path, 'comms_{}.log'.format(num_gpus))
algorithms = ['lumorph_4', 'ring', 'lumorph_2']
parsed_file_path_template = os.path.join(folder_path, 'comms_algorithm_{}_gpus_{}.log')

algo = args.algorithm
parsed_file_path = parsed_file_path_template.format(algo, num_gpus)
mcf_file_path_template = os.path.join(folder_path, 'mcf_paths_algorithm_{}_gpus_{}.json')
mcf_file_path = mcf_file_path_template.format(algo, num_gpus)

G: nx.Graph = generate_graph(rows, cols)
inorder_iteration(log_file_path, parsed_file_path, mcf_file_path, G, algo)
