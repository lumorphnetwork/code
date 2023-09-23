from utils.routing import generate_graph
from dnc_utils import log_stage, run_lightpath_solver, get_routes
import networkx as nx
from lightpath_solver import scheduled_circuit
from typing import List

num_lambdas = 4
num_chunks = 1
alpha = 0.7
link_bw = 300 * 1024 * 1024 / 1000 * 1024 # GB/ms

def get_comm_demands(file_path):
    fwd = 0
    bwd = 1

    model_parallel_comms = {}
    all_reduce_comms = {}

    with open(file_path, 'r') as comms_file:
        # AR;size;2,10,18,26,34,42,50,58 -> All reduce, different GPUs
        # L2L;size;2;1;7;7 -> Layer to layer, layer number, forward = 0/backward = 1, src gpu, dst gpu

        for line in comms_file:
            parts = line.strip().split(';')
            if parts[0] == 'L2L':
                size = int(parts[1])
                layer_id = int(parts[2])
                direction = int(parts[3])
                src_gpu = int(parts[4])
                dst_gpu = int(parts[5])
                if direction not in model_parallel_comms:
                    model_parallel_comms[direction] = {}
                if layer_id not in model_parallel_comms[direction]:
                    model_parallel_comms[direction][layer_id] = []
                model_parallel_comms[direction][layer_id].append(((src_gpu, dst_gpu), size))
            
            elif parts[0] == 'AR':
                gpus = parts[2].split(',')
                gpu_ids = []
                size = int(parts[1])
                for gpu in gpus:
                    gpu_ids.append(int(gpu))
                gpu_ids.sort()
                ar_key = ','.join([str(gpu_id) for gpu_id in gpu_ids])
                if ar_key not in all_reduce_comms:
                    all_reduce_comms[ar_key] = (gpu_ids, set())
                all_reduce_comms[ar_key][1].add(size)
    
    return model_parallel_comms, all_reduce_comms

def parse_model_parallel_comms(model_parallel_comms):
    for direction in model_parallel_comms:
        for layer_id in model_parallel_comms[direction]:
            # log_info('Direction {}, layer_id {}, number of circuits: {}, circuits: {}'.format(direction, layer_id, len(model_parallel_comms[direction][layer_id]), model_parallel_comms[direction][layer_id]))
            pass

def get_number_of_edges(num_rows, num_cols):
    # vertical + horizontal
    return (num_rows - 1) * num_cols + (num_cols - 1) * num_rows

def get_node_id_matrix(rows, cols):
    node_id = 0
    matrix = {}

    for col in range(cols):
        for row in range(rows):
            if row not in matrix:
                matrix[row] = {}
            if col not in matrix[row]:
                matrix[row][col] = {}
            matrix[row][col] = node_id
            node_id += 1
    
    return matrix

def log_info(data):
    print('[LOG]: {}'.format(data))

def get_updated_size(num_rows, num_cols, matrix, gpus, gpus_in_base_case, max_edges, current_size):
    edges = get_number_of_edges(num_rows, num_cols)
    gpus_in_current_grid = map_grid_size_to_gpus(int(num_rows), int(num_cols), matrix)
    relevant_gpus_in_grid = set(gpus).intersection(set(gpus_in_current_grid))
    if edges > max_edges and len(relevant_gpus_in_grid) == gpus_in_base_case:
        current_size = (int(num_rows), int(num_cols))
        max_edges = edges
    return max_edges, current_size

def get_base_grid_size(rows, cols, gpus, matrix):
    total_gpus = rows * cols
    gpus_per_part = int(total_gpus / len(gpus))

    log_info('Number of required GPUs: {}'.format(len(gpus)))
    log_info('GPUs per bucket: {}'.format(gpus_per_part))
    log_info('Number of buckets: {}'.format(total_gpus / gpus_per_part))

    gpus_in_base_case = 4
    gpus_across_parts = gpus_per_part * gpus_in_base_case
    max_edges = -1
    current_size = (-1, -1)

    for num_rows in range(1, rows + 1):
        if gpus_across_parts % num_rows == 0:
            num_cols = gpus_across_parts / num_rows
            if num_cols > cols:
                continue
            max_edges, current_size = get_updated_size(num_rows, num_cols, matrix, gpus, gpus_in_base_case, max_edges, current_size)
    
    if max_edges == -1:
        for num_cols in range(1, cols + 1):
            if gpus_across_parts % num_cols == 0:
                num_rows = gpus_across_parts / num_cols
                if num_rows > rows:
                    continue
                max_edges, current_size = get_updated_size(num_rows, num_cols, matrix, gpus, gpus_in_base_case, max_edges, current_size)
    
    return current_size

def double_grid_size(curr_rows, curr_cols):
    double_rows = curr_rows * 2
    double_cols = curr_cols * 2

    double_rows_edges = get_number_of_edges(double_rows, curr_cols)
    double_cols_edges = get_number_of_edges(curr_rows, double_cols)

    return (double_rows, curr_cols) if double_rows_edges > double_cols_edges else (curr_rows, double_cols)

def map_grid_size_to_gpus(num_rows, num_cols, matrix):
    nodes_in_grid = []
    for c in range(num_cols):
        for r in range(num_rows):
            nodes_in_grid.append(matrix[r][c])
    return nodes_in_grid

def print_schedule(schedule: List[scheduled_circuit]) -> int:
    rounds_taken = set()

    for circuit in schedule:
        rounds_taken.add(circuit.round_id)
        log_info('Sending chunk {} in round {} along {} on wave {}'.format(circuit.chunk, circuit.round_id, circuit.path, circuit.wavelength))

    return len(rounds_taken)

def get_conditions(sources, sinks, chunks):
    pre_condition = {}
    post_condition = {}
    chunk_id = 0
    for _ in range(chunks):
        pre_condition[chunk_id] = sources
        post_condition[chunk_id] = sources + sinks
        chunk_id += 1
    # for _ in range(chunks):
    #     pre_condition[chunk_id] = sinks
    #     post_condition[chunk_id] = sources + sinks
    #     chunk_id += 1
    
    return pre_condition, post_condition

def get_base_conditions(sources, sinks, num_chunks):
    pre_condition = {}
    post_condition = {}
    chunk_id = 0

    for node_id in range(len(sources)):
        for chunk in range(num_chunks):
            node = sources[node_id]
            pre_condition[chunk_id] = [node]
            post_condition[chunk_id] = sinks
            chunk_id += 1
    
    return pre_condition, post_condition

def get_optimal_rounds_chunks(sources, sinks, routes, cond_fn):
    max_chunks = 4
    current_chunks = 1
    rounds_used, chunks_used = 100, 1
    while(current_chunks <= max_chunks):
        pre_condition, post_condition = cond_fn(sources, sinks, current_chunks)
        schedule = run_lightpath_solver(1, 3, routes, pre_condition, post_condition, num_lambdas)
        num_rounds = print_schedule(schedule)
        ratio = num_rounds / current_chunks
        if ratio == 1:
            rounds_used = num_rounds
            chunks_used = current_chunks
            break

        if ratio < rounds_used / chunks_used:
            rounds_used = num_rounds
            chunks_used = current_chunks
        
        log_info('Min rounds/chunks = {}, current rounds/chunks = {}'.format((num_rounds / current_chunks), (rounds_used / chunks_used)))
        current_chunks += 1
    return rounds_used, chunks_used

def get_maximum_overlap(routes) -> int:
    overlaps = {}
    number_paths = 0
    route_infos = []
    for u in routes:
        for v in routes[u]:
            route_list = routes[u][v]
            for route_info in route_list:
                number_paths += 1
                route_infos.append(','.join([str(x) for x in route_info.nodes]))
                for edge in zip(route_info.nodes, route_info.nodes[1: ]):
                    e_u, e_v = edge
                    if e_u > e_v:
                        e_u, e_v = edge[1], edge[0]
                    edge_id = (e_u, e_v)
                    if edge_id not in overlaps:
                        overlaps[edge_id] = 0
                    overlaps[edge_id] += 1

    max_overlap = 0

    for edge in overlaps:
        if overlaps[edge] > max_overlap:
            max_overlap = overlaps[edge]

    log_info('MAX OVERLAP: {}, NUM_PATHS: {}'.format(max_overlap, number_paths))
    return max_overlap, route_infos

def get_lightpath_cost(rounds_used, chunk_size) -> int:
    bw = link_bw / (num_lambdas * 2)
    beta_cost = rounds_used * (chunk_size / bw)
    alpha_cost = rounds_used * (alpha)  # 0.7 from TACCL paper, 5 for reconfiguration, 5 for consensus
    alpha_cost /= 1000  # converting to ms
    alpha_cost += (rounds_used - 1) * (0.005 + 0.005)
    return alpha_cost + beta_cost

def get_nvlink_cost(rounds_used, chunk_size) -> int:
    beta_cost = rounds_used * (chunk_size / link_bw)
    alpha_cost = rounds_used * (alpha)
    alpha_cost /= 1000  # converting to ms
    return alpha_cost + beta_cost

def parse_ar_comms(all_reduce_comms) -> List[str]:
    global num_lambdas
    rows = 8
    cols = 8
    G: nx.Graph = generate_graph(rows, cols)
    ete_times = []
    # divide the list into groups of 4
    # Get pre condition
    # Get post condition
    # Get routes
    
    matrix = get_node_id_matrix(rows, cols)
    idx = 1
    for ar_key in all_reduce_comms:
        log_info('======================AR-call-{}==================='.format(idx))
        log_info(all_reduce_comms[ar_key])
        gpus, sizes = all_reduce_comms[ar_key]
        
        num_rows, num_cols = get_base_grid_size(rows, cols, gpus, matrix)
        nodes_in_grid = map_grid_size_to_gpus(num_rows, num_cols, matrix)
        H = G.subgraph(nodes_in_grid)
        total_gpus_seen = 4
        sources = [gpu for gpu in gpus if gpu in nodes_in_grid]
        assert(len(sources) == 4)
        log_info((num_rows, num_cols))
        log_info(nodes_in_grid)
        routes = get_routes(H, sources, num_lambdas)
        # pre_condition, post_condition = get_base_conditions(sources, sources, 1)
        # schedule = run_lightpath_solver(1, 3, routes, pre_condition, post_condition, num_lambdas)
        # num_rounds = print_schedule(schedule)
        num_rounds, chunks_used = get_optimal_rounds_chunks(sources, sources, routes, get_base_conditions)
        
        ete_schedule = []
        ete_schedule.append((num_rounds, chunks_used))
        

        while total_gpus_seen < len(gpus):
            total_gpus_seen *= 2
            sources = set(gpus).intersection(set(nodes_in_grid))
            num_rows, num_cols = double_grid_size(num_rows, num_cols)
            log_info((num_rows, num_cols))
            
            nodes_in_grid = map_grid_size_to_gpus(num_rows, num_cols, matrix)
            sinks = set(gpus).intersection(set(nodes_in_grid)).difference(sources)
            log_info(nodes_in_grid)

            sources = list(sources)
            sinks = list(sinks)
            data_nodes = sources + sinks
            H = G.subgraph(nodes_in_grid)
            log_stage(sources, sinks, H, 'AR_{}_{}_{}'.format(idx, len(gpus), total_gpus_seen))
            routes = get_routes(H, data_nodes, num_lambdas)
            get_maximum_overlap(routes)
            rounds_used, chunks_used = get_optimal_rounds_chunks(sources, sinks, routes, get_conditions)

            ete_schedule.append((rounds_used, chunks_used))
        
        # Parse the end to end schedule to get time taken for the collective
        for size in sizes:
            total_cost = 0
            total_nvlink_cost = 0
            for rounds_used, chunks_used in ete_schedule:
                chunk_size = size / chunks_used
                total_cost += get_lightpath_cost(rounds_used, chunk_size)
                total_nvlink_cost += get_nvlink_cost(rounds_used, chunk_size)
            # AR;size;gpu1,gpu2
            ete_times.append('AR;{};{}|{}|{}'.format(size, ar_key, total_cost, ete_schedule))

        idx += 1

    return ete_times

def get_ring_times(all_reduce_comms, bw):
    for ar_key in all_reduce_comms:
        gpus, sizes = all_reduce_comms[ar_key]
        for chunk_size in sizes:
            num_gpus = len(gpus)
            alpha_cost = 2 * (num_gpus - 1) * alpha / 1000
            beta_cost = 2 * (num_gpus - 1) * chunk_size / num_gpus / bw
            total_cost = alpha_cost + beta_cost
            print('Alpha cost: {}, beta_cost: {}'.format(alpha_cost, beta_cost))
            print('AR;{};{}|{}'.format(chunk_size, ar_key, total_cost))

def put_comm_demands(file_path):
    comm_demands = {}
    with open(file_path, 'r') as comm_file:
        for line in comm_file:
            line = line.strip()
            if line not in comm_demands:
                comm_demands[line] = 10
    
    for line in comm_demands:
        print('{}|{}'.format(line, comm_demands[line]))

def put_ete_times(times):
    for time_str in times:
        print(time_str)

# model_parallel_comms, all_reduce_comms = get_comm_demands('comms_topo.log')
# # parse_model_parallel_comms(model_parallel_comms)
# ete_times = parse_ar_comms(all_reduce_comms)
# put_ete_times(ete_times)
# print('=====Ring=====')
# get_ring_times(all_reduce_comms, link_bw)
# put_comm_demands('comms_topo.log')