from typing import List
import math
from parse_comms import get_comm_demands, get_maximum_overlap, log_info
import networkx as nx
from formulate_encoding import formulate_routes
import random

all_link_speed = 300 * 1024 * 1024 * 1024 # GB/s
single_link_speed = all_link_speed / 4
alpha = 0.7
reconfiguration_delay = 3.7 # micro second

class required_circuit:
    def __init__(self, src: int, dst: int, xfer_size: int) -> None:
        self.src = src
        self.dst = dst
        self.xfer_size = xfer_size

def recursive_allreduce(gpus, xfer_size) -> List[List[required_circuit]]:
    n_participants = len(gpus)
    num_steps = math.log2(n_participants)
    print('[LOG]:Number of participants: {}'.format(n_participants))
    transfers: List[List[required_circuit]] = []

    assert((n_participants & (n_participants -1)) == 0) # Has to be a power of 2
    print('[LOG]:Number of steps: {}'.format(num_steps))
    num_steps = int(num_steps)

    print('[LOG]:====================')
    print('[LOG]:Reduce scatter phase')
    print('[LOG]:====================')

    for i in range(num_steps):
        step_transfers: List[required_circuit] = []
        bucket_size = int(n_participants / math.pow(2, i))
        skip_length = bucket_size / 2
        peers = []
        transfer_size = xfer_size / math.pow(2, i + 1)
        bucket_start = -1

        for node in range(bucket_size):
            peer = (node + skip_length) % bucket_size
            peers.append(peer)
        
        assert(len(peers) == bucket_size)

        for g in range(n_participants):
            gpu_idx_in_bucket = -1
            peer_idx_in_bucket = -1
            real_peer_idx = -1
            
            if g % bucket_size == 0:
                bucket_start = g

            gpu_idx_in_bucket = g % bucket_size
            peer_idx_in_bucket = peers[gpu_idx_in_bucket]
            real_peer_idx = int(bucket_start + peer_idx_in_bucket)

            src_gpu = gpus[g]
            dst_gpu = gpus[real_peer_idx]
            
            step_transfers.append(required_circuit(src_gpu, dst_gpu, transfer_size))
            # print("Sending data from {} to {} of size: {}".format(src_gpu, dst_gpu, transfer_size))

        transfers.append(step_transfers)
    print('[LOG]:====================')
    print('[LOG]:All-gather phase')
    print('[LOG]:====================')

    for i in range(num_steps):
        step_transfers: List[required_circuit] = []
        bucket_size = int(math.pow(2, i + 1))
        skip_length = bucket_size / 2
        transfer_size = xfer_size / n_participants * math.pow(2, i)
        peers = []
        bucket_start = -1

        for node in range(bucket_size):
            peer = (node + skip_length) % bucket_size
            peers.append(peer)
        
        assert(len(peers) == bucket_size)

        for g in range(n_participants):
            gpu_idx_in_bucket = -1
            peer_idx_in_bucket = -1
            real_peer_idx = -1

            if g % bucket_size == 0:
                bucket_start = g

            gpu_idx_in_bucket = g % bucket_size
            peer_idx_in_bucket = peers[gpu_idx_in_bucket]
            real_peer_idx = int(bucket_start + peer_idx_in_bucket)


            src_gpu = gpus[g]
            dst_gpu = gpus[real_peer_idx]
            
            step_transfers.append(required_circuit(src_gpu, dst_gpu, transfer_size))

            # print("Sending data from {} to {} of size: {}".format(src_gpu, dst_gpu, transfer_size))
        
        transfers.append(step_transfers)

    return transfers

def get_quad_step_transfers(bucket_size, quad_size, skip_length, transfer_size, gpus):
    step_transfers = []
    current_bucket_start = 0
    for gpu_id in range(len(gpus)):
        if gpu_id % bucket_size == 0:
            current_bucket_start = gpu_id
        
        for neighour_idx in range(1, quad_size):
            peer_id = gpu_id + (neighour_idx * skip_length)
            peer_id = current_bucket_start + (peer_id % bucket_size)
            src_gpu = gpus[gpu_id]
            dst_gpu = gpus[peer_id]
            step_transfers.append(required_circuit(src_gpu, dst_gpu, transfer_size))
    return step_transfers

def quad_all_reduce(gpus, xfer_size) -> List[List[required_circuit]]:
    transfers: List[List[required_circuit]] = []
    # scatter-reduce
    bucket_size = 4
    skip_length = 1
    transfer_size = xfer_size / 4
    quad_size = 4

    while bucket_size <= len(gpus):
        step_transfers = get_quad_step_transfers(bucket_size, quad_size, skip_length, transfer_size, gpus)
                 
        bucket_size *= 4
        skip_length *= 4
        transfer_size /= 4

        transfers.append(step_transfers)
    
    last_bucket_size = int(bucket_size / 4)
    last_transfer_size = transfer_size * 4

    if last_bucket_size < len(gpus):
        step_transfers = []
        assert(last_bucket_size == len(gpus) / 2)
        for gpu_id in range(last_bucket_size):
            src_gpu = gpus[gpu_id]
            dst_gpu = gpus[gpu_id + last_bucket_size]
            step_transfers.append(required_circuit(src_gpu, dst_gpu, last_transfer_size / 2))
        transfers.append(step_transfers)

    # all gather
    transfer_size = xfer_size / len(gpus)
    bucket_size = 4
    skip_length = 1
    
    while bucket_size <= len(gpus):
        step_transfers = get_quad_step_transfers(bucket_size, quad_size, skip_length, transfer_size, gpus)
        
        bucket_size *= 4
        skip_length *= 4
        transfer_size *= 4

        transfers.append(step_transfers)
    last_bucket_size = int(bucket_size / 4)
    last_transfer_size = transfer_size / 4

    if last_bucket_size < len(gpus):
        step_transfers = []
        assert(last_bucket_size == len(gpus) / 2)
        for gpu_id in range(last_bucket_size):
            src_gpu = gpus[gpu_id]
            dst_gpu = gpus[gpu_id + last_bucket_size]
            step_transfers.append(required_circuit(src_gpu, dst_gpu, last_transfer_size * 4))
        transfers.append(step_transfers)
    
    return transfers

def simulate_all_reduce(gpus, transfers: List[List[required_circuit]], buffer_size):
    buffers = []
    for gpu_id in range(len(gpus)):
        buffer = [(gpu_id + 1) for _ in range(len(gpus))]
        buffers.append(buffer)
    assert(len(buffers) == len(gpus))

    chunk_size = buffer_size / len(gpus)

    transfer_idx = 0

    # scatter-reduce
    bucket_size = 4

    single_link_transfer_size = 0
    all_link_transfer_size = 0

    while bucket_size <= len(gpus):
        # dequeue steps and perform the transfers
        step_transfers = transfers[transfer_idx]
        single_link_transfer_size += step_transfers[0].xfer_size
        for pair_transfer in step_transfers:
            source_buffer = buffers[pair_transfer.src]
            destination_buffer = buffers[pair_transfer.dst]
            src_bucket_start = int(math.floor(pair_transfer.src / bucket_size)) * bucket_size
            dst_offset = pair_transfer.dst - src_bucket_start
            src_offset = pair_transfer.src - src_bucket_start
            num_buckets = int(len(gpus) / bucket_size)
            for idx in range(num_buckets):
                bucket_offset = idx * bucket_size
                data_offset = bucket_offset + dst_offset
                destination_buffer[data_offset] = destination_buffer[data_offset] + source_buffer[data_offset]

        transfer_idx += 1
        bucket_size *= 4
    
    last_bucket_size = bucket_size / 4
    if last_bucket_size < len(gpus):
        step_transfers = transfers[transfer_idx]
        all_link_transfer_size += step_transfers[0].xfer_size
        for pair_transfer in step_transfers:
            source_buffer = buffers[pair_transfer.src]
            destination_buffer = buffers[pair_transfer.dst]
            destination_buffer[pair_transfer.dst] += source_buffer[pair_transfer.dst]
            source_buffer[pair_transfer.src] += destination_buffer[pair_transfer.src]
        transfer_idx += 1

    for buffer_idx in range(len(buffers)):
        buffer = buffers[buffer_idx]
        new_buffer = [-1 for _ in range(len(buffer))]
        new_buffer[buffer_idx] = buffer[buffer_idx]
        buffers[buffer_idx] = new_buffer

    # all gather
    bucket_size = 4
    
    while bucket_size <= len(gpus):
        step_transfers = transfers[transfer_idx]
        chunks_to_send = []
        single_link_transfer_size += step_transfers[0].xfer_size
        for buffer_idx in range(len(buffers)):
            buffer = buffers[buffer_idx]
            chunks_in_buffer = set([idx for idx in range(len(buffer)) if buffer[idx] != -1])
            chunks_to_send.append(chunks_in_buffer)
        
        for pair_transfer in step_transfers:
            source_buffer = buffers[pair_transfer.src]
            destination_buffer = buffers[pair_transfer.dst]

            src_chunks_to_send = chunks_to_send[pair_transfer.src]
            dst_chunks_to_send = chunks_to_send[pair_transfer.dst]   

            copied_size = 0
            for data_idx in range(len(source_buffer)):
                if data_idx in src_chunks_to_send:
                    destination_buffer[data_idx] = source_buffer[data_idx]
                    copied_size += 1
                if data_idx in dst_chunks_to_send:
                    source_buffer[data_idx] = destination_buffer[data_idx]
            assert(copied_size == (pair_transfer.xfer_size / chunk_size))
        transfer_idx += 1
        bucket_size *= 4
    
    last_bucket_size = bucket_size / 4
    if last_bucket_size < len(gpus):
        step_transfers = transfers[transfer_idx]
        all_link_transfer_size += step_transfers[0].xfer_size
        for pair_transfer in step_transfers:
            source_buffer = buffers[pair_transfer.src]
            destination_buffer = buffers[pair_transfer.dst]
            copied_size = 0

            for data_idx in range(len(source_buffer)):
                if source_buffer[data_idx] != -1:
                    destination_buffer[data_idx] = source_buffer[data_idx]
                    copied_size += 1
                if destination_buffer[data_idx] != -1:
                    source_buffer[data_idx] = destination_buffer[data_idx]
            assert(copied_size == (pair_transfer.xfer_size / chunk_size))
    
    # for buffer in buffers:
    #     print(buffer)
    
    number_steps = len(transfers)
    alpha_cost = number_steps * (alpha + 5)
    single_link_beta_cost = single_link_transfer_size * 1000 * 1000 / single_link_speed # micro seconds
    all_link_beta_cost = all_link_transfer_size * 1000 * 1000 / all_link_speed # micro seconds
    beta_cost = single_link_beta_cost + all_link_beta_cost

    print('[LOG]:Number of steps: {}, alpha cost: {}'.format(number_steps, alpha_cost))
    print('[LOG]:Data transfered in single links: {}, all links: {}'.format(single_link_transfer_size, all_link_transfer_size))
    print('[LOG]:Single link beta cost: {}, all link beta cost: {}'.format(single_link_beta_cost, all_link_beta_cost))
    print('[LOG]:QUAD SIM: buffer size {}, alpha: {}, beta: {}'.format(buffer_size, alpha_cost, beta_cost))

    return alpha_cost + beta_cost

def get_ring_time(num_nodes, buffer_size):
    time_val = 0
    alpha_cost = 2 *  (num_nodes - 1) * 0.7 
    beta_cost = 2 * buffer_size * (num_nodes - 1) / num_nodes / all_link_speed
    beta_cost *= 1000 * 1000
    time_val = alpha_cost + beta_cost
    print('[LOG]:RING: buffer size {}, alpha: {}, beta: {}'.format(buffer_size, alpha_cost, beta_cost))
    return time_val

def get_quad_recursive_time(num_nodes, buffer_size):
    time_val = 0
    t_nodes = num_nodes / 1
    num_rounds = 2 * math.ceil(math.log(t_nodes, 4))
    alpha_cost = (num_rounds) * (alpha + 5)
    beta_cost = 2 * buffer_size * (t_nodes - 1) / t_nodes / all_link_speed * (4 / 3)
    beta_cost *= 1000 * 1000
    time_val = alpha_cost + beta_cost
    print('[LOG]:LIGHTPATH_4: buffer size {}, alpha: {}, beta: {}'.format(buffer_size, alpha_cost, beta_cost))
    return time_val

def is_all_to_all(links):
    neighbors = {}
    for link in links:
        u, v = link
        if u not in neighbors:
            neighbors[u] = set()
        neighbors[u].add(v)

    for u in neighbors:
        if len(neighbors[u]) != len(neighbors) - 1:
            return False

    return True


def get_random_pattern_rounds(required_links, num_lasers, G: nx.Graph):
    num_links = {}
    actual_links = set()
    for link in required_links:
        src, dst = link
        reverse_link = (dst, src)
        if link not in actual_links and reverse_link not in actual_links:
            actual_links.add(link)
            actual_links.add(reverse_link)

    for link in actual_links:
        src, dst = link
        if dst not in num_links:
            num_links[dst] = []
        num_links[dst].append(src)
    
    for link in required_links:
        src, dst = link
        reverse_link = (dst, src)
        assert(link in actual_links or reverse_link in actual_links)
    
    print('[LOG]: Actual links in Random L2L: {}'.format(len(actual_links)))

    round_id = 0
    all_parsed_routes = []
    while True:
        links_in_round = set()
        for dst in num_links:
            sources = num_links[dst]
            picked = []
            while len(picked) < num_lasers and len(sources) > 0:
                source = random.choice(sources)
                picked_link = (source, dst)
                assert(picked_link not in links_in_round)
                picked.append(picked_link)
                links_in_round.add(picked_link)
                sources.remove(source)

        # print('[LOG]:Links in round {} are {}'.format(round_id, links_in_round))
        round_id += 1

        routes_formulator = formulate_routes(1, G)
        all_routes = routes_formulator.get_routes_for(list(links_in_round), G)
        max_overlap, parsed_routes = get_maximum_overlap(all_routes)
        # print('[LOG]:MAX OVERLAP IS: {}'.format(max_overlap))
        all_parsed_routes.append(parsed_routes)
        should_break = True
        for dst in num_links:
            if len(num_links[dst]) > 0:
                # print('[LOG]:Num links in dst {} is {}'.format(dst, num_links[dst]))
                should_break = False
        
        if should_break:
            break
    
    return round_id, all_parsed_routes
            

def get_layer_to_layer_times(l2l_comms, G: nx.Graph):
    communication_strings = []
    layers = set()
    layer_to_layer_mcf_routes = {}
    for direction in l2l_comms:
        for layer_id in l2l_comms[direction]:
            layers.add(layer_id)
            comms_in_layer = l2l_comms[direction][layer_id]
            required_links = set()
            max_xfer_size = 0
            sources = set()
            nodes = set()
            for (src, dst), xfer_size in comms_in_layer:
                if src == dst:
                    continue
                nodes.add(src)
                nodes.add(dst)
                sources.add(src)
                link = (src, dst)
                if link not in required_links:
                    required_links.add(link)
                if xfer_size > max_xfer_size:
                    max_xfer_size = xfer_size
            all_to_all = is_all_to_all(required_links)
            comm_cost = -1
            parsed_routes = None
            if not all_to_all:
                # use_lightpath_encoding(required_links)
                num_rounds, parsed_routes = get_random_pattern_rounds(required_links, 4, G) 
                alpha_cost = num_rounds * (alpha + reconfiguration_delay)
                beta_cost = max_xfer_size / (all_link_speed / 8)
                beta_cost *= 1000 * 1000
                beta_cost *= num_rounds
                comm_cost = alpha_cost + beta_cost
                print('[LOG]: PATTERN: buffer size {}, alpha: {}, beta: {}, rounds: {}'.format(max_xfer_size, alpha_cost, beta_cost, num_rounds))
            else:
                comm_cost = get_index_all_to_all_time(len(nodes), max_xfer_size)
            if comm_cost > 0 :
                print('[LOG]:Layer: {}, direction {}, comm cost: {} us'.format(layer_id, direction, comm_cost))
                
            if not all_to_all and comm_cost > 0:
                layer_to_layer_mcf_routes['{}_direction_{}'.format(layer_id, direction)] = parsed_routes
            for (src, dst), xfer_size in comms_in_layer:
                comm_str = 'L2L;{};{};{};{};{}|{}\n'.format(xfer_size, layer_id, direction, src, dst, comm_cost / 1000)
                communication_strings.append(comm_str)
    
    return communication_strings, layer_to_layer_mcf_routes
            
def get_optimal_time(num_nodes, buffer_size):
    if num_nodes == 1:
        return 0
    time_val = 0
    t_nodes = num_nodes / 1
    num_rounds = 2 * math.log(t_nodes, 2)
    alpha_cost = (num_rounds) * (alpha + reconfiguration_delay)
    beta_cost = 2 * buffer_size * (t_nodes - 1) / t_nodes / all_link_speed 
    beta_cost *= 1000 * 1000
    time_val = alpha_cost + beta_cost
    print('[LOG]:LIGHTPATH_2: buffer size {}, alpha: {}, beta: {}'.format(buffer_size, alpha_cost, beta_cost))
    return time_val

def get_binary_tree_time(num_nodes, buffer_size):
    num_rounds = 2 * math.log(num_nodes, 2)
    alpha_cost = (num_rounds) * (alpha + reconfiguration_delay)
    beta_cost = 2 * buffer_size * 2 / all_link_speed
    beta_cost *= 1000 * 1000
    time_val = alpha_cost + beta_cost
    print('[LOG]:TREE: buffer size {}, alpha: {}, beta: {}'.format(buffer_size, alpha_cost, beta_cost))
    return time_val

def get_index_all_to_all_time(num_nodes, buffer_size):
    if num_nodes == 0:
        return 0
    num_rounds = math.log(num_nodes, 2)
    alpha_cost = (num_rounds) * (alpha + reconfiguration_delay)
    beta_cost = num_rounds * buffer_size / 2 / (all_link_speed / 2)
    beta_cost *= 1000 * 1000
    time_val = alpha_cost + beta_cost
    print('[LOG]:A2A: buffer size {}, alpha: {}, beta: {}'.format(buffer_size, alpha_cost, beta_cost))
    return time_val

def add_l2l_demands(communication_strings, l2l_comms):
    for direction in l2l_comms:
        for layer_id in l2l_comms[direction]:
            layer_transfers = l2l_comms[direction][layer_id]
            for pair_data in layer_transfers:
                src = pair_data[0]
                dst = pair_data[1]
                size = pair_data[2]
                # if (src != dst):
                #     log_info('ERROR: {}, {}'.format(src, dst))
                comm_cost = 0
                if src != dst:
                    alpha_cost = alpha + reconfiguration_delay
                    beta_cost = size / all_link_speed
                    comm_cost = alpha_cost + beta_cost
                comm_str = 'L2L;{};{};{};{};{}|{}\n'.format(size, layer_id, direction, src, dst, comm_cost / 1000)
                communication_strings.append(comm_str)


def generate_communication_file_for_simulation(communication_strings, file_name):
    with open(file_name, 'w') as log_file:
        for string in communication_strings:
            log_file.write(string)

def parse_quad_algorithm(log_file_path, parsed_file_path, G: nx.Graph):
    l2l_comms, ar_comms = get_comm_demands(log_file_path)
    parsed_file_path = parsed_file_path.format('quad')

    communication_strings = []
    for ar_key in ar_comms:
        gpus, xfer_sizes = ar_comms[ar_key]
        # gpus, xfer_sizes = [g for g in range(rows * cols)], [4096]
        for xfer_size in xfer_sizes:
            print(ar_key, xfer_size)
            transfers = quad_all_reduce(gpus, xfer_size)
            get_quad_recursive_time(len(gpus), xfer_size)
            comm_cost = simulate_all_reduce(gpus, transfers, xfer_size)
            for step_transfers in transfers:
                circuits = [(t.src, t.dst) for t in step_transfers]
                routes_formulator = formulate_routes(1, G)
                all_routes = routes_formulator.get_routes_for(circuits, G)
                get_maximum_overlap(all_routes)
            comm_string = 'AR;{};{}|{}\n'.format(xfer_size, ar_key, comm_cost / 1000)
            communication_strings.append(comm_string)
    add_l2l_demands(communication_strings, l2l_comms)
    generate_communication_file_for_simulation(communication_strings, parsed_file_path)

def parse_ring_algorithm(log_file_path, parsed_file_path, G: nx.Graph):
    l2l_comms, ar_comms = get_comm_demands(log_file_path)
    parsed_file_path = parsed_file_path.format('ring')

    communication_strings = []
    for ar_key in ar_comms:
        gpus, xfer_sizes = ar_comms[ar_key]
        # gpus, xfer_sizes = [g for g in range(rows * cols)], [4096]
        for xfer_size in xfer_sizes:
            print(ar_key, xfer_size)
            comm_cost = get_ring_time(len(gpus), xfer_size)
            comm_string = 'AR;{};{}|{}\n'.format(xfer_size, ar_key, comm_cost / 1000)
            communication_strings.append(comm_string)
    add_l2l_demands(communication_strings, l2l_comms)
    generate_communication_file_for_simulation(communication_strings, parsed_file_path)

def parse_bruck_algorithm(log_file_path, parsed_file_path, G: nx.Graph):
    l2l_comms, ar_comms = get_comm_demands(log_file_path)
    parsed_file_path = parsed_file_path.format('bruck')

    communication_strings = []
    for ar_key in ar_comms:
        gpus, xfer_sizes = ar_comms[ar_key]
        # gpus, xfer_sizes = [g for g in range(rows * cols)], [4096]
        for xfer_size in xfer_sizes:
            print(ar_key, xfer_size)
            comm_cost = get_optimal_time(len(gpus), xfer_size)
            comm_string = 'AR;{};{}|{}\n'.format(xfer_size, ar_key, comm_cost / 1000)
            communication_strings.append(comm_string)
    add_l2l_demands(communication_strings, l2l_comms)
    generate_communication_file_for_simulation(communication_strings, parsed_file_path)

def parse_tree_algorithm(log_file_path, parsed_file_path, G: nx.Graph):
    l2l_comms, ar_comms = get_comm_demands(log_file_path)
    parsed_file_path = parsed_file_path.format('tree')

    communication_strings = []
    for ar_key in ar_comms:
        gpus, xfer_sizes = ar_comms[ar_key]
        # gpus, xfer_sizes = [g for g in range(rows * cols)], [4096]
        for xfer_size in xfer_sizes:
            print(ar_key, xfer_size)
            comm_cost = get_binary_tree_time(len(gpus), xfer_size)
            comm_string = 'AR;{};{}|{}\n'.format(xfer_size, ar_key, comm_cost / 1000)
            communication_strings.append(comm_string)
    add_l2l_demands(communication_strings, l2l_comms)
    generate_communication_file_for_simulation(communication_strings, parsed_file_path)

def get_zeroed_ar_times(ar_comms):
    comm_strings = []
    for ar_key in ar_comms:
        gpus, xfer_sizes = ar_comms[ar_key]
        # gpus, xfer_sizes = [g for g in range(rows * cols)], [4096]
        for xfer_size in xfer_sizes:
            comm_string = 'AR;{};{}|{}\n'.format(xfer_size, ar_key, 0)
            comm_strings.append(comm_string)
    return comm_strings

def get_zero_l2l_times(l2l_comms, G):
    comm_strings = []
    for direction in l2l_comms:
        for layer_id in l2l_comms[direction]:
            layer_transfers = l2l_comms[direction][layer_id]
            for pair_data in layer_transfers:
                src, dst = pair_data[0]
                size = pair_data[1]
                # if (src != dst):
                #     log_info('ERROR: {}, {}'.format(src, dst))
                comm_cost = 0
                comm_str = 'L2L;{};{};{};{};{}|{}\n'.format(size, layer_id, direction, src, dst, comm_cost)
                comm_strings.append(comm_str)
    return comm_strings, 0

def parse_cover_set(cover_set, G: nx.Graph, algorithm=get_quad_recursive_time):
    num_non_overlapping_stages = len(cover_set)
    total_time = 0 # should be in milliseconds

    for i in range(num_non_overlapping_stages):
        stage_time = 0
        parallel_transfers = cover_set[i]
        for transfer in parallel_transfers:
            gpus, xfer_size = transfer
            runtime = algorithm(len(gpus), xfer_size)
            if runtime > stage_time:
                stage_time = runtime
        total_time += stage_time

    return total_time / 1000 