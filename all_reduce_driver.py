from collectives.routing import generate_graph
from formulate_encoding import formulate_lightpath, formulate_routes
from route_solver import traditional_routes
from collectives import ar
from typing import Dict, List
import matplotlib.pyplot as plt
import networkx as nx
import argparse

parser = argparse.ArgumentParser(description='Parse arguments for all-reduce DC.')
parser.add_argument('-n', '--num_chunks', type=int, default=1, help='Number of chunks to split the data into.', required=True)
parser.add_argument('-b', '--use_bi', action='store_true', help='When set, the encoding ignores symmetry and schedules all the circuits')
parser.add_argument('-l', '--lambdas', type=int, default=4, help='Number of lasers', required=True)
args = parser.parse_args()

rows = 2
cols = 2
max_nodes = 128
lambdas = args.lambdas
num_chunks = args.num_chunks
should_encode_bidirectionality = args.use_bi
print('[NUMBER OF CHUNKS]: {}'.format(num_chunks))

def run_lightpath_solver(min_rounds, max_rounds, routes, pre_condition: Dict[int, List[int]], post_condition: Dict[int, List[int]]):
    lightpath_formulator = formulate_lightpath(min_rounds=min_rounds, max_rounds=max_rounds, lambdas=lambdas, routes=routes)
    # set pre condition
    lightpath_formulator.set_precondition(pre_condition)
    # set post condition
    lightpath_formulator.set_postcondition(post_condition)
    # create variables
    lightpath_formulator.create_variables()
    # add constraints
    lightpath_formulator.add_constraints()
    lightpath_formulator.run_formulation()

    schedule = lightpath_formulator.get_schedule()
    for transfer in schedule:
        print(transfer)

def log_stage(grid, sources, sinks, G):
    print('[GENERATING SCHEDULE FOR GRID]: {}'.format(grid))
    print('Sources: {}, sinks: {}, grid: {}'.format(sources, sinks, grid))
    
    fig, ax = plt.subplots(figsize=(8, 10)) # create a figure and an axes
    pos = nx.get_node_attributes(G, 'pos')
    color_map = {}
    for source in sources:
        color_map[source] = 'teal'
    for sink in sinks:
        color_map[sink] = 'orange'
    colors = [color_map[node] for node in G.nodes]
    nx.draw(G, pos=pos, node_color=colors, with_labels=True, ax=ax) # draw the graph in the axes
    plt.title('Conquer: 2 {}-node grids'.format(len(sources)))
    plt.tight_layout()
    plt.savefig('grid_{}.png'.format(grid))


def get_routes(G):
    # traditional_routes_generator = traditional_routes()
    # all_routes = traditional_routes_generator.generate_all_routes(G, G.nodes(), lambdas)

    routes_formulator = formulate_routes(lambdas, G)
    all_routes = routes_formulator.get_routes(G.nodes(), G.nodes(), G)

    return all_routes

G = generate_graph(rows, cols)
conditions = ar.get_pattern(rows * cols)
pre_condition: Dict[int, List[int]] = conditions['pre']
post_condition: Dict[int, List[int]] = conditions['post']
all_routes = get_routes(G)
run_lightpath_solver(1, 10, all_routes, pre_condition, post_condition)

cols *= 2
while rows * cols <= max_nodes:
    sources = []
    sinks = []
    grid = '{}x{}'.format(rows, cols)
    G = generate_graph(rows, cols)

    all_nodes = [n for n in range(rows * cols)]

    if rows == cols:
        partition_point = int(len(all_nodes) / 2)
        for n in all_nodes[: partition_point]:
            sources.append(n)
        
        for n in all_nodes[partition_point: ]:
            sinks.append(n)

        cols *= 2

    elif rows < cols:
        partition_point = int(cols / 2)
        
        for r in range(rows):
            for n in range(partition_point):
                sources.append(r * cols + n)

        for r in range(rows):
            for n in range(partition_point, cols):
                sinks.append(r * cols + n)
        
        rows *= 2
    
    pre_condition = {}
    post_condition = {}
    for chunk in range(num_chunks):
        pre_condition[chunk] = sources
        post_condition[chunk] = sources + sinks

    if should_encode_bidirectionality:
        print('Encoding bidirectionality')
        for chunk in range(num_chunks):
            chunk_id = num_chunks + chunk
            pre_condition[chunk_id] = sinks
            post_condition[chunk_id] = sinks + sources
    
    log_stage(grid, sources, sinks, G)

    all_routes = get_routes(G)
    run_lightpath_solver(1, 5, all_routes, pre_condition, post_condition)
