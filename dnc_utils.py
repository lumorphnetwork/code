from formulate_encoding import formulate_lightpath, formulate_routes
from route_solver import traditional_routes
from typing import Dict, List
import matplotlib.pyplot as plt
import networkx as nx

def run_lightpath_solver(min_rounds, max_rounds, routes, pre_condition: Dict[int, List[int]], post_condition: Dict[int, List[int]], lambdas):
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
    return schedule

def log_stage(sources, sinks, G, file_name):
    
    fig, ax = plt.subplots(figsize=(8, 10)) # create a figure and an axes
    pos = nx.get_node_attributes(G, 'pos')
    color_map = {}
    for source in sources:
        color_map[source] = 'teal'
    for sink in sinks:
        color_map[sink] = 'orange'
    colors = [color_map[node] if node in color_map else 'white' for node in G.nodes]
    nx.draw(G, pos=pos, node_color=colors, with_labels=True, ax=ax) # draw the graph in the axes
    plt.title('Conquer: 2 {}-node grids'.format(len(sources)))
    plt.tight_layout()
    plt.savefig('{}.png'.format(file_name))


def get_routes(G, nodes, lambdas):
    # traditional_routes_generator = traditional_routes()
    # all_routes = traditional_routes_generator.generate_all_routes(G, G.nodes(), lambdas)

    routes_formulator = formulate_routes(lambdas, G)
    all_routes = routes_formulator.get_routes(nodes, nodes, G)

    return all_routes

