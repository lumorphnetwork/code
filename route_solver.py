import gurobipy as gp
from gurobipy import GRB, Var
from typing import List, Dict, Tuple
from itertools import islice
import networkx as nx
import random

class demand:

    def __init__(self, src, dst, key) -> None:
        self.src = src
        self.dst = dst
        self.k = key
    
    def serialize(self) -> str:
        return "{}_to_{}_ordeal_{}".format(self.src, self.dst, self.k)
    
    def parse(data: str) -> 'demand':
        parts = data.split('_')
        parsed_demand = demand(int(parts[0]), int(parts[2]), int(parts[4]))
        return parsed_demand

class route_solver:
    
    def __init__(self) -> None:
        self.nodes_info: Dict[int, List[(int , int)]] = {}
        self.demands: Dict[(int, int), Dict[int, List[Var]]] = {}
        self.model = gp.Model(name='routes')
        self.edge_variables: Dict[(int, int), List[int]] = {}
        self.overlap_count_vars: List[Var] = []
        

    def add_edge(self, u: int, v: int) -> None:
        edge = (u, v)
        if v < u: 
            edge = (v, u)
        if u not in self.nodes_info:
            self.nodes_info[u] = []
        if v not in self.nodes_info:
            self.nodes_info[v] = []
        
        self.nodes_info[u].append(edge)
        self.nodes_info[v].append(edge)
    
    def add_route_demand(self, d: demand) -> None:
        # total_flow for every node
        # edge_variables[(u, v)] = []
        # outgoing edges for every node
        # incoming edges for every node
        # add constraint based on outgoing and incoming
        # minimize edge_variables 
        local_edge_vars: Dict[(int, int), Var] = {}
        self.demands[d.serialize()] = {}
    
        for node in self.nodes_info:
            outgoing_vars = []
            incoming_vars = []

            for edge in self.nodes_info[node]:
                u, v = edge
                
                # might be reverse
                if v == node:
                    u, v = edge[1], edge[0]
                
                outgoing_edge = (u, v)
                incoming_edge = (v, u)
                
                # if edge not in self.edge_variables:
                #     self.edge_variables[edge] = []
                
                if outgoing_edge not in self.edge_variables:
                    self.edge_variables[outgoing_edge] = []
                
                if incoming_edge not in self.edge_variables:
                    self.edge_variables[incoming_edge] = []

                if outgoing_edge not in local_edge_vars:
                    outgoing_edge_var = self.model.addVar(vtype=GRB.BINARY, name='{}_{}'.format(outgoing_edge[0], outgoing_edge[1]))
                    local_edge_vars[outgoing_edge] = outgoing_edge_var
                    # self.edge_variables[edge].append(outgoing_edge_var)
                    self.edge_variables[outgoing_edge].append(outgoing_edge_var)
                
                if incoming_edge not in local_edge_vars:
                    incoming_edge_var = self.model.addVar(vtype=GRB.BINARY, name='{}_{}'.format(incoming_edge[0], incoming_edge[1]))
                    local_edge_vars[incoming_edge] = incoming_edge_var
                    # self.edge_variables[edge].append(incoming_edge_var)
                    self.edge_variables[incoming_edge].append(incoming_edge_var)
                
                incoming_edge_var = local_edge_vars[incoming_edge]
                outgoing_edge_var = local_edge_vars[outgoing_edge]
                
                incoming_vars.append(incoming_edge_var)
                outgoing_vars.append(outgoing_edge_var)
            
            if node != d.src and node != d.dst:
                self.model.addConstr(gp.quicksum(incoming_vars) <= 1)
                self.model.addConstr(gp.quicksum(outgoing_vars) <= 1)
                self.model.addConstr(gp.quicksum(incoming_vars) + gp.quicksum([-1 * out_var for out_var in outgoing_vars]) == 0)

            elif node == d.src:
                self.model.addConstr(gp.quicksum(outgoing_vars) >= 1)
                self.model.addConstr(gp.quicksum(outgoing_vars) <= 1)

                self.model.addConstr(gp.quicksum(incoming_vars) >= 0)
                self.model.addConstr(gp.quicksum(incoming_vars) <= 0)
            
            elif node == d.dst:
                self.model.addConstr(gp.quicksum(incoming_vars) >= 1)
                self.model.addConstr(gp.quicksum(incoming_vars) <= 1)

                self.model.addConstr(gp.quicksum(outgoing_vars) >= 0)
                self.model.addConstr(gp.quicksum(outgoing_vars) <= 0)
            
            self.demands[d.serialize()][node] = outgoing_vars


    def add_overlapping_count_variables(self):
        for edge in self.edge_variables:
            overlap_count = self.model.addVar(vtype=GRB.INTEGER, name='Count: {}'.format(edge))
            self.model.addConstr(overlap_count >= gp.quicksum(self.edge_variables[edge]) - 1)
            self.model.addConstr(overlap_count >= 0)
            self.overlap_count_vars.append(overlap_count)
        

    def solve(self) -> None:
        # self.model.setObjective(gp.quicksum(self.overlap_count_vars), GRB.MINIMIZE)
        max_over_lap_var = self.model.addVar(vtype=GRB.INTEGER, name='Mac overlap')
        for overlap_count_var in self.overlap_count_vars:
            self.model.addConstr(max_over_lap_var >= overlap_count_var)
        self.model.setObjective(max_over_lap_var, GRB.MINIMIZE)
        self.model.optimize()

    def get_routes(self, G: nx.Graph) -> Dict[str, List[int]]:
        routes: Dict[str, List[int]] = {}

        for demand_str in self.demands:
            d = demand.parse(demand_str)
            node_outgoing_vars = self.demands[demand_str]
            current_node = d.src
            routes[demand_str] = []
            routes[demand_str].append(current_node)
            while current_node != d.dst:
                outgoing_vars = node_outgoing_vars[current_node]
                for outgoing_variable in outgoing_vars:
                    if round(outgoing_variable.X) == 1:
                        parts = outgoing_variable.VarName.split('_')
                        current_node = int(parts[1])
                        routes[demand_str].append(current_node)
                        continue
            assert nx.is_path(G, routes[demand_str])
        return routes

class wave_route:
    def __init__(self, w: int, nodes: List[int]) -> None:
        self.w = w
        self.nodes = nodes

class routes_generator:

    def __init__(self, num_wavelengths: int) -> None:
        self.w = num_wavelengths
    
    def generate_routes(self, routes_without_wavelengths: Dict[str, List[int]]) -> Dict[int, Dict[int, List[wave_route]]]:
        routes_with_wavelengths: Dict[int, Dict[int, List[wave_route]]] = {}

        for demand_str in routes_without_wavelengths:
            d = demand.parse(demand_str)
            nodes = routes_without_wavelengths[demand_str]

            if d.src not in routes_with_wavelengths:
                routes_with_wavelengths[d.src] = {}
            if d.dst not in routes_with_wavelengths[d.src]:
                routes_with_wavelengths[d.src][d.dst] = []
            
            for w in range(self.w):
                assert nodes[0] == d.src
                assert nodes[-1] == d.dst
                routes_with_wavelengths[d.src][d.dst].append(wave_route(w, nodes))
        
        return routes_with_wavelengths

class traditional_routes:

    def __init__(self) -> None:
        pass
    
    def k_shortest_paths(self, G, source, target, k, weight=None):
        return list(
            islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
        )

    def generate_all_routes(self, G, node_ids, wavelengths, unused_nodes = [], seed=1):
        routes = {}
        random.seed(seed)
        for source in node_ids:
            for destination in node_ids:
                if source in unused_nodes or destination in unused_nodes:
                    continue
                if source != destination:
                    paths = self.k_shortest_paths(G, source, destination, 32)
                    paths = random.sample(paths, 1)
                    # print('Num paths between: {} {} is {}'.format(source, destination, len(paths)))
                    for path in paths:
                        for w in range(wavelengths):
                            if source not in routes:
                                routes[source] = {}
                            if destination not in routes[source]:
                                routes[source][destination] = []
                            routes[source][destination].append(wave_route(w, path)) 
        return routes

    def get_routes_for(self, circuits, G: nx.Graph, lambdas, seed = 1) -> Dict[int, Dict[int, List[wave_route]]]: 
        routes = {}
        random.seed(seed)
        for circuit in circuits:
            u, v = circuit
            paths = self.k_shortest_paths(G, u, v, 32)
            paths = random.sample(paths, 1)
            for path in paths:
                for w in range(lambdas):
                    if u not in routes:
                        routes[u] = {}
                    if v not in routes[u]:
                        routes[u][v] = []
                    routes[u][v].append(wave_route(w, path))
        return routes