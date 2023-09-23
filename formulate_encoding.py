from lightpath_solver import lightpath_solver, path, buffer_states, scheduled_circuit
from route_solver import route_solver, demand, routes_generator, wave_route
from typing import Dict, List
import networkx as nx

class formulate_lightpath:
    def __init__(self, min_rounds, max_rounds, lambdas, routes) -> None:
        self.solver = lightpath_solver(min_rounds, max_rounds)
        self.pre_condition: Dict[int, List[int]] = None
        self.lambdas = lambdas
        self.routes: Dict[int, Dict[int, List[wave_route]]] = routes
    
    def set_precondition(self, pre_condition: Dict[int, List[int]]):
        print('Filling pre condition buffers')
        pre_condition_buffer = buffer_states()
        for chunk in pre_condition:
            pre_condition_buffer.add_chunk_state(chunk, pre_condition[chunk])
        self.solver.enable_preconditions(pre_condition_buffer)
        self.pre_condition = pre_condition

    def set_postcondition(self, post_condition: Dict[int, List[int]]):
        print('Adding circuits for post conditions')
        for chunk in post_condition:
            for s in post_condition[chunk]:
                for d in post_condition[chunk]:
                    if d not in self.pre_condition[chunk] and d != s:
                        routes_between_s_d = self.routes[s][d]
                        assert len(routes_between_s_d) >= self.lambdas
                        for route in routes_between_s_d:
                            # w, nodes = route
                            path_obj = path(route.w, route.nodes)
                            # print('Maybe: {}->{} C {} over {}'.format(s, d, chunk, [route.w, route.nodes]))
                            self.solver.add_circuit(s, d, path_obj, chunk)
    
    def create_variables(self):
        print('Creating round variables')
        self.solver.create_round_indicators()
    
    def add_constraints(self):
        print('Adding chunk availability constraints')
        self.solver.add_chunk_availability_constraints()

        # print('Adding unique circuit per round constraint')
        # self.solver.add_unique_circuit_per_round_constraints()

        print('Adding unique constraint per destination constraint')
        self.solver.add_unique_circuit_per_destination_constraints()

        print('Adding rounds used constraints')
        self.solver.add_rounds_used_constraints()

        print('Minimum rounds used cosntraints')
        self.solver.add_minimum_rounds_used_constraint()

        print('Adding wavelength constraints')
        self.solver.add_edge_wavelength_constraints()

        print('Adding laser constraints')
        self.solver.add_laser_constraints()

        print('Adding photo diode constraints')
        self.solver.add_photodiode_constraints()

        print('Adding round dependency constraints')
        self.solver.add_round_dependency_constraints()

    def run_formulation(self):
        self.solver.run()
    
    def get_schedule(self) -> List[scheduled_circuit]:
        return self.solver.get_schedule()
    
class formulate_routes:
    def __init__(self, lambdas, G: nx.Graph) -> None:
        self.route_solver_model = route_solver()
        self.routes_generator_obj = routes_generator(lambdas)

        for e in G.edges():
            u, v = e
            self.route_solver_model.add_edge(u, v)
    
    def _solve_for_routes(self, G: nx.Graph):
        self.route_solver_model.add_overlapping_count_variables()
        self.route_solver_model.solve()
        routes_without_wavelengths = self.route_solver_model.get_routes(G)
        routes = self.routes_generator_obj.generate_routes(routes_without_wavelengths)
        return routes

    def get_routes(self, sources, sinks, G: nx.Graph) -> Dict[int, Dict[int, List[wave_route]]]:
        for u in sources:
            for v in sinks:
                if u != v:
                    d = demand(u, v, 0)
                    self.route_solver_model.add_route_demand(d)

        return self._solve_for_routes(G)
    
    def get_routes_for(self, circuits, G: nx.Graph) -> Dict[int, Dict[int, List[wave_route]]]:
        for circuit in circuits:
            u, v = circuit
            d = demand(u, v, 0)
            self.route_solver_model.add_route_demand(d)
        
        return self._solve_for_routes(G)
        
