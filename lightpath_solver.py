import gurobipy as gp
from gurobipy import GRB, Var
from typing import List, Dict

class buffer_states:
    def __init__(self) -> None:
        self.state = {}
    
    def add_chunk_state(self, chunk, nodes):
        """Marks nodes as having the chunk"""
        if chunk not in self.state:
            self.state[chunk] = []
        self.state[chunk].extend(nodes)
    
    def get_nodes(self, chunk):
        """returns the nodes having the chunk"""
        return self.state[chunk]
    
    def get_chunks(self):
        """returns the list of chunks in this state"""
        return list(self.state.keys())

class path:
    def __init__(self, w: int, nodes: List[int]) -> None:
        self.wavelength = w
        self.nodes = nodes
    
    def to_string(self):
        node_str = ','.join([str(node) for node in self.nodes])
        return 'w={}nodes={}'.format(self.wavelength, node_str)

class scheduled_circuit:
    def __init__(self, chunk: int, round_id: int, wavelength: int, path: List[int]) -> None:
        self.chunk = chunk
        self.round_id = round_id
        self.wavelength = wavelength
        self.path = path

class circuit:

    def __init__(self, src: int, dst: int, route: path, chunk: int, round_id: int) -> None:
        self.source = src
        self.destination = dst
        self.route = route
        self.chunk = chunk
        self.round = round_id
        self.name = 'r_{}_circuit_{}_{}_{}_{}'.format(round_id, src, dst, chunk, route.to_string())
        self.model_var: Var = None

    def set_model_variable(self, circuit_var: Var):
        self.model_var = circuit_var
    
    def schedule_info(self) -> scheduled_circuit:
        human_readable = "Chunk {} was sent from {} to {} in round {} along path: {}".format(self.chunk, self.source, self.destination, self.round, [self.route.wavelength, self.route.nodes])
        colon_separated = "{};{};{};{}".format(self.chunk, self.destination, self.round, [self.route.wavelength, self.route.nodes])
        return scheduled_circuit(self.chunk, self.round, self.route.wavelength, self.route.nodes)

class rounds_store:
    def __init__(self) -> None:
        self.round_indicators: List[Var] = []
    
    def add_round(self, round_var: Var):
        self.round_indicators.append(round_var)
    
    def get_num_rounds(self) -> int:
        return len(self.round_indicators)
    
    def get_round_var(self, r: int) -> Var:
        return self.round_indicators[r]

class circuits_store:
    def __init__(self) -> None:
        self.round_to_circuits: Dict[int, List[circuit]] = {}
        self.destination_to_circuits: Dict[int, List[circuit]] = {}
        self.chunk_to_circuits: Dict[int, List[circuit]] = {}
        self.source_to_circuits: Dict[int, List[circuit]] = {}

    def store_circuit(self, circuit_obj: circuit):
        destination = circuit_obj.destination
        chunk = circuit_obj.chunk
        round_id = circuit_obj.round
        source = circuit_obj.source

        if round_id not in self.round_to_circuits:
            self.round_to_circuits[round_id] = []
        if chunk not in self.chunk_to_circuits:
            self.chunk_to_circuits[chunk] = []
        if destination not in self.destination_to_circuits:
            self.destination_to_circuits[destination] = []
        if source not in self.source_to_circuits:
            self.source_to_circuits[source] = []
        
        self.round_to_circuits[round_id].append(circuit_obj)
        self.chunk_to_circuits[chunk].append(circuit_obj)
        self.destination_to_circuits[destination].append(circuit_obj)
        self.source_to_circuits[source].append(circuit_obj)
    
    def get_all_rounds(self) -> List[int]:
        return list(self.round_to_circuits.keys())
    
    def get_all_destinations(self) -> List[int]:
        return list(self.destination_to_circuits.keys())
    
    def get_all_chunks(self) -> List[int]:
        return list(self.chunk_to_circuits.keys())
    
    def get_all_sources(self) -> List[int]:
        return list(self.source_to_circuits.keys())

    def get_circuits_in_round(self, r: int) -> List[circuit]:
        return self.round_to_circuits[r]
    
    def get_circuits_in_destination(self, d: int) -> List[circuit]:
        return self.destination_to_circuits[d]

class lightpath_solver:
    
    def __init__(self, min_rounds: int, max_rounds: int, name='lightpath'):
        self.model = gp.Model(name=name)
        self.min_rounds = min_rounds
        self.max_rounds = max_rounds
        self.circuits_store = circuits_store()
        self.rounds_store = rounds_store()
        self.pre_conditions: buffer_states = None
    
    def create_round_indicators(self):
        """Creates indicator variables to indicate whether a round is used"""
        for r in range(self.max_rounds):
            round_variable = self.model.addVar(vtype=GRB.BINARY, name='round_{}'.format(r))
            self.rounds_store.add_round(round_variable)
    
    def enable_preconditions(self, pre_condition: buffer_states):
        """Sets circuit variables so that the chunks are present in the nodes specified by the preconditions"""
        self.pre_conditions = pre_condition
    
    def add_circuit(self, source: int, destination: int, route: path, chunk: int):
        """Adds the circuit to the store and creates a variable for it"""
        for r in range(self.max_rounds):
            
            circuit_obj = circuit(source, destination, route, chunk, r)
            circuit_variable = self.model.addVar(vtype=GRB.BINARY, name='{}'.format(circuit_obj.name))
            circuit_obj.set_model_variable(circuit_variable)

            self.circuits_store.store_circuit(circuit_obj)


    def add_unique_circuit_per_round_constraints(self):
        """In a round, at most one circuit can be used to send a given chunk to a destination"""
        rounds = self.circuits_store.get_all_rounds()

        for r in rounds:
            circuits = self.circuits_store.get_circuits_in_round(r)
            dst_chunk_to_circuits: Dict[int, Dict[int, List[Var]]] = {}
            for circuit_obj in circuits:
                d = circuit_obj.destination
                c = circuit_obj.chunk
                if d not in dst_chunk_to_circuits:
                    dst_chunk_to_circuits[d] = {}
                if c not in dst_chunk_to_circuits[d]:
                    dst_chunk_to_circuits[d][c] = []
                dst_chunk_to_circuits[d][c].append(circuit_obj.model_var)
            
            for dst in dst_chunk_to_circuits:
                for chunk in dst_chunk_to_circuits[dst]:
                    circuits = dst_chunk_to_circuits[dst][chunk]
                    self.model.addConstr(gp.quicksum(circuits) <= 1)

    def add_chunk_availability_constraints(self):
        """A circuit can be chosen only if the source has received the chunk in a previous round"""
        rounds = self.circuits_store.get_all_rounds()
        round_dst_chunk_to_circuits: Dict[int, Dict[int, Dict[int, List[circuit]]]] = {}

        for r in rounds:
            circuits = self.circuits_store.get_circuits_in_round(r)
            for circuit_obj in circuits:
                dst = circuit_obj.destination
                chunk = circuit_obj.chunk

                if r not in round_dst_chunk_to_circuits:
                    round_dst_chunk_to_circuits[r] = {}
                if dst not in round_dst_chunk_to_circuits[r]:
                    round_dst_chunk_to_circuits[r][dst] = {}
                if chunk not in round_dst_chunk_to_circuits[r][dst]:
                    round_dst_chunk_to_circuits[r][dst][chunk] = []
                
                round_dst_chunk_to_circuits[r][dst][chunk].append(circuit_obj)
        
        for r in rounds:
            circuits = self.circuits_store.get_circuits_in_round(r)
            for circuit_obj in circuits:
                src = circuit_obj.source
                chunk = circuit_obj.chunk
                previous_circuit_objs: List[Var] = []
                if r > 0:
                    for prev_r in range(r):
                        if prev_r in round_dst_chunk_to_circuits:
                            if src in round_dst_chunk_to_circuits[prev_r]:
                                if chunk in round_dst_chunk_to_circuits[prev_r][src]:
                                    circuit_objs = round_dst_chunk_to_circuits[prev_r][src][chunk]
                                    previous_circuit_objs.extend([c.model_var for c in circuit_objs])
                if True:
                    # use precondition
                    pre_chunks = self.pre_conditions.get_chunks()
                    if chunk in pre_chunks:
                        nodes = self.pre_conditions.get_nodes(chunk)
                        if src in nodes:
                            previous_circuit_objs.append(1)
                
                self.model.addConstr((circuit_obj.model_var == 1) >> (gp.quicksum(previous_circuit_objs) >= 1))
                # self.model.addConstr((circuit_obj.model_var == 1) >> (gp.quicksum(previous_circuit_objs) <= 1))


    def add_unique_circuit_per_destination_constraints(self):
        """A chunk is transferred to a destination only once across all rounds"""
        destinations = self.circuits_store.get_all_destinations()
        
        for d in destinations:
            circuits = self.circuits_store.get_circuits_in_destination(d)
            chunk_to_circuits: Dict[int, List[Var]] = {}
            for circuit_obj in circuits:
                chunk = circuit_obj.chunk
                if chunk not in chunk_to_circuits:
                    chunk_to_circuits[chunk] = []
                chunk_to_circuits[chunk].append(circuit_obj.model_var)
            
            for chunk in chunk_to_circuits:
                circuits_list = chunk_to_circuits[chunk]
                self.model.addConstr(gp.quicksum(circuits_list) >= 1)
                self.model.addConstr(gp.quicksum(circuits_list) <= 1)
    
    def add_rounds_used_constraints(self):
        """A round is useful if at least one chunk was transferred in that round"""
        rounds = self.circuits_store.get_all_rounds()
        for r in rounds:
            round_var = self.rounds_store.get_round_var(r)
            circuits = self.circuits_store.get_circuits_in_round(r)
            circuit_vars = [c_obj.model_var for c_obj in circuits]
            self.model.addConstr(round_var == gp.or_(circuit_vars))
            # for circuit_obj in circuits:
            #     self.model.addConstr((circuit_obj.model_var == 1) >> (round_var >= 1))

    def add_minimum_rounds_used_constraint(self):
        """Marks rounds as used. Also sets corresponding circuit used constraints since at least one circuit needs to be enabled in a round of the round to be marked as used"""
        for r in range(self.min_rounds):
            round_var = self.rounds_store.get_round_var(r)
            self.model.addConstr(round_var >= 1)
    
    def add_edge_wavelength_constraints(self):
        """A wavelength is used at most once on an edge in a round"""
        rounds = self.circuits_store.get_all_rounds()
        print("All rounds: {}".format(rounds))
        for r in rounds:
            edge_lambda_to_circuits: Dict[str, Dict[int, List[Var]]] = {}
            circuits = self.circuits_store.get_circuits_in_round(r)
            for circuit_obj in circuits:
                route = circuit_obj.route
                lambda_value = route.wavelength
                nodes = route.nodes
                for edge in zip(nodes, nodes[1:]):
                    edge = sorted(edge)
                    edge_id = '{}->{}'.format(edge[0], edge[1])
                    if edge_id not in edge_lambda_to_circuits:
                        edge_lambda_to_circuits[edge_id] = {}
                    if lambda_value not in edge_lambda_to_circuits[edge_id]:
                        edge_lambda_to_circuits[edge_id][lambda_value] = []
                    edge_lambda_to_circuits[edge_id][lambda_value].append(circuit_obj.model_var)

            for edge_id in edge_lambda_to_circuits:
                for wavelength in edge_lambda_to_circuits[edge_id]:
                    circuit_vars = edge_lambda_to_circuits[edge_id][wavelength]
                    self.model.addConstr(gp.quicksum(circuit_vars) <= 20)

    def _node_constraint_helpers(self, node_idx):
        rounds = self.circuits_store.get_all_rounds()

        for r in rounds:
            node_laser_to_circuits: Dict[int, Dict[int, List[Var]]] = {}
            circuits = self.circuits_store.get_circuits_in_round(r)
            for circuit_obj in circuits:
                route = circuit_obj.route
                wavelength = route.wavelength
                nodes = route.nodes
                start_node = nodes[node_idx]
                if start_node not in node_laser_to_circuits:
                    node_laser_to_circuits[start_node] = {}
                if wavelength not in node_laser_to_circuits[start_node]:
                    node_laser_to_circuits[start_node][wavelength] = []
                node_laser_to_circuits[start_node][wavelength].append(circuit_obj.model_var)
            
            for node in node_laser_to_circuits:
                for laser in node_laser_to_circuits[node]:
                    circuit_vars = node_laser_to_circuits[node][laser]
                    self.model.addConstr(gp.quicksum(circuit_vars) <= 1)

    def add_laser_constraints(self):
        """A given laser is used at most once on a node in a round"""
        self._node_constraint_helpers(0)

    def add_photodiode_constraints(self):
        """A given photo-diode is used at most once on a node in a round"""
        self._node_constraint_helpers(-1)

    def add_round_dependency_constraints(self):
        """If round i is used, then round i-1 must be used"""
        round_ids = [self.rounds_store.get_round_var(r) for r in range(self.rounds_store.get_num_rounds())]
        for r1, r2 in zip(round_ids[1: ], round_ids[2: ]):
            self.model.addConstr((r2 == 1) >> (r1 >= 1))
    
    def run(self):
        maximum_round_value = self.model.addVar(vtype=GRB.INTEGER, name='opt')
        for r in range(self.rounds_store.get_num_rounds()):
            round_var = self.rounds_store.get_round_var(r)
            self.model.addConstr(maximum_round_value >= r * round_var)
        
        self.model.setObjective(maximum_round_value, GRB.MINIMIZE)
        self.model.optimize()

    def get_schedule(self) -> List[scheduled_circuit]:
        rounds = self.circuits_store.get_all_rounds()
        schedule: List[scheduled_circuit] = []
        for r in rounds:
            circuits = self.circuits_store.get_circuits_in_round(r)
            for circuit_obj in circuits:
                if round(circuit_obj.model_var.X) >= 1:
                    schedule.append(circuit_obj.schedule_info())
        return schedule
