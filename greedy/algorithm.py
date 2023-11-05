#!/usr/bin/env python3
import itertools
import json
import sys
import os
from typing import List, Dict, Tuple, Any, Set, Optional
from collections import defaultdict, Counter
import traceback
from enum import Enum, unique
import networkx as nx
from pywasm.symbolic_execution import check_execution_from_ids

# General types we are using
var_T = str
id_T = str
instr_T = Dict[str, Any]
local_index_T = int


def idx_wrt_cstack(idx: int, cstack: List, fstack: List) -> int:
    """
    Given a position w.r.t fstack, returns the corresponding position w.r.t cstack
    """
    return idx - len(fstack) + len(cstack)


def idx_wrt_fstack(idx: int, cstack: List, fstack: List) -> int:
    """
    Given a position w.r.t cstack, returns the corresponding position w.r.t fstack
    """
    return idx - len(cstack) + len(fstack)


def top_relative_position_to_fstack(cstack: List[var_T], fstack: List[var_T]) -> int:
    return len(fstack) - len(cstack)


def extract_idx_from_id(instr_id: str) -> int:
    return int(instr_id.split('_')[-1])


def cheap(instr: instr_T) -> bool:
    """
    Cheap computations are those who take one instruction (i.e. inpt_sk is empty)
    """
    return len(instr['inpt_sk']) == 0


def remove_computation(id_: id_T, location: str, mops: List[id_T], sops: List[id_T], lops: Set[id_T]) -> None:
    if location == 'mops':
        mops.pop(0)
    elif location == 'sops':
        sops.pop(0)
    elif location == 'lops':
        lops.remove(id_)
    else:
        raise ValueError(f"Location {location} in _remove_computation is not valid")


@unique
class Location(Enum):
    stack = 0
    local = 1


class SymbolicState:
    """
    A symbolic state includes a stack, the register (locals) and a dict indicating the number of total uses of each
    instruction. With this dict, we can determine the liveness analysis
    """

    def __init__(self, stack: List[var_T], locals_: List[var_T], total_uses: Dict[id_T, int],
                 debug_mode: bool = True) -> None:
        self.stack: List[var_T] = stack
        self.locals: List[var_T] = locals_
        self.total_uses: Dict[id_T, int] = total_uses
        self.debug_mode: bool = debug_mode
        self.var_uses = self._computer_var_uses()
        self.liveness: List[bool] = self._compute_initial_liveness()

    def _computer_var_uses(self):
        var_uses = defaultdict(lambda: 0)

        # Count vars in the initial stack
        for var_stack in self.stack:
            var_uses[var_stack] += 1

        # Initial locals are not considered to count the stack vars because they are never in their positions

        return var_uses

    def _compute_initial_liveness(self) -> List[bool]:
        """
        Condition: a variable is live if it is used somewhere else, or it is in its correct position.
        As we are considering just the variables with some kind of change for locals, we just need to check if
        it's used somewhere else (i.e. it is used somewhere else)
        """
        return [self.var_uses[initial_var] < self.total_uses[initial_var] for initial_var in self.locals]

    def local_with_value(self, value: var_T) -> local_index_T:
        """
        Returns the first local in the list which contains the corresponding value. Otherwise, returns -1.
        """
        try:
            return self.locals.index(value)
        except:
            return -1

    def lset(self, x: int, in_position: bool) -> None:
        """
        Stores the top of the stack in the local with index x. in_position marks whether the element is
        solved in flocals
        """
        stack_var = self.stack.pop(0)
        self.locals[x] = stack_var

        # Var uses: if the local is solved, we are placing in its position and the var uses remains. Otherwise, as
        # we are storing it elsewhere, we are effectively removing one appearance
        self.var_uses[stack_var] += 0 if in_position else -1

        # Liveness: the variable is stored either in a position it must appear (hence, live) or to be used afterwards
        # Therefore, it is always live
        self.liveness[x] = True

    def ltee(self, x: int, in_position: bool) -> None:
        """
        Tee instruction in local with index x. in_position marks whether the element is solved in flocals
        """
        stack_var = self.stack[0]
        self.locals[x] = stack_var

        # Var uses: same reasoning as lget, but now we are both placing an element in its position and keeping it
        # computed.
        self.var_uses[stack_var] += 1 if in_position else 0

        # Liveness: the variable is stored either in a position it must appear (hence, live) or to be used afterwards
        # Therefore, it is always live
        self.liveness[x] = True

    def lget(self, x: int) -> var_T:
        """
        Get instruction in local x
        """
        stack_var = self.locals[x]
        self.stack.insert(0, stack_var)

        # Var uses: increased in one
        self.var_uses[stack_var] += 1

        # Liveness: the variable is still live if it needs to be used elsewhere
        self.liveness[x] = self.var_uses[stack_var] < self.total_uses[stack_var]
        return stack_var

    def drop(self):
        """
        Drops the last element
        """
        stack_var = self.stack.pop(0)
        # Var uses: we subtract one because the stack var is totally removed from the encoding
        self.var_uses[stack_var] -= 1
        # Liveness: not affected by dropping an element

    def uf(self, instr: instr_T) -> List[var_T]:
        """
        Symbolic execution of instruction instr. Additionally, checks the arguments match if debug mode flag is enabled
        """
        consumed_elements = [self.stack.pop(0) for _ in range(len(instr['inpt_sk']))]

        # Neither liveness nor var uses are affected by consuming elements, as these elements are just being embedded
        # into a new term

        # Debug mode to check the pop args from the stack match
        if self.debug_mode:
            if instr['commutative']:
                # Compare them as multisets
                assert Counter(consumed_elements) == Counter(instr['inpt_sk']), \
                    f"{instr['id']} is not consuming the correct elements from the stack"
            else:
                # Compare them as lists
                assert consumed_elements == instr['inpt_sk'], \
                    f"{instr['id']} is not consuming the correct elements from the stack"

        # We introduce the new elements
        for output_var in instr['outpt_sk']:
            self.stack.insert(0, output_var)

            # Var uses: increase one for each generated stack var
            self.var_uses[output_var] += 1

            # Liveness: not affected because we are not managing the stack

        return instr['outpt_sk']

    def top_stack(self) -> Optional[var_T]:
        return None if len(self.stack) == 0 else self.stack[0]

    def available_local(self) -> int:
        """
        Choose an available local to store an element. Otherwise, introduces an extra local to store it
        """
        # By reversing the liveness list, locals that have been introduced are prioritized over the ones that contain
        # valid values for the final stack
        # TODO: maybe this decision should also consider the final state somehow
        for x, is_live in enumerate(reversed(self.liveness)):
            if not is_live:
                return len(self.liveness) - x - 1

        # We just introduce an empty element and initialize the live variable to False
        self.locals.append('')
        self.liveness.append(False)
        return len(self.locals) - 1

    def __repr__(self):
        sentences = [f"Current stack: {self.stack}", "Current locals:",
                     *(f"{local}: {liveness}" for local, liveness in zip(self.locals, self.liveness))]
        return '\n'.join(sentences)


class SMSgreedy:

    def __init__(self, json_format, debug_mode: bool = False):
        self._bs: int = json_format['max_sk_sz']
        self._user_instr: List[instr_T] = json_format['user_instrs']
        self._b0: int = json_format["init_progr_len"]
        self._initial_stack: List[var_T] = json_format['src_ws']
        self._final_stack: List[var_T] = json_format['tgt_ws']
        self._vars: List[var_T] = json_format['vars']
        self._deps: List[Tuple[id_T, id_T]] = json_format['dependencies']
        self._max_registers: int = json_format['max_registers_sz']
        self._local_changes: List[Tuple[var_T, var_T]] = json_format['register_changes']
        self.debug_mode = debug_mode

        # We split into two different dicts the initial values and final values in locals
        self._initial_locals: List[var_T] = [local_repr[0] for local_repr in self._local_changes]
        self._final_locals: List[var_T] = [local_repr[1] for local_repr in self._local_changes]

        # Note: call instructions might have several variables in 'outpt_sk'
        self._var2instr = {var: ins for ins in self._user_instr for var in ins['outpt_sk']}
        self._id2instr = {ins['id']: ins for ins in self._user_instr}
        self._var2id = {var: ins['id'] for ins in self._user_instr for var in ins['outpt_sk']}
        self._var2pos_stack = self._compute_var2pos(self._final_stack)
        self._var2pos_locals = self._compute_var2pos(self._final_locals)

        self._var_total_uses = self._compute_var_total_uses()
        self._dep_graph = self._compute_dependency_graph()
        self._instr_dep_graph = self._dep_graph.edge_subgraph((u, v) for u, v, info in self._dep_graph.edges(data=True)
                                                              if info['weight'] > 0)
        # print(nx.find_cycle(self._dep_graph))

        # Nx interesting functions
        # nx.transitive_closure_dag()
        # nx.transitive_reduction()
        # nx.nx_pydot.write_dot()
        # nx.find_cycle() # for debugging
        with open('example.dot', 'w') as f:
            nx.nx_pydot.write_dot(self._dep_graph, f)

        # We include another list of local variables to indicate whether it is live or not
        # Condition: a variable is live if it is used somewhere else, or it is in its correct position.
        # As we are considering just the variables with some kind of change for locals, we just need to check if
        # it's used somewhere else (i.e. it is used somewhere else)

        self._mops, self._sops, self._lops, self._rops = self.split_ids_into_categories()

        # We need to compute the sub graph over the full dependency graph, as edges could be lost if we use the
        # transitive reduction instead. Hence, we need to compute the transitive_closure of the graph
        self._trans_sub_graph = nx.transitive_reduction(nx.transitive_closure_dag(self._dep_graph).subgraph(
            itertools.chain.from_iterable([self._mops, self._sops, self._lops])))

        with open('example2.dot', 'w') as f:
            nx.nx_pydot.write_dot(self._trans_sub_graph, f)

    def _compute_var_total_uses(self) -> Dict[var_T, int]:
        """
        Computes how many times each var appears either in the final stack, in the final locals or as a subterm
        of other terms.
        """
        var_uses = defaultdict(lambda: 0)

        # Count vars in the final stack
        for var_stack in self._final_stack:
            var_uses[var_stack] += 1

        # Count vars in the final locals
        for var_stack in self._final_locals:
            var_uses[var_stack] += 1

        # Count vars as input of other instrs
        for instr_id, instr in self._id2instr.items():
            for subterm_var in instr['inpt_sk']:
                var_uses[subterm_var] += 1

        return var_uses

    def _compute_var2pos(self, var_list: List[var_T]) -> Dict[var_T, List[int]]:
        var2pos = defaultdict(lambda: [])

        for i, stack_var in enumerate(var_list):
            var2pos[stack_var].append(i)

        return var2pos

    def _compute_dependency_graph(self) -> nx.DiGraph:
        # We annotate the edges with direct 0 or 1 to distinguish direct dependencies due to a subterm being embedded
        edge_list = []
        for instr in self._user_instr:
            instr_id = instr['id']

            for stack_elem in instr['inpt_sk']:
                # This means the stack element corresponds to another uninterpreted instruction
                if stack_elem in self._var2instr:
                    edge_list.append((self._var2id[stack_elem], instr_id, {'weight': 1}))
                # Otherwise, it corresponds to either a local or an initial element in the stack. We just add locals
                elif 'local' in stack_elem:
                    edge_list.append((stack_elem, instr_id, {'weight': 1}))

        # We need to consider also the order given by the tuples
        for id1, id2 in self._deps:
            edge_list.append((id1, id2, {'weight': 0}))

        # Also, the dependencies induced among locals that are live
        for ini_var, final_var in self._local_changes:
            # Either final var corresponds to a computation and appears in var2id or it is another local, which
            # we are referencing using the same name
            final_id = self._var2id.get(final_var, final_var)
            edge_list.append((ini_var, final_id, {'weight': 0}))
        return nx.DiGraph(edge_list)

    def _choose_local_to_store(self, var_elem: var_T, clocals_liveness: List[bool]) -> local_index_T:
        """
        Given an element, decides in which local it is going to be stored. If there are no locals available,
        introduces a new local and stores it there. Returns the index of the local it is being stored
        """

        # First we check whether the corresponding var elem appears in some final local and the contained value
        # is not used
        for x, final_local in enumerate(self._final_locals):
            if var_elem == final_local and not clocals_liveness[x]:
                return x

        # We traverse the locals to find a dead local in reversed order to prioritize added locals
        # If there is not such we need to add a new local to store the term
        return next((i for i in range(len(clocals_liveness) - 1, -1, -1) if clocals_liveness[i]),
                    len(clocals_liveness))

    def _can_be_computed(self, var_elem: var_T) -> bool:
        """
        A var element can be computed if it has no dependencies with instructions that have not been computed yet
        """
        instr_id = self._var2id.get(var_elem, None)
        # This should never contain a value
        assert instr_id is not None
        return True

    def _place_stack_elem_in_position(self, cstate: SymbolicState, positions: List[int]) -> List[id_T]:
        """
        Given a the set of positions an element must appear in a stack, it places them in their corresponding
        position by traversing the stack, storing the intermediate elements and loading them
        in reversed order.
        """
        ops = []
        deepest_position = max(positions)
        for i in range(deepest_position + 1):
            # If the element is already stored in a local, we just drop it
            if cstate.local_with_value(cstate.stack[i]) != -1:
                cstate.drop()
                ops.append('POP')
            else:
                x = cstate.available_local()

                # We need to set it and not tee because we have to 'remove' it from the stack. It cannot be in its
                # position because in that case it'd have been stored previously
                cstate.lset(x, False)
                ops.append(f"LSET_{x}")

        idx = top_relative_position_to_fstack(cstate.stack, self._final_stack)
        local_to_retrieve = cstate.local_with_value(self._final_stack[idx])
        # Then we place the elements from locals to the stack when possible
        while idx >= 0 and local_to_retrieve != -1:
            cstate.lget(local_to_retrieve)
            ops.append(f"LGET_{local_to_retrieve}")
            idx -= 1
            local_to_retrieve = cstate.local_with_value(self._final_stack[idx])
        return ops

    def _isolated_instr(self, instr_id: id_T) -> bool:
        """
        Given an instruction that has some deps and can appear embedded in a term (probably, a load or get instruction),
        checks it must be considered as a separate instruction to deal with.
        """
        # Access the successors it can be computed with
        other_successors = set(self._dep_graph.successors(instr_id))
        instr_successors = set(nx.nodes(nx.dfs_tree(self._instr_dep_graph, instr_id)))
        # We need to remove the same vertex
        instr_successors.remove(instr_id)
        other_successors.difference_update(instr_successors)

        if self.debug_mode:
            print("---- Detect isolated instr ----")
            for node_emb in instr_successors:
                if 'call' in node_emb or 'store' in node_emb:
                    for node_not in other_successors:
                        print(node_not, node_emb, nx.has_path(self._dep_graph, node_not, node_emb))

        # Condition: all maximal elements have some dependencies with another instruction
        # which does not use the term directly
        # TODO: improve condition
        return all(any(nx.has_path(self._dep_graph, node_not, node_emb) for node_not in other_successors)
                   for node_emb in instr_successors if 'call' in node_emb or 'store' in node_emb)

    def _available_positions_stack(self, var_elem: var_T, cstate: SymbolicState) -> List[int]:
        """
        Returns the set of positions w.r.t stack. in which we need to store the var elem and are currently available
        (i.e. there are enough elements)
        """
        positions_stack = self._var2pos_stack[var_elem]
        positions_available_stack = []

        # We determine in which positions both in the stack (enough elements) and in the locals the element
        # can be placed at this moment

        for x in positions_stack:
            idx_cstack = idx_wrt_cstack(x, cstate.stack, self._final_stack)
            # Corresponding position in cstack
            if idx_cstack >= 0:
                positions_available_stack.append(idx_cstack)

        return positions_available_stack

    def _available_positions_locals(self, var_elem: var_T, cstate: SymbolicState) -> List[int]:
        """
        Returns the set of positions w.r.t locals in which it is possible to store the var elem (i.e. doesn't contain
        it already and is not live)
        """
        positions_locals = self._var2pos_locals[var_elem]

        return [x for x in positions_locals if cstate.locals[x] != var_elem and not cstate.liveness[x]]

    def _locals_that_can_be_solved(self, var_elem: var_T, cstate: SymbolicState) -> List[int]:
        """
        Returns the set of positions in locals that can be solved: either they already contain the corresponding
        element or are not live anymore. Allows reasoning on which local to choose
        """
        positions_locals = self._var2pos_locals[var_elem]
        return [x for x in positions_locals if cstate.locals[x] == var_elem or not cstate.liveness[x]]

    def split_ids_into_categories(self) -> Tuple[Set[id_T], List[id_T], Set[id_T], Set[id_T]]:
        """
        Returns four sets of instruction ids: the ones that have some kind of dependency (mops), the ones that appear in
        fstack with no dependency (sops), the maximal elements that appear in flocals with no dependencies (lops)
        and the remaining ones (rops).
        This is useful to choose with computation used.
        """
        # We filter those mops that appear as a subterm of another term
        # TODO: Not consider load or get instructions if they can be computed when computing a superterm
        mops = {id_ for dep in self._deps for id_ in dep if self._dep_graph.out_degree(id_, 'weight') == 0
                or self._isolated_instr(id_)}
        sops = [self._var2id[stack_var] for stack_var in self._final_stack
                if stack_var in self._var2id and self._var2id[stack_var] not in mops and not cheap(
                self._var2instr[stack_var])]
        lops = {self._var2id[stack_var] for stack_var in self._final_locals
                if stack_var in self._var2id and self._var2id[stack_var] not in mops and not cheap(
                self._var2instr[stack_var])
                and self._var_total_uses[stack_var] == 1}
        rops = set(self._id2instr.keys()).difference(mops.union(sops).union(lops))
        return mops, sops, lops, rops

    def select_memory_ops_order(self, mops: Set[id_T]) -> List[id_T]:
        """
        Returns a compatible order w.r.t mops, considering the different dependencies that are formed both from
        deps and due to subterms embedded into terms
        """
        # As the dependency relation among instructions is represented as a happens-before, we need to reverse the
        # toposort to start with the deepest elementsh
        topo_order = nx.topological_sort(self._trans_sub_graph)
        # We must extract the order that only includes ids from mops
        return [id_ for id_ in topo_order if id_ in mops]

    def var_must_be_moved(self, var_elem: var_T, cstate: SymbolicState) -> bool:
        """
        By construction, a var element must be moved if it appears in no locals, unless it has one position tied which
        corresponds to current top of the stack
        """
        positions_stack = self._var2pos_stack[var_elem]

        local = cstate.local_with_value(var_elem)
        instr = self._var2instr.get(var_elem, None)

        # We don't need to move a var if it's already in locals and hence, they must satisfy that local == -1.
        # Also, if the var corresponds to a cheap instruction or all occurrences of that var have been already computed
        # and it must be at current position
        return local == -1 and not (self._var_total_uses[var_elem] == cstate.var_uses[var_elem] and len(positions_stack) >= 1 and
                                     idx_wrt_cstack(positions_stack[0], cstate.stack, self._final_stack) == 0) and not (instr is not None and cheap(instr))

    def can_be_placed_in_position(self, var_elem: var_T, clocals: List[var_T], clocals_liveness: List[bool]) -> bool:
        """
        Returns True if current element can be placed in its position
        """
        for x, final_local in enumerate(self._final_locals):
            if var_elem == final_local and not clocals_liveness[x]:
                return True
        return False

    def move_top_to_position(self, var_elem: var_T, cstate: SymbolicState) -> List[id_T]:
        """
        Tries to store current element in all the positions in which it is available to be moved
        """
        ops = []
        positions_available_locals = self._available_positions_locals(var_elem, cstate)
        positions_available_stack = self._available_positions_stack(var_elem, cstate)

        if self.debug_mode:
            print('---- Move top to position ----')
            print(cstate)
            print("Positions locals:", positions_available_locals)
            print("Positions stack:", positions_available_stack)
            print("")

        if len(positions_available_locals) == 0:
            # No position is available yet, so we just store it one available local
            y = cstate.available_local()
            cstate.lset(y, False)
            ops.append(f"LSET_{y}")

            # If it needs to be stored somewhere in the stack, we perform that step at this point
            if len(positions_available_stack) > 0:
                self._place_stack_elem_in_position(cstate, positions_available_stack)

        else:
            # There is at least one local in fstate in which we can place the state. We first store the
            # value in all locals with several ltee until the last one, which can be either a lset or ltee
            i = 0
            while i < len(positions_available_locals) - 1:
                local = positions_available_locals[i]
                cstate.ltee(local, True)
                ops.append(f"LTEE_{local}")
                i += 1

            # Finally, we check whether the last store to local can be a ltee (if there is an element left to place in
            # the stack at the same position) or a lset
            if len(positions_available_stack) == 1 and idx_wrt_cstack(positions_available_stack[0], cstate.stack,
                                                                      self._final_stack) == 0:
                local = positions_available_locals[i]
                cstate.ltee(local, True)
                ops.append(f"LTEE_{local}")
            else:
                y = cstate.available_local()
                cstate.lset(y, False)
                ops.append(f"LSET_{y}")

                if len(positions_available_stack) > 0:
                    self._place_stack_elem_in_position(cstate, positions_available_stack)

        return ops

    def choose_next_computation(self, cstate: SymbolicState, mops: List[id_T], sops: List[id_T], lops: Set[id_T]) -> Tuple[id_T, str]:
        """
        Returns an element from mops, sops or lops and where it came from (mops, sops or lops)
        TODO: Here we should try to devise a good heuristics to select the terms
        """

        # First we try to assign the next element which is top of the stack
        top_idx = idx_wrt_fstack(0, cstate.stack, self._final_stack) - 1
        if 0 <= top_idx < len(self._final_stack):
            top_var = self._final_stack[top_idx]
            top_id = self._var2id.get(top_var, None)

            # It corresponds to an initial value that should have been stored in a local
            if top_id is None:
                x = cstate.local_with_value(top_var)

                if x == -1:
                    raise ValueError("Initial value from choose_next_computation is not stored in a local")

                return top_var, 'local'

            instr = self._var2instr[top_var]
            # Cheap instruction can be computed directly
            if cheap(instr):
                return top_id, 'cheap'

            # If it is cheap to compute or has no dependencies, we can choose it
            elif self._trans_sub_graph.in_degree(top_id) == 0:
                return top_id, 'sops'

        # To determine the best candidate, we annotate which element can solve the most number of locals for the same
        # stack var. If > 1, we can chain one ltee that could save 1 instruction
        candidate = None
        max_number_solved = 0

        # First we try to assign an element from the list of final local values if it can be placed in all the gaps
        # We also determine the element which has more
        for id_ in lops:
            top_instr = self._id2instr[id_]

            # TODO: future maybe allow choosing operations that had been already chosen to allow just computing them
            #  from the stack and placing them in their positions

            # If there is an operation whose produced elements can be placed in all locals, we choose that element
            all_solved = True
            number_solved = 0

            # Call instructions might generate multiple values that we should take into account
            # TODO: maybe resolve ties somehow?
            for out_var in top_instr['outpt_sk']:
                # TODO consider in locals that can be solved how the operation could liberate some local registers
                avail_solved_flocals = self._locals_that_can_be_solved(out_var, cstate)
                pos_flocals = self._var2pos_locals[out_var]

                # Current heuristics: select as a candidate the instruction with the most number of positions that
                # can be solved
                all_solved = all_solved and len(pos_flocals) == len(avail_solved_flocals)
                number_solved = max(number_solved, len(avail_solved_flocals))

            if all_solved:
                return id_, 'lops'
            # >= to ensure at least one operation is solved
            elif number_solved >= max_number_solved:
                candidate = id_

        # After that, we try to assign an element of mops (if there are any)
        if len(mops) > 0:
            return mops[0], 'mops'

        # Finally, we just choose an element from locals that covers multiple gaps, which has been determined previously
        return candidate, 'lops'

    def compute_var(self, var_elem: var_T, cstate: SymbolicState) -> List[id_T]:
        """
        Given a stack_var and current state, computes the element and updates cstate accordingly. Returns the sequence of ids.
        Compute var considers it the var elem is already contained in a local and also updates the locals accordingly
        """
        # We assume current var_elem is not stored in a local, as compute_instr determines so
        instr = self._var2instr[var_elem]
        seq = self.compute_instr(instr, cstate)

        # Finally, we need to decide if the element is stored in a local or not. If so, we apply
        # a ltee instruction to place it in the corresponding local. We only store an instruction
        # if it is not cheap and it must be placed in more places
        if not cheap(instr) and cstate.var_uses[var_elem] < self._var_total_uses[var_elem]:
            avail_pos_locals = self._available_positions_locals(var_elem, cstate)

            # If there are more than one local in which the element is stored, we just do so
            if len(avail_pos_locals) > 0:
                for x in avail_pos_locals:
                    # We just apply as many tees as possible, so the element is placed in its position
                    cstate.ltee(x, True)
                    seq.append(f"LTEE_{x}")
            else:
                # Otherwise, we assign a new local
                x = cstate.available_local()
                cstate.ltee(x, False)
                seq.append(f"LTEE_{x}")
        return seq

    def compute_instr(self, instr: instr_T, cstate: SymbolicState) -> List[id_T]:
        """
        Given an instr and the current state, computes the corresponding term. This function is separated from compute_op because there
        are terms, such as var accesses or memory accesses that produce no var element as a result. Also it does not
        consider how to update the locals
        """
        seq = []

        # First we decide in which order we compute the arguments
        if instr['commutative']:
            # If it's commutative, study its dependencies.
            if self.debug_mode:
                assert len(instr['inpt_sk']) == 2, f'Commutative instruction {instr["id"]} has arity != 2'

            # TODO: add condition
            condition = False
            if condition:
                input_vars = instr['inpt_sk']
            else:
                input_vars = reversed(instr['inpt_sk'])
        else:
            input_vars = reversed(instr['inpt_sk'])

        for stack_var in input_vars:
            local = cstate.local_with_value(stack_var)
            if local != -1:
                # If it's already stored in a local, we just retrieve it
                cstate.lget(local)
                seq.append(f"LGET_{local}")
            else:
                # Otherwise, we must return generate it with a recursive call
                seq.extend(self.compute_var(stack_var, cstate))

        # Finally, we compute the element
        cstate.uf(instr)
        seq.append(instr["id"])
        return seq

    def solve_permutation(self, cstate: SymbolicState) -> List[id_T]:
        """
        After all terms have been computed, solve_permutation places all elements in their
        corresponding place.
        """
        optp = []
        stack_idx = len(self._final_stack) - len(cstate.stack) - 1

        # First we solve the values in the stack
        while stack_idx >= 0:
            # The corresponding value must be stored in some local register
            x = cstate.local_with_value(self._final_stack[stack_idx])

            if x == -1:
                raise ValueError("Stack var in solve_permutation is not stored in a local")

            cstate.lget(x)
            optp.append(f"LGET_{x}")
            stack_idx -= 1

        # Then we detect which locals have a value that appears in flocals and load them onto the stack
        outdated_locals = []
        for local_idx in range(len(self._final_locals)):
            if self._final_locals[local_idx] != cstate.locals[local_idx]:
                x = cstate.local_with_value(self._final_locals[local_idx])
                outdated_locals.append(local_idx)
                cstate.lget(x)
                optp.append(f"LGET_{x}")

            local_idx += 1

        # Finally, we store them in the corresponding local in reversed order
        for x in reversed(outdated_locals):
            cstate.lset(x, True)
            optp.append(f"LSET_{x}")

        return optp

    def _debug_initial(self, mops: List[id_T], lops: Set[id_T], sops: List[id_T]):
        if self.debug_mode:
            print("---- Initial Ops ----")
            print('Mops:', mops)
            print('Lops:', lops)
            print('Sops:', sops)
            print("")

    def _debug_loop(self, cstate: SymbolicState, optg: List[id_T], mops: List[id_T], sops: List[id_T], lops: List[id_T]):
        if self.debug_mode:
            print("---- While loop ----")
            print("Ids", optg)
            print('Ops', mops, sops, lops)
            print('State', cstate)
            print("")

    def _debug_drop(self, var_top: var_T, cstate: SymbolicState):
        if self.debug_mode:
            print("---- Drop term ----")
            print("Var Term", var_top)
            print('State', cstate)
            print("")

    def _debug_choose_computation(self, next_id: id_T, location: str, cstate: SymbolicState):
        if self.debug_mode:
            print("---- Computation chosen ----")
            print(next_id, location)
            print(cstate)
            print("")

    def greedy(self) -> List[id_T]:
        cstate: SymbolicState = SymbolicState(self._initial_stack.copy(), self._initial_locals.copy(),
                                              self._var_total_uses, self.debug_mode)

        # We split into three sets: mops (operations with dependencies), sops (elements that appear in fstack
        # that do not appear in mops) and rops (other operations with no restrictions)
        mops_unsorted, sops, lops = self._mops.copy(), self._sops.copy(), self._lops.copy()
        mops: List[id_T] = self.select_memory_ops_order(mops_unsorted)
        optg: List[id_T] = []

        self._debug_initial(mops, lops, sops)

        # For easier code, we end the while when we need to choose an operation and there are no operations left
        while True:
            var_top = cstate.top_stack()

            # Top of the stack must be removed, as it appears more time it is being used
            if var_top is not None and cstate.var_uses[var_top] > self._var_total_uses[var_top]:

                self._debug_drop(var_top, cstate)
                cstate.drop()
                optg.append("POP")

            # Top of the stack must be placed in some other position
            elif var_top is not None and self.var_must_be_moved(var_top, cstate):
                optg.extend(self.move_top_to_position(var_top, cstate))

            # Top of the stack cannot be moved to the corresponding position. As there is always the possibility
            # of storing in locals, this means that either the stack is empty or the current top of the stack
            # is already placed in the corresponding position. Hence, we just generate the following computation
            else:
                # There are no operations left to choose, so we stop the search
                if len(mops) + len(sops) + len(lops) == 0:
                    break

                next_id, location = self.choose_next_computation(cstate, mops, sops, lops)
                self._debug_choose_computation(next_id, location, cstate)

                # It is already stored in a local
                if location == 'local':
                    # We just load the stack var
                    x = cstate.local_with_value(next_id)
                    cstate.lget(x)
                    ops = [f'LGET_{x}']
                elif location == 'cheap':
                    # Cheap instructions are just computed, there is no need to erase elements from the lists
                    ops = self.compute_instr(self._id2instr[next_id], cstate)
                else:
                    remove_computation(next_id, location, mops, sops, lops)
                    ops = self.compute_instr(self._id2instr[next_id], cstate)

                    # Remove possible computations from mops that have been computed as part of the process
                    while len(mops) > 0 and mops[0] in ops:
                        remove_computation(mops[0], 'mops', mops, sops, lops)
                        self._trans_sub_graph.remove_node(mops[0])

                    self._trans_sub_graph.remove_node(next_id)

                optg.extend(ops)

        optg.extend(self.solve_permutation(cstate))

        if self.debug_mode:
            print("---- State after solving permutation ----")
            print(cstate)
            print(optg)
            print("")

        return optg


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        sfs = json.load(f)
    ids = SMSgreedy(sfs, True).greedy()
    print("Original")
    print("")
    print(sfs['original_instrs_with_ids'])
    print("")
    print("New")
    print("")
    print(ids)
    if check_execution_from_ids(sfs, ids):
        print("Check works!!")
