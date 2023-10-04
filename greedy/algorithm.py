#!/usr/bin/env python3

import json
import sys
import os
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict, Counter
import traceback
from enum import Enum, unique
import networkx as nx

# General types we are using
var_T = str
id_T = str
instr_T = Dict[str, Any]
local_index_T = int


def select_local_from_value(value: var_T, locals_: List[var_T]) -> local_index_T:
    """
    Returns the first local in the list which contains the corresponding value. Otherwise, raises an Exception.
    """
    return locals_.index(value)


def set_local_value(x: int, cstack: List[var_T], clocals: List[var_T]) -> Tuple[List[var_T], List[var_T]]:
    """
    Returns the state of the stack and locals after storing the top of the stack in the local with index x
    """
    elem = cstack.pop(0)
    clocals[x] = elem
    return cstack, clocals


@unique
class Location(Enum):
    stack = 0
    local = 1


class SymbolicState:
    """
    A symbolic state includes a stack, the register (locals) and the liveness analysis from the stack
    """

    def __init__(self, stack: List[var_T], locals_: List[var_T], liveness: List[bool], debug_mode: bool = True) -> None:
        self.stack: List[var_T] = stack
        self.locals: List[var_T] = locals_
        self.liveness: List[bool] = liveness
        self.debug_mode: bool = debug_mode
        self._var2position: Dict[var_T, List[Tuple[Location, int]]] = self._compute_var2position()
        self.var_uses = self._computer_var_uses()

    def _compute_var2position(self) -> Dict[var_T, List[Tuple[Location, int]]]:
        var2pos = defaultdict(lambda: [])
        for i, stack_var in enumerate(self.stack):
            var2pos[stack_var].append((Location.stack, i))

        for i, stack_var in enumerate(self.locals):
            var2pos[stack_var].append((Location.local, i))

        return var2pos

    def _computer_var_uses(self):
        var_uses = defaultdict(lambda: 0)

        # Count vars in the initial stack
        for var_stack in self.stack:
            var_uses[var_stack] += 1

        # Count vars in the initial locals
        for var_stack in self.locals:
            var_uses[var_stack] += 1

        return var_uses

    def var_locations(self, var: var_T) -> List[Tuple[Location, int]]:
        return self._var2position.get(var, [])

    def select_local_from_value(self, value: var_T) -> local_index_T:
        """
        Returns the first local in the list which contains the corresponding value. Otherwise, raises an Exception.
        """
        return self.locals.index(value)

    def lset(self, x: int) -> None:
        """
        Stores the top of the stack in the local with index x
        """
        self.locals[x] = self.stack.pop(0)

    def ltee(self, x: int) -> None:
        """
        Tee instruction in local with index x
        """
        self.locals[x] = self.stack[0]

    def lget(self, x: int) -> var_T:
        """
        Get instruction in local x
        """
        return self.locals[x]

    def drop(self):
        """
        Drops the last element
        """
        self.stack.pop(0)

    def uf(self, instr: instr_T):
        """
        Symbolic execution of instruction instr. Additionally, checks the arguments match if debug mode flag is enabled
        """
        consumed_elements = [self.stack.pop(0) for _ in range(len(instr['inpt_sk']))]

        # Debug mode to check the pop args from the stack match
        if self.debug_mode:
            if instr['comm']:
                # Compare them as multisets
                return Counter(consumed_elements) == Counter(instr['inpt_sk'])
            else:
                # Compare them as lists
                return consumed_elements == instr['inpt_sk']

        # We introduce the new elements
        for output_var in instr['outpt_sk']:
            self.stack.insert(0, output_var)



class SMSgreedy:

    def __init__(self, json_format):
        self._bs: int = json_format['max_sk_sz']
        self._user_instr: List[instr_T] = json_format['user_instrs']
        self._b0: int = json_format["init_progr_len"]
        self._initial_stack: List[var_T] = json_format['src_ws']
        self._final_stack: List[var_T] = json_format['tgt_ws']
        self._vars: List[var_T] = json_format['vars']
        self._deps: List[Tuple[id_T, id_T]] = json_format['dependencies']
        self._max_registers: int = json_format['max_registers_sz']
        self._local_changes: List[Tuple[var_T, var_T]] = json_format['register_changes']

        # We split into two different dicts the initial values and final values in locals
        self._initial_locals: List[var_T] = [local_repr[0] for local_repr in self._local_changes]
        self._final_locals: List[var_T] = [local_repr[1] for local_repr in self._local_changes]

        # Note: call instructions might have several variables in 'outpt_sk'
        self._var2instr = {var: ins for ins in self._user_instr for var in ins['outpt_sk']}
        self._id2instr = {ins['id']: ins for ins in self._user_instr}
        self._var2id = {var: ins['id'] for ins in self._user_instr for var in ins['outpt_sk']}

        self._var_total_uses = self._compute_var_total_uses()
        self._var_current_uses = self._computer_var_current_uses()
        self._dep_graph = self._compute_dependency_graph()
        self._trans_dep_graph = nx.transitive_closure_dag(self._dep_graph)
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
        self._local_liveness: List[bool] = [self._var_current_uses[initial_var] < self._var_total_uses[initial_var]
                                            for initial_var in self._initial_locals]

        self._mops, self._sops, self._lops, self._rops = self.split_ids_into_categories()

        self._needed_in_stack_map = {}
        self._dup_stack_ini = 0
        self.uses = {}

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

    def _computer_var_current_uses(self) -> Dict[var_T, int]:
        """
        Computes how many times are initially computed, i.e. 0 if terms do not appear in the initial stack or initial
        locals, x otherwise (probably 1 because initial elements are different, but this could change in the future)
        """
        var_uses = defaultdict(lambda: 0)

        # Count vars in the initial stack
        for var_stack in self._initial_stack:
            var_uses[var_stack] += 1

        # Count vars in the initial locals
        for var_stack in self._initial_locals:
            var_uses[var_stack] += 1

        return var_uses

    def _compute_dependency_graph(self) -> nx.DiGraph:
        edge_list = []
        for instr in self._user_instr:
            instr_id = instr['id']

            for stack_elem in instr['inpt_sk']:
                # This means the stack element corresponds to another uninterpreted instruction
                if stack_elem in self._var2instr:
                    edge_list.append((self._var2id[stack_elem], instr_id))
                # Otherwise, it corresponds to either a local or an initial element in the stack. We just add locals
                elif 'local' in stack_elem:
                    edge_list.append((stack_elem, instr_id))

        # We need to consider also the order given by the tuples
        for id1, id2 in self._deps:
            edge_list.append((id1, id2))

        # Also, the dependencies induced among locals that are live
        for ini_var, final_var in self._local_changes:
            # Either final var corresponds to a computation and appears in var2id or it is another local, which
            # we are referencing using the same name
            final_id = self._var2id.get(final_var, final_var)
            edge_list.append((ini_var, final_id))

        return nx.transitive_reduction(nx.DiGraph(edge_list))

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

    def split_ids_into_categories(self) -> Tuple[Set[id_T], List[id_T], Set[id_T], Set[id_T]]:
        """
        Returns four sets of instruction ids: the ones that have some kind of dependency (mops), the ones that appear in
        fstack with no dependency (sops), the maximal elements that appear in flocals with no dependencies (lops)
        and the remaining ones (rops).
        This is useful to choose with computation used.
        """
        mops = {id_ for dep in self._deps for id_ in dep}
        sops = [self._var2id[stack_var] for stack_var in self._final_stack
                if stack_var in self._var2id and self._var2id[stack_var] not in mops]
        lops = {self._var2id[stack_var] for stack_var in self._final_locals
                if stack_var in self._var2id and self._var2id[stack_var] not in mops and
                self._var_total_uses[stack_var] == 1}
        rops = set(self._id2instr.keys()).difference(mops.union(sops).union(lops))
        return mops, sops, lops, rops

    def permutation(self) -> bool:
        pass
    
    def select_memory_ops_order(self, mops: Set[id_T]) -> List[id_T]:
        """
        Returns a compatible order w.r.t mops, considering the different dependencies that are formed both from
        deps and due to subterms embedded into terms
        """
        # As the dependency relation among instructions is represented as a happens-before, we need to reverse the
        # toposort to start with the deepest elementsh
        topo_order = nx.topological_sort(self._dep_graph)
        # We must extract the order that only includes ids from mops
        return [id_ for id_ in topo_order if id_ in mops]

    def is_in_position_stack(self, var_elem: var_T, cstack: List[var_T]) -> bool:
        # Var elem has position 0 in cstack, hence why new pos ic computed this way
        new_pos = len(cstack) - len(self._final_stack)
        # Check it is within range and
        return 0 <= new_pos < len(self._final_stack) and self._final_stack[new_pos] == var_elem

    def can_be_placed_in_position(self, var_elem: var_T, clocals: List[var_T], clocals_liveness: List[bool]) -> bool:
        """
        Returns True if current element can be placed in its position
        """
        for x, final_local in enumerate(self._final_locals):
            if var_elem == final_local and not clocals_liveness[x]:
                return True
        return False

    def move_top_to_position(self, var_elem: var_T, cstack: List[var_T], clocals: List[var_T],
                             clocals_liveness: List[var_T]) -> List[id_T]:
        """
        Move top to position assumes the stack element is not in their corresponding position.
        If it must appear more than once, we move it to a local to duplicate it (checking if it appears in flocals).
        Otherwise,
        """
        pass

    def choose_next_computation(self, mops: List[id_T], sops: List[id_T], rops: Set[id_T],
                                cstack: List[var_T], clocals: List[var_T], clocals_liveness: List[bool]) -> id_T:
        """
        TODO: Here we should try to devise a good heuristics to select the terms
        """
        # First we try updating locals that are not live
        for x, final_var in enumerate(self._final_locals):
            if not clocals_liveness[x]:
                var_instr_id = self._var2id.get(final_var, None)
                if var_instr_id is not None:
                    return var_instr_id

        # We first prioritize the next element in the stack if it is possible to compute directly
        next_element = sops[0]
        if self._can_be_computed(next_element):
            return next_element

        # Finally, we just compute the next element from mops (which is always assumed to be computed)
        if self._can_be_computed(mops[0]):
            return mops[0]

        # This case should never be reached
        raise AssertionError('Choose next computation found an unexpected case')

    def compute_op(self, mops: List[id_T], sops: List[id_T], rops: Set[id_T], cstack: List[var_T],
                   clocals: List[var_T], clocals_liveness: List[bool]) -> List[id_T]:
        pass

    def choose_subterm_order(self, mops: List[id_T], sops: List[id_T], rops: Set[id_T], cstack: List[var_T],
                             clocals: List[var_T], clocals_liveness: List[bool]) -> List[id_T]:
        pass


    def solve_permutation(self, cstack: List[var_T], clocals: List[var_T]) -> List[id_T]:
        """
        After all terms have been computed, solve_permutation places all elements in their
        corresponding place.
        """
        optp = []
        stack_idx = len(self._final_stack) - len(cstack) - 1

        # First we solve the values in the stack
        while stack_idx >= 0:
            # The corresponding value must be stored in some local register
            x = select_local_from_value(self._final_stack[stack_idx], clocals)
            cstack.append(clocals[x])
            optp.append(f"local.get[{x}]")
            stack_idx -= 1

        # Then we detect which locals have a value that appears in flocals and load them onto the stack
        outdated_locals = []
        for local_idx in range(len(self._final_locals)):
            if self._final_locals[local_idx] != clocals[local_idx]:
                x = select_local_from_value(self._final_locals[local_idx], clocals)
                outdated_locals.append(local_idx)
                cstack.append(clocals[x])
                optp.append(f"local.get[{x}]")

            local_idx += 1

        # Finally, we store them in the corresponding local in reversed order
        for x in reversed(outdated_locals):
            cstack, clocals = set_local_value(x, cstack, clocals)
            optp.append(f"local.set[{x}]")

        return optp

    def greedy(self):
        cstack: List[var_T] = self._initial_stack.copy()
        clocals: List[var_T] = self._initial_locals.copy()
        clocals_liveness: List[bool] = self._local_liveness.copy()

        # We split into three sets: mops (operations with dependencies), sops (elements that appear in fstack
        # that do not appear in mops) and rops (other operations with no restrictions)
        mops_unsorted, sops, lops = self._mops.copy(), self._sops.copy(), self._lops.copy()
        seq: List[id_T] = []
        mops: List[id_T] = self.select_memory_ops_order(mops_unsorted)
        maximal_vars: List[id_T] = []

        optg: List[id_T] = []

        while mops != [] or sops != [] or lops != []:
            var_top = cstack[0] if len(cstack) > 0 else None

            # Top of the stack must be removed, as it appears more time it is being used
            if var_top is not None and self._var_current_uses[var_top] > self._var_total_uses[var_top]:
                # Decrease the number of occurrences and remove it from the stack
                self._var_current_uses[var_top] -= 1
                cstack.pop(0)
                optg.append("drop")

            # Top of the stack must be placed in the position it appears in the fstack or flocals
            elif var_top is not None and not self.is_in_position_stack(var_top, cstack):
                optg.extend(self.move_top_to_position(var_top, cstack, clocals, clocals_liveness))

            # Top of the stack cannot be moved to the corresponding position. As there is always the possibility
            # of storing in locals, this means that either the stack is empty or the current top of the stack
            # is already placed in the corresponding position. Hence, we just generate the following computation
            else:
                next_computation = self.choose_next_computation(mops, sops, lops, cstack, clocals, clocals_liveness)
                ops = self.choose_subterm_order(mops, sops, lops, cstack, clocals, clocals_liveness)
                optg.extend(ops)

        optg.extend(self.solve_permutation(cstack, clocals))


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        sfs = json.load(f)
    SMSgreedy(sfs).greedy()