#!/usr/bin/env python3

import json
import sys
import os
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict
import traceback

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


def iterative_topological_sort(graph: Dict, initial_node) -> List:
    seen = set()
    stack = []    # path variable is gone, stack and order are new
    order = []    # order will be in reverse order at first
    q = [initial_node]
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v)
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]:
                order.append(stack.pop())
            stack.append(v)

    return stack + order[::-1]   # new return value!


def toposort_instr_dependencies(graph: Dict) -> List:
    maximal_elements = list(set(instr_id for instr_id in graph).difference(
        set(instr_id for id_list in graph.values() for instr_id in id_list)))
    extended_dependency_graph = graph.copy()

    # We add a "dummy element" to function as the initial value
    extended_dependency_graph['dummy'] = maximal_elements
    topo_order = iterative_topological_sort(extended_dependency_graph, 'dummy')
    topo_order.pop(0)
    return topo_order


class SMSgreedy:

    def __init__(self, json_format):
        self._bs: int = json_format['max_sk_sz']
        self._user_instr: List[instr_T] = json_format['user_instrs']
        self._b0: int = json_format["init_progr_len"]
        self._initial_stack: List[var_T] = json_format['src_ws']
        self._final_stack: List[var_T] = json_format['tgt_ws']
        self._variables: List[var_T] = json_format['vars']
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

        # We include another list of local variables to indicate whether it is live or not
        # Condition: a variable is live if it is used somewhere else, or it is in its correct position.
        # As we are considering just the variables with some kind of change for locals, we just need to check if
        # it's used somewhere else (i.e. it is used somewhere else)
        self._local_liveness: List[bool] = [self._var_current_uses[initial_var] < self._var_total_uses[initial_var]
                                            for initial_var in self._initial_locals]

        self._dependency_graph, self._full_dependency_graph = self._compute_full_dependency_graph_elem()

        self._mops, self._sops, self._rops = self.split_ids_into_categories()

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

    def _compute_dependency_graph_elem(self, current_id: id_T, analyzed: Set[id_T],
                                       immediate_dependency_graph: Dict[id_T, List[id_T]],
                                       full_dependency_graph: Dict[id_T, Set[id_T]]):
        if current_id in analyzed:
            return

        instr = self._id2instr[current_id]
        analyzed.add(current_id)
        for input_var in instr['inpt_sk']:
            # This means the stack element corresponds to another uninterpreted instruction
            if input_var in self._var2instr:
                dep_id = self._var2instr[input_var]['id']
                immediate_dependency_graph[current_id].append(dep_id)
                full_dependency_graph[current_id].add(dep_id)

                self._compute_dependency_graph_elem(dep_id, analyzed, immediate_dependency_graph, full_dependency_graph)
                full_dependency_graph[current_id].update(full_dependency_graph[dep_id])

    def _compute_full_dependency_graph_elem(self) -> Tuple[Dict[id_T, List[id_T]], Dict[id_T, Set[id_T]]]:
        immediate_dependency_graph = defaultdict(lambda: [])
        full_dependency_graph = defaultdict(lambda: set())
        analyzed = set()
        for instr in self._user_instr:
            instr_id = instr['id']
            self._compute_dependency_graph_elem(instr_id, analyzed, immediate_dependency_graph, full_dependency_graph)

        # We need to consider also the order given by the tuples
        for id1, id2 in self._deps:
            immediate_dependency_graph[id2].append(id1)

            # To update the full dependency graph, we just need to include both the new term and its subterms
            full_dependency_graph[id2].add(id1)
            full_dependency_graph[id2].update(full_dependency_graph[id1])

        return immediate_dependency_graph, full_dependency_graph

    def _compute_dependency_graph(self) -> Dict[id_T, List[id_T]]:
        dependency_graph = {}
        for instr in self._user_instr:
            instr_id = instr['id']
            dependency_graph[instr_id] = []

            for stack_elem in instr['inpt_sk']:
                # This means the stack element corresponds to another uninterpreted instruction
                if stack_elem in self._var2instr:
                    dependency_graph[instr_id].append(self._var2instr[stack_elem]['id'])

        # We need to consider also the order given by the tuples
        for id1, id2 in self._deps:
            dependency_graph[id2].append(id1)

        return dependency_graph

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

    def split_ids_into_categories(self) -> Tuple[Set[id_T], List[id_T], Set[id_T]]:
        """
        Returns three set of instruction ids: the ones that have some kind of dependency (mops), the ones that appear in
        fstack with no dependency (sops) and the remaining ones (rops). This is useful to choose with computation used.
        The oterder
        """
        mops = {id_ for dep in self._deps for id_ in dep}
        sops = [self._var2id[stack_var] for stack_var in self._final_stack
                if stack_var in self._var2id and self._var2id[stack_var] not in mops]
        rops = set(self._id2instr.keys()).difference(mops.union(sops))
        return mops, sops, rops

    def permutation(self) -> bool:
        pass
    
    def select_memory_ops_order(self, mops: Set[id_T]) -> List[id_T]:
        """
        Returns a compatible order w.r.t mops, considering the different dependencies that are formed both from
        deps and due to subterms embedded into terms
        """
        # As the dependency relation among instructions is represented as a happens-before, we need to reverse the
        # toposort to start with the deepest elementsh
        topo_order = reversed(toposort_instr_dependencies(self._dependency_graph))

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

    def move_top_to_position(self, var_elem: var_T, cstack: List[var_T]) -> List[id_T]:
        pass

    def choose_next_computation(self) -> id_T:
        pass

    def compute_opt(self) -> List[id_T]:
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
        mops_unsorted, sops, rops = self._mops.copy(), self._sops.copy(), self._rops.copy()
        seq: List[id_T] = []
        mops: List[id_T] = self.select_memory_ops_order(mops_unsorted)
        maximal_vars: List[id_T] = []
        print(mops)
        # optg: List[id_T] = []
        # while mops != []:
        #     var_top = cstack[0] if len(cstack) > 0 else None
        #
        #     # Top of the stack must be removed, as it appears more time it is being used
        #     if var_top is not None and self._var_current_uses[var_top] > self._var_total_uses[var_top]:
        #         # Decrease the number of occurrences and remove it from the stack
        #         self._var_current_uses[var_top] -= 1
        #         cstack.pop(0)
        #         optg.append("drop")
        #
        #     # Top of the stack must be placed in the position it appears in the fstack or flocals
        #     elif var_top is not None and not self.is_in_position_stack(var_top, cstack):
        #         optg.extend(self.move_top_to_position(var_top, cstack))
        #
        #     # Top of the stack cannot be moved to the corresponding position. As there is always the possibility
        #     # of storing in locals, this means that either the stack is empty or the current top of the stack
        #     # is already placed in the corresponding position. Hence, we just generate the following computation
        #     else:
        #         next_computation, mops = self.choose_next_computation()
        #         ops = self.choose_subterm_order()
        #         optg.extend(ops)
        #
        # optg.extend(self.solve_permutation(cstack, clocals))


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        sfs = json.load(f)
    SMSgreedy(sfs).greedy()