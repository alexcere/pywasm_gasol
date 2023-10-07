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


@unique
class Location(Enum):
    stack = 0
    local = 1


class SymbolicState:
    """
    A symbolic state includes a stack, the register (locals) and a dict indicating the number of total uses of each
    instruction. With this dict, we can determine the liveness analysis
    """

    def __init__(self, stack: List[var_T], locals_: List[var_T], total_uses: Dict[id_T, int], debug_mode: bool = True) -> None:
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

        # Count vars in the initial locals
        for var_stack in self.locals:
            var_uses[var_stack] += 1

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

    def uf(self, instr: instr_T):
        """
        Symbolic execution of instruction instr. Additionally, checks the arguments match if debug mode flag is enabled
        """
        consumed_elements = [self.stack.pop(0) for _ in range(len(instr['inpt_sk']))]

        # Neither liveness nor var uses are affected by consuming elements, as these elements are just being embedded
        # into a new term

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

            # Var uses: increase one for each generated stack var
            self.var_uses[output_var] += 1

            # Liveness: not affected because we are not managing the stack

    def top_stack(self) -> Optional[var_T]:
        return None if len(self.stack) == 0 else self.stack[0]

    def available_local(self) -> int:
        """
        Choose an available local to store an element. Otherwise, introduces an extra local to store it
        :return:
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
        sentences = [f"Current stack: {self.stack}", "Current locals:", *(f"{local}: {liveness}" for local, liveness in zip(self.locals, self.liveness))]
        return '\n'.join(sentences)


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
        self._var2pos = self._compute_var2pos()

        self._var_total_uses = self._compute_var_total_uses()
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

        self._mops, self._sops, self._lops, self._rops = self.split_ids_into_categories()

        # We need to compute the sub graph over the full dependency graph, as edges could be lost if we use the
        # transitive reduction instead
        self._trans_sub_graph = nx.transitive_reduction(self._dep_graph.subgraph(itertools.chain.from_iterable([self._mops, self._sops, self._lops])))

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

    def _compute_var2pos(self) -> Dict[var_T, List[Tuple[Location, int]]]:
        var2pos = defaultdict(lambda: [])
        for i, stack_var in enumerate(self._final_stack):
            var2pos[stack_var].append((Location.stack, i))

        for i, stack_var in enumerate(self._final_locals):
            var2pos[stack_var].append((Location.local, i))

        return var2pos

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

    def _available_positions(self, var_elem: var_T, cstate: SymbolicState) -> Tuple[List[int], List[int]]:
        """
        Returns the set of positions w.r.t stack and locals resp. in which we need to store the var elem
        """
        positions = self._var2pos[var_elem]

        positions_available_stack = []
        positions_available_locals = []

        # We determine in which positions both in the stack (enough elements) and in the locals the element
        # can be placed at this moment
        for location, x in positions:
            if location == Location.stack:
                idx_cstack = idx_wrt_cstack(x, cstate.stack, self._final_stack)
                # Corresponding position in cstack
                if idx_cstack >= 0:
                    positions_available_locals.append(idx_cstack)
            else:
                if cstate.locals[x] != var_elem and not cstate.liveness[x]:
                    positions_available_locals.append(x)

        return positions_available_stack, positions_available_locals

    def split_ids_into_categories(self) -> Tuple[Set[id_T], List[id_T], Set[id_T], Set[id_T]]:
        """
        Returns four sets of instruction ids: the ones that have some kind of dependency (mops), the ones that appear in
        fstack with no dependency (sops), the maximal elements that appear in flocals with no dependencies (lops)
        and the remaining ones (rops).
        This is useful to choose with computation used.
        """
        mops = {id_ for dep in self._deps for id_ in dep}
        sops = [self._var2id[stack_var] for stack_var in self._final_stack
                if stack_var in self._var2id and self._var2id[stack_var] not in mops and not cheap(self._var2instr[stack_var])]
        lops = {self._var2id[stack_var] for stack_var in self._final_locals
                if stack_var in self._var2id and self._var2id[stack_var] not in mops and not cheap(self._var2instr[stack_var])
                and self._var_total_uses[stack_var] == 1}
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

    def var_must_be_moved(self, var_elem: var_T, cstate: SymbolicState) -> bool:
        """
        By construction, a var element must be moved if it appears in no locals, unless it has one position tied which
        corresponds to current top of the stack
        """
        positions = self._var2pos[var_elem]
        local = cstate.local_with_value(var_elem)

        assert len(positions) > 0, f'Variable {var_elem} does not appear anywhere else!'
        return local == -1 or len(positions) > 1 or positions[0][0] != 'stack' or idx_wrt_cstack(positions[0][1], cstate.stack, self._final_stack) != 0

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
        positions_available_locals, positions_available_stack = self._available_positions(var_elem, cstate)

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
            if len(positions_available_stack) == 1 and idx_wrt_cstack(positions_available_stack[0], cstate.stack, self._final_stack) == 0:
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


    # def solve_permutation(self, cstate: SymbolicState) -> List[id_T]:
    #     """
    #     After all terms have been computed, solve_permutation places all elements in their
    #     corresponding place.
    #     """
    #     optp = []
    #     stack_idx = len(self._final_stack) - len(cstack) - 1
    #
    #     # First we solve the values in the stack
    #     while stack_idx >= 0:
    #         # The corresponding value must be stored in some local register
    #         x = select_local_from_value(self._final_stack[stack_idx], clocals)
    #         cstack.append(clocals[x])
    #         optp.append(f"local.get[{x}]")
    #         stack_idx -= 1
    #
    #     # Then we detect which locals have a value that appears in flocals and load them onto the stack
    #     outdated_locals = []
    #     for local_idx in range(len(self._final_locals)):
    #         if self._final_locals[local_idx] != clocals[local_idx]:
    #             x = select_local_from_value(self._final_locals[local_idx], clocals)
    #             outdated_locals.append(local_idx)
    #             cstack.append(clocals[x])
    #             optp.append(f"local.get[{x}]")
    #
    #         local_idx += 1
    #
    #     # Finally, we store them in the corresponding local in reversed order
    #     for x in reversed(outdated_locals):
    #         cstack, clocals = set_local_value(x, cstack, clocals)
    #         optp.append(f"local.set[{x}]")
    #
    #     return optp

    def greedy(self):
        cstate: SymbolicState = SymbolicState(self._initial_stack.copy(), self._initial_locals.copy(),
                                              self._var_total_uses)

        # We split into three sets: mops (operations with dependencies), sops (elements that appear in fstack
        # that do not appear in mops) and rops (other operations with no restrictions)
        mops_unsorted, sops, lops = self._mops.copy(), self._sops.copy(), self._lops.copy()
        seq: List[id_T] = []
        mops: List[id_T] = self.select_memory_ops_order(mops_unsorted)
        maximal_vars: List[id_T] = []

        optg: List[id_T] = []

        while mops != [] or sops != [] or lops != []:
            var_top = cstate.top_stack()

            # Top of the stack must be removed, as it appears more time it is being used
            if var_top is not None and cstate.var_uses[var_top] > self._var_total_uses[var_top]:
                cstate.drop()
                optg.append("drop")

            # Top of the stack must be placed in some other position
            elif var_top is not None and self.var_must_be_moved(var_top, cstate):
                optg.extend(self.move_top_to_position(var_top, cstate))

            # Top of the stack cannot be moved to the corresponding position. As there is always the possibility
            # of storing in locals, this means that either the stack is empty or the current top of the stack
            # is already placed in the corresponding position. Hence, we just generate the following computation
            else:
                pass
                # next_computation = self.choose_next_computation(mops, sops, lops, cstack, clocals, clocals_liveness)
                #ops = self.choose_subterm_order(mops, sops, lops, cstack, clocals, clocals_liveness)
                #optg.extend(ops)

        # optg.extend(self.solve_permutation(cstate))


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        sfs = json.load(f)
    SMSgreedy(sfs).greedy()