"""
Adapted from gasol-optimizer/smt_encoding/json_with_dependencies.py
"""

import typing


def happens_before(current_id: str, prev_id: str, dependency_graph: typing.Dict[str, typing.List[str]]) -> bool:
    """
    Checks whether prev_id belongs to the dependency graph of current_id recursively
    """

    for other_id in dependency_graph[current_id]:
        if other_id == prev_id or happens_before(other_id, prev_id, dependency_graph):
            return True
    return False


def generate_dependency_graph_minimum(uninterpreted_instrs: typing.List[typing.Dict],
                                      order_tuples: typing.List[typing.List[str]]) -> typing.Dict[str, typing.List[str]]:
    # call instructions might generate multiple stack elements, and hence, we have to iterate over 'outpt_sk'
    stack_elem_to_id = {stack_elem: instruction["id"] for instruction in uninterpreted_instrs
                        for stack_elem in instruction['outpt_sk']}

    dependency_graph = {}
    for instr in uninterpreted_instrs:
        instr_id = instr["id"]
        dependency_graph[instr_id] = []

        for stack_elem in instr['inpt_sk']:

            # This means the stack element corresponds to another uninterpreted instruction
            if stack_elem in stack_elem_to_id:
                dependency_graph[instr['id']].append(stack_elem_to_id[stack_elem])

    # We need to consider also the order given by the tuples
    for id1, id2 in order_tuples:
        # Stronger check: if id1 happens before id2 at some point, then we don't consider it in the graph.
        # See test_lb_tight_dependencies in tests/test_instruction_bounds_with_dependencies
        dependency_graph[id2].append(id1)

    return dependency_graph


def immediate_dependencies(uninterpreted_instrs: typing.List[typing.Dict], order_tuples: typing.List[typing.List[str]]):
    # Stores the immediate dependencies
    # call instructions might generate multiple stack elements, and hence, we have to iterate over 'outpt_sk'
    stack_elem_to_id = {stack_elem: instruction["id"] for instruction in uninterpreted_instrs
                        for stack_elem in instruction['outpt_sk']}
    forbidden_ids = set(dep[0] for dep in order_tuples)

    allowed_deps, forbidden_deps = set(), set()
    for instr in uninterpreted_instrs:
        instr_id = instr["id"]

        for stack_elem in instr['inpt_sk']:
            stack_elem_id = stack_elem_to_id.get(stack_elem, None)

            # This means the stack element corresponds to another uninterpreted instruction
            if stack_elem_id is not None:
                if stack_elem_id in forbidden_ids:
                    forbidden_deps.add((stack_elem_id, instr_id))
                else:
                    allowed_deps.add((stack_elem_id, instr_id))

    return [list(dep) for dep in allowed_deps], [list(dep) for dep in forbidden_deps]
