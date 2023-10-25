import typing


def add_node(term: str, term_to_id: typing.Dict, color: str, f):
    new_idx = len(term_to_id)
    f.write(f"n_{new_idx} [color={color},label=\"{term}\"];\n")
    term_to_id[term] = new_idx


def add_edge(term1: str, term2: str, term_to_id: typing.Dict, label: str, f):
    f.write(f"n_{term_to_id[term1]} -> n_{term_to_id[term2]} [label=\"{label}\"];\n")


def add_consecutive_term(terms: typing.List[str], term_to_id: typing.Dict, label:str, f):
    for i in range(len(terms) - 1):
        add_edge(terms[i], terms[i+1], term_to_id, label, f)


def dot_for_term(term: str, user_instr: typing.List[typing.Dict], term_to_id: typing.Dict,
                 stack_var_to_id: typing.Dict, f):
    if term in term_to_id:
        return
    filtered_instr = [instr for instr in user_instr if instr["id"] == term]
    current_instr = filtered_instr[0] if len(filtered_instr) > 0 else None
    if current_instr is None:
        add_node(term, term_to_id, "red", f)
    else:
        add_node(term, term_to_id, "blue" if len(current_instr["outpt_sk"]) > 0 else "green", f)

        children = [stack_var_to_id.get(child, child) for child in current_instr['inpt_sk']]
        for child in children:
            dot_for_term(child, user_instr, term_to_id, stack_var_to_id, f)
            add_edge(term, child, term_to_id, "child", f)

        # If the instruction is not commutative, add a edge to represent the fixed order among elements
        if not current_instr["commutative"]:
            add_consecutive_term(list(reversed(children)), term_to_id, "bef", f)


# Given the blocks corresponding to the CFG of a program, and the string containing the input program,
# generates a graphical representation of the CFG as a .dot file.
def generate_CFG_dot(sfs_json: typing.Dict, dot_file_name):
    with open(dot_file_name, 'w') as f:
        f.write("digraph G {\n")
        final_stack = sfs_json["tgt_ws"]
        initial_stack = sfs_json["src_ws"]
        user_instrs = sfs_json["user_instrs"]
        local_changes = sfs_json["register_changes"]
        mem_deps = sfs_json["dependencies"]

        stack_var_to_id = {output_var: instr['id'] for instr in user_instrs for output_var in instr['outpt_sk']}
        term_to_id = {}

        # Add first initial terms
        for term in initial_stack:
            dot_for_term(term, user_instrs, term_to_id, stack_var_to_id, f)

        # Then name for locals
        for local, term in local_changes:
            dot_for_term(local, user_instrs, term_to_id, stack_var_to_id, f)

        add_node("tg_stk", term_to_id, "black", f)
        for term in final_stack:
            if term not in term_to_id:
                dot_for_term(stack_var_to_id.get(term, term), user_instrs, term_to_id, stack_var_to_id, f)
            add_edge("tg_stk", stack_var_to_id.get(term, term), term_to_id, "in", f)

        store_instrs = [instr for instr in user_instrs if len(instr['outpt_sk']) == 0 or 'call' in instr['id']]
        for store_instr in store_instrs:
            dot_for_term(store_instr['id'], user_instrs, term_to_id, stack_var_to_id, f)

        for local, term in local_changes:
            final_id = stack_var_to_id.get(term, None)
            if final_id is not None and final_id not in term_to_id:
                dot_for_term(stack_var_to_id[term], user_instrs, term_to_id, stack_var_to_id, f)
            add_edge(local, stack_var_to_id.get(term, term), term_to_id, "in", f)
            dot_for_term(stack_var_to_id.get(term, term), user_instrs, term_to_id, stack_var_to_id, f)

        for acc1, acc2 in mem_deps:
            add_edge(acc2, acc1, term_to_id, "dep", f)

        # for acc1, acc2 in local_deps:
        #     # Avoid repeating accesses for set and tee
        #     if "tee" in acc2:
        #         continue
        #
        #     # We need to filter which variable is accessed. acc2 is a set or tee access, and hence
        #     # we don't include it in our representation
        #     local_name = [instr["value"] for instr in user_instrs if instr["id"] == acc1][0]
        #     add_edge(local_name, acc1, term_to_id, "loc_dep", f)

        f.write("}\n")

