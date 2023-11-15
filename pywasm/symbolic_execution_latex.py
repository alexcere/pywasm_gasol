import json
from typing import List, Dict, Set, Tuple, Any
import re
import sys

id_T = str
var_T = str
instr_T = Dict[str, Any]

const_re = re.compile('\[(.*)]')
access_re = re.compile('_index\((.*)\)')


def idx_from_access(access: str) -> int:
    return int(re.search(access_re, access).group(1))


def instr_id_to_repr(instr_id, i) -> str:
    pass

def execute_instr(instr_name: str, pos: int, cstack: List[var_T], clocals: Dict[var_T, var_T], ilocals: List[var_T],
                  vars_: Set[var_T], user_instr: List[instr_T]) -> id_T:
    """
    Executes the instruction and returns the id from the instruction according to user_instr
    """
    # Case const: filter the instruction that has introduces that value
    if 'const' in instr_name:
        const = int(re.search(const_re, instr_name).group(1))
        filtered_instrs = [instr for instr in user_instr if f'const' in instr['disasm'] and instr['value'] == const]
        assert len(filtered_instrs) == 1
        instr = filtered_instrs[0]
        const_var = instr['outpt_sk'][0]
        assigned_instr = instr["id"]
        cstack.insert(0, const_var)
        vars_.add(const_var)

    # Drop: just remove the instruction that introduced the value
    elif instr_name == 'drop':
        vars_.add(cstack.pop(0))
        assigned_instr = 'POP'

    # load.get: get the value from the corresponding local
    elif 'local.get' in instr_name:
        # Either it appears as an instruction in user_instr or the local variable belongs to clocals
        local_idx = idx_from_access(instr_name)
        local_name = f"local_{local_idx}"
        local_val = clocals.get(local_name, None)
        if local_val is not None:
            cstack.insert(0, local_val)
            vars_.add(clocals[local_name])
            assigned_instr = f'LGET_{ilocals.index(local_name)}'
        else:
            # Check it exists exactly one instruction for loading
            filtered_instrs = [instr for instr in user_instr if
                               len(instr['outpt_sk']) > 0 and instr['outpt_sk'][0] == local_name]
            assert len(filtered_instrs) == 1
            instr = filtered_instrs[0]
            local_val = instr['outpt_sk'][0]
            assigned_instr = instr["id"]
            cstack.insert(0, local_val)
            vars_.add(local_val)

    # load.set: store the value in the corresponding local
    elif 'local.set' in instr_name or 'local.tee' in instr_name:
        # The value must belong to local variables
        local = idx_from_access(instr_name)
        local_name = f"local_{local}"
        new_local_value = cstack[0]
        clocals[local_name] = new_local_value
        vars_.add(new_local_value)

        # If it is set instead of tee, we remove the value from the stack
        if 'local.set' in instr_name:
            cstack.pop(0)
            assigned_instr = f'LSET_{ilocals.index(local_name)}'
        else:
            assigned_instr = f'LTEE_{ilocals.index(local_name)}'

    # Remaining instructions: filter those instructions whose disasm matches the instr name and consumes the same
    # values. For call and global instructions, we also use the access position to filter the instruction
    else:
        # if 'call' in instr_name:
        #    # Call instructions are of the form call[]_pos(args)
        #    filtered_instrs = [instr for instr in user_instr
        #                        if instr['id'].startswith(f"{instr_name}_{pos}")]
        if any(instr in instr_name for instr in ['call', 'global', 'load', 'store']):
            # Remaining instructions are of the form instr_name(args)_pos
            filtered_instrs = [instr for instr in user_instr
                               if instr['disasm'] in instr_name and instr['id'].endswith(f'_{pos}')]
        else:
            filtered_instrs = [instr for instr in user_instr if instr['disasm'] in instr_name and
                               all(cstack[i] == input_var for i, input_var in enumerate(instr['inpt_sk']))]

        # print(instr_name, pos, *[(instr['id'], instr['disasm']) for instr in user_instr])
        # print(instr_name, pos, *[(instr['id'], instr['disasm']) for instr in filtered_instrs])
        assert len(filtered_instrs) == 1
        instr = filtered_instrs[0]

        # We consume the elements
        for input_var in instr['inpt_sk']:
            assert cstack[0] == input_var
            cstack.pop(0)

        # We introduce the new elements
        for output_var in reversed(instr['outpt_sk']):
            cstack.insert(0, output_var)
            vars_.add(output_var)

        assigned_instr = instr['id']

    return assigned_instr


def extract_idx_from_id(instr_id: str) -> int:
    return int(instr_id.split('_')[-1])


def execute_instr_id(instr_id: str, i: int, cstack: List[var_T], clocals: List[var_T], user_instr: List[instr_T],
                     accesses: List[str]):
    """
    Executes the instr id according to user_instr
    """
    # Drop the value
    if instr_id == 'POP':
        cstack.pop(0)
        return 'POP'

    # load.get: get the value from the corresponding local
    elif 'LGET' in instr_id:
        idx = extract_idx_from_id(instr_id)
        local_val = clocals[idx]
        cstack.insert(0, local_val)
        return f'\GET{{{idx}}}'
    # load.set: store the value in the corresponding local
    elif 'LSET' in instr_id:
        idx = extract_idx_from_id(instr_id)

        if idx >= len(clocals):
            clocals.append('')

        clocals[idx] = cstack.pop(0)
        return f'\SET{{{idx}}}'


    # load.tee: store the value in the corresponding local without consuming the top of the stack
    elif 'LTEE' in instr_id:
        idx = extract_idx_from_id(instr_id)

        if idx >= len(clocals):
            clocals.append('')

        clocals[idx] = cstack[0]
        return f'\TEE{{{idx}}}'

    else:
        instr = [instr for instr in user_instr if instr['id'] == instr_id][0]
        operands = []
        # We consume the elements
        for input_var in instr['inpt_sk']:
            operands.append(str(cstack.pop(0)))

        joined_operands = ','.join(operands)
        if "local.get" in instr["disasm"]:
            idx = extract_idx_from_id(instr["outpt_sk"][0])
            instr_name = f"s_{idx}"
            final_instr_name = f'\GET{{{idx}}}'

        else:
            if "const" in instr["disasm"]:
                instr_name = f"\PUSH{{{instr['value']}}}"
            elif "shl" in instr["disasm"]:
                instr_name = "\ishl"
            elif "call" in instr["disasm"]:
                instr_name = "\LOADz"
            elif "store" in instr["disasm"]:
                instr_name = f"\MSTORE{{{i}}}"
            elif "rem" in instr["disasm"]:
                instr_name = f"\irem"
            else:
                raise ValueError("Not recognized option")
            final_instr_name = instr_name

        if any(other_instr in instr["disasm"] for other_instr in ["load", "call", "store"]):
            accesses.append(final_instr_name)

        if joined_operands == '':
            expression = instr_name
        else:
            expression = f'{instr_name}({joined_operands})'

        if len(instr['outpt_sk']) == 1:
            cstack.insert(0, expression)
        if len(instr['outpt_sk']) > 1:
            raise ValueError("Opcodes with more than one element returned are not supported")

        return final_instr_name

def print_state(instr_id: str, i: int, cstack: List[var_T], clocals: List[var_T], accesses: List[str]):
    print(f"\\rightarrow_{{\\tinycode{{{i}:{instr_id}}}}} \\\\")
    print(f"& ([{','.join(cstack)}],[{','.join(clocals)}],[{','.join(accesses)}])")


def symbolic_execution_from_sfs(sfs: Dict) -> List[id_T]:
    original_instr: str = sfs['original_instrs']
    user_instr: List[instr_T] = sfs['user_instrs']
    print(*(instr["disasm"] for instr in user_instr))
    instrs: List[str] = original_instr.split(' ')
    local_changes: List[Tuple[var_T, var_T]] = sfs['register_changes']
    dependencies: List[Tuple[id_T, id_T]] = sfs['dependencies']
    sfs_vars: Set[str] = set(sfs['vars'])

    # We split into two different dicts the initial values and final values in locals
    ilocals: List[var_T] = [local_repr[0] for local_repr in local_changes]
    clocals: Dict[var_T, var_T] = {local_repr[0]: local_repr[0] for local_repr in local_changes}
    flocals: Dict[var_T, var_T] = {local_repr[0]: local_repr[1] for local_repr in local_changes}
    cstack, fstack = sfs['src_ws'].copy(), sfs['tgt_ws']

    # We include directly the initial values in istack and ilocals
    vars_ = set(clocals.keys())
    vars_.update(cstack)

    final_instr_ids = [execute_instr(instr, i, cstack, clocals, ilocals, vars_, user_instr) for i, instr in
                       enumerate(instrs)]

    # Check that the ids returned generate the final state
    cstack, clocals_list = [f"s_{i}" for i in range(len(sfs["src_ws"]))], [f"s_{i}" for i in range(len(sfs["src_ws"]) + len(clocals))]
    flocal_list = [local_repr[1] for local_repr in local_changes]
    accesses = []

    print_state("", -1, cstack, clocals_list, accesses)
    for i, instr_id in enumerate(final_instr_ids):
        name = execute_instr_id(instr_id, i, cstack, clocals_list, user_instr, accesses)
        print_state(name, i, cstack, clocals_list, accesses)


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        loaded_sfs = json.load(f)
    symbolic_execution_from_sfs(loaded_sfs)
