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


def execute_instr(instr_name: str, pos: int, cstack: List[var_T], clocals: Dict[var_T, var_T],
                  vars_: Set[var_T], user_instr: List[instr_T]) -> id_T:
    """
    Executes the instruction and returns the id from the instruction according to user_instr
    """
    assigned_instr = None
    # Case const: filter the instruction that has introduces that value
    if 'const' in instr_name:
        const = int(re.search(const_re, instr_name).group(1))
        filtered_instrs = [instr for instr in user_instr if f'const' in instr['disasm'] and instr['value'] == const]
        assert len(filtered_instrs) == 1
        assigned_instr = filtered_instrs[0]
        const_var = assigned_instr['outpt_sk'][0]
        cstack.insert(0, const_var)
        vars_.add(const_var)

    # Drop: just remove the instruction that introduced the value
    elif instr_name == 'drop':
        vars_.add(cstack.pop(0))

    # load.get: get the value from the corresponding local
    elif 'local.get' in instr_name:
        # Either it appears as an instruction in user_instr or the local variable belongs to clocals
        local = idx_from_access(instr_name)
        local_name = f"local_{local}"
        local_val = clocals.get(local_name, None)
        if local_name is not None:
            cstack.insert(0, clocals[local_name])
            vars_.add(clocals[local_name])
        else:
            # Check it exists exactly one instruction for loading
            filtered_instr = [instr for instr in user_instr if instr['oupt_sk'] == local_val]
            assert len(filtered_instr) == 1
            assigned_instr = filtered_instr[0]
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

    # Remaining instructions: filter those instructions whose disasm matches the instr name and consumes the same
    # values. For call and global instructions, we also use the access position to filter the instruction
    else:
        if any(instr in instr_name for instr in ['call', 'global', 'load', 'store']):
            filtered_instrs = [instr for instr in user_instr
                               if instr['disasm'] in instr_name and instr['id'].endswith(f'_{pos}')]
        else:
            filtered_instrs = [instr for instr in user_instr if instr['disasm'] in instr_name and
                               all(cstack[i] == input_var for i, input_var in enumerate(instr['inpt_sk']))]

        # print(instr_name, pos, *[(instr['id'], instr['disasm']) for instr in user_instr])
        # print(instr_name, pos, *[(instr['id'], instr['disasm']) for instr in filtered_instrs])
        assert len(filtered_instrs) == 1
        assigned_instr = filtered_instrs[0]

        # We consume the elements
        for input_var in assigned_instr['inpt_sk']:
            assert cstack[0] == input_var
            cstack.pop(0)

        # We introduce the new elements
        for output_var in assigned_instr['outpt_sk']:
            cstack.insert(0, output_var)
            vars_.add(output_var)

    # If assigned_instr has a not null value, then it returns the id associated.
    # Otherwise, it just returns the instr_name
    return assigned_instr['id'] if assigned_instr is not None else instr_name


def symbolic_execution_from_sfs(sfs: Dict) -> List[id_T]:
    original_instr: str = sfs['original_instrs']
    id2instr: List[instr_T] = sfs['user_instrs']
    instrs: List[str] = original_instr.split(' ')
    local_changes: List[Tuple[var_T, var_T]] = sfs['register_changes']
    sfs_vars: Set[str] = set(sfs['vars'])

    # We split into two different dicts the initial values and final values in locals
    clocals: Dict[var_T, var_T] = {local_repr[0]: local_repr[0] for local_repr in local_changes}
    flocals: Dict[var_T, var_T] = {local_repr[0]: local_repr[1] for local_repr in local_changes}
    cstack, fstack = sfs['src_ws'], sfs['tgt_ws']

    # We include directly the initial values in istack and ilocals
    vars_ = set(clocals.keys())
    vars_.update(cstack)

    final_instr_ids = [execute_instr(instr, i, cstack, clocals, vars_, id2instr) for i, instr in enumerate(instrs)]

    assert cstack == fstack, 'Stack do not match'
    assert clocals == flocals, 'Locals do not match'
    assert vars_ == sfs_vars, 'Vars do not match'
    print("They match!")
    print(final_instr_ids)
    return final_instr_ids


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        loaded_sfs = json.load(f)
    symbolic_execution_from_sfs(loaded_sfs)
