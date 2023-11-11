"""
Module to process the results from the superoptimization with the SAT encoder (in the future, possibly with minizinc
as well)
"""
import resource
import traceback
from typing import Dict, Tuple, List, Any
from . import symbolic_execution, global_params
from evmopt.evmx.evmx_api import evmx_from_sms
import greedy.algorithm


def extract_idx_from_id(instr_id: str) -> int:
    return int(instr_id.split('_')[-1])


def extract_reg_from_id(instr_id: str, ini_locals: List[str]) -> int:
    """
    Given an instruction and the initial state of the locals, returns which local corresponds to in the inital code.
    """
    # first we determine which register we are
    # accessing, and then check its initial value, as the subindex of the initial value
    #  matches which local it represents in the initial code
    idx = extract_idx_from_id(instr_id)

    # If the idx is greater than the initial number of locals, this means we have to introduce an
    # auxiliary local to use. To represent this situation, we introduce a counter that assigns a
    # new local for the given one
    if idx >= len(ini_locals):
        return idx - len(ini_locals) - 1
    local_reg = extract_idx_from_id(ini_locals[idx])
    return local_reg


def id2disasm(instr_id: str, user_instrs: Dict[str, Dict[str, Any]], ini_locals: List[str]):
    """
    Executes the instr id according to user_instr
    """
    # Drop the value
    if instr_id == 'POP':
        return 'drop'

    # load.get, local.set and local.tee:
    elif 'LGET' in instr_id:
        return f"local.get[local_index({extract_reg_from_id(instr_id, ini_locals)})]"

    elif 'LSET' in instr_id:
        return f"local.set[local_index({extract_reg_from_id(instr_id, ini_locals)})]"

    elif 'LTEE' in instr_id:
        return f"local.tee[local_index({extract_reg_from_id(instr_id, ini_locals)})]"

    # At this point, there must be a corresponding instruction
    user_instr = user_instrs[instr_id]

    # Constants has a value field initialized
    if 'value' in user_instr:
        return f'{user_instr["disasm"]}[{user_instr["value"]}]'

    # Local gets that do not appear as part of local changes need to be reconstructed with the name of the
    # local it contains
    elif 'local.get' in user_instr["disasm"]:
        output_values = user_instr["outpt_sk"]

        # Ensure the representation is indeed correct
        if global_params.DEBUG_MODE:
            assert len(output_values) == 1
            assert output_values[0].startswith("local_")
        return f"local.get[local_index({output_values[0].split('_')[1]})]"
    else:
        return user_instr["disasm"]


def evmx_to_pywasm(sfs: Dict, timeout: float, parsed_args) -> Tuple[List[str], str, float, bool]:
    # There was an error when initializing the optimizer
    id_seq, optimization_outcome, time = evmx_from_sms(sfs, timeout, parsed_args, "wasm")
    instr_id_to_instr = {instr['id']: instr for instr in sfs['user_instrs']}
    ini_locals = [local_repr[0] for local_repr in sfs["register_changes"]]
    if 'optimal' in optimization_outcome:
        is_correct = symbolic_execution.check_execution_from_ids(sfs, [instr_id for instr_id in id_seq if instr_id != "NOP"])
    else:
        is_correct = True

    if global_params.DEBUG_MODE:
        print("Id seq:", id_seq)
    return ([id2disasm(instr_id, instr_id_to_instr, ini_locals) for instr_id in id_seq if instr_id != "NOP"],
            optimization_outcome, time, is_correct)


def greedy_to_pywasm(sfs: Dict, timeout: float, parsed_args) -> Tuple[List[str], str, float, bool]:
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    try:
        id_seq = greedy.algorithm.SMSgreedy(sfs, global_params.DEBUG_MODE).greedy()
        usage_stop = resource.getrusage(resource.RUSAGE_SELF)
        optimization_outcome = "non_optimal"
        is_correct = symbolic_execution.check_execution_from_ids(sfs, [instr_id for instr_id in id_seq if instr_id != "NOP"])

        if global_params.DEBUG_MODE:
            print("Id seq:", id_seq)
    except:
        usage_stop = resource.getrusage(resource.RUSAGE_SELF)
        print("EXEC", sfs["block"])
        traceback.print_exc()
        id_seq = []
        optimization_outcome = "error"
        # We assume if there is an exception, then it is incorrect
        is_correct = False

    instr_id_to_instr = {instr['id']: instr for instr in sfs['user_instrs']}
    ini_locals = [local_repr[0] for local_repr in sfs["register_changes"]]
    return ([id2disasm(instr_id, instr_id_to_instr, ini_locals) for instr_id in id_seq if instr_id != "NOP"],
            optimization_outcome, usage_stop.ru_utime + usage_stop.ru_stime - usage_start.ru_utime - usage_start.ru_stime, is_correct)


def generate_statistics_info(original_block: List[str], optimized_block: List[str], outcome: str, solver_time: float,
                             tout: int, initial_bound: int, used_bound: int, block_name: str,
                             rules_repr: str, is_correct: bool) -> Dict:

    statistics_row = {"block_id": block_name,  "previous_solution": ' '.join(original_block), "timeout": tout,
                      "solver_time_in_sec": round(solver_time, 3), "outcome": outcome,
                      "initial_n_instrs": initial_bound, "model_found": False, "shown_optimal": False,
                      "initial_length": len(original_block), "used_bound": used_bound, "saved_length": 0, "checker": is_correct}

    # The solver has returned a valid model
    if outcome in ["optimal", "non_optimal"]:
        shown_optimal = outcome == "optimal"

        statistics_row.update({"model_found": True, "shown_optimal": shown_optimal,
                               "solution_found": ' '.join(optimized_block),
                               "optimized_n_instrs": len(optimized_block), 'optimized_length': len(optimized_block),
                               'outcome': 'model', 'saved_length': len(original_block) - len(optimized_block),
                               "rules": rules_repr})

    return statistics_row
