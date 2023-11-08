"""
Module to generate random sequences of instructions and test the greedy algorithm
"""
import json
import shutil
import sys
from typing import List
from pathlib import Path
import random
import pywasm
import greedy

instruction = pywasm.instruction
binary = pywasm.binary
algorithm = greedy.algorithm
symbolic = pywasm.symbolic_execution

# We only consider i32 ops
# choices = [byte for byte, instr in instruction.opcode_info.items() if instr['type'] != instruction.InstructionType.control
#            and all(t not in instr["name"] for t in ['i64', 'f32', 'f64'])]

def weight(instr_):
    return 3 if "local" in instr_["name"] else 1


pair_choice_weight = [(byte, weight(instr)) for byte, instr in instruction.opcode_info.items() if "indirect" not in instr["name"] and
                      any(instr_name in instr["name"] for instr_name in ["local", "i32.const", "call", "i32.add", "i32.sub", "i32.neg"])]
choices = [byte for byte, _ in pair_choice_weight]

def random_opcode() -> str:
    """
    Generates a random opcode in the plain representation format used by pywasm.binary.Instruction.from_plain_repr.
    Uses this representation instead of directly generating binary.Instruction because we avoid processing calls
    """
    opcode = random.choices(choices,weights=[w for _, w in pair_choice_weight])[0]

    # Call instructions have their own format to be parsed
    if opcode == instruction.call:
        in_ar, out_ar = random.randint(0, 3), random.randint(0, 3)
        return f"call[{in_ar}, {out_ar}]"

    o = binary.Instruction()
    o.opcode = opcode
    o.name = instruction.opcode_info[o.opcode]["name"]
    o.type = instruction.opcode_info[o.opcode]["type"]
    o.in_arity = instruction.opcode_info[o.opcode]["in_ar"]
    o.out_arity = instruction.opcode_info[o.opcode]["out_ar"]
    o.comm = instruction.opcode_info[o.opcode]["comm"]
    o.args = []

    if opcode in [
        instruction.get_local,
        instruction.set_local,
        instruction.tee_local,
    ]:
        o.args = [binary.LocalIndex(random.randint(0, 1))]
    elif opcode in [
        instruction.get_global,
        instruction.set_global,
    ]:
        o.args = [binary.GlobalIndex(random.randint(0, 1))]
    elif opcode in [
        instruction.i32_load,
        instruction.i64_load,
        instruction.f32_load,
        instruction.f64_load,
        instruction.i32_load8_s,
        instruction.i32_load8_u,
        instruction.i32_load16_s,
        instruction.i32_load16_u,
        instruction.i64_load8_s,
        instruction.i64_load8_u,
        instruction.i64_load16_s,
        instruction.i64_load16_u,
        instruction.i64_load32_s,
        instruction.i64_load32_u,
        instruction.i32_store,
        instruction.i64_store,
        instruction.f32_store,
        instruction.f64_store,
        instruction.i32_store8,
        instruction.i32_store16,
        instruction.i64_store8,
        instruction.i64_store16,
        instruction.i64_store32,
    ]:
        o.args = [random.randint(0, 5), random.randint(0, 5)]
    elif o.opcode in [
        instruction.current_memory,
        instruction.grow_memory
    ]:
        o.args = [random.randint(0, 2)]
    elif o.opcode in [instruction.i32_const,
                    instruction.i64_const]:
        o.args = [random.randint(0, 100)]
    elif o.opcode in [instruction.f32_const,
                    instruction.f64_const]:
        o.args = [round(random.random(), 2) * random.randint(0, 10)]
    elif o.opcode not in instruction.opcode:
        raise Exception("unsupported opcode", o.opcode)

    return str(o)


def random_block(n: int) -> List[str]:
    return [random_opcode() for _ in range(n)]


def execute_symbolic_execution_and_greedy(instrs: List[str]) -> None:
    pywasm.symbolic_exec_from_instrs(instrs)

    with open(pywasm.global_params.FINAL_FOLDER.joinpath('isolated.json'), 'r') as f:
        sfs = json.load(f)

    ids = algorithm.SMSgreedy(sfs, False).greedy()
    print(ids)
    if symbolic.check_execution_from_ids(sfs, ids):
        print("Check works!!")


def execute_symbolic_execution_and_encoder(instrs: List[str]) -> None:
    pywasm.symbolic_exec_from_instrs(instrs)


if __name__ == "__main__":
    n = int(sys.argv[1])
    n_examples = int(sys.argv[2])

    root_folder = Path(sys.argv[3])

    for j in range(15, 17):
        final = root_folder.joinpath(f"{j}/")
        final.mkdir(exist_ok=True, parents=True)

        for i in range(n_examples):
            plain_instrs = random_block(j)
            pywasm.global_params.DEBUG_MODE = True # Enable debug mode
            pywasm.global_params.FINAL_FOLDER = final
            print(' '.join(plain_instrs))
            execute_symbolic_execution_and_encoder(plain_instrs)
            shutil.copy(final.joinpath("isolated.json"), final.joinpath(f"{i}.json"))
