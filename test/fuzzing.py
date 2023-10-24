"""
Module to generate random sequences of instructions and test the greedy algorithm
"""
import json
import sys
from typing import List
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

choices = [byte for byte, instr in instruction.opcode_info.items() if "local" in instr["name"] or "call" == instr["name"]]

def random_opcode() -> str:
    """
    Generates a random opcode in the plain representation format used by pywasm.binary.Instruction.from_plain_repr.
    Uses this representation instead of directly generating binary.Instruction because we avoid processing calls
    """
    opcode = random.choice(choices)

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
        o.args = [binary.LocalIndex(random.randint(0, 5))]
    elif opcode in [
        instruction.get_global,
        instruction.set_global,
    ]:
        o.args = [binary.GlobalIndex(random.randint(0, 5))]
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
    plain_instrs = random_block(n)
    pywasm.global_params.DEBUG_MODE = True # Enable debug mode
    pywasm.global_params.ALL_EXECUTED = True # Allow executing any block
    print(' '.join(plain_instrs))
    execute_symbolic_execution_and_encoder(plain_instrs)