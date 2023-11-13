from typing import List
from .binary import Instruction


def split_instructions(initial_block: List[Instruction], n_instructions: int) -> List[List[Instruction]]:
    # We try to split in memory instructions
    possible_splits = [i+1 for i, instr in enumerate(initial_block[:-1])
                       for instr_name in ["call", "global.set", "store"] if instr_name in instr.name]
    blocks = []
    i = 0
    for j in range(1, len(possible_splits)):
        current_diff = possible_splits[j] - i
        # If there are more instructions than the number allowed, split here
        if current_diff > n_instructions:
            blocks.append(initial_block[i:possible_splits[j-1]])
            i = possible_splits[j-1]

    if i < len(initial_block):
        blocks.append(initial_block[i:len(initial_block)])
    return blocks


def split_simple(initial_block: List[Instruction], n_instructions: int) -> List[List[Instruction]]:
    return [initial_block[i:i + n_instructions] for i in range(0, len(initial_block), n_instructions)]