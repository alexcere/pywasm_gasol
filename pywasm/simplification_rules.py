"""
Module to include simplification rules
"""
from collections import defaultdict
from . import execution
from . import binary
from . import instruction
from typing import List, Union, Optional, Tuple, Callable, Dict


# ====== Auxiliary functions ======

def value_is_zero(value: 'execution.Value'):
    # First check it is indeed a value (of the four types)
    # It doesn't matter if it's an int or a numpy.floatx in the check
    if value.type in [execution.convention.i32, execution.convention.i64,
                      execution.convention.f32, execution.convention.f64]:
        return value.val() == 0


def value_is_one(value: 'execution.Value'):
    # Same as value_is_zero
    if value.type in [execution.convention.i32, execution.convention.i64,
                      execution.convention.f32, execution.convention.f64]:
        return value.val() == 1

# ====== Simplification rules ======


# === Arithmetic operations ===

def add_x_0(operands: List['execution.Value']):
    if value_is_zero(operands[0]):
        return operands[1]
    elif value_is_zero(operands[1]):
        return operands[0]


def sub_x_0(operands: List['execution.Value']):
    if value_is_zero(operands[1]):
        return operands[1]


def _sub_x_x(num_type: str, operands: List['execution.Value']):
    if operands[0] == operands[1]:
        return execution.Value


def _mul_x_0(num_type: str, operands: List['execution.Value']):
    pass


def mul_x_1(operands: List[Union['execution.Term', 'execution.Value']]):
    if value_is_one(operands[0]):
        return operands[1]
    elif value_is_one(operands[1]):
        return operands[0]


# === Equality operations

def eq_zero(num_type: str, operands: List['execution.Value']):
    """
    Parametric function that applies rule instr.eq(x, 0) -> instr.eqz(x) for instr in [i32, i64, f32, f64].
    The concrete type must be instantiated
    """
    if value_is_zero(operands[0]):
        eqz_instr = binary.Instruction.from_plain_repr(f"{num_type}.eqz", [])
        return execution.Term(eqz_instr, [operands[1]])
    elif value_is_zero(operands[1]):
        eqz_instr = binary.Instruction.from_plain_repr(f"{num_type}.eqz", [])
        return execution.Term(eqz_instr, [operands[0]])


def iszero_iszero_iszero(operands: List['execution.Value']):
    if operands[0].type == binary.convention.term and operands[0].val().instr.opcode == instruction.i32_eqz:
        third_instr = operands[0].val().ops[0]
        print("Third thing", third_instr)

        # Thid instr might be either i32.zero or i64.zero
        if (third_instr.type == binary.convention.term and
                third_instr.val().instr.opcode in [instruction.i32_eqz, instruction.i64_eqz]):
            return third_instr


# === Bitwise operations ===

def and_x_0(operands: List['execution.Value']):
    if value_is_zero(operands[0]):
        return operands[0]
    elif value_is_zero(operands[1]):
        return operands[1]


def and_x_x(operands: List['execution.Value']):
    if operands[0] == operands[1]:
        return operands[0]

# ====== Rule generation ======


# List of rules to apply. Structure: function to apply, instructions over
# which it can be applied, textual representation and how many instructions can be decreased.
# To add a new rule, include a new item following the format
rules = [
    (add_x_0, [instruction.i32_add, instruction.i64_add, instruction.f32_add, instruction.f64_add], "[i,f]xx.add(X,0) -> X", 2),
    (sub_x_0, [instruction.i32_sub, instruction.i64_sub, instruction.f32_sub, instruction.f64_sub], "[i,f]xx.sub(X,0) -> X", 2),

    (mul_x_1, [instruction.i32_mul, instruction.i64_mul, instruction.f32_mul, instruction.f64_mul], "[i,f]xx.mul(X,1) -> X", 2),
    (lambda ops: eq_zero("i32", ops), [instruction.i32_eq], "i32.eq(X,0) -> i32.eqz(X)", 1),
    (lambda ops: eq_zero("i64", ops), [instruction.i64_eq], "i64.eq(X,0) -> i64.eqz(X)", 1),
    (iszero_iszero_iszero, [instruction.i32_eqz], "i32.eqz(i32.eqz(ixx.eqz(X))) -> ixx.eqz(X)", 2),
    (and_x_0, [instruction.i32_and, instruction.i64_and], "ixx.and(X, 0) -> 0", 2),
    (and_x_x, [instruction.i32_and, instruction.i64_and], "ixx.and(X, X) -> X", 2)
]


def process_rules(r: List[Tuple[Callable, List[int], str, int]]) -> Dict[int, List[Tuple[Callable, str]]]:
    """
    Associates each opcode with the list of rules that can be applied
    """
    r_dict = defaultdict(lambda: [])
    for tup in r:
        for opcode in tup[1]:
            r_dict[opcode].append([tup[0], tup[2], tup[3]])
    return r_dict


rule_dict = process_rules(rules)


class SimplificationRules:

    def __init__(self):
        self._rules_applied = []
        self.instructions_saved = 0

    def apply_rule(self, instr: binary.Instruction, operands: List['execution.Value']) -> (
            Optional)[Union['execution.Term', 'execution.Value']]:
        possible_rules = rule_dict[instr.opcode]
        for rule, rule_repr, saved_instrs in possible_rules:
            simplified_term = rule(operands)
            if simplified_term is not None:
                self._rules_applied.append(rule_repr)
                self.instructions_saved += saved_instrs
                return simplified_term

    def rules_applied(self) -> List[str]:
        return self._rules_applied
