"""
Module to include simplification rules.
Note that Wasm evaluates the operands from bottom to top instead of top
to bottom, so rules have to be applied in that order.
For instance, sub(x,y) = y - x, so the rule must be sub(0,x) -> x
"""
from collections import defaultdict
from . import execution
from . import binary
from . import instruction
from typing import List, Union, Optional, Tuple, Callable, Dict
from functools import partial

# IMPORTANT: to instantiate template functions, use partial and not lambda x. It doesn't behave well with iterators

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


def zero(num_type: str):
    # First determine the value type
    if num_type == "i32":
        t = execution.convention.i32
    elif num_type == "i64":
        t = execution.convention.i64
    elif num_type == "f32":
        t = execution.convention.f32
    elif num_type == "f64":
        t = execution.convention.f64
    else:
        raise ValueError("Not valid type for zero value function")

    return execution.Value.new(t, 0)


def one(num_type: str):
    # First determine the value type
    if num_type == "i32":
        t = execution.convention.i32
    elif num_type == "i64":
        t = execution.convention.i64
    elif num_type == "f32":
        t = execution.convention.f32
    elif num_type == "f64":
        t = execution.convention.f64
    else:
        raise ValueError("Not valid type for zero value function")

    return execution.Value.new(t, 1)


# ====== Simplification rules ======


# === Arithmetic operations ===

def add_0_x(operands: List['execution.Value']):
    if value_is_zero(operands[0]):
        return operands[1]
    elif value_is_zero(operands[1]):
        return operands[0]


def sub_0_x(operands: List['execution.Value']):
    """
    Consider the arguments are reversed, according to Wasm specification sub(x,y) = y-x
    """
    if value_is_zero(operands[0]):
        return operands[0]


def sub_x_x(num_type: str, operands: List['execution.Value']):
    if operands[0] == operands[1]:
        return zero(num_type)


def mul_0_x(num_type: str, operands: List['execution.Value']):
    if value_is_zero(operands[0]) or value_is_zero(operands[0]):
        return zero(num_type)


def mul_1_x(operands: List['execution.Value']):
    if value_is_one(operands[0]):
        return operands[1]
    elif value_is_one(operands[1]):
        return operands[0]


def div_1_x(operands: List['execution.Value']):
    if value_is_one(operands[0]):
        return operands[1]


def div_x_x(number_type: str, operands: List['execution.Value']):
    if operands[0] == operands[1]:
        return one(number_type)


def rem_1_x(operands: List['execution.Value']):
    if value_is_one(operands[0]):
        return operands[1]


def rem_x_x(operands: List['execution.Value']):
    if operands[0] == operands[1]:
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
    (add_0_x, [instruction.i32_add, instruction.i64_add, instruction.f32_add, instruction.f64_add], "any.add(0,X) -> X", 2),
    (sub_0_x, [instruction.i32_sub, instruction.i64_sub, instruction.f32_sub, instruction.f64_sub], "any.sub(0,X) -> X", 2),
    *[(partial(sub_x_x, num_type), [instr], f"{num_type}.sub(X,X) -> 0", 2) for num_type, instr in
      [("i32", instruction.i32_sub), ("i64", instruction.i64_sub), ("f32", instruction.f32_sub), ("f64", instruction.f64_sub)]],
    *[(partial(mul_0_x, num_type), [instr], f"{num_type}.mul(0,X) -> 0", 2) for num_type, instr in
      [("i32", instruction.i32_mul), ("i64", instruction.i64_mul), ("f32", instruction.i32_mul), ("f64", instruction.f64_mul)]],
    (mul_1_x, [instruction.i32_mul, instruction.i64_mul, instruction.f32_mul, instruction.f64_mul], "any.mul(1,X) -> X", 2),
    (div_1_x, [instruction.i32_divs, instruction.i32_divu, instruction.i64_divs, instruction.i64_divu,
               instruction.f32_div, instruction.f64_div], "any.div[s,u](1,X) -> X", 2),
    *[(partial(div_x_x, num_type), instrs, f"{num_type}.div[s,u](X,X) -> 0", 2) for num_type, instrs in
      [("i32", [instruction.i32_divs, instruction.i32_divu]), ("i64", [instruction.i64_divs, instruction.i64_divu]),
       ("f32", [instruction.f32_div]), ("f64", [instruction.f64_div])]],
    (partial(eq_zero, "i32"), [instruction.i32_eq], "i32.eq(X,0) -> i32.eqz(X)", 1),
    (partial(eq_zero,"i64"), [instruction.i64_eq], "i64.eq(X,0) -> i64.eqz(X)", 1),
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
