import json
import unittest
import pywasm
import greedy

instruction = pywasm.instruction
binary = pywasm.binary
algorithm = greedy.algorithm
symbolic = pywasm.symbolic_execution
global_params = pywasm.global_params


def load_sfs():
    with open(global_params.FINAL_FOLDER.joinpath("isolated.json"), 'r') as f:
        return json.load(f)


class SimplificationRulesTest(unittest.TestCase):
    def test_add_x_0(self):
        instr_sequence = ["i32.const[0]", "i32.add"]
        pywasm.symbolic_exec_from_instrs(instr_sequence)
        sfs = load_sfs()
        print(sfs)
        self.assertEqual(len(sfs["user_instrs"]), 0, "add_x_0 does not work properly")

    def test_mul_x_1(self):
        instr_sequence = ["i64.const[1]", "i64.mul"]
        pywasm.symbolic_exec_from_instrs(instr_sequence)
        sfs = load_sfs()
        self.assertEqual(len(sfs["user_instrs"]), 0, "mul_x_1 does not work properly")

    def test_iszero_iszero_iszero(self):
        instr_sequence = ["i32.eqz", "i32.eqz", "i32.eqz"]
        pywasm.symbolic_exec_from_instrs(instr_sequence)
        sfs = load_sfs()
        self.assertEqual(len(sfs["user_instrs"]), 1, "Chained i32.eqz does not work")

        instr_sequence = ["i64.eqz", "i32.eqz", "i32.eqz"]
        pywasm.symbolic_exec_from_instrs(instr_sequence)
        sfs = load_sfs()
        self.assertEqual(len(sfs["user_instrs"]), 1, "Chained i64.eqz does not work")

    def test_and_x_x(self):
        instr_sequence = ["i64.const[1]", "i64.mul", "local.tee[local_index(0)]",
                          "local.get[local_index(0)]", "i64.and"]
        pywasm.symbolic_exec_from_instrs(instr_sequence)
        sfs = load_sfs()
        self.assertEqual(len(sfs["user_instrs"]), 0, "and_x_x does not work properly")

    def test_div_x_x(self):
        instr_sequence = ["local.tee[local_index(0)]", "local.get[local_index(0)]", "i64.div_s"]
        pywasm.symbolic_exec_from_instrs(instr_sequence)
        sfs = load_sfs()
        self.assertEqual(sfs["user_instrs"][0]["disasm"], "i64.const", "div_x_x with f64.div does not work properly")

        instr_sequence = ["local.tee[local_index(0)]", "local.get[local_index(0)]", "f64.div"]
        pywasm.symbolic_exec_from_instrs(instr_sequence)
        sfs = load_sfs()
        self.assertEqual(sfs["user_instrs"][0]["disasm"], "f64.const", "div_x_x with f64.div does not work properly")


    def test_zero(self):
        i32_zero = pywasm.simplification_rules.zero("i32")
        self.assertEqual(i32_zero.val(), 0, "i32 zero is not 0")

        i64_zero = pywasm.simplification_rules.zero("i64")
        self.assertEqual(i64_zero.val(), 0, "i64 zero is not 0")

        f32_zero = pywasm.simplification_rules.zero("f32")
        self.assertEqual(f32_zero.val(), 0, "f32 zero is not 0")

        f64_zero = pywasm.simplification_rules.zero("f64")
        self.assertEqual(f64_zero.val(), 0, "f64 zero is not 0")

    def test_one(self):
        i32_one = pywasm.simplification_rules.one("i32")
        self.assertEqual(i32_one.val(), 1, "i32 zero is not 0")

        i64_one = pywasm.simplification_rules.one("i64")
        self.assertEqual(i64_one.val(), 1, "i64 zero is not 0")

        f32_one = pywasm.simplification_rules.one("f32")
        self.assertEqual(f32_one.val(), 1, "f32 zero is not 0")

        f64_one = pywasm.simplification_rules.one("f64")
        self.assertEqual(f64_one.val(), 1, "f64 zero is not 0")


if __name__ == '__main__':
    unittest.main()
