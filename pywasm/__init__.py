import json
import typing
import pandas as pd

from . import binary
from . import convention
from . import execution
from . import instruction
from . import leb128
from . import log
from . import num
from . import option
from . import validation
from . import symbolic_execution
from . import global_params


class Runtime:
    # A webassembly runtime manages Store, stack, and other runtime structure. They forming the WebAssembly abstract.

    def __init__(self, module: binary.Module, imps: typing.Dict = None, opts: typing.Optional[option.Option] = None):
        self.machine = execution.Machine()
        self.machine.opts = opts or option.Option()

        # For compatibility with older 0.4.x versions
        self.store = self.machine.store

        imps = imps if imps else {}
        extern_value_list: typing.List[execution.ExternValue] = []
        for e in module.import_list:
            if e.module not in imps or e.name not in imps[e.module]:
                # Remove exception to allow not including imports. Only works with symbolic execution
                pass
            if isinstance(e.desc, binary.TypeIndex):
                # If we try to import a function not defined in imps, we just return the id function
                a = execution.HostFunc(module.type_list[e.desc], imps.get(e.module, {e.name: lambda x: x})[e.name])
                addr = self.machine.store.allocate_host_function(a)
                extern_value_list.append(addr)
                continue
            if isinstance(e.desc, binary.TableType):
                addr = execution.TableAddress(len(self.store.table_list))
                table = imps[e.module][e.name]
                self.store.table_list.append(table)
                extern_value_list.append(addr)
                continue
            if isinstance(e.desc, binary.MemoryType):
                addr = execution.MemoryAddress(len(self.store.memory_list))
                memory = imps[e.module][e.name]
                if self.machine.opts.pages_limit > 0 and e.desc.limits.n > self.machine.opts.pages_limit:
                    raise Exception('pywasm: out of memory limit')
                memory.grow(e.desc.limits.n)
                self.store.memory_list.append(memory)
                extern_value_list.append(addr)
                continue
            if isinstance(e.desc, binary.GlobalType):
                addr = self.store.allocate_global(
                    e.desc,
                    execution.Value.new(e.desc.value_type, imps[e.module][e.name])
                )
                extern_value_list.append(addr)
                continue

        self.machine.instantiate(module, extern_value_list)

    def func_addr(self, name: str) -> execution.FunctionAddress:
        # Get a function address denoted by the function name
        for e in self.machine.module.export_list:
            if e.name == name and isinstance(e.value, execution.FunctionAddress):
                return e.value
        raise Exception('pywasm: function not found')

    def exec_accu(self, name: str, args: typing.List[execution.Value]) -> execution.Result:
        # Invoke a function denoted by the function address with the provided arguments.
        func_addr = self.func_addr(name)
        return self.machine.invocate(func_addr, args)

    def exec(self, name: str, args: typing.List[typing.Union[int, float]]) -> execution.Result:
        func_addr = self.func_addr(name)
        func = self.machine.store.function_list[func_addr]
        func_args = []
        # Mapping check for python valtype to webAssembly valtype
        for i, e in enumerate(func.type.args.data):
            v = execution.Value.new(e, args[i])
            func_args.append(v)
        resp = self.exec_accu(name, func_args)
        if len(resp.data) == 0:
            return None
        if len(resp.data) == 1:
            return resp.data[0].val()
        return [e.val() for e in resp.data]

    def symbolic_exec_func(self, func: typing.Union[execution.WasmFunc, execution.HostFunc], func_address: int):
        func_args = []
        # Mapping check for python valtype to webAssembly valtype
        for i, e in enumerate(func.type.args.data):
            v = execution.Value.new(convention.symbolic, f"in_{i}")
            func_args.append(v)
        return self.machine.invocate_symbolic(func, func_args, func_address)

    def symbolic_exec(self, name: str):
        func_addr = self.func_addr(name)
        func = self.machine.store.function_list[func_addr]
        self.symbolic_exec_func(func, func_addr)

    def all_symbolic_exec(self):
        csv_rows = []
        for func_addr, func in enumerate(self.machine.store.function_list):
            print(f"{func_addr} {func.type}")
            if isinstance(func, execution.WasmFunc):
                csv_rows.extend(self.symbolic_exec_func(func, func_addr))
        # Finally, we store the results in a csv file
        pd.DataFrame(csv_rows).to_csv(global_params.CSV_FILE)


# Using the pywasm API.
# If you have already compiled a module from another language using tools like Emscripten, or loaded and run the code
# by Javascript yourself, the pywasm API is easy to learn.

def on_debug():
    log.lvl = 1


def load(name: str, imps: typing.Dict = None, opts: typing.Optional[option.Option] = None) -> Runtime:
    # Generate a runtime directly by loading a file from disk.
    with open(name, 'rb') as f:
        module = binary.Module.from_reader(f)
        return Runtime(module, imps, opts)


def symbolic_exec_from_instrs(plain_instrs: typing.List[str]):
    function_addresses = []
    instrs = [binary.Instruction.from_plain_repr(plain_instr, function_addresses) for plain_instr in plain_instrs]

    # We retrieve the maximum index accessed from the locals and globals
    local_args = [instr.args[0] for instr in instrs if 'local' in instr.name]
    n_locals = max(local_args) + 1 if len(local_args) > 0 else 0

    global_args = [instr.args[0] for instr in instrs if 'global' in instr.name]
    n_globals = max(global_args) + 1 if len(global_args) > 0 else 0
    execution.symbolic_execution_from_instrs(instrs, function_addresses, n_locals, n_globals)


def symbolic_exec_from_block(file_name: str):
    # Generate a runtime directly by loading a file from disk.
    with open(file_name, 'r') as f:
        plain_instrs = f.read().split()
        symbolic_exec_from_instrs(plain_instrs)


def symbolic_exec_from_sfs(file_name: str):
    # Generate a runtime directly by loading a file from disk.
    with open(file_name, 'r') as f:
        sfs = json.load(f)
        csv_info = execution.superopt_from_json(sfs, file_name.split(".")[0], 10)

        with open(global_params.CSV_FILE, 'w') as f:
            json.dump(csv_info, f)


Store = execution.Store
Memory = execution.MemoryInstance
Value = execution.Value
Table = execution.TableInstance
Global = execution.GlobalInstance
Limits = binary.Limits
FunctionAddress = execution.FunctionAddress
TableAddress = execution.TableAddress
MemoryAddress = execution.MemoryAddress
GlobalAddress = execution.GlobalAddress
Option = option.Option

# For compatibility with older 0.4.x versions
Ctx = Store
