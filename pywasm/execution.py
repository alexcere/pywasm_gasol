import collections
import copy
import json
import typing

import numpy
import networkx as nx

from . import binary
from . import convention
from . import instruction
from . import log
from . import num
from . import option
from . import global_params
from . import dataflow_dot
from . import dependencies

import sys
# ======================================================================================================================
# Execution Runtime Structure
# ======================================================================================================================


class Value:
    # Values are represented by themselves.
    def __init__(self):
        self.type: binary.ValueType
        self.data: typing.Union[bytearray, num.sym, 'Term'] = bytearray(8)

    def __repr__(self):
        return f'{self.type} {self.val()}'

    def __str__(self):
        return f'{self.val()}'

    @classmethod
    def new(cls, type: binary.ValueType, data: typing.Union[int, float, str, 'Term']):
        return {
            convention.i32: Value.from_i32,
            convention.i64: Value.from_i64,
            convention.f32: lambda x: Value.from_f32(num.f32(x)),
            convention.f64: lambda x: Value.from_f64(num.f64(x)),
            convention.symbolic: Value.from_symbolic,
            convention.term: Value.from_term,
        }[type](data)

    @classmethod
    def raw(cls, type: binary.ValueType, data: bytearray):
        o = Value()
        o.type = type
        o.data = data
        return o

    def val(self) -> typing.Union[num.i32, num.i64, num.f32, num.f64]:
        return {
            convention.i32: self.i32,
            convention.i64: self.i64,
            convention.f32: self.f32,
            convention.f64: self.f64,
            convention.symbolic: self.sym,
            convention.term: self.term,
        }[self.type]()

    def opcode(self):
        op = self.data.instr.opcode if self.type == convention.term else None
        return {
            convention.i32: 0x41,
            convention.i64: 0x42,
            convention.f32: 0x43,
            convention.f64: 0x44,
            convention.symbolic: None,
            convention.term: op,
        }[self.type]


    def i32(self) -> num.i32:
        return num.LittleEndian.i32(self.data[0:4])

    def i64(self) -> num.i64:
        return num.LittleEndian.i64(self.data[0:8])

    def u32(self) -> num.u32:
        return num.LittleEndian.u32(self.data[0:4])

    def u64(self) -> num.u64:
        return num.LittleEndian.u64(self.data[0:8])

    def f32(self) -> num.f32:
        return num.LittleEndian.f32(self.data[0:4])

    def f64(self) -> num.f64:
        return num.LittleEndian.f64(self.data[0:8])

    def sym(self) -> num.sym:
        return self.data

    def term(self) -> 'Term':
        return self.data

    @classmethod
    def from_i32(cls, n: num.i32):
        o = Value()
        o.type = binary.ValueType(convention.i32)
        o.data[0:4] = num.LittleEndian.pack_i32(num.int2i32(n))
        return o

    @classmethod
    def from_i64(cls, n: num.i64):
        o = Value()
        o.type = binary.ValueType(convention.i64)
        o.data[0:8] = num.LittleEndian.pack_i64(num.int2i64(n))
        return o

    @classmethod
    def from_u32(cls, n: num.u32):
        o = Value()
        o.type = binary.ValueType(convention.i32)
        o.data[0:4] = num.LittleEndian.pack_u32(num.int2u32(n))
        return o

    @classmethod
    def from_u64(cls, n: num.u64):
        o = Value()
        o.type = binary.ValueType(convention.i64)
        o.data[0:8] = num.LittleEndian.pack_u64(num.int2u64(n))
        return o

    @classmethod
    def from_f32(cls, n: num.f32):
        assert isinstance(n, num.f32)
        o = Value()
        o.type = binary.ValueType(convention.f32)
        o.data[0:4] = num.LittleEndian.pack_f32(n)
        return o

    @classmethod
    def from_f32_u32(cls, n: num.u32):
        o = Value.from_u32(n)
        o.type = binary.ValueType(convention.f32)
        return o

    @classmethod
    def from_f64(cls, n: num.f64):
        assert isinstance(n, num.f64)
        o = Value()
        o.type = binary.ValueType(convention.f64)
        o.data[0:8] = num.LittleEndian.pack_f64(n)
        return o

    @classmethod
    def from_f64_u64(cls, n: num.u64):
        o = Value.from_u64(n)
        o.type = binary.ValueType(convention.f64)
        return o

    @classmethod
    def from_symbolic(cls, n: num.sym):
        o = Value()
        o.type = binary.ValueType(convention.symbolic)
        o.data = n
        return o

    @classmethod
    def from_term(cls, t: 'Term'):
        o = Value()
        o.type = binary.ValueType(convention.term)
        o.data = t
        return o


def term_to_string(instr: binary.Instruction, operands: typing.List[typing.Union['Term', Value]],
                   sub_index: typing.Optional[int] = None) -> str:
    """
    String that represents a term unequivocally

    :param instr: instruction to represent
    :param operands: list of terms/values that are the operands of the operation. Assume __str__ is defined
    :param sub_index: the subindex is used to identify different subterms if a term represents different values.
        For instance, if call(a,b) returns two values in the stack, then we could create terms call_1(a,b) and call_2(a,b)
    :return: the str representation
    """
    if instr.comm:
        joined_operands = f"({','.join(sorted(map(str, operands)))})" if operands else ""
    else:
        joined_operands = f"({','.join(map(str, operands))})" if operands else ""

    sub_index = f"_{sub_index}" if sub_index is not None else ""
    return f"{instr}{sub_index}{joined_operands}"


class Term:
    # A term consists of either a value or an instruction applied to other terms.

    def __init__(self, instr: binary.Instruction, operands: typing.List[typing.Union['Term', Value]],
                 sub_index: typing.Optional[int] = None, term_repr: str = None):
        # We allow passing the representation as an argument to allow repeating computations
        self.instr: binary.Instruction = instr
        self.ops: typing.List[typing.Union['Term', Value]] = operands
        self.sub_index: int = sub_index
        self.repr: str = term_to_string(instr, operands, sub_index) if term_repr is None else term_repr

        # Sub index is used to identify different values that are generated from the same instruction.
        # super_term is the representation of the general term
        self.super_term: str = self.repr if sub_index is None else term_to_string(instr, operands, None)

    def __str__(self):
        return self.repr

    def __repr__(self):
        return self.repr


class TermFactory:
    """
    Factory to create terms. So far, it only avoids creating repeated terms
    """
    def __init__(self):
        self._terms_created: typing.Dict = {}

    def term(self, instr: binary.Instruction, operands: typing.List[typing.Union['Term', Value]],
             sub_index: typing.Optional[int] = None) -> typing.Tuple['Term', str]:
        term_repr = term_to_string(instr, operands, sub_index)
        if term_repr in self._terms_created:
            return self._terms_created[term_repr], term_repr
        new_term = Term(instr, operands, sub_index, term_repr)
        self._terms_created[term_repr] = new_term
        return new_term, term_repr

    def created(self) -> typing.Dict:
        return self._terms_created


class StackVarFactory:
    """
    Class to generate the stack vars associated to terms
    """

    def __init__(self):
        self._value2var: typing.Dict[str, typing.List[str]] = {}
        self._cont = 0
        self._accessed: typing.Set[str] = set()

    def stack_var(self, term_repr: str) -> typing.List[str]:
        """
        Returns the stack var associated to term_repr, assuming if it corresponds to a term then it has been
        assigned previously. Also marks an access has been made
        """
        self._accessed.add(term_repr)
        stack_repr = self._value2var.get(term_repr, None)
        if stack_repr is None:
            self._value2var[term_repr] = [term_repr]
            return [term_repr]
        return stack_repr

    def assign_stack_var(self, term_repr: str) -> str:
        """
        Assigns an arbitrary stack var to term_repr. Checks if there are repetitions
        """
        var_term = self._value2var.get(term_repr, None)
        if var_term is None:
            var_term = [f"s({self._cont})"]
            self._value2var[term_repr] = var_term
            self._cont += 1
        return var_term[0]

    def has_been_accessed(self, term_repr: str) -> bool:
        return term_repr in self._accessed

    def set_term(self, term_repr: str, var_list: typing.List[str]) -> None:
        """
        Assigns var_list to term_repr term
        """
        self._value2var[term_repr] = var_list

    def vars(self) -> typing.List[str]:
        return [self._value2var[term_repr][0] for term_repr in self._accessed if len(self._value2var[term_repr]) == 1]


class Result:
    # A result is the outcome of a computation. It is either a sequence of values or a trap.
    def __init__(self, data: typing.List[Value]):
        self.data = data

    def __repr__(self):
        return self.data.__repr__()


class FunctionAddress(int):
    def __repr__(self):
        return f'FunctionAddress({super().__repr__()})'


class TableAddress(int):
    def __repr__(self):
        return f'TableAddress({super().__repr__()})'


class MemoryAddress(int):
    def __repr__(self):
        return f'MemoryAddress({super().__repr__()})'


class GlobalAddress(int):
    def __repr__(self):
        return f'GlobalAddress({super().__repr__()})'


class ModuleInstance:
    # A module instance is the runtime representation of a module. It is created by instantiating a module, and
    # collects runtime representations of all entities that are imported, defined, or exported by the module.
    #
    # moduleinst ::= {
    #     types functype∗
    #     funcaddrs funcaddr∗
    #     tableaddrs tableaddr∗
    #     memaddrs memaddr∗
    #     globaladdrs globaladdr∗
    #     exports exportinst∗
    # }
    def __init__(self):
        self.type_list: typing.List[binary.FunctionType] = []
        self.function_addr_list: typing.List[FunctionAddress] = []
        self.table_addr_list: typing.List[TableAddress] = []
        self.memory_addr_list: typing.List[MemoryAddress] = []
        self.global_addr_list: typing.List[GlobalAddress] = []
        self.export_list: typing.List[ExportInstance] = []


class WasmFunc:
    def __init__(self, function_type: binary.FunctionType, module: ModuleInstance, code: binary.Function):
        self.type = function_type
        self.module = module
        self.code = code

    def __repr__(self):
        return f'wasm_func({self.type})'


class HostFunc:
    # A host function is a function expressed outside WebAssembly but passed to a module as an import. The definition
    # and behavior of host functions are outside the scope of this specification. For the purpose of this
    # specification, it is assumed that when invoked, a host function behaves non-deterministically, but within certain
    # constraints that ensure the integrity of the runtime.
    def __init__(self, function_type: binary.FunctionType, hostcode: typing.Callable):
        self.type = function_type
        self.hostcode = hostcode

    def __repr__(self):
        return self.hostcode.__name__


# A function instance is the runtime representation of a function. It effectively is a closure of the original
# function over the runtime module instance of its originating module. The module instance is used to resolve
# references to other definitions during execution of the function.
#
# funcinst ::= {type functype,module moduleinst,code func}
#            | {type functype,hostcode hostfunc}
# hostfunc ::= ...
FunctionInstance = typing.Union[WasmFunc, HostFunc]


class TableInstance:
    # A table instance is the runtime representation of a table. It holds a vector of function elements and an optional
    # maximum size, if one was specified in the table type at the table’s definition site.
    #
    # Each function element is either empty, representing an uninitialized table entry, or a function address. Function
    # elements can be mutated through the execution of an element segment or by external means provided by the embedder.
    #
    # tableinst ::= {elem vec(funcelem), max u32?}
    # funcelem ::= funcaddr?
    #
    # It is an invariant of the semantics that the length of the element vector never exceeds the maximum size, if
    # present.
    def __init__(self, element_type: int, limits: binary.Limits):
        self.element_type = element_type
        self.element_list: typing.List[typing.Optional[FunctionAddress]] = [None for _ in range(limits.n)]
        self.limits = limits


class MemoryInstance:
    # A memory instance is the runtime representation of a linear memory. It holds a vector of bytes and an optional
    # maximum size, if one was specified at the definition site of the memory.
    #
    # meminst ::= {data vec(byte), max u32?}
    #
    # The length of the vector always is a multiple of the WebAssembly page size, which is defined to be the constant
    # 65536 – abbreviated 64Ki. Like in a memory type, the maximum size in a memory instance is given in units of this
    # page size.
    #
    # The bytes can be mutated through memory instructions, the execution of a data segment, or by external means
    # provided by the embedder.
    #
    # It is an invariant of the semantics that the length of the byte vector, divided by page size, never exceeds the
    # maximum size, if present.
    def __init__(self, type: binary.MemoryType):
        self.type = type
        self.data = bytearray()
        self.size = 0
        self.grow(type.limits.n)

    def grow(self, n: int):
        if self.type.limits.m and self.size + n > self.type.limits.m:
            raise Exception('pywasm: out of memory limit')
        # If len is larger than 2**16, then fail
        if self.size + n > convention.memory_page:
            raise Exception('pywasm: out of memory limit')
        self.data.extend([0x00 for _ in range(n * convention.memory_page_size)])
        self.size += n


class GlobalInstance:
    # A global instance is the runtime representation of a global variable. It holds an individual value and a flag
    # indicating whether it is mutable.
    #
    # globalinst ::= {value val, mut mut}
    #
    # The value of mutable globals can be mutated through variable instructions or by external means provided by the
    # embedder.
    def __init__(self, value: Value, mut: binary.Mut):
        self.value = value
        self.mut = mut


# An external value is the runtime representation of an entity that can be imported or exported. It is an address
# denoting either a function instance, table instance, memory instance, or global instances in the shared store.
#
# externval ::= func funcaddr
#             | table tableaddr
#             | mem memaddr
#             | global globaladdr
ExternValue = typing.Union[FunctionAddress, TableAddress, MemoryAddress, GlobalAddress]


class Store:
    # The store represents all global state that can be manipulated by WebAssembly programs. It consists of the runtime
    # representation of all instances of functions, tables, memories, and globals that have been allocated during the
    # life time of the abstract machine
    # Syntactically, the store is defined as a record listing the existing instances of each category:
    # store ::= {
    #     funcs funcinst∗
    #     tables tableinst∗
    #     mems meminst∗
    #     globals globalinst∗
    # }
    #
    # Addresses are dynamic, globally unique references to runtime objects, in contrast to indices, which are static,
    # module-local references to their original definitions. A memory address memaddr denotes the abstract address of
    # a memory instance in the store, not an offset inside a memory instance.
    def __init__(self):
        self.function_list: typing.List[FunctionInstance] = []
        self.table_list: typing.List[TableInstance] = []
        self.memory_list: typing.List[MemoryInstance] = []
        self.global_list: typing.List[GlobalInstance] = []

        # For compatibility with older 0.4.x versions
        self.mems = self.memory_list

    def allocate_wasm_function(self, module: ModuleInstance, function: binary.Function) -> FunctionAddress:
        function_address = FunctionAddress(len(self.function_list))
        function_type = module.type_list[function.type_index]
        wasmfunc = WasmFunc(function_type, module, function)
        self.function_list.append(wasmfunc)
        return function_address

    def allocate_host_function(self, hostfunc: HostFunc) -> FunctionAddress:
        function_address = FunctionAddress(len(self.function_list))
        self.function_list.append(hostfunc)
        return function_address

    def allocate_table(self, table_type: binary.TableType) -> TableAddress:
        table_address = TableAddress(len(self.table_list))
        table_instance = TableInstance(convention.funcref, table_type.limits)
        self.table_list.append(table_instance)
        return table_address

    def allocate_memory(self, memory_type: binary.MemoryType) -> MemoryAddress:
        memory_address = MemoryAddress(len(self.memory_list))
        memory_instance = MemoryInstance(memory_type)
        self.memory_list.append(memory_instance)
        return memory_address

    def allocate_global(self, global_type: binary.GlobalType, value: Value) -> GlobalAddress:
        global_address = GlobalAddress(len(self.global_list))
        global_instance = GlobalInstance(value, global_type.mut)
        self.global_list.append(global_instance)
        return global_address


class ExportInstance:
    # An export instance is the runtime representation of an export. It defines the export's name and the associated
    # external value.
    #
    # exportinst ::= {name name, value externval}
    def __init__(self, name: str, value: ExternValue):
        self.name = name
        self.value = value

    def __repr__(self):
        return f'export_instance({self.name}, {self.value})'


class Label:
    # Labels carry an argument arity n and their associated branch target, which is expressed syntactically as an
    # instruction sequence:
    #
    # label ::= labeln{instr∗}
    #
    # Intuitively, instr∗ is the continuation to execute when the branch is taken, in place of the original control
    # construct.
    def __init__(self, arity: int, continuation: int):
        self.arity = arity
        self.continuation = continuation

    def __repr__(self):
        return f'label({self.arity})'


class Frame:
    # Activation frames carry the return arity n of the respective function, hold the values of its locals
    # (including arguments) in the order corresponding to their static local indices, and a reference to the function's
    # own module instance.

    def __init__(self, module: ModuleInstance,
                 local_list: typing.List[Value],
                 expr: binary.Expression,
                 arity: int):
        self.module = module
        self.local_list = local_list
        self.expr = expr
        self.arity = arity

    def __repr__(self):
        return f'frame({self.arity}, {self.local_list})'


class Stack:
    # Besides the store, most instructions interact with an implicit stack. The stack contains three kinds of entries:
    #
    # Values: the operands of instructions.
    # Labels: active structured control instructions that can be targeted by branches.
    # Activations: the call frames of active function calls.
    #
    # These entries can occur on the stack in any order during the execution of a program. Stack entries are described
    # by abstract syntax as follows.
    def __init__(self):
        self.data: typing.List[typing.Union[Value, Label, Frame]] = []

    def len(self):
        return len(self.data)

    def append(self, v: typing.Union[Value, Label, Frame]):
        self.data.append(v)

    def pop(self):
        return self.data.pop()


# ======================================================================================================================
# Execution Runtime Import Matching
# ======================================================================================================================


def match_limits(a: binary.Limits, b: binary.Limits) -> bool:
    if a.n >= b.n:
        if b.m == 0:
            return 1
        if a.m != 0 and b.m != 0:
            if a.m < b.m:
                return 1
    return 0


def match_function(a: binary.FunctionType, b: binary.FunctionType) -> bool:
    return a.args.data == b.args.data and a.rets.data == b.rets.data


def match_memory(a: binary.MemoryType, b: binary.MemoryType) -> bool:
    return match_limits(a.limits, b.limits)


# ======================================================================================================================
# Abstract Machine
# ======================================================================================================================

class Configuration:
    # A configuration consists of the current store and an executing thread.
    # A thread is a computation over instructions that operates relative to a current frame referring to the module
    # instance in which the computation runs, i.e., where the current function originates from.
    #
    # config ::= store;thread
    # thread ::= frame;instr∗
    def __init__(self, store: Store):
        self.store = store
        self.frame: typing.Optional[Frame] = None
        self.stack = Stack()
        self.depth = 0
        self.pc = 0
        self.opts: option.Option = option.Option()

    def get_label(self, i: int) -> Label:
        l = self.stack.len()
        x = i
        for a in range(l):
            j = l - a - 1
            v = self.stack.data[j]
            if isinstance(v, Label):
                if x == 0:
                    return v
                x -= 1

    def set_frame(self, frame: Frame):
        self.frame = frame
        self.stack.append(frame)
        self.stack.append(Label(frame.arity, len(frame.expr.data) - 1))

    def call(self, function_addr: FunctionAddress, function_args: typing.List[Value]) -> Result:
        function = self.store.function_list[function_addr]
        log.debugln(f'call {function}({function_args})')
        for e, t in zip(function_args, function.type.args.data):
            assert e.type == t
        assert len(function.type.rets.data) < 2

        if isinstance(function, WasmFunc):
            local_list = [Value.new(e, 0) for e in function.code.local_list]
            frame = Frame(
                module=function.module,
                local_list=function_args + local_list,
                expr=function.code.expr,
                arity=len(function.type.rets.data),
            )
            self.set_frame(frame)
            return self.exec()
        if isinstance(function, HostFunc):
            r = function.hostcode(self.store, *[e.val() for e in function_args])
            l = len(function.type.rets.data)
            if l == 0:
                return Result([])
            if l == 1:
                return Result([Value.new(function.type.rets.data[0], r)])
            return [Value.new(e, r[i]) for i, e in enumerate(function.type.rets.data)]
        raise Exception(f'pywasm: unknown function type: {function}')

    def exec(self):
        instruction_list = self.frame.expr.data
        instruction_list_len = len(instruction_list)
        while self.pc < instruction_list_len:
            i = instruction_list[self.pc]
            if self.opts.cycle_limit > 0:
                c = self.opts.cycle + self.opts.instruction_cycle_func(i)
                if c > self.opts.cycle_limit:
                    raise Exception(f'pywasm: out of cycles')
                self.opts.cycle = c
            ArithmeticLogicUnit.exec(self, i)
            self.pc += 1
        r = [self.stack.pop() for _ in range(self.frame.arity)][::-1]
        l = self.stack.pop()
        assert l == self.frame
        return Result(r)


class AbstractConfiguration:
    # A configuration consists of the current store and an executing thread.
    # A thread is a computation over instructions that operates relative to a current frame referring to the module
    # instance in which the computation runs, i.e., where the current function originates from.
    #
    # config ::= store;thread
    # thread ::= frame;instr∗
    def __init__(self, store: Store):
        self.initial_store = store
        self.store = None
        self.frame = None
        self.stack = None
        self.depth = None
        self.pc = None
        self.opts = None
        self.max_stack_size = None
        self.global_count: typing.Optional[int] = None
        self.term_factory: typing.Optional[TermFactory] = None
        self.term2var: typing.Optional[StackVarFactory] = None

    def init_stack_size(self, block: typing.List[binary.Instruction]):
        current_stack = 0
        init_stack = 0
        max_stack = 0

        for instr in block:

            if instr.name == "call":
                instr = ArithmeticLogicUnit.instr_from_call(self, instr)
            elif instr.name == "call_indirect":
                instr = ArithmeticLogicUnit.instr_from_call(self, instr)
            else:
                pass

            consumed_elements = instr.in_arity
            produced_elements = instr.out_arity

            if consumed_elements > current_stack:
                diff = consumed_elements - current_stack
                init_stack += diff
                current_stack = current_stack + diff - consumed_elements + produced_elements
            else:
                current_stack = current_stack - consumed_elements + produced_elements
            max_stack = max(current_stack, max_stack)

        return init_stack, max_stack

    def update_store(self) -> None:
        """
        Initializes a new set of global values after call and stores it in the store global list. It is used after
        simulating symbolic calls, as we don't know the exact contents of globals afterwards
        """
        symbolic_globals = [global_value if not global_value.mut else
                            GlobalInstance(Value.new(convention.symbolic, f"global_{self.global_count}_{i}"), global_value.mut)
                            for i, global_value in enumerate(self.store.global_list)]
        self.global_count += 1
        self.store.global_list = symbolic_globals

    def initialize_store(self):
        self.global_count = 1
        self.store = copy.deepcopy(self.initial_store)

        # Change store global values by generic values
        # Non-mutable values remain the same as initially
        symbolic_globals = [global_value if not global_value.mut else
                            GlobalInstance(Value.new(convention.symbolic, f"global_{i}"), global_value.mut)
                            for i, global_value in enumerate(self.store.global_list)]
        self.store.global_list = symbolic_globals

    def initialize_stack(self, block: typing.List[binary.Instruction]):
        stack_size, max_stack_size = self.init_stack_size(block)
        stack = Stack()
        initial_values = [Value.new(convention.symbolic, f"in({i})") for i in range(stack_size)]
        for val in initial_values:
            stack.append(val)
        self.stack = stack
        self.max_stack_size = max_stack_size + 5

    def initialize_block(self, block: typing.List[binary.Instruction]):
        self.initialize_store()
        self.initialize_stack(block)
        self.depth = 0
        self.pc = 0
        self.term_factory = TermFactory()
        self.opts: option.Option = option.Option()
        self.term2var = StackVarFactory()

    def get_label(self, i: int) -> Label:
        l = self.stack.len()
        x = i
        for a in range(l):
            j = l - a - 1
            v = self.stack.data[j]
            if isinstance(v, Label):
                if x == 0:
                    return v
                x -= 1

    def set_frame(self, frame: Frame):
        self.frame = frame

    def call_symbolic(self, function: typing.Union[HostFunc, WasmFunc], function_args: typing.List[Value], func_address: int):
        log.debugln(f'call {function}({function_args})')
        # for e, t in zip(function_args, function.type.args.data):
        #     assert e.type == t
        assert len(function.type.rets.data) < 2

        if isinstance(function, WasmFunc):
            # Group function args and local list as a single list of "local_{i}" symbolic values
            local_list = [Value.new(convention.symbolic, f"local_{i}") for i in range(len(function_args) + len(function.code.local_list))]
            frame = Frame(
                module=function.module,
                local_list=local_list,
                expr=function.code.expr,
                arity=len(function.type.rets.data),
            )
            self.set_frame(frame)
            self.exec_symbolic(function, function_args, func_address)
            return
        if isinstance(function, HostFunc):
            raise Exception(f'pywasm_gasol: host function not allowed to be executed')
        raise Exception(f'pywasm: unknown function type: {function}')

    def exec_symbolic(self, function: typing.Union[HostFunc, WasmFunc], function_args: typing.List[Value], func_address: int):

        blocks = binary.Expression.blocks_from_instructions(self.frame.expr.data)
        for i, block in enumerate(blocks):
            # Group function args and local list as a single list of "local_{i}" symbolic values
            local_list = [Value.new(convention.symbolic, f"local_{i}") for i in range(len(function_args) + len(function.code.local_list))]
            frame = Frame(
                module=function.module,
                local_list=local_list,
                expr=function.code.expr,
                arity=len(function.type.rets.data),
            )
            self.set_frame(frame)
            print(f"Analyzing block {i}")
            self.exec_symbolic_block(block, f"function_{func_address}_block_{i}")

    def print_block(self, stack, memory_accesses, var_accesses, call_accesses):
        print(f"Stack:")
        print('\n'.join(str(elem) for elem in stack.data[::-1]))
        print("")
        print("Memory access:")
        print('\n'.join([f'({idx}, {str(term)})' for idx, term in memory_accesses]))
        print("")
        print("Variable access:")
        print('\n'.join([f'({idx}, {str(term)})' for idx, term in var_accesses]))
        print("")
        print("Call access:")
        print('\n'.join([f'({idx}, {str(term)})' for idx, term in call_accesses]))
        print("")
        print("Global access:")
        print('\n'.join([f'({idx}, {str(global_instance.value)})' for idx, global_instance in enumerate(self.store.global_list)]))
        print("")
        print("Local access:")
        print('\n'.join([f'({idx}, {str(local_instance)})' for idx, local_instance in enumerate(self.frame.local_list)]))
        print("")

    def exec_symbolic_block(self, block: typing.List[binary.Instruction], block_name: str):
        # Remove labels
        basic_block = [instr for instr in block if instr.opcode not in instruction.beginning_basic_block_instrs and
                       instr.opcode not in instruction.end_basic_block_instrs]
        # if len(basic_block) > 300:
        #     print("Big!")
        #     return
        self.initialize_block(basic_block)

        memory_accesses = []
        var_accesses = []
        call_accesses = []
        # states = []
        initial_stack = [str(elem) for elem in self.stack.data[::-1]]
        initial_locals = copy.deepcopy(self.frame.local_list)

        for i, instr in enumerate(basic_block):
            # states.append([i, instr, copy.deepcopy(self.stack), copy.deepcopy(memory_accesses),
            #               copy.deepcopy(var_accesses), copy.deepcopy(call_accesses)])
            # print(i, instr, self.term_factory.created())
            ArithmeticLogicUnit.exec_symbolic(self, instr, i, memory_accesses, var_accesses, call_accesses)

        final_locals = self.frame.local_list
        global_accesses = [var_access for var_access in var_accesses if "global" in var_access[1].instr.name]

        if interesting_block(var_accesses):
            print("")
            print("Instructions:")
            print('\n'.join([str(i) for i in basic_block]))
            print("")
            print("")
            json_sat = sfs_with_local_changes(initial_stack, self.stack, memory_accesses, global_accesses, call_accesses,
                                              basic_block, initial_locals, final_locals, self.max_stack_size, self.term2var)
            # json_sat["instr_dependencies"] = dependencies.generate_dependency_graph_minimum(json_sat["user_instrs"], json_sat["dependencies"])
            store_json(json_sat, block_name)

            # dataflow_dot.generate_CFG_dot(json_sat, global_params.FINAL_FOLDER.joinpath(f"{block_name}.dot"))

            print(f"Final state:")
            self.print_block(self.stack, memory_accesses, var_accesses, call_accesses)


def interesting_block(var_accesses: typing.List[typing.Tuple[int, Term]]) -> bool:
    """
    Blocks are interesting when a variable is that is used is then stored using local.set (instead of local.tee)
    """
    values_args = set()
    for idx, term in var_accesses:
        current_args = term.instr.args
        if len(current_args) > 0:
            assert len(current_args) == 1
            current_arg = current_args[0]
            if current_arg in values_args:
                return True
            elif term.instr.name == "local.set":
                values_args.add(current_arg)
    return False

# ======================================================================================================================
# Instruction Set
# ======================================================================================================================


def symbolic_func(config: AbstractConfiguration, i: binary.Instruction) -> Term:

    # First we remove from the stack the elements that have been consumed
    operands = [config.stack.pop() for _ in range(i.in_arity)]

    result, result_repr = config.term_factory.term(i, operands)
    ar = i.out_arity
    # Then we introduce the values in the stack
    # If ar > 1, then we create a term per introduced value in the stack
    if i.out_arity > 1:
        output_sk = []
        while ar > 0:
            # We introduce a stack var for each subterm and then the list of stack vars for the super term
            subterm, subterm_repr = config.term_factory.term(i, operands, ar)
            term_value = Value.from_term(subterm)
            config.stack.append(term_value)
            output_sk.insert(0, config.term2var.assign_stack_var(subterm_repr))
            ar -= 1
        # Assign the list of super terms
        config.term2var.set_term(result_repr, list(reversed(output_sk)))

    elif i.out_arity == 1:
        config.stack.append(Value.from_term(result))
        config.term2var.assign_stack_var(result_repr)
    return result


def term_from_func(config: AbstractConfiguration, i: binary.Instruction) -> Term:
    # First we remove from the stack the elements that have been consumed
    operands = [elem for elem in config.stack.data[-1:-i.in_arity - 1:-1]]
    result, _ = config.term_factory.term(i, operands)
    return result


def introduce_term(term: Term, current_ops: typing.Dict, new_index_per_instr: typing.Dict,
                   stack_var_factory: StackVarFactory) -> str:
    # First we obtain the stack vars associated to all input values
    input_values = [instruction_from_value(input_term, current_ops, new_index_per_instr, stack_var_factory)
                    for input_term in term.ops]
    opcode_name = term.instr.name
    term_repr = term.repr
    term_vars = stack_var_factory.stack_var(term_repr)
    term_info = {"id": f"{opcode_name}_{new_index_per_instr[opcode_name]}", "disasm": opcode_name,
                 "opcode": hex(term.instr.opcode)[2:], "inpt_sk": input_values,
                 "outpt_sk": term_vars, 'push': False, "commutative": term.instr.comm,
                 'storage': any(instr in opcode_name for instr in ["call", "store"]), 'gas': 1, 'size': 1}
    current_ops[term_repr] = term_info
    new_index_per_instr[opcode_name] += 1
    return term_vars[0]


def introduce_constant(opcode: int, current_ops: typing.Dict, new_index_per_instr: typing.Dict, constant,
                       stack_var_factory: StackVarFactory) -> str:
    opcode_info = instruction.opcode_info[opcode]
    term_repr = str(constant)
    term_var = stack_var_factory.assign_stack_var(term_repr)
    term_info = {"id": f"PUSH_{new_index_per_instr['PUSH']}", "disasm": opcode_info["name"],
                 "opcode": hex(opcode)[2:], "inpt_sk": [], "outpt_sk": [term_var], 'push': True,
                 "commutative": False, 'storage': False, 'value': constant, 'gas': 1, 'size': 1}
    current_ops[term_repr] = term_info
    new_index_per_instr['PUSH'] += 1
    return term_var


# Change from the function above: no instruction is generated for loading locals
def instruction_from_value(val: Value, current_ops: typing.Dict, new_index_per_instr: typing.Dict,
                           stack_var_factory: StackVarFactory):
    value = val.val()
    value_rep = str(value)
    # If it is a subterm from a term, we just return the first stack var and don't introduce any op.
    # IMPORTANT: these terms are guaranteed to correspond exactly to one stack var
    if stack_var_factory.has_been_accessed(value_rep) or (type(val) == Term and value.instr.sub_index is not None):
        return stack_var_factory.stack_var(value_rep)[0]
    else:
        if val.type == convention.term:
            return introduce_term(value, current_ops, new_index_per_instr, stack_var_factory)
        elif val.type == convention.symbolic:
            # For symbolic, we just returns the corresponding stack var (it only has one)
            return stack_var_factory.stack_var(value_rep)[0]
        else:
            opcode = val.opcode()
            return introduce_constant(opcode, current_ops, new_index_per_instr, value, stack_var_factory)


def introduce_local_register(var_name: str, current_ops: typing.Dict, new_index_per_instr: typing.Dict) -> str:
    opcode = 0x20
    opcode_info = instruction.opcode_info[opcode]
    opcode_name = opcode_info["name"]
    opcode_info = instruction.opcode_info[opcode]
    term_info = {"id": f"{opcode_name}_{new_index_per_instr[opcode_name]}", "disasm": opcode_info["name"],
                 "opcode": hex(opcode)[2:], "inpt_sk": [], "outpt_sk": [var_name], 'push': False,
                 "commutative": False, 'storage': False, 'gas': 1, 'size': 1}
    current_ops[var_name] = term_info
    new_index_per_instr[opcode_name] += 1
    return var_name


def access_representation(access: int, term: Term) -> str:
    return f"{term}_{access}"


def introduce_access(term: Term, access: int, current_ops: typing.Dict, new_index_per_instr: typing.Dict,
                     stack_var_factory: StackVarFactory) -> None:
    """
    Assumes the stack_var_factory already contains all the stack vars associated to its subterms.
    Also, to avoid two accesses with the same arguments to be assigned only once, we add the position in the sequence in
    which it was accessed
    """
    # First we obtain the stack vars associated to all input values. They are guaranteed to have only one term
    input_values = [instruction_from_value(input_term, current_ops, new_index_per_instr, stack_var_factory)
                    for input_term in term.ops]
    opcode_name = term.instr.name
    term_repr = term.repr
    outpt_sk = stack_var_factory.stack_var(term_repr)
    term_info = {"id": f"{opcode_name}_{access}", "disasm": opcode_name,
                 "opcode": hex(term.instr.opcode)[2:], "inpt_sk": input_values,
                 "outpt_sk": outpt_sk, 'push': False, "commutative": term.instr.comm,
                 'storage': any(instr in opcode_name for instr in ["call", "store"]), 'gas': 1, 'size': 1}
    current_ops[access_representation(access, term)] = term_info
    new_index_per_instr[opcode_name] += 1


def operands_from_stack(stack: Stack, current_ops: typing.Dict, new_index_per_instr: typing.Dict,
                        stack_var_factory: StackVarFactory):
    symbolic_stack = []
    for val in stack.data[::-1]:
        stack_term = instruction_from_value(val, current_ops, new_index_per_instr, stack_var_factory)
        symbolic_stack.append(stack_term)
    return symbolic_stack


def operands_from_accesses(accesses: typing.List[typing.Tuple[int, Term]], current_ops: typing.Dict,
                           new_index_per_instr: typing.Dict, stack_var_factory: StackVarFactory):
    for pos, val in accesses:
        introduce_access(val, pos, current_ops, new_index_per_instr, stack_var_factory)


def set_instruction(arg_num: int):
    o = binary.Instruction()
    o.opcode = 0x21
    o.name = instruction.opcode_info[o.opcode]["name"]
    o.type = instruction.opcode_info[o.opcode]["type"]
    o.in_arity = instruction.opcode_info[o.opcode]["in_ar"]
    o.out_arity = instruction.opcode_info[o.opcode]["out_ar"]
    o.comm = instruction.opcode_info[o.opcode]["comm"]
    o.args = [binary.LocalIndex(arg_num)]
    return o


def tee_instruction(arg_num: int):
    o = binary.Instruction()
    o.opcode = 0x22
    o.name = instruction.opcode_info[o.opcode]["name"]
    o.type = instruction.opcode_info[o.opcode]["type"]
    o.in_arity = instruction.opcode_info[o.opcode]["in_ar"]
    o.out_arity = instruction.opcode_info[o.opcode]["out_ar"]
    o.comm = instruction.opcode_info[o.opcode]["comm"]
    o.args = [binary.LocalIndex(arg_num)]
    return o


def get_instruction(arg_num: int):
    o = binary.Instruction()
    o.opcode = 0x20
    o.name = instruction.opcode_info[o.opcode]["name"]
    o.type = instruction.opcode_info[o.opcode]["type"]
    o.in_arity = instruction.opcode_info[o.opcode]["in_ar"]
    o.out_arity = instruction.opcode_info[o.opcode]["out_ar"]
    o.comm = instruction.opcode_info[o.opcode]["comm"]
    o.args = [binary.LocalIndex(arg_num)]
    return o


def mem_access_range(mem_access: Term):
    effective_address = mem_access.ops[0].val() + mem_access.instr.args[0] if mem_access.ops[0].type != convention.term and mem_access.ops[0].type != convention.symbolic else (mem_access.ops[0].val(), mem_access.instr.args[0])
    instr_name = mem_access.instr.name
    if instr_name == "i32.load8_s" or instr_name == "i32.load8_u" or instr_name == "i64.load8_s" or instr_name == "i64.load8_u"\
            or instr_name == "i32.store8" or instr_name == "i64.store8":
        offset = 1
    elif instr_name == "i32.load16_s" or instr_name == "i32.load16_u" or instr_name == "i64.load16_s" or instr_name == "i64.load16_u" \
            or instr_name == "i32.store16" or instr_name == "i64.store16":
        offset = 2
    elif instr_name == "i32.load" or instr_name == "f32.load" or instr_name == "i64.load32_s" or instr_name == "i64.load32_u" \
            or instr_name == "i32.store" or instr_name == "f32.store" or instr_name == "i64.store32":
        offset = 4
    elif instr_name == "i64.load" or instr_name == "f64.load" or instr_name == "i64.store" or instr_name == "f64.store":
        offset = 8
    else:
        raise ValueError(f"{instr_name} not recognized in mem_access_range")
    return [effective_address, offset]


def overlap_address(add1: int, off1: int, add2: int, off2: int) -> bool:
    return add1 <= add2 < add1 + off1 or add2 <= add1 < add2 + off2


def are_dependent_mem_access(mem_access1: Term, mem_access2: Term):
    if "load" in mem_access1.instr.name and "load" in mem_access2.instr.name:
        return False
    # We don't know if call instructions are modifying the memory,
    # so we assume it is dependent with the remaining instructions
    elif "call" in mem_access1.instr.name or "call" in mem_access2.instr.name:
        return True
    elif "grow" in mem_access1.instr.name or "grow" in mem_access2.instr.name:
        # We assume growing the memory is dependent with the remaining instructions
        return True
    elif "size" in mem_access1.instr.name or "size" in mem_access2.instr.name:
        # As neither instruction can be "grow", size is not dependent with any other instruction
        return False

    effective1, offset1 = mem_access_range(mem_access1)
    effective2, offset2 = mem_access_range(mem_access2)

    if type(effective1) == int and type(effective2) == int:
        return overlap_address(effective1, offset1, effective2, offset2)
    elif type(effective1) == tuple and type(effective2) == tuple:
        # As they share the same symbolic value, we don't care the exact address
        if effective1[0] == effective2[0]:
            return overlap_address(effective1[1], offset1, effective2[1], offset2)

    # One constant and one symbolic access are assumed to be dependent
    return True


def are_dependent_var_access(var_access1: Term, var_access2: Term):
    return ("call" in var_access1.instr.name or "call" in var_access2.instr.name) or \
        (("get" not in var_access1.instr.name or "get" not in var_access2.instr.name)
         and var_access1.instr.args[0] == var_access2.instr.args[0])


def simplify_dependencies(deps: typing.List[typing.Tuple[int, int]]) -> typing.List[typing.Tuple[int, int]]:
    dg = nx.DiGraph(deps)
    tr = nx.transitive_reduction(dg)
    return list(tr.edges)


def deps_from_var_accesses(var_accesses: typing.List[typing.Tuple[int, Term]], current_ops: typing.Dict) -> typing.List[typing.Tuple[str, str]]:
    dependencies = [(i, j+i+1) for i, (_, var_access1) in enumerate(var_accesses)
                    for j, (_, var_access2) in enumerate(var_accesses[i+1:])
                    if are_dependent_var_access(var_access1, var_access2)]
    return [(current_ops[access_representation(*var_accesses[i])]['id'], current_ops[access_representation(*var_accesses[j])]['id'])
            for i, j in simplify_dependencies(dependencies)]


def deps_from_mem_accesses(mem_accesses: typing.List[typing.Tuple[int, Term]], current_ops: typing.Dict) -> typing.List[typing.Tuple[str, str]]:
    dependencies = [(i, j+i+1) for i, (_, mem_access1) in enumerate(mem_accesses)
                    for j, (_, mem_access2) in enumerate(mem_accesses[i+1:])
                    if are_dependent_mem_access(mem_access1, mem_access2)]
    return [(current_ops[access_representation(*mem_accesses[i])]['id'], current_ops[access_representation(*mem_accesses[j])]['id'])
            for i, j in simplify_dependencies(dependencies)]


def state_from_local_variables(initial_locals: typing.List[Value], final_locals: typing.List[Value], current_ops: typing.Dict,
                               new_index_per_instr: typing.Dict, stack_var_factory: StackVarFactory):
    """
    Returns a list of tuples representing the local variables. Each tuple contains the initial variable and final
    variable of a local, and only those locals in which they differ are included.
    """
    modified_locals = []
    for i, (ini_local, final_local) in enumerate(zip(initial_locals, final_locals)):
        ini_local_repr, final_local_repr = str(ini_local), str(final_local)
        if ini_local_repr != final_local_repr:
            final_local_var = instruction_from_value(final_local, current_ops, new_index_per_instr, stack_var_factory)

            # Final local could contain a value in the init stack or a value from other local. Hence, it may not have
            # an instruction associated
            modified_locals.append((stack_var_factory.stack_var(ini_local_repr)[0], final_local_var))
    return modified_locals


def local_loads(initial_locals: typing.List[Value], final_locals: typing.List[Value], current_ops: typing.Dict,
                new_index_per_instr: typing.Dict, stack_var_factory: StackVarFactory):
    """
    Introduces a load instruction for each local that is accessed but not modified
    """
    for i, (ini_local, final_local) in enumerate(zip(initial_locals, final_locals)):
        ini_local_repr, final_local_repr = str(ini_local), str(final_local)
        if ini_local_repr == final_local_repr and stack_var_factory.has_been_accessed(ini_local_repr):
            introduce_local_register(final_local_repr, current_ops, new_index_per_instr)


def deps_from_modified_locals(initial_locals: typing.List[Value], final_locals: typing.List[Value], current_ops: typing.Dict):
    return [(current_ops[str(ini_local)]['id'], current_ops[str(final_local)]['id'])
            for ini_local, final_local in zip(initial_locals, final_locals)
            if str(ini_local) != str(final_local) and str(ini_local) in current_ops]


def initial_length(instrs: typing.List[binary.Instruction]):
    return len(instrs)


def sfs_with_local_changes(initial_stack: typing.List[str], final_stack: Stack, memory_accesses: typing.List[typing.Tuple[int, Term]],
                           global_accesses: typing.List[typing.Tuple[int, Term]],
                           call_accesses: typing.List[typing.Tuple[int, Term]], instrs: typing.List[binary.Instruction],
                           initial_locals: typing.List[Value], final_locals: typing.List[Value], max_sk_sz: int,
                           stack_var_factory: StackVarFactory) -> typing.Dict:
    current_ops = {}
    new_index_per_instr = collections.defaultdict(lambda: 0)

    # We access each value in the initial stack to add them to the list of stack variables that are accessed
    for stack_var in initial_stack:
        stack_var_factory.stack_var(stack_var)

    # First we extract values from call instructions to ensure they are created when dealing with its subterms
    tgt_stack = operands_from_stack(final_stack, current_ops, new_index_per_instr, stack_var_factory)

    # TODO: filter repeated accesses
    operands_from_accesses(call_accesses, current_ops, new_index_per_instr, stack_var_factory)
    operands_from_accesses(memory_accesses, current_ops, new_index_per_instr, stack_var_factory)
    operands_from_accesses(global_accesses, current_ops, new_index_per_instr, stack_var_factory)

    combined_accesses = sorted([*memory_accesses, *call_accesses], key=lambda kv: kv[0])
    mem_deps = deps_from_mem_accesses(combined_accesses, current_ops)

    combined_accesses = sorted([*global_accesses, *call_accesses], key=lambda kv: kv[0])
    global_deps = [(acc1, acc2) for acc1, acc2 in deps_from_var_accesses(combined_accesses, current_ops)
                   if "call" not in acc1 or "call" not in acc2]

    local_changes = state_from_local_variables(initial_locals, final_locals, current_ops, new_index_per_instr,
                                               stack_var_factory)
    local_loads(initial_locals, final_locals, current_ops, new_index_per_instr, stack_var_factory)

    # Include vars from the instructions, initial stack and initial locals
    used_vars = stack_var_factory.vars()
    b0 = initial_length(instrs)
    bs = max_sk_sz
    # Number of elements in current ops
    n_locals = len(current_ops)
    sfs = {'init_progr_len': b0, 'vars': list(used_vars), 'max_sk_sz': bs, "src_ws": initial_stack, "tgt_ws": tgt_stack,
           "user_instrs": list(current_ops.values()), 'dependencies': [*mem_deps, *global_deps],
           'original_instrs': ' '.join((str(instr) for instr in instrs)), 'max_registers_sz': n_locals,
           'register_changes': local_changes}
    return sfs


def store_json(sfs_json: typing.Dict, block_name: str) -> None:
    with open(global_params.FINAL_FOLDER.joinpath(f"{block_name}.json"), 'w') as f:
        json.dump(sfs_json, f)


class ArithmeticLogicUnit:

    @staticmethod
    def exec(config: Configuration, i: binary.Instruction):
        if log.lvl > 0:
            log.println('|', i)
        func = _INSTRUCTION_TABLE[i.opcode]
        func(config, i)


    @staticmethod
    def exec_symbolic(config: AbstractConfiguration, i: binary.Instruction, idx: int,
                      memory_accesses: typing.List[typing.Tuple[int, Term]],
                      var_accesses: typing.List[typing.Tuple[int, Term]],
                      call_accesses: typing.List[typing.Tuple[int, Term]]):
        # Exec symbolic: execute if the concrete instruction if the args are concrete.
        # Otherwise, consume the corresponding elements and annotate the mem access
        # (globals, locals or the linear memory)

        if log.lvl > 0:
            log.println('|', i)

        func = _INSTRUCTION_TABLE[i.opcode]

        if i.type == instruction.InstructionType.control:
            # Control instructions include nop and call.
            # Call indirect is not allowed because the types are only known in execution time
            if i.opcode == instruction.nop:
                pass
            elif i.opcode == instruction.call:
                call_expr = ArithmeticLogicUnit.call_symbolic(config, i)
                call_accesses.append((idx, call_expr))
            # elif i.opcode == instruction.call_indirect:
            #     call_expr = ArithmeticLogicUnit.call_indirect_symbolic(config, i)
            #     call_accesses.append(f"{call_expr}_{idx}")

        elif i.type == instruction.InstructionType.variable:
            var_expr = term_from_func(config, i)
            # Apply the symbolic function
            func(config, i)
            var_accesses.append((idx, var_expr))

        elif i.type == instruction.InstructionType.memory:
            mem_expr = symbolic_func(config, i)
            memory_accesses.append((idx, mem_expr))
        else:
            # Either numeric or parametric instruction. We only try executing if all the values in the stack
            # are not symbolic
            if any(val.type == convention.symbolic or val.type == convention.term for val in config.stack.data[-1:-i.in_arity-1:-1]):
                symbolic_func(config, i)
            else:
                func(config, i)


    @staticmethod
    def unreachable(config: Configuration, i: binary.Instruction):
        raise Exception('pywasm: unreachable')

    @staticmethod
    def nop(config: Configuration, i: binary.Instruction):
        pass

    @staticmethod
    def block(config: Configuration, i: binary.Instruction):
        if i.args[0] == convention.empty:
            arity = 0
        else:
            arity = 1
        continuation = config.frame.expr.position[config.pc][1]
        config.stack.append(Label(arity, continuation))

    @staticmethod
    def loop(config: Configuration, i: binary.Instruction):
        if i.args[0] == convention.empty:
            arity = 0
        else:
            arity = 1
        continuation = config.frame.expr.position[config.pc][0]
        config.stack.append(Label(arity, continuation))

    @staticmethod
    def if_(config: Configuration, i: binary.Instruction):
        c = config.stack.pop().i32()
        if i.args[0] == convention.empty:
            arity = 0
        else:
            arity = 1
        continuation = config.frame.expr.position[config.pc][1]
        config.stack.append(Label(arity, continuation))
        if c == 0:
            if len(config.frame.expr.position[config.pc]) == 3:
                config.pc = config.frame.expr.position[config.pc][2]
            else:
                config.pc = config.frame.expr.position[config.pc][1]
                config.stack.pop()

    @staticmethod
    def else_(config: Configuration, i: binary.Instruction):
        L = config.get_label(0)
        v = [config.stack.pop() for _ in range(L.arity)][::-1]
        while True:
            if isinstance(config.stack.pop(), Label):
                break
        for e in v:
            config.stack.append(e)
        config.pc = config.frame.expr.position[config.pc][1]

    @staticmethod
    def end(config: Configuration, i: binary.Instruction):
        L = config.get_label(0)
        v = [config.stack.pop() for _ in range(L.arity)][::-1]
        while True:
            if isinstance(config.stack.pop(), Label):
                break
        for e in v:
            config.stack.append(e)

    @staticmethod
    def br_label(config: Configuration, l: int):
        # Let L be the l-th label appearing on the stack, starting from the top and counting from zero.
        L = config.get_label(l)
        # Pop the values n from the stack.
        v = [config.stack.pop() for _ in range(L.arity)][::-1]
        # Repeat l+1 times
        #     While the top of the stack is a value, pop the value from the stack
        #     Assert: due to validation, the top of the stack now is a label
        #     Pop the label from the stack
        s = 0
        # For loops we keep the label which represents the loop on the stack since the continuation of a loop is
        # beginning back at the beginning of the loop itself.
        if L.continuation < config.pc:
            n = l
        else:
            n = l + 1
        while s != n:
            e = config.stack.pop()
            if isinstance(e, Label):
                s += 1
        # Push the values n to the stack
        for e in v:
            config.stack.append(e)
        # Jump to the continuation of L
        config.pc = L.continuation

    @staticmethod
    def br(config: Configuration, i: binary.Instruction):
        l = i.args[0]
        return ArithmeticLogicUnit.br_label(config, l)

    @staticmethod
    def br_if(config: Configuration, i: binary.Instruction):
        if config.stack.pop().i32() == 0:
            return
        l = i.args[0]
        return ArithmeticLogicUnit.br_label(config, l)

    @staticmethod
    def br_table(config: Configuration, i: binary.Instruction):
        a = i.args[0]
        l = i.args[1]
        c = config.stack.pop().i32()
        if c >= 0 and c < len(a):
            l = a[c]
        return ArithmeticLogicUnit.br_label(config, l)

    @staticmethod
    def return_(config: Configuration, i: binary.Instruction):
        v = [config.stack.pop() for _ in range(config.frame.arity)][::-1]
        while True:
            e = config.stack.pop()
            if isinstance(e, Frame):
                config.stack.append(e)
                break
        for e in v:
            config.stack.append(e)
        # Jump to the instruction after the original call that pushed the frame
        config.pc = len(config.frame.expr.data) - 1

    @staticmethod
    def call_function_addr(config: Configuration, function_addr: FunctionAddress):
        if config.depth > convention.call_stack_depth:
            raise Exception('pywasm: call stack exhausted')

        function: FunctionInstance = config.store.function_list[function_addr]
        function_type = function.type
        function_args = [config.stack.pop() for _ in function_type.args.data][::-1]

        subcnf = Configuration(config.store)
        subcnf.depth = config.depth + 1
        subcnf.opts = config.opts
        r = subcnf.call(function_addr, function_args)
        for e in r.data:
            config.stack.append(e)


    @staticmethod
    def call_instruction_from_address(config: AbstractConfiguration, function_addr: FunctionAddress):
        function: FunctionInstance = config.store.function_list[function_addr]
        function_type = function.type

        new_instr = binary.Instruction()

        # We identify the instructions by its address
        new_instr.opcode = 0x10
        new_instr.name = f"call[{function_addr}]"
        new_instr.args: typing.List[typing.Any] = []
        new_instr.type: instruction.InstructionType = instruction.InstructionType.control
        new_instr.in_arity: int = len(function_type.args.data)
        new_instr.out_arity: int = len(function.type.rets.data)
        new_instr.comm = False
        return new_instr

    @staticmethod
    def call(config: Configuration, i: binary.Instruction):
        function_addr: binary.FunctionIndex = i.args[0]
        ArithmeticLogicUnit.call_function_addr(config, function_addr)


    @staticmethod
    def instr_from_call(config: AbstractConfiguration, i: binary.Instruction):
        function_addr: binary.FunctionIndex = i.args[0]
        instr = ArithmeticLogicUnit.call_instruction_from_address(config, function_addr)
        return instr

    @staticmethod
    def call_symbolic(config: AbstractConfiguration, i: binary.Instruction):
        instr = ArithmeticLogicUnit.instr_from_call(config, i)
        # We need to update the store to initialize new globals
        config.update_store()
        return symbolic_func(config, instr)

    @staticmethod
    def call_indirect(config: Configuration, i: binary.Instruction):
        if i.args[1] != 0x00:
            raise Exception("pywasm: zero byte malformed in call_indirect")
        ta = config.frame.module.table_addr_list[0]
        tab = config.store.table_list[ta]
        idx = config.stack.pop().i32()
        if not 0 <= idx < len(tab.element_list):
            raise Exception('pywasm: undefined element')
        function_addr = tab.element_list[idx]
        if function_addr is None:
            raise Exception('pywasm: uninitialized element')
        ArithmeticLogicUnit.call_function_addr(config, function_addr)

    @staticmethod
    def instr_from_call_indirect(config: AbstractConfiguration, i: binary.Instruction):
        if i.args[1] != 0x00:
            raise Exception("pywasm: zero byte malformed in call_indirect")
        ta = config.frame.module.table_addr_list[0]
        tab = config.store.table_list[ta]
        idx = config.stack.pop().i32()
        if not 0 <= idx < len(tab.element_list):
            raise Exception('pywasm: undefined element')
        function_addr = tab.element_list[idx]
        if function_addr is None:
            raise Exception('pywasm: uninitialized element')
        instr = ArithmeticLogicUnit.call_instruction_from_address(config, function_addr)
        return symbolic_func(config, instr)

    @staticmethod
    def call_indirect_symbolic(config: AbstractConfiguration, i: binary.Instruction):
        instr = ArithmeticLogicUnit.instr_from_call_indirect(config, i)
        # We need to update the store to initialize new globals
        config.update_store()
        return symbolic_func(config, instr)

    @staticmethod
    def drop(config: Configuration, i: binary.Instruction):
        config.stack.pop()

    @staticmethod
    def select(config: Configuration, i: binary.Instruction):
        c = config.stack.pop().i32()
        b = config.stack.pop()
        a = config.stack.pop()
        if c:
            config.stack.append(a)
        else:
            config.stack.append(b)

    @staticmethod
    def get_local(config: Configuration, i: binary.Instruction):
        r = config.frame.local_list[i.args[0]]
        o = Value()
        o.type = r.type
        if o.type == convention.symbolic or o.type == convention.term:
            o.data = r.data
        else:
            o.data = r.data.copy()
        config.stack.append(o)

    @staticmethod
    def set_local(config: Configuration, i: binary.Instruction):
        r = config.stack.pop()
        config.frame.local_list[i.args[0]] = r

    @staticmethod
    def tee_local(config: Configuration, i: binary.Instruction):
        r = config.stack.data[-1]
        o = Value()
        o.type = r.type
        if o.type == convention.symbolic or o.type == convention.term:
            o.data = r.data
        else:
            o.data = r.data.copy()
        config.frame.local_list[i.args[0]] = o

    @staticmethod
    def get_global(config: Configuration, i: binary.Instruction):
        a = config.frame.module.global_addr_list[i.args[0]]
        glob = config.store.global_list[a]
        r = glob.value
        config.stack.append(r)

    @staticmethod
    def set_global(config: Configuration, i: binary.Instruction):
        a = config.frame.module.global_addr_list[i.args[0]]
        glob = config.store.global_list[a]
        assert glob.mut == convention.var
        glob.value = config.stack.pop()

    @staticmethod
    def mem_load(config: Configuration, i: binary.Instruction, size: int) -> bytearray:
        memory_addr = config.frame.module.memory_addr_list[0]
        memory = config.store.memory_list[memory_addr]
        offset = i.args[1]
        addr = config.stack.pop().i32() + offset
        if addr < 0 or addr + size > len(memory.data):
            raise Exception('pywasm: out of bounds memory access')
        return memory.data[addr:addr + size]

    @staticmethod
    def i32_load(config: Configuration, i: binary.Instruction):
        r = Value.from_i32(num.LittleEndian.i32(ArithmeticLogicUnit.mem_load(config, i, 4)))
        config.stack.append(r)

    @staticmethod
    def i64_load(config: Configuration, i: binary.Instruction):
        r = Value.from_i64(num.LittleEndian.i64(ArithmeticLogicUnit.mem_load(config, i, 8)))
        config.stack.append(r)

    @staticmethod
    def f32_load(config: Configuration, i: binary.Instruction):
        r = Value.from_f32(num.LittleEndian.f32(ArithmeticLogicUnit.mem_load(config, i, 4)))
        config.stack.append(r)

    @staticmethod
    def f64_load(config: Configuration, i: binary.Instruction):
        r = Value.from_f64(num.LittleEndian.f64(ArithmeticLogicUnit.mem_load(config, i, 8)))
        config.stack.append(r)

    @staticmethod
    def i32_load8_s(config: Configuration, i: binary.Instruction):
        r = Value.from_i32(num.LittleEndian.i8(ArithmeticLogicUnit.mem_load(config, i, 1)))
        config.stack.append(r)

    @staticmethod
    def i32_load8_u(config: Configuration, i: binary.Instruction):
        r = Value.from_i32(ArithmeticLogicUnit.mem_load(config, i, 1)[0])
        config.stack.append(r)

    @staticmethod
    def i32_load16_s(config: Configuration, i: binary.Instruction):
        r = Value.from_i32(num.LittleEndian.i16(ArithmeticLogicUnit.mem_load(config, i, 2)))
        config.stack.append(r)

    @staticmethod
    def i32_load16_u(config: Configuration, i: binary.Instruction):
        r = Value.from_i32(num.LittleEndian.u16(ArithmeticLogicUnit.mem_load(config, i, 2)))
        config.stack.append(r)

    @staticmethod
    def i64_load8_s(config: Configuration, i: binary.Instruction):
        r = Value.from_i64(num.LittleEndian.i8(ArithmeticLogicUnit.mem_load(config, i, 1)))
        config.stack.append(r)

    @staticmethod
    def i64_load8_u(config: Configuration, i: binary.Instruction):
        r = Value.from_i64(ArithmeticLogicUnit.mem_load(config, i, 1)[0])
        config.stack.append(r)

    @staticmethod
    def i64_load16_s(config: Configuration, i: binary.Instruction):
        r = Value.from_i64(num.LittleEndian.i16(ArithmeticLogicUnit.mem_load(config, i, 2)))
        config.stack.append(r)

    @staticmethod
    def i64_load16_u(config: Configuration, i: binary.Instruction):
        r = Value.from_i64(num.LittleEndian.u16(ArithmeticLogicUnit.mem_load(config, i, 2)))
        config.stack.append(r)

    @staticmethod
    def i64_load32_s(config: Configuration, i: binary.Instruction):
        r = Value.from_i64(num.LittleEndian.i32(ArithmeticLogicUnit.mem_load(config, i, 4)))
        config.stack.append(r)

    @staticmethod
    def i64_load32_u(config: Configuration, i: binary.Instruction):
        r = Value.from_i64(num.LittleEndian.u32(ArithmeticLogicUnit.mem_load(config, i, 4)))
        config.stack.append(r)

    @staticmethod
    def mem_store(config: Configuration, i: binary.Instruction, size: int):
        memory_addr = config.frame.module.memory_addr_list[0]
        memory = config.store.memory_list[memory_addr]
        r = config.stack.pop()
        offset = i.args[1]
        addr = config.stack.pop().i32() + offset
        if addr < 0 or addr + size > len(memory.data):
            raise Exception('pywasm: out of bounds memory access')
        memory.data[addr:addr + size] = r.data[0:size]

    @staticmethod
    def i32_store(config: Configuration, i: binary.Instruction):
        ArithmeticLogicUnit.mem_store(config, i, 4)

    @staticmethod
    def i64_store(config: Configuration, i: binary.Instruction):
        ArithmeticLogicUnit.mem_store(config, i, 8)

    @staticmethod
    def f32_store(config: Configuration, i: binary.Instruction):
        ArithmeticLogicUnit.mem_store(config, i, 4)

    @staticmethod
    def f64_store(config: Configuration, i: binary.Instruction):
        ArithmeticLogicUnit.mem_store(config, i, 8)

    @staticmethod
    def i32_store8(config: Configuration, i: binary.Instruction):
        ArithmeticLogicUnit.mem_store(config, i, 1)

    @staticmethod
    def i32_store16(config: Configuration, i: binary.Instruction):
        ArithmeticLogicUnit.mem_store(config, i, 2)

    @staticmethod
    def i64_store8(config: Configuration, i: binary.Instruction):
        ArithmeticLogicUnit.mem_store(config, i, 4)

    @staticmethod
    def i64_store16(config: Configuration, i: binary.Instruction):
        ArithmeticLogicUnit.mem_store(config, i, 2)

    @staticmethod
    def i64_store32(config: Configuration, i: binary.Instruction):
        ArithmeticLogicUnit.mem_store(config, i, 4)

    @staticmethod
    def current_memory(config: Configuration, i: binary.Instruction):
        memory_addr = config.frame.module.memory_addr_list[0]
        memory = config.store.memory_list[memory_addr]
        r = Value.from_i32(memory.size)
        config.stack.append(r)

    @staticmethod
    def grow_memory(config: Configuration, i: binary.Instruction):
        memory_addr = config.frame.module.memory_addr_list[0]
        memory = config.store.memory_list[memory_addr]
        size = memory.size
        r = config.stack.pop().i32()
        if config.opts.pages_limit > 0 and memory.size + r > config.opts.pages_limit:
            raise Exception('pywasm: out of memory limit')
        try:
            memory.grow(r)
            config.stack.append(Value.from_i32(size))
        except Exception:
            config.stack.append(Value.from_i32(-1))

    @staticmethod
    def i32_const(config: Configuration, i: binary.Instruction):
        config.stack.append(Value.from_i32(i.args[0]))

    @staticmethod
    def i64_const(config: Configuration, i: binary.Instruction):
        config.stack.append(Value.from_i64(i.args[0]))

    @staticmethod
    def f32_const(config: Configuration, i: binary.Instruction):
        r = Value.from_i32(i.args[0])
        r.type = binary.ValueType(convention.f32)
        config.stack.append(r)

    @staticmethod
    def f64_const(config: Configuration, i: binary.Instruction):
        r = Value.from_i64(i.args[0])
        r.type = binary.ValueType(convention.f64)
        config.stack.append(r)

    @staticmethod
    def i32_eqz(config: Configuration, i: binary.Instruction):
        config.stack.append(Value.from_i32(config.stack.pop().i32() == 0))

    @staticmethod
    def i32_eq(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        config.stack.append(Value.from_i32(a == b))

    @staticmethod
    def i32_ne(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        config.stack.append(Value.from_i32(a != b))

    @staticmethod
    def i32_lts(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        config.stack.append(Value.from_i32(a < b))

    @staticmethod
    def i32_ltu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u32()
        a = config.stack.pop().u32()
        config.stack.append(Value.from_i32(a < b))

    @staticmethod
    def i32_gts(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        config.stack.append(Value.from_i32(a > b))

    @staticmethod
    def i32_gtu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u32()
        a = config.stack.pop().u32()
        config.stack.append(Value.from_i32(a > b))

    @staticmethod
    def i32_les(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        config.stack.append(Value.from_i32(a <= b))

    @staticmethod
    def i32_leu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u32()
        a = config.stack.pop().u32()
        config.stack.append(Value.from_i32(a <= b))

    @staticmethod
    def i32_ges(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        config.stack.append(Value.from_i32(int(a >= b)))

    @staticmethod
    def i32_geu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u32()
        a = config.stack.pop().u32()
        config.stack.append(Value.from_i32(int(a >= b)))

    @staticmethod
    def i64_eqz(config: Configuration, i: binary.Instruction):
        config.stack.append(Value.from_i32(config.stack.pop().i64() == 0))

    @staticmethod
    def i64_eq(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        config.stack.append(Value.from_i32(a == b))

    @staticmethod
    def i64_ne(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        config.stack.append(Value.from_i32(a != b))

    @staticmethod
    def i64_lts(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        config.stack.append(Value.from_i32(a < b))

    @staticmethod
    def i64_ltu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u64()
        a = config.stack.pop().u64()
        config.stack.append(Value.from_i32(a < b))

    @staticmethod
    def i64_gts(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        config.stack.append(Value.from_i32(a > b))

    @staticmethod
    def i64_gtu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u64()
        a = config.stack.pop().u64()
        config.stack.append(Value.from_i32(a > b))

    @staticmethod
    def i64_les(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        config.stack.append(Value.from_i32(a <= b))

    @staticmethod
    def i64_leu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u64()
        a = config.stack.pop().u64()
        config.stack.append(Value.from_i32(a <= b))

    @staticmethod
    def i64_ges(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        config.stack.append(Value.from_i32(a >= b))

    @staticmethod
    def i64_geu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u64()
        a = config.stack.pop().u64()
        config.stack.append(Value.from_i32(a >= b))

    @staticmethod
    def f32_eq(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        config.stack.append(Value.from_i32(a == b))

    @staticmethod
    def f32_ne(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        config.stack.append(Value.from_i32(a != b))

    @staticmethod
    def f32_lt(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        config.stack.append(Value.from_i32(a < b))

    @staticmethod
    def f32_gt(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        config.stack.append(Value.from_i32(a > b))

    @staticmethod
    def f32_le(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        config.stack.append(Value.from_i32(a <= b))

    @staticmethod
    def f32_ge(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        config.stack.append(Value.from_i32(a >= b))

    @staticmethod
    def f64_eq(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        config.stack.append(Value.from_i32(a == b))

    @staticmethod
    def f64_ne(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        config.stack.append(Value.from_i32(a != b))

    @staticmethod
    def f64_lt(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        config.stack.append(Value.from_i32(a < b))

    @staticmethod
    def f64_gt(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        config.stack.append(Value.from_i32(a > b))

    @staticmethod
    def f64_le(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        config.stack.append(Value.from_i32(a <= b))

    @staticmethod
    def f64_ge(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        config.stack.append(Value.from_i32(a >= b))

    @staticmethod
    def i32_clz(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i32()
        c = 0
        while c < 32 and (a & 0x80000000) == 0:
            c += 1
            a = a << 1
        config.stack.append(Value.from_i32(c))

    @staticmethod
    def i32_ctz(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i32()
        c = 0
        while c < 32 and (a & 0x01) == 0:
            c += 1
            a = a >> 1
        config.stack.append(Value.from_i32(c))

    @staticmethod
    def i32_popcnt(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i32()
        c = 0
        for _ in range(32):
            if a & 0x01:
                c += 1
            a = a >> 1
        config.stack.append(Value.from_i32(c))

    @staticmethod
    def i32_add(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        c = Value.from_i32(a + b)
        config.stack.append(c)

    @staticmethod
    def i32_sub(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        c = Value.from_i32(a - b)
        config.stack.append(c)

    @staticmethod
    def i32_mul(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        c = Value.from_i32(a * b)
        config.stack.append(c)

    @staticmethod
    def i32_divs(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        if b == 0:
            raise Exception('pywasm: integer divide by zero')
        if b == -1 and a == -2**31:
            raise Exception('pywasm: integer overflow')
        # Integer division that rounds towards 0 (like C)
        r = Value.from_i32(a // b if a * b > 0 else (a + (-a % b)) // b)
        config.stack.append(r)

    @staticmethod
    def i32_divu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u32()
        a = config.stack.pop().u32()
        if b == 0:
            raise Exception('pywasm: integer divide by zero')
        r = Value.from_i32(a // b)
        config.stack.append(r)

    @staticmethod
    def i32_rems(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        if b == 0:
            raise Exception('pywasm: integer divide by zero')
        # Integer remainder that rounds towards 0 (like C)
        r = Value.from_i32(a % b if a * b > 0 else -(-a % b))
        config.stack.append(r)

    @staticmethod
    def i32_remu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u32()
        a = config.stack.pop().u32()
        if b == 0:
            raise Exception('pywasm: integer divide by zero')
        r = Value.from_i32(a % b)
        config.stack.append(r)

    @staticmethod
    def i32_and(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        c = Value.from_i32(a & b)
        config.stack.append(c)

    @staticmethod
    def i32_or(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        c = Value.from_i32(a | b)
        config.stack.append(c)

    @staticmethod
    def i32_xor(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        c = Value.from_i32(a ^ b)
        config.stack.append(c)

    @staticmethod
    def i32_shl(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        c = Value.from_i32(a << (b % 0x20))
        config.stack.append(c)

    @staticmethod
    def i32_shrs(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().i32()
        c = Value.from_i32(a >> (b % 0x20))
        config.stack.append(c)

    @staticmethod
    def i32_shru(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u32()
        a = config.stack.pop().u32()
        c = Value.from_i32(a >> (b % 0x20))
        config.stack.append(c)

    @staticmethod
    def i32_rotl(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().u32()
        c = Value.from_i32((((a << (b % 0x20)) & 0xffffffff) | (a >> (0x20 - (b % 0x20)))))
        config.stack.append(c)

    @staticmethod
    def i32_rotr(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i32()
        a = config.stack.pop().u32()
        c = Value.from_i32(((a >> (b % 0x20)) | ((a << (0x20 - (b % 0x20))) & 0xffffffff)))
        config.stack.append(c)

    @staticmethod
    def i64_clz(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i64()
        c = 0
        while c < 64 and (a & 0x8000000000000000) == 0:
            c += 1
            a = a << 1
        config.stack.append(Value.from_i64(c))

    @staticmethod
    def i64_ctz(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i64()
        c = 0
        while c < 64 and (a & 0x01) == 0:
            c += 1
            a = a >> 1
        config.stack.append(Value.from_i64(c))

    @staticmethod
    def i64_popcnt(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i64()
        c = 0
        for _ in range(64):
            if a & 0x01:
                c += 1
            a = a >> 1
        config.stack.append(Value.from_i64(c))

    @staticmethod
    def i64_add(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        c = Value.from_i64(a + b)
        config.stack.append(c)

    @staticmethod
    def i64_sub(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        c = Value.from_i64(a - b)
        config.stack.append(c)

    @staticmethod
    def i64_mul(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        c = Value.from_i64(a * b)
        config.stack.append(c)

    @staticmethod
    def i64_divs(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        if b == 0:
            raise Exception('pywasm: integer divide by zero')
        if b == -1 and a == -2**63:
            raise Exception('pywasm: integer overflow')
        r = Value.from_i64(a // b if a * b > 0 else (a + (-a % b)) // b)
        config.stack.append(r)

    @staticmethod
    def i64_divu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u64()
        a = config.stack.pop().u64()
        if b == 0:
            raise Exception('pywasm: integer divide by zero')
        r = Value.from_i64(a // b)
        config.stack.append(r)

    @staticmethod
    def i64_rems(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        if b == 0:
            raise Exception('pywasm: integer divide by zero')
        # Integer remainder that rounds towards 0 (like C)
        r = Value.from_i64(a % b if a * b > 0 else -(-a % b))
        config.stack.append(r)

    @staticmethod
    def i64_remu(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u64()
        a = config.stack.pop().u64()
        if b == 0:
            raise Exception('pywasm: integer divide by zero')
        r = Value.from_i64(a % b)
        config.stack.append(r)

    @staticmethod
    def i64_and(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        c = Value.from_i64(a & b)
        config.stack.append(c)

    @staticmethod
    def i64_or(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        c = Value.from_i64(a | b)
        config.stack.append(c)

    @staticmethod
    def i64_xor(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        c = Value.from_i64(a & b)
        config.stack.append(c)

    @staticmethod
    def i64_shl(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        c = Value.from_i64(a << (b % 0x40))
        config.stack.append(c)

    @staticmethod
    def i64_shrs(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().i64()
        c = Value.from_i64(a >> (b % 0x40))
        config.stack.append(c)

    @staticmethod
    def i64_shru(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().u64()
        a = config.stack.pop().u64()
        c = Value.from_i64(a >> (b % 0x40))
        config.stack.append(c)

    @staticmethod
    def i64_rotl(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().u64()
        c = Value.from_i64((((a << (b % 0x40)) & 0xffffffffffffffff) | (a >> (0x40 - (b % 0x40)))))
        config.stack.append(c)

    @staticmethod
    def i64_rotr(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().i64()
        a = config.stack.pop().u64()
        c = Value.from_i64(((a >> (b % 0x40)) | ((a << (0x40 - (b % 0x40))) & 0xffffffffffffffff)))
        config.stack.append(c)

    @staticmethod
    def f32_abs(config: Configuration, i: binary.Instruction):
        a = config.stack.pop()
        a.data[3] = a.data[3] & 0x7f
        config.stack.append(a)

    @staticmethod
    def f32_neg(config: Configuration, i: binary.Instruction):
        a = config.stack.pop()
        if a.data[3] & 0x80 != 0x00:
            a.data[3] = a.data[3] & 0x7f
        else:
            a.data[3] = a.data[3] | 0x80
        config.stack.append(a)

    @staticmethod
    def f32_ceil(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f32()
        r = Value.from_f32(numpy.ceil(a))
        config.stack.append(r)

    @staticmethod
    def f32_floor(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f32()
        r = Value.from_f32(numpy.floor(a))
        config.stack.append(r)

    @staticmethod
    def f32_trunc(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f32()
        r = Value.from_f32(numpy.trunc(a))
        config.stack.append(r)

    @staticmethod
    def f32_nearest(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f32()
        r = Value.from_f32(numpy.round(a))
        config.stack.append(r)

    @staticmethod
    def f32_sqrt(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f32()
        r = Value.from_f32(numpy.sqrt(a))
        config.stack.append(r)

    @staticmethod
    def f32_add(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        r = Value.from_f32(a + b)
        config.stack.append(r)

    @staticmethod
    def f32_sub(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        r = Value.from_f32(a - b)
        config.stack.append(r)

    @staticmethod
    def f32_mul(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        r = Value.from_f32(a * b)
        config.stack.append(r)

    @staticmethod
    def f32_div(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        r = Value.from_f32(a / b)
        config.stack.append(r)

    @staticmethod
    def f32_min(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        if a == b == 0 and (numpy.signbit(a) or numpy.signbit(b)):
            return config.stack.append(Value.from_f32_u32(convention.f32_negative_zero))
        config.stack.append(Value.from_f32(numpy.min([a, b])))

    @staticmethod
    def f32_max(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        if a == b == 0 and not (numpy.signbit(a) and numpy.signbit(b)):
            return config.stack.append(Value.from_f32_u32(convention.f32_positive_zero))
        config.stack.append(Value.from_f32(numpy.max([a, b])))

    @staticmethod
    def f32_copysign(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f32()
        a = config.stack.pop().f32()
        r = Value.from_f32(numpy.copysign(a, b))
        config.stack.append(r)

    @staticmethod
    def f64_abs(config: Configuration, i: binary.Instruction):
        a = config.stack.pop()
        a.data[7] = a.data[7] & 0x7f
        config.stack.append(a)

    @staticmethod
    def f64_neg(config: Configuration, i: binary.Instruction):
        a = config.stack.pop()
        if a.data[7] & 0x80 != 0x00:
            a.data[7] = a.data[7] & 0x7f
        else:
            a.data[7] = a.data[7] | 0x80
        config.stack.append(a)

    @staticmethod
    def f64_ceil(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f64()
        r = Value.from_f64(numpy.ceil(a))
        config.stack.append(r)

    @staticmethod
    def f64_floor(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f64()
        r = Value.from_f64(numpy.floor(a))
        config.stack.append(r)

    @staticmethod
    def f64_trunc(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f64()
        r = Value.from_f64(numpy.trunc(a))
        config.stack.append(r)

    @staticmethod
    def f64_nearest(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f64()
        r = Value.from_f64(numpy.round(a))
        config.stack.append(r)

    @staticmethod
    def f64_sqrt(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f64()
        r = Value.from_f64(numpy.sqrt(a))
        config.stack.append(r)

    @staticmethod
    def f64_add(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        r = Value.from_f64(a + b)
        config.stack.append(r)

    @staticmethod
    def f64_sub(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        r = Value.from_f64(a - b)
        config.stack.append(r)

    @staticmethod
    def f64_mul(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        r = Value.from_f64(a * b)
        config.stack.append(r)

    @staticmethod
    def f64_div(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        r = Value.from_f64(a / b)
        config.stack.append(r)

    @staticmethod
    def f64_min(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        if a == b == 0 and (numpy.signbit(a) or numpy.signbit(b)):
            return config.stack.append(Value.from_f64_u64(convention.f64_negative_zero))
        config.stack.append(Value.from_f64(numpy.min([a, b])))

    @staticmethod
    def f64_max(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        if a == b == 0 and not (numpy.signbit(a) and numpy.signbit(b)):
            return config.stack.append(Value.from_f64_u64(convention.f64_positive_zero))
        config.stack.append(Value.from_f64(numpy.max([a, b])))

    @staticmethod
    def f64_copysign(config: Configuration, i: binary.Instruction):
        b = config.stack.pop().f64()
        a = config.stack.pop().f64()
        r = Value.from_f64(numpy.copysign(a, b))
        config.stack.append(r)

    @staticmethod
    def i32_wrap_i64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i64()
        config.stack.append(Value.from_i32(a))

    @staticmethod
    def i32_trunc_sf32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f32()
        if a > (1 << 31) - 1 or a < -(1 << 31):
            raise Exception('pywasm: integer overflow')
        try:
            b = int(a)
        except:
            raise Exception('pywasm: invalid conversion to integer')
        r = Value.from_i32(b)
        config.stack.append(r)

    @staticmethod
    def i32_trunc_uf32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f32()
        if a > (1 << 32) - 1 or a <= -1:
            raise Exception('pywasm: integer overflow')
        try:
            b = int(a)
        except:
            raise Exception('pywasm: invalid conversion to integer')
        r = Value.from_i32(b)
        config.stack.append(r)

    @staticmethod
    def i32_trunc_sf64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f64()
        if a > (1 << 31) - 1 or a < -(1 << 31):
            raise Exception('pywasm: integer overflow')
        try:
            b = int(a)
        except:
            raise Exception('pywasm: invalid conversion to integer')
        r = Value.from_i32(b)
        config.stack.append(r)

    @staticmethod
    def i32_trunc_uf64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f64()
        if a > (1 << 32) - 1 or a <= -1:
            raise Exception('pywasm: integer overflow')
        try:
            b = int(a)
        except:
            raise Exception('pywasm: invalid conversion to integer')
        r = Value.from_i32(b)
        config.stack.append(r)

    @staticmethod
    def i64_extend_si32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i32()
        r = Value.from_i64(a)
        config.stack.append(r)

    @staticmethod
    def i64_extend_ui32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().u32()
        r = Value.from_i64(a)
        config.stack.append(r)

    @staticmethod
    def i64_trunc_sf32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f32()
        if a > (1 << 63) - 1 or a < -(1 << 63):
            raise Exception('pywasm: integer overflow')
        try:
            b = int(a)
        except:
            raise Exception('pywasm: invalid conversion to integer')
        r = Value.from_i64(b)
        config.stack.append(r)

    @staticmethod
    def i64_trunc_uf32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f32()
        if a > (1 << 64) - 1 or a <= -1:
            raise Exception('pywasm: integer overflow')
        try:
            b = int(a)
        except:
            raise Exception('pywasm: invalid conversion to integer')
        r = Value.from_i64(b)
        config.stack.append(r)

    @staticmethod
    def i64_trunc_sf64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f64()
        if a > (1 << 63) - 1 or a < -(1 << 63):
            raise Exception('pywasm: integer overflow')
        try:
            b = int(a)
        except:
            raise Exception('pywasm: invalid conversion to integer')
        r = Value.from_i64(b)
        config.stack.append(r)

    @staticmethod
    def i64_trunc_uf64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f64()
        if a > (1 << 64) - 1 or a <= -1:
            raise Exception('pywasm: integer overflow')
        try:
            b = int(a)
        except:
            raise Exception('pywasm: invalid conversion to integer')
        r = Value.from_i64(b)
        config.stack.append(r)

    @staticmethod
    def f32_convert_si32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i32()
        r = Value.from_f32(num.f32(a))
        config.stack.append(r)

    @staticmethod
    def f32_convert_ui32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().u32()
        r = Value.from_f32(num.f32(a))
        config.stack.append(r)

    @staticmethod
    def f32_convert_si64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i64()
        r = Value.from_f32(num.f32(a))
        config.stack.append(r)

    @staticmethod
    def f32_convert_ui64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().u64()
        r = Value.from_f32(num.f32(a))
        config.stack.append(r)

    @staticmethod
    def f32_demote_f64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f64()
        r = Value.from_f32(num.f32(a))
        config.stack.append(r)

    @staticmethod
    def f64_convert_si32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i32()
        r = Value.from_f64(num.f64(a))
        config.stack.append(r)

    @staticmethod
    def f64_convert_ui32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().u32()
        r = Value.from_f64(num.f64(a))
        config.stack.append(r)

    @staticmethod
    def f64_convert_si64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().i64()
        r = Value.from_f64(num.f64(a))
        config.stack.append(r)

    @staticmethod
    def f64_convert_ui64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().u64()
        r = Value.from_f64(num.f64(a))
        config.stack.append(r)

    @staticmethod
    def f64_promote_f32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop().f32()
        r = Value.from_f64(num.f64(a))
        config.stack.append(r)

    @staticmethod
    def i32_reinterpret_f32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop()
        a.type = binary.ValueType(convention.i32)
        config.stack.append(a)

    @staticmethod
    def i64_reinterpret_f64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop()
        a.type = binary.ValueType(convention.i64)
        config.stack.append(a)

    @staticmethod
    def f32_reinterpret_i32(config: Configuration, i: binary.Instruction):
        a = config.stack.pop()
        a.type = binary.ValueType(convention.f32)
        config.stack.append(a)

    @staticmethod
    def f64_reinterpret_i64(config: Configuration, i: binary.Instruction):
        a = config.stack.pop()
        a.type = binary.ValueType(convention.f64)
        config.stack.append(a)


def _make_instruction_table():
    table = [None for i in range(max(instruction.opcode) + 1)]

    table[instruction.unreachable] = ArithmeticLogicUnit.unreachable
    table[instruction.nop] = ArithmeticLogicUnit.nop
    table[instruction.block] = ArithmeticLogicUnit.block
    table[instruction.loop] = ArithmeticLogicUnit.loop
    table[instruction.if_] = ArithmeticLogicUnit.if_
    table[instruction.else_] = ArithmeticLogicUnit.else_
    table[instruction.end] = ArithmeticLogicUnit.end
    table[instruction.br] = ArithmeticLogicUnit.br
    table[instruction.br_if] = ArithmeticLogicUnit.br_if
    table[instruction.br_table] = ArithmeticLogicUnit.br_table
    table[instruction.return_] = ArithmeticLogicUnit.return_
    table[instruction.call] = ArithmeticLogicUnit.call
    table[instruction.call_indirect] = ArithmeticLogicUnit.call_indirect
    table[instruction.drop] = ArithmeticLogicUnit.drop
    table[instruction.select] = ArithmeticLogicUnit.select
    table[instruction.get_local] = ArithmeticLogicUnit.get_local
    table[instruction.set_local] = ArithmeticLogicUnit.set_local
    table[instruction.tee_local] = ArithmeticLogicUnit.tee_local
    table[instruction.get_global] = ArithmeticLogicUnit.get_global
    table[instruction.set_global] = ArithmeticLogicUnit.set_global
    table[instruction.i32_load] = ArithmeticLogicUnit.i32_load
    table[instruction.i64_load] = ArithmeticLogicUnit.i64_load
    table[instruction.f32_load] = ArithmeticLogicUnit.f32_load
    table[instruction.f64_load] = ArithmeticLogicUnit.f64_load
    table[instruction.i32_load8_s] = ArithmeticLogicUnit.i32_load8_s
    table[instruction.i32_load8_u] = ArithmeticLogicUnit.i32_load8_u
    table[instruction.i32_load16_s] = ArithmeticLogicUnit.i32_load16_s
    table[instruction.i32_load16_u] = ArithmeticLogicUnit.i32_load16_u
    table[instruction.i64_load8_s] = ArithmeticLogicUnit.i64_load8_s
    table[instruction.i64_load8_u] = ArithmeticLogicUnit.i64_load8_u
    table[instruction.i64_load16_s] = ArithmeticLogicUnit.i64_load16_s
    table[instruction.i64_load16_u] = ArithmeticLogicUnit.i64_load16_u
    table[instruction.i64_load32_s] = ArithmeticLogicUnit.i64_load32_s
    table[instruction.i64_load32_u] = ArithmeticLogicUnit.i64_load32_u
    table[instruction.i32_store] = ArithmeticLogicUnit.i32_store
    table[instruction.i64_store] = ArithmeticLogicUnit.i64_store
    table[instruction.f32_store] = ArithmeticLogicUnit.f32_store
    table[instruction.f64_store] = ArithmeticLogicUnit.f64_store
    table[instruction.i32_store8] = ArithmeticLogicUnit.i32_store8
    table[instruction.i32_store16] = ArithmeticLogicUnit.i32_store16
    table[instruction.i64_store8] = ArithmeticLogicUnit.i64_store8
    table[instruction.i64_store16] = ArithmeticLogicUnit.i64_store16
    table[instruction.i64_store32] = ArithmeticLogicUnit.i64_store32
    table[instruction.current_memory] = ArithmeticLogicUnit.current_memory
    table[instruction.grow_memory] = ArithmeticLogicUnit.grow_memory
    table[instruction.i32_const] = ArithmeticLogicUnit.i32_const
    table[instruction.i64_const] = ArithmeticLogicUnit.i64_const
    table[instruction.f32_const] = ArithmeticLogicUnit.f32_const
    table[instruction.f64_const] = ArithmeticLogicUnit.f64_const
    table[instruction.i32_eqz] = ArithmeticLogicUnit.i32_eqz
    table[instruction.i32_eq] = ArithmeticLogicUnit.i32_eq
    table[instruction.i32_ne] = ArithmeticLogicUnit.i32_ne
    table[instruction.i32_lts] = ArithmeticLogicUnit.i32_lts
    table[instruction.i32_ltu] = ArithmeticLogicUnit.i32_ltu
    table[instruction.i32_gts] = ArithmeticLogicUnit.i32_gts
    table[instruction.i32_gtu] = ArithmeticLogicUnit.i32_gtu
    table[instruction.i32_les] = ArithmeticLogicUnit.i32_les
    table[instruction.i32_leu] = ArithmeticLogicUnit.i32_leu
    table[instruction.i32_ges] = ArithmeticLogicUnit.i32_ges
    table[instruction.i32_geu] = ArithmeticLogicUnit.i32_geu
    table[instruction.i64_eqz] = ArithmeticLogicUnit.i64_eqz
    table[instruction.i64_eq] = ArithmeticLogicUnit.i64_eq
    table[instruction.i64_ne] = ArithmeticLogicUnit.i64_ne
    table[instruction.i64_lts] = ArithmeticLogicUnit.i64_lts
    table[instruction.i64_ltu] = ArithmeticLogicUnit.i64_ltu
    table[instruction.i64_gts] = ArithmeticLogicUnit.i64_gts
    table[instruction.i64_gtu] = ArithmeticLogicUnit.i64_gtu
    table[instruction.i64_les] = ArithmeticLogicUnit.i64_les
    table[instruction.i64_leu] = ArithmeticLogicUnit.i64_leu
    table[instruction.i64_ges] = ArithmeticLogicUnit.i64_ges
    table[instruction.i64_geu] = ArithmeticLogicUnit.i64_geu
    table[instruction.f32_eq] = ArithmeticLogicUnit.f32_eq
    table[instruction.f32_ne] = ArithmeticLogicUnit.f32_ne
    table[instruction.f32_lt] = ArithmeticLogicUnit.f32_lt
    table[instruction.f32_gt] = ArithmeticLogicUnit.f32_gt
    table[instruction.f32_le] = ArithmeticLogicUnit.f32_le
    table[instruction.f32_ge] = ArithmeticLogicUnit.f32_ge
    table[instruction.f64_eq] = ArithmeticLogicUnit.f64_eq
    table[instruction.f64_ne] = ArithmeticLogicUnit.f64_ne
    table[instruction.f64_lt] = ArithmeticLogicUnit.f64_lt
    table[instruction.f64_gt] = ArithmeticLogicUnit.f64_gt
    table[instruction.f64_le] = ArithmeticLogicUnit.f64_le
    table[instruction.f64_ge] = ArithmeticLogicUnit.f64_ge
    table[instruction.i32_clz] = ArithmeticLogicUnit.i32_clz
    table[instruction.i32_ctz] = ArithmeticLogicUnit.i32_ctz
    table[instruction.i32_popcnt] = ArithmeticLogicUnit.i32_popcnt
    table[instruction.i32_add] = ArithmeticLogicUnit.i32_add
    table[instruction.i32_sub] = ArithmeticLogicUnit.i32_sub
    table[instruction.i32_mul] = ArithmeticLogicUnit.i32_mul
    table[instruction.i32_divs] = ArithmeticLogicUnit.i32_divs
    table[instruction.i32_divu] = ArithmeticLogicUnit.i32_divu
    table[instruction.i32_rems] = ArithmeticLogicUnit.i32_rems
    table[instruction.i32_remu] = ArithmeticLogicUnit.i32_remu
    table[instruction.i32_and] = ArithmeticLogicUnit.i32_and
    table[instruction.i32_or] = ArithmeticLogicUnit.i32_or
    table[instruction.i32_xor] = ArithmeticLogicUnit.i32_xor
    table[instruction.i32_shl] = ArithmeticLogicUnit.i32_shl
    table[instruction.i32_shrs] = ArithmeticLogicUnit.i32_shrs
    table[instruction.i32_shru] = ArithmeticLogicUnit.i32_shru
    table[instruction.i32_rotl] = ArithmeticLogicUnit.i32_rotl
    table[instruction.i32_rotr] = ArithmeticLogicUnit.i32_rotr
    table[instruction.i64_clz] = ArithmeticLogicUnit.i64_clz
    table[instruction.i64_ctz] = ArithmeticLogicUnit.i64_ctz
    table[instruction.i64_popcnt] = ArithmeticLogicUnit.i64_popcnt
    table[instruction.i64_add] = ArithmeticLogicUnit.i64_add
    table[instruction.i64_sub] = ArithmeticLogicUnit.i64_sub
    table[instruction.i64_mul] = ArithmeticLogicUnit.i64_mul
    table[instruction.i64_divs] = ArithmeticLogicUnit.i64_divs
    table[instruction.i64_divu] = ArithmeticLogicUnit.i64_divu
    table[instruction.i64_rems] = ArithmeticLogicUnit.i64_rems
    table[instruction.i64_remu] = ArithmeticLogicUnit.i64_remu
    table[instruction.i64_and] = ArithmeticLogicUnit.i64_and
    table[instruction.i64_or] = ArithmeticLogicUnit.i64_or
    table[instruction.i64_xor] = ArithmeticLogicUnit.i64_xor
    table[instruction.i64_shl] = ArithmeticLogicUnit.i64_shl
    table[instruction.i64_shrs] = ArithmeticLogicUnit.i64_shrs
    table[instruction.i64_shru] = ArithmeticLogicUnit.i64_shru
    table[instruction.i64_rotl] = ArithmeticLogicUnit.i64_rotl
    table[instruction.i64_rotr] = ArithmeticLogicUnit.i64_rotr
    table[instruction.f32_abs] = ArithmeticLogicUnit.f32_abs
    table[instruction.f32_neg] = ArithmeticLogicUnit.f32_neg
    table[instruction.f32_ceil] = ArithmeticLogicUnit.f32_ceil
    table[instruction.f32_floor] = ArithmeticLogicUnit.f32_floor
    table[instruction.f32_trunc] = ArithmeticLogicUnit.f32_trunc
    table[instruction.f32_nearest] = ArithmeticLogicUnit.f32_nearest
    table[instruction.f32_sqrt] = ArithmeticLogicUnit.f32_sqrt
    table[instruction.f32_add] = ArithmeticLogicUnit.f32_add
    table[instruction.f32_sub] = ArithmeticLogicUnit.f32_sub
    table[instruction.f32_mul] = ArithmeticLogicUnit.f32_mul
    table[instruction.f32_div] = ArithmeticLogicUnit.f32_div
    table[instruction.f32_min] = ArithmeticLogicUnit.f32_min
    table[instruction.f32_max] = ArithmeticLogicUnit.f32_max
    table[instruction.f32_copysign] = ArithmeticLogicUnit.f32_copysign
    table[instruction.f64_abs] = ArithmeticLogicUnit.f64_abs
    table[instruction.f64_neg] = ArithmeticLogicUnit.f64_neg
    table[instruction.f64_ceil] = ArithmeticLogicUnit.f64_ceil
    table[instruction.f64_floor] = ArithmeticLogicUnit.f64_floor
    table[instruction.f64_trunc] = ArithmeticLogicUnit.f64_trunc
    table[instruction.f64_nearest] = ArithmeticLogicUnit.f64_nearest
    table[instruction.f64_sqrt] = ArithmeticLogicUnit.f64_sqrt
    table[instruction.f64_add] = ArithmeticLogicUnit.f64_add
    table[instruction.f64_sub] = ArithmeticLogicUnit.f64_sub
    table[instruction.f64_mul] = ArithmeticLogicUnit.f64_mul
    table[instruction.f64_div] = ArithmeticLogicUnit.f64_div
    table[instruction.f64_min] = ArithmeticLogicUnit.f64_min
    table[instruction.f64_max] = ArithmeticLogicUnit.f64_max
    table[instruction.f64_copysign] = ArithmeticLogicUnit.f64_copysign
    table[instruction.i32_wrap_i64] = ArithmeticLogicUnit.i32_wrap_i64
    table[instruction.i32_trunc_sf32] = ArithmeticLogicUnit.i32_trunc_sf32
    table[instruction.i32_trunc_uf32] = ArithmeticLogicUnit.i32_trunc_uf32
    table[instruction.i32_trunc_sf64] = ArithmeticLogicUnit.i32_trunc_sf64
    table[instruction.i32_trunc_uf64] = ArithmeticLogicUnit.i32_trunc_uf64
    table[instruction.i64_extend_si32] = ArithmeticLogicUnit.i64_extend_si32
    table[instruction.i64_extend_ui32] = ArithmeticLogicUnit.i64_extend_ui32
    table[instruction.i64_trunc_sf32] = ArithmeticLogicUnit.i64_trunc_sf32
    table[instruction.i64_trunc_uf32] = ArithmeticLogicUnit.i64_trunc_uf32
    table[instruction.i64_trunc_sf64] = ArithmeticLogicUnit.i64_trunc_sf64
    table[instruction.i64_trunc_uf64] = ArithmeticLogicUnit.i64_trunc_uf64
    table[instruction.f32_convert_si32] = ArithmeticLogicUnit.f32_convert_si32
    table[instruction.f32_convert_ui32] = ArithmeticLogicUnit.f32_convert_ui32
    table[instruction.f32_convert_si64] = ArithmeticLogicUnit.f32_convert_si64
    table[instruction.f32_convert_ui64] = ArithmeticLogicUnit.f32_convert_ui64
    table[instruction.f32_demote_f64] = ArithmeticLogicUnit.f32_demote_f64
    table[instruction.f64_convert_si32] = ArithmeticLogicUnit.f64_convert_si32
    table[instruction.f64_convert_ui32] = ArithmeticLogicUnit.f64_convert_ui32
    table[instruction.f64_convert_si64] = ArithmeticLogicUnit.f64_convert_si64
    table[instruction.f64_convert_ui64] = ArithmeticLogicUnit.f64_convert_ui64
    table[instruction.f64_promote_f32] = ArithmeticLogicUnit.f64_promote_f32
    table[instruction.i32_reinterpret_f32] = ArithmeticLogicUnit.i32_reinterpret_f32
    table[instruction.i64_reinterpret_f64] = ArithmeticLogicUnit.i64_reinterpret_f64
    table[instruction.f32_reinterpret_i32] = ArithmeticLogicUnit.f32_reinterpret_i32
    table[instruction.f64_reinterpret_i64] = ArithmeticLogicUnit.f64_reinterpret_i64

    return table


_INSTRUCTION_TABLE = _make_instruction_table()


class Machine:
    # Execution behavior is defined in terms of an abstract machine that models the program state. It includes a stack,
    # which records operand values and control constructs, and an abstract store containing global state.
    def __init__(self):
        self.module: ModuleInstance = ModuleInstance()
        self.store: Store = Store()
        self.opts: option.Option = option.Option()

    def instantiate(self, module: binary.Module, extern_value_list: typing.List[ExternValue]):
        self.module.type_list = module.type_list

        # [TODO] If module is not valid, then panic

        # Assert: module is valid with external types classifying its imports
        for e in extern_value_list:
            if isinstance(e, FunctionAddress):
                assert e < len(self.store.function_list)
            if isinstance(e, TableAddress):
                assert e < len(self.store.table_list)
            if isinstance(e, MemoryAddress):
                assert e < len(self.store.memory_list)
            if isinstance(e, GlobalAddress):
                assert e < len(self.store.global_list)

        # If the number m of imports is not equal to the number n of provided external values, then fail
        # assert len(module.import_list) == len(extern_value_list)

        # For each external value and external type, do:
        # If externval is not valid with an external type in store S, then fail
        # If externtype does not match externtype, then fail
        for i, e in enumerate(extern_value_list):
            if isinstance(e, FunctionAddress):
                pass
                # Avoid checking FunctionAddresses, as imports are not required
                # a = self.store.function_list[e].type
                # b = module.type_list[module.import_list[i].desc]
                # assert match_function(a, b)
            if isinstance(e, TableAddress):
                a = self.store.table_list[e].limits
                b = module.import_list[i].desc.limits
                assert match_limits(a, b)
            if isinstance(e, MemoryAddress):
                a = self.store.memory_list[e].type
                b = module.import_list[i].desc
                assert match_memory(a, b)
            if isinstance(e, GlobalAddress):
                assert module.import_list[i].desc.value_type == self.store.global_list[e].value.type
                assert module.import_list[i].desc.mut == self.store.global_list[e].mut

        # Let vals be the vector of global initialization values determined by module and externvaln
        global_values: typing.List[Value] = []
        aux = ModuleInstance()
        aux.global_addr_list = [e for e in extern_value_list if isinstance(e, GlobalAddress)]
        for e in module.global_list:
            log.debugln(f'init global value')
            frame = Frame(aux, [], e.expr, 1)
            config = Configuration(self.store)
            config.opts = self.opts
            config.set_frame(frame)
            r = config.exec().data[0]
            global_values.append(r)

        # Let moduleinst be a new module instance allocated from module in store S with imports externval and global
        # initializer values, and let S be the extended store produced by module allocation.
        self.allocate(module, extern_value_list, global_values)

        for element_segment in module.element_list:
            log.debugln('init elem')
            # Let F be the frame, push the frame F to the stack
            frame = Frame(self.module, [], element_segment.offset, 1)
            config = Configuration(self.store)
            config.opts = self.opts
            config.set_frame(frame)
            r = config.exec().data[0]
            offset = r.val()
            table_addr = self.module.table_addr_list[element_segment.table_index]
            table_instance = self.store.table_list[table_addr]
            for i, e in enumerate(element_segment.init):
                table_instance.element_list[offset + i] = e

        for data_segment in module.data_list:
            log.debugln('init data')
            frame = Frame(self.module, [], data_segment.offset, 1)
            config = Configuration(self.store)
            config.opts = self.opts
            config.set_frame(frame)
            r = config.exec().data[0]
            offset = r.val()
            memory_addr = self.module.memory_addr_list[data_segment.memory_index]
            memory_instance = self.store.memory_list[memory_addr]
            memory_instance.data[offset: offset + len(data_segment.init)] = data_segment.init

        # [TODO] Assert: due to validation, the frame F is now on the top of the stack.

        # If the start function module.start is not empty, invoke the function instance
        if module.start is not None:
            log.debugln(f'running start function {module.start}')
            self.invocate(self.module.function_addr_list[module.start.function_idx], [])

    def allocate(
        self,
        module: binary.Module,
        extern_value_list: typing.List[ExternValue],
        global_values: typing.List[Value],
    ):
        # Let funcaddr be the list of function addresses extracted from externval, concatenated with funcaddr
        # Let tableaddr be the list of table addresses extracted from externval, concatenated with tableaddr
        # Let memaddr be the list of memory addresses extracted from externval, concatenated with memaddr
        # Let globaladdr be the list of global addresses extracted from externval, concatenated with globaladdr
        for e in extern_value_list:
            if isinstance(e, FunctionAddress):
                self.module.function_addr_list.append(e)
            if isinstance(e, TableAddress):
                self.module.table_addr_list.append(e)
            if isinstance(e, MemoryAddress):
                self.module.memory_addr_list.append(e)
            if isinstance(e, GlobalAddress):
                self.module.global_addr_list.append(e)

        # For each function func in module.funcs, do:
        for e in module.function_list:
            function_addr = self.store.allocate_wasm_function(self.module, e)
            self.module.function_addr_list.append(function_addr)

        # For each table in module.tables, do:
        for e in module.table_list:
            table_addr = self.store.allocate_table(e.type)
            self.module.table_addr_list.append(table_addr)

        # For each memory module.mems, do:
        for e in module.memory_list:
            memory_addr = self.store.allocate_memory(e.type)
            self.module.memory_addr_list.append(memory_addr)

        # For each global in module.globals, do:
        for i, e in enumerate(module.global_list):
            global_addr = self.store.allocate_global(e.type, global_values[i])
            self.module.global_addr_list.append(global_addr)

        # For each export in module.exports, do:
        for e in module.export_list:
            if isinstance(e.desc, binary.FunctionIndex):
                addr = self.module.function_addr_list[e.desc]
            if isinstance(e.desc, binary.TableIndex):
                addr = self.module.table_addr_list[e.desc]
            if isinstance(e.desc, binary.MemoryIndex):
                addr = self.module.memory_addr_list[e.desc]
            if isinstance(e.desc, binary.GlobalIndex):
                addr = self.module.global_addr_list[e.desc]
            export_inst = ExportInstance(e.name, addr)
            self.module.export_list.append(export_inst)

    def invocate(self, function_addr: FunctionAddress, function_args: typing.List[Value]) -> Result:
        config = Configuration(self.store)
        config.opts = self.opts
        return config.call(function_addr, function_args)

    def invocate_symbolic(self, function: typing.Union[HostFunc, WasmFunc], function_args: typing.List[Value], func_address: int) -> Result:
        config = AbstractConfiguration(self.store)
        config.opts = self.opts
        return config.call_symbolic(function, function_args, func_address)
