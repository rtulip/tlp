
from dataclasses import dataclass, field
from enum import Enum, auto
from operator import indexOf
from subprocess import call
from typing import List, Union, Dict, Optional, Any, Tuple, Callable
from lexer import tokenize, Token, Intrinsic, MiscTokenKind, Keyword, tokenize_string
import sys
from copy import deepcopy


class OpType(Enum):
    PUSH_UINT = auto()
    PUSH_BOOL = auto()
    PUSH_PTR = auto()
    PUSH_STRING = auto()
    INTRINSIC = auto()
    JUMP_COND = auto()
    JUMP = auto()
    RETURN = auto()
    CALL = auto()
    NOP = auto()


@dataclass(frozen=True)
class DataType:
    ident: str
    generic: bool = False
    struct: bool = False
    size: int = 1
    fnptr: Optional[str] = None


INT = DataType("int")
BOOL = DataType("bool")
PTR = DataType("ptr")
STR = DataType("Str", struct=True, size=2)
T = DataType("T", generic=True)
A = DataType("A", generic=True)
B = DataType("B", generic=True)
C = DataType("C", generic=True)
D = DataType("D", generic=True)
E = DataType("E", generic=True)
F = DataType("F", generic=True)

TUPLE_IDENT_COUNT = 0

TypeDict: Dict[str, DataType] = {
    "int": INT,
    "bool": BOOL,
    "ptr": PTR,
}

ArgList = List[DataType]
StructMembers: Dict[DataType, ArgList] = {}
StructGenerics: Dict[DataType, ArgList] = {}


@ dataclass
class Op():
    op: OpType
    tok: Token
    operand: Optional[Any]


@ dataclass
class Signature:
    pops: ArgList
    puts: ArgList
    rpops: ArgList = field(default_factory=lambda: [])
    rputs: ArgList = field(default_factory=lambda: [])


TypeSignatures: Dict[DataType, Signature] = {}
Program = List[Op]


@ dataclass
class Function:
    ident: str
    tok: Token
    generics: ArgList
    signature: Signature
    stub: bool
    program: Program


IncludedFiles: List[str] = []

FunctionMeta = Dict[str, Function]

ConstMap = Dict[str, Op]
MemoryMap = Dict[str, Tuple[int, Token]]

signatures = {
    OpType.PUSH_UINT: Signature(pops=[], puts=[INT]),
    OpType.PUSH_BOOL: Signature(pops=[], puts=[BOOL]),
    OpType.PUSH_PTR: Signature(pops=[], puts=[PTR]),
    OpType.PUSH_STRING: Signature(pops=[], puts=[STR]),
    OpType.JUMP_COND: Signature(pops=[BOOL], puts=[]),
    OpType.JUMP: Signature(pops=[], puts=[]),
    OpType.CALL: Signature(pops=[], puts=[]),
    OpType.RETURN: Signature(pops=[], puts=[]),
    OpType.NOP: Signature(pops=[], puts=[]),
    Intrinsic.ADD: Signature(pops=[INT, INT], puts=[INT]),
    Intrinsic.SUB: Signature(pops=[INT, INT], puts=[INT]),
    Intrinsic.DIV: Signature(pops=[INT, INT], puts=[INT]),
    Intrinsic.MUL: Signature(pops=[INT, INT], puts=[INT]),
    Intrinsic.MOD: Signature(pops=[INT, INT], puts=[INT]),
    Intrinsic.AND: Signature(pops=[BOOL, BOOL], puts=[BOOL]),
    Intrinsic.OR: Signature(pops=[INT, INT], puts=[INT]),
    Intrinsic.LSL: Signature(pops=[INT, INT], puts=[INT]),
    Intrinsic.EQ: Signature(pops=[INT, INT], puts=[BOOL]),
    Intrinsic.LE: Signature(pops=[INT, INT], puts=[BOOL]),
    Intrinsic.LT: Signature(pops=[INT, INT], puts=[BOOL]),
    Intrinsic.BW_AND: Signature(pops=[INT, INT], puts=[INT]),
    Intrinsic.READ64: Signature(pops=[PTR], puts=[INT]),
    Intrinsic.READ8: Signature(pops=[PTR], puts=[INT]),
    Intrinsic.WRITE64: Signature(pops=[A, PTR], puts=[]),
    Intrinsic.WRITE8: Signature(pops=[A, PTR], puts=[]),
    Intrinsic.GT: Signature(pops=[INT, INT], puts=[BOOL]),
    Intrinsic.PUTU: Signature(pops=[INT], puts=[]),
    Intrinsic.DUP: Signature(pops=[T], puts=[T, T]),
    Intrinsic.DROP: Signature(pops=[T], puts=[]),
    Intrinsic.SWAP: Signature(pops=[A, B], puts=[B, A]),
    Intrinsic.RPUSH: Signature(pops=[T], puts=[], rputs=[T]),
    Intrinsic.RPOP: Signature(pops=[], puts=[T], rpops=[T]),
    Intrinsic.SIZE_OF: Signature(pops=[], puts=[INT]),
    # Intrinsic.SPLIT creates signature dynamically based on the struct
    # Intrinsic.CAST creates signatures dynamically based on the struct
    # Intrinsic.INNER_TUPLE creates a signature dynamically based on the tuple on the top of the stack.
    # Intrinsic.CAST_TUPLE  creates signatures dynamically based on the number of elements asked to group
    # Intrinsic.ADDR_OF creates a new type based on the function specified.
    # Intrinsic.CALL creates a new signature dynamically.
    Intrinsic.SYSCALL0: Signature(pops=[INT], puts=[INT]),
    Intrinsic.SYSCALL1: Signature(pops=[A, INT], puts=[INT]),
    Intrinsic.SYSCALL2: Signature(pops=[A, B, INT], puts=[INT]),
    Intrinsic.SYSCALL3: Signature(pops=[A, B, C, INT], puts=[INT]),
    Intrinsic.SYSCALL4: Signature(pops=[A, B, C, D, INT], puts=[INT]),
    Intrinsic.SYSCALL5: Signature(pops=[A, B, C, D, E, INT], puts=[INT]),
    Intrinsic.SYSCALL6: Signature(pops=[A, B, C, D, E, F, INT], puts=[INT]),

}

N_DYNAMIC_INTRINSICS = 6
N_IGNORED_OP = 1
N_SUM_IGNORE = N_DYNAMIC_INTRINSICS + N_IGNORED_OP
assert len(signatures) == len(OpType) - N_SUM_IGNORE + len(Intrinsic), \
    f"Not all OpTypes and Intrinsics have a signature. Expected {len(OpType) - N_SUM_IGNORE + len(Intrinsic)} Found {len(signatures)}"

PRELUDE_SIZE = 0


def compiler_error(predicate: bool, token: Token, msg: str, notes: List[str] = []):
    if not predicate:
        print(f"{token.loc} [ERROR]: {msg}", file=sys.stderr)
        for note in notes:
            print(f"    [NOTE]: {note}", file=sys.stderr)
        exit(1)


def check_for_name_conflict(
    name: str,
    tok: Token,
    fn_meta: FunctionMeta,
    const_values: ConstMap,
    reserved_memory: MemoryMap,
    ignore_fn: bool = False,
    ignore_type: bool = False,
    ignore_const: bool = False,
    ignore_mem: bool = False,
):
    if name in fn_meta.keys() and not ignore_fn:
        compiler_error(
            False,
            tok,
            f"Redefinition of `{name}`. Previously defined here: {fn_meta[name].tok.loc}"
        )

    if name in TypeDict.keys() and not ignore_type:
        compiler_error(
            False,
            tok,
            f"Redefinition of Type `{name}`."
        )

    if name in const_values.keys() and not ignore_const:
        compiler_error(
            False,
            tok,
            f"Redefinition of `{name}`. Previously defined here: {const_values[name].tok.loc}"
        )

    if name in reserved_memory.keys() and not ignore_mem:
        compiler_error(
            False,
            tok,
            f"Redefinition of `{name}`. Previously defined here: {reserved_memory[name][1].loc}"
        )


class FnDefState(Enum):
    Start = auto()
    Name = auto()
    Inputs = auto()
    Outputs = auto()
    Open = auto()
    End = auto()


def tokens_until_keywords(
    start_tok: Token,
    tokens: List[Token],
    expected_keywords: List[Keyword]
) -> Tuple[Token, List[Token]]:

    toks: List[Token] = []
    while len(tokens) > 0:
        tok = tokens.pop()

        if tok.typ in expected_keywords:
            break
        elif type(tok.typ) == Keyword:
            compiler_error(
                False,
                tok,
                f"Unexpected keyword {tok.typ}"
            )
        else:
            toks.append(tok)

    compiler_error(
        tok.typ in expected_keywords,
        start_tok,
        f"Expected one of {expected_keywords}, but found end of file instead"
    )

    return (tok, toks)


def parse_tokens_until_keywords(
    tokens: List[Token],
    expected_keywords: List[Keyword],
    generic_types: List[DataType],
    program: Program,
    fn_meta: FunctionMeta,
    const_values: ConstMap,
    reserved_memory: MemoryMap,
) -> Optional[Token]:

    if len(tokens) == 0:
        return None

    while len(tokens) > 0:
        tok = tokens.pop()
        if type(tok.typ) == MiscTokenKind:

            if tok.typ == MiscTokenKind.INT:
                program.append(Op(
                    op=OpType.PUSH_UINT,
                    operand=tok.value,
                    tok=tok,
                ))
            elif tok.typ == MiscTokenKind.BOOL:
                program.append(Op(
                    op=OpType.PUSH_BOOL,
                    operand=tok.value == "true",
                    tok=tok
                ))

            elif tok.typ == MiscTokenKind.WORD:

                if tok.value in fn_meta.keys():

                    compiler_error(
                        len(fn_meta[tok.value].generics) == 0,
                        tok,
                        f"Cannot call generic function `{tok.value}` without casting types first.",
                        [
                            f"Use a `WITH`-`DO` block to declare types.",
                            f"Undefined generics: {pretty_print_arg_list(fn_meta[tok.value].generics)}"
                        ]
                    )

                    program.append(Op(
                        op=OpType.CALL,
                        operand=tok.value,
                        tok=tok,
                    ))
                elif tok.value in const_values.keys():
                    program.append(const_values[tok.value])
                elif tok.value in reserved_memory.keys():
                    program.append(Op(
                        op=OpType.PUSH_PTR,
                        tok=tok,
                        operand=tok.value
                    ))
                else:
                    compiler_error(
                        False,
                        tok,
                        f"Unrecognized WORD {tok.value}."
                    )

            elif tok.typ == MiscTokenKind.STRING:

                program.append(Op(
                    op=OpType.PUSH_STRING,
                    operand=tok.value,
                    tok=tok
                ))

            else:
                assert False, "Unreachable... Unknown MiscTokenKind"

        elif type(tok.typ) == Intrinsic:
            assert tok.typ in Intrinsic, f"Unknown Intrinsic {tok.typ}"

            if tok.typ == Intrinsic.CAST_TUPLE:
                global PRELUDE_SIZE
                compiler_error(
                    len(program) > PRELUDE_SIZE,
                    tok,
                    f"`GROUP` expects a preceding `UINT`, but found end of file instead"
                )

                tuple_size_op = program.pop()

                compiler_error(
                    tuple_size_op.op == OpType.PUSH_UINT,
                    tok,
                    f"`GROUP` expectes a preceding `UINT` for how many elements to group. Found {tuple_size_op.tok.typ} instead."
                )

                n_tuple_elements = tuple_size_op.operand
                tok.value = n_tuple_elements
                program.append(Op(
                    op=OpType.INTRINSIC,
                    operand=Intrinsic.CAST_TUPLE,
                    tok=tok
                ))
            else:
                program.append(Op(
                    op=OpType.INTRINSIC,
                    operand=tok.typ,
                    tok=tok
                ))

        elif type(tok.typ) == Keyword:

            if tok.typ == Keyword.IF:
                parse_if_block_from_tokens(
                    tok,
                    tokens,
                    generic_types,
                    program,
                    fn_meta,
                    const_values,
                    reserved_memory,
                )
            elif tok.typ == Keyword.WHILE:
                parse_while_block_from_tokens(
                    tok,
                    tokens,
                    generic_types,
                    program,
                    fn_meta,
                    const_values,
                    reserved_memory,
                )
            elif tok.typ == Keyword.INCLUDE:
                parse_include_statement(
                    tok,
                    tokens,
                    program,
                    fn_meta,
                    const_values,
                )
            elif tok.typ == Keyword.RESERVE:
                parse_reserve_statement(
                    tok,
                    tokens,
                    fn_meta,
                    const_values,
                    reserved_memory,
                )
            elif tok.typ == Keyword.WITH and Keyword.WITH not in expected_keywords:
                tok, types = parse_with_block_from_tokens(
                    tok,
                    tokens,
                )
                call_generic_fn_with(
                    tok,
                    tokens,
                    fn_meta,
                    types,
                    generic_types,
                    program
                )
            else:

                compiler_error(
                    tok.typ in expected_keywords,
                    tok,
                    f"Unexpected Keyword: {tok.typ}"
                )
                return tok

        else:
            assert False, f"Unhandled Token: {tok}"

    return tok


def eval_const_ops(program: Program, tok: Token) -> Optional[Op]:
    stack = []
    program.reverse()

    compiler_error(
        len(program) > 0,
        tok,
        f"`CONST` body must evaluate to a single value"
    )

    while len(program) > 0:
        op = program.pop()

        if op.op == OpType.PUSH_UINT:
            stack.append(op.operand)
        elif op.op == OpType.INTRINSIC:
            if op.operand == Intrinsic.ADD:
                b = stack.pop()
                a = stack.pop()
                assert isinstance(a, int)
                assert isinstance(b, int)
                stack.append(a+b)
            elif op.operand == Intrinsic.SUB:
                b = stack.pop()
                a = stack.pop()
                assert isinstance(a, int)
                assert isinstance(b, int)
                stack.append(a-b)
            elif op.operand == Intrinsic.MUL:
                b = stack.pop()
                a = stack.pop()
                assert isinstance(a, int)
                assert isinstance(b, int)
                stack.append(a*b)
            elif op.operand == Intrinsic.OR:
                b = stack.pop()
                a = stack.pop()
                assert isinstance(a, int)
                assert isinstance(b, int)
                stack.append(a | b)
            elif op.operand == Intrinsic.LSL:
                b = stack.pop()
                a = stack.pop()
                assert isinstance(a, int)
                assert isinstance(b, int)
                stack.append(a << b)
            elif op.operand == Intrinsic.SIZE_OF:
                compiler_error(
                    op.tok.value in TypeDict.keys(),
                    op.tok,
                    f"Cannot get size of unknown type `{op.tok.value}`."
                )

                stack.append(TypeDict[op.tok.value].size * 8)
            else:
                compiler_error(
                    False,
                    op.tok,
                    f"Intrinsic {op.operand} is not supported in constant expressions yet."
                )
        else:
            compiler_error(
                False,
                op.tok,
                f"Operation {op.op}:{op.operand} is not supported in constant expressions yet."
            )

    compiler_error(
        len(stack) == 1,
        op.tok,
        f"`CONST` expressions must evaluate to a single type"
    )

    value = stack.pop()

    if type(value) == int:
        return Op(op=OpType.PUSH_UINT, operand=value, tok=tok)
    else:
        compiler_error(
            False,
            op.tok,
            f"Unsupported CONST evaluation type: {value}"
        )

    return None


def parse_with_block_from_tokens(
    start_tok: Token,
    tokens: List[Token],
) -> Tuple[Token, List[DataType]]:
    assert start_tok.typ == Keyword.WITH

    compiler_error(
        len(tokens) > 0,
        start_tok,
        "Expected generic type identifiers after `WITH`, but found end of file instead."
    )

    end_tok, type_list = parse_type_list(
        start_tok, tokens, [], [Keyword.STRUCT, Keyword.FN, Keyword.DO], allow_undefined_generics=True)

    compiler_error(
        len(type_list) > 0,
        start_tok,
        "`WITH` statement must include at least one generic type identifier."
    )

    return end_tok, type_list


def parse_reserve_statement(
    start_tok: Token,
    tokens: List[Token],
    fn_meta: FunctionMeta,
    const_values: ConstMap,
    reserved_memory: MemoryMap,
):

    assert start_tok.typ == Keyword.RESERVE
    compiler_error(
        len(tokens) > 0,
        start_tok,
        f"Expected identifier after `RESERVE` statement, but found end of file instead"
    )

    memory_name = tokens.pop()
    compiler_error(
        memory_name.typ == MiscTokenKind.WORD,
        memory_name,
        f"Expected an identifier after `RESERVE` statement, but found {memory_name.typ}:{memory_name.value} instead"
    )

    check_for_name_conflict(
        memory_name.value,
        memory_name,
        fn_meta,
        const_values,
        reserved_memory,
    )

    compiler_error(
        len(tokens) > 0,
        memory_name,
        f"Expected number after reserved memory identifier, but found end of file instead"
    )

    const_expr: Program = []
    tok = parse_tokens_until_keywords(
        tokens,
        [Keyword.END],
        [],
        const_expr,
        fn_meta,
        const_values,
        reserved_memory,
    )

    if not isinstance(tok, Token):
        compiler_error(
            False,
            start_tok,
            "Expected `END` to close reserve block. Found end of file instead"
        )

    assert isinstance(tok, Token)

    compiler_error(
        tok.typ == Keyword.END,
        tok,
        f"Expected `END` to close reserve block. found {tok.typ}:{tok.value} instead"
    )

    value = eval_const_ops(const_expr, start_tok)
    assert value is not None

    compiler_error(
        value.op == OpType.PUSH_UINT,
        start_tok,
        "Constant Expression in Reserve block must evaluate to an int.",
    )
    assert isinstance(value.operand, int)

    reserved_memory[memory_name.value] = (value.operand, start_tok)


def parse_const_expr(
    start_tok: Token,
    tokens: List[Token],
    fn_meta: FunctionMeta,
    const_values: ConstMap,
    reserved_memory: MemoryMap
):

    assert start_tok.typ == Keyword.CONST

    compiler_error(
        len(tokens) > 0,
        start_tok,
        f"Expected identifier after `CONST` statement, but found end of file instead"
    )

    const_ident = tokens.pop()

    compiler_error(
        const_ident.typ == MiscTokenKind.WORD,
        const_ident,
        f"Expected an identifier after `CONST` statement, but found {const_ident.typ}:{const_ident.value} instead"
    )

    check_for_name_conflict(
        const_ident.value,
        const_ident,
        fn_meta,
        const_values,
        reserved_memory,
    )

    compiler_error(
        len(tokens) > 0,
        const_ident,
        f"Expected `CONST` body, but found end of file instead"
    )

    const_ops: Program = []
    parse_tokens_until_keywords(
        tokens,
        [Keyword.END],
        [],
        const_ops,
        fn_meta,
        const_values,
        reserved_memory,
    )
    value = eval_const_ops(const_ops, start_tok)
    assert value is not None
    const_values[const_ident.value] = value


def parse_include_statement(start_tok: Token, tokens: List[Token], program: Program, fn_meta: FunctionMeta, const_values: ConstMap):

    compiler_error(
        len(tokens) > 0,
        start_tok,
        f"Expected a string after a `USE` statement, but found end of file instead"
    )

    include_str_tok = tokens.pop()

    compiler_error(
        include_str_tok.typ == MiscTokenKind.STRING,
        include_str_tok,
        f"Expected a string after a `USE` statement, but found {include_str_tok.typ} instead"
    )

    include_str = include_str_tok.value
    if include_str not in IncludedFiles:
        included_tokens = tokenize(include_str)
        included_tokens.reverse()
        tokens += included_tokens
        IncludedFiles.append(include_str)


def parse_generic_struct_type(
    start_tok: Token,
    tokens: List[Token],
    concrete_types: List[DataType],
    generic_types: List[DataType]
) -> DataType:

    compiler_error(
        len(tokens) > 0,
        start_tok,
        "Expected struct name, but found end of file instead."
    )

    struct_tok = tokens.pop()
    compiler_error(
        struct_tok.typ == MiscTokenKind.WORD,
        struct_tok,
        f"Expected struct name, but found `{struct_tok.typ}:{struct_tok.value}` instead."
    )

    struct_name = struct_tok.value
    assert struct_name is not None
    compiler_error(
        struct_name in TypeDict.keys(),
        struct_tok,
        f"Unknown struct `{struct_name}`."
    )

    compiler_error(
        TypeDict[struct_name].struct,
        struct_tok,
        f"Type `{struct_name}` is not a struct."
    )

    gen_struct_t = TypeDict[struct_name]
    compiler_error(
        gen_struct_t in StructGenerics.keys(),
        struct_tok,
        f"Struct `{struct_name}` is not generic."
    )

    compiler_error(
        len(concrete_types) == len(StructGenerics[gen_struct_t]),
        struct_tok,
        f"""Incorrect number of types provided.
    [Note]: `{struct_name}` is generic over {pretty_print_arg_list(StructGenerics[gen_struct_t])}
    [Note]: These types were provided: {pretty_print_arg_list(concrete_types)}"""
    )

    concrete_name = f"{struct_name}{pretty_print_arg_list(concrete_types, open='<', close='>')}"
    concrete_members = convert_to_concrete_arg_list(
        StructMembers[gen_struct_t], concrete_types, StructGenerics[gen_struct_t])
    # [T if not T.generic else
    # concrete_types[indexOf(StructGenerics[gen_struct_t], T)] for T in StructMembers[gen_struct_t]]

    TypeDict[concrete_name] = DataType(
        ident=concrete_name,
        generic=any([T.generic for T in concrete_members]),
        struct=True,
        size=sum([T.size for T in concrete_members])
    )
    StructMembers[TypeDict[concrete_name]] = concrete_members
    StructGenerics[TypeDict[concrete_name]] = generic_types

    return TypeDict[concrete_name]


def parse_type_from_with_block(start_tok: Token, tokens: List[Token], generic_types: List[DataType]) -> DataType:
    tok, type_list = parse_type_list(
        start_tok, tokens, generic_types, [Keyword.ARROW, Keyword.FN_TYPE])

    if tok.typ == Keyword.ARROW:
        return parse_generic_struct_type(tok, tokens, type_list, generic_types)
    elif tok.typ == Keyword.FN_TYPE:
        return parse_fn_ptr_type(tok, tokens, type_list, generic_types)

    assert False, "... Unreachable"


def parse_fn_ptr_type(start_tok: Token, tokens: List[Token], concrete_types: List[DataType], generic_types: List[DataType]) -> DataType:

    tok, sig = parse_fn_signature(start_tok, tokens, generic_types)
    compiler_error(
        tok.typ == Keyword.END,
        tok,
        f"Unexpected keyword `{tok.typ}`. Expected `END`"
    )

    t = DataType(
        ident=f'fn{pretty_print_signature(sig)}',
        generic=len(generic_types) > 0,
    )

    if t not in TypeSignatures.keys():
        TypeSignatures[t] = sig

    assert t in TypeSignatures.keys()
    return t


def parse_type_list(
    start_tok: Token,
    tokens: List[Token],
    generic_types: List[DataType],
    delimiters=List[Keyword],
    allow_undefined_generics=False,
) -> Tuple[Token, List[DataType]]:

    assert len(tokens) > 0

    types: List[DataType] = []
    while True:
        end_tok, input_tokens = tokens_until_keywords(
            start_tok, tokens, delimiters + [Keyword.WITH, Keyword.FN_TYPE])

        for tok in input_tokens:
            if tok.value in TypeDict.keys():
                types.append(TypeDict[tok.value])
            elif tok.value in [T.ident for T in generic_types]:
                types.append(
                    [T for T in generic_types if T.ident == tok.value][0])
            elif allow_undefined_generics:
                types.append(DataType(tok.value, generic=True))
            else:
                compiler_error(
                    False,
                    tok,
                    f"Unrecognized token `{tok.typ}:{tok.value}` in type list."
                )

        if end_tok.typ in delimiters:
            break
        elif end_tok.typ == Keyword.WITH:
            t = parse_type_from_with_block(
                end_tok, tokens, generic_types)
            types.append(t)
        elif end_tok.typ == Keyword.FN_TYPE:
            fn_ptr_t = parse_fn_ptr_type(
                end_tok, tokens, [], generic_types)
            types.append(fn_ptr_t)
            assert fn_ptr_t in TypeSignatures
        else:
            assert False, "... unreachable"

    # print(pretty_print_arg_list(types))

    return (end_tok, types)


def parse_fn_signature(
    start_tok: Token,
    tokens: List[Token],
    generic_types: List[DataType]
) -> Tuple[Token, Signature]:

    puts: ArgList = []
    pops: ArgList = []
    tok, pops = parse_type_list(
        start_tok,
        tokens,
        generic_types,
        [Keyword.ARROW, Keyword.DO, Keyword.END]
    )

    if tok.typ == Keyword.ARROW:
        end_tok, puts = parse_type_list(
            tok,
            tokens,
            generic_types,
            [Keyword.DO, Keyword.END]
        )

        compiler_error(
            len(puts) > 0,
            tok,
            "Invalid signature. Arrow must be followed with at least one output type",
            ["Consider removing the `->` if the function leaves nothing on the stack"]
        )
        tok = end_tok

    return tok, Signature(pops, puts)


def convert_to_concrete_arg_list(gen_list: List[DataType], concrete_types: List[DataType], generics: List[DataType]):
    type_list: List[DataType] = []
    for T in gen_list:
        if T.generic and T.struct:

            assert T in StructMembers.keys()
            assert T in StructGenerics.keys()

            T_members = StructMembers[T]
            T_generics = StructGenerics[T]
            T_assignment_tag = T.ident[indexOf(T.ident, '<'):]

            for G in T_generics:
                T_assignment_tag = T_assignment_tag.replace(
                    G.ident, concrete_types[indexOf(T_generics, G)].ident)

            T_concrete_name = f"{T.ident[:indexOf(T.ident, '<')]}{T_assignment_tag}"
            T_concrete_members = convert_to_concrete_arg_list(
                T_members, concrete_types, T_generics)

            TypeDict[T_concrete_name] = DataType(
                ident=T_concrete_name,
                generic=any([T.generic for T in T_concrete_members]),
                struct=True,
                size=sum([T.size for T in T_concrete_members])
            )
            StructMembers[TypeDict[T_concrete_name]] = T_concrete_members
            type_list.append(TypeDict[T_concrete_name])
        elif T.generic and T in TypeSignatures.keys():
            sig = TypeSignatures[T]
            pops = convert_to_concrete_arg_list(
                sig.pops, concrete_types, generics)
            puts = convert_to_concrete_arg_list(
                sig.puts, concrete_types, generics
            )
            concrete_sig = Signature(pops, puts)
            concrete_fn_ptr_name = f"fn{pretty_print_signature(concrete_sig)}"
            if concrete_fn_ptr_name not in TypeDict:
                TypeDict[concrete_fn_ptr_name] = DataType(
                    ident=concrete_fn_ptr_name,
                    generic=any(T.generic for T in concrete_types),
                    fnptr=T.fnptr
                )

                TypeSignatures[TypeDict[concrete_fn_ptr_name]] = concrete_sig

            type_list.append(TypeDict[concrete_fn_ptr_name])
        elif T.generic:
            type_list.append(concrete_types[indexOf(generics, T)])
        else:
            type_list.append(T)

    return type_list


def fn_reduce_generics(fn_name: str, fn_meta: FunctionMeta, concrete_types: List[DataType], generic_types: List[DataType]) -> str:
    if '<' in fn_name and '>' == fn_name[-1]:
        fn_generics = fn_meta[fn_name].generics
        fn_assignment_tag = fn_name[indexOf(fn_name, '<'):]
        for G in fn_generics:
            fn_assignment_tag = fn_assignment_tag.replace(
                G.ident, concrete_types[indexOf(fn_generics, G)].ident)
        concrete_name = f"{fn_name[:indexOf(fn_name, '<')]}{fn_assignment_tag}"
    else:
        concrete_name = f"{fn_name}{pretty_print_arg_list(concrete_types, '<', '>')}"

    if concrete_name not in fn_meta.keys():
        fn_sig = fn_meta[fn_name].signature
        fn_generics = fn_meta[fn_name].generics
        pops = convert_to_concrete_arg_list(
            fn_sig.pops,
            concrete_types,
            fn_generics
        )
        puts = convert_to_concrete_arg_list(
            fn_sig.puts,
            concrete_types,
            fn_generics
        )
        concrete_sig = Signature(pops, puts)

        fn_meta[concrete_name] = Function(
            concrete_name,
            deepcopy(fn_meta[fn_name].tok),
            generics=generic_types,
            signature=concrete_sig,
            stub=False,
            program=deepcopy(fn_meta[fn_name].program)
        )

        if len(generic_types) == 0:
            for op in fn_meta[concrete_name].program:
                if op.op == OpType.CALL:
                    assert op.operand is not None
                    if len(fn_meta[op.operand].generics) > 0:
                        op.operand = fn_reduce_generics(
                            op.operand, fn_meta, concrete_types, generic_types)

    return concrete_name


def call_generic_fn_with(
    start_tok: Token,
    tokens: List[Token],
    fn_meta: FunctionMeta,
    concrete_types: List[DataType],
    generic_types: List[DataType],
    program: Program
):
    fn_tok = tokens.pop()

    compiler_error(
        fn_tok.typ == MiscTokenKind.WORD,
        fn_tok,
        f"Expected a function name after `WITH`-`DO` block, but found `{fn_tok.typ}:{fn_tok.value}` instead."
    )
    fn_name = fn_tok.value
    assert isinstance(fn_name, str)

    compiler_error(
        fn_name in fn_meta.keys(),
        fn_tok,
        f"Unkonwn function `{fn_name}`."
    )

    compiler_error(
        len(fn_meta[fn_name].generics) > 0,
        fn_tok,
        f"Cannot assign generics to non-generic function `{fn_name}`"
    )

    compiler_error(
        not fn_meta[fn_name].stub,
        fn_tok,
        f"Cannot call a function stub."
    )

    compiler_error(
        len(fn_meta[fn_name].generics) == len(concrete_types),
        fn_tok,
        f"""Invalid number of types for generic assignment.
    [Note]: Function `{fn_name}` is generic over {pretty_print_arg_list(fn_meta[fn_name].generics)}
    [Note]: The following types were provided: {pretty_print_arg_list(concrete_types)}"""
    )

    compiler_error(
        not any(T.generic and T not in generic_types for T in concrete_types),
        fn_tok,
        f"""Calling generic functions requires providing concrete types.
    [Note]: The following types are generic: {pretty_print_arg_list([T for T in concrete_types if T.generic])}"""
    )

    concrete_name = fn_reduce_generics(
        fn_name, fn_meta, concrete_types, generic_types)

    program.append(Op(
        OpType.CALL,
        fn_tok,
        operand=concrete_name
    ))


def parse_fn_from_tokens(
    start_tok: Token,
    tokens: List[Token],
    generic_types: List[DataType],
    fn_meta: FunctionMeta,
    const_values: ConstMap,
    reserved_memory: MemoryMap
):
    assert start_tok.typ == Keyword.FN

    compiler_error(
        len(tokens) > 0,
        start_tok,
        f"Expected function name, found end of file"
    )

    name_tok = tokens.pop()

    compiler_error(
        name_tok.typ == MiscTokenKind.WORD,
        name_tok,
        f"Expected function name, found {name_tok.typ} instead"
    )

    assert isinstance(name_tok.value, str)
    fn_name = name_tok.value

    if not (fn_name in fn_meta.keys() and fn_meta[fn_name].stub):
        check_for_name_conflict(
            fn_name,
            name_tok,
            fn_meta,
            const_values,
            reserved_memory,
        )

    compiler_error(
        len(tokens) > 0,
        name_tok,
        "Expected function signature, but found end of file instead"
    )

    tok, signature = parse_fn_signature(start_tok, tokens, generic_types)
    program: Program = []

    # Chedk for redefinitions
    if fn_name in fn_meta.keys():
        if tok.typ == Keyword.END:
            compiler_error(
                fn_name not in fn_meta.keys(),
                start_tok,
                f"Cannot pre-define a function more than once.",
                [f"Function `{fn_name}` initially defined here: {fn_meta[fn_name].tok.loc}"]
            )
        else:
            compiler_error(
                fn_meta[fn_name].stub,
                fn_meta[fn_name].tok,
                f"Redefinition of function `{fn_name}`.",
                [f"Function `{fn_name}` initially defined here: {fn_meta[fn_name].tok.loc}"]
            )

            stub_sig = fn_meta[fn_name].signature
            compiler_error(
                signature == fn_meta[fn_name].signature,
                start_tok,
                f"Function signature for {fn_name} must match pre-declaration.",
                [
                    f"Initially defined here: {fn_meta[fn_name].tok.loc}",
                    f"Expected Signature: {pretty_print_arg_list(stub_sig.pops)} -> {pretty_print_arg_list(stub_sig.puts)}",
                    f"Found Signature: {pretty_print_arg_list(signature.pops)} -> {pretty_print_arg_list(signature.puts)}"
                ]
            )

    check_for_name_conflict(
        fn_name,
        tok,
        fn_meta,
        const_values,
        reserved_memory,
        ignore_fn=True
    )

    # Add fn to meta for recursion purposes.
    fn_meta[fn_name] = Function(
        fn_name,
        start_tok,
        generic_types,
        signature,
        stub=tok.typ == Keyword.END,
        program=program
    )

    if tok.typ == Keyword.DO:

        tok_to_end = parse_tokens_until_keywords(
            tokens,
            [Keyword.END],
            generic_types,
            program,
            fn_meta,
            const_values,
            reserved_memory,
        )

        compiler_error(
            isinstance(tok_to_end, Token),
            name_tok,
            "Unclosed Function definition. Expected `END`, but found end of file instead"
        )
        assert isinstance(tok_to_end, Token)

        compiler_error(
            tok_to_end.typ == Keyword.END,
            tok_to_end,
            "Unclosed Function definition. Expected `END`, but found end of file instead"
        )

        program.append(Op(
            op=OpType.RETURN,
            operand=None,
            tok=tok_to_end
        ))


def generate_read_write_fns(start_tok: Token, new_struct: DataType, members: ArgList) -> List[Token]:
    assert new_struct.struct

    tokens: List[Token] = []
    name = new_struct.ident
    loc = start_tok.loc

    tokens += tokenize_string(f"fn !{name} {name} ptr -> ptr do")
    tokens += tokenize_string(f"push split pop")
    m = members.copy()
    while len(m) > 0:
        t = m.pop()
        tokens += tokenize_string(f"!{t.ident}")
    tokens += tokenize_string("end")

    tokens += tokenize_string(f"fn @{name} ptr -> {name} ptr do")
    m = members.copy()
    m.reverse()
    while len(m) > 0:
        t = m.pop()
        tokens += tokenize_string(f"@{t.ident}")
    tokens += tokenize_string(f"push cast({name}) pop")
    tokens += tokenize_string("end")

    for tok in tokens:
        tok.loc = loc

    return tokens


def generate_accessor_tokens(start_tok: Token, new_struct: DataType, members: ArgList) -> List[Token]:

    assert new_struct.struct

    tokens: List[Token] = []

    name = new_struct.ident
    loc = start_tok.loc

    for i, typ in enumerate(members):
        tokens += tokenize_string(f"fn {name}.{i} {name} -> {typ.ident} do")
        tokens += tokenize_string(f"split")
        for _ in range(i, len(members)-1):
            tokens += tokenize_string("drop")

        for _ in range(i):
            tokens += tokenize_string("swap drop")
        tokens += tokenize_string("end")

    for tok in tokens:
        tok.loc = loc

    tokens += generate_read_write_fns(start_tok, new_struct, members)

    tokens.reverse()

    return tokens


def flatten_types(args: ArgList) -> ArgList:
    types: List[DataType] = []
    for T in args:
        if T.struct:
            types += flatten_types(StructMembers[T])
            types.append(T)
        elif T in TypeSignatures.keys():
            types += flatten_types(TypeSignatures[T].pops)
            types += flatten_types(TypeSignatures[T].puts)
        else:
            types.append(T)

    return list(set(types))


def parse_struct_from_tokens(
    start_tok: Token,
    tokens: List[Token],
    generic_types: List[DataType],
    fn_meta: FunctionMeta,
    const_values: ConstMap,
    reserved_memory: MemoryMap,
):
    assert start_tok.typ == Keyword.STRUCT

    compiler_error(
        len(tokens) > 0,
        start_tok,
        f"Expected struct name, but found end of file instead"
    )

    name_tok = tokens.pop()
    compiler_error(
        name_tok.typ == MiscTokenKind.WORD,
        name_tok,
        f"Expected struct name, but found {name_tok.typ}:{name_tok.value} instead"
    )

    check_for_name_conflict(
        name_tok.value,
        name_tok,
        fn_meta,
        const_values,
        reserved_memory,
    )

    compiler_error(
        len(tokens) > 0,
        name_tok,
        f"Expected struct body, but found end of file instead"
    )

    compiler_error(
        len(generic_types) == len(set(generic_types)),
        start_tok,
        f"Generic identifiers must be unique in struct definition"
    )

    assert isinstance(name_tok.value, str)

    end_tok, members = parse_type_list(
        name_tok, tokens, generic_types, [Keyword.END]
    )

    compiler_error(
        len(members) > 0,
        end_tok,
        "Structs must have at least one member"
    )

    # Todo make it so that this can nest indefiniately.

    for T in generic_types:
        compiler_error(
            T in flatten_types(members),
            start_tok,
            f"""Unused generic identifier `{T.ident}`.
    [Note]: Consider removing it from the `WITH` statement above"""
        )

    new_struct = DataType(
        ident=name_tok.value,
        struct=True,
        generic=len(generic_types) > 0,
        size=sum(member.size for member in members)
    )
    TypeDict[new_struct.ident] = new_struct
    StructMembers[new_struct] = members

    if len(generic_types) == 0:
        generated_tokens = generate_accessor_tokens(
            start_tok, new_struct, members)
        tokens += generated_tokens
    else:
        StructGenerics[new_struct] = generic_types


def parse_if_block_from_tokens(
    start_tok: Token,
    tokens: List[Token],
    generic_types: List[DataType],
    program: Program,
    fn_meta: FunctionMeta,
    const_values: ConstMap,
    reserved_memory: MemoryMap,
):
    assert start_tok.typ == Keyword.IF
    jumps: List[int] = []

    program.append(Op(
        op=OpType.NOP,
        operand=None,
        tok=start_tok
    ))

    compiler_error(
        len(tokens) > 0,
        start_tok,
        "Unclosed `IF` block. Expected `DO`, but found end of file"
    )

    tok_do = parse_tokens_until_keywords(
        tokens,
        [Keyword.DO],
        generic_types,
        program,
        fn_meta,
        const_values,
        reserved_memory,
    )

    compiler_error(
        tok_do is not None,
        start_tok,
        "Expected token, found end of file instead"
    )
    assert isinstance(tok_do, Token)

    compiler_error(
        tok_do.typ == Keyword.DO,
        tok_do,
        f"Expected Keyword `DO` after `IF`"
    )
    compiler_error(
        len(tokens) > 0,
        tok_do,
        f"Unclosed `IF` block. Expected `END` after `DO`"
    )

    jumps.append(len(program))
    program.append(Op(
        op=OpType.JUMP_COND,
        operand=None,
        tok=tok_do
    ))

    # Handle all else cases until end
    expected_keywords = [Keyword.ELSE, Keyword.END]
    while len(tokens) > 0:
        tok = parse_tokens_until_keywords(
            tokens,
            expected_keywords,
            generic_types,
            program,
            fn_meta,
            const_values,
            reserved_memory,
        )

        compiler_error(
            tok is not None,
            start_tok,
            "Expected token, found end of file instead"
        )
        assert isinstance(tok, Token)
        assert len(
            expected_keywords) == 2, "Exhaustive handling of keywords in `IF` block"

        assert len(jumps) <= 2
        # Assign jump destinations
        while len(jumps) > 0:
            idx = jumps.pop()

            assert program[idx].tok.typ in [Keyword.DO, Keyword.ELSE], \
                "Pointed to the wrong place"

            assert program[idx].operand == None, "Already filled in jump dest"

            if program[idx].tok.typ == Keyword.DO and tok.typ == Keyword.ELSE:
                program[idx].operand = len(program) + 1
            elif program[idx].tok.typ == Keyword.ELSE and tok.typ == Keyword.ELSE:
                program[idx].operand = len(program)
            elif tok.typ == Keyword.END:
                program[idx].operand = len(program)
            else:
                assert False, "Unreachable..."

        if tok.typ == Keyword.ELSE:
            # Add the else to the jump stack
            jumps.append(len(program))
            program.append(Op(
                op=OpType.JUMP,
                operand=None,
                tok=tok
            ))

            # Look for DO or END
            tok_next = parse_tokens_until_keywords(
                tokens,
                [Keyword.DO, Keyword.END],
                generic_types,
                program,
                fn_meta,
                const_values,
                reserved_memory,
            )

            compiler_error(
                isinstance(tok_next, Token),
                tok,
                f"Expected Keyword `DO` or `END` after `ELSE`, but found end of file instead"
            )
            assert isinstance(tok_next, Token)
            compiler_error(
                tok_next.typ in [Keyword.DO, Keyword.END],
                tok_next,
                f"Expected Keyword `DO` or `END` after `ELSE`"
            )

            # If do, add it to the jump stack and continue
            if tok_next.typ == Keyword.DO:
                jumps.append(len(program))
                program.append(Op(
                    op=OpType.JUMP_COND,
                    operand=None,
                    tok=tok_next
                ))

            # If end, cleanup and exit
            elif tok_next.typ == Keyword.END:
                assert len(jumps) == 1
                idx = jumps.pop()

                assert program[idx].tok.typ == Keyword.ELSE
                assert program[idx].operand == None

                program[idx].operand = len(program)
                program.append(Op(
                    op=OpType.JUMP,
                    operand=len(program) + 1,
                    tok=tok_next
                ))
                return

            else:
                assert False, f"Unreachable..., {tok_next}"

        elif tok.typ == Keyword.END:
            program.append(Op(
                op=OpType.JUMP,
                operand=len(program) + 1,
                tok=tok
            ))
            return
        else:
            assert False, "Unreachable..."


def parse_while_block_from_tokens(
    start_tok: Token,
    tokens: List[Token],
    generic_types: List[DataType],
    program: Program,
    fn_meta: FunctionMeta,
    const_values: ConstMap,
    reserved_memory: MemoryMap,
):
    assert start_tok.typ == Keyword.WHILE

    do_tok_loc: int = 0
    program.append(Op(
        op=OpType.NOP,
        operand=start_tok.typ,
        tok=start_tok,
    ))
    start_loc = len(program) - 1

    compiler_error(
        len(tokens) > 0,
        start_tok,
        f"Unclosed `WHILE` block. Expected `DO`, but found end of file instead"
    )

    tok = parse_tokens_until_keywords(
        tokens,
        [Keyword.DO],
        generic_types,
        program,
        fn_meta,
        const_values,
        reserved_memory,
    )

    compiler_error(
        tok is not None,
        start_tok,
        "Expected token, found end of file instead"
    )
    assert isinstance(tok, Token)
    compiler_error(
        tok.typ == Keyword.DO,
        tok,
        f"Expected Keyword `DO`. Found {tok.typ} instead"
    )
    compiler_error(
        len(tokens) > 0,
        tok,
        "Unclosed `WHILE` block. Expected `END` but found end of file instead"
    )

    do_tok_loc = len(program)
    program.append(Op(
        op=OpType.JUMP_COND,
        operand=None,
        tok=tok
    ))

    tok = parse_tokens_until_keywords(
        tokens,
        [Keyword.END],
        generic_types,
        program,
        fn_meta,
        const_values,
        reserved_memory,
    )

    compiler_error(
        tok is not None,
        start_tok,
        "Expected token, found end of file instead"
    )
    assert isinstance(tok, Token)

    compiler_error(
        tok.typ == Keyword.END,
        tok,
        f"Expected Keyword `END`. Found {tok.typ} instead"
    )

    assert program[do_tok_loc].operand == None
    assert program[do_tok_loc].op == OpType.JUMP_COND

    program.append(Op(
        op=OpType.JUMP,
        tok=tok,
        operand=start_loc,
    ))
    program[do_tok_loc].operand = len(program)


def program_from_prelude() -> Tuple[Program, FunctionMeta, MemoryMap]:
    prelude_tokens = tokenize("prelude/prelude.tlp")
    program, fn_meta, reserved_mem = program_from_tokens(prelude_tokens)
    global PRELUDE_SIZE
    PRELUDE_SIZE = len(program)
    return (program, fn_meta, reserved_mem)


def program_from_tokens(
    tokens: List[Token],
    program: Program = [],
    fn_meta: FunctionMeta = {},
    reserved_memory: MemoryMap = {},
) -> Tuple[Program, FunctionMeta, MemoryMap]:

    const_values: ConstMap = {}
    tokens.reverse()

    expected_keywords: List[Keyword] = [
        Keyword.WITH,
        Keyword.FN,
        Keyword.STRUCT,
        Keyword.CONST
    ]

    while len(tokens) > 0:

        types: List[DataType] = []
        tok = parse_tokens_until_keywords(
            tokens,
            expected_keywords,
            types,
            program,
            fn_meta,
            const_values,
            reserved_memory
        )

        assert isinstance(tok, Token)
        assert len(
            expected_keywords) == 4, "Exhaustive handling of expected keywords"

        if tok.typ == Keyword.WITH:
            tok, types = parse_with_block_from_tokens(
                tok,
                tokens,
            )

        if tok.typ == Keyword.FN:
            parse_fn_from_tokens(
                tok,
                tokens,
                types,
                fn_meta,
                const_values,
                reserved_memory
            )
        elif tok.typ == Keyword.STRUCT:
            parse_struct_from_tokens(
                tok,
                tokens,
                types,
                fn_meta,
                const_values,
                reserved_memory
            )
        elif tok.typ == Keyword.CONST:
            parse_const_expr(
                tok,
                tokens,
                fn_meta,
                const_values,
                reserved_memory
            )
        elif tok.typ == Keyword.DO and len(types) > 0:
            call_generic_fn_with(tok, tokens, fn_meta, types, [], program)
        elif len(tokens) == 0:
            break
        else:
            assert False, f"Unreachable... {tok}:{tok.value} "

    return (program, fn_meta, reserved_memory)


def asm_header(out):
    out.write("segment .text\n")
    # Credit to Tsoding from his first video on impl porth.
    # TODO: Revisit this code gand do my own impl.
    out.write("putu:\n")
    out.write("    mov     r9, -3689348814741910323\n")
    out.write("    sub     rsp, 40\n")
    out.write("    mov     BYTE [rsp+31], 10\n")
    out.write("    lea     rcx, [rsp+30]\n")
    out.write(".L2:\n")
    out.write("    mov     rax, rdi\n")
    out.write("    lea     r8, [rsp+32]\n")
    out.write("    mul     r9\n")
    out.write("    mov     rax, rdi\n")
    out.write("    sub     r8, rcx\n")
    out.write("    shr     rdx, 3\n")
    out.write("    lea     rsi, [rdx+rdx*4]\n")
    out.write("    add     rsi, rsi\n")
    out.write("    sub     rax, rsi\n")
    out.write("    add     eax, 48\n")
    out.write("    mov     BYTE [rcx], al\n")
    out.write("    mov     rax, rdi\n")
    out.write("    mov     rdi, rdx\n")
    out.write("    mov     rdx, rcx\n")
    out.write("    sub     rcx, 1\n")
    out.write("    cmp     rax, 9\n")
    out.write("    ja      .L2\n")
    out.write("    lea     rax, [rsp+32]\n")
    out.write("    mov     edi, 1\n")
    out.write("    sub     rdx, rax\n")
    out.write("    xor     eax, eax\n")
    out.write("    lea     rsi, [rsp+32+rdx]\n")
    out.write("    mov     rdx, r8\n")
    out.write("    mov     rax, 1\n")
    out.write("    syscall\n")
    out.write("    add     rsp, 40\n")
    out.write("    ret\n")
    out.write("global _start\n")
    out.write("_start:\n")
    out.write("    mov rax, ret_stack_end\n")
    out.write("    mov [ret_stack_rsp], rax\n")


def asm_exit(out, strings, reserved_memory: MemoryMap):
    out.write("segment .data\n")
    for i, data in enumerate(strings):
        out.write(f"    string_{i}: db {','.join(map(hex, list(data)))}\n")
    out.write("segment .bss\n")
    out.write(f"    ret_stack_rsp: resq 1\n")
    out.write(f"    ret_stack: resb {8192}\n")
    out.write(f"    ret_stack_end:\n")
    for k, v in reserved_memory.items():
        out.write(f"    mem_{k}: resb {v[0]}\n")


def type_check_cond_jump(
    ip: int,
    program: Program,
    fn_meta: FunctionMeta,
    current_stack: List[DataType],
    return_stack: List[DataType]
) -> Tuple[int, List[List[DataType]], List[List[DataType]]]:
    assert program[
        ip].op == OpType.JUMP_COND, f"Bug in type checking pointed to the wrong place. {program[ip]}"
    evaluate_signature(
        program[ip],
        signatures[OpType.JUMP_COND],
        current_stack,
        return_stack,
    )

    end_ip, stack_if_true, ret_stack_if_true = type_check_program(
        program,
        fn_meta,
        start_from=ip + 1,
        starting_stack=current_stack.copy(),
        starting_rstack=return_stack.copy(),
        break_on=[lambda op: op.tok.typ == Keyword.END]
    )

    assert program[end_ip].tok.typ == Keyword.END

    jump_loc = program[ip].operand
    assert isinstance(jump_loc, int)

    false_path_ip, stack_if_false, ret_stack_if_false = type_check_program(
        program,
        fn_meta,
        start_from=jump_loc,
        starting_stack=current_stack.copy(),
        starting_rstack=return_stack.copy(),
        break_on=[
            lambda op: op.tok.typ == Keyword.END,
            lambda op: op.op == OpType.JUMP_COND
        ]
    )

    if program[false_path_ip].tok.typ == Keyword.END and program[end_ip].tok.typ == Keyword.END and false_path_ip == end_ip:
        return (end_ip, [stack_if_true, stack_if_false], [ret_stack_if_true, ret_stack_if_false])

    elif program[false_path_ip].op == OpType.JUMP_COND:
        _, branch_types, ret_branch_types = type_check_cond_jump(
            false_path_ip,
            program,
            fn_meta,
            stack_if_false.copy(),
            ret_stack_if_false.copy(),
        )
        return (end_ip, [stack_if_true] + branch_types, [ret_stack_if_true] + ret_branch_types)

    else:
        assert False, f"Well this was unexpected... {end_ip} vs {false_path_ip} {program[false_path_ip].op}:{program[false_path_ip].operand}"


def pretty_print_arg_list(arg_list: List[DataType], open="[", close="]") -> str:
    s = open
    for t in arg_list[:-1]:
        s += f"{t.ident} "
    if len(arg_list) > 0:
        t = arg_list[-1]
        s += f"{t.ident}"
    s += close
    return s


def pretty_print_signature(sig: Signature, pops_open="(", pops_close=")", puts_open='[', puts_close=']') -> str:
    s = pretty_print_arg_list(sig.pops, open=pops_open, close=pops_close)
    s += " -> "
    s += pretty_print_arg_list(sig.puts, open=puts_open, close=puts_close)
    return s


def pretty_print_stack_options(possible_stacks: List[List[DataType]]) -> str:
    s = "\n"
    for i, stack in enumerate(possible_stacks):
        s += f"        Branch {i+1}: {pretty_print_arg_list(stack)}\n"
    return s


def type_check_if_block(ip: int, program: Program, fn_meta: FunctionMeta, current_stack: List[DataType], return_stack: List[DataType]) -> Tuple[int, List[DataType], List[DataType]]:
    possile_stacks: List[List[DataType]] = []

    assert program[ip].tok.typ == Keyword.IF, "Bug in type checking pointed to the wrong place."
    ip2, stack_before_jumpc, ret_stack_before_jumpc = type_check_program(
        program,
        fn_meta,
        start_from=ip+1,
        starting_stack=current_stack.copy(),
        starting_rstack=return_stack.copy(),
        break_on=[lambda op: op.op == OpType.JUMP_COND]
    )

    end_ip, possible_stacks, ret_possible_stacks = type_check_cond_jump(
        ip2,
        program,
        fn_meta,
        stack_before_jumpc.copy(),
        ret_stack_before_jumpc.copy()
    )

    match = True
    for i in range(len(possible_stacks)):
        for j in range(i+1, len(possible_stacks)):
            match &= possible_stacks[i] == possible_stacks[j]
            match &= ret_possible_stacks[i] == ret_possible_stacks[j]

    compiler_error(
        match,
        program[ip].tok,
        f"""
    Each branch of an IF Block must produce a similare stack.
    Possible outputs: {pretty_print_stack_options(possible_stacks)}
    Possible pushed values: {pretty_print_stack_options(ret_possible_stacks)}
        """
    )
    return (end_ip, possible_stacks[0], ret_possible_stacks[0])


def type_check_while_block(ip: int, program: Program, fn_meta: FunctionMeta, current_stack: List[DataType], return_stack: List[DataType]) -> Tuple[int, List[DataType], List[DataType]]:

    assert program[ip].tok.typ == Keyword.WHILE, "Bug in type checking pointed to the wrong spot."

    ip2, current_stack, ret_current_stack = type_check_program(
        program,
        fn_meta,
        start_from=ip+1,
        starting_stack=current_stack.copy(),
        starting_rstack=return_stack.copy(),
        break_on=[lambda op: op.op == OpType.JUMP_COND]
    )

    evaluate_signature(
        program[ip2],
        signatures[OpType.JUMP_COND],
        current_stack,
        ret_current_stack
    )

    stack_before = current_stack.copy()
    ret_stack_before = ret_current_stack.copy()

    end_ip, final_stack, final_ret_stack = type_check_program(
        program,
        fn_meta,
        start_from=ip2+1,
        starting_stack=current_stack.copy(),
        starting_rstack=ret_current_stack.copy(),
        break_on=[lambda op: op.tok.typ == Keyword.END]
    )

    compiler_error(
        stack_before == final_stack and ret_stack_before == final_ret_stack,
        program[ip].tok,
        f"""
    While loops cannot change the stack outside of the loop
    [Note]: Stack at start of loop : {pretty_print_arg_list(stack_before)}
    [Note]: Stack at end of loop   : {pretty_print_arg_list(final_stack)}
    [Note]: Pushed at start of loop: {pretty_print_arg_list(ret_stack_before)}
    [Note]: Pushed at end of loop  : {pretty_print_arg_list(final_ret_stack)}
        """
    )

    return (end_ip+1, final_stack, final_ret_stack)


def assign_sizes(op: Op, sig: Signature):

    if op.operand == Intrinsic.DUP:
        assert len(sig.pops) == 1
        assert op.tok.value == None, f"{op.tok.loc} -- {op.op}:{op.operand} - {op.tok.value}"
        op.tok.value = sig.pops[0].size
    elif op.operand == Intrinsic.SWAP:
        assert len(sig.pops) == 2
        assert op.tok.value == None
        op.tok.value = (sig.pops[0].size, sig.pops[1].size)
    elif op.operand == Intrinsic.DROP:
        assert len(sig.pops) == 1
        assert op.tok.value == None
        op.tok.value = sig.pops[0].size
    elif op.operand == Intrinsic.RPUSH:
        assert len(sig.pops) == 1
        assert op.tok.value == None
        op.tok.value = sig.pops[0].size
    if op.operand == Intrinsic.RPOP:
        assert len(sig.rpops) == 1
        assert op.tok.value == None
        op.tok.value = sig.rpops[0].size


def check_args_match(args: ArgList, stack: ArgList) -> bool:
    return [DataType(T.ident, T.generic, T.struct, T.size) for T in args] == \
           [DataType(T.ident, T.generic, T.struct, T.size) for T in stack]


def evaluate_signature(op: Op, sig: Signature, type_stack: List[DataType], return_stack: List[DataType]):

    assert True if len(sig.pops) == 0 else [not T.generic for T in sig.pops], \
        f"{op.op}:{op.operand} Expected non-generic signature. Pops {pretty_print_arg_list(sig.pops)}"
    assert True if len(sig.puts) == 0 else [not T.generic for T in sig.puts], \
        f"{op.op}:{op.operand} Expected non-generic signature. Puts {pretty_print_arg_list(sig.puts)}"
    assert True if len(sig.rpops) == 0 else [not T.generic for T in sig.rpops], \
        f"{op.op}:{op.operand} Expected non-generic signature. Rpops {pretty_print_arg_list(sig.rpops)}"
    assert True if len(sig.rputs) == 0 else [not T.generic for T in sig.rputs], \
        f"{op.op}:{op.operand} Expected non-generic signature. Rputs {pretty_print_arg_list(sig.rputs)}"

    assign_sizes(op, sig)
    if len(sig.pops) > 0:
        if check_args_match(sig.pops, type_stack[-len(sig.pops):]):
            for _ in sig.pops:
                type_stack.pop()
        else:
            compiler_error(
                False,
                op.tok,
                f"""
    Didn't find a matching signature for {op.op}:{op.operand}.
    Expected: {pretty_print_arg_list(sig.pops)}
    Found   : {pretty_print_arg_list(type_stack[-len(sig.pops):])}
                """
            )

    if len(sig.rpops) > 0:
        if check_args_match(sig.rpops, return_stack[-len(sig.rpops):]):
            for _ in sig.rpops:
                return_stack.pop()
        else:
            compiler_error(
                False,
                op.tok,
                f"""
    Didn't find a matching signature for {op.op}:{op.operand}.
    Expected Return Stack: {pretty_print_arg_list(sig.rpops)}
    Found Return Stack   : {pretty_print_arg_list(return_stack[-len(sig.rpops):])}
                """
            )
    type_stack += sig.puts
    return_stack += sig.rputs


def generate_concrete_struct(op: Op, type_stack: List[DataType]) -> DataType:
    assert isinstance(op.operand, Intrinsic)
    assert op.tok.value in TypeDict.keys()
    gen_struct = TypeDict[op.tok.value]
    assert gen_struct in StructGenerics.keys()
    assert gen_struct in StructMembers.keys()

    generics = StructGenerics[gen_struct]
    members = StructMembers[gen_struct]

    compiler_error(
        len(type_stack) >= len(members),
        op.tok,
        f"""Cannot assign generics during cast, insufficient elements on the stack.
    [Note]: Expected: {pretty_print_arg_list(members)}
    [Note]: Found:    {pretty_print_arg_list(type_stack)}"""
    )

    concrete_members: List[DataType] = []
    generic_map: Dict[DataType, DataType] = {}
    for i, t in enumerate(members):
        if t.generic and t in TypeSignatures.keys():
            compiler_error(
                type_stack[-len(members) + i] in TypeSignatures,
                op.tok,
                f"""Generic Type resolution Failure.
    [Note]: Expected funtion poiner like: {t.ident}.
    [Note]: Type `{type_stack[-len(members) + i]}` was found instead."""
            )

            pops = convert_to_concrete_arg_list(
                TypeSignatures[t].pops,
                TypeSignatures[type_stack[-len(members) + i]].pops,
                generics
            )
            puts = convert_to_concrete_arg_list(
                TypeSignatures[t].puts,
                TypeSignatures[type_stack[-len(members) + i]].puts,
                generics
            )

            for concrete_pop, gen_pop in zip(pops, TypeSignatures[t].pops):
                if gen_pop.generic:
                    if gen_pop in generic_map.keys():
                        compiler_error(
                            generic_map[gen_pop] == concrete_pop,
                            op.tok,
                            f"""Generic Type Resolution Failure.
    [Note]: Generic `{T.ident}` was assigned `{generic_map[gen_pop].ident}` yet `{concrete_pop.ident}` was found
    [Note]: Signature: {pretty_print_arg_list(members)}
    [Note]: Stack    : {pretty_print_arg_list(type_stack[-len(members):])}"""
                        )
                    else:
                        generic_map[gen_pop] = concrete_pop

            for concrete_put, gen_put in zip(puts, TypeSignatures[t].puts):
                if gen_put.generic:
                    if gen_put in generic_map.keys():
                        compiler_error(
                            generic_map[gen_put] == concrete_put,
                            op.tok,
                            f"""Generic Type Resolution Failure.
    [Note]: Generic `{T.ident}` was assigned `{generic_map[gen_put].ident}` yet `{concrete_put.ident}` was found
    [Note]: Signature: {pretty_print_arg_list(members)}
    [Note]: Stack    : {pretty_print_arg_list(type_stack[-len(members):])}"""
                        )
                    else:
                        generic_map[gen_put] = concrete_put

            sig = Signature(pops, puts)
            concrete_name = f"fn{pretty_print_signature(sig)}"
            if concrete_name not in TypeDict.keys():
                TypeDict[concrete_name] = DataType(
                    ident=concrete_name,
                    fnptr=type_stack[-len(members) + i].fnptr
                )

                TypeSignatures[TypeDict[concrete_name]] = sig

            concrete_members.append(TypeDict[concrete_name])

        elif t.generic:

            assert t in generics
            concrete_type = type_stack[-len(members) + i]
            if t in generic_map.keys():
                compiler_error(
                    generic_map[t] == concrete_type,
                    op.tok,
                    f"""Generic Type Resolution Failure.
    [Note]: Generic `{T.ident}` was assigned `{generic_map[t].ident}` yet `{concrete_type.ident}` was found
    [Note]: Signature: {pretty_print_arg_list(members)}
    [Note]: Stack: {pretty_print_arg_list(type_stack[-len(members):])}"""
                )
            else:
                generic_map[t] = concrete_type
            concrete_members.append(concrete_type)
        else:
            concrete_members.append(t)

    concrete_struct_name = f"{gen_struct.ident}{pretty_print_arg_list(list(generic_map.values()), open='<', close='>')}"
    if concrete_struct_name not in TypeDict.keys():
        TypeDict[concrete_struct_name] = DataType(
            ident=concrete_struct_name,
            struct=True,
            size=sum(t.size for t in concrete_members)
        )

        StructMembers[TypeDict[concrete_struct_name]] = concrete_members

    return TypeDict[concrete_struct_name]


def struct_is_instance(T: DataType, S: DataType) -> bool:
    assert S.struct
    return T.ident.startswith(S.ident)


def assign_generics(op: Op, sig: Signature, type_stack: List[DataType], return_stack: List[DataType]) -> Signature:

    n_args_expected = len(sig.pops)
    n_rargs_expected = len(sig.rpops)

    compiler_error(
        len(type_stack) >= n_args_expected,
        op.tok,
        f"""
        Operation {op.op}: {op.operand} Requires {n_args_expected} arguments. {len(type_stack)} found.
        [Note]: Expected {pretty_print_arg_list(sig.pops)}
        [Note]: Found    {pretty_print_arg_list(type_stack)}
        """
    )

    compiler_error(
        len(return_stack) >= n_rargs_expected,
        op.tok,
        f"""
        Operation {op.op}: {op.operand} Requires {n_rargs_expected} pushed arguments. {len(return_stack)} found.
        [Note]: Expected {pretty_print_arg_list(sig.rpops)}
        [Note]: Found    {pretty_print_arg_list(return_stack)}
        """
    )

    generic_map: Dict[DataType, DataType] = {}

    for i, T in enumerate(sig.pops):
        if T.generic:
            if not T in generic_map:
                if T.struct:
                    compiler_error(
                        struct_is_instance(
                            type_stack[-n_args_expected:][i],
                            T
                        ),
                        op.tok,
                        f"""
    Didn't find a matching signature for {op.op}: {op.operand}.
    Expected: {pretty_print_arg_list(sig.pops)}
    Found: {pretty_print_arg_list(type_stack[-len(sig.pops):])}"""
                    )

                generic_map[T] = type_stack[-n_args_expected:][i]
            else:
                compiler_error(
                    generic_map[T] == type_stack[-n_args_expected:][i],
                    op.tok,
                    f"""
    Generic Type Resolution Failure.
    [Note]: Generic `{T.ident}` was assigned `{generic_map[T].ident}` yet `{type_stack[-n_args_expected:][i].ident}` was found
    [Note]: Signature: {pretty_print_arg_list(sig.pops)}
    [Note]: Stack: {pretty_print_arg_list(type_stack[-n_args_expected:])}
                    """
                )

    for i, T in enumerate(sig.rpops):
        if T.generic:
            if not T in generic_map:
                generic_map[T] = return_stack[-n_rargs_expected:][i]
            else:
                compiler_error(
                    generic_map[T] == return_stack[-n_rargs_expected:][i],
                    op.tok,
                    f"""
    Generic Type Resolution Failure.
    [Note]: Generic `{T.ident}` was assigned `{generic_map[T].ident}` yet `{return_stack[-n_rargs_expected:][i].ident}` was found
    [Note]: Signature: {pretty_print_arg_list(sig.rpops)}
    [Note]: Stack: {pretty_print_arg_list(return_stack[-n_rargs_expected:])}
                    """
                )

    for T in sig.pops:
        compiler_error(
            not T.generic or T in generic_map.keys(),
            op.tok,
            f"Undefined generic `{T.ident}`"
        )

    new_sig = Signature(
        pops=[T if not T.generic else generic_map[T] for T in sig.pops],
        puts=[T if not T.generic else generic_map[T] for T in sig.puts],
        rpops=[T if not T.generic else generic_map[T] for T in sig.rpops],
        rputs=[T if not T.generic else generic_map[T] for T in sig.rputs],
    )
    return new_sig


def type_check_fn(fn: Function, fn_meta: FunctionMeta):

    _, out_stack, out_ret_stack = type_check_program(
        fn.program,
        fn_meta,
        0,  # Start 1 past the fn name token
        starting_stack=fn.signature.pops.copy(),
        starting_rstack=[],
    )

    puts = fn.signature.puts
    assert isinstance(puts, list)
    compiler_error(
        out_stack == fn.signature.puts,
        fn.tok,
        f"""
    Function `{fn.ident}` output doesn't match signature.
    [Note]: Expected Output Stack: {pretty_print_arg_list(puts)}
    [Note]: Actual Output Stack: {pretty_print_arg_list(out_stack)}
        """
    )

    compiler_error(
        len(out_ret_stack) == 0,
        fn.tok,
        f"""Function `{fn.ident}` doesn't leave an empty return stack.
    [Note]: Expected Empty Return Stack, but found: {pretty_print_arg_list(out_ret_stack)}"""
    )


def type_check_program(
    program: Program,
    fn_meta: FunctionMeta,
    start_from: int = 0,
    starting_stack: List[DataType] = [],
    starting_rstack: List[DataType] = [],
    break_on: List[Callable[[Op], bool]] = [],
    skip_fn_eval: bool = True
) -> Tuple[int, List[DataType], List[DataType]]:

    type_stack: List[DataType] = starting_stack.copy()
    ret_stack: List[DataType] = starting_rstack.copy()
    ip = start_from

    if not skip_fn_eval:
        for fn in fn_meta.values():
            if len(fn.generics) != 0:
                continue
            type_check_fn(fn, fn_meta)
    while ip < len(program):
        op = program[ip]
        if any([cond(op) for cond in break_on]):
            break

        assert op.op != OpType.JUMP_COND, f"{op.tok.loc} Type Checking error: Unhandled conditional jump"

        if op.op == OpType.NOP and op.tok.typ == Keyword.IF:
            ip, type_stack, ret_stack = type_check_if_block(
                ip,
                program,
                fn_meta,
                type_stack,
                ret_stack,
            )
            ip += 1
        elif op.op == OpType.NOP and op.tok.typ == Keyword.WHILE:
            ip, type_stack, ret_stack = type_check_while_block(
                ip,
                program,
                fn_meta,
                type_stack,
                ret_stack
            )
        else:
            # Make sure that there are sufficient arguments on the stack
            if op.op == OpType.INTRINSIC:
                assert isinstance(op.operand, Intrinsic)

                assert N_DYNAMIC_INTRINSICS == 6
                if op.operand == Intrinsic.CAST:
                    compiler_error(
                        op.tok.value in TypeDict.keys(),
                        op.tok,
                        f"Unrecognized Data Type `{op.tok.value}`"
                    )

                    t = TypeDict[op.tok.value]
                    if t.struct:
                        assert t in StructMembers.keys(), f"{t}"
                        if t.generic:
                            t = generate_concrete_struct(op, type_stack)
                        sig = Signature(
                            pops=StructMembers[t].copy(),
                            puts=[t]
                        )
                    else:
                        if t.ident == INT.ident:
                            if len(type_stack) > 0:
                                compiler_error(
                                    type_stack[-1] in [INT, BOOL, PTR],
                                    op.tok,
                                    f"Only `INT`, `BOOl`, and `PTR` types can be cast to int. Found `{type_stack[-1]}`"
                                )

                            sig = Signature(
                                pops=[T],
                                puts=[INT]
                            )
                        elif t.ident == PTR.ident:
                            sig = Signature(
                                pops=[INT],
                                puts=[PTR]
                            )
                        elif t.ident == BOOL.ident:
                            sig = Signature(
                                pops=[INT],
                                puts=[BOOL]
                            )
                        else:
                            compiler_error(
                                False,
                                op.tok,
                                f"Cannot cast to built in type {t.ident}"
                            )
                elif op.operand == Intrinsic.CAST_TUPLE:
                    global TUPLE_IDENT_COUNT
                    n = op.tok.value

                    # Make sure there are enough elements on the stack to group together
                    compiler_error(
                        len(type_stack) >= n,
                        op.tok,
                        f"Grouping {n} elements requires there be at least {n} elements on the stack"
                    )

                    # Determine the size and the type of the struct
                    tuple_size = 0
                    members = type_stack[-n:].copy()
                    for t in members:
                        tuple_size += t.size

                    tuple = DataType(
                        ident=f"Anonstruct_{TUPLE_IDENT_COUNT}_{n}",
                        struct=True,
                        size=tuple_size
                    )

                    assert tuple.ident not in TypeDict
                    assert tuple not in StructMembers

                    TypeDict[tuple.ident] = tuple
                    StructMembers[tuple] = members

                    sig = Signature(
                        pops=members.copy(),
                        puts=[tuple]
                    )

                    TUPLE_IDENT_COUNT += 1
                elif op.operand == Intrinsic.INNER_TUPLE:
                    compiler_error(
                        len(type_stack) > 0,
                        op.tok,
                        "Cannot get group inner element, stack is empty"
                    )

                    t = type_stack[-1]

                    compiler_error(
                        t.ident.startswith("Anonstruct_"),
                        op.tok,
                        f"Expected to find an `GROUP` on the top of the stack. Found {t.ident} instead"
                    )

                    assert t in StructMembers.keys()

                    compiler_error(
                        len(StructMembers[t]) > op.tok.value,
                        op.tok,
                        f"`GROUP` only has {len(StructMembers[t])} members. Cannot access element {op.tok.value}"
                    )

                    sig = Signature(
                        pops=[t],
                        puts=[StructMembers[t][op.tok.value]]
                    )

                    op.tok.value = (op.tok.value, t)
                elif op.operand == Intrinsic.SPLIT:
                    compiler_error(
                        len(type_stack) > 0,
                        op.tok,
                        "Cannot split struct/group, stack is empty"
                    )

                    t = type_stack[-1]

                    compiler_error(
                        t.struct,
                        op.tok,
                        f"{op.op}:{op.operand} expects a `Struct` on the top of the stack. Found an `{t.ident}` instead"
                    )

                    assert t in StructMembers.keys()
                    sig = Signature(
                        pops=[t],
                        puts=StructMembers[t]
                    )
                elif op.operand == Intrinsic.ADDR_OF:
                    compiler_error(
                        op.tok.value in fn_meta.keys(),
                        op.tok,
                        f"Unknown function `{op.tok.value}`"
                    )

                    fn_ptr_t = DataType(
                        ident=f'fn{pretty_print_signature(fn_meta[op.tok.value].signature)}',
                        generic=len(fn_meta[op.tok.value].generics) > 0,
                        fnptr=op.tok.value
                    )

                    if fn_ptr_t not in TypeSignatures:
                        TypeSignatures[fn_ptr_t] = fn_meta[op.tok.value].signature

                    sig = Signature(
                        pops=[],
                        puts=[fn_ptr_t]
                    )
                elif op.operand == Intrinsic.CALL:
                    compiler_error(
                        len(type_stack) > 0,
                        op.tok,
                        f"{op.op}:{op.operand} Expects the top element of the stack to be a function pointer. Stack was empty."
                    )

                    t = type_stack.pop()
                    compiler_error(
                        t in TypeSignatures.keys(),
                        op.tok,
                        f"Expected function pointer on the top of the stack, but found `{t.ident}` instead"
                    )
                    sig = TypeSignatures[t]
                else:
                    sig = signatures[op.operand]

            elif op.op == OpType.CALL:
                assert isinstance(op.operand, str)
                sig = fn_meta[op.operand].signature
            else:
                sig = signatures[op.op]

            sig = assign_generics(op, sig, type_stack, ret_stack)
            evaluate_signature(op, sig, type_stack, ret_stack)

            if (op.op == OpType.JUMP):
                assert isinstance(op.operand, int)
                ip = op.operand
            else:
                ip += 1

    if ip == len(program):
        ip -= 1
    return (ip, type_stack, ret_stack)


def print_ret_stack_rsp(out):
    out.write(f"    mov rbx, [ret_stack_rsp]\n")
    out.write(f"    mov rdi, rbx\n")
    out.write(f"    call putu\n")
    out.write(f"    mov rdi, [rbx]\n")
    out.write(f"    call putu\n")


def op_drop_to_asm(out, N):
    for i in range(N):
        out.write("    pop     rax\n")


def op_ret_stack_push(out, N):
    # print_ret_stack_rsp(out)
    for i in range(N):
        out.write(f"    sub     qword [ret_stack_rsp], 8\n")
        out.write(f"    pop     rax\n")
        out.write(f"    mov     rbx, [ret_stack_rsp]\n")
        out.write(f"    mov     [rbx], rax\n")
        # print_ret_stack_rsp(out)
    # out.write(f"    mov rdi, [ret_stack_rsp]\n")
    # out.write(f"    call putu\n")


# [..... (rsp)]
#          ^  ^end


def op_ret_stack_pop(out, N):
    for i in range(N):
        out.write(f"    mov     rbx, [ret_stack_rsp]\n")
        out.write(f"    mov     rax, [rbx]\n")
        out.write(f"    push    rax\n")
        out.write(f"    add     qword [ret_stack_rsp], 8\n")
        # print_ret_stack_rsp(out)


def op_swap_to_asm(out, ip, n, m):
    out.write(f"    mov     rdi, {n}\n")
    out.write(f"loop_{ip}:\n")
    out.write(f"    mov     rbx, rsp\n")
    out.write(f"    mov     rcx, rsp\n")
    out.write(f"    add     rbx, {8 * (n + m - 1)}\n")
    out.write(f"    add     rcx, {8 * (n + m - 2)}\n")
    out.write(f"    mov     rsi, 0\n")  # How many
    out.write(f"rotate_{ip}:\n")
    out.write(f"    mov     rax, [rbx]\n")
    out.write(f"    xchg    rax, [rcx]\n")
    out.write(f"    mov     [rbx], rax\n")
    out.write(f"    sub     rbx, 8\n")
    out.write(f"    sub     rcx, 8\n")
    out.write(f"    add     rsi, 1\n")
    out.write(f"    cmp     rsi, {n + m - 1}\n")
    out.write(f"    jl      rotate_{ip}\n")
    out.write(f"    sub     rdi, 1\n")
    out.write(f"    cmp     rdi, 0\n")
    out.write(f"    jg      loop_{ip}\n")


def compile_ops(out, ip, program: Program, fn_meta, reserved_memory, strings) -> int:
    start_ip = ip
    for op in program:
        out.write(f"op_{ip}: ")
        if op.op == OpType.PUSH_UINT:
            out.write(f";; --- {op.op} {op.operand} --- \n")
            out.write(f"    push    {op.operand}\n")
        elif op.op == OpType.PUSH_BOOL:
            assert isinstance(op.operand, bool)
            out.write(f";; --- {op.op} {op.operand} --- \n")
            out.write(f"    push    {int(op.operand)}\n")
        elif op.op == OpType.PUSH_PTR:
            assert isinstance(op.operand, str)
            assert op.operand in reserved_memory.keys()
            out.write(f";; --- {op.op} {op.operand} --- \n")
            out.write(f"    push    mem_{op.operand}\n")
        elif op.op == OpType.PUSH_STRING:
            assert isinstance(op.operand, str)
            out.write(f";; --- {op.op} --- \n")
            string = op.operand + '\0'
            encoded = string.encode('utf-8')
            out.write(f"    mov     rax, {len(encoded) - 1}\n")
            out.write(f"    push    rax\n")
            out.write(f"    push    string_{len(strings)}\n")
            strings.append(encoded)
        elif op.op == OpType.INTRINSIC:

            if op.operand == Intrinsic.ADD:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")
                out.write(f"    pop     rbx\n")
                out.write(f"    add     rax, rbx\n")
                out.write(f"    push    rax\n")
            elif op.operand == Intrinsic.SUB:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")
                out.write(f"    pop     rbx\n")
                out.write(f"    sub     rbx, rax\n")
                out.write(f"    push    rbx\n")
            elif op.operand == Intrinsic.MUL:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rcx\n")
                out.write(f"    pop     rax\n")
                out.write(f"    mul     rcx\n")
                out.write(f"    push    rax\n")
            elif op.operand == Intrinsic.DIV:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    mov     rdx, 0\n")
                out.write(f"    pop     rcx\n")
                out.write(f"    pop     rax\n")
                out.write(f"    div     rcx\n")
                out.write(f"    push    rax\n")
            elif op.operand == Intrinsic.MOD:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    mov     rdx, 0\n")
                out.write(f"    pop     rcx\n")
                out.write(f"    pop     rax\n")
                out.write(f"    div     rcx\n")
                out.write(f"    push    rdx\n")
            elif op.operand == Intrinsic.LSL:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rcx\n")
                out.write(f"    pop     rbx\n")
                out.write(f"    shl     rbx, cl\n")
                out.write(f"    push    rbx\n")
            elif op.operand == Intrinsic.AND:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")
                out.write(f"    pop     rbx\n")
                out.write(f"    and     rbx, rax\n")
                out.write(f"    push    rbx\n")
            elif op.operand == Intrinsic.OR:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")
                out.write(f"    pop     rbx\n")
                out.write(f"    or      rbx, rax\n")
                out.write(f"    push    rbx\n")
            elif op.operand == Intrinsic.BW_AND:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")
                out.write(f"    pop     rbx\n")
                out.write(f"    and     rbx, rax\n")
                out.write(f"    push    rbx\n")
            elif op.operand == Intrinsic.PUTU:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rdi\n")
                out.write(f"    call    putu\n")
            elif op.operand == Intrinsic.DUP:
                out.write(
                    f";; --- {op.op} {op.operand} {op.tok.value} --- \n")
                out.write(f"    mov     rbx, rsp\n")
                out.write(f"    mov     rcx, rsp\n")
                out.write(f"    add     rcx, {(op.tok.value - 1) * 8} \n")
                out.write(f"loop_{ip}:\n")
                out.write(f"    mov     rax, [rcx]\n")
                out.write(f"    push    rax\n")
                out.write(f"    sub     rcx, 8\n")
                out.write(f"    cmp     rbx, rcx\n")
                out.write(f"    jle     loop_{ip}\n")
            elif op.operand == Intrinsic.DROP:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                op_drop_to_asm(out, op.tok.value)
            elif op.operand == Intrinsic.SWAP:
                out.write(
                    f";; --- {op.op} {op.operand} {op.tok.value} --- \n")
                n, m = op.tok.value
                op_swap_to_asm(out, ip, n, m)
            elif op.operand == Intrinsic.RPUSH:
                out.write(
                    f";; --- {op.op} {op.operand} {op.tok.value} --- \n")
                op_ret_stack_push(out, op.tok.value)
            elif op.operand == Intrinsic.RPOP:
                out.write(
                    f";; --- {op.op} {op.operand} {op.tok.value} --- \n")
                op_ret_stack_pop(out, op.tok.value)
            elif op.operand == Intrinsic.SPLIT:
                out.write(f";; --- {op.op} {op.operand} --- \n")
            elif op.operand == Intrinsic.READ64:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")
                out.write(f"    mov     rax, [rax]\n")
                out.write(f"    push    rax\n")
            elif op.operand == Intrinsic.READ8:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")
                out.write(f"    xor     rbx, rbx\n")
                out.write(f"    mov     bl, [rax]\n")
                out.write(f"    push    rbx\n")
            elif op.operand == Intrinsic.WRITE64:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop rax\n")
                out.write(f"    pop rbx\n")
                out.write(f"    mov [rax], rbx\n")
            elif op.operand == Intrinsic.WRITE8:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop rax\n")
                out.write(f"    pop rbx\n")
                out.write(f"    mov [rax], bl\n")
            elif op.operand == Intrinsic.EQ:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    mov     rcx, 0\n")
                out.write(f"    mov     rdx, 1\n")
                out.write(f"    pop     rbx\n")
                out.write(f"    pop     rax\n")
                out.write(f"    cmp     rax, rbx\n")
                out.write(f"    cmove   rcx, rdx\n")
                out.write(f"    push    rcx\n")
            elif op.operand == Intrinsic.LE:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    mov     rcx, 0\n")
                out.write(f"    mov     rdx, 1\n")
                out.write(f"    pop     rbx\n")
                out.write(f"    pop     rax\n")
                out.write(f"    cmp     rax, rbx\n")
                out.write(f"    cmovle  rcx, rdx\n")
                out.write(f"    push    rcx\n")
            elif op.operand == Intrinsic.LT:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    mov     rcx, 0\n")
                out.write(f"    mov     rdx, 1\n")
                out.write(f"    pop     rbx\n")
                out.write(f"    pop     rax\n")
                out.write(f"    cmp     rax, rbx\n")
                out.write(f"    cmovl   rcx, rdx\n")
                out.write(f"    push    rcx\n")
            elif op.operand == Intrinsic.GT:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    mov     rcx, 0\n")
                out.write(f"    mov     rdx, 1\n")
                out.write(f"    pop     rbx\n")
                out.write(f"    pop     rax\n")
                out.write(f"    cmp     rax, rbx\n")
                out.write(f"    cmovg   rcx, rdx\n")
                out.write(f"    push    rcx\n")
            elif op.operand == Intrinsic.CAST:
                out.write(
                    f";; --- {op.op} {op.operand} {op.tok.value} --- \n")
            elif op.operand == Intrinsic.CAST_TUPLE:
                out.write(
                    f";; --- {op.op} {op.operand} {op.tok.value} --- \n")
            elif op.operand == Intrinsic.INNER_TUPLE:
                out.write(
                    f";; --- {op.op} {op.operand} {op.tok.value} --- \n")
                index = op.tok.value[0]
                members = StructMembers[op.tok.value[1]].copy()

                for i in range(index, len(members)-1):
                    out.write(f";; Drop\n")
                    op_drop_to_asm(out, members.pop().size)

                for i in range(index):
                    out.write(f";; SWAP DROP {i}\n")
                    op_swap_to_asm(
                        out, f"{ip}_{i}", members[-2].size, members[-1].size)
                    op_drop_to_asm(out, members[-2].size)
                    del members[-2]
            elif op.operand == Intrinsic.SIZE_OF:
                compiler_error(
                    op.tok.value in TypeDict.keys(),
                    op.tok,
                    f"Cannot get size of unknown type `{op.tok.value}`."
                )
                out.write(
                    f";; --- {op.op} {op.operand} {op.tok.value} --- \n")
                out.write(
                    f"    push    {TypeDict[op.tok.value].size * 8}\n")
            elif op.operand == Intrinsic.ADDR_OF:
                out.write(
                    f";; --- {op.op} {op.operand} {op.tok.value} --- \n")
                out.write(
                    f"    mov     rax, fn_{indexOf(list(fn_meta), op.tok.value)}\n")
                out.write(f"    push    rax\n")
            elif op.operand == Intrinsic.CALL:
                out.write(
                    f";; --- {op.op} {op.operand} {op.tok.value} --- \n")
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rbx\n")
                out.write(f"    mov     rax, rsp\n")
                out.write(f"    mov     rsp, [ret_stack_rsp]\n")
                out.write(f"    call    rbx\n")
                out.write(f"    mov     [ret_stack_rsp], rsp\n")
                out.write(f"    mov     rsp, rax\n")
            elif op.operand == Intrinsic.SYSCALL0:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")  # SYSCALL NUM
                out.write(f"    syscall\n")
                out.write(f"    push    rax\n")  # push result
            elif op.operand == Intrinsic.SYSCALL1:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")  # SYSCALL NUM
                out.write(f"    pop     rdi\n")  # Arg 0
                out.write(f"    syscall\n")
                out.write(f"    push    rax\n")  # push result
            elif op.operand == Intrinsic.SYSCALL2:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")  # SYSCALL NUM
                out.write(f"    pop     rdi\n")  # Arg 0
                out.write(f"    pop     rsi\n")  # Arg 1
                out.write(f"    syscall\n")
                out.write(f"    push    rax\n")  # push result
            elif op.operand == Intrinsic.SYSCALL3:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")  # SYSCALL NUM
                out.write(f"    pop     rdi\n")  # Arg 0
                out.write(f"    pop     rsi\n")  # Arg 1
                out.write(f"    pop     rdx\n")  # Arg 2
                out.write(f"    syscall\n")
                out.write(f"    push    rax\n")  # push result
            elif op.operand == Intrinsic.SYSCALL4:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")  # SYSCALL NUM
                out.write(f"    pop     rdi\n")  # Arg 0
                out.write(f"    pop     rsi\n")  # Arg 1
                out.write(f"    pop     rdx\n")  # Arg 2
                out.write(f"    pop     r10\n")  # Arg 3
                out.write(f"    syscall\n")
                out.write(f"    push    rax\n")  # push result
            elif op.operand == Intrinsic.SYSCALL5:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")  # SYSCALL NUM
                out.write(f"    pop     rdi\n")  # Arg 0
                out.write(f"    pop     rsi\n")  # Arg 1
                out.write(f"    pop     rdx\n")  # Arg 2
                out.write(f"    pop     r10\n")  # Arg 3
                out.write(f"    pop     r8\n")   # Arg 4
                out.write(f"    syscall\n")
                out.write(f"    push    rax\n")  # push result
            elif op.operand == Intrinsic.SYSCALL6:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    pop     rax\n")  # SYSCALL NUM
                out.write(f"    pop     rdi\n")  # Arg 0
                out.write(f"    pop     rsi\n")  # Arg 1
                out.write(f"    pop     rdx\n")  # Arg 2
                out.write(f"    pop     r10\n")  # Arg 3
                out.write(f"    pop     r8\n")   # Arg 4
                out.write(f"    pop     r9\n")   # Arg 5
                out.write(f"    syscall\n")
                out.write(f"    push    rax\n")  # push result
            else:
                assert False, f"Unhandled Intrinsic: {op.operand}"
        elif op.op == OpType.JUMP_COND:
            out.write(f";; --- {op.op} {op.operand} --- \n")
            out.write(f"    pop     rax\n")
            out.write(f"    test    rax, rax\n")
            out.write(f"    jz      op_{start_ip + op.operand}\n")
        elif op.op == OpType.JUMP:
            out.write(f";; --- {op.op} {op.operand} --- \n")
            if ip + 1 != op.operand:
                out.write(f"    jmp     op_{start_ip + op.operand}\n")
        elif op.op == OpType.NOP:
            out.write("\n")
        elif op.op == OpType.RETURN:
            out.write(f";; --- {op.op} {op.operand} --- \n")
            out.write(f"    mov     rax, rsp\n")
            out.write(f"    mov     rsp, [ret_stack_rsp]\n")
            out.write(f"    ret\n")
        elif op.op == OpType.CALL:
            assert isinstance(op.operand, str)
            out.write(f";; --- {op.op} {op.operand} --- \n")
            out.write(f"    mov     rax, rsp\n")
            out.write(f"    mov     rsp, [ret_stack_rsp]\n")
            out.write(
                f"    call    fn_{indexOf(list(fn_meta), op.operand)}\n")
            out.write(f"    mov     [ret_stack_rsp], rsp\n")
            out.write(f"    mov     rsp, rax\n")
        else:
            print(f"Operation {op.op} is not supported yet")
            exit(1)
        ip += 1

    return ip


def compile_program(out_path: str, program: Program, fn_meta: FunctionMeta, reserved_memory: MemoryMap):

    strings: List[bytes] = []
    ip = 0

    with open(f"{out_path}.asm", 'w') as out:
        asm_header(out)

        ip = compile_ops(out, ip, program, fn_meta, reserved_memory, strings)
        out.write(f"op_{ip}:\n")
        ip += 1
        out.write("exit:\n")
        out.write("    mov rax, 60\n")
        out.write("    mov rdi, 0\n")
        out.write("    syscall\n")
        out.write("\n")
        out.write(
            f";; ---------------------- FUNCTIONS -------------------------\n")
        for i, fn in enumerate(fn_meta.values()):
            if len(fn.generics) != 0:
                continue

            out.write(f"fn_{i}:  ;; --- {fn.ident} ---\n")
            out.write(f"    mov     [ret_stack_rsp], rsp\n")
            out.write(f"    mov     rsp, rax\n")

            ip = compile_ops(out, ip, fn.program, fn_meta,
                             reserved_memory, strings)

        asm_exit(out, strings, reserved_memory)

    call(["nasm", "-felf64", f"{out_path}.asm"])
    call(["ld", "-o", f"{out_path}", f"{out_path}.o"])


def usage(msg: str):
    print("Usage:")
    print(f"    python3 {__file__} <FILE>")
    print(f"[Error]: {msg}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage("Must provide an file to compile")
        exit(1)

    filepath = sys.argv[1]
    tokens = tokenize(filepath)
    # print("-------------------------------------------")
    # for i, tok in enumerate(tokens):
    #     print(f"{i} -- {tok.typ}: {tok.value}")
    # print("-------------------------------------------")
    program, fn_meta, reserved_memory = program_from_prelude()
    program, fn_meta, reserved_memory = program_from_tokens(
        tokens,
        program, fn_meta, reserved_memory
    )
    # print("-------------------------------------------")
    # for ip, op in enumerate(program):
    #     print(f"{ip} -- {op.op}: {op.operand} TokenType: {op.tok.typ}")
    # print("-------------------------------------------")

    # print(f"Len Program: {len(program)}")

    ip, type_stack, ret_stack = type_check_program(
        program,
        fn_meta,
        skip_fn_eval=False
    )

    if len(program) > 0:

        compiler_error(
            len(type_stack) == 0,
            program[ip].tok,
            f"""Unhandled data on the datastack.
    [Note]: Expected an empty stack found: {pretty_print_arg_list(type_stack)}"""
        )

        compiler_error(
            len(ret_stack) == 0,
            program[ip].tok,
            f"""Unhandled pushed data.
    [Note]: Expected no data to be pushed found {pretty_print_arg_list(ret_stack)}"""
        )

        compile_program("output", program, fn_meta, reserved_memory)
    else:
        print("Empty Program...")
        exit(1)
