
from dataclasses import dataclass
from enum import Enum, auto
from subprocess import call
from typing import List, Union, Dict, Optional, Any, Tuple, Callable
from lexer import tokenize, Token, Intrinsic, MiscTokenKind, Keyword
import sys


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
    Ident: str
    Generic: bool = False
    Struct: bool = False
    Size: int = 1


INT = DataType("int")
BOOL = DataType("bool")
PTR = DataType("ptr")
STR = DataType("Str", Struct=True, Size=2)
T = DataType("T", Generic=True)
A = DataType("A", Generic=True)
B = DataType("B", Generic=True)
C = DataType("C", Generic=True)
D = DataType("D", Generic=True)
E = DataType("E", Generic=True)
F = DataType("F", Generic=True)

TUPLE_IDENT_COUNT = 0

TypeDict: Dict[str, DataType] = {
    "int": INT,
    "bool": BOOL,
    "ptr": PTR,
    "Str": STR,
}

ArgList = List[DataType]
StructMembers: Dict[DataType, ArgList] = {}
StructMembers[STR] = [INT, PTR]


@ dataclass
class Op():
    op: OpType
    tok: Token
    operand: Optional[Any]


@ dataclass
class Signature:
    pops: ArgList
    puts: Union[ArgList, Callable[[ArgList], Optional[ArgList]]]


@ dataclass
class Function:
    ident: str
    signature: Signature
    tok: Token
    start_ip: int
    end_ip: Optional[int]


FunctionMeta = Dict[str, Function]
Program = List[Op]
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
    Intrinsic.OR: Signature(pops=[INT, INT], puts=[INT]),
    Intrinsic.LSL: Signature(pops=[INT, INT], puts=[INT]),
    Intrinsic.EQ: Signature(pops=[INT, INT], puts=[BOOL]),
    Intrinsic.LE: Signature(pops=[INT, INT], puts=[BOOL]),
    Intrinsic.LT: Signature(pops=[INT, INT], puts=[BOOL]),
    Intrinsic.READ64: Signature(pops=[PTR], puts=[INT]),
    Intrinsic.READ8: Signature(pops=[PTR], puts=[INT]),
    Intrinsic.GT: Signature(pops=[INT, INT], puts=[BOOL]),
    Intrinsic.PUTU: Signature(pops=[INT], puts=[]),
    Intrinsic.DUP: Signature(pops=[T], puts=[T, T]),
    Intrinsic.DROP: Signature(pops=[T], puts=[]),
    Intrinsic.SWAP: Signature(pops=[A, B], puts=[B, A]),
    Intrinsic.SPLIT: Signature(pops=[T], puts=lambda List_T: StructMembers[List_T[0]] if List_T[0].Struct else None),
    Intrinsic.CAST_INT: Signature(pops=[T], puts=lambda pops: [INT] if pops[0] in [INT, BOOL, PTR] else None),
    Intrinsic.CAST_PTR: Signature(pops=[INT], puts=[PTR]),
    # Intrinsic.CAST_STRUCT creates signatures dynamically based on the struct
    # Intrinsic.INNER_TUPLE creates a signature dynamically based on the tuple on the top of the stack.
    # Intrinsic.CAST_TUPLE  creates signatures dynamically based on the number of elements asked to group

    Intrinsic.SYSCALL0: Signature(pops=[INT], puts=[INT]),
    Intrinsic.SYSCALL1: Signature(pops=[A, INT], puts=[INT]),
    Intrinsic.SYSCALL2: Signature(pops=[A, B, INT], puts=[INT]),
    Intrinsic.SYSCALL3: Signature(pops=[A, B, C, INT], puts=[INT]),
    Intrinsic.SYSCALL4: Signature(pops=[A, B, C, D, INT], puts=[INT]),
    Intrinsic.SYSCALL5: Signature(pops=[A, B, C, D, E, INT], puts=[INT]),
    Intrinsic.SYSCALL6: Signature(pops=[A, B, C, D, E, F, INT], puts=[INT]),

}

N_DYNAMIC_INTRINSICS = 3
N_IGNORED_OP = 1
N_SUM_IGNORE = N_DYNAMIC_INTRINSICS + N_IGNORED_OP
assert len(signatures) == len(OpType) - N_SUM_IGNORE + len(Intrinsic), \
    f"Not all OpTypes and Intrinsics have a signature. Expected {len(OpType) - N_SUM_IGNORE + len(Intrinsic)} Found {len(signatures)}"


def token_loc_str(token: Token):
    return f"{token.loc.file}:{token.loc.line}:{token.loc.column}"


def compiler_error(predicate: bool, token: Token, msg):
    if not predicate:
        print(
            f"{token_loc_str(token)} [ERROR]: {msg}", file=sys.stderr)
        exit(1)


def check_for_name_conflict(
    name: str,
    tok: Token,
    fn_meta: FunctionMeta,
    const_values: ConstMap,
    reserved_memory: MemoryMap
):
    if name in fn_meta.keys():
        compiler_error(
            False,
            tok,
            f"Redefinition of `{name}`. Previously defined here: {token_loc_str(fn_meta[name].tok)}"
        )

    if name in TypeDict.keys():
        compiler_error(
            False,
            tok,
            f"Redefinition of Type `{name}`."
        )

    if name in const_values.keys():
        compiler_error(
            False,
            tok,
            f"Redefinition of `{name}`. Previously defined here: {token_loc_str(const_values[name].tok)}"
        )

    if name in reserved_memory.keys():
        compiler_error(
            False,
            tok,
            f"Redefinition of `{name}`. Previously defined here: {token_loc_str(reserved_memory[name][1])}"
        )


class FnDefState(Enum):
    Start = auto()
    Name = auto()
    Inputs = auto()
    Outputs = auto()
    Open = auto()
    End = auto()


def parse_tokens_until_keywords(
    tokens: List[Token],
    expected_keywords: List[Keyword],
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
                compiler_error(
                    False,
                    tok,
                    f"Unhandled MiscTokenKind: {tok}"
                )

        elif type(tok.typ) == Intrinsic:
            compiler_error(
                tok.typ in Intrinsic,
                tok,
                f"Unrecognized Intrinsic {tok}"
            )

            if tok.typ == Intrinsic.CAST_TUPLE:
                compiler_error(
                    len(program) > 0,
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
                    program,
                    fn_meta,
                    const_values,
                    reserved_memory,
                )
            elif tok.typ == Keyword.WHILE:
                parse_while_block_from_tokens(
                    tok,
                    tokens,
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

    included_tokens = tokenize(include_str)
    included_tokens.reverse()

    tokens += included_tokens


def parse_fn_from_tokens(
    start_tok: Token,
    tokens: List[Token],
    program: Program,
    fn_meta: FunctionMeta,
    const_values: ConstMap,
    reserved_memory: MemoryMap
):
    signature = Signature(pops=[], puts=[])
    assert isinstance(signature.puts, list)
    start_loc = len(program)
    assert start_tok.typ == Keyword.FN
    program.append(Op(
        op=OpType.NOP,
        operand=None,
        tok=start_tok
    ))

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

    check_for_name_conflict(
        fn_name,
        name_tok,
        fn_meta,
        const_values,
        reserved_memory,
    )

    program[start_loc].operand = fn_name

    pops = True

    compiler_error(
        len(tokens) > 0,
        name_tok,
        "Expected function signature, but found end of file instead"
    )

    while len(tokens) > 0:
        tok = tokens.pop()

        if tok.typ == MiscTokenKind.WORD:
            if tok.value in TypeDict.keys():
                if pops:
                    signature.pops.append(TypeDict[tok.value])
                else:
                    signature.puts.append(TypeDict[tok.value])
            else:
                if pops:
                    signature.pops.append(
                        DataType(
                            Ident=tok.value,
                            Generic=True
                        )
                    )
                else:
                    signature.puts.append(
                        DataType(
                            Ident=tok.value,
                            Generic=True
                        )
                    )

        elif tok.typ == Keyword.ARROW:

            compiler_error(
                pops,
                tok,
                "Only one arrow statement can be used in function signature definition"
            )

            pops = False

        elif tok.typ == Keyword.DO:
            compiler_error(
                pops or len(signature.puts) > 0,
                tok,
                "Invalid signature. Arrow notation must be followed with at least one output type"
            )
            break

        else:
            compiler_error(
                False,
                tok,
                f"""Unexpected token in function signature definition: {tok.typ}:{tok.value}.
    [Note]: Expected type names, `ARROW` or `DO`."""
            )

    compiler_error(
        tok.typ == Keyword.DO,
        tok,
        "Expected `DO` after function signature. Found end of file instead"
    )

    fn_meta[fn_name] = Function(
        ident=fn_name,
        signature=signature,
        tok=start_tok,
        start_ip=start_loc,
        end_ip=None
    )

    tok_to_end = parse_tokens_until_keywords(
        tokens,
        [Keyword.END],
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

    program.append(Op(
        op=OpType.NOP,
        operand=Keyword.END,
        tok=tok_to_end
    ))

    end_ip = len(program)

    assert fn_name in fn_meta.keys()
    assert fn_meta[fn_name].end_ip == None
    fn_meta[fn_name].end_ip = end_ip


def generate_accessor_tokens(start_tok: Token, new_struct: DataType, members: ArgList) -> List[Token]:

    assert new_struct.Struct

    tokens: List[Token] = []

    name = new_struct.Ident
    loc = start_tok.loc
    for i, typ in enumerate(members):
        # Fn
        tokens.append(
            Token(
                typ=Keyword.FN,
                value=None,
                loc=loc
            )
        )

        tokens.append(Token(
            typ=MiscTokenKind.WORD,
            value=name+f".{i}",
            loc=loc
        ))

        tokens.append(Token(
            typ=MiscTokenKind.WORD,
            value=name,
            loc=loc
        ))

        tokens.append(Token(
            typ=Keyword.ARROW,
            value=None,
            loc=loc
        ))

        tokens.append(Token(
            typ=MiscTokenKind.WORD,
            value=typ.Ident,
            loc=loc
        ))

        tokens.append(Token(
            typ=Keyword.DO,
            value=None,
            loc=loc
        ))

        tokens.append(Token(
            typ=Intrinsic.SPLIT,
            value=None,
            loc=loc
        ))

        for _ in range(i, len(members)-1):
            tokens.append(Token(
                typ=Intrinsic.DROP,
                value=None,
                loc=loc
            ))

        for _ in range(i):
            tokens.append(Token(
                typ=Intrinsic.SWAP,
                value=None,
                loc=loc
            ))
            tokens.append(Token(
                typ=Intrinsic.DROP,
                value=None,
                loc=loc
            ))

        tokens.append(Token(
            typ=Keyword.END,
            value=None,
            loc=loc
        ))

    tokens.reverse()

    return tokens


def parse_struct_from_tokens(
    start_tok: Token,
    tokens: List[Token],
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

    assert isinstance(name_tok.value, str)

    members = []

    while len(tokens) > 0:
        tok = tokens.pop()

        if tok.typ == Keyword.END:
            break

        elif tok.typ == MiscTokenKind.WORD:
            assert isinstance(tok.value, str)
            if tok.value in TypeDict.keys():
                members.append(TypeDict[tok.value])
            else:
                compiler_error(
                    False,
                    tok,
                    f"Unknown type `{tok.value}` in struct definition"
                )
        else:
            compiler_error(
                False,
                tok,
                f"Unexpected token during struct definition. Expected type identifiers or `END`, but found {tok.typ}:{tok.value} instead"
            )

    compiler_error(
        tok.typ == Keyword.END,
        tok,
        f"Unclosed struct definition. Expected `END`, but found end of file instead"
    )

    compiler_error(
        len(members) > 0,
        tok,
        f"Structs must have at least one member"
    )

    new_struct = DataType(
        Ident=name_tok.value,
        Struct=True,
        Size=sum(member.Size for member in members)
    )
    TypeDict[new_struct.Ident] = new_struct
    StructMembers[new_struct] = members

    generated_tokens = generate_accessor_tokens(start_tok, new_struct, members)
    tokens += generated_tokens


def parse_if_block_from_tokens(
    start_tok: Token,
    tokens: List[Token],
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


def program_from_tokens(tokens: List[Token]) -> Tuple[Program, FunctionMeta, MemoryMap]:
    program: Program = []
    fn_meta: FunctionMeta = {}
    const_values: ConstMap = {}
    reserved_memory: MemoryMap = {}
    tokens.reverse()

    expected_keywords: List[Keyword] = [
        Keyword.FN, Keyword.STRUCT, Keyword.CONST]

    while len(tokens) > 0:

        tok = parse_tokens_until_keywords(
            tokens,
            expected_keywords,
            program,
            fn_meta,
            const_values,
            reserved_memory
        )

        assert isinstance(tok, Token)
        assert len(
            expected_keywords) == 3, "Exhaustive handling of expected keywords"

        if tok.typ == Keyword.FN:
            parse_fn_from_tokens(
                tok,
                tokens,
                program,
                fn_meta,
                const_values,
                reserved_memory
            )
        elif tok.typ == Keyword.STRUCT:
            parse_struct_from_tokens(
                tok,
                tokens,
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
    out.write("exit:\n")
    out.write("    mov rax, 60\n")
    out.write("    mov rdi, 0\n")
    out.write("    syscall\n")
    out.write("\n")
    out.write("segment .data\n")
    for i, data in enumerate(strings):
        out.write(f"    string_{i}: db {','.join(map(hex, list(data)))}\n")
    out.write("segment .bss\n")
    out.write(f"    ret_stack_rsp: resq 1\n")
    out.write(f"    ret_stack: resb {8192}\n")
    out.write(f"    ret_stack_end:\n")
    for k, v in reserved_memory.items():
        out.write(f"    mem_{k}: resb {v[0]}\n")


def type_check_cond_jump(ip: int, program: Program, fn_meta: FunctionMeta, current_stack: List[DataType]) -> Tuple[int, List[List[DataType]]]:
    assert program[
        ip].op == OpType.JUMP_COND, f"Bug in type checking pointed to the wrong place. {program[ip]}"
    evaluate_signature(
        program[ip],
        signatures[OpType.JUMP_COND],
        current_stack
    )

    end_ip, stack_if_true = type_check_program(
        program,
        fn_meta,
        start_from=ip + 1,
        starting_stack=current_stack.copy(),
        break_on=[lambda op: op.tok.typ == Keyword.END]
    )

    assert program[end_ip].tok.typ == Keyword.END

    jump_loc = program[ip].operand
    assert isinstance(jump_loc, int)

    false_path_ip, stack_if_false = type_check_program(
        program,
        fn_meta,
        start_from=jump_loc,
        starting_stack=current_stack.copy(),
        break_on=[
            lambda op: op.tok.typ == Keyword.END,
            lambda op: op.op == OpType.JUMP_COND
        ]
    )

    if program[false_path_ip].tok.typ == Keyword.END and program[end_ip].tok.typ == Keyword.END and false_path_ip == end_ip:
        return (end_ip, [stack_if_true, stack_if_false])

    elif program[false_path_ip].op == OpType.JUMP_COND:
        _, branch_types = type_check_cond_jump(
            false_path_ip,
            program,
            fn_meta,
            stack_if_false.copy()
        )
        return (end_ip, [stack_if_true] + branch_types)

    else:
        assert False, f"Well this was unexpected... {end_ip} vs {false_path_ip} {program[false_path_ip].op}:{program[false_path_ip].operand}"


def type_check_if_block(ip: int, program: Program, fn_meta: FunctionMeta, current_stack: List[DataType]) -> Tuple[int, List[DataType]]:
    possile_stacks: List[List[DataType]] = []

    assert program[ip].tok.typ == Keyword.IF, "Bug in type checking pointed to the wrong place."
    ip2, stack_before_jumpc = type_check_program(
        program,
        fn_meta,
        start_from=ip+1,
        starting_stack=current_stack.copy(),
        break_on=[lambda op: op.op == OpType.JUMP_COND]
    )

    end_ip, possible_stacks = type_check_cond_jump(
        ip2,
        program,
        fn_meta,
        stack_before_jumpc.copy()
    )

    match = True
    for i in range(len(possible_stacks)):
        for j in range(i+1, len(possible_stacks)):
            match &= possible_stacks[i] == possible_stacks[j]

    compiler_error(
        match,
        program[ip].tok,
        f"""
    Each branch of an IF Block must produce a similare stack.
    Possible outputs: {possible_stacks}
        """
    )
    return (end_ip, possible_stacks[0])


def type_check_while_block(ip: int, program: Program, fn_meta: FunctionMeta, current_stack: List[DataType]) -> Tuple[int, List[DataType]]:

    assert program[ip].tok.typ == Keyword.WHILE, "Bug in type checking pointed to the wrong spot."

    ip2, current_stack = type_check_program(
        program,
        fn_meta,
        start_from=ip+1,
        starting_stack=current_stack.copy(),
        break_on=[lambda op: op.op == OpType.JUMP_COND]
    )

    evaluate_signature(
        program[ip2],
        signatures[OpType.JUMP_COND],
        current_stack
    )

    stack_before = current_stack.copy()

    end_ip, final_stack = type_check_program(
        program,
        fn_meta,
        start_from=ip2+1,
        starting_stack=current_stack.copy(),
        break_on=[lambda op: op.tok.typ == Keyword.END]
    )
    compiler_error(
        stack_before == final_stack,
        program[ip].tok,
        f"""
    While loops cannot change the stack outside of the loop
    [Note]: Stack at start of loop: {stack_before}
    [Note]: Stack at end of loop  : {final_stack}
        """
    )

    return (end_ip+1, final_stack)


def evaluate_signature(op: Op, sig: Signature, type_stack: List[DataType]):
    n_args_expected = len(sig.pops)
    # Check to see if there the correct number of arguments
    compiler_error(
        len(type_stack) >= n_args_expected,
        op.tok,
        f"""
        Operation {op.op}:{op.operand} Requires {n_args_expected} arguments. {len(type_stack)} found.
        [Note]: Expected {sig.pops}
        [Note]: Found    {type_stack}
        """
    )

    generic_map: Dict[DataType, DataType] = {}

    pop_sig: List[DataType] = []

    if isinstance(sig.puts, list):

        puts = sig.puts.copy()

    if n_args_expected > 0:

        # Assign Generics
        for i, T in enumerate(sig.pops.copy()):

            if T.Generic:
                if not T in generic_map:
                    generic_map[T] = type_stack[-n_args_expected:][i]
                else:
                    compiler_error(
                        generic_map[T] == type_stack[-n_args_expected:][i],
                        op.tok,
                        f"""
        Generic Type Resolution Failure.
        [Note]: Generic `{T.Ident}` was assigned `{generic_map[T].Ident}` yet `{type_stack[-n_args_expected:][i].Ident}` was found
        [Note]: Signature: {sig.pops}
        [Note]: Stack    : {type_stack[-n_args_expected:]}
                        """
                    )
            pop_sig.append(T if not T.Generic else generic_map[T])

        if type_stack[-n_args_expected:] == pop_sig:
            if callable(sig.puts):
                assert not isinstance(sig.puts, list)
                puts = sig.puts(pop_sig.copy())
                compiler_error(
                    puts != None,
                    op.tok,
                    f"""Invalid inputs for {op.op}:{op.operand}.
    [Note]: Found {type_stack[-n_args_expected:]}"""
                )

            if op.operand == Intrinsic.DUP:
                assert len(pop_sig) == 1
                op.tok.value = pop_sig[0].Size
            elif op.operand == Intrinsic.SWAP:
                assert len(pop_sig) == 2
                op.tok.value = (pop_sig[0].Size, pop_sig[1].Size)
            elif op.operand == Intrinsic.DROP:
                assert len(pop_sig) == 1
                op.tok.value = pop_sig[0].Size
            for _ in sig.pops:
                type_stack.pop()
        else:
            compiler_error(
                False,
                op.tok,
                f"""
    Didn't find a matching signature for {op.op}:{op.operand}.
    Expected: {sig.pops}
    Found   : {type_stack[-n_args_expected:]}
                """
            )

    for T in puts:
        type_stack.append(T if not T.Generic else generic_map[T])


def type_check_program(
    program: Program,
    fn_meta: FunctionMeta,
    start_from: int = 0,
    starting_stack: List[DataType] = [],
    break_on: List[Callable[[Op], bool]] = [],
    skip_fn_eval: bool = True
) -> Tuple[int, List[DataType]]:
    type_stack: List[DataType] = starting_stack.copy()
    ip = start_from

    if not skip_fn_eval:
        for fn in fn_meta.values():
            end_ip, out_stack = type_check_program(
                program,
                fn_meta,
                fn.start_ip + 1,    # Start one instruction past the FN name marker
                starting_stack=fn.signature.pops.copy(),
                break_on=[
                    lambda op: op.op == OpType.NOP and op.tok.typ == Keyword.END
                ]
            )

            compiler_error(
                out_stack == fn.signature.puts,
                fn.tok,
                f"""
    Function `{fn.ident}` output doesn't match signature.
    [Note]: Expected Output Stack: {fn.signature.puts}
    [Note]: Actual Output Stack  : {out_stack}
                """
            )

    while ip < len(program):
        op = program[ip]
        if any([cond(op) for cond in break_on]):
            break

        assert op.op != OpType.JUMP_COND, f"{token_loc_str(op.tok)} Type Checking error: Unhandled conditional jump"

        if op.op == OpType.NOP and op.tok.typ == Keyword.IF:
            ip, type_stack = type_check_if_block(
                ip,
                program,
                fn_meta,
                type_stack
            )
            ip += 1
        elif op.op == OpType.NOP and op.tok.typ == Keyword.WHILE:
            ip, type_stack = type_check_while_block(
                ip,
                program,
                fn_meta,
                type_stack
            )
        else:
            # Make sure that there are sufficient arguments on the stack
            if op.op == OpType.INTRINSIC:
                assert isinstance(op.operand, Intrinsic)

                if op.tok.typ == Intrinsic.CAST_STRUCT:
                    compiler_error(
                        op.tok.value in TypeDict.keys(),
                        op.tok,
                        f"Unrecognized Data Type `{op.tok.value}`"
                    )
                    struct_t = TypeDict[op.tok.value]
                    assert struct_t.Struct and struct_t in StructMembers.keys()

                    sig = Signature(
                        pops=StructMembers[struct_t].copy(),
                        puts=[struct_t]
                    )
                elif op.tok.typ == Intrinsic.CAST_TUPLE:
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
                        tuple_size += t.Size

                    tuple = DataType(
                        Ident=f"AnonStruct_{TUPLE_IDENT_COUNT}_{n}",
                        Struct=True,
                        Size=tuple_size
                    )

                    assert tuple.Ident not in TypeDict
                    assert tuple not in StructMembers

                    TypeDict[tuple.Ident] = tuple
                    StructMembers[tuple] = members

                    sig = Signature(
                        pops=members.copy(),
                        puts=[tuple]
                    )

                    TUPLE_IDENT_COUNT += 1

                elif op.tok.typ == Intrinsic.INNER_TUPLE:

                    compiler_error(
                        len(type_stack) > 0,
                        op.tok,
                        "Cannot get group inner element, stack is empty"
                    )

                    t = type_stack[-1]

                    compiler_error(
                        t.Ident.startswith("AnonStruct_"),
                        op.tok,
                        f"Expected to find an `GROUP` on the top of the stack. Found {t.Ident} instead"
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

                else:
                    sig = signatures[op.operand]

            elif op.op == OpType.CALL:
                assert isinstance(op.operand, str)
                sig = fn_meta[op.operand].signature
            else:
                sig = signatures[op.op]

            evaluate_signature(op, sig, type_stack)

            if (op.op == OpType.JUMP):
                assert isinstance(op.operand, int)
                ip = op.operand
            elif op.op == OpType.NOP and op.operand in fn_meta.keys():
                assert isinstance(op.operand, str)
                tmp = fn_meta[op.operand].end_ip
                assert isinstance(tmp, int), f"{tmp}, {type(tmp)}"
                ip = tmp
            else:
                ip += 1

    if ip == len(program):
        ip -= 1
    return (ip, type_stack)


def op_drop_to_asm(out, N):
    for i in range(N):
        out.write("    pop     rax\n")


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


def compile_program(out_path: str, program: Program, fn_meta: FunctionMeta, reserved_memory: MemoryMap):

    strings: List[bytes] = []

    with open(f"{out_path}.asm", 'w') as out:
        asm_header(out)
        for ip, op in enumerate(program):
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
                elif op.operand == Intrinsic.LSL:
                    out.write(f";; --- {op.op} {op.operand} --- \n")
                    out.write(f"    pop     rcx\n")
                    out.write(f"    pop     rbx\n")
                    out.write(f"    shl     rbx, cl\n")
                    out.write(f"    push    rbx\n")
                elif op.operand == Intrinsic.OR:
                    out.write(f";; --- {op.op} {op.operand} --- \n")
                    out.write(f"    pop     rax\n")
                    out.write(f"    pop     rbx\n")
                    out.write(f"    or      rbx, rax\n")
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
                elif op.operand == Intrinsic.CAST_INT:
                    out.write(f";; --- {op.op} {op.operand} --- \n")
                elif op.operand == Intrinsic.CAST_PTR:
                    out.write(f";; --- {op.op} {op.operand} --- \n")
                elif op.operand == Intrinsic.CAST_STRUCT:
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
                        op_drop_to_asm(out, members.pop().Size)

                    for i in range(index):
                        out.write(f";; SWAP DROP {i}\n")
                        op_swap_to_asm(
                            out, f"{ip}_{i}", members[-2].Size, members[-1].Size)
                        op_drop_to_asm(out, members[-2].Size)
                        del members[-2]

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
                out.write(f"    jz      op_{op.operand}\n")
            elif op.op == OpType.JUMP:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                if ip + 1 != op.operand:
                    out.write(f"    jmp     op_{op.operand}\n")
            elif op.op == OpType.NOP:
                if op.tok.typ == Keyword.FN:
                    out.write(";; --- START OF FN ---\n")
                    assert isinstance(op.operand, str)
                    assert op.operand in fn_meta.keys()
                    after_fn_def = fn_meta[op.operand].end_ip
                    assert after_fn_def != None
                    out.write(f"jmp op_{after_fn_def}\n")
                    out.write(f"fn_{op.operand}:\n")
                    out.write(f"    mov     [ret_stack_rsp], rsp\n")
                    out.write(f"    mov     rsp, rax\n")
                else:
                    out.write("\n")
            elif op.op == OpType.RETURN:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    mov     rax, rsp\n")
                out.write(f"    mov     rsp, [ret_stack_rsp]\n")
                out.write(f"    ret\n")
            elif op.op == OpType.CALL:
                out.write(f";; --- {op.op} {op.operand} --- \n")
                out.write(f"    mov     rax, rsp\n")
                out.write(f"    mov     rsp, [ret_stack_rsp]\n")
                out.write(f"    call    fn_{op.operand}\n")
                out.write(f"    mov     [ret_stack_rsp], rsp\n")
                out.write(f"    mov     rsp, rax\n")
            else:
                print(f"Operation {op.op} is not supported yet")
                exit(1)
        out.write(f"op_{len(program)}:\n")
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

    program, fn_meta, reserved_memory = program_from_tokens(tokens)
    # print("-------------------------------------------")
    # for ip, op in enumerate(program):
    #     print(f"{ip} -- {op.op}: {op.operand} TokenType: {op.tok.typ}")
    # print("-------------------------------------------")

    ip, type_stack = type_check_program(program, fn_meta, skip_fn_eval=False)

    if len(program) > 0:

        compiler_error(
            len(type_stack) == 0,
            program[ip].tok,
            f"""Unhandled data on the datastack.
        [Note]: Expected an empty stack found {type_stack} instead"""
        )
        compile_program("output", program, fn_meta, reserved_memory)
    else:
        print("Empty Program...")
