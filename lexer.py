from os import unsetenv
from sys import argv
from enum import Enum, unique
from dataclasses import dataclass
import re
from typing import Union, Tuple, List, Dict, Optional, Any

"""Keywords

Define the list of keywords to search for with the lexer.
"""

WORD_REGEX = "[^0-9\s]\S*"


@unique
class Keyword(Enum):

    IF = "if"
    DO = "do"
    ELSE = "else"
    END = "end"
    WHILE = "while"
    FN = "fn"
    FN_TYPE = "&fn"
    STRUCT = "struct"
    ARROW = "->"
    INCLUDE = 'use'
    CONST = 'const'
    RESERVE = 'reserve'
    WITH = 'with'


"""Intrinsics

These are operations which may consume and or add to the stack.
"""


@unique
class Intrinsic(Enum):
    ADD = "\+"
    SUB = "-"
    DIV = "/"
    MUL = "\*"
    MOD = "%"
    EQ = "=="
    LE = "<="
    LSL = "<<"
    LT = "<"
    GT = ">"
    BW_AND = "&"
    READ64 = "@64"
    READ8 = "@8"
    WRITE64 = "!64"
    WRITE8 = "!8"
    OR = "or"
    AND = "and"
    RPUSH = "push"
    RPOP = 'pop'
    PUTU = "putu"
    DUP = 'dup'
    DROP = 'drop'
    SWAP = 'swap'
    SPLIT = 'split'
    CAST = f'cast\({WORD_REGEX}\)'
    INNER_TUPLE = 'group\.[0-9]+'
    CAST_TUPLE = 'group'
    SIZE_OF = f'SizeOf\({WORD_REGEX}\)'
    CALL = "call"
    ADDR_OF = f"&{WORD_REGEX}"
    SYSCALL0 = 'syscall0'
    SYSCALL1 = 'syscall1'
    SYSCALL2 = 'syscall2'
    SYSCALL3 = 'syscall3'
    SYSCALL4 = 'syscall4'
    SYSCALL5 = 'syscall5'
    SYSCALL6 = 'syscall6'


""" Other Tokens

This is how things like ints, strings, words are recognized by the lexer.
"""


@unique
class MiscTokenKind(Enum):
    INT = "[0-9]+"
    STRING = "\"[^\"]*\""
    BOOL = "true|false"
    WORD = "[^0-9\s]\S*"


""" Ignorable patterns
This enum defines what will be ignored
"""


class Ignore(Enum):
    WHITESPACE = "\s+"
    NEWLINE = "\n"
    COMMENT = "//.*"


TokenType = Union[Keyword, Intrinsic, MiscTokenKind]


@dataclass
class Loc:
    line: int
    column: int
    file: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class Token:
    typ: TokenType
    value: Any
    loc: Loc

    def __str__(self) -> str:
        if self.typ == MiscTokenKind.STRING:
            return f"{self.loc}: {self.typ.name}: \"{repr(self.value)[1:-1]}\""
        elif type(self.typ) == Keyword:
            return f"{self.loc}: Keyword: {self.typ.name}"
        elif type(self.typ) == Intrinsic:
            return f"{self.loc}: Intrinsic: {self.typ.name}"
        else:
            return f"{self.loc}: {self.typ.name}: {self.value}"


@dataclass
class PatternHolder:
    keywords: List[Keyword]
    Intrinsics: List[Intrinsic]
    intermediate_repr: List[MiscTokenKind]
    ignorable: List[Ignore]

    def search(self, line: str) -> Tuple[Optional[TokenType], re.Match]:
        # Search for ignorable values first
        for ig in self.ignorable:
            match = re.search("^" + ig.value, line)
            if match:
                return (None, match)

        # Search for Keywords next
        for kw in self.keywords:
            match = re.search("^" + kw.value + "(?=$|\s)", line)
            if match:
                return (kw, match)

        # Search for Intrinsics next
        for oper in self.Intrinsics:
            match = re.search("^" + oper.value + "(?=$|\s)", line)
            if match:
                return (oper, match)

        # Finally look for intermediate representation values
        for ir in self.intermediate_repr:
            match = re.search("^" + ir.value + "(?=$|\s)", line)
            if match:
                return (ir, match)

        print(f"Lexing Error: Unrecognized token at: {line}")
        exit(1)


patterns = PatternHolder(
    keywords=[kw for kw in Keyword],
    Intrinsics=[oper for oper in Intrinsic],
    intermediate_repr=[ir for ir in MiscTokenKind],
    ignorable=[ig for ig in Ignore]
)


def unescape_string(s: str) -> str:
    # NOTE: unicode_escape assumes latin-1 encoding, so we kinda have
    # to do this weird round trip
    return s.encode('utf-8').decode('unicode_escape').encode('latin-1').decode('utf-8')


def to_value(s: str, tok: TokenType) -> Any:
    if isinstance(tok, Keyword):
        return None
    elif isinstance(tok, Intrinsic):
        if tok == Intrinsic.CAST:
            return s[5:-1]
        elif tok == Intrinsic.SIZE_OF:
            return s[7:-1]
        elif tok == Intrinsic.INNER_TUPLE:
            return int(s[s.find('.')+1:])
        elif tok == Intrinsic.ADDR_OF:
            return s[1:]
        else:
            return None
    elif isinstance(tok, MiscTokenKind):
        if tok == MiscTokenKind.INT:
            return int(s)
        elif tok == MiscTokenKind.STRING:
            return unescape_string(s[1:-1])
        elif tok == MiscTokenKind.WORD:
            return s
        elif tok == MiscTokenKind.BOOL:
            return s
        else:
            print(f"Unhandled Intermediate Repr: {tok}")
            exit(1)
    else:
        print(f"Unhandled token type: {tok}")
        exit(1)


def tokenize_line(line: str, line_num: int, filename: str) -> List[Token]:
    tokens = []
    cursor_pos = 1
    while len(line) > 0:
        typ, m = patterns.search(line)
        # Ignore comments and whitespace
        if not typ:
            line = line[m.end():]
            cursor_pos += m.end()
            continue

        token_str = line[m.start():m.end()]

        contains_comment = re.search(Ignore.COMMENT.value, token_str)
        if contains_comment:
            token_str = token_str[contains_comment.start():
                                  contains_comment.end()]
            line = line[contains_comment.start():]
            pos_change = contains_comment.start()
        else:
            line = line[m.end():]
            pos_change = m.end()

        tokens.append(Token(typ=typ,
                            value=to_value(token_str, typ),
                            loc=Loc(line=line_num, column=cursor_pos, file=filename)))
        cursor_pos += pos_change

    return tokens


def tokenize_string(string: str) -> List[Token]:
    return tokenize_line(string, 0, "")


def tokenize(filepath: str) -> List[Token]:
    with open(filepath) as file:
        file_lines = file.readlines()

    parsed_tokens = []
    for line_num, line in enumerate(file_lines):
        # Add one to the line num to start from 1 instead of 0
        parsed_tokens += tokenize_line(line, line_num + 1, filepath)

    return parsed_tokens


if __name__ == "__main__":
    if len(argv) < 2:
        print("Must include Filepath")
        exit(1)

    tokens = tokenize(argv[1])
    # tokens.reverse()
    for token in tokens:
        print(token)
