# Tulip
This is a compiled, statically typed, and stack based toy language inspired by [Porth](https://gitlab.com/tsoding/porth). The goal of this project is for me to explore the basics of compiler and language development.

Ultimately, I'm aiming to make the compiler self-hosted. 

**NOTE: Everything is subject to change during this early stage of development**
## Examples
Below are a few examples to demonstrate some of the basic concepts of the language.

Hello world:
```c
use "std.tlp"

"Hello World\n" puts
```

Print numbers 0 to 99:
```c
0 while dup 100 < do
    dup putu
    1 +
end drop
```

## Quick Start
Before the compiler is self-hosted, the Python3 compiler can be used. 

### Requirements:
* Python 3.7+
* NASM v 2.13+ 

### Compiling and running a Tulip program
Compile the `fib.tlp` example program. This will generate an executable file `output`.
```bash
pyton3 tulip.py examples/fib.tlp
```

Run the program with
```bash
./output
```

### Running the tests.
I've got a number of tests (there's still a bunch missing) to make sure that the compiler's working properly. The `ci.py` program runs each of the tests and make sure the output matches the expected value.

```bash
python3 ci.py
```

## Language Overview
This is a brief overview of the features currently in the language. I'll try to keep this up to date as new features are introduced.

### Literals
#### Integers
A sequence of digits are treated as an unsigned integer, and pushed onto the stack. 
```c
10 20 +
```

#### Booleans
`true` and `false` are parsed as booleans and are represented with `1` and `0` respectively.

Booleans are treated separately from integers thus the following code would not compile:
```c
1 true +
```

#### Strings
A string  must be contained within two `"`. A string is a structure within `Tulip` that has both a size (`int`) and a pointer to the data (`ptr`)

```c
// This is the internal representation of a Str
struct Str
    int // size
    ptr // data
end
```

When a string token is encountered, the `Str` structure is pushed onto the stack. As will be discussed later, structures can be treated as a single element. 

When a string literal is compiled, a null terminator is placed at the end for convenience for working with the operating system. String operations do not rely on this null terminator, and rather use the size of the string. The size of the string does not include the null terminator.

### Intrinsics

These are the built in operations for the language.

### Stack Manipulation

| Operation | Signature | Description |
| --------- | --------- | ----------- |
| `dup` | `T -> T T`| Duplicates the top element on the stack. |
| `swap` | `A B -> B A` | Swaps the order of the top two elements |
| `drop` | `T -> ` | Consumes the top element from the stack |
| `putu` | `int -> ` | Consumes and prints the top integer on the stack |
| `push` | `T -> R[T]` | Consumes the top element of the stack, and pushes it onto the return stack.
| `pop` | `R[T] -> T` | Consumes the top element of the return stack and pushes it onto the stack.

### Comparison Operators

Not all comparison operators have been implemented yet.

| Operation | Signature | Description |
| --------- | --------- | ----------- |
| `==` | `a: int b: int -> bool` | Pushes `a == b` onto the stack |
| `<=` | `a: int b: int -> bool` | Pushes `a <= b` onto the stack |
| `<` | `a: int b: int -> bool` | Pushes `a < b`  onto the stack |
| `>` | `a: int b:int -> bool` | Pushes `a > b` onto the stack |

### Syscalls

| Operation | Signature | Description |
| --------- | --------- | ----------- |
| `syscall<n>` | `T1, T2, ... Tn id: int -> int` | Performs the syscall with the corresponding `id`, with up to n arguments. 0 <= n <= 6 |

### Group/Struct Operations

`Tulip` supports the creation of structures as well as anonymous structures or `groups`. 

| Operation | Signature | Description |
| --------- | --------- | ----------- |
| `<n> group` | `T1, T2, ... TN -> Group<n>` | Groups the top n elements into one element |
| `group.<n>` | `Group<n> -> T` | Consumes the `group` and pushes the nth element onto the stack |
| `cast(<name>)` | `T1, T2, ... TN -> struct` | Groups the top elements of the stack into a `struct` |
| `<name>.<n>` | `struct -> T` | Consumes the struct and pushes the nth element onto the stack |
| `split` | `struct -> T1, T2, ... TN` | Breaks the `struct`/`group` into it's constituent parts |

`Structs` and `groups` are treated as if they were a single element. For example the `swap` operation will swap the entire `struct`/`group` with the element below the `struct`/`group`, while preserving the order of elements within the `struct`/`group`.

For example the following program would print `1`, then `3`, then finally `2`.
```c
1 2 3   // Stack: 1 2 3 
2 group // Stack: 1 [2 3] 
swap    // Stack: [2 3] 1
putu    // Output: `1`
split   // Stack: 2 3
putu    // Output: `3`
putu    // Output: `2`
```

### Control Flow

### If Conditions
Type checking requires that each branch (at least two) of the if statement produces the same types onto the stack. For instance, if one branch pushes an `int` onto the stack, and another pushes two `int`s onto the stack, this will not compile. 

```c
if <condition> do
    <branch body>
else <condition> do
    <branch body>
else 
    <branch body>
end
```

### While loops
Type checking requires that the types on the stack do not change from before the loop and after the loop. You cannot, for example, push an int onto the stack with each iteration of the loop.

```c
while <condition> do
    <body>
end
```

### Functions

``` rust
fn <name> <Input Types> (-> <Output types>) do
    <function body>
end

// Eg. Function that takes an int and returns an int
fn foo int -> int do
    // ...
end

// Eg. Function that takes a bool and returns nothing
fn bar bool do

end
```

### Constant Expression
`Tulip` supports a very limited number of operations as constant expressions.

```
const <name> <expr> end
```

### Reserving Memory
You can reserve fixed amounts of memory (such as for an array) with
`reserve` blocks.

```
reserve <name> <int> end
```

### Types
There are only four types in Tulip by default: `int`, `bool`, `ptr`, and `Str`.

### Including From Multiple Files

You can include other files with `use` statements. Paths can be absolute or relative to the `tulip.py` compiler.

```
use "std.tlp"
```