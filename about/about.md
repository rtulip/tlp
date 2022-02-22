# Tlp - A Language Experiment

Tlp is a stack based, statically typed, concatinative programming language created as a learning exercise to better understand how a programming language is made. The purpose of this document is to detail the development process, some of the intereresting details about the language, as well as some of the shortcomings and failings of the language.

I'm choosing to leave this project where it is so that I can do further experiments with the lessons I've learned from this experience. My original goal with this language was to write it's compiler in `tlp` to make it self-hosted, but I'll aim to exaplain why I don't think it's worth putting in the effort to do so at time of writing.

# Language Summary

Before diving into the details, lets look at some basic programs and use that to expalain some of the core concepts of the langue. Let's start with a hello world:

### HelloWorld.tlp
```
use "std.tlp"

"Hello World\n" puts
```

Step by step, here's what this program does. 
 * The `use` statment includes the standard library (needed for the `puts` function).
 * The string literal `"Hello World\n"` is pushed onto the stack. 
 * The function `puts` consumes a string from the top of the stack and prints it to `stdout`.

Much like `Python`, the top of the file is the entry point, and so execution starts from there. Any literal, (Strings, ints, booleans) is pushed automatically to the top of the stack. Functions are called just by name, and will use the stack for any arguments. Here's what writing a function in `tlp` looks like.

### Function.tlp
```
fn add_wrapper int int -> int do
    +
end
```

Ok, there's a bit more going on here, so lets break it down.
 * `fn` keyword notes this is the start of a function definition
 * `add_wrapper` is the name of the function
 * `int int -> int` is the function signature. I.e. the it expectes two ints on the top of the stack, and ends by leaving one int on top. The `->` separates inputs from outputs.
 * `do` keyword indicates the start of the function body. 
 * `end` keyword indicates the end of the function definition.

This function can then be called the same way as described earlier.
```
// adds 10 and 5 and prints 15 to stdout
10 5 add_wrapper putu
```

As mentioned above, this is a statically typed language, so the function signature will be checked at compile time, thus the following code wouldn't compile:
```
// will not compile
10 "Hello World" add_wrapper

----------- Compiler Error ------------

about/examples/Function.tlp:5:20 [ERROR]: 
    Didn't find a matching signature for OpType.CALL:add_wrapper.
    Expected: [int int]
    Found   : [int Str]
```

We'll go over type checking in more detail later in this document.

### Intrinsics.tlp

The last important thing I want to touch upon in this section is some of the intrinsics. As we saw in the `add_wrapper` function, the body just consisted of the `+` operator. These operators are intrinsics, which themselves have signatures and are type checked accordingly. So for example the `<` intrinsic takes two ints and reutnrs a boolean.

```
// For example
10 5 <
// false on top of stack.
```

But in `tlp` intrinsics arent' limited to operators, there's also intrinsics for doing stack manipulation. For example, the `dup` intrinsic, pushes a copy of the top element of the stack, the `swap` intrinsic swaps the top two elements of the stack, and the `drop` intrinsic removes the top element of the stack. I'm not going to go over the full list of intrinsics here, but know that as a result, `tlp` has many keywords (~35), so there is certainly an up-front challenge to learning/understanding `tlp`. See the `README` for a more thorough breakdown of the intrinsics.

With the brief rundown of the language out of the way, I want to reflect upon what language features worked, what didn't, and what problems I never found a way to resolve.

# The Good, the Bad, and The Ugly

I first stared this project after being inspired by [tsoding daily](https://www.youtube.com/c/TsodingDaily)'s work on Porth. I had never heard of a stack based programming language, and was intrigued by the different approach to programming where variables didn't exist. Every problem seemed like a puzzle, where you had to try to figure out how to manipulate the stack to get the desired result. 

What prompted me to bulid `tlp` was the desire to support structures. I had noticed in `tsoding`'s videos that string operations became complex, becase a string had two components: a pointer to the data and a size. I wanted to be able to group those two elements together and treat them as a single unit.

With that point of context, lets have a look at what worked.

## The Good

### Structures

My initial goal to support structures, was certainly one of the things which I think was a success with `tlp`. Without having any bindings or variables (ahem, in theory at least... ), being able to perform stack on operations on logical units was extremely convenient, and made reasoning about the code a lot easier (for me at least).

For example, here's two pieces of code which duplicate a string with and without the `Str` structure

```
fn dup_str1 Str -> Str Str do
    dup
end 

fn dup_str2 int ptr -> int ptr int ptr do
    push dup            // copy the size
    pop swap push dup   // copy the pointer
    pop swap            // get the order right
end
```

In theory, the `dup_str2` function could have been made easier by adding other intrinsics such as `over` (which would copy the 2nd element of the stack i.e. `A B -> A B A`) continues to add to the already large number of intrinsics. Instead, the number of intrinsics can be kept lower and made to be more versitile.

### Implicit Returns in Functions

```
fn three_nums -> int int int do
    // This just works!
    1 2 3
end
```

One of the pleasant surprises with `tlp` was returning values from functions implicitly through the stack. It was shockingly convenient to be able to incrementally "return" values by pushing them to the stack, and then continuing to do othe processing. Additionally, being able to return more than one thing from a function is great (cough cough `C`).

### Type Checking

I've come to greatly appreciate stronly-typed languages, but I found this to be particularly nessisary with `tlp`. Given how different programming without arguments felt, it was very easy to make mistakes when tracking what you're doing to the stack.

By having the compiler track the types for you, entire classes of mistakes were eliminated automatically. Without this, I wouldn't have been anywhere near as productive with the language.

It does enforce some limitations which help reasoning about the program a lot easier. For instance, each branch of an if statement must produce a similar stack.

```
// This compiles
if <cond> do
    1
else <cond> do
    2
else
    3
end

// This doesnt
if <cond> do
    1
else <cond> do
    "Hello World\n"
else
    true
end

----------- Compiler Error ------------

about/examples/If.tlp:11:1 [ERROR]: 
    Each branch of an IF Block must produce a similare stack.
    Possible outputs: 
        Branch 1: [int]
        Branch 2: [Str]
        Branch 3: [bool]

    Possible pushed values: 
        Branch 1: []
        Branch 2: []
        Branch 3: []
```
This limitation makes it possible to reason about the stack under branching conditions. Similarly, the type checker ensures that loop iterations will produce similar stacks.

```
// This doesn't compile
while true do
    1
end

----------- Compiler Error ------------

about/examples/While.tlp:2:1 [ERROR]: While loops cannot change the stack outside of the loop
    [NOTE]: Stack at start of loop : []
    [NOTE]: Stack at end of loop   : [int]
    [NOTE]: Pushed at start of loop: []
    [NOTE]: Pushed at end of loop  : []
```


### Generics

I haven't touched on this yet, but `tlp` has support for some generics in both data structures and in functions.

```
// Define a structure that is generic over type T
with T
struct Pair
    T T
end 

// Define a function that operates on a generic type T
with T
fn use_pair
    with T -> Pair  // How to define an instance of Pair<T>
do
    // do work...
end 

// Type Inferred during cast
5 3 cast(Pair)

// Explictly use T=int to call use_pair
with int do use_pair
```
And it interacts with the type checking system as you'd expect. 

```
//This doesn't compile
5 "Hello World\n" cast(Pair)

----------- Compiler Error ------------

about/examples/Generics.tlp:17:19 [ERROR]: Generic assignment error.
    [NOTE]: Cannot assign generic type `T` to `Str`.
    [NOTE]: Type `T` was previously assigned to `int`
```
The syntax isn't the prettiest, but it worked. Ultimately, this allowed for some form of polymorphism. For example, here's how we can define a generic stack as a data structure. 

```
// Create a structure to define how to read/write a type `T` into memory.
with T
struct TypeInfo
    // Note: This is the syntax for defining a function pointer
    with T &fn ptr -> T ptr end // read callback 
    with T &fn T ptr -> ptr end // write callback 
    int                         // SizeOf(T)
end

with T
struct Stack
    with T -> TypeInfo          // TypeInfo for T  
    ptr                         // Pointer to empty space in stack
    ptr                         // Pointer to # of elements on the stack
    int                         // Capacity of stack.
end

with T
fn Stack.Push
    with T -> Stack
    T
do 
    // push T into stack ... 
end

with T
fn Stack.Pop
    with T -> Stack
    ->
    T
do 
    // Pop T from stack ... 
end

```

This reduced the amount of duplicated code that was required to implement data structures for similar types. Generics were fleshed out near the end of my work on `tlp`, was easliy the feature which made `tlp` the most bearable to use.

## The Bad
So what didn't work as well? In no particular order, here's some things I don't like with where the language is at.

### Reserved Memory
You can manually allocate reserve memory for data structures like lists or stacks. This is currently implemented in a very clunky way, where all memory is globally accessible. A pointer to the beginning of the reserved memory is pushed onto the stack with the memory's identifyer. Intrinsics, such as `@64` and `!64`, can will read and write repsectively at a given pointer.

```
// Reserve 8 bytes.
reserve foo 8 end

// Save 5 into foo
5 foo !64

// ...

// Read 5 from foo
foo @64 putu
```

Beyond being clunky to use, this design had two major pitfalls. First, this indirectly allows for variables in a language in a language based around not having variables. In the example above, `foo` is really just an `int` which is globally accessible without the stack. 

Secondly, the type system currently doesn't support pointers to different types, just raw pointers. The type system could obviously be improved to supported typed pointers, but this isn't the biggest problem the language faces.

### The Second Stack

Yeah, so much like `Forth`, `tlp` has a second "return" stack. This can act as temporary storage for the top elements of the stack, and makes some operations easy to express. For example de-structuring the 2nd element on the stack. 

```
"Hello World" 42
push split pop
// Stack is [ptr int int]
```

The type checker enforces similar invariants as before. `if` and `while` blocks cannot modify the return stack, just as they can't the data stack. It's also enforced that the return stack is empty at the end of a function.

While the return stack makes some things easier, I believe it is an inelegant solution to the problem of accessing the nth element from the top of the stack. In order to access an arbitrary element, the code would look something like this:

```
// stack has some things on it 
push push   // ... keep pushing the top n elements
do_work     // do work with the nth element
pop pop     // ... ensure you pop n elements back
```

I think this is bad for a few reasons. First of all, this is really tedious to write. Because of the limitations of the type checker, you cannot write loops to repeatedly `push`/`pop`. Secondly, this makes the mental load of writing `tlp` programs even greater. Now not only is the programmer expected to track the state of the stack (with the assistance of the type checker), but they also must now also reason about a second stack. 

It's difficult to write programs which will only ever need operate on 2-3 operands. The return stack feels like a band-aid solution to cover up problems which arise from trying to work with than 3 elements at once.

## The Ugly

So what was it that made me reconsider continuing development with this language? While there are a variety of small problems which could be fixed by expanding the capability of the type checker, the largest problem has to be the maintainability of the code.

As mentioned earlier, `tlp` adds additional mental load to the programmer by forcing them to track the state of multiple stacks at once. This mental load is already a burden when you're actively working on a function, but makes it near impossible to return to previous work and understand what's going on without having to track the stack from the beginning of the function.

In this section, I want to show the piece of code I wrote, and then tried to return to, which I think perfectly demonstrates the fundamental problem of a stack based language without any binding mechanisms (which was one of the goals of this language).

I was trying to create a generic data structure for a stack. Again this is what the data structures looked like:

```
// Create a structure to define how to read/write a type `T` into memory.
with T
struct TypeInfo
    // Note: This is the syntax for defining a function pointer
    with T &fn ptr -> T ptr end // read callback 
    with T &fn T ptr -> ptr end // write callback 
    int                         // SizeOf(T)
end

with T
struct Stack
    with T -> TypeInfo          // TypeInfo for T  
    ptr                         // Pointer to empty space in stack
    ptr                         // Pointer to # of elements on the stack
    int                         // Capacity of stack.
end
```

and here's the (working) function to push an element onto a stack which made me realize that this language doesn't lead to maintainable code.

```with T
fn Stack.Push
    with T -> Stack
    T
do
    
    push split          // ???                            
    push dup @64 pop    // ???
    if < not do         // what are we comparing?
        "Cannot push to stack. It's full.\n" eputs 
        1 exit
    end 

    dup @64 1 + swap !64            // ???
    pop swap dup push @64           // ???
    cast(ptr) push swap pop swap    // ???????
    with T do TypeInfo.Write call   // Call write I guess...
    pop !64                         // ???

    // I guess that we pushed T onto the stack?

end
```

In my opinion, this code ends up being a mess. It's a cryptic chanin of intrinsics which requires knowing the layout of the Stack structure in order to even try to follow. 

Maybe there's a better way to try to structure this piece of code, but in principal, there's no way to parse the logic of this function at a glance. This is the primary flaw with this language, and one I don't see a way to work around. 

`Tsoding` has since introduced the `let` and `peek` keywords in `Porth`, which binds the top n elements of the stack to a variable names to try and mitigate this problem, but I wanted to stay true to not having any bindings.

I'll continue to think on how this problem could be solved, but the difficulty of returning to old code is the primary reason I'm not currently going to continue to work on `tlp`. I'd love to be able to get the compiler to be self-hosted, but as the language evolves and changes it'll continue to be more and more difficult to maintain the code base.

# Conclusion
This has been a very interesting experiment. I'd never built a language before, and I've learned a ton. I've got to see how each component of the compiler, from parsing to code generation all comes together.

I think the design of `tlp` is flawed enough in it's current iteration, that it's worth leaving it here, and starting fresh. 

-- rtulip