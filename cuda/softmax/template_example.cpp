// template_example.cpp
//
// A gentle tour of C++ templates, in three steps:
//   1. Function template with a TYPE parameter  (the classic case)
//   2. Class template with a type parameter
//   3. Template with a VALUE parameter          (the trick used in
//      example_softmax.cu's blockReduce<ReduceOp::MAX>)
//
// Build:  g++ -std=c++17 -Wall template_example.cpp -o template_example
// Run:    ./template_example

#include <cstdio>

// ---------------------------------------------------------------------------
// 1. Function template with a type parameter.
//
// `T` is a placeholder the compiler fills in at each call site. Calling
// myMax(2, 7) makes it generate myMax<int>; calling myMax(2.5, 1.5) makes
// it generate myMax<double>. Each generated copy is ordinary, fully-typed
// code — there is no runtime cost compared to writing both by hand.
// ---------------------------------------------------------------------------
template <typename T>
T myMax(T a, T b) {
    return (a > b) ? a : b;
}

// ---------------------------------------------------------------------------
// 2. Class template.
//
// Same idea, but the type parameter belongs to the whole class. You must
// name the type when you declare a variable: Pair<int> and Pair<double>
// are two distinct types.
// ---------------------------------------------------------------------------
template <typename T>
struct Pair {
    T first;
    T second;

    T sum() const { return first + second; }
};

// ---------------------------------------------------------------------------
// 3. Template with a VALUE parameter (non-type template parameter).
//
// The parameter is not a type but a compile-time constant. This is exactly
// how `template <ReduceOp op> blockReduce(...)` works in the softmax
// kernel: the compiler stamps out one specialized copy per value, and
// `if (op == ...)` is resolved at compile time, so each copy only contains
// the branch it actually needs.
// ---------------------------------------------------------------------------
enum class Op { ADD, MUL };

template <Op op>
int apply(int a, int b) {
    if (op == Op::ADD)  // decided by the compiler, not at runtime
        return a + b;
    else
        return a * b;
}

int main() {
    // 1. Type is deduced from the arguments...
    printf("myMax(2, 7)       = %d\n", myMax(2, 7));          // T = int
    printf("myMax(2.5, 1.5)   = %.2f\n", myMax(2.5, 1.5));    // T = double

    // ...or written explicitly: myMax<double>(2, 7) would convert 2 and 7
    // to double and use the double copy.

    // 2. Class template: the type must be named explicitly.
    Pair<int> p{3, 4};
    Pair<double> q{1.5, 2.25};
    printf("Pair<int>{3,4}.sum()      = %d\n", p.sum());
    printf("Pair<double>{1.5,2.25}.sum() = %.2f\n", q.sum());

    // 3. Value parameter: one specialized function per enum value.
    printf("apply<Op::ADD>(3, 4) = %d\n", apply<Op::ADD>(3, 4));
    printf("apply<Op::MUL>(3, 4) = %d\n", apply<Op::MUL>(3, 4));

    return 0;
}
