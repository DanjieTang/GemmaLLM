// enum_class_example.cpp
//
// A short tour of `enum class` (scoped enums, C++11):
//   1. Basic declaration and use (scoping)
//   2. No implicit conversion to int (type safety)
//   3. Switching over an enum class
//   4. Custom underlying values
//   5. Enum class as a template VALUE parameter (the softmax trick)
//
// Build:  g++ -std=c++17 -Wall enum_class_example.cpp -o enum_class_example
// Run:    ./enum_class_example

#include <cstdio>

// ---------------------------------------------------------------------------
// 1. Declaration. The names live INSIDE the enum: you write Color::RED,
//    not just RED. Two enums can reuse the same name without colliding.
// ---------------------------------------------------------------------------
enum class Color { RED, GREEN, BLUE };
enum class TrafficLight { RED, YELLOW, GREEN };  // GREEN again — no conflict

// ---------------------------------------------------------------------------
// 4. You can pick explicit values (and an underlying integer type).
//    Handy for bit flags or matching an external protocol/file format.
// ---------------------------------------------------------------------------
enum class Permission : unsigned char {
    READ    = 1,  // 0b001
    WRITE   = 2,  // 0b010
    EXECUTE = 4,  // 0b100
};

// ---------------------------------------------------------------------------
// 3. switch works naturally, and the compiler warns if you forget a case.
// ---------------------------------------------------------------------------
const char* toString(Color c) {
    switch (c) {
        case Color::RED:   return "red";
        case Color::GREEN: return "green";
        case Color::BLUE:  return "blue";
    }
    return "unknown";  // unreachable if every case is handled
}

// ---------------------------------------------------------------------------
// 5. Enum class value as a template parameter: one specialized function
//    per value, decided at compile time (same idea as blockReduce<...>).
// ---------------------------------------------------------------------------
enum class Op { ADD, MUL };

template <Op op>
int apply(int a, int b) {
    if (op == Op::ADD)  // resolved by the compiler, zero runtime cost
        return a + b;
    else
        return a * b;
}

int main() {
    // 1. Scoping: values must be qualified with the enum name.
    Color c = Color::GREEN;
    TrafficLight t = TrafficLight::GREEN;
    printf("Color and TrafficLight both define GREEN, no collision: %d vs %d\n",
           static_cast<int>(c), static_cast<int>(t));

    // 2. Type safety: no silent conversion to int.
    // int x = c;                          // ERROR: won't compile
    // if (c == t) { ... }                 // ERROR: different enum types
    int x = static_cast<int>(c);           // OK — conversion must be explicit
    printf("static_cast<int>(Color::GREEN) = %d\n", x);  // 1 (0-based default)

    // 3. switch over an enum class.
    printf("toString(Color::BLUE) = %s\n", toString(Color::BLUE));

    // 4. Explicit values / underlying type.
    printf("sizeof(Permission) = %zu byte, WRITE = %d\n",
           sizeof(Permission), static_cast<int>(Permission::WRITE));

    // Combine flags manually with bitwise ops on the underlying values:
    unsigned char rw = static_cast<unsigned char>(Permission::READ) |
                       static_cast<unsigned char>(Permission::WRITE);
    printf("READ | WRITE = %d\n", rw);  // 3

    // 5. Value template parameter: one specialized copy per enum value.
    printf("apply<Op::ADD>(3, 4) = %d\n", apply<Op::ADD>(3, 4));  // 7
    printf("apply<Op::MUL>(3, 4) = %d\n", apply<Op::MUL>(3, 4));  // 12

    return 0;
}
