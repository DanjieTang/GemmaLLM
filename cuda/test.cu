#include <stdio.h>
#include <string>

int main() {
    std::string str1 = "Hello, world!";
    std::string str2 = str1;
    str2 = "Hi world";
    printf("%s (length %zu)\n", str1.c_str(), str1.size());
    printf("%s (length %zu)\n", str2.c_str(), str2.size());
    return 0;
}
