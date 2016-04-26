#include <iostream>

using namespace std;

extern "C" {
    const char* getHelloWorld() {
        return "Hello world!";
    }

    int add(int a, int b) {
        return a + b;
    }

    int mult(int a, int b) {
        return a * b;
    }
}

int main() {
    cout << "Hello, World!" << endl;

    return 0;
}