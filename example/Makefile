all: build

CFLAGS := -std=c++11 -O2 -lm

build: reverse_example.cpp forward_example.cpp
	gcc $(CFLAGS) reverse_example.cpp -o reverse_example
	gcc $(CFLAGS) forward_example.cpp -o forward_example

clean:
	rm -rf reverse_example forward_example
