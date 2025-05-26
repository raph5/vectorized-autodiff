all: build

CFLAGS := -std=c++11 -O2 -lm

build: forward.cpp reverse.cpp
	gcc $(CFLAGS) forward.cpp -o forward
	gcc $(CFLAGS) reverse.cpp -o reverse

clean:
	rm -rf forward reverse
