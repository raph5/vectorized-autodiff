all: build

CC := clang
CFLAGS := -std=c++11 -O2 -lm

build: forward.cpp reverse.cpp
	$(CC) $(CFLAGS) forward.cpp -o forward
	$(CC) $(CFLAGS) reverse.cpp -o reverse

clean:
	rm -rf forward reverse
