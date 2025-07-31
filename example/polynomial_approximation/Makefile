all: build

CC := clang
CFLAGS := -std=c++11 -O2 -lm

build: forward.cpp reverse.cpp forward_parallel.cpp
	$(CC) $(CFLAGS) forward.cpp -o forward
	$(CC) $(CFLAGS) reverse.cpp -o reverse
	$(CC) $(CFLAGS) -pthread forward_parallel.cpp -o forward_parallel

dev: forward.cpp reverse.cpp forward_parallel.cpp
	$(CC) -std=c++11 -g -lm forward.cpp -o forward
	$(CC) -std=c++11 -g -lm reverse.cpp -o reverse
	$(CC) -std=c++11 -g -lm -pthread forward_parallel.cpp -o forward_parallel

clean:
	rm -rf forward reverse forward_parallel
