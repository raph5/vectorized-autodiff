all: build

CC := clang
CFLAGS := -std=c++11 -O2 -lm

primal: primal.cpp
	$(if $(DEG),,$(error Must set DEG))
	$(CC) $(CFLAGS) -DDEG=$(DEG) primal.cpp -o primal_build_$(DEG)

reverse: reverse.cpp
	$(if $(DEG),,$(error Must set DEG))
	$(CC) $(CFLAGS) -DDEG=$(DEG) reverse.cpp -o reverse_build_$(DEG)

forward: forward.cpp
	$(if $(DEG),,$(error Must set DEG))
# I renamed GRADLEN to GL to avoid the overriding of GRADLEN
	$(eval GL := $(shell echo ${DEG}+1 | bc))
	$(CC) $(CFLAGS) -DDEG=$(DEG) -DGRADLEN=$(GL) forward.cpp -o forward_build_$(DEG)

forward_novec: forward.cpp
	$(if $(DEG),,$(error Must set DEG))
# I renamed GRADLEN to GL to avoid the overriding of GRADLEN
	$(eval GL := $(shell echo ${DEG}+1 | bc))
	$(CC) $(CFLAGS) -fno-vectorize -fno-slp-vectorize -DDEG=$(DEG) -DGRADLEN=$(GL) forward.cpp -o forward_build_novec_$(DEG)

forward_gradlen: forward.cpp
	$(if $(DEG),,$(error Must set DEG))
	$(if $(GRADLEN),,$(error Must set GRADLEN))
	$(CC) $(CFLAGS) -DDEG=$(DEG) -DGRADLEN=$(GRADLEN) forward.cpp -o forward_build_gradlen_$(DEG)_$(GRADLEN)

parallel: forward_parallel.cpp
	$(if $(DEG),,$(error Must set DEG))
	$(CC) $(CFLAGS) -DDEG=$(DEG) forward_parallel.cpp -o parallel_build_$(DEG)

parallel_workers: forward_parallel.cpp
	$(if $(DEG),,$(error Must set DEG))
	$(if $(WORKERS),,$(error Must set WORKERS))
	$(CC) $(CFLAGS) -DDEG=$(DEG) -DRI_WORKERS=$(WORKERS) forward_parallel.cpp -o parallel_build_workers_$(DEG)_$(WORKERS)


# use -j option to run build in parallel
build: reverse forward forward_novec forward_gradlen

clean:
	rm primal_build_* forward_build_* reverse_build_* parallel_build_*
