
# Automatic Differentiation - Reverse VS Forward Vectorized

This repository shares the benchmarks and the AD implementations presented in
[this article](https://raph5.github.io/blog/vectorized-autodiff/)

## Build and run

Along with the `reverse.h` and `forward.h` headers, two examples and one
benchmark are provided.

- `examples/hello_world` is a program that differentiates the function `(a, b,
c, d) -> (sqrt(a / (b + c*a) + exp(1/d)))^d`
- `examples/polynomial_approximation` is an example of an application to
automatic differentiation that I used as benchmarking task to test the
performances of `reverse.h` and `forward.h`. It's a program that performs a
gradient descent on polynomial coefficients in order to find a polynomial that
approximates the function `t -> exp(1/t^2)`

To build either of those two examples, make sure that you have `clang` and
`make` installed and run the command `make` into the corresponding example
directory.

`benchmarks/polynomial_approximation` contains multiple .cpp files that measure
the runtime of 1 iteration of the gradient descent algorithm described earlier.
Alongside those .cpp files are some bash scripts that gather those measurements
as functions of some parameter.

- `benchmark.sh` compares the different AD implementations relative to gradient
size.
- `benchmark_gradlen.sh` compares the runtime of chunked forward AD with
different values for the parametter α.
- `benchmark_workers.sh` compares the runtime of parallelized chunked forward
AD with different workers count.
- `benchmark_parallel.sh` and `benchmark_reverse.sh` are quick measruements
of the performances of parallelized chunked forward AD and reverse AD to avoid
having to run `benchmark.sh` which is slow.

If you happen to interrupt one of those benchmarks, you will be left with a
series of executables that would have been deleted at the end of the benchmark.
To get rid of those run `make clean`.
