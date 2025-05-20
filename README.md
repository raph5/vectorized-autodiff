
# Automatic Differentiation - Reverse VS Forward Vectorized

See youtu.be/watch?v=wG_nF1awSSY for an introduction to automatic
differentiation.

When working with automatic differentiation, a common and important question
arises: should you use forward or reverse mode? The answer isn't always
straightforward, especially in applications with many inputs and outputs.
However, for optimization problems—typically characterized by many inputs and a
single output—reverse mode is often considered the ideal choice. But is it
really?

It's true that, in this context, forward mode requires significantly more
FLOPS. However, it's also *embarrassingly parallel*. And in practice, hardware
vendors have been optimizing for exactly this kind of workload for decades.
GPUs are the obvious example, but even modern CPUs are highly efficient at
vectorized computation thanks to SIMD instruction sets.

On the other hand, reverse mode tends to be much more memory intensive. This is
because it requires storing the entire computation graph for the backward pass.
For perspective, on a modern CPU, a floating-point operation (FLOP) takes
roughly 0.01 nanoseconds, while an L2 cache access takes around 10 nanoseconds.

This disparity helps explain why, in some cases, a vectorized forward mode
implementation can outperform reverse mode—even when the number of inputs far
exceeds the number of outputs.

This repository includes multiple implementations of reverse and forward mode
AD inspired by [[1]](https://arxiv.org/abs/1811.05031) and
[[2]](https://github.com/Janko-dev/autodiff). These implementations aim to be
representative of the current state of AD libraries.

# Links

- [[1] A Review of automatic differentiation and its efficient implementation](https://arxiv.org/abs/1811.05031)
- [[2] Simple Automatic Differentiation library](https://github.com/Janko-dev/autodiff)
