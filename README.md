# PP-GSP (Pauli Propagation-Ground State Preparation)

A high-performance implementation of ADAPT-VQE for preparing ground states of quantum (spin) systems, built on top of [PauliPropagation.jl](https://github.com/MSRudolph/PauliPropagation.jl).
This project was realised as a master's thesis (30 ECTS) in the Laboratory of Quantum Information and Computation at EPFL from September 2025 - January 2026. 

## Abstract

Quantum simulations are promising for studying quantum-chemical problems, such as the preparation of molecular and lattice Hamiltonians' ground states, with the ultimate goal of advancing materials discovery. 
For efficient use of quantum hardware, we require a classical ground-state preparation framework that can serve as an input and benchmark for quantum hardware simulations.
We implement and benchmark variational ground-state preparation using the classical simulation algorithm Pauli Propagation. Pauli Propagation computes expectation values in the Heisenberg picture by back-propagating observables through a circuit. 
Even though the algorithm relies on truncations to remain computationally feasible and scalable, it has proven to be accurate in many applications.
The ansatz we use to approximate the ground state is ADAPT-VQE, a hybrid quantum-classical algorithm which we have adapted to work fully classically. 
To enable scaling to larger system sizes, we employ an operator-pool tiling strategy. This takes operators relevant for expressing the ground state of small systems and extends them to match the larger system size.
We focus on spin Hamiltonians in 1D and 2D and validate approximate ground states by re-evaluating them at stricter truncation settings and by checking symmetry-based observables. 
We benchmark the accuracy of our results against exact solutions where available, as well as against state-of-the-art numerical methods such as Density Matrix Renormalization Group and Neural Quantum States. 
We further examine the scalability of our approach by considering time and memory behaviour in interplay with the accuracy of specific implementation decisions, such as the optimization procedure, the operator tile size, as well as threading options on high-performance computers.

## Features

- **Flexible Hamiltonian Support**: Pre-implemented Hamiltonians for common spin and fermionic systems
  - Heisenberg model (1D, 2D)
  - Transverse-field Ising model (TFIM)
  - Transverse-vector model (TV)
  - J1-J2 model (1D, 2D square lattice with OBC)
  - Fermi-Hubbard model
  
- **Three AD Backends**: Choose the best automatic differentiation backend for your problem
  - **ForwardDiff**: Simple, uses `min_abs_coeff` truncation, no tape overhead
  - **Mooncake**: Fast after tape compilation, uses `min_abs_coeff` truncation, supports gradient tapes, high memory usage
  - **ReverseDiff**: Uses `max_freq` truncation, supports gradient tapes, similar speed to ForwardDiff

- **Parallel Computing**: ThreadsX parallelization for gradient computation across all backends

- **Modular Design**: Easy to extend with custom Hamiltonians and initial states

- **Scalable**: Efficient handling of large operator pools through tiling and truncation


**Dependencies**: PauliPropagation.jl, Mooncake.jl, ReverseDiff.jl, ForwardDiff.jl, NLopt.jl, ThreadsX.jl, DifferentiationInterface.jl, Bits.jl, LinearAlgebra


## Module Structure

```
PP-GSP/
├── adapt_modular.jl          # Main module file
├── Adapt/
│   ├── hamiltonians.jl       # Hamiltonian constructors
│   ├── system_hamiltonian.jl # Type-based dispatch system
│   ├── utilities.jl          # Bit operations and utilities
│   ├── optimizers.jl         # L-BFGS optimizers for all backends
│   ├── adapt_steps.jl        # Gradient calculation, operator selection
│   └── adapt_algo.jl         # Main ADAPT-VQE algorithms
├── adapt_benchmarks.jl       # Benchmarking utilities
└── cluster_files/            # Example scripts for HPC clusters
```
