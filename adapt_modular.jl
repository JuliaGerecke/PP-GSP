"""
Ground State Search for Spin Systems using ADAPT-VQE
==============================================================

This module provides a general implementation of ADAPT-VQE that works with
a set of implemented Hamiltonian and initial state overlap function,
but can be expanded by the user to any Hamiltonian and topology without modifying the main code.

Key features:
- Flexible overlap function specification (e.g., Néel, Plus)
- Support for three AD backends: Mooncake, ReverseDiff, ForwardDiff
- ThreadsX parallelization for all backends
- Hamiltonian-agnostic design 
- Modular file structure for maintainability

AD Backend Options:
- Mooncake: Fast, uses min_abs_coeff truncation, supports gradient tapes
- ForwardDiff: Simple, uses min_abs_coeff truncation, no tape overhead
- ReverseDiff: Memory efficient, uses max_freq truncation, supports gradient tapes

ThreadsX Parallelization:
- Each Hamiltonian term's gradient computed in parallel
- Available for all three backends (lbfgs_optimizer_threadsx_MC/FD/RD)
- Optional gradient tape caching for Mooncake and ReverseDiff
- Use adaptVQE_2d_threadsx() with backend=:mooncake/:forwarddiff/:reversediff

Module Structure:
- Adapt/hamiltonians.jl: Hamiltonian constructors and overlap functions
- Adapt/utilities.jl: Bit operations and Pauli string conversions
- Adapt/system_hamiltonian.jl: Type-based Hamiltonian dispatch system
- Adapt/optimizers.jl: Loss functions and L-BFGS optimizers
- Adapt/adapt_steps.jl: Gradient calculation, operator selection, convergence checking, tile generation
- Adapt/adapt_algo.jl: Main ADAPT-VQE algorithms and pool generation

For usage examples, see the test scripts (folder cluster_files, can be run locally with reasonable truncations) or refer to function docstrings
"""

using PauliPropagation
# Plots is optional - only load if available (for local testing)
#using Plots
using ReverseDiff
#using ReverseDiff: GradientTape, compile
using Bits
using Random
using NLopt
import ForwardDiff
import DifferentiationInterface
import Mooncake
using ThreadsX
using LinearAlgebra

# Control BLAS threads
BLAS.set_num_threads(1)

# Check number of threads available
n_threads = Base.Threads.nthreads()
println("Julia using $n_threads threads")

# Mooncake tangent type definitions for PauliPropagation internal types
# Custom UInt types are only defined when a psum with appropriate qubit count is created
# Each qubit needs 2 bits (4 Pauli operators: I, X, Y, Z), so we need UIntN types for:
# 9-12 qubits: UInt24, 13-20 qubits: UInt40, 21-24 qubits: UInt48, 25-28 qubits: UInt56
# 33-36 qubits: UInt72, 37-40 qubits: UInt80, 41-44 qubits: UInt88, 45-48 qubits: UInt96, 49-50 qubits: UInt104
# Create dummy pauli sums to trigger type definitions (one per range)
_ = PauliSum(Float64, 9)   # UInt24
_ = PauliSum(Float64, 17)  # UInt40
_ = PauliSum(Float64, 21)  # UInt48
_ = PauliSum(Float64, 25)  # UInt56
_ = PauliSum(Float64, 35)  # UInt72
_ = PauliSum(Float64, 37)  # UInt80
_ = PauliSum(Float64, 41)  # UInt88
_ = PauliSum(Float64, 45)  # UInt96
_ = PauliSum(Float64, 50)  # UInt104

# Import all custom UInt types from PauliPropagation
import PauliPropagation: UInt24, UInt40, UInt48, UInt56, UInt72, UInt80, UInt88, UInt96, UInt104

# Define tangent types for Mooncake AD (all custom UInt types have no tangent)
Mooncake.tangent_type(::Type{UInt24}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{UInt40}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{UInt48}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{UInt56}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{UInt72}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{UInt80}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{UInt88}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{UInt96}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{UInt104}) = Mooncake.NoTangent

# ============================================================================
# LOAD ALL MODULES
# ============================================================================

# Include all module files in correct dependency order
include("Adapt/utilities.jl")
include("Adapt/hamiltonians.jl")
include("Adapt/system_hamiltonian.jl")  # Type-based Hamiltonian dispatch
include("Adapt/optimizers.jl")
include("Adapt/adapt_steps.jl")
include("Adapt/adapt_algo.jl")

# ============================================================================
# EXPORTED INTERFACE
# ============================================================================

# Export main functions for external use
export adaptVQE_2d_threadsx, adaptVQE_2d_threadsx_loop
export scaled_pool_generation_2d, scaled_pool_selection_2d, generate_obc_square_tiles
export generate_full_bit_pool
export heisenberg_hamiltonian, tfim_hamiltonian, tv_hamiltonian
export j1j2_1d_hamiltonian, j1j2_2d_square_obc_hamiltonian, fermi_hubbard_hamiltonian
export overlapwithneel, overlapwithplus, overlapwithfock, neel_bits
# Export type-based Hamiltonian system
export SystemHamiltonian, get_hamiltonian, get_default_kwargs, get_hamiltonian_constructor

println("\nADAPT-VQE module loaded successfully!")
# ============================================================================