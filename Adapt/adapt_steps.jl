"""
ADAPT-VQE Step Functions
=========================

This file contains functions for individual ADAPT-VQE steps: gradient calculation,
operator selection, convergence checking, and pool generation with tiles approach.
"""

using PauliPropagation
using ThreadsX
using Random
using ForwardDiff

# Include dependencies
include("utilities.jl")
include("hamiltonians.jl")

# ============================================================================
# GRADIENT CALCULATION
# ============================================================================

"""
    compute_operator_pool_commutators(bit_pool, hamiltonian, nq; threads_oppool=false)

Compute commutators [H, P] for all operators P in the operator pool.

# Arguments
- `bit_pool`: Pool of operators in bit representation
- `hamiltonian`: Hamiltonian (PauliSum)
- `nq`: Number of qubits
- `threads_oppool`: Use ThreadsX parallelization (default: false)

# Returns
- Vector of commutators [H, P] for each operator in bit_pool
"""
function compute_operator_pool_commutators(bit_pool, hamiltonian, nq; threads_oppool=false)
    if threads_oppool
        return ThreadsX.collect([begin
            P_op, _, _ = bit_to_paulistring_general(bit_repr, nq; sites=nothing, total_nq=nq)
            P = PauliSum(nq)
            add!(P, P_op)
            PauliPropagation.commutator(hamiltonian, P)
        end for bit_repr in bit_pool])
    else
        return [begin
            P_op, _, _ = bit_to_paulistring_general(bit_repr, nq; sites=nothing, total_nq=nq)
            P = PauliSum(nq)
            add!(P, P_op)
            PauliPropagation.commutator(hamiltonian, P)
        end for bit_repr in bit_pool]
    end
end

"""
    calculate_adapt_gradients(bit_pool, hamiltonian, nq, overlap_func; 
                              calc_grads=:phys, circuit=nothing, thetas=nothing, 
                              threads_oppool=false, min_abs_coeff=0.0, max_weight=Inf,
                              calc_grad_kwargs=nothing, overlap_kwargs...)

Calculate ADAPT-VQE gradients using specified method with timing and memory tracking.

# Arguments
- `bit_pool`: Pool of operators in bit representation
- `hamiltonian`: Hamiltonian (PauliSum)
- `nq`: Number of qubits
- `overlap_func`: Function to compute overlap with initial state
- `calc_grads`: Gradient calculation method (:phys or :fd)
- `circuit`: Current circuit (default: nothing)
- `thetas`: Current parameters (default: nothing)
- `threads_oppool`: Use ThreadsX for commutator computation (default: false)
- `min_abs_coeff`: Default minimum absolute coefficient (default: 0.0)
- `max_weight`: Default maximum Pauli weight (default: Inf)
- `calc_grad_kwargs`: Named tuple to override defaults (default: nothing)
  Can specify (min_abs_coeff=value, max_weight=value) to override defaults
- `overlap_kwargs`: Additional keyword arguments for overlap function

# Returns
Named tuple with:
- `grads`: Vector of gradients
- `commutator_time`: Time for commutator computation (0.0 for :fd method)
- `commutator_memory`: Memory for commutator computation (0 for :fd method)
- `gradient_time`: Time for gradient computation
- `gradient_memory`: Memory for gradient computation
"""
function calculate_adapt_gradients(bit_pool, hamiltonian, nq, overlap_func; 
                                   calc_grads::Symbol=:phys,
                                   circuit=nothing,
                                   thetas=nothing,
                                   threads_oppool::Bool=false,
                                   min_abs_coeff::Real=0.0,
                                   max_weight::Real=Inf,
                                   calc_grad_kwargs=nothing,
                                   overlap_kwargs...)
    
    # Set truncation parameters for gradient calculation
    # If calc_grad_kwargs is provided, use those values; otherwise use defaults
    calc_grad_mac::Real = isnothing(calc_grad_kwargs) ? min_abs_coeff : get(calc_grad_kwargs, :min_abs_coeff, min_abs_coeff)
    calc_grad_mw::Real = isnothing(calc_grad_kwargs) ? max_weight : get(calc_grad_kwargs, :max_weight, max_weight)
    
    if calc_grads == :phys
        # Pre-compute commutators [H, P] for all pool operators
        comm_result = @timed compute_operator_pool_commutators(bit_pool, hamiltonian, nq; 
                                                                threads_oppool=threads_oppool)
        commutators = comm_result.value
        commutator_time = comm_result.time
        commutator_memory = comm_result.bytes

        # Calculate gradients using physical formula
        grad_result = @timed calc_gradients_phys(bit_pool, commutators, nq, overlap_func; 
                                    circuit=circuit, params=thetas, 
                                    calc_grad_mac=calc_grad_mac, calc_grad_mw=calc_grad_mw,
                                    tol=1e-12, verbose=false,
                                    overlap_kwargs...)
        grads = grad_result.value
        gradient_time = grad_result.time
        gradient_memory = grad_result.bytes
        
        return (grads=grads, 
                commutator_time=commutator_time, 
                commutator_memory=commutator_memory,
                gradient_time=gradient_time, 
                gradient_memory=gradient_memory)
        
    elseif calc_grads == :fd
        # Calculate gradients using ForwardDiff
        grad_result = @timed calc_gradients_FD(bit_pool, hamiltonian, nq, overlap_func; 
                                    circuit=circuit, params=thetas, 
                                    tol=1e-12, verbose=false, overlap_kwargs...)
        grads = grad_result.value
        gradient_time = grad_result.time
        gradient_memory = grad_result.bytes
        
        return (grads=grads, 
                commutator_time=0.0,  # No commutator computation for FD
                commutator_memory=0,
                gradient_time=gradient_time, 
                gradient_memory=gradient_memory)
    else
        error("Unknown gradient calculation method: $calc_grads. Use :phys or :fd")
    end
end

"""
    calc_gradients_phys(bit_pool, commutators, nq, overlap_func; kwargs...)

Calculate ADAPT-VQE gradients: g_P = i⟨φ₀|U†[H,P]U|φ₀⟩

# Arguments
- `bit_pool`: Pool of operators in bit representation
- `commutators`: Pre-computed commutators [H,P] for each operator P in the pool
- `nq`: Number of qubits
- `overlap_func`: Function to compute overlap with initial state
- `circuit`: Current circuit (default: nothing)
- `params`: Current parameters (default: nothing)
- `calc_grad_mac`: Minimum absolute coefficient for gradient calculation (default: 0.0)
- `calc_grad_mw`: Maximum Pauli weight for gradient calculation (default: Inf)
- `tol`: Tolerance for imaginary part warning (default: 1e-12)
- `verbose`: Print detailed output (default: false)
- `overlap_kwargs`: Additional keyword arguments for overlap function (e.g., up_on_odd=true)

# Notes
This function expects pre-computed commutators to avoid redundant calculations.
Commutators should be computed once and passed in for all pool operators.
Use `compute_operator_pool_commutators` to generate commutators.
"""
function calc_gradients_phys(bit_pool, commutators, nq, overlap_func;
                              circuit::Union{Nothing,Any}=nothing,
                              params::Union{Nothing,AbstractVector}=nothing,
                              calc_grad_mac::Real=0.0,
                              calc_grad_mw::Real=Inf,
                              tol::Float64=1e-12,
                              verbose::Bool=false,
                              overlap_kwargs...)
    
    grads = Float64[]

    for (k, bit_repr) in enumerate(bit_pool)
        C = commutators[k]

        if !(iterate(C) !== nothing)
            verbose && println("op[$k]: commutator=0 → grad=0.0")
            push!(grads, 0.0)
            continue
        end

        C_prop = C
        if circuit !== nothing
            C_prop = deepcopy(C)
            propagate!(circuit, C_prop, params, min_abs_coeff=calc_grad_mac, max_weight=calc_grad_mw)
            
            # Check if propagation with truncation resulted in empty PauliSum
            if !(iterate(C_prop) !== nothing)
                verbose && println("op[$k]: propagated commutator truncated to empty → grad=0.0")
                push!(grads, 0.0)
                continue
            end
        end
        # this is equivalent to the theoretical formula that calculates the overlap at the new layer but set the parameter of the new layer to zero
        g = overlap_func(im * C_prop, nq; overlap_kwargs...)

        if abs(imag(g)) > tol
            @warn "Gradient has non-negligible imaginary part" imag=imag(g)
        end

        push!(grads, real(g))
    end

    return grads
end

"""
    calc_gradients_FD(bit_pool, H, nq, overlap_func; kwargs...)

Calculate ADAPT-VQE gradients using ForwardDiff automatic differentiation:
g_P = d/dθ ⟨φ₀|U†(θ) H_prop U(θ)|φ₀⟩|_{θ=0}
where U(θ) = exp(-iθP) and H_prop = U_ansatz† H U_ansatz

# Arguments
- `bit_pool`: Pool of operators in bit representation
- `H`: Hamiltonian (PauliSum)
- `nq`: Number of qubits
- `overlap_func`: Function to compute overlap with initial state
- `circuit`: Current circuit (default: nothing)
- `params`: Current parameters (default: nothing)
- `tol`: Tolerance for imaginary part warning (default: 1e-12)
- `verbose`: Print detailed output (default: false)
- `overlap_kwargs`: Additional keyword arguments for overlap function

# Returns
- Vector of gradients for each operator in the pool
"""
function calc_gradients_FD(bit_pool, H, nq, overlap_func;
                              circuit::Union{Nothing,Any}=nothing,
                              params::Union{Nothing,AbstractVector}=nothing,
                              tol::Float64=1e-12,
                              verbose::Bool=false,
                              overlap_kwargs...)

    grads = Float64[]

    # Pre-propagate H once: H_prop = U† H U (only if a circuit is provided)
    H_prop = H
    if circuit !== nothing
        H_prop = deepcopy(H)
        propagate!(circuit, H_prop, params)
    end

    # For each pool element, pre-propagate P (to P_prop) and form [H_prop, P_prop]
    for (k, bit_repr) in enumerate(bit_pool)
        
        gate = pauli_rotation_from_bits_general(bit_repr, nq; sites=nothing)
        temp_circuit = [gate]

        # Define the energy function for a single parameter theta
        function energy_func(theta_vec::Vector{T}) where T
            theta = theta_vec[1]
            # We need a fresh copy of H_prop for each evaluation inside the gradient calculation
            # and we need to promote the coefficients to the Dual number type
            H_theta = promote_paulisum_coeffs(H_prop, T)
            
            propagate!(temp_circuit, H_theta, [theta]; min_abs_coeff=0)
            
            # The imaginary part should be zero for Hermitian observables, but can have small numerical noise
            return real(overlap_func(H_theta, nq; overlap_kwargs...))
        end

        # Calculate the gradient at theta=0
        # ForwardDiff.gradient expects a vector input
        g = ForwardDiff.gradient(energy_func, [1e-10])[1] 

        push!(grads, g)
        verbose && println("op[$k]: ", bit_repr, "  grad=", g)
    end

    return grads
end

# ============================================================================
# OPERATOR SELECTION AND CONVERGENCE
# ============================================================================

"""
    pick_top_operator(gradients, operators; rng=Random.GLOBAL_RNG)

Select operator with largest gradient magnitude (random tie-breaking).
"""
function pick_top_operator(gradients::AbstractVector, operators::AbstractVector; 
                           rng=Random.GLOBAL_RNG)
    length(gradients) == length(operators) || 
        throw(ArgumentError("gradients and operators must have same length"))
    isempty(gradients) && throw(ArgumentError("gradients must not be empty"))
    
    mags = abs.(gradients)
    order = sortperm(mags, rev=true)
    
    max_mag = mags[order[1]]
    tied_top = filter(i -> mags[i] == max_mag, order)
    
    chosen_idx = rand(rng, tied_top)
    
    gradients_sorted = gradients[order]
    operators_sorted = operators[order]
    
    return operators[chosen_idx], gradients[chosen_idx], gradients_sorted, operators_sorted
end

"""
    check_convergence(gradients, max_grads_history; tol=1e-4, stagnation_layers=5, stagnation_tol=0.1, 
                      min_iters_for_stagnation=20)

Check convergence using three criteria:
1. Zero gradients: all gradients are zero (likely due to aggressive truncation)
2. Original ADAPT: max gradient below tolerance (works for unitary circuits)
3. Stagnation detection: max gradient change below threshold for multiple consecutive layers
   (better for non-unitary PP evolution where gradients stall at non-zero floor)

# Arguments
- `gradients`: Current iteration gradients
- `max_grads_history`: Vector of max gradients from all previous iterations
- `tol`: Absolute gradient convergence threshold (default: 1e-4)
- `stagnation_layers`: Number of consecutive layers to check for stagnation (default: 5)
- `stagnation_tol`: Maximum allowed change in max gradient to consider stagnation (default: 0.1)
- `min_iters_for_stagnation`: Minimum iterations before stagnation detection is enabled (default: 20)

# Returns
- `converged`: Boolean indicating convergence
- `max_grad`: Current maximum gradient
- `reason`: String describing convergence reason (or empty if not converged)
"""
function check_convergence(gradients, max_grads_history; tol=1e-4, stagnation_layers=5, stagnation_tol=0.1, 
                           min_iters_for_stagnation=20)
    max_grad = maximum(abs.(gradients))
    
    # Criterion 0: All gradients are zero (truncation issue)
    if all(abs.(grads) .< 1e-15 for grads in gradients)
        @warn "All gradients are zero. This may indicate overly aggressive truncation parameters."
        return true, max_grad, "all gradients are zero"
    end
    
    # Criterion 1: Original ADAPT (gradient below absolute tolerance)
    if max_grad < tol
        return true, max_grad, "gradient below tolerance ($tol)"
    end
    
    # Criterion 2: Stagnation detection (for non-unitary PP truncations)
    # Only check for stagnation after minimum number of iterations
    if length(max_grads_history) >= min_iters_for_stagnation && length(max_grads_history) >= stagnation_layers
        # Check last N layers for stagnation
        recent_grads = max_grads_history[end-(stagnation_layers-1):end]
        max_change = maximum(abs.(diff(recent_grads)))
        
        if max_change < stagnation_tol
            return true, max_grad, "gradient stagnation detected (max_change=$max_change < $stagnation_tol over $stagnation_layers layers)"
        end
    end
    
    return false, max_grad, ""
end

# ============================================================================
# TILES APPROACH FOR SCALABLE 2D SYSTEMS
# ============================================================================

"""
    generate_obc_square_tiles(tile_width::Int, scaled_lattice_width::Int)

Generate tiles for OBC square lattice topology.

# Arguments
- `tile_width`: Size of tile (e.g., 2 for 2x2 tile gives 4 qubits)
- `scaled_lattice_width`: Size of the full lattice (e.g., 3 for 3x3 lattice gives 9 qubits total)

# Returns
- Vector of vectors, where each inner vector contains the qubit indices for one tile

# Example
```julia
# For a 3x3 lattice with 2x2 tiles
tiles = generate_obc_square_tiles(2, 3)
# Returns 4 tiles: [[1,2,4,5], [2,3,5,6], [4,5,7,8], [5,6,8,9]]
```
"""
function generate_obc_square_tiles(tile_width::Int, scaled_lattice_width::Int)
    tiles_list = Vector{Int}[]
    
    # Generate the first tile (bottom left corner)
    first_tile = Int[]
    for i in 0:(tile_width-1)
        for j in 1:tile_width
            push!(first_tile, i * scaled_lattice_width + j)
        end
    end
    
    # Generate all tiles by shifting the first tile
    for j in 0:(scaled_lattice_width - tile_width)
        for i in 0:(scaled_lattice_width - tile_width)
            new_tile = (first_tile .+ j * scaled_lattice_width) .+ i
            push!(tiles_list, new_tile)
        end
    end
    
    return tiles_list
end

