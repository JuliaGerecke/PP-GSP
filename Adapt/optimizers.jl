"""
Optimization Functions for ADAPT-VQE
=====================================

This file contains loss functions, L-BFGS optimizers, and optimization wrappers
for different AD backends (Mooncake, ReverseDiff, ForwardDiff).
"""

using PauliPropagation
using NLopt
using ReverseDiff
using ForwardDiff
using Mooncake
using ThreadsX
using DifferentiationInterface

#using Plots
# Include dependencies
include("utilities.jl")

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

"""
    fulllossfunction_MC(thetas, circuit, nq, hamiltonian, overlap_func; kwargs...)

Mooncake-compatible loss function with min_abs_coeff truncation.

# Arguments
- `thetas`: Circuit parameters
- `circuit`: Quantum circuit
- `nq`: Number of qubits
- `hamiltonian`: Hamiltonian (PauliSum)
- `overlap_func`: Function to compute overlap with initial state
- `min_abs_coeff`: Minimum absolute coefficient for truncation
- `max_weight`: Maximum Pauli weight
- `overlap_kwargs`: Additional keyword arguments for overlap function
"""
function fulllossfunction_MC(thetas, circuit, nq, hamiltonian, overlap_func; 
                             min_abs_coeff=0.0, max_weight=Inf, overlap_kwargs...)
    H = deepcopy(hamiltonian)
    H = propagate!(circuit, H, thetas; min_abs_coeff, max_weight)
    return overlap_func(H, nq; overlap_kwargs...)
end

"""
    fulllossfunction_FD(thetas, circuit, nq, hamiltonian, overlap_func; kwargs...)

ForwardDiff-compatible loss function with min_abs_coeff truncation.
Promotes Hamiltonian coefficients to match theta type for AD.

# Arguments
- `thetas`: Circuit parameters
- `circuit`: Quantum circuit
- `nq`: Number of qubits
- `hamiltonian`: Hamiltonian (PauliSum)
- `overlap_func`: Function to compute overlap with initial state
- `min_abs_coeff`: Minimum absolute coefficient for truncation
- `max_weight`: Maximum Pauli weight
- `overlap_kwargs`: Additional keyword arguments for overlap function
"""
function fulllossfunction_FD(thetas, circuit, nq, hamiltonian, overlap_func; 
                             min_abs_coeff=0.0, max_weight=Inf, overlap_kwargs...)
    # Promote Hamiltonian coefficients to match theta type (for ForwardDiff Dual numbers)
    CT = eltype(thetas)
    H = promote_paulisum_coeffs(hamiltonian, CT)
    H_prop = propagate!(circuit, H, thetas; min_abs_coeff, max_weight)
    return overlap_func(H_prop, nq; overlap_kwargs...)
end

"""
    fulllossfunction_RD(thetas, circuit, nq, hamiltonian_constructor, overlap_func; kwargs...)

ReverseDiff-compatible loss function with max_freq truncation.

# Arguments
- `thetas`: Circuit parameters
- `circuit`: Quantum circuit
- `nq`: Number of qubits
- `hamiltonian_constructor`: Function to construct Hamiltonian with given coefficient type
  Should have signature: (CT, nq, topology; ham_params...) -> PauliSum{CT}
- `overlap_func`: Function to compute overlap with initial state
- `topology`: System topology
- `max_freq`: Maximum frequency for truncation
- `max_weight`: Maximum Pauli weight
- `overlap_kwargs`: Additional keyword arguments for overlap function
"""
function fulllossfunction_RD(thetas, circuit, nq, hamiltonian_constructor, overlap_func; 
                             topology=nothing, max_freq=Inf, max_weight=Inf, overlap_kwargs...)
    # differentiation libraries use custom types to trace through the computation
    # we need to make all of our objects typed like that so that nothing breaks
    CT = eltype(thetas)
    
    # Reconstruct Hamiltonian with correct coefficient type
    H = hamiltonian_constructor(CT, nq, topology)
    # wrap the coefficients into PauliFreqTracker so that we can use `max_freq` truncation.
    # usually this happens automatically but the in-place propagate!() function does not allow that.
    wrapped_H = wrapcoefficients(H, PauliFreqTracker)
    
    # we also need to run the in-place version with `!`, because by default we copy the Pauli sum
    output_H = propagate!(circuit, wrapped_H, thetas; max_freq, max_weight);
    return overlap_func(output_H, nq; overlap_kwargs...)
end

"""
    indiv_term_lossfunction_MC(thetas, which, circuit, nq, hamiltonian, overlap_func; kwargs...)

Loss function for individual Hamiltonian term (ThreadsX parallelization).
Compatible with Mooncake and ForwardDiff backends.

# Arguments
- `thetas`: Circuit parameters
- `which`: Index of Hamiltonian term
- `circuit`: Quantum circuit
- `nq`: Number of qubits
- `hamiltonian`: Hamiltonian (PauliSum)
- `overlap_func`: Function to compute overlap with initial state
- `min_abs_coeff`: Minimum absolute coefficient
- `max_weight`: Maximum Pauli weight
- `overlap_kwargs`: Additional keyword arguments for overlap function
"""
function indiv_term_lossfunction_MC(thetas, which, circuit, nq, hamiltonian, overlap_func; 
                                     min_abs_coeff=0.0, max_weight=Inf, overlap_kwargs...)
    # Extract individual Hamiltonian term (no coefficient promotion needed for Mooncake)
    pstr = topaulistrings(hamiltonian)[which]
    psum = PauliSum(pstr)
    
    psum_prop = propagate!(circuit, psum, thetas; min_abs_coeff, max_weight)
    
    if iterate(psum_prop) === nothing
        # Return type-stable zero
        return 0.0
    end
    
    result = overlap_func(psum_prop, nq; overlap_kwargs...)
    return real(result)  # Ensure real return type
end

"""
    indiv_term_lossfunction_RD(thetas, which, circuit, nq, hamiltonian_constructor, overlap_func; kwargs...)

Loss function for individual Hamiltonian term using ReverseDiff (max_freq truncation).

# Arguments
- `thetas`: Circuit parameters
- `which`: Index of Hamiltonian term
- `circuit`: Quantum circuit
- `nq`: Number of qubits
- `hamiltonian_constructor`: Function to construct Hamiltonian with given CT
- `overlap_func`: Function to compute overlap with initial state
- `topology`: System topology
- `max_freq`: Maximum frequency
- `max_weight`: Maximum Pauli weight
- `overlap_kwargs`: Additional keyword arguments for overlap function
"""
function indiv_term_lossfunction_RD(thetas, which, circuit, nq, hamiltonian_constructor, overlap_func; 
                                     hamiltonian_kwargs=NamedTuple(),
                                     topology=nothing, max_freq=Inf, max_weight=Inf, overlap_kwargs...)
    # Reconstruct Hamiltonian with correct coefficient type
    CT = eltype(thetas)
    H = hamiltonian_constructor(CT, nq, topology; hamiltonian_kwargs...)
    
    # Extract individual term
    pstr = topaulistrings(H)[which]
    psum = PauliSum(pstr)
    
    # Wrap coefficients for frequency tracking
    wrapped_psum = wrapcoefficients(psum, PauliFreqTracker)
    
    # Propagate
    output_psum = propagate!(circuit, wrapped_psum, thetas; max_freq, max_weight)
    
    if iterate(output_psum) === nothing
        return convert(CT, 0.0)
    end
    
    return overlap_func(output_psum, nq; overlap_kwargs...)
end

"""
    fulllossfunction_H2_MC(thetas, circuit, nq, hamiltonian, overlap_func; kwargs...)

Loss function for H² expectation value (used in v-score calculation).

Computes ⟨ψ|H²|ψ⟩ where |ψ⟩ = U(thetas)|φ₀⟩.

# Arguments
- `thetas`: Circuit parameters
- `circuit`: Quantum circuit
- `nq`: Number of qubits
- `hamiltonian`: Hamiltonian (PauliSum)
- `overlap_func`: Function to compute overlap with initial state
- `min_abs_coeff`: Minimum absolute coefficient for truncation
- `max_weight`: Maximum Pauli weight
- `overlap_kwargs`: Additional keyword arguments for overlap function

# Returns
- Expectation value ⟨H²⟩
"""
function fulllossfunction_H2_MC(thetas, circuit, nq, hamiltonian, overlap_func; 
                             min_abs_coeff=0.0, max_weight=Inf, overlap_kwargs...)
    H = deepcopy(hamiltonian)
    H2 = H*H # brute-force computation of H^2, computation with anticommutator is linearly better
    H2_prop = propagate!(circuit, H2, thetas; min_abs_coeff, max_weight)
    return overlap_func(H2_prop, nq; overlap_kwargs...)
end

"""
    vscore_calculation_MC(thetas, circuit, nq, hamiltonian, overlap_func; kwargs...)

Calculate v-score metric using Mooncake backend.

The v-score quantifies how well a variational ansatz approximates the ground state:
v-score = Var(E) × N / (E - E_inf)²

where:
- Var(E) = ⟨H²⟩ - ⟨H⟩² is the energy variance
- N is the number of qubits
- E_inf is the exact ground state energy (used as reference)

# Arguments
- `thetas`: Circuit parameters
- `circuit`: Quantum circuit
- `nq`: Number of qubits
- `hamiltonian`: Hamiltonian (PauliSum)
- `overlap_func`: Function to compute overlap with initial state
- `min_abs_coeff_vscore`: Minimum absolute coefficient for v-score calculation (default: 0.0)
- `max_weight_vscore`: Maximum Pauli weight for v-score calculation (default: Inf)
- `E_inf`: Exact ground state energy (default: 0.0)
- `overlap_kwargs`: Additional keyword arguments for overlap function

# Returns
- `vscore`: The v-score value
"""
function vscore_calculation_MC(thetas, circuit, nq, hamiltonian, overlap_func; verbose=false,
                             min_abs_coeff_vscore=0.0, max_weight_vscore=Inf, E_inf=0.0, overlap_kwargs...)
    E = fulllossfunction_MC(thetas, circuit, nq, hamiltonian, overlap_func; 
                             min_abs_coeff=min_abs_coeff_vscore, max_weight=max_weight_vscore, overlap_kwargs...)
    E2 = fulllossfunction_H2_MC(thetas, circuit, nq, hamiltonian, overlap_func; 
                             min_abs_coeff=min_abs_coeff_vscore, max_weight=max_weight_vscore, overlap_kwargs...)
    variance = real(E2 - E^2)  # Take real part to avoid complex arithmetic
    vscore = variance*nq / (E-E_inf)^2
    if verbose
        println("Energy = $E, nq= $nq, Variance = $variance, vscore = $vscore")
    end
    return vscore
end

# ============================================================================
# L-BFGS OPTIMIZERS
# ============================================================================

"""
    lbfgs_optimizer_MC(thetas_init, lossfun, nq; max_optim_iters=300, refresh_grad_tape=300)

L-BFGS optimizer using Mooncake backend for automatic differentiation.

# Arguments
- `thetas_init`: Initial parameters
- `lossfun`: Loss function to minimize
- `nq`: Number of qubits (for normalization)
- `max_optim_iters`: Maximum number of L-BFGS optimization steps (default: 300)
- `refresh_grad_tape`: Refresh gradient tape every N iterations (default: 300)

# Returns
- `thetas`: Optimized parameters
- `opt_energy`: Energy per qubit at each iteration
"""
function lbfgs_optimizer_MC(thetas_init, lossfun, nq; max_optim_iters=300, refresh_grad_tape=300)
    n = length(thetas_init)

    # Mooncake backend + prepared gradient object
    backend  = DifferentiationInterface.AutoMooncake(; config=nothing)
    prep_ref = Ref(DifferentiationInterface.prepare_gradient(lossfun, backend, thetas_init))

    iter_ref = Ref(0)

    opt = NLopt.Opt(:LD_LBFGS, n)
    NLopt.maxeval!(opt, max_optim_iters)

    opt_energy = Float64[]

    NLopt.min_objective!(opt, (x, grad) -> begin
        iter_ref[] += 1

        # Re-prepare periodically using the *current* x (helps if shapes/aliases change)
        if refresh_grad_tape > 0 && (iter_ref[] % refresh_grad_tape == 0)
            prep_ref[] = DifferentiationInterface.prepare_gradient(lossfun, backend, x)
        end

        fx = lossfun(x)::Float64

        if !isempty(grad)
            g = DifferentiationInterface.gradient(lossfun, prep_ref[], backend, x)
            @inbounds @simd for i in 1:n
                grad[i] = g[i]
            end
        end

        push!(opt_energy, fx)
        fx
    end)

    _minf, thetas, _ret = NLopt.optimize(opt, thetas_init)
    return thetas, opt_energy ./ nq
end

"""
    lbfgs_optimizer_RD(thetas_init, lossfun, nq; max_optim_iters=300, refresh_grad_tape=300)

L-BFGS optimizer using ReverseDiff backend for automatic differentiation.

# Arguments
- `thetas_init`: Initial parameters
- `lossfun`: Loss function to minimize
- `nq`: Number of qubits (for normalization)
- `max_optim_iters`: Maximum number of L-BFGS optimization steps (default: 300)
- `refresh_grad_tape`: Refresh gradient tape every N iterations (default: 300)

# Returns
- `thetas`: Optimized parameters
- `opt_energy`: Energy per qubit at each iteration
"""
function lbfgs_optimizer_RD(thetas_init, lossfun, nq; max_optim_iters=300, refresh_grad_tape=300)
    n = length(thetas_init)
    tape_ref = Ref(GradientTape(lossfun, thetas_init)) # Ref is only necessary in the LBFGS case because the closure otherwise can't update an outer variable that NLopt is still using.
    compile(tape_ref[])
    gradbuf = zeros(n)
    iter_ref = Ref(0)
    opt = NLopt.Opt(:LD_LBFGS, n)
    NLopt.maxeval!(opt, max_optim_iters)

    opt_energy = Float64[]
    NLopt.min_objective!(opt, (x, grad) -> begin

        iter_ref[] += 1
        # Re-record the tape every `refresh_grad_tape` evaluations
        # record with the current x so shapes/aliasing match
        if refresh_grad_tape > 0 && (iter_ref[] % refresh_grad_tape == 0)
            tape_ref[] = GradientTape(lossfun, x)
            compile(tape_ref[])
        end

        fx = lossfun(x)::Float64
        if !isempty(grad)
            ReverseDiff.gradient!(gradbuf, tape_ref[], x) #TODO test
            grad .= gradbuf
        end
        push!(opt_energy, fx)
        return fx
    end)
    minf, thetas, ret = NLopt.optimize(opt, thetas_init)
    return thetas, opt_energy ./ nq
end

"""
    lbfgs_optimizer_FD(thetas_init, lossfun, nq; max_optim_iters=300, refresh_grad_tape=300)

L-BFGS optimizer using ForwardDiff backend for automatic differentiation.

# Arguments
- `thetas_init`: Initial parameters
- `lossfun`: Loss function to minimize
- `nq`: Number of qubits (for normalization)
- `max_optim_iters`: Maximum number of L-BFGS optimization steps (default: 300)
- `refresh_grad_tape`: Included for API compatibility (not used with ForwardDiff)

# Returns
- `thetas`: Optimized parameters
- `opt_energy`: Energy per qubit at each iteration
"""
function lbfgs_optimizer_FD(thetas_init, lossfun, nq; max_optim_iters=300, refresh_grad_tape=300)
    n = length(thetas_init)
    opt = NLopt.Opt(:LD_LBFGS, n)
    NLopt.maxeval!(opt, max_optim_iters)

    opt_energy = Float64[]
    
    NLopt.min_objective!(opt, (x, grad) -> begin
        if !isempty(grad)
            g = ForwardDiff.gradient(lossfun, x)
            @inbounds @simd for i in 1:n # TODO: check simd
                grad[i] = g[i]
            end
        end
        
        fx = lossfun(x)
        fx_val = real(fx)  # Extract real value (handles both Float64 and Dual)
        push!(opt_energy, fx_val)
        return fx_val
    end)
    
    _minf, thetas, _ret = NLopt.optimize(opt, thetas_init)
    return thetas, opt_energy ./ nq
end

# ============================================================================
# THREADSX PARALLELIZED OPTIMIZERS
# ============================================================================

"""
    lbfgs_optimizer_threadsx_MC(thetas_init, lossfun_list, nq; kwargs...)

L-BFGS optimizer with ThreadsX-parallelized gradient computation using Mooncake backend.

# Arguments
- `thetas_init`: Initial parameters
- `lossfun_list`: List of loss functions (one per Hamiltonian term)
- `nq`: Number of qubits
- `use_threadsx`: Use ThreadsX parallelization (default: true)
- `use_tape`: Use prepared gradient tapes (default: false)
- `max_optim_iters`: Maximum L-BFGS optimization steps (default: 300)
- `refresh_grad_tape`: Refresh tapes every N iterations (default: 300)
"""
function lbfgs_optimizer_threadsx_MC(thetas_init, lossfun_list, nq; 
                                     use_threadsx=true, 
                                     use_tape=false,
                                     max_optim_iters=300, 
                                     refresh_grad_tape=300)
    n = length(thetas_init)
    backend = DifferentiationInterface.AutoMooncake(; config=nothing)
    
    tape_refs = if use_tape
        [Ref(DifferentiationInterface.prepare_gradient(func, backend, thetas_init)) 
         for func in lossfun_list]
    else
        nothing
    end
    
    iter_ref = Ref(0)
    
    opt = NLopt.Opt(:LD_LBFGS, n)
    NLopt.maxeval!(opt, max_optim_iters)
    
    opt_energy = Float64[]
    
    NLopt.min_objective!(opt, (x, grad) -> begin
        iter_ref[] += 1
        
        # Refresh tapes periodically
        if use_tape && refresh_grad_tape > 0 && (iter_ref[] % refresh_grad_tape == 0)
            for (i, func) in enumerate(lossfun_list)
                tape_refs[i][] = DifferentiationInterface.prepare_gradient(func, backend, x)
            end
        end
        
        # Compute full loss
        fx = sum(func(x) for func in lossfun_list)
        
        if !isempty(grad)
            # Compute gradient (parallel or serial)
            if use_threadsx
                if use_tape && tape_refs !== nothing
                    g = ThreadsX.sum(DifferentiationInterface.gradient(func, tape_refs[i][], backend, x) 
                                    for (i, func) in enumerate(lossfun_list))
                else
                    g = ThreadsX.sum(DifferentiationInterface.gradient(func, backend, x) 
                                    for func in lossfun_list)
                end
            else
                if use_tape && tape_refs !== nothing
                    g = sum(DifferentiationInterface.gradient(func, tape_refs[i][], backend, x) 
                           for (i, func) in enumerate(lossfun_list))
                else
                    g = sum(DifferentiationInterface.gradient(func, backend, x) 
                           for func in lossfun_list)
                end
            end
            
            @inbounds @simd for i in 1:n
                grad[i] = g[i]
            end
        end
        
        push!(opt_energy, fx)
        fx
    end)
    
    _minf, thetas, _ret = NLopt.optimize(opt, thetas_init)
    return thetas, opt_energy ./ nq
end

"""
    lbfgs_optimizer_threadsx_FD(thetas_init, lossfun_list, nq; kwargs...)

L-BFGS optimizer with ThreadsX-parallelized gradient computation using ForwardDiff backend.

# Arguments
- `thetas_init`: Initial parameters
- `lossfun_list`: List of loss functions (one per Hamiltonian term)
- `nq`: Number of qubits
- `use_threadsx`: Use ThreadsX parallelization (default: true)
- `max_optim_iters`: Maximum L-BFGS optimization steps (default: 300)
"""
function lbfgs_optimizer_threadsx_FD(thetas_init, lossfun_list, nq; 
                                     use_threadsx=true, 
                                     max_optim_iters=300)
    n = length(thetas_init)
    
    iter_ref = Ref(0)
    
    opt = NLopt.Opt(:LD_LBFGS, n)
    NLopt.maxeval!(opt, max_optim_iters)
    
    opt_energy = Float64[]
    
    NLopt.min_objective!(opt, (x, grad) -> begin
        iter_ref[] += 1
        
        # Compute full loss first (type-stable)
        fx_val = 0.0
        for func in lossfun_list
            result = func(x)
            fx_val += real(result)
        end
        
        if !isempty(grad)
            # Compute gradient (parallel or serial)
            if use_threadsx
                g = ThreadsX.sum(ForwardDiff.gradient(func, x) for func in lossfun_list)
            else
                g = sum(ForwardDiff.gradient(func, x) for func in lossfun_list)
            end
            
            @inbounds @simd for i in 1:n
                grad[i] = g[i]
            end
        end
        
        push!(opt_energy, fx_val)
        return fx_val
    end)
    
    _minf, thetas, _ret = NLopt.optimize(opt, thetas_init)
    return thetas, opt_energy ./ nq
end

"""
    lbfgs_optimizer_threadsx_RD(thetas_init, lossfun_list, nq; kwargs...)

L-BFGS optimizer with ThreadsX-parallelized gradient computation using ReverseDiff backend.

# Arguments
- `thetas_init`: Initial parameters
- `lossfun_list`: List of loss functions (one per Hamiltonian term)
- `nq`: Number of qubits
- `use_threadsx`: Use ThreadsX parallelization (default: true)
- `use_tape`: Use gradient tapes (default: false)
- `max_optim_iters`: Maximum L-BFGS optimization steps (default: 300)
- `refresh_grad_tape`: Refresh tapes every N iterations (default: 300)
"""
function lbfgs_optimizer_threadsx_RD(thetas_init, lossfun_list, nq; 
                                     use_threadsx=true,
                                     use_tape=false,
                                     max_optim_iters=300,
                                     refresh_grad_tape=300)
    n = length(thetas_init)
    
    tape_refs = if use_tape
        [Ref(GradientTape(func, thetas_init)) for func in lossfun_list]
    else
        nothing
    end
    
    # Compile tapes if using them
    if use_tape && tape_refs !== nothing
        for tape_ref in tape_refs
            compile(tape_ref[])
        end
    end
    
    iter_ref = Ref(0)
    
    opt = NLopt.Opt(:LD_LBFGS, n)
    NLopt.maxeval!(opt, max_optim_iters)
    
    opt_energy = Float64[]
    
    NLopt.min_objective!(opt, (x, grad) -> begin
        iter_ref[] += 1
        
        # Refresh tapes periodically
        if use_tape && refresh_grad_tape > 0 && (iter_ref[] % refresh_grad_tape == 0)
            for (i, func) in enumerate(lossfun_list)
                tape_refs[i][] = GradientTape(func, x)
                compile(tape_refs[i][])
            end
        end
        
        # Compute full loss
        fx = sum(func(x) for func in lossfun_list)
        
        if !isempty(grad)
            # Compute gradient (parallel or serial)
            if use_threadsx
                if use_tape && tape_refs !== nothing
                    # Parallel with tapes
                    grad_buffers = [zeros(n) for _ in 1:length(lossfun_list)]
                    Threads.@threads for i in 1:length(lossfun_list)
                        ReverseDiff.gradient!(grad_buffers[i], tape_refs[i][], x)
                    end
                    g = sum(grad_buffers)
                else
                    # Parallel without tapes
                    g = ThreadsX.sum(ReverseDiff.gradient(func, x) for func in lossfun_list)
                end
            else
                if use_tape && tape_refs !== nothing
                    # Serial with tapes
                    grad_buffer = zeros(n)
                    g = zeros(n)
                    for tape_ref in tape_refs
                        ReverseDiff.gradient!(grad_buffer, tape_ref[], x)
                        g .+= grad_buffer
                    end
                else
                    # Serial without tapes
                    g = sum(ReverseDiff.gradient(func, x) for func in lossfun_list)
                end
            end
            
            @inbounds @simd for i in 1:n
                grad[i] = g[i]
            end
        end
        
        push!(opt_energy, fx)
        fx
    end)
    
    _minf, thetas, _ret = NLopt.optimize(opt, thetas_init)
    return thetas, opt_energy ./ nq
end

# ============================================================================
# OPTIMIZATION WRAPPERS
# ============================================================================

"""
    target_optimization_MC(nq, circuit, thetas, hamiltonian, overlap_func; kwargs...)

Optimize circuit parameters using Mooncake backend.

# Arguments
- `nq`: Number of qubits
- `circuit`: Quantum circuit
- `thetas`: Initial parameters
- `hamiltonian`: Hamiltonian (PauliSum)
- `overlap_func`: Function to compute overlap with initial state
- `min_abs_coeff`: Minimum absolute coefficient
- `max_weight`: Maximum Pauli weight
- `verbose`: Print optimization progress
- `refresh_grad_tape`: Refresh gradient tape frequency
- `overlap_kwargs`: Additional keyword arguments for overlap function
"""
function target_optimization_MC(nq, circuit, thetas, hamiltonian, overlap_func; 
                                min_abs_coeff=0.0, max_weight=Inf, verbose=false, 
                                refresh_grad_tape=300, overlap_kwargs...)
    # Create closure with all necessary variables
    closed_lossfunction = let const_circ=circuit, const_nq=nq, const_hamiltonian=hamiltonian, 
                             const_overlap_func=overlap_func, const_min_abs_coeff=min_abs_coeff, 
                             const_max_weight=max_weight, const_overlap_kwargs=overlap_kwargs
        theta -> fulllossfunction_MC(theta, const_circ, const_nq, const_hamiltonian, const_overlap_func; 
                                    min_abs_coeff=const_min_abs_coeff, max_weight=const_max_weight, 
                                    const_overlap_kwargs...)
    end
    
    opt_thetas, opt_energy_gd = lbfgs_optimizer_MC(thetas, closed_lossfunction, nq; refresh_grad_tape=refresh_grad_tape)
    
    if verbose
        println("Optimized thetas: ", opt_thetas)
        println("Optimized energy per qubit: ", opt_energy_gd[end])
        # if PLOTS_AVAILABLE
        #     plot(opt_energy_gd)
        #     display(plot!(title="Energy optimisation", xlabel="runs", ylabel="E/Q"))
        # end
    end

    return opt_thetas, opt_energy_gd
end

"""
    target_optimization_RD(nq, circuit, thetas, hamiltonian_constructor, overlap_func; kwargs...)

Optimize circuit parameters using ReverseDiff backend.

# Arguments
- `nq`: Number of qubits
- `circuit`: Quantum circuit
- `thetas`: Initial parameters
- `hamiltonian_constructor`: Function to construct Hamiltonian with given CT
- `overlap_func`: Function to compute overlap with initial state
- `topology`: System topology
- `max_freq`: Maximum frequency
- `max_weight`: Maximum Pauli weight
- `verbose`: Print optimization progress
- `refresh_grad_tape`: Refresh gradient tape frequency
- `overlap_kwargs`: Additional keyword arguments for overlap function
"""
function target_optimization_RD(nq, circuit, thetas, hamiltonian_constructor, overlap_func; 
                                hamiltonian_kwargs=NamedTuple(),
                                topology=nothing, max_freq=Inf, max_weight=Inf, 
                                verbose=false, refresh_grad_tape=300, overlap_kwargs...)
    # Wrap hamiltonian_constructor to include kwargs
    ham_constructor_with_kwargs = (CT, nq, topo) -> hamiltonian_constructor(CT, nq, topo; hamiltonian_kwargs...)
    
    # Create closure with all necessary variables
    closed_lossfunction = let const_nq=nq, const_ham_constructor=ham_constructor_with_kwargs,
                             const_overlap_func=overlap_func, const_topology=topology, 
                             const_max_freq=max_freq, const_max_weight=max_weight,
                             const_overlap_kwargs=overlap_kwargs
        theta -> fulllossfunction_RD(theta, circuit, const_nq, const_ham_constructor, const_overlap_func; 
                                    topology=const_topology, max_freq=const_max_freq, 
                                    max_weight=const_max_weight, const_overlap_kwargs...)
    end
    
    opt_thetas, opt_energy_gd = lbfgs_optimizer_RD(thetas, closed_lossfunction, nq; refresh_grad_tape=refresh_grad_tape)
    
    if verbose
        println("Optimized thetas: ", opt_thetas)
        println("Optimized energy per qubit: ", opt_energy_gd[end])
        # if PLOTS_AVAILABLE
        #     plot(opt_energy_gd)
        #     display(plot!(title="Energy optimisation", xlabel="runs", ylabel="E/Q"))
        # end
    end

    return opt_thetas, opt_energy_gd
end

"""
    target_optimization_FD(nq, circuit, thetas, hamiltonian, overlap_func; kwargs...)

Optimize circuit parameters using ForwardDiff backend (no tape).
Uses min_abs_coeff truncation instead of max_freq.

# Arguments
- `nq`: Number of qubits
- `circuit`: Quantum circuit
- `thetas`: Initial parameters
- `hamiltonian`: Hamiltonian (PauliSum)
- `overlap_func`: Function to compute overlap with initial state
- `min_abs_coeff`: Minimum absolute coefficient
- `max_weight`: Maximum Pauli weight
- `verbose`: Print optimization progress
- `refresh_grad_tape`: Included for API compatibility (not used)
- `overlap_kwargs`: Additional keyword arguments for overlap function
"""
function target_optimization_FD(nq, circuit, thetas, hamiltonian, overlap_func; 
                                min_abs_coeff=0.0, max_weight=Inf, 
                                verbose=false, refresh_grad_tape=300, overlap_kwargs...)
    # Define loss function using ForwardDiff-compatible version
    lossfun = theta -> fulllossfunction_FD(theta, circuit, nq, hamiltonian, overlap_func; 
                                          min_abs_coeff=min_abs_coeff, max_weight=max_weight, 
                                          overlap_kwargs...)
    
    opt_thetas, opt_energy_gd = lbfgs_optimizer_FD(thetas, lossfun, nq; refresh_grad_tape=refresh_grad_tape)
    
    if verbose
        println("Optimized thetas: ", opt_thetas)
        println("Optimized energy per qubit: ", opt_energy_gd[end])
        # if PLOTS_AVAILABLE
        #     plot(opt_energy_gd)
        #     display(plot!(title="Energy optimisation", xlabel="runs", ylabel="E/Q"))
        # end
    end

    return opt_thetas, opt_energy_gd
end

"""
    target_optimization_threadsx(nq, circuit, thetas, hamiltonian, overlap_func; backend=:mooncake, kwargs...)

Optimize with ThreadsX-parallelized gradient computation (backend-specific).

# Arguments
- `nq`: Number of qubits
- `circuit`: Quantum circuit
- `thetas`: Initial parameters
- `hamiltonian`: Hamiltonian (PauliSum) - for Mooncake/ForwardDiff backends
- `hamiltonian_constructor`: Function to construct Hamiltonian (for ReverseDiff)
- `overlap_func`: Function to compute overlap with initial state
- `backend`: `:mooncake` (default), `:forwarddiff`, or `:reversediff`
- `topology`: System topology (for ReverseDiff)
- `min_abs_coeff`: Minimum absolute coefficient (Mooncake/ForwardDiff)
- `max_freq`: Maximum frequency (ReverseDiff)
- `max_weight`: Maximum Pauli weight
- `use_threadsx`: Use ThreadsX parallelization
- `use_tape`: Use prepared gradient tapes
- `refresh_grad_tape`: Refresh gradient tape frequency
- `overlap_kwargs`: Additional keyword arguments for overlap function
"""
function target_optimization_threadsx(nq, circuit, thetas, hamiltonian, overlap_func; 
                                      hamiltonian_constructor=nothing,
                                      hamiltonian_kwargs=NamedTuple(),
                                      backend::Symbol=:mooncake,
                                      topology=nothing,
                                      min_abs_coeff=0.0,
                                      max_freq=Inf,
                                      max_weight=Inf,
                                      use_threadsx=true, 
                                      use_tape=false,
                                      refresh_grad_tape=300,
                                      overlap_kwargs...)
    
    # If use_threadsx=false, route to non-threaded optimizers with full lossfunction
    if !use_threadsx
        if backend == :mooncake
            return target_optimization_MC(nq, circuit, thetas, hamiltonian, overlap_func; 
                                         min_abs_coeff=min_abs_coeff, max_weight=max_weight,
                                         verbose=false, refresh_grad_tape=refresh_grad_tape,
                                         overlap_kwargs...)
        elseif backend == :forwarddiff
            return target_optimization_FD(nq, circuit, thetas, hamiltonian, overlap_func; 
                                         min_abs_coeff=min_abs_coeff, max_weight=max_weight,
                                         verbose=false, refresh_grad_tape=refresh_grad_tape,
                                         overlap_kwargs...)
        elseif backend == :reversediff
            if isnothing(hamiltonian_constructor) || isnothing(topology)
                throw(ArgumentError("ReverseDiff backend requires hamiltonian_constructor and topology"))
            end
            return target_optimization_RD(nq, circuit, thetas, hamiltonian_constructor, overlap_func; 
                                         hamiltonian_kwargs=hamiltonian_kwargs,
                                         topology=topology, max_freq=max_freq, max_weight=max_weight,
                                         verbose=false, refresh_grad_tape=refresh_grad_tape,
                                         overlap_kwargs...)
        else
            throw(ArgumentError("Unsupported backend: $backend. Use :mooncake, :forwarddiff, or :reversediff"))
        end
    else
        # use_threadsx=true: Use parallelized optimizers with per-term lossfunctions
        if backend == :mooncake
        num_terms = length(hamiltonian)
        
        # Create loss function for each Hamiltonian term
        lossfun_list = [let circuit=circuit, nq=nq, hamiltonian=hamiltonian, 
                            overlap_func=overlap_func, min_abs_coeff=min_abs_coeff, 
                            max_weight=max_weight, which=which, overlap_kwargs=overlap_kwargs
            theta -> indiv_term_lossfunction_MC(theta, which, circuit, nq, hamiltonian, overlap_func; 
                                                min_abs_coeff=min_abs_coeff, max_weight=max_weight, 
                                                overlap_kwargs...)
        end for which in 1:num_terms]
        
        opt_thetas, opt_energy = lbfgs_optimizer_threadsx_MC(thetas, lossfun_list, nq; 
                                                              use_threadsx=use_threadsx,
                                                              use_tape=use_tape, 
                                                              refresh_grad_tape=refresh_grad_tape)
        
    elseif backend == :forwarddiff
        num_terms = length(hamiltonian)
        
        # Create loss function for each Hamiltonian term
        # Promote coefficients to handle ForwardDiff Dual numbers correctly
        lossfun_list = [let circuit=circuit, nq=nq, hamiltonian=hamiltonian, 
                            overlap_func=overlap_func, min_abs_coeff=min_abs_coeff, 
                            max_weight=max_weight, which=which, overlap_kwargs=overlap_kwargs
            function (theta)
                CT = eltype(theta)
                pstr = topaulistrings(hamiltonian)[which]
                psum_base = PauliSum(pstr)
                psum = promote_paulisum_coeffs(psum_base, CT)
                psum_prop = propagate!(circuit, psum, theta; min_abs_coeff=min_abs_coeff, max_weight=max_weight)
                
                if iterate(psum_prop) === nothing
                    return zero(CT)
                end
                
                return real(overlap_func(psum_prop, nq; overlap_kwargs...))
            end
        end for which in 1:num_terms]
        
        opt_thetas, opt_energy = lbfgs_optimizer_threadsx_FD(thetas, lossfun_list, nq; 
                                                              use_threadsx=use_threadsx)
        
    elseif backend == :reversediff
        if isnothing(hamiltonian_constructor) || isnothing(topology)
            throw(ArgumentError("ReverseDiff backend requires hamiltonian_constructor and topology"))
        end
        
        # Get number of terms by constructing with Float64
        temp_ham = hamiltonian_constructor(Float64, nq, topology; hamiltonian_kwargs...)
        num_terms = length(temp_ham)
        
        # Create loss function for each Hamiltonian term
        lossfun_list = [let circuit=circuit, nq=nq, ham_constructor=hamiltonian_constructor,
                            overlap_func=overlap_func, topology=topology, max_freq=max_freq,
                            max_weight=max_weight, which=which, hamiltonian_kwargs=hamiltonian_kwargs,
                            overlap_kwargs=overlap_kwargs
            theta -> indiv_term_lossfunction_RD(theta, which, circuit, nq, ham_constructor, overlap_func; 
                                                hamiltonian_kwargs=hamiltonian_kwargs,
                                                topology=topology, max_freq=max_freq, 
                                                max_weight=max_weight, overlap_kwargs...)
        end for which in 1:num_terms]
        
        opt_thetas, opt_energy = lbfgs_optimizer_threadsx_RD(thetas, lossfun_list, nq; 
                                                              use_threadsx=use_threadsx,
                                                              use_tape=use_tape,
                                                              refresh_grad_tape=refresh_grad_tape)
        else
            throw(ArgumentError("Unsupported backend: $backend. Use :mooncake, :forwarddiff, or :reversediff"))
        end
        
        return opt_thetas, opt_energy
    end
end

"""
    run_multiple_optimizations(nq, circuit, thetas, hamiltonian, overlap_func;
                               rerun_optim=1, verbose=false,
                               hamiltonian_constructor=nothing, hamiltonian_kwargs=NamedTuple(),
                               backend=:mooncake, topology=nothing,
                               min_abs_coeff=0.0, max_freq=Inf, max_weight=Inf,
                               use_threadsx=true, use_tape=false, refresh_grad_tape=300,
                               overlap_kwargs...)

Run optimization multiple times and return the best result with timing information.

# Arguments
- `nq`: Number of qubits
- `circuit`: Quantum circuit
- `thetas`: Initial parameters
- `hamiltonian`: Hamiltonian (PauliSum)
- `overlap_func`: Function to compute overlap with initial state
- `rerun_optim`: Number of optimization runs (default: 1)
- `verbose`: Print detailed progress (default: false)
- Additional kwargs passed to `target_optimization_threadsx`

# Returns
Named tuple with:
- `best_thetas`: Optimized parameters with lowest energy
- `best_opt_energy`: Energy trajectory of best run
- `total_time`: Total optimization time across all runs
- `total_memory`: Total memory used across all runs
"""
function run_multiple_optimizations(nq, circuit, thetas, hamiltonian, overlap_func;
                                   rerun_optim::Int=1,
                                   verbose::Bool=false,
                                   hamiltonian_constructor=nothing,
                                   hamiltonian_kwargs=NamedTuple(),
                                   backend::Symbol=:mooncake,
                                   topology=nothing,
                                   min_abs_coeff::Real=0.0,
                                   max_freq::Real=Inf,
                                   max_weight::Real=Inf,
                                   use_threadsx::Bool=true,
                                   use_tape::Bool=false,
                                   refresh_grad_tape::Int=300,
                                   overlap_kwargs...)
    
    best_thetas = nothing
    best_opt_energy = nothing
    best_final_energy = Inf
    total_time = 0.0
    total_memory = 0
    
    for run_idx in 1:rerun_optim
        opt_result = @timed target_optimization_threadsx(nq, circuit, thetas, hamiltonian, overlap_func; 
                                                  hamiltonian_constructor=hamiltonian_constructor,
                                                  hamiltonian_kwargs=hamiltonian_kwargs,
                                                  backend=backend,
                                                  topology=topology,
                                                  min_abs_coeff=min_abs_coeff,
                                                  max_freq=max_freq,
                                                  max_weight=max_weight,
                                                  use_threadsx=use_threadsx,
                                                  use_tape=use_tape,
                                                  refresh_grad_tape=refresh_grad_tape,
                                                  overlap_kwargs...)
        
        run_thetas, run_opt_energy = opt_result.value
        run_final_energy = run_opt_energy[end]
        
        total_time += opt_result.time
        total_memory += opt_result.bytes
        
        # Keep the best run (lowest final energy)
        if run_final_energy < best_final_energy
            best_thetas = run_thetas
            best_opt_energy = run_opt_energy
            best_final_energy = run_final_energy
            if rerun_optim > 1 && verbose
                println("    Run $run_idx/$rerun_optim: E/Q = $run_final_energy (new best)")
            end
        elseif rerun_optim > 1 && verbose
            println("    Run $run_idx/$rerun_optim: E/Q = $run_final_energy")
        end
    end
    
    return (best_thetas=best_thetas,
            best_opt_energy=best_opt_energy,
            total_time=total_time,
            total_memory=total_memory)
end
