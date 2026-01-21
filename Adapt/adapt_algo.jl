"""
ADAPT-VQE Main Algorithms
==========================

This file contains the main ADAPT-VQE algorithm implementations with ThreadsX support,
loop versions with progressively refined truncations, and scaled pool generation for 2D systems.
"""

using PauliPropagation
using Random
using ThreadsX

# Include all dependencies
include("utilities.jl")
include("hamiltonians.jl")
include("optimizers.jl")
include("adapt_steps.jl")

# ============================================================================
# MAIN ADAPT-VQE ALGORITHMS
# ============================================================================

"""
    adaptVQE_2d_threadsx(total_nq, hamiltonian, bit_pool, overlap_func; backend=:mooncake, kwargs...)

ADAPT-VQE with ThreadsX-parallelized optimization.

Uses target_optimization_threadsx for optimization with ThreadsX support, and supports 
ThreadsX parallelization for gradient calculations via the threads_oppool parameter.

# Arguments
All arguments from adaptVQE_2d, plus ThreadsX-specific parameters:
- `total_nq`: Total number of qubits in the system
- `hamiltonian`: Hamiltonian (PauliSum)
- `bit_pool`: Operator pool in bit representation
- `overlap_func`: Function to compute overlap with initial state
- `hamiltonian_constructor`: Function to construct Hamiltonian with given CT (for ReverseDiff)
- `backend`: `:mooncake` (default), `:reversediff`, or `:forwarddiff`
- `topology`: System topology
- `use_threadsx`: Use ThreadsX parallelization in optimization (default: true)
- `use_tape`: Use prepared gradient tapes (default: false)
- `threads_oppool`: Use ThreadsX parallelization for operator pool gradients (default: false)
- All other arguments same as adaptVQE_2d

# Returns
Same as adaptVQE_2d: (circuit, thetas, chosen_ops, energy_per_loop, max_grads, final_vscore, timings)
"""
function adaptVQE_2d_threadsx(total_nq, hamiltonian, bit_pool, overlap_func; 
                              hamiltonian_constructor=nothing,
                              hamiltonian_kwargs=NamedTuple(),
                              backend::Symbol=:mooncake,
                              topology=nothing,
                              max_iters=40,
                              min_abs_coeff=0.0,
                              max_freq=Inf,
                              max_weight=Inf,
                              conv_tol=1e-4,
                              stagnation_layers=5,
                              stagnation_tol=0.1,
                              min_iters_for_stagnation=30,
                              use_threadsx=true,
                              use_tape=false,
                              refresh_grad_tape=300,
                              verbose=true, 
                              vscore=false, 
                              min_abs_coeff_vscore=1e-7, 
                              max_weight_vscore=20,
                              calc_grads::Symbol=:phys,
                              threads_oppool::Bool=false,
                              calc_grad_kwargs=nothing,
                              rerun_optim::Int=1,
                              circuit=nothing,
                              thetas=nothing,
                              overlap_kwargs...)
    
    # Validate inputs
    if backend == :reversediff && (isnothing(hamiltonian_constructor) || isnothing(topology))
        throw(ArgumentError("ReverseDiff backend requires hamiltonian_constructor and topology"))
    end
    
    # Initialize circuit and thetas (warm start if provided)
    if isnothing(circuit)
        circuit = Any[]
    end
    if isnothing(thetas)
        thetas = Float64[]
    end
    
    energy_per_loop = Float64[]
    max_grads = Float64[]
    chosen_ops = eltype(bit_pool)[]
    final_vscore = nothing

    # Timing data collection
    commutator_times = Float64[]
    gradient_times = Float64[]
    optimization_times = Float64[]
    
    # Memory data collection 
    commutator_memory = Float64[]
    gradient_memory = Float64[]
    optimization_memory = Float64[]

    # Store final optimized thetas for each ADAPT iteration
    optimized_thetas_per_iter = Vector{Float64}[]

    for iter in 1:max_iters
        # 1) Calculate gradients with timing and memory tracking
        grad_results = calculate_adapt_gradients(bit_pool, hamiltonian, total_nq, overlap_func;
                                                 calc_grads=calc_grads,
                                                 circuit=circuit,
                                                 thetas=thetas,
                                                 threads_oppool=threads_oppool,
                                                 min_abs_coeff=min_abs_coeff,
                                                 max_weight=max_weight,
                                                 calc_grad_kwargs=calc_grad_kwargs,
                                                 overlap_kwargs...)
        
        grads = grad_results.grads
        push!(commutator_times, grad_results.commutator_time)
        push!(commutator_memory, grad_results.commutator_memory)
        push!(gradient_times, grad_results.gradient_time)
        push!(gradient_memory, grad_results.gradient_memory)

        # 2) Check convergence
        converged, max_grad, conv_reason = check_convergence(grads, max_grads; 
                                                             tol=conv_tol, 
                                                             stagnation_layers=stagnation_layers,
                                                             stagnation_tol=stagnation_tol,
                                                             min_iters_for_stagnation=min_iters_for_stagnation)
        push!(max_grads, max_grad)
        
        if converged
            verbose && println("Convergence achieved: $conv_reason")
            break
        end
        
        # 3) Pick top operator
        chose_op, grad_op, grads_sorted, _ = pick_top_operator(grads, bit_pool)
        push!(chosen_ops, chose_op) 
        verbose && println("Iteration $iter: Selected operator with gradient $grad_op")

        # 4) Append operator to circuit using generalized function
        # For full pool 2D: sites=nothing means sequential qubits [1:total_nq]
        new_theta_init = rand() * 2π - π
        circuit, thetas = append_from_bits_general!(circuit, thetas, chose_op, total_nq; 
                                                    sites=nothing, theta_init=new_theta_init)

        if verbose
            println("  Circuit length: ", length(circuit))
            #println("Circuit: ", circuit)
            println("  Parameters: ", length(thetas))
        end

        # 5) Run optimization (possibly multiple times) and keep best result
        opt_results = run_multiple_optimizations(total_nq, circuit, thetas, hamiltonian, overlap_func;
                                                rerun_optim=rerun_optim,
                                                verbose=verbose,
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
        
        # Extract results
        thetas = opt_results.best_thetas
        opt_energy = opt_results.best_opt_energy
        push!(optimization_times, opt_results.total_time)
        push!(optimization_memory, opt_results.total_memory)
        push!(optimized_thetas_per_iter, copy(thetas))

        push!(energy_per_loop, opt_energy[end])
        verbose && println("  Energy/qubit: ", opt_energy[end])
    end

    # 6) Plot convergence if verbose (jed cluster environment has problem with Julia Plots package (does not support current version))
    # if verbose && PLOTS_AVAILABLE
    #     plot(max_grads, marker=:o)
    #     display(plot!(title="Max Gradient Convergence", 
    #                  xlabel="Iteration", ylabel="Max Gradient"))
    # end

    # 7) V-score calculation
    vscore_time = 0.0
    if vscore==true
        vscore_result = @timed vscore_calculation_MC(thetas, circuit, total_nq, hamiltonian, overlap_func; 
                             min_abs_coeff_vscore=min_abs_coeff_vscore, 
                             max_weight_vscore=max_weight_vscore,
                             overlap_kwargs...)
        final_vscore = vscore_result.value
        vscore_time = vscore_result.time
        verbose && println("Final v-score: ", final_vscore)
    end
    
    # Aggregate timing data
    timings = Dict(
        "commutator_time_total" => sum(commutator_times),
        "gradient_time_total" => sum(gradient_times),
        "optimization_time_total" => sum(optimization_times),
        "vscore_time" => vscore_time,
        "commutator_times" => commutator_times,
        "gradient_times" => gradient_times,
        "optimization_times" => optimization_times,
        "commutator_memory_total" => sum(commutator_memory),
        "gradient_memory_total" => sum(gradient_memory),
        "optimization_memory_total" => sum(optimization_memory),
        "commutator_memory" => commutator_memory,
        "gradient_memory" => gradient_memory,
        "optimization_memory" => optimization_memory,
        "num_iterations" => length(gradient_times),
        "optimized_thetas_per_iter" => optimized_thetas_per_iter
    )
    
    if verbose
        println("All chosen ops: ", chosen_ops)
    end
    
    return circuit, thetas, chosen_ops, energy_per_loop, max_grads, final_vscore, timings
end

"""
    adaptVQE_2d_threadsx_loop(total_nq, hamiltonian, bit_pool, overlap_func; kwargs...)

ThreadsX-parallelized version of ADAPT-VQE loop for 2D (and generalized 1D/2D) systems.
Wrapper around adaptVQE_2d that progressively reduces truncation parameters.

# Arguments
All arguments from adaptVQE_2d_loop, including:
- `total_nq`: Total number of qubits in the system
- `hamiltonian`: Hamiltonian (PauliSum)
- `bit_pool`: Operator pool in bit representation (pre-generated, e.g., from scaled_pool_generation_2d)
- `overlap_func`: Function to compute overlap with initial state
- `hamiltonian_constructor`: Function to construct Hamiltonian (for ReverseDiff)
- `backend`: `:mooncake` (default), `:forwarddiff`, or `:reversediff`
- `topology`: System topology (1D or 2D)
- `max_iters`: Maximum ADAPT iterations per run
- `mw_sequence`: Array of max_weight values for each run (default: [4, 5, 6])
- `mac_sequence`: Array of min_abs_coeff values for each run (default: [1e-4, 1e-5, 1e-6])
- `max_freq_sequence`: Array of max_freq values for each run (default: [50, 100, 150], ReverseDiff only)
- `min_iters_for_stagnation_sequence`: Array of min_iters_for_stagnation for each run (default: [30, 20, 10])
- `threads_oppool`: Use ThreadsX parallelization for operator pool gradients (default: false)
- All other kwargs from adaptVQE_2d are supported

# Returns
- `circuit`: Final converged circuit
- `thetas`: Final optimized parameters
- `chosen_ops`: All chosen operators across all runs
- `energy_per_loop`: Energy history across all runs
- `max_grads`: Gradient history across all runs
- `final_vscore`: V-score of final circuit (if vscore=true)
- `timings`: Timing data from final run
- `all_timings`: Array of timing dicts from all runs
- `all_energies`: Array of energy arrays from all runs
- `all_max_grads`: Array of gradient arrays from all runs

# Example
```julia
# For 2D system with Mooncake backend
circuit, thetas, ops, energies, grads, vscore, timings, all_timings, all_energies, all_grads = 
    adaptVQE_2d_threadsx_loop(9, H, pool, overlapwithneel; 
                           backend=:mooncake,
                           topology=rectangletopology(3, 3),
                           mw_sequence=[4, 5, 6],
                           mac_sequence=[1e-4, 1e-5, 1e-6],
                           min_iters_for_stagnation_sequence=[30, 20, 10],
                           threads_oppool=true)

# Works with 1D topology too
circuit, thetas, ops, energies, grads, vscore, timings, all_timings, all_energies, all_grads = 
    adaptVQE_2d_threadsx_loop(6, H_1d, pool_1d, overlapwithneel; 
                           backend=:mooncake,
                           topology=bricklayertopology(6),
                           mw_sequence=[4, 5, 6],
                           mac_sequence=[1e-4, 1e-5, 1e-6],
                           threads_oppool=true)
```
"""
function adaptVQE_2d_threadsx_loop(total_nq, hamiltonian, bit_pool, overlap_func;
                                hamiltonian_constructor=nothing,
                                hamiltonian_kwargs=NamedTuple(),
                                backend::Symbol=:mooncake,
                                topology=nothing,
                                max_iters=40,
                                mw_sequence=[4, 5, 6],
                                mac_sequence=[1e-4, 1e-5, 1e-6],
                                max_freq_sequence=[50, 100, 150],
                                min_iters_for_stagnation_sequence=[30, 20, 10],
                                conv_tol=1e-4,
                                stagnation_layers=5,
                                stagnation_tol=0.1,
                                use_threadsx=true,
                                use_tape=false,
                                refresh_grad_tape=300,
                                verbose=true,
                                vscore=false,
                                min_abs_coeff_vscore=1e-7,
                                max_weight_vscore=20,
                                calc_grads::Symbol=:phys,
                                threads_oppool::Bool=false,
                                calc_grad_kwargs=nothing,
                                rerun_optim::Int=1,
                                initial_circuit=nothing,
                                initial_thetas=nothing,
                                overlap_kwargs...)
    
    # Validate sequence lengths
    num_runs = length(mw_sequence)
    if backend == :reversediff
        # For ReverseDiff, use max_freq_sequence
        if length(max_freq_sequence) != num_runs
            throw(ArgumentError("max_freq_sequence length ($(length(max_freq_sequence))) must match mw_sequence length ($num_runs)"))
        end
    else
        # For Mooncake/ForwardDiff, use mac_sequence
        if length(mac_sequence) != num_runs
            throw(ArgumentError("mac_sequence length ($(length(mac_sequence))) must match mw_sequence length ($num_runs)"))
        end
    end
    
    if length(min_iters_for_stagnation_sequence) != num_runs
        throw(ArgumentError("min_iters_for_stagnation_sequence length ($(length(min_iters_for_stagnation_sequence))) must match mw_sequence length ($num_runs)"))
    end
    
    # Initialize with provided circuit and thetas (or empty if not provided)
    current_circuit = initial_circuit
    current_thetas = initial_thetas
    
    # Storage for all intermediate results
    all_timings = []
    all_energies = []
    all_max_grads = []
    all_chosen_ops = []
    
    # Final results (will be updated after each run)
    final_circuit = nothing
    final_thetas = nothing
    final_chosen_ops = nothing
    final_energy_per_loop = nothing
    final_max_grads = nothing
    final_vscore = nothing
    final_timings = nothing
    
    for run_idx in 1:num_runs
        current_mw = mw_sequence[run_idx]
        current_min_iters_stag = min_iters_for_stagnation_sequence[run_idx]
        
        if backend == :reversediff
            current_max_freq = max_freq_sequence[run_idx]
            current_mac = 0.0  # Not used for ReverseDiff
            truncation_str = "max_freq=$current_max_freq, max_weight=$current_mw"
        else
            current_mac = mac_sequence[run_idx]
            current_max_freq = Inf  # Not used for Mooncake/ForwardDiff
            truncation_str = "min_abs_coeff=$current_mac, max_weight=$current_mw"
        end
        
        verbose && println("\n" * "="^80)
        verbose && println("ADAPT-VQE 2D ThreadsX Loop: Run $run_idx/$num_runs")
        verbose && println("Truncation parameters: $truncation_str")
        verbose && println("min_iters_for_stagnation: $current_min_iters_stag")
        verbose && println("="^80)
        
        # Run adaptVQE_2d_threadsx with current truncation parameters and warm start
        circuit, thetas, chosen_ops, energy_per_loop, max_grads, run_vscore, timings = 
            adaptVQE_2d_threadsx(total_nq, hamiltonian, bit_pool, overlap_func;
                     hamiltonian_constructor=hamiltonian_constructor,
                     hamiltonian_kwargs=hamiltonian_kwargs,
                     backend=backend,
                     topology=topology,
                     max_iters=max_iters,
                     min_abs_coeff=current_mac,
                     max_freq=current_max_freq,
                     max_weight=current_mw,
                     conv_tol=conv_tol,
                     stagnation_layers=stagnation_layers,
                     stagnation_tol=stagnation_tol,
                     min_iters_for_stagnation=current_min_iters_stag,
                     use_threadsx=use_threadsx,
                     use_tape=use_tape,
                     refresh_grad_tape=refresh_grad_tape,
                     verbose=verbose,
                     vscore=vscore && (run_idx == num_runs),  # Only compute vscore on final run
                     min_abs_coeff_vscore=min_abs_coeff_vscore,
                     max_weight_vscore=max_weight_vscore,
                     calc_grads=calc_grads,
                     threads_oppool=threads_oppool,
                     calc_grad_kwargs=calc_grad_kwargs,
                     rerun_optim=rerun_optim,
                     circuit=current_circuit,  # Warm start with previous circuit
                     thetas=current_thetas,  # Warm start with previous thetas
                     overlap_kwargs...)
        
        # Store intermediate results
        push!(all_timings, timings)
        push!(all_energies, energy_per_loop)
        push!(all_max_grads, max_grads)
        push!(all_chosen_ops, chosen_ops)
        
        # Update final results
        final_circuit = circuit
        final_thetas = thetas
        final_chosen_ops = vcat(all_chosen_ops...)  # Concatenate all chosen ops
        final_energy_per_loop = vcat(all_energies...)  # Concatenate all energies
        final_max_grads = vcat(all_max_grads...)  # Concatenate all gradients
        final_vscore = run_vscore
        final_timings = timings
        
        # Set current circuit and thetas for next iteration (warm start)
        current_circuit = circuit
        current_thetas = thetas
        
        verbose && println("\nRun $run_idx/$num_runs completed:")
        verbose && !isempty(energy_per_loop) && println("  Final energy/qubit: $(energy_per_loop[end])")
        verbose && println("  Layers added this run: $(length(chosen_ops))")
        verbose && println("  Total layers: $(length(circuit))")
    end
    
    verbose && println("\n" * "="^80)
    verbose && println("ADAPT-VQE 2D ThreadsX Loop completed after $num_runs runs")
    verbose && println("Final circuit has $(length(final_circuit)) layers")
    verbose && !isempty(final_energy_per_loop) && println("Final energy/qubit: $(final_energy_per_loop[end])")
    verbose && println("="^80)
    
    return final_circuit, final_thetas, final_chosen_ops, final_energy_per_loop, 
           final_max_grads, final_vscore, final_timings, all_timings, all_energies, all_max_grads
end

# ============================================================================
# SCALED POOL GENERATION (TILES APPROACH)
# ============================================================================

"""
    scaled_pool_selection_2d(total_nq, hamiltonian, full_bit_pool; topology=nothing, kwargs...)

Run multiple ADAPT-VQE instances to select important operators for 2D systems.
Backward compatible with 1D systems when topology is 1D.

# Arguments
- `total_nq`: Total number of qubits in the tile system
- `hamiltonian`: Hamiltonian for the tile system
- `full_bit_pool`: Full operator pool for the tile system
- `topology`: System topology (required for 2D)
- All other kwargs same as scaled_pool_selection
"""
function scaled_pool_selection_2d(total_nq, hamiltonian, full_bit_pool; 
                                   backend::Symbol=:mooncake,
                                   topology=nothing,
                                   num_runs=5, 
                                   run_iters=5,
                                   conv_tol=1e-4, 
                                   refresh_grad_tape=300,
                                   verbose=false,
                                   overlap_func=overlapwithneel,
                                   overlap_kwargs...)
    all_chosen_ops = eltype(full_bit_pool)[]
    
    for _ in 1:num_runs
        circuit, thetas, chosen_ops, _, _, _, _ = 
            adaptVQE_2d_threadsx(total_nq, hamiltonian, full_bit_pool, overlap_func; 
                        use_threadsx=false,
                        backend=backend,
                        topology=topology,
                        max_iters=run_iters, 
                        conv_tol=conv_tol,
                        refresh_grad_tape=refresh_grad_tape, 
                        verbose=false,
                        overlap_kwargs...)
        append!(all_chosen_ops, chosen_ops)
    end
    
    return unique(all_chosen_ops)
end

"""
    scaled_pool_generation_2d(tile_nq, scaled_nq, hamiltonian, full_bit_pool; tiles=nothing, kwargs...)

Generate scaled operator pool using tiles approach for 2D systems (backward compatible with 1D).

This function generalizes the tiles approach to work with arbitrary tile configurations.
For 2D systems, use `generate_obc_square_tiles` or similar to create the tiles.
For 1D systems, tiles defaults to 1D sliding window behavior.

# Arguments
- `tile_nq`: Number of qubits in the tile
- `scaled_nq`: Total number of qubits in the scaled system
- `hamiltonian`: Hamiltonian for the tile system
- `full_bit_pool`: Full operator pool for the tile
- `tiles`: Vector of vectors, where each inner vector contains qubit indices for one tile.
          If `nothing`, defaults to 1D sliding window with qubit_stride
- `topology`: Topology for the tile system (for operator selection)
- `qubit_stride`: Stride for 1D sliding window (default: 1). Only used if tiles=nothing
- All other kwargs same as scaled_pool_generation

# Returns
- Scaled operator pool (unique operators)

# Example for 2D
```julia
# 2x2 tile → 3x3 scaled system
tile_width = 2
scaled_width = 3
tiles = generate_obc_square_tiles(tile_width, scaled_width)
scaled_pool = scaled_pool_generation_2d(4, 9, tile_H, tile_pool; 
                                        tiles=tiles, 
                                        topology=tile_topo)
```

# Example for 1D (backward compatible)
```julia
# Same as scaled_pool_generation
scaled_pool = scaled_pool_generation_2d(3, 6, tile_H, tile_pool; 
                                        qubit_stride=1)
```
"""
function scaled_pool_generation_2d(tile_nq, scaled_nq, hamiltonian, full_bit_pool; 
                                    backend::Symbol=:mooncake,
                                    topology=nothing,
                                    tiles=nothing,
                                    num_runs=5, 
                                    run_iters=5,
                                    conv_tol=1e-4, 
                                    refresh_grad_tape=300,
                                    verbose=false,
                                    qubit_stride::Int=1,
                                    overlap_func=overlapwithneel,
                                    overlap_kwargs...)
    UIntType = PauliPropagation.getinttype(scaled_nq)
    
    # Select operators from tile system
    chosen_ops = scaled_pool_selection_2d(tile_nq, hamiltonian, full_bit_pool; 
                                          backend=backend,
                                          topology=topology,
                                          num_runs=num_runs, 
                                          run_iters=run_iters,
                                          conv_tol=conv_tol, 
                                          refresh_grad_tape=refresh_grad_tape,
                                          verbose=verbose,
                                          overlap_func=overlap_func,
                                          overlap_kwargs...)
    
    verbose && println("Chosen operators from tile ADAPT-VQE: ", length(chosen_ops))
    
    scaled_pool = UIntType[]
    
    # If tiles not provided, use 1D sliding window (backward compatible)
    if isnothing(tiles)
        verbose && println("Using 1D sliding window with stride=$qubit_stride")
        
        for op in chosen_ops
            ps, paulis, sites = bit_to_paulistring_general(op, tile_nq; sites=nothing, total_nq=tile_nq)
            
            # Preserve internal geometry
            min_site = minimum(sites)
            rel_sites = sites .- min_site
            block_span = maximum(rel_sites) + 1
            
            # For fermionic systems with qubit_stride=2, validate site boundaries
            if qubit_stride == 2
                valid_operator = true
                for q in sites
                    partner = (q % 2 == 1) ? q + 1 : q - 1
                    if !(partner in sites)
                        valid_operator = false
                        verbose && println("  Skipping operator - qubit $q missing partner $partner")
                        break
                    end
                end
                if !valid_operator
                    continue
                end
            end
            
            # Slide block across system
            for start in 1:qubit_stride:(scaled_nq - block_span + 1)
                new_sites = start .+ rel_sites
                new_op = PauliString(scaled_nq, paulis, collect(new_sites), 1.0)
                push!(scaled_pool, new_op.term)
            end
        end
    else
        # Use provided tiles configuration (2D or custom)
        verbose && println("Using provided tiles configuration with $(length(tiles)) tiles")
        
        for op in chosen_ops
            # Convert bit representation to Pauli string for the tile
            ps, paulis, active_sites = bit_to_paulistring_general(op, tile_nq; sites=nothing, total_nq=tile_nq)
            
            # Apply this operator pattern to each tile in the scaled system
            for tile_sites in tiles
                # Map the operator to this tile's qubit sites
                mapped_pstr, mapped_paulis, mapped_active_sites = 
                    bit_to_paulistring_general(op, tile_nq; 
                                              sites=tile_sites, 
                                              total_nq=scaled_nq)
                
                # Add to scaled pool
                push!(scaled_pool, mapped_pstr.term)
            end
        end
    end
    
    return unique(scaled_pool)
end
