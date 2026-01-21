"""
Scaling Benchmark Script for ADAPT-VQE 
=======================================

This script runs systematic benchmarks for ADAPT-VQE simulations on any spin model,
varying truncation parameters (max_weight, min_abs_coeff) and logging all results
for later analysis.

Functions:
Log related:
- `save_run_results`: Save results to JLD2 file.
- `append_to_log`: Append results to existing JLD2 log file or create new one.

Re-evaluation:
- `reevaluate_circuit_MC`: Re-evaluate a circuit at different truncation parameters (Mooncake backend).
- `reevaluate_circuit_RD`: Re-evaluate a circuit at different truncation parameters (ReverseDiff backend).

Benchmarks:
- `run_single_loop_benchmark_2d`: Run ADAPT-VQE loop for 2D/1D systems (generalized with ThreadsX).
- `run_loop_scaling_benchmarks_2d`: Run systematic loop benchmarks for 2D/1D systems.

Features:
- ThreadsX parallelization for all backends (ForwardDiff, ReverseDiff, Mooncake)
- Systematic parameter sweeps over max_weight and min_abs_coeff
- Progressive truncation refinement with warm-start optimization (Loop Implementation)
- Support for both 1D (chain) and 2D (square lattice) topologies
- Automatic topology detection and pool generation
- Comprehensive logging of circuits, parameters, and energy curves
- Optional re-evaluation at lower truncation levels
- Timing measurements for each run
- Incremental saving to preserve partial results

"""

using PauliPropagation
using Random
using Dates
using JLD2
using BenchmarkTools
using Printf
#using Plots # Plots is optional - only load if available  


import Base: merge

# Import the ADAPT-VQE implementation
include("adapt_modular.jl")


# ============================================================================
# MEMORY MONITORING UTILITIES
# ============================================================================

"""
    get_process_memory_mb()

Get current process memory usage in MB using /proc/self/status (Linux).
Returns VmRSS (Resident Set Size) - actual physical memory used.
"""
function get_process_memory_mb()
    try
        # Read /proc/self/status to get VmRSS
        status = read("/proc/self/status", String)
        vmrss_match = match(r"VmRSS:\s+(\d+)\s+kB", status)
        if !isnothing(vmrss_match)
            vmrss_kb = parse(Float64, vmrss_match.captures[1])
            return vmrss_kb / 1024.0  # Convert to MB
        end
    catch e
        # Fallback for non-Linux systems or if /proc is unavailable
        @warn "Could not read /proc/self/status: $e. Memory monitoring disabled."
    end
    return 0.0
end

"""
    format_memory_size(bytes)

Format memory size in bytes to human-readable string (KB, MB, GB).
"""
function format_memory_size(bytes::Real)
    if bytes < 1024
        return @sprintf("%.1f B", bytes)
    elseif bytes < 1024^2
        return @sprintf("%.1f KB", bytes / 1024)
    elseif bytes < 1024^3
        return @sprintf("%.1f MB", bytes / 1024^2)
    else
        return @sprintf("%.2f GB", bytes / 1024^3)
    end
end


# ============================================================================
# LOGGING AND DATA MANAGEMENT
# ============================================================================

"""
    save_run_results(filename, data_dict)

Save results to JLD2 file.
"""
function save_run_results(filename::String, data_dict::Dict)
    @save filename data_dict
    println("  ✓ Results saved to: $filename")
end


"""
    append_to_log(log_file, run_data)

Append results to existing JLD2 log file or create new one.
"""
function append_to_log(log_file::String, run_data::Dict)
    if isfile(log_file)
        # Load existing data
        existing_data = load(log_file)
        
        # The file should have a "runs" key with an array of run dictionaries
        if haskey(existing_data, "runs")
            runs = existing_data["runs"]
        else
            # If structure is wrong, start fresh
            runs = []
        end
        
        # Append new run
        push!(runs, run_data)
        
        println("  ✓ Loaded existing master log with $(length(runs)-1) runs")
        
        # Save back with updated runs array
        @save log_file runs
        
        println("  ✓ Appended to master log (now $(length(runs)) runs)")
    else
        # Create new file with runs array
        runs = [run_data]
        @save log_file runs
        println("  ✓ Created new master log with 1 run")
    end
end


# ============================================================================
# RE-EVALUATION AT LOWER TRUNCATION 
# ============================================================================

"""
    reevaluate_circuit_MC(circuit, thetas, nq, hamiltonian, overlap_func; 
                       max_weight_new, min_abs_coeff_new, overlap_kwargs...)

Re-evaluate a circuit at different truncation parameters (Mooncake backend).
"""
function reevaluate_circuit_MC(circuit, thetas, nq, hamiltonian, overlap_func; 
                           max_weight_new::Int, 
                           min_abs_coeff_new::Float64,
                           overlap_kwargs...)
    println("    Re-evaluating at max_weight=$max_weight_new, min_abs_coeff=$min_abs_coeff_new")
    
    # Create loss function with new truncation parameters
    closed_lossfunction = let const_circ=circuit, const_nq=nq, 
                             const_hamiltonian=hamiltonian, const_overlap_func=overlap_func,
                             const_min_abs_coeff=min_abs_coeff_new, 
                             const_max_weight=max_weight_new,
                             const_overlap_kwargs=overlap_kwargs
        theta -> fulllossfunction_MC(theta, const_circ, const_nq, const_hamiltonian, const_overlap_func; 
                                    min_abs_coeff=const_min_abs_coeff, 
                                    max_weight=const_max_weight,
                                    const_overlap_kwargs...)
    end
    
    # Evaluate energy with current parameters
    energy = closed_lossfunction(thetas)
    energy_per_qubit = energy / nq
    
    println("      → Energy/qubit: $energy_per_qubit")
    
    return energy_per_qubit
end

#TODO: no need for closure here?

"""
    reevaluate_circuit_RD(circuit, thetas, nq, topology; 
                          max_weight_new, max_freq_new, hamiltonian_kwargs)

Re-evaluate a circuit at different truncation parameters (ReverseDiff backend).
"""
function reevaluate_circuit_RD(circuit, thetas, nq, hamiltonian_constructor, overlap_func; 
                              hamiltonian_kwargs=NamedTuple(),
                              topology=nothing,
                              max_weight_new::Int, 
                              max_freq_new::Float64, overlap_kwargs...)
    println("    Re-evaluating at max_weight=$max_weight_new, max_freq=$max_freq_new")
    
    # Wrap hamiltonian_constructor to include kwargs
    ham_constructor_with_kwargs = (CT, nq, topo) -> hamiltonian_constructor(CT, nq, topo; hamiltonian_kwargs...)
    
    # Create loss function with new truncation parameters
    closed_lossfunction = let const_nq=nq, const_ham_constructor=ham_constructor_with_kwargs,
                             const_overlap_func=overlap_func, const_topology=topology, 
                             const_max_freq=max_freq_new, const_max_weight=max_weight_new,
                             const_overlap_kwargs=overlap_kwargs
        theta -> fulllossfunction_RD(theta, circuit, const_nq, const_ham_constructor, const_overlap_func; 
                                    topology=const_topology, max_freq=const_max_freq, 
                                    max_weight=const_max_weight, const_overlap_kwargs...)
    end

    
    # Evaluate energy with current parameters
    energy = closed_lossfunction(thetas)
    energy_per_qubit = energy / nq
    
    println("      → Energy/qubit: $energy_per_qubit")
    
    return energy_per_qubit
end






"""
    run_single_loop_benchmark_2d(total_nq, hamiltonian, scaled_pool, topology, overlap_func, 
                                  hamiltonian_constructor; kwargs...)

Generalized benchmark function for 2D (and 1D) systems with progressive truncation refinement.
This function is backward compatible with both `run_single_benchmark` and `run_single_loop_benchmark`.

Uses `adaptVQE_2d_threadsx_loop` internally, which works for both 1D and 2D topologies with 
ThreadsX parallelization support. The scaled_pool is assumed to be pre-generated (e.g., using 
`scaled_pool_generation_2d`), so no tiles argument is needed.

# Backward Compatibility
- When called with single max_weight/min_abs_coeff/max_freq (no sequences), behaves like run_single_benchmark
- When called with sequences, behaves like run_single_loop_benchmark with progressive refinement
- Works with 1D topologies (bricklayertopology) and 2D topologies (rectangletopology)

# Arguments
- `total_nq`: Total number of qubits in the system
- `hamiltonian`: Hamiltonian operator
- `scaled_pool`: Pre-generated operator pool (from scaled_pool_generation_2d or similar)
- `topology`: System topology (1D or 2D)
- `overlap_func`: Overlap function for initial state
- `hamiltonian_constructor`: Function to construct Hamiltonian (for ReverseDiff backend)
- `hamiltonian_kwargs`: Keyword arguments for Hamiltonian constructor
- `mw_sequence`: Vector of max_weight values for progressive refinement (default: [4, 5, 6])
                 If single value, runs without loop
- `mac_sequence`: Vector of min_abs_coeff values (Mooncake/ForwardDiff, default: [1e-4, 1e-5, 1e-6])
- `max_freq_sequence`: Vector of max_freq values (ReverseDiff, default: [50.0, 100.0, 150.0])
- `min_iters_for_stagnation_sequence`: Vector of stagnation thresholds (default: [30, 20, 10])
- `backend`: `:mooncake` (default), `:forwarddiff`, or `:reversediff`
- `threads`: Use threading (default: true) - automatically used via adaptVQE_2d_threadsx
- `threads_oppool`: Use ThreadsX parallelization for operator pool gradients (default: false)
- All other kwargs same as run_single_loop_benchmark

# Returns
- `circuit`: Final optimized circuit
- `thetas`: Final optimized parameters
- `run_data`: Dictionary with comprehensive benchmark results
- `chosen_ops`: Bit representations of chosen operators

# Examples
```julia
# Single run (like run_single_benchmark)
circuit, thetas, data, ops = run_single_loop_benchmark_2d(
    9, H, pool, topo, overlap, ham_constr;
    mw_sequence=[5], mac_sequence=[1e-4], min_iters_for_stagnation_sequence=[20])

# Progressive loop (like run_single_loop_benchmark)
circuit, thetas, data, ops = run_single_loop_benchmark_2d(
    9, H, pool, topo, overlap, ham_constr;
    mw_sequence=[4,5,6], mac_sequence=[1e-4,1e-5,1e-6])

# Works with 1D topology too
circuit, thetas, data, ops = run_single_loop_benchmark_2d(
    6, H_1d, pool_1d, bricklayertopology(6), overlap, ham_constr)
```
"""
function run_single_loop_benchmark_2d(total_nq::Int, 
                                      hamiltonian, scaled_pool, topology,
                                      overlap_func, hamiltonian_constructor;
                                      hamiltonian_kwargs=NamedTuple(),
                                      mw_sequence::Vector{Int}=[4, 5, 6],
                                      mac_sequence::Vector{Float64}=[1e-4, 1e-5, 1e-6],
                                      max_freq_sequence::Vector{Float64}=[50.0, 100.0, 150.0],
                                      min_iters_for_stagnation_sequence::Vector{Int}=[30, 20, 10],
                                      tile_nq::Int=0,  # Not used, kept for compatibility
                                      refresh_grad_tape::Int=300,
                                      max_iters::Int=40,
                                      conv_tol::Float64=1e-2,
                                      stagnation_layers::Int=5,
                                      stagnation_tol::Float64=0.1,
                                      backend::Symbol=:mooncake,
                                      threads::Bool=true,
                                      calc_grads::Symbol=:phys,
                                      threads_oppool::Bool=false,
                                      calc_grad_kwargs=nothing,
                                      verbose::Bool=false, 
                                      vscore::Bool=false,
                                      num_reruns::Int=1,  # Number of reruns for the same optimization
                                      initial_circuit=nothing,
                                      initial_thetas=nothing,
                                      overlap_kwargs...)
    
    # Determine if this is a single run or a loop
    num_runs = length(mw_sequence)
    is_loop = num_runs > 1
    
    # Check number of threads available
    n_threads = Base.Threads.nthreads()
    println("  Julia using $n_threads threads")
    
    if is_loop
        println("  Running ADAPT-VQE 2D Loop with backend=$backend, threads=$threads")
        println("    mw_sequence=$mw_sequence")
        if backend == :reversediff
            println("    max_freq_sequence=$max_freq_sequence")
        else
            println("    mac_sequence=$mac_sequence")
        end
        println("    min_iters_for_stagnation_sequence=$min_iters_for_stagnation_sequence")
        if num_reruns > 1
            println("    num_reruns=$num_reruns (will select best energy)")
        end
    else
        println("  Running ADAPT-VQE 2D (single) with backend=$backend, threads=$threads, " *
                "max_weight=$(mw_sequence[1]), " *
                (backend == :reversediff ? "max_freq=$(max_freq_sequence[1])" : "min_abs_coeff=$(mac_sequence[1])"))
        if num_reruns > 1
            println("    num_reruns=$num_reruns (will select best energy)")
        end
    end
    
    # Track memory usage before run
    memory_before_mb = get_process_memory_mb()
    gc_stats_before = Base.gc_num()
    
    # Storage for multiple reruns - MODIFIED to only store best
    best_result = nothing
    best_energy = Inf
    best_rerun_idx = 1
    all_rerun_final_energies = Vector{Float64}(undef, num_reruns)
    all_rerun_elapsed_times = Vector{Float64}(undef, num_reruns)
    all_rerun_gc_times = Vector{Float64}(undef, num_reruns)
    all_rerun_memory_bytes = Vector{Int}(undef, num_reruns)
    
    total_elapsed_time = 0.0
    total_gc_time = 0.0
    total_memory_bytes = 0
    
    for rerun_idx in 1:num_reruns
        if num_reruns > 1
            println("\n  === Rerun $rerun_idx/$num_reruns ===")
        end
        
        # Run the appropriate ADAPT-VQE variant
        result = @timed begin
            # Use adaptVQE_2d_threadsx_loop (works for both loop and single-run cases)
            circuit, thetas, chosen_ops, energy_curve, max_grads, final_vscore, timings, 
            all_timings, all_energies, all_max_grads = 
                adaptVQE_2d_threadsx_loop(total_nq, hamiltonian, scaled_pool, overlap_func;
                                       hamiltonian_constructor=hamiltonian_constructor,
                                       hamiltonian_kwargs=hamiltonian_kwargs,
                                       backend=backend,
                                       topology=topology,
                                       max_iters=max_iters,
                                       mw_sequence=mw_sequence,
                                       mac_sequence=mac_sequence,
                                       max_freq_sequence=max_freq_sequence,
                                       min_iters_for_stagnation_sequence=min_iters_for_stagnation_sequence,
                                       conv_tol=conv_tol,
                                       stagnation_layers=stagnation_layers,
                                       stagnation_tol=stagnation_tol,
                                       refresh_grad_tape=refresh_grad_tape,
                                       verbose=verbose,
                                       vscore=vscore,
                                       min_abs_coeff_vscore=1e-7,
                                       max_weight_vscore=20,
                                       calc_grads=calc_grads,
                                       threads_oppool=threads_oppool,
                                       calc_grad_kwargs=calc_grad_kwargs,
                                       rerun_optim=1,
                                       initial_circuit=initial_circuit,
                                       initial_thetas=initial_thetas,
                                       overlap_kwargs...)
        end
        
        elapsed_time = result.time
        memory_bytes = result.bytes
        gc_time = result.gctime
        
        # Get final energy
        final_energy = energy_curve[end]
        all_rerun_final_energies[rerun_idx] = final_energy
        all_rerun_elapsed_times[rerun_idx] = elapsed_time
        all_rerun_gc_times[rerun_idx] = gc_time
        all_rerun_memory_bytes[rerun_idx] = memory_bytes
        
        # Accumulate totals
        total_elapsed_time += elapsed_time
        total_gc_time += gc_time
        total_memory_bytes += memory_bytes
        
        if num_reruns > 1
            println("    Rerun $rerun_idx: Final energy = $final_energy (depth=$(length(circuit)))")
        end
        
        # Check if this is the best run so far - ONLY STORE BEST
        if final_energy < best_energy
            best_energy = final_energy
            # Replace with new best result (only store one complete result)
            best_result = (circuit=circuit, thetas=thetas, chosen_ops=chosen_ops,
                          energy_curve=energy_curve, max_grads=max_grads,
                          final_vscore=final_vscore, timings=timings,
                          all_timings=all_timings, all_energies=all_energies,
                          all_max_grads=all_max_grads,
                          elapsed_time=elapsed_time, memory_bytes=memory_bytes,
                          gc_time=gc_time)
            best_rerun_idx = rerun_idx
        end
        
        # Force garbage collection between reruns
        if rerun_idx < num_reruns
            GC.gc()
        end
    end
    
    # Select the best result
    if num_reruns > 1
        println("\n  ✓ Best result: Rerun $best_rerun_idx with energy = $best_energy")
        println("    All final energies: $all_rerun_final_energies")
    end
    
    # Extract the best result components
    circuit = best_result.circuit
    thetas = best_result.thetas
    chosen_ops = best_result.chosen_ops
    energy_curve = best_result.energy_curve
    max_grads = best_result.max_grads
    final_vscore = best_result.final_vscore
    timings = best_result.timings
    all_timings = best_result.all_timings
    all_energies = best_result.all_energies
    all_max_grads = best_result.all_max_grads
    elapsed_time = best_result.elapsed_time
    memory_bytes = best_result.memory_bytes
    gc_time = best_result.gc_time
    
    # Track memory usage after run
    memory_after_mb = get_process_memory_mb()
    gc_stats_after = Base.gc_num()
    
    # Calculate memory metrics
    memory_increase_mb = memory_after_mb - memory_before_mb
    gc_time_ratio = total_gc_time / total_elapsed_time
    total_allocations = gc_stats_after.allocd - gc_stats_before.allocd
    
    if num_reruns == 1
        println("    ✓ Completed in $(round(elapsed_time, digits=2))s " *
                "(GC: $(round(gc_time, digits=2))s [$(round(gc_time_ratio*100, digits=1))%])")
        println("    Memory allocated: $(format_memory_size(memory_bytes))")
    else
        println("\n    ✓ All reruns completed in $(round(total_elapsed_time, digits=2))s " *
                "(GC: $(round(total_gc_time, digits=2))s [$(round(gc_time_ratio*100, digits=1))%])")
        println("    Total memory allocated: $(format_memory_size(total_memory_bytes))")
    end
    println("    RSS increase: $(round(memory_increase_mb, digits=1)) MB")
    println("    Peak RSS: $(round(memory_after_mb, digits=1)) MB")
    println("    Final energy/qubit: $(energy_curve[end])")
    println("    Final circuit depth: $(length(circuit))")
    if vscore
        println("    Final v-score: $final_vscore")
    end
    
    # Build result dictionary
    if is_loop
        # Build nested dictionary structure for loop: Dict(run_idx => run_data)
        loop_run_data = Dict{Any, Dict{String, Any}}()
        
        for run_idx in 1:num_runs
            current_mw = mw_sequence[run_idx]
            current_min_iters_stag = min_iters_for_stagnation_sequence[run_idx]
            
            if backend == :reversediff
                current_max_freq = max_freq_sequence[run_idx]
                current_mac = 0.0
            else
                current_mac = mac_sequence[run_idx]
                current_max_freq = Inf
            end
            
            run_timings = all_timings[run_idx]
            run_energies = all_energies[run_idx]
            run_grads = all_max_grads[run_idx]
            
            # Extract per-iteration memory arrays
            gradient_memory = haskey(run_timings, "gradient_memory") ? run_timings["gradient_memory"] : Float64[]
            commutator_memory = haskey(run_timings, "commutator_memory") ? run_timings["commutator_memory"] : Float64[]
            optimization_memory = haskey(run_timings, "optimization_memory") ? run_timings["optimization_memory"] : Float64[]
            
            # Extract per-iteration RSS arrays
            rss_after_commutator = haskey(run_timings, "rss_after_commutator") ? run_timings["rss_after_commutator"] : Float64[]
            rss_after_gradient = haskey(run_timings, "rss_after_gradient") ? run_timings["rss_after_gradient"] : Float64[]
            rss_after_optimization = haskey(run_timings, "rss_after_optimization") ? run_timings["rss_after_optimization"] : Float64[]
            rss_per_iteration = haskey(run_timings, "rss_per_iteration") ? run_timings["rss_per_iteration"] : Float64[]
            
            run_data = Dict(
                "backend" => string(backend),
                "threads" => threads,
                "n_threads" => Base.Threads.nthreads(),
                "threads_oppool" => threads_oppool,
                "calc_grads" => string(calc_grads),
                "max_weight" => current_mw,
                "min_abs_coeff" => current_mac,
                "max_freq" => current_max_freq,
                "min_iters_for_stagnation" => current_min_iters_stag,
                "tile_nq" => tile_nq,
                "scaled_nq" => total_nq,
                "refresh_grad_tape" => refresh_grad_tape,
                "conv_tol" => conv_tol,
                "stagnation_layers" => stagnation_layers,
                "stagnation_tol" => stagnation_tol,
                "energy_curve" => run_energies,
                "max_grads" => run_grads,
                "final_energy_per_qubit" => isempty(run_energies) ? NaN : run_energies[end],
                "overlap_func" => string(overlap_func),
                "hamiltonian" => hamiltonian,
                "topology" => topology,
                "hamiltonian_kwargs" => Dict(pairs(hamiltonian_kwargs)),
                "calc_grad_kwargs" => isnothing(calc_grad_kwargs) ? nothing : Dict(pairs(calc_grad_kwargs)),
                # Per-iteration memory arrays (in bytes for allocations, MB for RSS)
                "gradient_memory" => gradient_memory,
                "commutator_memory" => commutator_memory,
                "optimization_memory" => optimization_memory,
                "rss_after_commutator" => rss_after_commutator,
                "rss_after_gradient" => rss_after_gradient,
                "rss_after_optimization" => rss_after_optimization,
                "rss_per_iteration" => rss_per_iteration,
                "converged" => length(run_energies) < max_iters,
                "timestamp" => Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
            )
            
            merge!(run_data, run_timings)
            loop_run_data[run_idx] = run_data
        end
        
        # Add overall loop summary
        loop_run_data["loop_summary"] = Dict(
            "num_runs" => num_runs,
            "n_threads" => Base.Threads.nthreads(),
            "total_elapsed_time_s" => num_reruns > 1 ? total_elapsed_time : elapsed_time,
            "total_gc_time_s" => num_reruns > 1 ? total_gc_time : gc_time,
            "total_gc_time_ratio" => gc_time_ratio,
            "total_memory_bytes" => num_reruns > 1 ? total_memory_bytes : memory_bytes,
            "total_memory_gb" => (num_reruns > 1 ? total_memory_bytes : memory_bytes) / 1e9,
            "best_rerun_elapsed_time_s" => elapsed_time,
            "best_rerun_gc_time_s" => gc_time,
            "best_rerun_memory_bytes" => memory_bytes,
            "peak_rss_mb" => memory_after_mb,
            "rss_increase_mb" => memory_increase_mb,
            "total_allocations" => total_allocations,
            #"circuit_bit_generators" => collect(UInt128, chosen_ops),
            "circuit" => circuit,
            "optimized_thetas" => thetas,
            "final_energy_per_qubit" => energy_curve[end],
            "final_circuit_depth" => length(circuit),
            "vscore" => vscore ? final_vscore : nothing,
            "combined_energy_curve" => energy_curve,
            "combined_max_grads" => max_grads,
            "num_reruns" => num_reruns,
            "best_rerun_idx" => best_rerun_idx,
            "all_rerun_final_energies" => all_rerun_final_energies,
            "all_rerun_elapsed_times" => all_rerun_elapsed_times,
            "all_rerun_gc_times" => all_rerun_gc_times,
            "all_rerun_memory_bytes" => all_rerun_memory_bytes,
            "timestamp" => Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
        )
        
        run_data = loop_run_data
    else
        # Single run - create nested structure with loop_summary for consistency
        current_mw = mw_sequence[1]
        current_min_iters_stag = min_iters_for_stagnation_sequence[1]
        
        if backend == :reversediff
            current_max_freq = max_freq_sequence[1]
            current_mac = 0.0
        else
            current_mac = mac_sequence[1]
            current_max_freq = Inf
        end
        
        # Extract per-iteration memory arrays
        gradient_memory = haskey(timings, "gradient_memory") ? timings["gradient_memory"] : Float64[]
        commutator_memory = haskey(timings, "commutator_memory") ? timings["commutator_memory"] : Float64[]
        optimization_memory = haskey(timings, "optimization_memory") ? timings["optimization_memory"] : Float64[]
        
        # Extract per-iteration RSS arrays
        rss_after_commutator = haskey(timings, "rss_after_commutator") ? timings["rss_after_commutator"] : Float64[]
        rss_after_gradient = haskey(timings, "rss_after_gradient") ? timings["rss_after_gradient"] : Float64[]
        rss_after_optimization = haskey(timings, "rss_after_optimization") ? timings["rss_after_optimization"] : Float64[]
        rss_per_iteration = haskey(timings, "rss_per_iteration") ? timings["rss_per_iteration"] : Float64[]
        
        # Create nested structure like multi-run case
        loop_run_data = Dict{Any, Dict{String, Any}}()
        
        # Single run data (indexed by 1)
        loop_run_data[1] = Dict(
            "backend" => string(backend),
            "threads" => threads,
            "n_threads" => Base.Threads.nthreads(),
            "threads_oppool" => threads_oppool,
            "calc_grads" => string(calc_grads),
            "max_weight" => current_mw,
            "min_abs_coeff" => current_mac,
            "max_freq" => current_max_freq,
            "min_iters_for_stagnation" => current_min_iters_stag,
            "tile_nq" => tile_nq,
            "scaled_nq" => total_nq,
            "refresh_grad_tape" => refresh_grad_tape,
            "conv_tol" => conv_tol,
            "stagnation_layers" => stagnation_layers,
            "stagnation_tol" => stagnation_tol,
            "overlap_func" => string(overlap_func),
            "hamiltonian" => hamiltonian,
            "topology" => topology,
            "hamiltonian_kwargs" => Dict(pairs(hamiltonian_kwargs)),
            "calc_grad_kwargs" => isnothing(calc_grad_kwargs) ? nothing : Dict(pairs(calc_grad_kwargs)),
            # Per-iteration memory arrays (in bytes for allocations, MB for RSS)
            "gradient_memory" => gradient_memory,
            "commutator_memory" => commutator_memory,
            "optimization_memory" => optimization_memory,
            "rss_after_commutator" => rss_after_commutator,
            "rss_after_gradient" => rss_after_gradient,
            "rss_after_optimization" => rss_after_optimization,
            "rss_per_iteration" => rss_per_iteration,
            "converged" => length(energy_curve) < max_iters,
            "timestamp" => Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
        )
        merge!(loop_run_data[1], timings)
        
        # Add loop_summary for consistency with multi-run case
        loop_run_data["loop_summary"] = Dict(
            "num_runs" => 1,
            "n_threads" => Base.Threads.nthreads(),
            "total_elapsed_time_s" => num_reruns > 1 ? total_elapsed_time : elapsed_time,
            "total_gc_time_s" => num_reruns > 1 ? total_gc_time : gc_time,
            "total_gc_time_ratio" => gc_time_ratio,
            "total_memory_bytes" => num_reruns > 1 ? total_memory_bytes : memory_bytes,
            "total_memory_gb" => (num_reruns > 1 ? total_memory_bytes : memory_bytes) / 1e9,
            "best_rerun_elapsed_time_s" => elapsed_time,
            "best_rerun_gc_time_s" => gc_time,
            "best_rerun_memory_bytes" => memory_bytes,
            "peak_rss_mb" => memory_after_mb,
            "rss_increase_mb" => memory_increase_mb,
            "total_allocations" => total_allocations,
            "circuit" => circuit,
            "optimized_thetas" => thetas,
            "final_energy_per_qubit" => energy_curve[end],
            "final_circuit_depth" => length(circuit),
            "vscore" => vscore ? final_vscore : nothing,
            "combined_energy_curve" => energy_curve,
            "combined_max_grads" => max_grads,
            "num_reruns" => num_reruns,
            "best_rerun_idx" => best_rerun_idx,
            "all_rerun_final_energies" => all_rerun_final_energies,
            "all_rerun_elapsed_times" => all_rerun_elapsed_times,
            "all_rerun_gc_times" => all_rerun_gc_times,
            "all_rerun_memory_bytes" => all_rerun_memory_bytes,
            "timestamp" => Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
        )
        
        run_data = loop_run_data
    end
    
    return circuit, thetas, run_data, chosen_ops
end

# ============================================================================
# LOOP SCALING BENCHMARKS - 2D SYSTEMS (AND 1D BACKWARD COMPATIBLE)
# ============================================================================

"""
    run_loop_scaling_benchmarks_2d(;
        hamiltonian::SystemHamiltonian,
        topology::SystemTopology,
        tile_total_nq=4,
        scaled_total_nq=9,
        mw_sequences=[[4,5,6], [3,4,5]],
        mac_sequences=[[1e-4,1e-5,1e-6], [1e-3,1e-4,1e-5]],
        max_freq_sequences=[[50,100,150]],
        min_iters_for_stagnation_sequences=[[30,20,10]],
        ...)

Run systematic benchmarks using ADAPT-VQE loops with progressive truncation for 2D systems.
Backward compatible with 1D systems (chain, chain_pbc).

This function extends `run_loop_scaling_benchmarks` to support 2D topologies using 
`scaled_pool_generation_2d` and `run_single_loop_benchmark_2d`. It automatically detects
whether the system is 1D or 2D based on the topology type and configures the appropriate
topology and pool generation method.

# Key Features:
- Auto-detection of 1D vs 2D based on topology type
- Uses `scaled_pool_generation_2d` with appropriate tiles configuration for 2D
- Uses `run_single_loop_benchmark_2d` which works for both 1D and 2D
- Backward compatible with run_loop_scaling_benchmarks for 1D systems

# Topology Support:
- 1D: "chain" (OBC), "chain_pbc" (PBC) → uses bricklayertopology
- 2D: "square_obc" (square lattice with OBC) → uses rectangletopology

# Arguments
- `hamiltonian`: SystemHamiltonian instance (e.g., SystemHamiltonian(:Heisenberg))
- `topology`: SystemTopology instance (e.g., SystemTopology(:chain), SystemTopology(:square_obc))
- `tile_total_nq`: Total number of qubits in tile (e.g., 4 for 2x2 square)
- `scaled_total_nq`: Total number of qubits in scaled system (e.g., 9 for 3x3 square)
- `mw_sequences`: Array of max_weight sequences, e.g. [[4,5,6], [3,4,5]]
- `mac_sequences`: Array of min_abs_coeff sequences (Mooncake/ForwardDiff), e.g. [[1e-4,1e-5,1e-6], [1e-3,1e-4,1e-5]]
- `max_freq_sequences`: Array of max_freq sequences (ReverseDiff), e.g. [[50,100,150]]
- `min_iters_for_stagnation_sequences`: Array of min_iters_for_stagnation sequences, e.g. [[30,20,10]]
- `refresh_grad_tape`: Frequency of gradient tape refresh
- `max_iters`: Maximum ADAPT iterations per sub-run
- `conv_tol`: Convergence tolerance
- `stagnation_layers`: Number of layers to check for gradient stagnation
- `stagnation_tol`: Tolerance for gradient stagnation detection
- `backend`: `:mooncake` (default), `:reversediff`, or `:forwarddiff`
- `threads`: Use threading (default: true)
- `calc_grads`: Gradient calculation method (`:phys` or `:fd`)
- `threads_oppool`: Use ThreadsX parallelization for operator pool gradients
- `calc_grad_kwargs`: Named tuple to override truncation for gradient calculations
- `output_dir`: Directory for output files
- `verbose`: Print detailed progress
- `vscore`: Calculate and log v-score
- `overlap_func`: Function to compute overlap with initial state
- `hamiltonian_constructor`: Function to construct Hamiltonian (auto-set if not provided)
- `hamiltonian_kwargs`: Kwargs to pass to hamiltonian constructor
- `overlap_kwargs`: Kwargs to pass to overlap function

# Returns
3-layer nested dictionary (same structure as run_loop_scaling_benchmarks)

# Examples
```julia
# 2D square lattice: 2x2 tile → 3x3 scaled
results = run_loop_scaling_benchmarks_2d(
    hamiltonian=SystemHamiltonian(:Heisenberg),
    topology=SystemTopology(:square_obc),
    tile_total_nq=4,
    scaled_total_nq=9,
    mw_sequences=[[3,4,5]],
    mac_sequences=[[1e-3,1e-4,1e-5]],
    min_iters_for_stagnation_sequences=[[20,15,10]]
)

# 1D chain (backward compatible)
results = run_loop_scaling_benchmarks_2d(
    hamiltonian=SystemHamiltonian(:Heisenberg),
    topology=SystemTopology(:chain),
    tile_total_nq=3,
    scaled_total_nq=6,
    mw_sequences=[[3,4,5]],
    mac_sequences=[[1e-3,1e-4,1e-5]]
)
```
"""
function run_loop_scaling_benchmarks_2d(;
    hamiltonian::SystemHamiltonian,  # Type-based Hamiltonian (e.g., SystemHamiltonian(:Heisenberg))
    topology::SystemTopology,  # Type-based Topology (e.g., SystemTopology(:chain))
    tile_total_nq::Int=4,
    scaled_total_nq::Int=9,
    mw_sequences::Vector{Vector{Int}}=[[4, 5, 6]],
    mac_sequences::Vector{Vector{Float64}}=[[1e-4, 1e-5, 1e-6]],
    max_freq_sequences::Vector{Vector{Float64}}=[[50.0, 100.0, 150.0]],
    min_iters_for_stagnation_sequences::Vector{Vector{Int}}=[[30, 20, 10]],
    refresh_grad_tape::Int=300,
    max_iters::Int=40,
    conv_tol::Float64=1e-2,
    stagnation_layers::Int=5,
    stagnation_tol::Float64=0.1,
    backend::Symbol=:mooncake,
    threads::Bool=true,
    calc_grads::Symbol=:phys,
    threads_oppool::Bool=false,
    calc_grad_kwargs=nothing,
    output_dir::String="benchmark_results_loop_2d",
    verbose::Bool=false,
    vscore::Bool=false,
    num_reruns::Int=1,  # Number of reruns for the same optimization (default=1 for backward compatibility)
    # Hamiltonian-specific parameters
    overlap_func=overlapwithneel,
    hamiltonian_kwargs=NamedTuple(),
    overlap_kwargs=(up_on_odd=true,),
    # Initial circuit parameters for warm starting
    initial_circuit=nothing,
    initial_thetas=nothing)
    
    # Create output directory
    mkpath(output_dir)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    # Get default kwargs for the Hamiltonian type and merge with user-provided
    default_kwargs = get_default_kwargs(hamiltonian)
    merged_hamiltonian_kwargs = merge(default_kwargs, hamiltonian_kwargs)
    
    # Get hamiltonian constructor for potential ReverseDiff usage
    hamiltonian_constructor = get_hamiltonian_constructor(hamiltonian)
    
    println("\n" * "="^80)
    println("ADAPT-VQE Loop Scaling Benchmarks 2D (Progressive Truncation)")
    println("="^80)
    println("Hamiltonian: $hamiltonian with parameters $merged_hamiltonian_kwargs")
    println("Topology: $topology")
    
    # Detect dimensionality using type-based dispatch
    is_2d_bool = is_2d(topology)
    
    if is_2d_bool
        # Infer dimensions from total qubits for 2D
        tile_width = get_lattice_width(topology, tile_total_nq)
        scaled_lattice_width = get_lattice_width(topology, scaled_total_nq)
        
        println("Tile: $(tile_width)x$(tile_width) = $tile_total_nq qubits")
        println("Scaled: $(scaled_lattice_width)x$(scaled_lattice_width) = $scaled_total_nq qubits")
    else
        # For 1D, width equals the number of qubits (chain length)
        tile_width = tile_total_nq
        scaled_lattice_width = scaled_total_nq
        
        println("Tile qubits: $tile_total_nq → Scaled qubits: $scaled_total_nq")
    end
    
    println("Number of sequence sets: $(length(mw_sequences))")
    println("  MW sequences: $mw_sequences")
    if backend == :reversediff
        println("  Max freq sequences: $max_freq_sequences")
    else
        println("  MAC sequences: $mac_sequences")
    end
    println("  Min iters for stagnation sequences: $min_iters_for_stagnation_sequences")
    println("Backend: $backend")
    println("Threading: $threads")
    println("Refresh grad tape: $refresh_grad_tape")
    println("Number of reruns per optimization: $num_reruns")
    println("Output directory: $output_dir")
    println("="^80 * "\n")
    
    # Validate sequence lengths
    num_sequence_sets = length(mw_sequences)
    if backend == :reversediff
        if length(max_freq_sequences) != num_sequence_sets
            throw(ArgumentError("max_freq_sequences length must match mw_sequences length"))
        end
    else
        if length(mac_sequences) != num_sequence_sets
            throw(ArgumentError("mac_sequences length must match mw_sequences length"))
        end
    end
    if length(min_iters_for_stagnation_sequences) != num_sequence_sets
        throw(ArgumentError("min_iters_for_stagnation_sequences length must match mw_sequences length"))
    end
    
    # ========================================================================
    # SETUP: Generate Hamiltonians and Pools
    # ========================================================================
    
    println("Setting up Hamiltonians and operator pools...")
    
    # Construct topologies using type-based dispatch
    tile_topology = get_topology(topology, tile_total_nq)
    scaled_topology = get_topology(topology, scaled_total_nq)
    
    # Get tiles for pool generation (2D uses tiles, 1D uses sliding window)
    tiles = get_tiles(topology, tile_width, scaled_lattice_width)
    
    if is_2d_bool
        println("  2D topology detected: $(topology)")
        println("  Tile topology: $(tile_width)x$(tile_width) sites")
        println("  Scaled topology: $(scaled_lattice_width)x$(scaled_lattice_width) sites")
        println("  Number of tiles: $(length(tiles))")
    else
        println("  1D topology detected: $(topology)")
    end
    
    # Create Hamiltonians using type-based dispatch
    println("  Constructing Hamiltonians using type-based dispatch")
    
    # Special handling for J1J2_2d_obc which needs lattice_width
    if hamiltonian isa SystemHamiltonian{:J1J2_2d_obc}
        if !is_2d_bool
            error("J1J2_2d_obc Hamiltonian requires 2D topology (square_obc)")
        end
        tile_hamiltonian = get_hamiltonian(hamiltonian, tile_total_nq, tile_topology, tile_width; merged_hamiltonian_kwargs...)
        scaled_hamiltonian = get_hamiltonian(hamiltonian, scaled_total_nq, scaled_topology, scaled_lattice_width; merged_hamiltonian_kwargs...)
    else
        tile_hamiltonian = get_hamiltonian(hamiltonian, tile_total_nq, tile_topology; merged_hamiltonian_kwargs...)
        scaled_hamiltonian = get_hamiltonian(hamiltonian, scaled_total_nq, scaled_topology; merged_hamiltonian_kwargs...)
    end
    
    tile_full_bit_pool = generate_full_bit_pool(tile_total_nq)
    
    # Generate scaled pool using generalized 2D function
    println("  Generating scaled operator pool...")
    
    if is_2d_bool
        println("  Using tiles-based generation for 2D")
        scaled_pool = scaled_pool_generation_2d(tile_total_nq, scaled_total_nq,
                                                tile_hamiltonian, tile_full_bit_pool;
                                                backend=:mooncake,
                                                topology=tile_topology,
                                                tiles=tiles,
                                                num_runs=15,
                                                run_iters=10,
                                                conv_tol=1e-2,
                                                verbose=false,
                                                overlap_func=overlap_func,
                                                overlap_kwargs...)
    else
        println("  Using 1D sliding window generation")
        scaled_pool = scaled_pool_generation_2d(tile_total_nq, scaled_total_nq,
                                                tile_hamiltonian, tile_full_bit_pool;
                                                backend=:mooncake,
                                                topology=tile_topology,
                                                tiles=nothing,  # 1D mode
                                                num_runs=15,
                                                run_iters=10,
                                                conv_tol=1e-2,
                                                verbose=false,
                                                overlap_func=overlap_func,
                                                overlap_kwargs...)
    end
    
    println("  ✓ Scaled pool size: $(length(scaled_pool))")
    
    # ========================================================================
    # BENCHMARK LOOP - 3 LAYER NESTED STRUCTURE
    # ========================================================================
    
    # Layer 1: Dict indexed by sequence set number
    # Always maintains nested structure with loop_summary for consistency
    all_results = Dict{Any, Dict{Any, Dict{String, Any}}}()
    
    # Define master log path
    #topology_sym = typeof(topology).parameters[1] (but this woudl oynl work for predefined topos)
    master_log = joinpath(output_dir,
        "master_log_loop_$(hamiltonian)_$(topology)_$(timestamp)_tile$(tile_total_nq)_scaled$(scaled_total_nq).jld2")
    
    for seq_idx in 1:num_sequence_sets
        current_mw_seq = mw_sequences[seq_idx]
        current_min_iters_stag_seq = min_iters_for_stagnation_sequences[seq_idx]
        
        if backend == :reversediff
            current_max_freq_seq = max_freq_sequences[seq_idx]
            current_mac_seq = [0.0]  # Not used
            seq_desc = "mw=$current_mw_seq, max_freq=$current_max_freq_seq"
        else
            current_mac_seq = mac_sequences[seq_idx]
            current_max_freq_seq = [Inf]  # Not used
            seq_desc = "mw=$current_mw_seq, mac=$current_mac_seq"
        end
        
        println("\n" * "-"^80)
        println("Sequence Set $seq_idx/$num_sequence_sets:")
        println("  $seq_desc")
        println("  min_iters_for_stagnation=$current_min_iters_stag_seq")
        println("-"^80)
        
        # Run loop benchmark using 2D version
        circuit, thetas, loop_run_data, chosen_ops = run_single_loop_benchmark_2d(
            scaled_total_nq,
            scaled_hamiltonian, scaled_pool, scaled_topology,
            overlap_func, hamiltonian_constructor;
            hamiltonian_kwargs=hamiltonian_kwargs,
            mw_sequence=current_mw_seq,
            mac_sequence=current_mac_seq,
            max_freq_sequence=current_max_freq_seq,
            min_iters_for_stagnation_sequence=current_min_iters_stag_seq,
            tile_nq=tile_total_nq,  # Pass for compatibility
            refresh_grad_tape=refresh_grad_tape,
            max_iters=max_iters,
            conv_tol=conv_tol,
            stagnation_layers=stagnation_layers,
            stagnation_tol=stagnation_tol,
            backend=backend,
            threads=threads,
            calc_grads=calc_grads,
            threads_oppool=threads_oppool,
            calc_grad_kwargs=calc_grad_kwargs,
            verbose=verbose,
            vscore=vscore,
            num_reruns=num_reruns,
            initial_circuit=initial_circuit,
            initial_thetas=initial_thetas,
            overlap_kwargs...
        )
        
        # Store in layer 1 (indexed by sequence set)
        all_results[seq_idx] = loop_run_data
        
        # Add metadata about this sequence set
        all_results[seq_idx]["sequence_metadata"] = Dict(
            "sequence_set_index" => seq_idx,
            "mw_sequence" => current_mw_seq,
            "mac_sequence" => current_mac_seq,
            "max_freq_sequence" => current_max_freq_seq,
            "min_iters_for_stagnation_sequence" => current_min_iters_stag_seq,
            "timestamp" => Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
        )
        
        # Force garbage collection between sequence sets
        GC.gc()
    end
    
    # Save master log with all results
    println("\n" * "="^80)
    println("Saving master log...")
    println("="^80)
    
    @save master_log all_results
    println("  ✓ Master log saved to: $master_log")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    println("\n" * "="^80)
    println("LOOP BENCHMARK 2D SUMMARY")
    println("="^80)
    println("Total sequence sets completed: $num_sequence_sets")
    println("Backend: $backend, Threading: $threads")
    println("\nResults by sequence set:")
    println("-"^80)
    
    if backend == :reversediff
        println("Set | MW Seq      | MaxFreq Seq       | Final E/Q     | Depth | Total Time (s)")
        println("-"^80)
        
        for seq_idx in 1:num_sequence_sets
            seq_data = all_results[seq_idx]
            summary = seq_data["loop_summary"]
            metadata = seq_data["sequence_metadata"]
            
            s = Printf.@sprintf("%-3d | %-11s | %-17s | %-13.6f | %-5d | %.2f",
                                seq_idx,
                                string(metadata["mw_sequence"]),
                                string(metadata["max_freq_sequence"]),
                                summary["final_energy_per_qubit"],
                                summary["final_circuit_depth"],
                                summary["total_elapsed_time_s"])
            println(s)
        end
    else
        println("Set | MW Seq      | MAC Seq           | Final E/Q     | Depth | Total Time (s)")
        println("-"^80)
        
        for seq_idx in 1:num_sequence_sets
            seq_data = all_results[seq_idx]
            summary = seq_data["loop_summary"]
            metadata = seq_data["sequence_metadata"]
            
            s = Printf.@sprintf("%-3d | %-11s | %-17s | %-13.6f | %-5d | %.2f",
                                seq_idx,
                                string(metadata["mw_sequence"]),
                                string(metadata["mac_sequence"]),
                                summary["final_energy_per_qubit"],
                                summary["final_circuit_depth"],
                                summary["total_elapsed_time_s"])
            println(s)
        end
    end
    println("-"^80)
    
    println("\n" * "="^80)
    println("All results saved to: $output_dir")
    println("Master log: $(basename(master_log))")
    println("="^80 * "\n")
    
    return all_results
end


