using PauliPropagation
using Random
using Distributions: Uniform
using Dates
using JLD2
using BenchmarkTools
using Printf
# using Plots  # Not needed for cluster runs

# Import the ADAPT-VQE implementation
include("../../adapt_modular.jl")
# Import the general execution script
include("../../adapt_benchmarks.jl")

function loop_j1j2_benchmarks(tile_total_nq, scaled_total_nq; use_initial_circuit=false)
    # Example: Create an initial circuit with an XX gate on qubits 1 and 2
    initial_circuit = nothing
    initial_thetas = nothing
    
    #placeholder for DQA tests
    if use_initial_circuit
        println("Creating initial circuit with XX gate on qubits 1 and 2")
        xx_gate = PauliRotation([:X, :X], [1, 2])
        initial_circuit = Any[xx_gate]  # Use Any[] to match ADAPT's circuit type! (otherwise we get nested PauliRotations)
        initial_thetas = [π/4]  # Example angle
    end

    exact_grads = (max_weight=Inf, min_abs_coeff=0.0)
    approx_grads = (max_weight=Inf,)

    # Create SystemHamiltonian and SystemTopology instances - type-based interface
    hamiltonian = SystemHamiltonian(:J1J2_1d)
    topology = SystemTopology(:chain)
    backend = :forwarddiff
    calc_grad_kwargs = approx_grads

    results = run_loop_scaling_benchmarks_2d(
        hamiltonian=hamiltonian,  # Type-based Hamiltonian
        topology=topology,  # Type-based Topology
        tile_total_nq=tile_total_nq,
        scaled_total_nq=scaled_total_nq,
        mw_sequences=[[4]],# [5], [5]], #[[4,5],[4,5]],#[[3,3],[3,4]],
        mac_sequences=[[1e-4]],#,[1e-4],[1e-4],[1e-5]], #[[1e-3,1e-4],[1e-4,1e-5]],#[[1e-2,1e-3],[1e-3,1e-4]],
        #max_freq_sequences=[[200.0]],
        min_iters_for_stagnation_sequences=[[80]],#,[40],[40],[40]], #[[40,20],[40,20]],#[[40,20],[40,20]],
        max_iters=200, # avoid ooM by fixing max_iters
        conv_tol=1e-2, # not relevant for non-unitary evolution
        stagnation_layers=20,  # Check last 10 layers
        stagnation_tol=0.05,  # Detect if gradient changes less than 0.05
        backend=backend,  # no tape!
        threads=false, # main time and memory contribution
        calc_grads=:phys, 
        threads_oppool=false, # too big overhead for small systems
        calc_grad_kwargs=calc_grad_kwargs,  # approx gradients
        output_dir="PP-GSP/cluster_data/Benchmark_$(hamiltonian)_$(backend)_grad_$(calc_grad_kwargs)_tile_nq_$(tile_total_nq)_rerun",
        verbose=false, 
        vscore=false, # careful with vscore and large systems
        overlap_func=overlapwithneel,
        hamiltonian_kwargs=(J1=1.0, J2=0.5),
        overlap_kwargs=(up_on_odd=true,),
        # Pass the initial circuit and thetas
        initial_circuit=initial_circuit,
        initial_thetas=initial_thetas,
        num_reruns = 2
    )
    return results
end

tile_total_nq = 2

#threading comparison runs
# loop_j1j2_benchmarks(tile_total_nq,40; use_initial_circuit=true)
# loop_j1j2_benchmarks(tile_total_nq,50; use_initial_circuit=false)
# loop_j1j2_benchmarks(tile_total_nq,80; use_initial_circuit=false)
# loop_j1j2_benchmarks(tile_total_nq,90; use_initial_circuit=false)
loop_j1j2_benchmarks(tile_total_nq,100; use_initial_circuit=false)

# # ED regime
# for i in 4:15
#     loop_j1j2_benchmarks(tile_total_nq, i)
# end

#NQS regime
# nqubits = [20, 25, 30, 35, 40, 45, 50]
# for i in nqubits
#     loop_j1j2_benchmarks(tile_total_nq, i)
# end

# #NQS regime extended
# nqubits = [60,70]#,80,90,100]
# for i in nqubits
#     loop_j1j2_benchmarks(tile_total_nq, i)
# end 
