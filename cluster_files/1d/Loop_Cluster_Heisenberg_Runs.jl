using PauliPropagation
using Random
using Distributions: Uniform
using Dates
using JLD2
using BenchmarkTools
using Printf

# Import the ADAPT-VQE implementation
include("../../adapt_modular.jl")
# Import the general execution script
include("../../adapt_benchmarks.jl")

function loop_heisenberg_benchmarks(tile_total_nq, scaled_total_nq; use_initial_circuit=false)
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
    hamiltonian = SystemHamiltonian(:Heisenberg)
    topology = SystemTopology(:chain)
    backend = :forwarddiff
    calc_grad_kwargs = approx_grads

    results = run_loop_scaling_benchmarks_2d(
        hamiltonian=hamiltonian,  # Type-based Hamiltonian
        topology=topology,  # Type-based Topology
        tile_total_nq=tile_total_nq,
        scaled_total_nq=scaled_total_nq,
        mw_sequences=[[4]],
        mac_sequences=[[1e-4]],
        min_iters_for_stagnation_sequences=[[40]],
        max_iters= 150, # avoid ooM by fixing max_iters
        conv_tol=1e-2, # not relevant for non-unitaty evolution
        stagnation_layers=15,  # Check last 10 layers
        stagnation_tol=0.05,  # Detect if gradient changes less than 0.05
        backend=backend,  # no tape!
        threads=true,
        calc_grads=:phys, 
        threads_oppool=false,
        calc_grad_kwargs=calc_grad_kwargs,  # approx gradients
        output_dir="PP-GSP/Benchmark_$(hamiltonian)_$(backend)_grad_$(calc_grad_kwargs)_tile_nq_$(tile_total_nq)_rerun",
        verbose=false,
        vscore=false, # careful with vscore and large systems
        overlap_func=overlapwithneel,
        hamiltonian_kwargs=(J=1.0,),
        overlap_kwargs=(up_on_odd=true,),
        # Pass the initial circuit and thetas
        initial_circuit=initial_circuit,
        initial_thetas=initial_thetas,
        num_reruns = 5
    )
    return results
end

tile_total_nq = 2
# loop_heisenberg_benchmarks(tile_total_nq,20; use_initial_circuit=false)

# # ED regime
# for i in 4:15
#     loop_heisenberg_benchmarks(3, i)
# end

# NQS regime
nqubits = [20, 25, 30, 35, 40, 45, 50]
for i in nqubits
    loop_heisenberg_benchmarks(tile_total_nq, i)
end

#NQS regime extended
nqubits = [60,70,80,90,100]
for i in nqubits
    loop_heisenberg_benchmarks(tile_total_nq, i)
end 