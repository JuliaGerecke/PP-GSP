using PauliPropagation
using Random
using Distributions: Uniform
using Dates
using JLD2
using BenchmarkTools
using Printf
#using Plots

# Import the ADAPT-VQE implementation
include("../../adapt_modular.jl")
# Import the general execution script
include("../../adapt_benchmarks.jl")

"""
Run loop scaling benchmarks for 2D TFIM model on square lattice with OBC.

# Arguments
- `tile_width`: Width of the tile (e.g., 2 for 2x2 tile = 4 qubits)
- `scaled_lattice_width`: Width of the scaled lattice (e.g., 3 for 3x3 = 9 qubits)
"""
function loop_tfim_2d_benchmarks(tile_width, scaled_lattice_width; use_initial_circuit=false)

    initial_circuit = nothing
    initial_thetas = nothing
    
    if use_initial_circuit
        println("Creating initial circuit with XX gate on qubits 1 and 2")
        # Create a simple XX gate using PauliRotation
        # XX corresponds to Pauli X on qubits 1 and 2
        xx_gate = PauliRotation([:X, :X], [1, 2])
        initial_circuit = [xx_gate]
        initial_thetas = [π/4]  # Example angle
    end

    tile_total_nq = tile_width^2
    scaled_total_nq = scaled_lattice_width^2
    
    exact_grads = (max_weight=Inf, min_abs_coeff=0.0)
    approx_grads = (max_weight=Inf,)

    # Create SystemHamiltonian instance - this replaces the old string-based interface
    hamiltonian = SystemHamiltonian(:TFIM)
    topology = SystemTopology(:square_obc)
    backend = :forwarddiff
    calc_grad_kwargs = approx_grads

    results = run_loop_scaling_benchmarks_2d(
        hamiltonian=hamiltonian, 
        topology= topology,
        tile_total_nq=tile_total_nq,
        scaled_total_nq=scaled_total_nq,
        mw_sequences=[[4]],
        mac_sequences=[[1e-4]],
        min_iters_for_stagnation_sequences=[[40]],
        refresh_grad_tape=300,
        max_iters=150,
        conv_tol=1e-2,
        stagnation_layers=15,
        stagnation_tol=0.05,
        backend=backend,
        threads=false,
        calc_grads=:phys,
        threads_oppool=false,
        calc_grad_kwargs=calc_grad_kwargs,
        output_dir="PP-GSP/cluster_data/Benchmark_2D_$(hamiltonian)_$(backend)_grad_$(calc_grad_kwargs)_stag",
        verbose=false,
        vscore=false,
        overlap_func=overlapwithplus,
        hamiltonian_kwargs=(J=1.0, h=1.0),
        overlap_kwargs=NamedTuple(),
        # Pass the initial circuit and thetas
        initial_circuit=initial_circuit,
        initial_thetas=initial_thetas
    )
    return results
end

# Example runs for different system sizes
scaled_width = [3,4,5,6]
# Small systems (ED regime)
for i in scaled_width
    loop_tfim_2d_benchmarks(2, i; use_initial_circuit=false)  # 2x2 → ixi (4 → i^2 qubits)
end
