"""
System Hamiltonian Type-Based Dispatch
=======================================

This file implements a type-based dispatch system for Hamiltonian construction,
following the Strategy Pattern. This allows for extensible, type-safe Hamiltonian
definitions without modifying core code.

# Design Pattern

```julia
# Define a new Hamiltonian type:
ham = SystemHamiltonian(:MyModel)

# Get default parameters:
defaults = get_default_kwargs(ham)

# Construct Hamiltonian:
H = get_hamiltonian(ham, nq, topology; custom_param=value)

# For ReverseDiff (with coefficient type):
H = get_hamiltonian(ham, Float64, nq, topology; custom_param=value)
```

# Adding New Hamiltonians

To add a new Hamiltonian, simply define new methods (no need to modify existing code):

```julia
get_default_kwargs(::SystemHamiltonian{:MyModel}) = (α=1.0, β=2.0)

function get_hamiltonian(::SystemHamiltonian{:MyModel}, nq, topology; kwargs...)
    merged_kwargs = merge(get_default_kwargs(SystemHamiltonian(:MyModel)), kwargs)
    # ... construction logic
    return hamiltonian
end
```
"""

using PauliPropagation

# ============================================================================
# Core Type Definition
# ============================================================================

"""
    SystemHamiltonian{name}

Type-parameterized struct for Hamiltonian dispatch.

# Example
```julia
heisenberg = SystemHamiltonian(:Heisenberg)
tfim = SystemHamiltonian(:TFIM)
```
"""
struct SystemHamiltonian{name} end
SystemHamiltonian(name::Symbol) = SystemHamiltonian{name}()

# ============================================================================
# Default Parameters
# ============================================================================

"""
    get_default_kwargs(ham::SystemHamiltonian)

Return default keyword arguments for a given Hamiltonian type.
"""
get_default_kwargs(::SystemHamiltonian{:Heisenberg}) = (J=1.0,)
get_default_kwargs(::SystemHamiltonian{:TFIM}) = (J=1.0, h=1.0)
get_default_kwargs(::SystemHamiltonian{:tV}) = (t=1.0, V=1.0)
get_default_kwargs(::SystemHamiltonian{:J1J2_1d}) = (J1=1.0, J2=0.5)
get_default_kwargs(::SystemHamiltonian{:J1J2_2d_obc}) = (J1=1.0, J2=0.5)
get_default_kwargs(::SystemHamiltonian{:Hubbard}) = (t=1.0, U=4.0)

# ============================================================================
# Hamiltonian Construction (Float64 - for Mooncake/ForwardDiff)
# ============================================================================

"""
    get_hamiltonian(ham::SystemHamiltonian, nq, topology; kwargs...)

Construct a Hamiltonian with Float64 coefficients (for Mooncake/ForwardDiff backends).

# Arguments
- `ham`: SystemHamiltonian type instance
- `nq`: Number of qubits
- `topology`: Pre-constructed topology (e.g., from bricklayertopology or rectangletopology)
- `kwargs...`: Override default parameters

# Returns
- `PauliSum`: The constructed Hamiltonian
"""
function get_hamiltonian(ham::SystemHamiltonian{:Heisenberg}, nq::Int, topology; kwargs...)
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    return heisenberg_hamiltonian(nq, topology; merged_kwargs...)
end

function get_hamiltonian(ham::SystemHamiltonian{:TFIM}, nq::Int, topology; kwargs...)
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    return tfim_hamiltonian(nq, topology; merged_kwargs...)
end

function get_hamiltonian(ham::SystemHamiltonian{:tV}, nq::Int, topology; kwargs...)
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    return tv_hamiltonian(nq, topology; merged_kwargs...)
end

function get_hamiltonian(ham::SystemHamiltonian{:J1J2_1d}, nq::Int, topology; kwargs...)
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    return j1j2_1d_hamiltonian(nq, topology; merged_kwargs...)
end

function get_hamiltonian(ham::SystemHamiltonian{:J1J2_2d_obc}, nq::Int, topology, lattice_width::Int; kwargs...)
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    return j1j2_2d_square_obc_hamiltonian(nq, topology, lattice_width; merged_kwargs...)
end

function get_hamiltonian(ham::SystemHamiltonian{:Hubbard}, nq::Int, topology; kwargs...)
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    Nsites = div(nq, 2)  # Hubbard uses 2 qubits per site (spin up/down)
    return fermi_hubbard_hamiltonian(Nsites, topology; merged_kwargs...)
end

# ============================================================================
# Hamiltonian Construction (Parameterized CT - for ReverseDiff)
# ============================================================================

"""
    get_hamiltonian(ham::SystemHamiltonian, ::Type{CT}, nq, topology; kwargs...) where CT

Construct a Hamiltonian with custom coefficient type (for ReverseDiff backend).

# Arguments
- `ham`: SystemHamiltonian type instance
- `CT`: Coefficient type (e.g., Float64, ForwardDiff.Dual, etc.)
- `nq`: Number of qubits
- `topology`: Pre-constructed topology
- `kwargs...`: Override default parameters

# Returns
- `PauliSum{CT}`: The constructed Hamiltonian with specified coefficient type
"""
function get_hamiltonian(ham::SystemHamiltonian{:Heisenberg}, ::Type{CT}, nq::Int, topology; kwargs...) where CT
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    return heisenberg_hamiltonian(CT, nq, topology; merged_kwargs...)
end

function get_hamiltonian(ham::SystemHamiltonian{:TFIM}, ::Type{CT}, nq::Int, topology; kwargs...) where CT
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    return tfim_hamiltonian(CT, nq, topology; merged_kwargs...)
end

function get_hamiltonian(ham::SystemHamiltonian{:tV}, ::Type{CT}, nq::Int, topology; kwargs...) where CT
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    return tv_hamiltonian(CT, nq, topology; merged_kwargs...)
end

function get_hamiltonian(ham::SystemHamiltonian{:J1J2_1d}, ::Type{CT}, nq::Int, topology; kwargs...) where CT
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    return j1j2_1d_hamiltonian(CT, nq, topology; merged_kwargs...)
end

function get_hamiltonian(ham::SystemHamiltonian{:J1J2_2d_obc}, ::Type{CT}, nq::Int, topology, lattice_width::Int; kwargs...) where CT
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    return j1j2_2d_square_obc_hamiltonian(CT, nq, topology, lattice_width; merged_kwargs...)
end

function get_hamiltonian(ham::SystemHamiltonian{:Hubbard}, ::Type{CT}, nq::Int, topology; kwargs...) where CT
    merged_kwargs = merge(get_default_kwargs(ham), kwargs)
    Nsites = div(nq, 2)  # Hubbard uses 2 qubits per site (spin up/down)
    return fermi_hubbard_hamiltonian(CT, Nsites, topology; merged_kwargs...)
end

# ============================================================================
# Helper: Get Hamiltonian Constructor Function (for backward compat if needed)
# ============================================================================

"""
    get_hamiltonian_constructor(ham::SystemHamiltonian)

Return the underlying Hamiltonian constructor function.
This can be useful for ReverseDiff tape construction.
"""
get_hamiltonian_constructor(::SystemHamiltonian{:Heisenberg}) = heisenberg_hamiltonian
get_hamiltonian_constructor(::SystemHamiltonian{:TFIM}) = tfim_hamiltonian
get_hamiltonian_constructor(::SystemHamiltonian{:tV}) = tv_hamiltonian
get_hamiltonian_constructor(::SystemHamiltonian{:J1J2_1d}) = j1j2_1d_hamiltonian
get_hamiltonian_constructor(::SystemHamiltonian{:J1J2_2d_obc}) = j1j2_2d_square_obc_hamiltonian
get_hamiltonian_constructor(::SystemHamiltonian{:Hubbard}) = fermi_hubbard_hamiltonian

# ============================================================================
# Pretty Printing
# ============================================================================

Base.show(io::IO, ::SystemHamiltonian{name}) where name = print(io, "SystemHamiltonian(:$name)")

# ============================================================================
# TOPOLOGY TYPE-BASED DISPATCH
# ============================================================================

"""
    SystemTopology{name}

Type-parameterized struct for topology dispatch.

# Example
```julia
chain = SystemTopology(:chain)
square = SystemTopology(:square_obc)
```
"""
struct SystemTopology{name} end
SystemTopology(name::Symbol) = SystemTopology{name}()

"""
    get_topology(topo::SystemTopology, nq; kwargs...)

Construct a topology for the given number of qubits.

# Arguments
- `topo`: SystemTopology type instance
- `nq`: Number of qubits (or sites)
- `kwargs...`: Topology-specific parameters (e.g., `periodic=false`)

# Returns
- Vector of tuples representing edges
"""
function get_topology(::SystemTopology{:chain}, nq::Int; periodic=false)
    return bricklayertopology(nq; periodic=periodic)
end

function get_topology(::SystemTopology{:chain_pbc}, nq::Int)
    return bricklayertopology(nq; periodic=true)
end

function get_topology(::SystemTopology{:square_obc}, nq::Int)
    width = Int(sqrt(nq))
    if width^2 != nq
        error("nq=$nq must be a perfect square for square_obc topology")
    end
    return rectangletopology(width, width; periodic=false)
end

"""
    is_2d(topo::SystemTopology)

Check if topology is 2D.
"""
is_2d(::SystemTopology{:square_obc}) = true
is_2d(::SystemTopology) = false  # Default: 1D

"""
    get_tiles(topo::SystemTopology, tile_width, scaled_width)

Generate tiles for operator pool construction (2D only).

# Returns
- Vector of qubit index vectors for each tile, or `nothing` for 1D (uses sliding window)
"""
function get_tiles(::SystemTopology{:square_obc}, tile_width::Int, scaled_width::Int)
    return generate_obc_square_tiles(tile_width, scaled_width)
end
get_tiles(::SystemTopology, args...) = nothing  # Default: 1D sliding window

"""
    get_lattice_width(topo::SystemTopology, nq)

Get lattice width for 2D topologies (needed for some Hamiltonians like J1J2_2d_obc).
"""
function get_lattice_width(::SystemTopology{:square_obc}, nq::Int)
    width = Int(sqrt(nq))
    if width^2 != nq
        error("nq=$nq must be a perfect square for square_obc topology")
    end
    return width
end
get_lattice_width(::SystemTopology, nq::Int) = nq  # For 1D, just return nq

# Pretty printing
Base.show(io::IO, ::SystemTopology{name}) where name = print(io, "SystemTopology(:$name)")
