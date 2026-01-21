"""
Hamiltonian Constructors and Overlap Functions
===============================================

This file contains all Hamiltonian construction functions and initial state
overlap functions for ADAPT-VQE.
"""

using PauliPropagation

# ============================================================================
# HAMILTONIAN CONSTRUCTORS
# ============================================================================

"""
    heisenberg_hamiltonian(::Type{CT}, nq::Int, topology; J=1.0) where CT

Construct Heisenberg Hamiltonian: H = J Σ_{⟨i,j⟩} (XᵢXⱼ + YᵢYⱼ + ZᵢZⱼ)

# Arguments
- `CT`: Coefficient type (typically Float64)
- `nq`: Number of qubits
- `topology`: Vector of tuples representing edges (e.g., [(1,2), (2,3)])
              If `nothing`, defaults to 1D chain with open boundary conditions
- `J`: Coupling constant (default: 1.0)

# Returns
- `PauliSum`: The Heisenberg Hamiltonian

# Example
```julia
nq = 4
topology = bricklayertopology(nq; periodic=false)
H = heisenberg_hamiltonian(nq, topology)
```
"""
function heisenberg_hamiltonian(::Type{CT}, nq::Int, topology; J=1.0) where CT
    psum = PauliSum(CT, nq)
    Jc = convert(CT, J)
    
    if isnothing(topology)
        topology = bricklayertopology(nq; periodic=false)
    end

    for pair in topology
        add!(psum, [:X, :X], collect(pair), Jc)
        add!(psum, [:Y, :Y], collect(pair), Jc)
        add!(psum, [:Z, :Z], collect(pair), Jc)
    end
    
    return psum
end

# Convenience method with Float64 default
heisenberg_hamiltonian(nq::Int, topology; J=1.0) = 
    heisenberg_hamiltonian(Float64, nq, topology; J=J)


"""
    tfim_hamiltonian(::Type{CT}, nq::Int, topology; J=1.0, h=1.0) where CT

Construct Transverse Field Ising Model Hamiltonian: H = J Σ_{⟨i,j⟩} ZᵢZⱼ + h Σᵢ Xᵢ

# Arguments
- `CT`: Coefficient type (typically Float64)
- `nq`: Number of qubits
- `topology`: Vector of tuples representing edges
              If `nothing`, defaults to 1D chain with open boundary conditions
- `J`: Coupling constant (default: 1.0)
- `h`: Transverse field strength (default: 1.0)

# Returns
- `PauliSum`: The TFIM Hamiltonian
"""
function tfim_hamiltonian(::Type{CT}, nq::Int, topology; J=1.0, h=1.0) where CT
    psum = PauliSum(CT, nq)
    Jc = convert(CT, J)
    hc = convert(CT, h)
    
    if isnothing(topology)
        topology = bricklayertopology(nq; periodic=false)
    end
    
    # ZZ coupling terms
    for pair in topology
        add!(psum, [:Z, :Z], collect(pair), Jc)
    end
    
    # Transverse field terms
    for qind in 1:nq
        add!(psum, :X, qind, hc)
    end
    
    return psum
end

# Convenience method with Float64 default
tfim_hamiltonian(nq::Int, topology; J=1.0, h=1.0) = 
    tfim_hamiltonian(Float64, nq, topology; J=J, h=h)

"""
    tv_hamiltonian(::Type{CT}, nq::Int, topology; t=1.0, V=1.0) where CT

Construct spinless t–V Hamiltonian using Jordan–Wigner mapping:
H = -t Σ_{⟨i,j⟩}(c†ᵢcⱼ + c†ⱼcᵢ) + V Σ_{⟨i,j⟩} nᵢ nⱼ

Each term is mapped to Pauli strings.
"""
function tv_hamiltonian(::Type{CT}, nq::Int, topology; t=1.0, V=1.0) where CT
    psum = PauliSum(CT, nq)
    tc = convert(CT, t)
    Vc = convert(CT, V)

    if isnothing(topology)
        topology = bricklayertopology(nq; periodic=false)
    end

    # --- hopping terms (-t) ---
    for (i, j) in topology
        i, j = sort([i, j])
        zsites = collect((i+1):(j-1))
        nZ = length(zsites)

        xstring = Symbol[:X; fill(:Z, nZ)...; :X]
        ystring = Symbol[:Y; fill(:Z, nZ)...; :Y]

        add!(psum, xstring, [i; zsites; j], -tc/2)
        add!(psum, ystring, [i; zsites; j], -tc/2)
    end

    # --- density–density interactions (V n_i n_j) ---
    for (i, j) in topology
        # n_i n_j = ¼ (I - Z_i - Z_j + Z_i Z_j)
        add!(psum, [:I, :I], [i, j],  Vc/4)
        add!(psum, [:Z], [i], -Vc/4)
        add!(psum, [:Z], [j], -Vc/4)
        add!(psum, [:Z, :Z], [i, j], Vc/4)
    end

    return psum
end

# Convenience wrapper
tv_hamiltonian(nq::Int, topology; t=1.0, V=1.0) =
    tv_hamiltonian(Float64, nq, topology; t=t, V=V)


"""
    j1j2_1d_hamiltonian(::Type{CT}, nq::Int, topology; J1::Float64=1.0, J2::Float64=0.5) where CT

Construct J1-J2 Heisenberg Hamiltonian for 1D chain:
H = J1 Σ_{⟨i,j⟩} (XᵢXⱼ + YᵢYⱼ + ZᵢZⱼ) + J2 Σ_{⟨⟨i,j⟩⟩} (XᵢXⱼ + YᵢYⱼ + ZᵢZⱼ)

# Arguments
- `CT`: Coefficient type (typically Float64)
- `nq`: Number of qubits
- `topology`: Vector of tuples representing nearest-neighbor edges
- `J1`: Nearest-neighbor coupling constant (default: 1.0)
- `J2`: Next-nearest-neighbor coupling constant (default: 0.5)

# Returns
- `PauliSum`: The J1-J2 Heisenberg Hamiltonian
"""
function j1j2_1d_hamiltonian(::Type{CT}, nq::Int, topology; J1::Float64=1.0, J2::Float64=0.5) where CT
    psum = PauliSum(CT,nq)
    J1c = convert(CT, J1)
    J2c = convert(CT, J2)

    # J1: Heisenberg on nearest neighbors
    for pair in topology
        add!(psum, [:X, :X],collect(pair), J1c)
        add!(psum, [:Y, :Y],collect(pair), J1c)
        add!(psum, [:Z, :Z],collect(pair), J1c)
    end

    # J2: Heisenberg on next-nearest neighbors
    for qind in 1:(length(topology)-1)
        pair = mod1.([qind, qind+2],nq)
        add!(psum, [:X, :X], collect(pair), J2c)
        add!(psum, [:Y, :Y], collect(pair), J2c)
        add!(psum, [:Z, :Z], collect(pair), J2c)
    end

    return psum
end

# Convenience wrapper
j1j2_1d_hamiltonian(nq::Int, topology; J1::Float64=1.0, J2::Float64=0.5) = 
    j1j2_1d_hamiltonian(Float64, nq, topology; J1=J1, J2=J2)

"""
    j1j2_2d_square_obc_hamiltonian(::Type{CT}, nq::Int, topology, lattice_width::Int; J1::Float64=1.0, J2::Float64=0.5) where CT

Construct J1-J2 Heisenberg Hamiltonian for 2D square lattice with open boundary conditions:
H = J1 Σ_{⟨i,j⟩} (XᵢXⱼ + YᵢYⱼ + ZᵢZⱼ) + J2 Σ_{⟨⟨i,j⟩⟩} (XᵢXⱼ + YᵢYⱼ + ZᵢZⱼ)

J1 couples nearest neighbors (horizontal and vertical), J2 couples diagonal neighbors.

# Arguments
- `CT`: Coefficient type (typically Float64)
- `nq`: Number of qubits (total sites in the lattice)
- `topology`: Vector of tuples representing nearest-neighbor edges (horizontal and vertical)
- `lattice_width`: Width of the square lattice (e.g., 3 for a 3x3 lattice)
- `J1`: Nearest-neighbor coupling constant (default: 1.0)
- `J2`: Diagonal-neighbor coupling constant (default: 0.5)

# Returns
- `PauliSum`: The J1-J2 Heisenberg Hamiltonian for 2D square lattice
"""
function j1j2_2d_square_obc_hamiltonian(::Type{CT}, nq::Int, topology, lattice_width::Int; J1::Float64=1.0, J2::Float64=0.5) where CT
    psum = PauliSum(CT, nq)
    J1c = convert(CT, J1)
    J2c = convert(CT, J2)

    # J1: Heisenberg on nearest neighbors (horizontal and vertical)
    for pair in topology
        add!(psum, [:X, :X], collect(pair), J1c)
        add!(psum, [:Y, :Y], collect(pair), J1c)
        add!(psum, [:Z, :Z], collect(pair), J1c)
    end

    # J2: Heisenberg on diagonal neighbors
    lattice_height = nq ÷ lattice_width
    
    for idx in 1:nq
        row = (idx - 1) ÷ lattice_width  # 0-indexed
        col = (idx - 1) % lattice_width   # 0-indexed
        
        # Check all 4 diagonal neighbors
        diagonal_neighbors = []
        
        # Southeast: (row+1, col+1)
        if row + 1 < lattice_height && col + 1 < lattice_width
            push!(diagonal_neighbors, (row + 1) * lattice_width + (col + 1) + 1)
        end
        
        # Southwest: (row+1, col-1)
        if row + 1 < lattice_height && col > 0
            push!(diagonal_neighbors, (row + 1) * lattice_width + (col - 1) + 1)
        end
        
        # Northeast: (row-1, col+1)
        if row > 0 && col + 1 < lattice_width
            push!(diagonal_neighbors, (row - 1) * lattice_width + (col + 1) + 1)
        end
        
        # Northwest: (row-1, col-1)
        if row > 0 && col > 0
            push!(diagonal_neighbors, (row - 1) * lattice_width + (col - 1) + 1)
        end
        
        # Add Heisenberg interaction for each diagonal pair
        for neighbor in diagonal_neighbors
            if neighbor > idx  # Avoid double counting
                add!(psum, [:X, :X], [idx, neighbor], J2c)
                add!(psum, [:Y, :Y], [idx, neighbor], J2c)
                add!(psum, [:Z, :Z], [idx, neighbor], J2c)
            end
        end
    end

    return psum
end

# Convenience method with Float64 default
j1j2_2d_square_obc_hamiltonian(nq::Int, topology, lattice_width::Int; J1::Float64=1.0, J2::Float64=0.5) = 
    j1j2_2d_square_obc_hamiltonian(Float64, nq, topology, lattice_width; J1=J1, J2=J2) 

"""
    fermi_hubbard_hamiltonian(::Type{CT}, Nsites::Int, topology; t=1.0, U=4.0) where CT

Construct spinful Fermi–Hubbard Hamiltonian:
H = -t Σ_{⟨i,j⟩,σ} (c†_{iσ} c_{jσ} + h.c.) + U Σ_i n_{i↑} n_{i↓}

Using Jordan–Wigner mapping with 2*Nsites qubits.
"""
function fermi_hubbard_hamiltonian(::Type{CT}, Nsites::Int, topology; t=1.0, U=4.0) where CT
    nq = 2 * Nsites
    psum = PauliSum(CT, nq)
    tc = convert(CT, t)
    Uc = convert(CT, U)

    if isnothing(topology)
        topology = bricklayertopology(Nsites; periodic=false)
    end

    # --- hopping terms for each spin ---
    for (i, j) in topology
        for spin in 0:1  # 0 = up, 1 = down
            i_spin = 2*i - spin
            j_spin = 2*j - spin
            i_spin, j_spin = sort([i_spin, j_spin])
            zsites = collect((i_spin+1):(j_spin-1))
            nZ = length(zsites)

            xstring = Symbol[:X; fill(:Z, nZ)...; :X]
            ystring = Symbol[:Y; fill(:Z, nZ)...; :Y]

            add!(psum, xstring, [i_spin; zsites; j_spin], -tc/2)
            add!(psum, ystring, [i_spin; zsites; j_spin], -tc/2)
        end
    end

    # --- on-site interaction U n_i↑ n_i↓ ---
    for i in 1:Nsites
        i_up = 2*i - 1
        i_dn = 2*i
        # n_i↑ n_i↓ = ¼ (I - Z↑ - Z↓ + Z↑Z↓)
        add!(psum, [:I, :I], [i_up, i_dn],  Uc/4)
        add!(psum, [:Z], [i_up], -Uc/4)
        add!(psum, [:Z], [i_dn], -Uc/4)
        add!(psum, [:Z, :Z], [i_up, i_dn], Uc/4)
    end

    return psum
end

# Convenience wrapper
fermi_hubbard_hamiltonian(Nsites::Int, topology; t=1.0, U=4.0) =
    fermi_hubbard_hamiltonian(Float64, Nsites, topology; t=t, U=U)


# ============================================================================
# OVERLAP FUNCTIONS
# ============================================================================

"""
    neel_bits(nq::Int; up_on_odd::Bool=true)

Generate bit representation for Néel state.

# Arguments
- `nq`: Number of qubits
- `up_on_odd`: If true, |01010...⟩; if false, |10101...⟩

# Returns
- Vector of indices where bit is "1"
"""
function neel_bits(nq::Int; up_on_odd::Bool=true)
    if up_on_odd
        return collect(2:2:nq)  # |0101...⟩
    else
        return collect(1:2:nq)  # |1010...⟩
    end
end

"""
    overlapwithneel(operator, nq::Int; up_on_odd::Bool=true, params=nothing)

Compute overlap of an operator with the Néel state.
For Heisenberg model ground state search.
"""
function overlapwithneel(operator, nq::Int; up_on_odd::Bool=true, params=nothing)
    nb = neel_bits(nq; up_on_odd=up_on_odd)
    return overlapwithcomputational(operator, nb)
end

"""
    overlapwithplus(operator, nq::Int; params=nothing)

Compute overlap of an operator with the |+⟩ state (equal superposition).
For TFIM ground state search.
"""
function overlapwithplus(operator, nq::Int; params=nothing)
    return PauliPropagation.overlapwithplus(operator) 
end

"""
    overlapwithfock(operator, nq::Int; occs::Vector{Int}, params=nothing)

Compute overlap ⟨Fock(occs)| operator |Fock(occs)⟩
for fermionic ADAPT-VQE initialization.
"""
function overlapwithfock(operator, nq::Int; occs::Vector{Int}, params=nothing)
    return overlapwithcomputational(operator, occs)
end
