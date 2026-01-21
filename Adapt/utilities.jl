"""
Utility Functions for ADAPT-VQE
================================

This file contains utility functions for bit operations, Pauli string conversions,
and operator pool generation.
"""

using PauliPropagation
using Random

# ============================================================================
# BIT OPERATIONS AND PAULI STRING CONVERSIONS
# ============================================================================

"""
    generate_full_bit_pool(nq::Int)

Generate full operator pool as bit representations.
Uses PauliPropagation's internal UInt type system.
"""
function generate_full_bit_pool(nq::Int)
    UIntType = PauliPropagation.getinttype(nq)
    pool = UIntType[]
    
    for i in 1:(4^nq - 1)
        push!(pool, UIntType(i))
    end
    
    return pool
end

"""
    bit_to_paulistring_general(bit_repr, nq; sites=nothing, total_nq=nq)

Convert bit representation to PauliString with custom qubit site mapping.

# Arguments
- `bit_repr`: Bit representation of Pauli string
- `nq`: Number of qubits in the bit representation (size of the tile/local region)
- `sites`: Vector of qubit indices where the Pauli operators should be applied.
           If `nothing`, defaults to [1, 2, ..., nq]
- `total_nq`: Total number of qubits in the full system (for PauliString construction)

# Returns
- `PauliString`: The Pauli string with operators applied to specified sites
- `paulis`: Vector of Pauli operator symbols
- `active_sites`: Vector of qubit indices where non-identity operators are applied

# Example
```julia
# For a 2x2 tile on a 3x3 lattice, tile covers qubits [1, 2, 4, 5]
bit_repr = UInt8(0b00001111)  # Some Pauli string on 4 qubits
pstr, paulis, active_sites = bit_to_paulistring_general(bit_repr, 4; sites=[1, 2, 4, 5], total_nq=9)
```
"""
function bit_to_paulistring_general(bit_repr, nq; sites=nothing, total_nq=nq)
    # Default sites to sequential qubits if not provided
    if isnothing(sites)
        sites = collect(1:nq)
    end
    
    # Validate input
    if length(sites) != nq
        throw(ArgumentError("Length of sites ($(length(sites))) must match nq ($nq)"))
    end
    
    paulis = Symbol[]
    active_sites = Int[]
    
    # Map each bit position to the corresponding site
    for (bit_idx, qubit_site) in enumerate(sites)
        pauli_val = getpauli(bit_repr, bit_idx)
        if pauli_val != 0
            pauli_symbol = [:I, :X, :Y, :Z][pauli_val + 1]
            push!(paulis, pauli_symbol)
            push!(active_sites, qubit_site)
        end
    end
    
    return PauliString(total_nq, paulis, active_sites, 1.0), paulis, active_sites
end

"""
    pauli_rotation_from_bits_general(bit_repr, nq; sites=nothing)

Convert bit representation to PauliRotation gate with custom qubit site mapping.

# Arguments
- `bit_repr`: Bit representation of Pauli string
- `nq`: Number of qubits in the bit representation
- `sites`: Vector of qubit indices where the rotation should be applied.
           If `nothing`, defaults to [1, 2, ..., nq]
"""
function pauli_rotation_from_bits_general(bit_repr, nq; sites=nothing)
    _, paulis, active_sites = bit_to_paulistring_general(bit_repr, nq; sites=sites, total_nq=nq)
    return PauliRotation(paulis, active_sites)
end

"""
    promote_paulisum_coeffs(psum::PauliSum, ::Type{CT}) where {CT}

Promote PauliSum coefficients to target numeric type CT.

This is needed for automatic differentiation with ForwardDiff, where coefficients
need to be promoted to Dual number types to enable gradient computation.

# Arguments
- `psum`: Input PauliSum with original coefficient type
- `CT`: Target coefficient type (e.g., Float64 or ForwardDiff.Dual)

# Returns
- New PauliSum with coefficients converted to type CT
"""
function promote_paulisum_coeffs(psum::PauliSum, ::Type{CT}) where {CT}
    newpsum = PauliSum(CT, psum.nqubits)
    for (pstr, coeff) in psum
        # `pstr` is a PauliStringType; add! accepts pstr directly
        add!(newpsum, pstr, convert(CT, tonumber(coeff)))
    end
    return newpsum
end

"""
    append_from_bits_general!(circuit, thetas, chose_op, nq; sites=nothing, theta_init=rand())

Append operator to circuit with initial parameter.

# Arguments
- `circuit`: Circuit to append to
- `thetas`: Parameter vector to append to
- `chose_op`: Chosen operator in bit representation
- `nq`: Number of qubits in the bit representation (tile size for tiled approach)
- `sites`: Vector of qubit indices where the operator should be applied (default: nothing → [1:nq])
- `theta_init`: Initial parameter value (default: random)

# Returns
- Updated `circuit` and `thetas`
"""
function append_from_bits_general!(circuit, thetas, chose_op, nq; sites=nothing, theta_init=rand())
    gate = pauli_rotation_from_bits_general(chose_op, nq; sites=sites)
    push!(circuit, gate)
    push!(thetas, theta_init)
    return circuit, thetas
end
