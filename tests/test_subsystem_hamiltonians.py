import numpy as np
import numpy.typing as npt

from rydopt.gates.subsystem_hamiltonians_general import (
    H_1_atom_general,
    H_2_atoms_general,
    H_3_atoms_general,
    H_4_atoms_general,
)


def generate_hamiltonian(
    N: int,
    Delta: float,
    Xi: float,
    Omega: float,
    decay: float,
    interactions: list[tuple[int, int, float]],
    s: list[float],
) -> npt.NDArray[np.complex128]:
    dim = 2**N
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)

    # Diagonal terms
    for idx in range(dim):
        n_exc = idx.bit_count()
        hamiltonian[idx, idx] = -n_exc * Delta - 1j * (n_exc * decay / 2.0)

        for i, j, Vij in interactions:
            bi = (idx >> (N - i)) & 1
            bj = (idx >> (N - j)) & 1
            hamiltonian[idx, idx] += Vij * (bi & bj)

    # Off-diagonal terms
    for row in range(dim):
        for atom in range(1, N + 1):
            bitpos = N - atom
            col = row ^ (1 << bitpos)
            row_bit = (row >> bitpos) & 1
            phase = np.exp(-1j * Xi) if row_bit == 0 else np.exp(1j * Xi)
            hamiltonian[row, col] = 0.5 * s[atom - 1] * Omega * phase

    return hamiltonian


def test_H_1_atom_general() -> None:
    Delta, Xi, Omega, decay = 0.7, 0.3, 1.2, 0.4
    s1 = 0.9

    hamiltonian_rydopt = np.asarray(H_1_atom_general(Delta=Delta, Xi=Xi, Omega=Omega, decay=decay, s1=s1))
    hamiltonian_ref = generate_hamiltonian(
        N=1,
        Delta=Delta,
        Xi=Xi,
        Omega=Omega,
        decay=decay,
        interactions=[],
        s=[s1],
    )

    assert np.allclose(hamiltonian_rydopt, hamiltonian_ref)


def test_H_2_atoms_general() -> None:
    Delta, Xi, Omega, decay = 0.7, 0.3, 1.2, 0.4
    V12 = -0.85
    s1, s2 = 0.9, 1.1

    hamiltonian_rydopt = np.asarray(
        H_2_atoms_general(Delta=Delta, Xi=Xi, Omega=Omega, decay=decay, V12=V12, s1=s1, s2=s2)
    )
    hamiltonian_ref = generate_hamiltonian(
        N=2,
        Delta=Delta,
        Xi=Xi,
        Omega=Omega,
        decay=decay,
        interactions=[(1, 2, V12)],
        s=[s1, s2],
    )

    assert np.allclose(hamiltonian_rydopt, hamiltonian_ref)


def test_H_3_atoms_general() -> None:
    Delta, Xi, Omega, decay = 0.7, 0.3, 1.2, 0.4
    V12, V13, V23 = -0.85, 0.33, 1.05
    s1, s2, s3 = 0.9, 1.1, 0.8

    hamiltonian_rydopt = np.asarray(
        H_3_atoms_general(
            Delta=Delta,
            Xi=Xi,
            Omega=Omega,
            decay=decay,
            V12=V12,
            V13=V13,
            V23=V23,
            s1=s1,
            s2=s2,
            s3=s3,
        )
    )
    hamiltonian_ref = generate_hamiltonian(
        N=3,
        Delta=Delta,
        Xi=Xi,
        Omega=Omega,
        decay=decay,
        interactions=[(1, 2, V12), (1, 3, V13), (2, 3, V23)],
        s=[s1, s2, s3],
    )

    assert np.allclose(hamiltonian_rydopt, hamiltonian_ref)


def test_H_4_atoms_general() -> None:
    Delta, Xi, Omega, decay = 0.7, 0.3, 1.2, 0.4
    V12, V13, V14 = -0.85, 0.33, -0.12
    V23, V24, V34 = 1.05, 0.5, -0.7
    s1, s2, s3, s4 = 0.9, 1.1, 0.8, 1.3

    hamiltonian_rydopt = np.asarray(
        H_4_atoms_general(
            Delta=Delta,
            Xi=Xi,
            Omega=Omega,
            decay=decay,
            V12=V12,
            V13=V13,
            V14=V14,
            V23=V23,
            V24=V24,
            V34=V34,
            s1=s1,
            s2=s2,
            s3=s3,
            s4=s4,
        )
    )
    hamiltonian_ref = generate_hamiltonian(
        N=4,
        Delta=Delta,
        Xi=Xi,
        Omega=Omega,
        decay=decay,
        interactions=[(1, 2, V12), (1, 3, V13), (1, 4, V14), (2, 3, V23), (2, 4, V24), (3, 4, V34)],
        s=[s1, s2, s3, s4],
    )

    assert np.allclose(hamiltonian_rydopt, hamiltonian_ref)
