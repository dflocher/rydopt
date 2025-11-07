import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# internal function that provides Hamiltonians and fidelity functions


# given the number of atoms and interaction strengths, the function provides the Hamiltonians that govern the appropriate subsystem dynamics,
# and it provides functions to calculate state fidelities given all subsystem states
def get_subsystem_Hamiltonians(
    n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, decay
):
    # Hamiltonian for subspace |001> -- |00r>
    def H_2LS_1(Phi, Delta):
        return jnp.array(
            [
                [0.0, 0.5 * (jnp.cos(Phi) - 1j * jnp.sin(Phi))],
                [0.5 * (jnp.cos(Phi) + 1j * jnp.sin(Phi)), Delta - 1j * 0.5 * decay],
            ]
        )

    # Hamiltonian for subspace |011> -- (|01r> + |0r1>)  (Vnn = infinity)
    def H_2LS_sqrt2(Phi, Delta):
        return jnp.array(
            [
                [0.0, 0.5 * jnp.sqrt(2) * (jnp.cos(Phi) - 1j * jnp.sin(Phi))],
                [
                    0.5 * jnp.sqrt(2) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    Delta - 1j * 0.5 * decay,
                ],
            ]
        )

    # Hamiltonian for subspace |111> -- |W>  (Vnn = Vnnn = infinity)
    def H_2LS_sqrt3(Phi, Delta):
        return jnp.array(
            [
                [0.0, 0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi))],
                [
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    Delta - 1j * 0.5 * decay,
                ],
            ]
        )

    # Hamiltonian for subspace |1111> -- |W4>  (Vnn = Vnnn = infinity)
    def H_2LS_sqrt4(Phi, Delta):
        return jnp.array(
            [
                [0.0, (jnp.cos(Phi) - 1j * jnp.sin(Phi))],
                [(jnp.cos(Phi) + 1j * jnp.sin(Phi)), Delta - 1j * 0.5 * decay],
            ]
        )

    # Hamiltonian for subspace |101> -- (|10r> + |r01>) -- |r0r>
    def H_3LS_Vnnn(Phi, Delta):
        return jnp.array(
            [
                [0.0, 0.5 * jnp.sqrt(2) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)), 0],
                [
                    0.5 * jnp.sqrt(2) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    Delta - 1j * 0.5 * decay,
                    0.5 * jnp.sqrt(2) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                ],
                [
                    0,
                    0.5 * jnp.sqrt(2) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    2 * Delta + Vnnn - 1j * decay,
                ],
            ]
        )

    # Hamiltonian for subspace |011> -- (|01r> + |0r1>) -- |0rr>
    def H_3LS_Vnn(Phi, Delta):
        return jnp.array(
            [
                [0.0, 0.5 * jnp.sqrt(2) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)), 0],
                [
                    0.5 * jnp.sqrt(2) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    Delta - 1j * 0.5 * decay,
                    0.5 * jnp.sqrt(2) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                ],
                [
                    0,
                    0.5 * jnp.sqrt(2) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    2 * Delta + Vnn - 1j * decay,
                ],
            ]
        )

    # Hamiltonian for subspace |111> -- |W> -- |X> -- |r1r>  (Vnn = infinity)
    def H_4LS(Phi, Delta):
        return jnp.array(
            [
                [0.0, 0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)), 0.0, 0.0],
                [
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    Delta - 1j * 0.5 * decay,
                    0.0,
                    (1 / jnp.sqrt(3)) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                ],
                [
                    0.0,
                    0.0,
                    Delta - 1j * 0.5 * decay,
                    (1 / jnp.sqrt(6)) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                ],
                [
                    0.0,
                    (1 / jnp.sqrt(3)) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    (1 / jnp.sqrt(6)) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    2 * Delta + Vnnn - 1j * decay,
                ],
            ]
        )

    # Hamiltonian for subspace |111> -- |W> -- |X> -- |X'> -- |W'> -- |rrr>
    def H_6LS(Phi, Delta):
        return jnp.array(
            [
                [
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    Delta - 1j * 0.5 * decay,
                    0.0,
                    0.0,
                    (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    Delta - 1j * 0.5 * decay,
                    -0.5 * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    -0.5 * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    (1 / 3) * Vnn + (2 / 3) * Vnnn + 2 * Delta - 1j * decay,
                    (1 / 3) * jnp.sqrt(2) * (Vnn - Vnnn),
                    0.0,
                ],
                [
                    0.0,
                    (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    0.0,
                    (1 / 3) * jnp.sqrt(2) * (Vnn - Vnnn),
                    (2 / 3) * Vnn + (1 / 3) * Vnnn + 2 * Delta - 1j * decay,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    2 * Vnn + Vnnn + 3 * Delta - 1j * 1.5 * decay,
                ],
            ]
        )

    # Hamiltonian for subspace |1110> -- |W>|0> -- |W'>|0> -- |rrr0>
    def H_4LS_Vnnn(Phi, Delta):
        return jnp.array(
            [
                [0.0, 0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)), 0.0, 0.0],
                [
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    Delta - 1j * 0.5 * decay,
                    (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.0,
                ],
                [
                    0.0,
                    (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    Vnnn + 2 * Delta - 1j * decay,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                ],
                [
                    0.0,
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    3 * Vnnn + 3 * Delta - 1j * 1.5 * decay,
                ],
            ]
        )

    # Hamiltonian for subspace |1111> -- |W> -- |Z> -- |S> -- |rrr1>  (Vnn = infinity)
    # |Z> = sqrt(3)/2 * |111r> - 1/sqrt(12) * (|11r1> + |1r11> + |r111>)
    # |S> = 1/sqrt(3) * (|1rr1> + |r1r1> + |rr11>)
    def H_5LS(Phi, Delta):
        return jnp.array(
            [
                [0.0, (jnp.cos(Phi) - 1j * jnp.sin(Phi)), 0.0, 0.0, 0.0],
                [
                    (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    Delta - 1j * 0.5 * decay,
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    Delta - 1j * 0.5 * decay,
                    0.5 * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.0,
                ],
                [
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    0.5 * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    Vnnn + 2 * Delta - 1j * decay,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    3 * Vnnn + 3 * Delta - 1j * 1.5 * decay,
                ],
            ]
        )

    # Hamiltonian for subspace |1111> -- |W> -- ... -- |rrrr>
    def H_8LS(Phi, Delta):
        return jnp.array(
            [
                [0.0, (jnp.cos(Phi) - 1j * jnp.sin(Phi)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    Delta - 1j * 0.5 * decay,
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    Delta - 1j * 0.5 * decay,
                    -0.5 * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.5 * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    -0.5 * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    2 * Delta - 1j * decay + Vnn,
                    0.0,
                    (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    0.5 * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    0.0,
                    2 * Delta - 1j * decay + Vnnn,
                    0.5 * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    0.5 * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    3 * Delta - 1j * 1.5 * decay + 2 * Vnn + Vnnn,
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    0.0,
                    3 * Delta - 1j * 1.5 * decay + 3 * Vnnn,
                    0.5 * (jnp.cos(Phi) - 1j * jnp.sin(Phi)),
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5 * jnp.sqrt(3) * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    0.5 * (jnp.cos(Phi) + 1j * jnp.sin(Phi)),
                    4 * Delta - 1j * 2 * decay + 3 * Vnn + 3 * Vnnn,
                ],
            ]
        )

    def fidelity_2qubits(subsystem_states):
        # populations and phases of evolved states
        # |10>, |01>
        psi_A = subsystem_states[0]
        rA = jnp.absolute(psi_A[0, 0])
        pA = jnp.angle(psi_A[0, 0])
        # |11>
        psi_B = subsystem_states[1]
        rB = jnp.absolute(psi_B[0, 0])
        pB = jnp.angle(psi_B[0, 0])
        # calculate gate fidelity
        F = (
            1
            + 4 * rA**2
            + rB**2
            + 4 * rA
            + 2 * rB * (1 + 2 * rA) * jnp.cos(pB - 2 * pA - theta)
        ) / 16
        return -F

    def fidelity_3qubits_3subsystems_Vinf(subsystem_states):
        rA, rC, rD, pA, pC, pD = extract_3qubits_3subsystems(subsystem_states)
        return three_qubit_fidelity(rA, rC, rC, rD, pA, pC, pC, pD, theta, eps, lamb)

    def fidelity_3qubits_3subsystems_Vinf_ThetaArbitrary(subsystem_states):
        rA, rC, rD, pA, pC, pD = extract_3qubits_3subsystems(subsystem_states)
        return three_qubit_fidelity(
            rA, rC, rC, rD, pA, pC, pC, pD, pC - 2 * pA, pC - 2 * pA, lamb
        )

    def fidelity_3qubits_3subsystems_VnnnZero(subsystem_states):
        rA, rC, rD, pA, pC, pD = extract_3qubits_3subsystems(subsystem_states)
        return three_qubit_fidelity(
            rA, rA**2, rC, rD, pA, 2 * pA, pC, pD, theta, eps, lamb
        )

    def fidelity_3qubits_3subsystems_VnnnZero_ThetaArbitrary(subsystem_states):
        rA, rC, rD, pA, pC, pD = extract_3qubits_3subsystems(subsystem_states)
        return three_qubit_fidelity(
            rA, rA**2, rC, rD, pA, 2 * pA, pC, pD, pC - 2 * pA, 0.0, lamb
        )

    def fidelity_3qubits_4subsystems(subsystem_states):
        rA, rB, rC, rD, pA, pB, pC, pD = extract_3qubits_4subsystems(subsystem_states)
        return three_qubit_fidelity(rA, rB, rC, rD, pA, pB, pC, pD, theta, eps, lamb)

    def fidelity_3qubits_4subsystems_epsArbitrary(subsystem_states):
        rA, rB, rC, rD, pA, pB, pC, pD = extract_3qubits_4subsystems(subsystem_states)
        return three_qubit_fidelity(
            rA, rB, rC, rD, pA, pB, pC, pD, theta, pB - 2 * pA, lamb
        )

    def fidelity_3qubits_4subsystems_epsArbitraryThetaLinked(subsystem_states):
        rA, rB, rC, rD, pA, pB, pC, pD = extract_3qubits_4subsystems(subsystem_states)
        return three_qubit_fidelity(
            rA, rB, rC, rD, pA, pB, pC, pD, pB - 2 * pA, pB - 2 * pA, lamb
        )

    def fidelity_3qubits_4subsystems_epsThetaArbitrary(subsystem_states):
        rA, rB, rC, rD, pA, pB, pC, pD = extract_3qubits_4subsystems(subsystem_states)
        return three_qubit_fidelity(
            rA, rB, rC, rD, pA, pB, pC, pD, pC - 2 * pA, pB - 2 * pA, lamb
        )

    def fidelity_4qubits_4subsystems_Vinf(subsystem_states):
        rA, rB, rD, rF, pA, pB, pD, pF = extract_4qubits_4subsystems(subsystem_states)
        return four_qubit_fidelity(
            rA,
            rB,
            rB,
            rD,
            rD,
            rF,
            pA,
            pB,
            pB,
            pD,
            pD,
            pF,
            theta,
            eps,
            lamb,
            delta,
            kappa,
        )

    def fidelity_4qubits_4subsystems_VnnnZero(subsystem_states):
        rA, rB, rD, rF, pA, pB, pD, pF = extract_4qubits_4subsystems(subsystem_states)
        return four_qubit_fidelity(
            rA,
            rB,
            rA**2,
            rD,
            rA**3,
            rF,
            pA,
            pB,
            2 * pA,
            pD,
            3 * pA,
            pF,
            theta,
            eps,
            lamb,
            delta,
            kappa,
        )

    def fidelity_4qubits_6subsystems(subsystem_states):
        rA, rB, rC, rD, rE, rF, pA, pB, pC, pD, pE, pF = extract_4qubits_6subsystems(
            subsystem_states
        )
        return four_qubit_fidelity(
            rA,
            rB,
            rC,
            rD,
            rE,
            rF,
            pA,
            pB,
            pC,
            pD,
            pE,
            pF,
            theta,
            eps,
            lamb,
            delta,
            kappa,
        )

    def fidelity_4qubits_6subsystems_epsDeltaArbitrary(subsystem_states):
        rA, rB, rC, rD, rE, rF, pA, pB, pC, pD, pE, pF = extract_4qubits_6subsystems(
            subsystem_states
        )
        return four_qubit_fidelity(
            rA,
            rB,
            rC,
            rD,
            rE,
            rF,
            pA,
            pB,
            pC,
            pD,
            pE,
            pF,
            theta,
            pC - 2 * pA,
            lamb,
            pE - 3 * pA - 3 * (pC - 2 * pA),
            kappa,
        )

    if n_atoms == 2 and Vnn == float("inf"):
        Hamiltonians = (H_2LS_1, H_2LS_sqrt2)
        input_states = (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
        )
        fidelity_fn = fidelity_2qubits
    elif n_atoms == 2:
        Hamiltonians = (H_2LS_1, H_3LS_Vnn)
        input_states = (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        )
        fidelity_fn = fidelity_2qubits
    elif n_atoms == 3 and Vnn == float("inf") and Vnnn == float("inf"):
        Hamiltonians = (H_2LS_1, H_2LS_sqrt2, H_2LS_sqrt3)
        input_states = (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
        )
        if theta is None and eps is None:
            fidelity_fn = fidelity_3qubits_3subsystems_Vinf_ThetaArbitrary
        else:
            fidelity_fn = fidelity_3qubits_3subsystems_Vinf
            if theta != eps:
                raise IOError("for Vnn = Vnnn = infinity: eps must be equal to theta")
    elif n_atoms == 3 and Vnn == float("inf") and Vnnn == 0:
        Hamiltonians = (H_2LS_1, H_2LS_sqrt2, H_4LS)
        input_states = (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        )
        if theta is None and eps is None:
            fidelity_fn = fidelity_3qubits_3subsystems_VnnnZero_ThetaArbitrary
        else:
            fidelity_fn = fidelity_3qubits_3subsystems_VnnnZero
            if eps != 0:
                raise IOError("for Vnnn = 0: eps must be equal to zero")
    elif n_atoms == 3 and Vnn == float("inf"):
        Hamiltonians = (H_2LS_1, H_3LS_Vnnn, H_2LS_sqrt2, H_4LS)
        input_states = (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        )
        if theta is None and eps is None:
            fidelity_fn = fidelity_3qubits_4subsystems_epsThetaArbitrary
        elif theta == "eps" and eps is None:
            fidelity_fn = fidelity_3qubits_4subsystems_epsArbitraryThetaLinked
        elif eps is None:
            fidelity_fn = fidelity_3qubits_4subsystems_epsArbitrary
        else:
            fidelity_fn = fidelity_3qubits_4subsystems
    elif n_atoms == 3:
        Hamiltonians = (H_2LS_1, H_3LS_Vnnn, H_3LS_Vnn, H_6LS)
        input_states = (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array(
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
            ),
        )
        if theta is None and eps is None:
            fidelity_fn = fidelity_3qubits_4subsystems_epsThetaArbitrary
        elif theta == "eps" and eps is None:
            fidelity_fn = fidelity_3qubits_4subsystems_epsArbitraryThetaLinked
        elif eps is None:
            fidelity_fn = fidelity_3qubits_4subsystems_epsArbitrary
        else:
            fidelity_fn = fidelity_3qubits_4subsystems
    elif n_atoms == 4 and Vnn == float("inf") and Vnnn == float("inf"):
        Hamiltonians = (H_2LS_1, H_2LS_sqrt2, H_2LS_sqrt3, H_2LS_sqrt4)
        input_states = (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
        )
        fidelity_fn = fidelity_4qubits_4subsystems_Vinf
        if theta != eps or lamb != delta:
            raise IOError(
                "for Vnn = Vnnn = infinity: eps must be equal to theta and delta must be equal to lambda"
            )
    elif n_atoms == 4 and Vnn == float("inf") and Vnnn == 0:
        Hamiltonians = (H_2LS_1, H_2LS_sqrt2, H_4LS, H_5LS)
        input_states = (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        )
        fidelity_fn = fidelity_4qubits_4subsystems_VnnnZero
        if eps != 0 or delta != 0:
            raise IOError(
                "for Vnnn = 0: eps must be equal to zero and delta must be equal to 0"
            )
    elif n_atoms == 4 and Vnn == float("inf"):
        Hamiltonians = (H_2LS_1, H_2LS_sqrt2, H_3LS_Vnnn, H_4LS, H_4LS_Vnnn, H_5LS)
        input_states = (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        )
        if eps is None and delta is None:
            fidelity_fn = fidelity_4qubits_6subsystems_epsDeltaArbitrary
        else:
            fidelity_fn = fidelity_4qubits_6subsystems
    elif n_atoms == 4:
        Hamiltonians = (H_2LS_1, H_3LS_Vnn, H_3LS_Vnnn, H_6LS, H_4LS_Vnnn, H_8LS)
        input_states = (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array(
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
            ),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array(
                [
                    1.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ]
            ),
        )
        if eps is None and delta is None:
            fidelity_fn = fidelity_4qubits_6subsystems_epsDeltaArbitrary
        else:
            fidelity_fn = fidelity_4qubits_6subsystems
    else:
        raise IOError(
            "The requested combination of atoms and interaction strengths is not supported."
        )
    return Hamiltonians, input_states, fidelity_fn


def extract_3qubits_3subsystems(subsystem_states):
    # populations and phases of evolved states
    # |100>, |010>, |001>
    psi_A = subsystem_states[0]
    rA = jnp.absolute(psi_A[0, 0])
    pA = jnp.angle(psi_A[0, 0])
    # |110>, |011>
    psi_C = subsystem_states[1]
    rC = jnp.absolute(psi_C[0, 0])
    pC = jnp.angle(psi_C[0, 0])
    # |111>
    psi_D = subsystem_states[2]
    rD = jnp.absolute(psi_D[0, 0])
    pD = jnp.angle(psi_D[0, 0])
    return rA, rC, rD, pA, pC, pD


def extract_3qubits_4subsystems(subsystem_states):
    # population and phases of evolved states
    # |100>, |010>, |001>
    psi_A = subsystem_states[0]
    rA = jnp.absolute(psi_A[0, 0])
    pA = jnp.angle(psi_A[0, 0])
    # |101>
    psi_B = subsystem_states[1]
    rB = jnp.absolute(psi_B[0, 0])
    pB = jnp.angle(psi_B[0, 0])
    # |110>, |011>
    psi_C = subsystem_states[2]
    rC = jnp.absolute(psi_C[0, 0])
    pC = jnp.angle(psi_C[0, 0])
    # |111>
    psi_D = subsystem_states[3]
    rD = jnp.absolute(psi_D[0, 0])
    pD = jnp.angle(psi_D[0, 0])
    return rA, rB, rC, rD, pA, pB, pC, pD


def extract_4qubits_4subsystems(subsystem_states):
    # population and phases of evolved states
    # |0001>, |0010>, |0100>, |1000>
    psi_A = subsystem_states[0]
    rA = jnp.absolute(psi_A[0, 0])
    pA = jnp.angle(psi_A[0, 0])
    # |0011>, |0101>, |1001>
    psi_B = subsystem_states[1]
    rB = jnp.absolute(psi_B[0, 0])
    pB = jnp.angle(psi_B[0, 0])
    # |0111>, |1011>, |1101>
    psi_D = subsystem_states[2]
    rD = jnp.absolute(psi_D[0, 0])
    pD = jnp.angle(psi_D[0, 0])
    # |1111>
    psi_F = subsystem_states[3]
    rF = jnp.absolute(psi_F[0, 0])
    pF = jnp.angle(psi_F[0, 0])
    return rA, rB, rD, rF, pA, pB, pD, pF


def extract_4qubits_6subsystems(subsystem_states):
    # population and phases of evolved states
    # |0001>, |0010>, |0100>, |1000>
    psi_A = subsystem_states[0]
    rA = jnp.absolute(psi_A[0, 0])
    pA = jnp.angle(psi_A[0, 0])
    # |0011>, |0101>, |1001>
    psi_B = subsystem_states[1]
    rB = jnp.absolute(psi_B[0, 0])
    pB = jnp.angle(psi_B[0, 0])
    # |0110>, |1010>, |1100>
    psi_C = subsystem_states[2]
    rC = jnp.absolute(psi_C[0, 0])
    pC = jnp.angle(psi_C[0, 0])
    # |0111>, |1011>, |1101>
    psi_D = subsystem_states[3]
    rD = jnp.absolute(psi_D[0, 0])
    pD = jnp.angle(psi_D[0, 0])
    # |1110>
    psi_E = subsystem_states[4]
    rE = jnp.absolute(psi_E[0, 0])
    pE = jnp.angle(psi_E[0, 0])
    # |1111>
    psi_F = subsystem_states[5]
    rF = jnp.absolute(psi_F[0, 0])
    pF = jnp.angle(psi_F[0, 0])
    return rA, rB, rC, rD, rE, rF, pA, pB, pC, pD, pE, pF


def three_qubit_fidelity(rA, rB, rC, rD, pA, pB, pC, pD, theta, eps, lamb):
    a_1 = pB - 2 * pA - eps
    a_2 = pC - 2 * pA - theta
    a_3 = pD - 3 * pA - lamb - 2 * theta - eps
    F = (
        1
        + 9 * rA**2
        + rB**2
        + 4 * rC**2
        + rD**2
        + 6 * rA
        + 2 * rB * (1 + 3 * rA) * jnp.cos(a_1)
        + 4 * rC * (1 + 3 * rA) * jnp.cos(a_2)
        + 2 * rD * (1 + 3 * rA) * jnp.cos(a_3)
        + 4 * rB * rC * jnp.cos(a_1 - a_2)
        + 2 * rB * rD * jnp.cos(a_1 - a_3)
        + 4 * rC * rD * jnp.cos(a_2 - a_3)
    ) / 64
    return -F


def four_qubit_fidelity(
    rA, rB, rC, rD, rE, rF, pA, pB, pC, pD, pE, pF, theta, eps, lamb, delta, kappa
):
    a_1 = pB - 2 * pA - theta
    a_2 = pC - 2 * pA - eps
    a_3 = pD - 3 * pA - lamb - 2 * theta - eps
    a_4 = pE - 3 * pA - delta - 3 * eps
    a_5 = pF - 4 * pA - kappa - delta - 3 * lamb - 3 * theta - 3 * eps
    # F = abs((1 + 4*rA + 3*rB*jnp.exp(1j*a_1) + 3*rC*jnp.exp(1j*a_2) + 3*rD*jnp.exp(1j*a_3) + rE*jnp.exp(1j*a_4) + rF*jnp.exp(1j*a_5)) / 16) ** 2
    F = (
        1
        + 16 * rA**2
        + 9 * rB**2
        + 9 * rC**2
        + 9 * rD**2
        + rE**2
        + rF**2
        + 8 * rA
        + 6 * rB * (1 + 4 * rA) * jnp.cos(a_1)
        + 6 * rC * (1 + 4 * rA) * jnp.cos(a_2)
        + 6 * rD * (1 + 4 * rA) * jnp.cos(a_3)
        + 2 * rE * (1 + 4 * rA) * jnp.cos(a_4)
        + 2 * rF * (1 + 4 * rA) * jnp.cos(a_5)
        + 18 * rB * rC * jnp.cos(a_1 - a_2)
        + 18 * rB * rD * jnp.cos(a_1 - a_3)
        + 6 * rB * rE * jnp.cos(a_1 - a_4)
        + 6 * rB * rF * jnp.cos(a_1 - a_5)
        + 18 * rC * rD * jnp.cos(a_2 - a_3)
        + 6 * rC * rE * jnp.cos(a_2 - a_4)
        + 6 * rC * rF * jnp.cos(a_2 - a_5)
        + 6 * rD * rE * jnp.cos(a_3 - a_4)
        + 6 * rD * rF * jnp.cos(a_3 - a_5)
        + 2 * rE * rF * jnp.cos(a_4 - a_5)
    ) / 256
    return -F
