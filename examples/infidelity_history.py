import jax

jax.config.update("jax_platforms", "cuda,cpu")
import rydopt as ro  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

if __name__ == "__main__":
    tol = 1e-8
    num_initializations = 100000
    num_steps = 1000

    # Gate
    gate = ro.gates.ThreeQubitGateIsosceles(
        phi=None,
        theta=np.pi,
        eps=np.pi,
        lamb=np.pi,
        Vnn=float("inf"),
        Vnnn=float("inf"),
        decay=0.0,
    )

    # Pulse
    pulse = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab
    )

    # Parameter bounds for choosing random initial parameters
    min_initial_params = (10, (-3,), (-3, -3, 100, -3), ())
    max_initial_params = (17, (3,), (3, 3, 100, 3), ())
    fixed_initial_params = (False, (False,), (False, False, True, False), ())

    # Run optimization
    r = ro.optimization.multi_start_adam(
        gate,
        pulse,
        min_initial_params,
        max_initial_params,
        fixed_initial_params=fixed_initial_params,
        num_steps=num_steps,
        num_initializations=num_initializations,
        min_converged_initializations=num_initializations,
        tol=tol,
        return_all=True,
        return_history=True,
    )

    plt.plot(r.history)
    plt.yscale("log")
    plt.xlim(0, num_steps)
    plt.ylim(tol, 1)
    plt.xlabel("Optimization step")
    plt.ylabel("Infidelity")
    plt.savefig("infidelity_history.png")
