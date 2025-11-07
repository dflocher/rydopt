import os
import numpy as np
from rydopt import pulses, gate_search_cluster


# run pulse optimizations on a cluster


if __name__ == "__main__":
    # # # # read the parameters at program start when running on the cluster
    # n_atoms_str = sys.argv[1]
    # theta_str = sys.argv[2]
    # eps_str = sys.argv[3]
    # lamb_str = sys.argv[4]
    # delta_str = sys.argv[5]
    # kappa_str = sys.argv[6]
    # Vnn_str = sys.argv[7]
    # Vnnn_str = sys.argv[8]
    # pulse_str = sys.argv[9]
    # N_params_str = sys.argv[10]
    # decay_str = sys.argv[11]
    # run_str = sys.argv[12]
    # N_searches_str = sys.argv[13]
    # N_epochs_str = sys.argv[14]
    # learning_rate_str = sys.argv[15]
    # T_default_str = sys.argv[16]
    # T_penalty_str = sys.argv[17]

    # # # specify the parameters here when running the program locally
    n_atoms_str = "3"
    theta_str = "pi"
    eps_str = "pi"
    lamb_str = "pi"
    delta_str = "0"
    kappa_str = "0"
    Vnn_str = "inf"
    Vnnn_str = "inf"
    pulse_str = "pulse_detuning_cos_crab"
    N_params_str = "8"
    decay_str = "0.0001"
    run_str = "0"
    N_searches_str = "20"
    N_epochs_str = "1500"
    learning_rate_str = "0.05"
    T_default_str = "10.0"
    T_penalty_str = "0.0"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # try to cast the inputs
    n_atoms = int(n_atoms_str)
    if theta_str == "pi":
        theta = np.pi
    elif theta_str == "None":
        theta = None
    elif theta_str == "eps":
        theta = "eps"
    else:
        theta = float(theta_str)
    if eps_str == "pi":
        eps = np.pi
    elif eps_str == "None":
        eps = None
    else:
        eps = float(eps_str)
    if lamb_str == "pi":
        lamb = np.pi
    else:
        lamb = float(lamb_str)
    if delta_str == "pi":
        delta = np.pi
    elif delta_str == "None":
        delta = None
    else:
        delta = float(delta_str)
    if kappa_str == "pi":
        kappa = np.pi
    else:
        kappa = float(kappa_str)
    Vnn = float(Vnn_str)
    Vnnn = float(Vnnn_str)
    pulse = pulses.get_pulse(pulse_str)
    N_params = int(N_params_str)
    decay = float(decay_str)
    if decay == 0:
        decay_on_off = "__decay_off"
    else:
        decay_on_off = "__decay_on"
    run = int(run_str)
    N_searches = int(N_searches_str)
    N_epochs = int(N_epochs_str)
    learning_rate = float(learning_rate_str)
    T_default = float(T_default_str)
    T_penalty = float(T_penalty_str)

    all_costs, all_params, runtime = gate_search_cluster(
        n_atoms,
        Vnn,
        Vnnn,
        theta,
        eps,
        lamb,
        delta,
        kappa,
        pulse,
        T_default,
        N_searches,
        N_params,
        N_epochs,
        learning_rate,
        T_penalty,
        decay,
    )

    directory = os.path.dirname(__file__)
    file = (
        directory
        + "/../Data/data_"
        + n_atoms_str
        + "_atoms"
        + "__theta_"
        + theta_str
        + "__eps_"
        + eps_str
        + "__lamb_"
        + lamb_str
        + "__delta_"
        + delta_str
        + "__kappa_"
        + kappa_str
        + "__Vnn_"
        + Vnn_str
        + "__Vnnn_"
        + Vnnn_str
        + "__"
        + pulse_str
        + "__params_"
        + N_params_str
        + decay_on_off
        + "__run_"
        + run_str
        + ".npz"
    )

    np.savez(
        file,
        n_atoms=n_atoms,
        Vnn=Vnn,
        Vnnn=Vnnn,
        theta=theta,
        eps=eps,
        lamb=lamb,
        delta=delta,
        kappa=kappa,
        pulse=pulse_str,
        T_default=T_default,
        N_searches=N_searches,
        N_params=N_params,
        N_epochs=N_epochs,
        learning_rate=learning_rate,
        T_penalty=T_penalty,
        decay=decay,
        run=run,
        runtime=runtime,
        all_costs=all_costs,
        all_params=all_params,
    )
