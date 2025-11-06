import numpy as np
import pulse_visualization
import pulse_verification


# utility functions to translate a detuning-pulse to a phase-pulse and vice versa;
# function that analyzes gate fidelities etc. for sets of pulses


# sin_cos=False: translates the parameters for a pulse phase_sin_crab to parameters for the pulse detuning_cos_crab
# sin_cos=True: translates the parameters for a pulse phase_sin_cos_crab to parameters for the pulse detuning_cos_sin_crab
def translate_phase_to_detuning(params, sin_cos=False):
    params = params.copy()
    T = params[0]
    if sin_cos:
        step = 4
    else:
        step = 2
    n = 0
    for i in range(3, len(params), step):
        n += 1
        params[i] = params[i] * 2 * np.pi * n / T * (1 + 0.5 * np.tanh(params[i - 1]))
    if sin_cos:
        m = 0
        for j in range(5, len(params), step):
            m += 1
            params[j] = -params[j] * 2 * np.pi * m / T * (1 + 0.5 * np.tanh(params[j - 1]))
    return params


# sin_cos=False: translates the parameters for a pulse detuning_cos_crab to parameters for the pulse phase_sin_crab
# sin_cos=True: translates the parameters for a pulse detuning_cos_sin_crab to parameters for the pulse phase_sin_cos_crab
def translate_detuning_to_phase(params, sin_cos=False):
    params = params.copy()
    T = params[0]
    if sin_cos:
        step = 4
    else:
        step = 2
    n = 0
    for i in range(3, len(params), step):
        n += 1
        params[i] = params[i] * T / (2 * np.pi * n * (1 + 0.5 * np.tanh(params[i - 1])))
    if sin_cos:
        m = 0
        for j in range(5, len(params), step):
            m += 1
            params[j] = -params[j] * T / (2 * np.pi * m * (1 + 0.5 * np.tanh(params[j - 1])))
    return params


# for pulses phase_sin_crab/phase_sin_cos_crab this function translates a part of the static (actual) detuning into a linear contribution
# to the phase profile in order to let the pulse begin smoothly (for a phase_sin_crab pulse it then also ends smoothly)
def make_phase_pulse_smooth(params, sin_cos=False):
    slope = 0.0
    T = params[0]
    if sin_cos:
        step = 4
    else:
        step = 2
    n = 0
    for i in range(3, len(params), step):
        n += 1
        slope += params[i] * 2 * np.pi * n / T * (1 + 0.5 * np.tanh(params[i - 1])) * np.cos(np.pi * n * (1 + 0.5 * np.tanh(params[i - 1])))
    if sin_cos:
        m = 0
        for j in range(5, len(params), step):
            m += 1
            slope += params[j] * 2 * np.pi * m / T * (1 + 0.5 * np.tanh(params[j - 1])) * np.sin(np.pi * m * (1 + 0.5 * np.tanh(params[j - 1])))
    params_new = np.zeros(len(params) + 1)
    params_new[0] = T
    params_new[1] = params[1] + slope
    params_new[2] = -slope
    params_new[3::] = params[2::]
    return params_new


# takes a list of pulses and calculates important properties such as infidelities
def postprocess_pulses(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params_list, decay, Vnnn_list=None):
    infidelities_nodecay = []
    infidelities_nodecay_2 = []
    infidelities_decay = []
    infidelities_decay_2 = []
    TRs = []
    Ts = []
    if Vnnn_list is None:
        Vnnn_list = Vnnn * np.ones(len(params_list))
    for params, vnnn in zip(params_list, Vnnn_list):

        f_decay = pulse_visualization.visualize_subsystem_dynamics(n_atoms, Vnn, vnnn, theta, eps, lamb, delta, kappa, pulse, params, decay, plot=False)
        f_nodecay = pulse_visualization.visualize_subsystem_dynamics(n_atoms, Vnn, vnnn, theta, eps, lamb, delta, kappa, pulse, params, 0.0, plot=False)
        f_decay_2, _ = pulse_verification.verify(n_atoms, Vnn, vnnn, theta, eps, lamb, delta, kappa, pulse, params, decay)
        f_nodecay_2, tr = pulse_verification.verify(n_atoms, Vnn, vnnn, theta, eps, lamb, delta, kappa, pulse, params, 0.0)

        infidelities_nodecay.append(1 - f_nodecay)
        infidelities_nodecay_2.append(1 - f_nodecay_2)
        infidelities_decay.append(1 - f_decay)
        infidelities_decay_2.append(1 - f_decay_2)
        TRs.append(tr)
        Ts.append(params[0])

    print('\n')
    print('Ts               : [' + ', '.join('{p:.6f}'.format(p=p) for p in Ts) + ']')
    print('infids no decay  : [' + ', '.join('{p:.4e}'.format(p=p) for p in infidelities_nodecay) + ']')
    print('infids no decay 2: [' + ', '.join('{p:.4e}'.format(p=p) for p in infidelities_nodecay_2) + ']')
    print('infids w/ decay  : [' + ', '.join('{p:.4e}'.format(p=p) for p in infidelities_decay) + ']')
    print('infids w/ decay 2: [' + ', '.join('{p:.4e}'.format(p=p) for p in infidelities_decay_2) + ']')
    print('TRs              : [' + ', '.join('{p:.6f}'.format(p=p) for p in TRs) + ']')


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)

    # optimization parameters
    params_p = [10.97094681, 0.19566367, 0.43131090, -1.16460209, 1.05669771, -0.70545851, 0.88054914, -0.22756692]
    params_d = [10.97094681, 0.19566367, 0.43131090, -0.80251671, 1.05669771, -1.12496327, 0.88054914, -0.52914581]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # params_detuning = translate_phase_to_detuning(params_p, sin_cos=False)
    # print('[' + ', '.join('{p:.8f}'.format(p=p) for p in params_detuning) + ']')
    # pulse_visualization.visualize_pulse('pulse_detuning_cos_crab', params_detuning)

    # params_phase = translate_detuning_to_phase(params_d, sin_cos=False)
    # print('[' + ', '.join('{p:.8f}'.format(p=p) for p in params_phase) + ']')
    # pulse_visualization.visualize_pulse('pulse_phase_sin_crab', params_phase)

    # params_phase_smooth = make_phase_pulse_smooth(params_p, sin_cos=False)
    # print('[' + ', '.join('{p:.8f}'.format(p=p) for p in params_phase_smooth) + ']')
    # pulse_visualization.visualize_pulse('pulse_phase_sin_crab_smooth', params_phase_smooth)
