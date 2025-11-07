import numpy as np


# pulse ansätze in a form such that they can be processed by the qutip functions for pulse visualization and pulse verification


# pulse that expands the detuning in cosines and leaves the phase fixed at zero
def pulse_detuning_cos_crab(params):
    def Delta_of_t(t):
        Delta = c0 * np.cos(0 * t)
        for j in range(len(coeffs)):
            Delta += coeffs[j] * np.cos(
                2
                * np.pi
                * (j + 1)
                * (1 + 0.5 * np.tanh(frequency_params[j]))
                / T
                * (t - T / 2)
            )
        return Delta

    def Phi_of_t(t):
        return 0.0 * t

    T = params[0]
    c0 = params[1]
    frequency_params = params[2::2]
    coeffs = params[3::2]
    return Delta_of_t, Phi_of_t


# pulse that expands the detuning in cosines and sines and leaves the phase fixed at zero
def pulse_detuning_cos_sin_crab(params):
    def Delta_of_t(t):
        Delta = c0 * np.cos(0 * t)
        for j in range(len(coeffs_cos)):
            Delta += coeffs_cos[j] * np.cos(
                2
                * np.pi
                * (j + 1)
                * (1 + 0.5 * np.tanh(frequency_params_cos[j]))
                / T
                * (t - T / 2)
            )
        for k in range(len(coeffs_sin)):
            Delta += coeffs_sin[k] * np.sin(
                2
                * np.pi
                * (k + 1)
                * (1 + 0.5 * np.tanh(frequency_params_sin[k]))
                / T
                * (t - T / 2)
            )
        return Delta

    def Phi_of_t(t):
        return 0.0 * t

    T = params[0]
    c0 = params[1]
    frequency_params_cos = params[2::4]
    coeffs_cos = params[3::4]
    frequency_params_sin = params[4::4]
    coeffs_sin = params[5::4]
    return Delta_of_t, Phi_of_t


# pulse that expands the phase in sines and leaves the detuning fixed
def pulse_phase_sin_crab(params):
    def Delta_of_t(t):
        return Delta + 0.0 * t

    def Phi_of_t(t):
        phi = 0.0
        for j in range(len(coeffs)):
            phi += coeffs[j] * np.sin(
                2
                * np.pi
                * (j + 1)
                * (1 + 0.5 * np.tanh(frequency_params[j]))
                / T
                * (t - T / 2)
            )
        return phi

    T = params[0]
    Delta = params[1]
    frequency_params = params[2::2]
    coeffs = params[3::2]
    return Delta_of_t, Phi_of_t


# pulse that expands the phase in sines and a linear part, and leaves the detuning fixed
# this allows one to split the detuning contribution into an actual detuning part and a linear phase part
def pulse_phase_sin_crab_smooth(params):
    def Delta_of_t(t):
        return Delta + 0.0 * t

    def Phi_of_t(t):
        phi = phi_linear * (t - T / 2)
        for j in range(len(coeffs)):
            phi += coeffs[j] * np.sin(
                2
                * np.pi
                * (j + 1)
                * (1 + 0.5 * np.tanh(frequency_params[j]))
                / T
                * (t - T / 2)
            )
        return phi

    T = params[0]
    Delta = params[1]
    phi_linear = params[2]
    frequency_params = params[3::2]
    coeffs = params[4::2]
    return Delta_of_t, Phi_of_t


# pulse that expands the phase in sines and cosines and leaves the detuning fixed
def pulse_phase_sin_cos_crab(params):
    def Delta_of_t(t):
        return Delta + 0.0 * t

    def Phi_of_t(t):
        phi = 0.0
        for j in range(len(coeffs_cos)):
            phi += coeffs_cos[j] * np.cos(
                2
                * np.pi
                * (j + 1)
                * (1 + 0.5 * np.tanh(frequency_params_cos[j]))
                / T
                * (t - T / 2)
            )
        for k in range(len(coeffs_sin)):
            phi += coeffs_sin[k] * np.sin(
                2
                * np.pi
                * (k + 1)
                * (1 + 0.5 * np.tanh(frequency_params_sin[k]))
                / T
                * (t - T / 2)
            )
        return phi

    T = params[0]
    Delta = params[1]
    frequency_params_sin = params[2::4]
    coeffs_sin = params[3::4]
    frequency_params_cos = params[4::4]
    coeffs_cos = params[5::4]
    return Delta_of_t, Phi_of_t


# pulse that expands the phase in sines, cosines, and a linear part, and leaves the detuning fixed
# this allows one to split the detuning contribution into an actual detuning part and a linear phase part
def pulse_phase_sin_cos_crab_smooth(params):
    def Delta_of_t(t):
        return Delta + 0.0 * t

    def Phi_of_t(t):
        phi = phi_linear * (t - T / 2)
        for j in range(len(coeffs_cos)):
            phi += coeffs_cos[j] * np.cos(
                2
                * np.pi
                * (j + 1)
                * (1 + 0.5 * np.tanh(frequency_params_cos[j]))
                / T
                * (t - T / 2)
            )
        for k in range(len(coeffs_sin)):
            phi += coeffs_sin[k] * np.sin(
                2
                * np.pi
                * (k + 1)
                * (1 + 0.5 * np.tanh(frequency_params_sin[k]))
                / T
                * (t - T / 2)
            )
        return phi

    T = params[0]
    Delta = params[1]
    phi_linear = params[2]
    frequency_params_sin = params[3::4]
    coeffs_sin = params[4::4]
    frequency_params_cos = params[5::4]
    coeffs_cos = params[6::4]
    return Delta_of_t, Phi_of_t


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_pulse(name):
    pulse_dict = {
        "pulse_detuning_cos_crab": pulse_detuning_cos_crab,
        "pulse_detuning_cos_sin_crab": pulse_detuning_cos_sin_crab,
        "pulse_phase_sin_crab": pulse_phase_sin_crab,
        "pulse_phase_sin_crab_smooth": pulse_phase_sin_crab_smooth,
        "pulse_phase_sin_cos_crab": pulse_phase_sin_cos_crab,
        "pulse_phase_sin_cos_crab_smooth": pulse_phase_sin_cos_crab_smooth,
    }
    return pulse_dict[name]
