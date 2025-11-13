import jax.numpy as jnp

# pulse ansätze in a form such that they can be processed by the ODE solver in the pulse optimization


# pulse that expands the detuning in cosines and leaves the phase fixed at zero
def pulse_detuning_cos_crab(H, params):
    def schroedinger_equation(t, y, args):
        times, c0, frequency_params, coeffs = args
        T = times[-1]
        Delta_of_t = c0 * jnp.cos(0 * t)
        for j in range(len(coeffs)):
            Delta_of_t += coeffs[j] * jnp.cos(
                2
                * jnp.pi
                * (j + 1)
                * (1 + 0.5 * jnp.tanh(frequency_params[j]))
                / T
                * (t - T / 2)
            )
        return -1j * H(Delta_of_t, 0.0, 1) @ y

    T = params[0]
    times = jnp.array([T])
    c0 = params[1]
    frequency_params = params[2::2]
    coeffs = params[3::2]
    args = (times, c0, frequency_params, coeffs)
    return schroedinger_equation, args


# pulse that expands the detuning in cosines and sines and leaves the phase fixed at zero
def pulse_detuning_cos_sin_crab(H, params):
    def schroedinger_equation(t, y, args):
        (
            times,
            c0,
            frequency_params_cos,
            coeffs_cos,
            frequency_params_sin,
            coeffs_sin,
        ) = args
        T = times[-1]
        Delta_of_t = c0 * jnp.cos(0 * t)
        for j in range(len(coeffs_cos)):
            Delta_of_t += coeffs_cos[j] * jnp.cos(
                2
                * jnp.pi
                * (j + 1)
                * (1 + 0.5 * jnp.tanh(frequency_params_cos[j]))
                / T
                * (t - T / 2)
            )
        for k in range(len(coeffs_sin)):
            Delta_of_t += coeffs_sin[k] * jnp.sin(
                2
                * jnp.pi
                * (k + 1)
                * (1 + 0.5 * jnp.tanh(frequency_params_sin[k]))
                / T
                * (t - T / 2)
            )
        return -1j * H(Delta_of_t, 0.0, 1) @ y

    T = params[0]
    times = jnp.array([T])
    c0 = params[1]
    frequency_params_cos = params[2::4]
    coeffs_cos = params[3::4]
    frequency_params_sin = params[4::4]
    coeffs_sin = params[5::4]
    args = (
        times,
        c0,
        frequency_params_cos,
        coeffs_cos,
        frequency_params_sin,
        coeffs_sin,
    )
    return schroedinger_equation, args


# pulse that expands the phase in sines and leaves the detuning fixed
def pulse_phase_sin_crab(H, params):
    def schroedinger_equation(t, y, args):
        times, Delta, frequency_params, coeffs = args
        T = times[-1]
        phi_of_t = 0.0
        for j in range(len(coeffs)):
            phi_of_t += coeffs[j] * jnp.sin(
                2
                * jnp.pi
                * (j + 1)
                * (1 + 0.5 * jnp.tanh(frequency_params[j]))
                / T
                * (t - T / 2)
            )
        return -1j * H(Delta, phi_of_t, 1) @ y

    T = params[0]
    times = jnp.array([T])
    Delta = params[1]
    frequency_params = params[2::2]
    coeffs = params[3::2]
    args = (times, Delta, frequency_params, coeffs)
    return schroedinger_equation, args


# pulse that expands the phase in sines and cosines and leaves the detuning fixed
def pulse_phase_sin_cos_crab(H, params):
    def schroedinger_equation(t, y, args):
        (
            times,
            Delta,
            frequency_params_sin,
            coeffs_sin,
            frequency_params_cos,
            coeffs_cos,
        ) = args
        T = times[-1]
        phi_of_t = 0.0
        for j in range(len(coeffs_cos)):
            phi_of_t += coeffs_cos[j] * jnp.cos(
                2
                * jnp.pi
                * (j + 1)
                * (1 + 0.5 * jnp.tanh(frequency_params_cos[j]))
                / T
                * (t - T / 2)
            )
        for k in range(len(coeffs_sin)):
            phi_of_t += coeffs_sin[k] * jnp.sin(
                2
                * jnp.pi
                * (k + 1)
                * (1 + 0.5 * jnp.tanh(frequency_params_sin[k]))
                / T
                * (t - T / 2)
            )
        return -1j * H(Delta, phi_of_t, 1) @ y

    T = params[0]
    times = jnp.array([T])
    Delta = params[1]
    frequency_params_sin = params[2::4]
    coeffs_sin = params[3::4]
    frequency_params_cos = params[4::4]
    coeffs_cos = params[5::4]
    args = (
        times,
        Delta,
        frequency_params_sin,
        coeffs_sin,
        frequency_params_cos,
        coeffs_cos,
    )
    return schroedinger_equation, args


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_pulse(name):
    pulse_dict = {
        "pulse_detuning_cos_crab": pulse_detuning_cos_crab,
        "pulse_detuning_cos_sin_crab": pulse_detuning_cos_sin_crab,
        "pulse_phase_sin_crab": pulse_phase_sin_crab,
        "pulse_phase_sin_cos_crab": pulse_phase_sin_cos_crab,
    }
    return pulse_dict[name]
