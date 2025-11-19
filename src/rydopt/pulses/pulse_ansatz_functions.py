import jax.numpy as jnp
from typing import Protocol


class PulseAnsatzFunction(Protocol):
    def __call__(
        self, t: jnp.ndarray | float, duration: float, params: jnp.ndarray
    ) -> jnp.ndarray | float: ...


def sin_crab(
    t: jnp.ndarray | float, duration: float, params: jnp.ndarray
) -> jnp.ndarray | float:
    r"""Sine-only CRAB pulse ansatz.

    .. math::

       \xi(t)
       = \sum_{n=1}^N \alpha_n
         \sin\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(A_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`\xi(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`2N` entries
            :math:`(A_1, \alpha_1, \dots, A_N, \alpha_N)`.

    Returns:
        Values of :math:`\xi(t)`.
    """
    t = jnp.asarray(t)

    freq_params = params[0::2]
    coeffs = params[1::2]

    n = jnp.arange(len(coeffs)) + 1
    freqs = (1 + 0.5 * jnp.tanh(freq_params)) * n / duration
    phase = 2 * jnp.pi * (t - duration / 2.0)[..., None] * freqs
    return jnp.sum(coeffs * jnp.sin(phase), axis=-1)


def cos_crab(
    t: jnp.ndarray | float, duration: float, params: jnp.ndarray
) -> jnp.ndarray | float:
    r"""Cosine-only CRAB pulse ansatz.

    .. math::

       \xi(t)
       = \sum_{n=1}^N \beta_n
         \cos\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(B_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`\xi(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`2N` entries
            :math:`(B_1, \beta_1, \dots, B_N, \beta_N)`.

    Returns:
        Values of :math:`\xi(t)`.
    """
    t = jnp.asarray(t)

    freq_params = params[0::2]
    coeffs = params[1::2]

    n = jnp.arange(len(coeffs)) + 1
    freqs = (1 + 0.5 * jnp.tanh(freq_params)) * n / duration
    phase = 2 * jnp.pi * (t - duration / 2.0)[..., None] * freqs
    return jnp.sum(coeffs * jnp.cos(phase), axis=-1)


def sin_cos_crab(
    t: jnp.ndarray | float, duration: float, params: jnp.ndarray
) -> jnp.ndarray | float:
    r"""Combined sine and cosine CRAB pulse ansatz.

    .. math::

       \xi(t)
       = \sum_{n=1}^N \alpha_n
         \sin\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(A_n)\right)
           (t - T/2)
         \right)
       \\
       \quad
       + \sum_{n=1}^N \beta_n
         \cos\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(B_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`\xi(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`4N` entries
            :math:`(A_1, \alpha_1, B_1, \beta_1, \dots, A_N, \alpha_N, B_N, \beta_N)`.

    Returns:
        Values of :math:`\xi(t)`.
    """
    freq_params_sin = params[0::4]
    coeffs_sin = params[1::4]
    freq_params_cos = params[2::4]
    coeffs_cos = params[3::4]

    sin_params = jnp.column_stack((freq_params_sin, coeffs_sin)).ravel()
    cos_params = jnp.column_stack((freq_params_cos, coeffs_cos)).ravel()

    return sin_crab(t, duration, sin_params) + cos_crab(t, duration, cos_params)


def const(
    t: jnp.ndarray | float, duration: float, params: jnp.ndarray
) -> jnp.ndarray | float:
    r"""Constant pulse.

    .. math::

       \xi(t) = c_0

    Args:
        t: Time samples at which :math:`\xi(t)` is evaluated.
        duration: Pulse duration :math:`T` (unused).
        params: Array with entry :math:`(c_0)`.

    Returns:
        Values of :math:`\xi(t)`.
    """
    c0 = params[0]

    return jnp.zeros_like(t) + c0


def const_sin_crab(
    t: jnp.ndarray | float, duration: float, params: jnp.ndarray
) -> jnp.ndarray | float:
    r"""Constant offset plus sine CRAB pulse ansatz.

    .. math::

       \xi(t)
       = c_0
       + \sum_{n=1}^N \alpha_n
         \sin\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(A_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`\xi(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`2N+1` entries
            :math:`(c_0, A_1, \alpha_1, \dots, A_N, \alpha_N)`.

    Returns:
        Values of :math:`\xi(t)`.
    """
    c0 = params[0]

    return c0 + sin_crab(t, duration, params[1:])


def const_cos_crab(
    t: jnp.ndarray | float, duration: float, params: jnp.ndarray
) -> jnp.ndarray | float:
    r"""Constant offset plus cosine CRAB pulse ansatz.

    .. math::

       \xi(t)
       = c_0
       + \sum_{n=1}^N \beta_n
         \cos\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(B_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`\xi(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`2N+1` entries
            :math:`(c_0, B_1, \beta_1, \dots, B_N, \beta_N)`.

    Returns:
        Values of :math:`\xi(t)`.
    """
    c0 = params[0]

    return c0 + cos_crab(t, duration, params[1:])


def const_sin_cos_crab(
    t: jnp.ndarray | float, duration: float, params: jnp.ndarray
) -> jnp.ndarray | float:
    r"""Constant offset plus combined sine and cosine CRAB pulse ansatz.

    .. math::

       \xi(t)
       = c_0
       + \sum_{n=1}^N \alpha_n
         \sin\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(A_n)\right)
           (t - T/2)
         \right)
       \\
       \quad
       + \sum_{n=1}^N \beta_n
         \cos\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(B_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`\xi(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`4N+1` entries
            :math:`(c_0, A_1, \alpha_1, B_1, \beta_1, \dots, A_N, \alpha_N, B_N, \beta_N)`.

    Returns:
        Values of :math:`\xi(t)`.
    """

    c0 = params[0]

    return c0 + sin_cos_crab(t, duration, params[1:])
