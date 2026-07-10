import jax
import jax.numpy as jnp


def sin_series(t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
    r"""Sine-series pulse ansatz with fixed integer harmonics.

    .. math::

       f(t)
       = \sum_{n=1}^N \alpha_n
         \sin\!\left(
           \frac{2\pi n}{T} t
         \right)

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        ansatz_params: Array with :math:`N` entries
            :math:`(\alpha_1, \dots, \alpha_N)`, where :math:`\alpha_n`
            is the amplitude of the :math:`n`-th harmonic.

    Returns:
        Values of :math:`f(t)`.

    """
    t = jnp.asarray(t)

    n = jnp.arange(1, len(ansatz_params) + 1)
    phase = 2 * jnp.pi * t[..., None] * n / duration

    return jnp.sum(ansatz_params * jnp.sin(phase), axis=-1)


def cos_series(t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
    r"""Cosine-series pulse ansatz with fixed odd half-integer harmonics.

    .. math::

       f(t)
       = \sum_{n=1}^N \beta_n
         \cos\!\left(
           \frac{(2n - 1)\pi}{T} t
         \right)

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        ansatz_params: Array with :math:`N` entries
            :math:`(\beta_1, \dots, \beta_N)`, where :math:`\beta_n`
            is the amplitude of the :math:`n`-th odd half-integer cosine mode.

    Returns:
        Values of :math:`f(t)`.

    """
    t = jnp.asarray(t)

    n = jnp.arange(1, len(ansatz_params) + 1)
    phase = jnp.pi * t[..., None] * (2 * n - 1) / duration

    return jnp.sum(ansatz_params * jnp.cos(phase), axis=-1)


def sin_crab(t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
    r"""Sine-only CRAB pulse ansatz.

    .. math::

       f(t)
       = \sum_{n=1}^N \alpha_n
         \sin\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(A_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        ansatz_params: Array with :math:`2N` entries
            :math:`(A_1, \alpha_1, \dots, A_N, \alpha_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    t = jnp.asarray(t)

    freq_params = ansatz_params[0::2]
    coeffs = ansatz_params[1::2]

    n = jnp.arange(len(coeffs)) + 1
    freqs = (1 + 0.5 * jnp.tanh(freq_params)) * n / duration
    phase = 2 * jnp.pi * (t - duration / 2.0)[..., None] * freqs
    return jnp.sum(coeffs * jnp.sin(phase), axis=-1)


def cos_crab(t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
    r"""Cosine-only CRAB pulse ansatz.

    .. math::

       f(t)
       = \sum_{n=1}^N \beta_n
         \cos\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(B_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        ansatz_params: Array with :math:`2N` entries
            :math:`(B_1, \beta_1, \dots, B_N, \beta_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    t = jnp.asarray(t)

    freq_params = ansatz_params[0::2]
    coeffs = ansatz_params[1::2]

    n = jnp.arange(len(coeffs)) + 1
    freqs = (1 + 0.5 * jnp.tanh(freq_params)) * n / duration
    phase = 2 * jnp.pi * (t - duration / 2.0)[..., None] * freqs
    return jnp.sum(coeffs * jnp.cos(phase), axis=-1)


def sin_cos_crab(t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
    r"""Combined sine and cosine CRAB pulse ansatz.

    .. math::

       f(t)
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
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        ansatz_params: Array with :math:`4N` entries
            :math:`(A_1, \alpha_1, B_1, \beta_1, \dots, A_N, \alpha_N, B_N, \beta_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    freq_params_sin = ansatz_params[0::4]
    coeffs_sin = ansatz_params[1::4]
    freq_params_cos = ansatz_params[2::4]
    coeffs_cos = ansatz_params[3::4]

    sin_params = jnp.column_stack((freq_params_sin, coeffs_sin)).ravel()
    cos_params = jnp.column_stack((freq_params_cos, coeffs_cos)).ravel()

    return sin_crab(t, duration, sin_params) + cos_crab(t, duration, cos_params)


def cos_sin_crab(t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
    r"""Combined cosine and sine CRAB pulse ansatz.

    .. math::

       f(t)
       = \sum_{n=1}^N \beta_n
         \cos\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(B_n)\right)
           (t - T/2)
         \right)
       \\
       \quad
       + \sum_{n=1}^N \alpha_n
         \sin\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(A_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        ansatz_params: Array with :math:`4N` entries
            :math:`(B_1, \beta_1, A_1, \alpha_1, \dots, B_N, \beta_N, A_N, \alpha_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    freq_params_cos = ansatz_params[0::4]
    coeffs_cos = ansatz_params[1::4]
    freq_params_sin = ansatz_params[2::4]
    coeffs_sin = ansatz_params[3::4]

    cos_params = jnp.column_stack((freq_params_cos, coeffs_cos)).ravel()
    sin_params = jnp.column_stack((freq_params_sin, coeffs_sin)).ravel()

    return cos_crab(t, duration, cos_params) + sin_crab(t, duration, sin_params)


def const(t: float | jax.Array, _duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
    r"""Constant pulse.

    .. math::

       f(t) = c_0

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        _duration: Pulse duration :math:`T` (unused).
        ansatz_params: Array with entry :math:`(c_0)`.

    Returns:
        Values of :math:`f(t)`.

    """
    c0 = ansatz_params[0]
    return c0 + jnp.zeros_like(t)


def polynomial(
    t: float | jax.Array,
    duration: float | jax.Array,
    ansatz_params: jax.Array,
) -> jax.Array:
    r"""Polynomial pulse ansatz.

    .. math::

       f(t)
       = \sum_{n=0}^{N}
         c_n
         \left(
             \frac{2t}{T}-1
         \right)^n,

    where :math:`c_n` denotes the coefficient of the :math:`n`-th monomial.

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        ansatz_params: Array with :math:`N+1` entries
            :math:`(c_0,\dots,c_N)`, where :math:`c_n`
            is the coefficient of the :math:`n`-th polynomial term.

    Returns:
        Values of :math:`f(t)`.

    """
    x = 2.0 * t / duration - 1.0

    pulse = jnp.zeros_like(x)
    power = jnp.ones_like(x)

    for coeff in ansatz_params:
        pulse += coeff * power
        power *= x

    return pulse


def chebyshev(
    t: float | jax.Array,
    duration: float | jax.Array,
    ansatz_params: jax.Array,
) -> jax.Array:
    r"""Chebyshev polynomial pulse ansatz.

    .. math::

       f(t)
       = \sum_{n=0}^{N}
         c_n\,
         T_n\!\left(
           \frac{2t}{T}-1
         \right),

    where :math:`T_n(x)` denotes the Chebyshev polynomial of the first kind.

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        ansatz_params: Array with :math:`N+1` entries
            :math:`(c_0,\dots,c_N)`, where :math:`c_n`
            is the coefficient of the :math:`n`-th Chebyshev polynomial.

    Returns:
        Values of :math:`f(t)`.

    """
    x = 2.0 * t / duration - 1.0

    pulse = ansatz_params[0]
    if ansatz_params.size == 1:
        return pulse

    Tm2 = jnp.ones_like(x)
    Tm1 = x

    pulse = pulse + ansatz_params[1] * Tm1

    for n in range(2, ansatz_params.size):
        Tn = 2.0 * x * Tm1 - Tm2
        pulse = pulse + ansatz_params[n] * Tn
        Tm2 = Tm1
        Tm1 = Tn

    return pulse


def legendre(
    t: float | jax.Array,
    duration: float | jax.Array,
    ansatz_params: jax.Array,
) -> jax.Array:
    r"""Legendre polynomial pulse ansatz.

    .. math::

       f(t)
       = \sum_{n=0}^{N}
         c_n\,
         P_n\!\left(
           \frac{2t}{T}-1
         \right),

    where :math:`P_n(x)` denotes the Legendre polynomial of degree
    :math:`n`.

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        ansatz_params: Array with :math:`N+1` entries
            :math:`(c_0,\dots,c_N)`, where :math:`c_n`
            is the coefficient of the :math:`n`-th Legendre polynomial.

    Returns:
        Values of :math:`f(t)`.

    """
    x = 2.0 * t / duration - 1.0

    pulse = ansatz_params[0]
    if ansatz_params.size == 1:
        return pulse

    Pm2 = jnp.ones_like(x)
    Pm1 = x

    pulse = pulse + ansatz_params[1] * Pm1

    for n in range(2, ansatz_params.size):
        Pn = ((2 * n - 1) * x * Pm1 - (n - 1) * Pm2) / n
        pulse = pulse + ansatz_params[n] * Pn
        Pm2 = Pm1
        Pm1 = Pn

    return pulse


def bspline(
    t: float | jax.Array,
    duration: float | jax.Array,
    ansatz_params: jax.Array,
) -> jax.Array:
    r"""Uniform cubic B-spline pulse ansatz.

    .. math::

       f(t)
       = \sum_{i=0}^{N-1}
         c_i\,
         B_i^{(3)}
         \!\left(
           \frac{N-3}{T}t
         \right),

    where :math:`B_i^{(3)}` denotes the :math:`i`-th cubic B-spline basis
    function and :math:`c_i` are the corresponding control point values.

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        ansatz_params: Array with :math:`N` entries
            :math:`(c_0,\dots,c_{N-1})`, where :math:`c_i`
            is the value of the :math:`i`-th spline control point.

    Returns:
        Values of :math:`f(t)`.

    """
    p = ansatz_params
    n = p.size

    if n < 4:
        raise ValueError("Need at least four control points.")

    x = t / duration * (n - 3)

    i = jnp.floor(x).astype(int)
    i = jnp.clip(i, 0, n - 4)

    u = x - i

    B0 = (1 - u) ** 3 / 6
    B1 = (3 * u**3 - 6 * u**2 + 4) / 6
    B2 = (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6
    B3 = u**3 / 6

    return B0 * p[i] + B1 * p[i + 1] + B2 * p[i + 2] + B3 * p[i + 3]


def piecewise_constant(
    t: float | jax.Array,
    duration: float | jax.Array,
    ansatz_params: jax.Array,
) -> jax.Array:
    r"""Piecewise constant pulse ansatz.

    .. math::

       f(t)
       = c_i,
       \qquad
       t \in
       \left[
       \frac{iT}{N},
       \frac{(i+1)T}{N}
       \right),

    where :math:`N` is the number of pulse segments.

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        ansatz_params: Array with :math:`N` entries
            :math:`(c_0,\dots,c_{N-1})`, where :math:`c_i`
            is the amplitude of the :math:`i`-th segment.

    Returns:
        Values of :math:`f(t)`.

    """
    n = ansatz_params.size

    idx = jnp.floor(t / duration * n).astype(int)
    idx = jnp.clip(idx, 0, n - 1)

    return ansatz_params[idx]
