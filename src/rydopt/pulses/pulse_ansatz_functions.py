import jax.numpy as jnp


def sin_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
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
        params: Array with :math:`2N` entries
            :math:`(A_1, \alpha_1, \dots, A_N, \alpha_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    t = jnp.asarray(t)

    freq_params = params[0::2]
    coeffs = params[1::2]

    n = jnp.arange(len(coeffs)) + 1
    freqs = (1 + 0.5 * jnp.tanh(freq_params)) * n / duration
    phase = 2 * jnp.pi * (t - duration / 2.0)[..., None] * freqs
    return jnp.sum(coeffs * jnp.sin(phase), axis=-1)


def cos_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
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
        params: Array with :math:`2N` entries
            :math:`(B_1, \beta_1, \dots, B_N, \beta_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    t = jnp.asarray(t)

    freq_params = params[0::2]
    coeffs = params[1::2]

    n = jnp.arange(len(coeffs)) + 1
    freqs = (1 + 0.5 * jnp.tanh(freq_params)) * n / duration
    phase = 2 * jnp.pi * (t - duration / 2.0)[..., None] * freqs
    return jnp.sum(coeffs * jnp.cos(phase), axis=-1)


def sin_cos_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
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
        params: Array with :math:`4N` entries
            :math:`(A_1, \alpha_1, B_1, \beta_1, \dots, A_N, \alpha_N, B_N, \beta_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    freq_params_sin = params[0::4]
    coeffs_sin = params[1::4]
    freq_params_cos = params[2::4]
    coeffs_cos = params[3::4]

    sin_params = jnp.column_stack((freq_params_sin, coeffs_sin)).ravel()
    cos_params = jnp.column_stack((freq_params_cos, coeffs_cos)).ravel()

    return sin_crab(t, duration, sin_params) + cos_crab(t, duration, cos_params)


def cos_sin_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
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
        params: Array with :math:`4N` entries
            :math:`(B_1, \beta_1, A_1, \alpha_1, \dots, B_N, \beta_N, A_N, \alpha_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    freq_params_cos = params[0::4]
    coeffs_cos = params[1::4]
    freq_params_sin = params[2::4]
    coeffs_sin = params[3::4]

    cos_params = jnp.column_stack((freq_params_cos, coeffs_cos)).ravel()
    sin_params = jnp.column_stack((freq_params_sin, coeffs_sin)).ravel()

    return cos_crab(t, duration, cos_params) + sin_crab(t, duration, sin_params)


def const(t: jnp.ndarray | float, _duration: float, params: jnp.ndarray) -> jnp.ndarray:
    r"""Constant pulse.

    .. math::

       f(t) = c_0

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        _duration: Pulse duration :math:`T` (unused).
        params: Array with entry :math:`(c_0)`.

    Returns:
        Values of :math:`f(t)`.

    """
    c0 = params[0]
    return c0 + jnp.zeros_like(t)


def const_sin_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
    r"""Constant offset plus sine CRAB pulse ansatz.

    .. math::

       f(t)
       = c_0
       + \sum_{n=1}^N \alpha_n
         \sin\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(A_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`2N+1` entries
            :math:`(c_0, A_1, \alpha_1, \dots, A_N, \alpha_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    c0 = params[0]
    return c0 + sin_crab(t, duration, params[1:])


def const_cos_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
    r"""Constant offset plus cosine CRAB pulse ansatz.

    .. math::

       f(t)
       = c_0
       + \sum_{n=1}^N \beta_n
         \cos\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(B_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`2N+1` entries
            :math:`(c_0, B_1, \beta_1, \dots, B_N, \beta_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    c0 = params[0]
    return c0 + cos_crab(t, duration, params[1:])


def const_sin_cos_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
    r"""Constant offset plus combined sine and cosine CRAB pulse ansatz.

    .. math::

       f(t)
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
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`4N+1` entries
            :math:`(c_0, A_1, \alpha_1, B_1, \beta_1, \dots, A_N, \alpha_N, B_N, \beta_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    c0 = params[0]
    return c0 + sin_cos_crab(t, duration, params[1:])


def const_cos_sin_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
    r"""Constant offset plus combined cosine and sine CRAB pulse ansatz.

    .. math::

       f(t)
       = c_0
       + \sum_{n=1}^N \beta_n
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
        params: Array with :math:`4N+1` entries
            :math:`(c_0, B_1, \beta_1, A_1, \alpha_1, \dots, B_N, \beta_N, A_N, \alpha_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    c0 = params[0]
    return c0 + cos_sin_crab(t, duration, params[1:])


def lin_sin_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
    r"""Straight line plus sine CRAB pulse ansatz.

    .. math::

       f(t)
       = c_1 (t - T/2)
       + \sum_{n=1}^N \alpha_n
         \sin\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(A_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`2N+1` entries
            :math:`(c_1, A_1, \alpha_1, \dots, A_N, \alpha_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    c1 = params[0]
    return c1 * (t - duration / 2.0) + sin_crab(t, duration, params[1:])


def lin_cos_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
    r"""Straight line plus cosine CRAB pulse ansatz.

    .. math::

       f(t)
       = c_1 (t - T/2)
       + \sum_{n=1}^N \beta_n
         \cos\!\left(
           \frac{2\pi}{T}\,
           n\left(1 + \tfrac{1}{2}\tanh(B_n)\right)
           (t - T/2)
         \right)

    Args:
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`2N+1` entries
            :math:`(c_1, B_1, \beta_1, \dots, B_N, \beta_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    c1 = params[0]
    return c1 * (t - duration / 2.0) + cos_crab(t, duration, params[1:])


def lin_sin_cos_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
    r"""Straight line plus combined sine and cosine CRAB pulse ansatz.

    .. math::

       f(t)
       = c_1 (t - T/2)
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
        t: Time samples at which :math:`f(t)` is evaluated.
        duration: Pulse duration :math:`T`.
        params: Array with :math:`4N+1` entries
            :math:`(c_1, A_1, \alpha_1, B_1, \beta_1, \dots, A_N, \alpha_N, B_N, \beta_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    c1 = params[0]
    return c1 * (t - duration / 2.0) + sin_cos_crab(t, duration, params[1:])


def lin_cos_sin_crab(t: jnp.ndarray | float, duration: float, params: jnp.ndarray) -> jnp.ndarray:
    r"""Straight line plus combined cosine and sine CRAB pulse ansatz.

    .. math::

       f(t)
       = c_1 (t - T/2)
       + \sum_{n=1}^N \beta_n
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
        params: Array with :math:`4N+1` entries
            :math:`(c_1, B_1, \beta_1, A_1, \alpha_1, \dots, B_N, \beta_N, A_N, \alpha_N)`.

    Returns:
        Values of :math:`f(t)`.

    """
    c1 = params[0]
    return c1 * (t - duration / 2.0) + cos_sin_crab(t, duration, params[1:])


def softbox_hann(
    t: jnp.ndarray | float,
    duration: float,
    params: jnp.ndarray,
) -> jnp.ndarray:
    r"""Soft-box pulse ansatz with Hann-shaped edges.

    For a pulse of duration :math:`T` and cutoff frequency :math:`\nu_c`,
    the rise/fall time :math:`\tau_e` is chosen as

    .. math::

       \tau_e
       = \min\!\left(
           \frac{T}{2},
           \frac{2}{\nu_c}
         \right),

    which corresponds to placing the first zero of a 2-term Hann window at
    approximately :math:`\nu_c`. The Hann window on :math:`\xi \in [0, 1]`
    is

    .. math::

       w(\xi)
       = a_0 - a_1 \cos(2\pi \xi),

    with :math:`a_0 = 0.5`, :math:`a_1 = 0.5`.
    The pulse ansatz :math:`f(t)` uses the rising and falling halves of
    this window:

    .. math::

       f(t)
       =
       \begin{cases}
         0, &
         t < 0 \ \text{or}\ t > T, \\[4pt]
         A\,w\!\left(\dfrac{t}{2\tau_e}\right), &
         0 \le t < \tau_e, \\[8pt]
         A, &
         \tau_e \le t \le T - \tau_e, \\[8pt]
         A\,w\!\left(
           1 - \dfrac{T - t}{2\tau_e}
         \right), &
         T - \tau_e < t \le T.
       \end{cases}

    Args:
        t:
            Time samples :math:`t` at which :math:`f(t)` is evaluated.
        duration:
            Pulse duration :math:`T`.
        params:
            Array with two entries :math:`(A, \nu_c)`.

    Returns:
        Values of :math:`f(t)`.

    """
    amplitude, cutoff = params
    t = jnp.asarray(t)

    edge_fraction_inv = jnp.maximum(cutoff * duration / 2.0, 2.0)
    edge_duration = duration / edge_fraction_inv

    # 2-term Hann window
    def hann(s: jnp.ndarray) -> jnp.ndarray:
        return 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * s)

    # Determine edge regions
    end_rising_edge = edge_duration
    start_falling_edge = duration - edge_duration

    is_outside = (t < 0.0) | (t > duration)
    is_rising = (t >= 0.0) & (t < end_rising_edge)
    is_falling = (t <= duration) & (t > start_falling_edge)

    # Map physical time to Hann window coordinate
    position_within_rising_edge = 0.5 * t / edge_duration
    position_within_falling_edge = 1.0 - 0.5 * (duration - t) / edge_duration

    # Assemble the pulse
    return amplitude * jnp.select(
        [
            is_outside,
            is_rising,
            is_falling,
        ],
        [
            jnp.zeros_like(t),
            hann(position_within_rising_edge),
            hann(position_within_falling_edge),
        ],
        default=1.0,
    )


def softbox_blackman(
    t: jnp.ndarray | float,
    duration: float,
    params: jnp.ndarray,
) -> jnp.ndarray:
    r"""Soft-box pulse ansatz with Blackman-shaped edges.

    For a pulse of duration :math:`T` and cutoff frequency :math:`\nu_c`,
    the rise/fall time :math:`\tau_e` is chosen as

    .. math::

       \tau_e
       = \min\!\left(
           \frac{T}{2},
           \frac{3}{\nu_c}
         \right),

    which corresponds to placing the first zero of a 3-term Blackman
    window at approximately :math:`\nu_c`. The Blackman window on
    :math:`\xi \in [0, 1]` is

    .. math::

       w(\xi)
       = a_0
         - a_1 \cos(2\pi \xi)
         + a_2 \cos(4\pi \xi),

    with :math:`a_0 = 0.42`, :math:`a_1 = 0.5`, :math:`a_2 = 0.08`.
    The pulse ansatz :math:`f(t)` uses the rising and falling halves of
    this window:

    .. math::

       f(t)
       =
       \begin{cases}
         0, &
         t < 0 \ \text{or}\ t > T, \\[4pt]
         A\,w\!\left(\dfrac{t}{2\tau_e}\right), &
         0 \le t < \tau_e, \\[8pt]
         A, &
         \tau_e \le t \le T - \tau_e, \\[8pt]
         A\,w\!\left(
           1 - \dfrac{T - t}{2\tau_e}
         \right), &
         T - \tau_e < t \le T.
       \end{cases}

    Args:
        t:
            Time samples :math:`t` at which :math:`f(t)` is evaluated.
        duration:
            Pulse duration :math:`T`.
        params:
            Array with two entries :math:`(A, \nu_c)`.

    Returns:
        Values of :math:`f(t)`.

    """
    amplitude, cutoff = params
    t = jnp.asarray(t)

    edge_fraction_inv = jnp.maximum(cutoff * duration / 3.0, 2.0)
    edge_duration = duration / edge_fraction_inv

    # 3-term Blackman window
    def blackman(s: jnp.ndarray) -> jnp.ndarray:
        a0, a1, a2 = 0.42, 0.5, 0.08
        return a0 - a1 * jnp.cos(2.0 * jnp.pi * s) + a2 * jnp.cos(4.0 * jnp.pi * s)

    # Determine edge regions
    end_rising_edge = edge_duration
    start_falling_edge = duration - edge_duration

    is_outside = (t < 0.0) | (t > duration)
    is_rising = (t >= 0.0) & (t < end_rising_edge)
    is_falling = (t <= duration) & (t > start_falling_edge)

    # Map physical time to Blackman window coordinate
    position_within_rising_edge = 0.5 * t / edge_duration
    position_within_falling_edge = 1.0 - 0.5 * (duration - t) / edge_duration

    # Assemble the pulse
    return amplitude * jnp.select(
        [
            is_outside,
            is_rising,
            is_falling,
        ],
        [
            jnp.zeros_like(t),
            blackman(position_within_rising_edge),
            blackman(position_within_falling_edge),
        ],
        default=1.0,
    )


def softbox_nuttall(
    t: jnp.ndarray | float,
    duration: float,
    params: jnp.ndarray,
) -> jnp.ndarray:
    r"""Soft-box pulse ansatz with Nuttall-shaped edges.

    For a pulse of duration :math:`T` and cutoff frequency :math:`\nu_c`, the rise/fall time is chosen as

    .. math::

       \tau_e
       = \min\!\left(
           \frac{T}{2},
           \frac{4}{\nu_c}
         \right),

    which corresponds to placing the first zero of a  4-term Nuttall
    window at approximately :math:`\nu_c`. The Nuttall window on :math:`\xi \in [0, 1]` is

    .. math::

       w(\xi) = a_0
          - a_1 \cos(2\pi \xi)
          + a_2 \cos(4\pi \xi)
          - a_3 \cos(6\pi \xi),

    with :math:`a_0 = 0.355768`, :math:`a_1 = 0.487396`, :math:`a_2 = 0.144232`, :math:`a_3 = 0.012604`.
    The pulse ansatz :math:`f(t)` uses the rising and falling halves of this window:

    .. math::

       f(t)
       =
       \begin{cases}
         0, &
         t < 0 \ \text{or}\ t > T, \\[4pt]
         A\,w\!\left(\dfrac{t}{2\tau_e}\right), &
         0 \le t < \tau_e, \\[8pt]
         A, &
         \tau_e \le t \le T - \tau_e, \\[8pt]
         A\,w\!\left(
           1 - \dfrac{T - t}{2\tau_e}
         \right), &
         T - \tau_e < t \le T.
       \end{cases}

    Args:
        t:
            Time samples :math:`t` at which :math:`f(t)` is evaluated.
        duration:
            Pulse duration :math:`T`.
        params:
            Array with two entries :math:`(A, \nu_c)`.

    Returns:
        Values of :math:`f(t)`.

    """
    amplitude, cutoff = params
    t = jnp.asarray(t)

    edge_fraction_inv = jnp.maximum(cutoff * duration / 4.0, 2.0)
    edge_duration = duration / edge_fraction_inv

    # 4-term Nuttall window
    def nuttall(s: jnp.ndarray) -> jnp.ndarray:
        a0, a1, a2, a3 = 0.355768, 0.487396, 0.144232, 0.012604
        return a0 - a1 * jnp.cos(2.0 * jnp.pi * s) + a2 * jnp.cos(4.0 * jnp.pi * s) - a3 * jnp.cos(6.0 * jnp.pi * s)

    # Determine edge regions
    end_rising_edge = edge_duration
    start_falling_edge = duration - edge_duration

    is_outside = (t < 0.0) | (t > duration)
    is_rising = (t >= 0.0) & (t < end_rising_edge)
    is_falling = (t <= duration) & (t > start_falling_edge)

    # Map physical time to Nuttall window coordinate
    position_within_rising_edge = 0.5 * t / edge_duration
    position_within_falling_edge = 1.0 - 0.5 * (duration - t) / edge_duration

    # Assemble the pulse
    return amplitude * jnp.select(
        [
            is_outside,
            is_rising,
            is_falling,
        ],
        [
            jnp.zeros_like(t),
            nuttall(position_within_rising_edge),
            nuttall(position_within_falling_edge),
        ],
        default=1.0,
    )
