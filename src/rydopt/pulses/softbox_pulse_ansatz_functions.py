import jax.nn as jnn
import jax.numpy as jnp


def softbox_hann(
    t: jnp.ndarray | float,
    duration: float,
    ansatz_params: jnp.ndarray,
) -> jnp.ndarray:
    r"""Soft-box pulse ansatz with Hann-shaped edges, also known as Tukey window.

    The Hann window on :math:`\xi \in [0, 1]` is

    .. math::

       w(\xi) = a_0 - a_1 \cos(2\pi \xi),

    with :math:`a_0 = 0.5`, :math:`a_1 = 0.5`.
    The pulse ansatz :math:`f(t)` uses the rising and falling halves of
    this window:

    .. math::

       f(t)
       =
       \begin{cases}
         0, &
         t < 0 \ \text{or}\ t > T, \\[4pt]
         A\,w\!\left(\dfrac{t}{\alpha T}\right), &
         0 \le t < \alpha T / 2, \\[8pt]
         A, &
         \alpha T / 2 \le t \le T - \alpha T / 2, \\[8pt]
         A\,w\!\left(
           1 - \dfrac{T - t}{\alpha T}
         \right), &
         T - \alpha T / 2 < t \le T.
       \end{cases}

    Args:
        t:
            Time samples :math:`t` at which :math:`f(t)` is evaluated.
        duration:
            Pulse duration :math:`T`.
        ansatz_params:
            Array with two entries :math:`(A, \alpha)`.

    Returns:
        Values of :math:`f(t)`.

    """
    amplitude, alpha = ansatz_params
    alpha = jnp.clip(alpha, 0.0, 1.0)
    t = jnp.asarray(t)

    edge_duration = duration * alpha / 2.0

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
    ansatz_params: jnp.ndarray,
) -> jnp.ndarray:
    r"""Soft-box pulse ansatz with Blackman-shaped edges.

    The Blackman window on :math:`\xi \in [0, 1]` is

    .. math::

       w(\xi) = a_0 - a_1 \cos(2\pi \xi) + a_2 \cos(4\pi \xi),

    with :math:`a_0 = 0.42`, :math:`a_1 = 0.5`, :math:`a_2 = 0.08`.
    The pulse ansatz :math:`f(t)` uses the rising and falling halves of
    this window:

    .. math::

       f(t)
       =
       \begin{cases}
         0, &
         t < 0 \ \text{or}\ t > T, \\[4pt]
         A\,w\!\left(\dfrac{t}{\alpha T}\right), &
         0 \le t < \alpha T / 2, \\[8pt]
         A, &
         \alpha T / 2 \le t \le T - \alpha T / 2, \\[8pt]
         A\,w\!\left(
           1 - \dfrac{T - t}{\alpha T}
         \right), &
         T - \alpha T / 2 < t \le T.
       \end{cases}

    Args:
        t:
            Time samples :math:`t` at which :math:`f(t)` is evaluated.
        duration:
            Pulse duration :math:`T`.
        ansatz_params:
            Array with two entries :math:`(A, \alpha)`.

    Returns:
        Values of :math:`f(t)`.

    """
    amplitude, alpha = ansatz_params
    alpha = jnp.clip(alpha, 0.0, 1.0)
    t = jnp.asarray(t)

    edge_duration = duration * alpha / 2.0

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
    ansatz_params: jnp.ndarray,
) -> jnp.ndarray:
    r"""Soft-box pulse ansatz with Nuttall-shaped edges.

    The Nuttall window on :math:`\xi \in [0, 1]` is

    .. math::

       w(\xi) = a_0 - a_1 \cos(2\pi \xi) + a_2 \cos(4\pi \xi) - a_3 \cos(6\pi \xi),

    with :math:`a_0 = 0.355768`, :math:`a_1 = 0.487396`, :math:`a_2 = 0.144232`, :math:`a_3 = 0.012604`.
    The pulse ansatz :math:`f(t)` uses the rising and falling halves of this window:

    .. math::

       f(t)
       =
       \begin{cases}
         0, &
         t < 0 \ \text{or}\ t > T, \\[4pt]
         A\,w\!\left(\dfrac{t}{\alpha T}\right), &
         0 \le t < \alpha T / 2, \\[8pt]
         A, &
         \alpha T / 2 \le t \le T - \alpha T / 2, \\[8pt]
         A\,w\!\left(
           1 - \dfrac{T - t}{\alpha T}
         \right), &
         T - \alpha T / 2 < t \le T.
       \end{cases}

    Args:
        t:
            Time samples :math:`t` at which :math:`f(t)` is evaluated.
        duration:
            Pulse duration :math:`T`.
        ansatz_params:
            Array with two entries :math:`(A, \alpha)`.

    Returns:
        Values of :math:`f(t)`.

    """
    amplitude, alpha = ansatz_params
    alpha = jnp.clip(alpha, 0.0, 1.0)
    t = jnp.asarray(t)

    edge_duration = duration * alpha / 2.0

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


def softbox_planck(
    t: jnp.ndarray | float,
    duration: float,
    ansatz_params: jnp.ndarray,
) -> jnp.ndarray:
    r"""Planck-taper window.

    The Planck-taper on :math:`\xi \in (0, 1)` is

    .. math::

       w(\xi)
       =
       \frac{1}{
         \exp\!\left(
           \frac{1}{\xi}
           - \frac{1}{1 - \xi}
         \right) + 1
       },

    with :math:`w(0)=0` and :math:`w(1)=1` by continuity.
    The pulse ansatz :math:`f(t)` uses a rising and a falling Planck taper:

    .. math::

       f(t)
       =
       \begin{cases}
         0, &
         t < 0 \ \text{or}\ t > T, \\[4pt]
         A\,w\!\left(\dfrac{t}{\alpha T/2}\right), &
         0 \le t < \alpha T / 2, \\[8pt]
         A, &
         \alpha T / 2 \le t \le T - \alpha T / 2, \\[8pt]
         A\,w\!\left(
           \dfrac{T - t}{\alpha T/2}
         \right), &
         T - \alpha T / 2 < t \le T.
       \end{cases}

    Args:
        t:
            Time samples :math:`t` at which :math:`f(t)` is evaluated.
        duration:
            Pulse duration :math:`T`.
        ansatz_params:
            Array with two entries :math:`(A, \alpha)`.

    Returns:
        Values of :math:`f(t)`.

    """
    amplitude, alpha = ansatz_params
    alpha = jnp.clip(alpha, 0.0, 1.0)
    t = jnp.asarray(t)

    edge_duration = duration * alpha / 2.0

    # Planck-taper
    def planck_taper(s: jnp.ndarray) -> jnp.ndarray:
        t = (2.0 * s - 1.0) / (s * (1.0 - s))
        return jnn.sigmoid(t)

    # Determine edge regions
    end_rising_edge = edge_duration
    start_falling_edge = duration - edge_duration

    is_outside = (t < 0.0) | (t > duration)
    is_rising = (t >= 0.0) & (t < end_rising_edge)
    is_falling = (t <= duration) & (t > start_falling_edge)

    # Map physical time to Planck-taper coordinate
    position_within_rising_edge = t / edge_duration
    position_within_falling_edge = (duration - t) / edge_duration

    # Assemble the pulse
    return amplitude * jnp.select(
        [
            is_outside,
            is_rising,
            is_falling,
        ],
        [
            jnp.zeros_like(t),
            planck_taper(position_within_rising_edge),
            planck_taper(position_within_falling_edge),
        ],
        default=1.0,
    )


def softbox_fifth_order_smoothstep(
    t: jnp.ndarray | float,
    duration: float,
    ansatz_params: jnp.ndarray,
) -> jnp.ndarray:
    r"""Soft-box pulse ansatz with 5th-order-smoothstep-shaped edges.

    The 5th-order smoothstep on :math:`\xi \in [0, 1]` is

    .. math::

       w(\xi) = 6\xi^5 - 15\xi^4 + 10\xi^3,

    which interpolates smoothly from 0 to 1 with vanishing first and
    second derivatives at both endpoints.
    The pulse ansatz :math:`f(t)` uses a rising and a falling smoothstep:

    .. math::

       f(t)
       =
       \begin{cases}
         0, &
         t < 0 \ \text{or}\ t > T, \\[4pt]
         A\,w\!\left(\dfrac{t}{\alpha T/2}\right), &
         0 \le t < \alpha T / 2, \\[8pt]
         A, &
         \alpha T / 2 \le t \le T - \alpha T / 2, \\[8pt]
         A\,w\!\left(
           \dfrac{T - t}{\alpha T/2}
         \right), &
         T - \alpha T / 2 < t \le T.
       \end{cases}

    Args:
        t:
            Time samples :math:`t` at which :math:`f(t)` is evaluated.
        duration:
            Pulse duration :math:`T`.
        ansatz_params:
            Array with two entries :math:`(A, \alpha)`.

    Returns:
        Values of :math:`f(t)`.

    """
    amplitude, alpha = ansatz_params
    alpha = jnp.clip(alpha, 0.0, 1.0)
    t = jnp.asarray(t)

    edge_duration = duration * alpha / 2.0

    # 5th-order smoothstep
    def quintic_smoothstep(s: jnp.ndarray) -> jnp.ndarray:
        return 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3

    # Determine edge regions
    end_rising_edge = edge_duration
    start_falling_edge = duration - edge_duration

    is_outside = (t < 0.0) | (t > duration)
    is_rising = (t >= 0.0) & (t < end_rising_edge)
    is_falling = (t <= duration) & (t > start_falling_edge)

    # Map physical time to quintic smoothstep coordinate
    position_within_rising_edge = t / edge_duration
    position_within_falling_edge = (duration - t) / edge_duration

    # Assemble the pulse
    return amplitude * jnp.select(
        [
            is_outside,
            is_rising,
            is_falling,
        ],
        [
            jnp.zeros_like(t),
            quintic_smoothstep(position_within_rising_edge),
            quintic_smoothstep(position_within_falling_edge),
        ],
        default=1.0,
    )


def softbox_seventh_order_smoothstep(
    t: jnp.ndarray | float,
    duration: float,
    ansatz_params: jnp.ndarray,
) -> jnp.ndarray:
    r"""Soft-box pulse ansatz with 7th-order-smoothstep-shaped edges.

    The 7th-order smoothstep :math:`S_3` on :math:`\xi \in [0, 1]` is

    .. math::

       w(\xi) = -20\xi^7 + 70\xi^6 - 84\xi^5 + 35\xi^4,

    which interpolates smoothly from 0 to 1 with vanishing derivatives
    up to third order at both endpoints.
    The pulse ansatz :math:`f(t)` uses a rising and a falling smoothstep:

    .. math::

       f(t)
       =
       \begin{cases}
         0, &
         t < 0 \ \text{or}\ t > T, \\[4pt]
         A\,w\!\left(\dfrac{t}{\alpha T/2}\right), &
         0 \le t < \alpha T / 2, \\[8pt]
         A, &
         \alpha T / 2 \le t \le T - \alpha T / 2, \\[8pt]
         A\,w\!\left(
           \dfrac{T - t}{\alpha T/2}
         \right), &
         T - \alpha T / 2 < t \le T.
       \end{cases}

    Args:
        t:
            Time samples :math:`t` at which :math:`f(t)` is evaluated.
        duration:
            Pulse duration :math:`T`.
        ansatz_params:
            Array with two entries :math:`(A, \alpha)`.

    Returns:
        Values of :math:`f(t)`.

    """
    amplitude, alpha = ansatz_params
    alpha = jnp.clip(alpha, 0.0, 1.0)
    t = jnp.asarray(t)

    edge_duration = duration * alpha / 2.0

    # 7th-order smoothstep
    def seventh_order_smoothstep(s: jnp.ndarray) -> jnp.ndarray:
        return -20.0 * s**7 + 70.0 * s**6 - 84.0 * s**5 + 35.0 * s**4

    # Determine edge regions
    end_rising_edge = edge_duration
    start_falling_edge = duration - edge_duration

    is_outside = (t < 0.0) | (t > duration)
    is_rising = (t >= 0.0) & (t < end_rising_edge)
    is_falling = (t <= duration) & (t > start_falling_edge)

    # Map physical time to smoothstep coordinate
    position_within_rising_edge = t / edge_duration
    position_within_falling_edge = (duration - t) / edge_duration

    # Assemble the pulse
    return amplitude * jnp.select(
        [
            is_outside,
            is_rising,
            is_falling,
        ],
        [
            jnp.zeros_like(t),
            seventh_order_smoothstep(position_within_rising_edge),
            seventh_order_smoothstep(position_within_falling_edge),
        ],
        default=1.0,
    )
