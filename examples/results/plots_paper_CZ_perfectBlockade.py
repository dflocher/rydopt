import numpy as np
import matplotlib.pyplot as plt
from rydopt import pulses_qutip
from rydopt import pulse_postprocessing


plt.rcParams['font.sans-serif'] = "Helvetica"
plt.rcParams['mathtext.fontset'] = "cm"
# plt.rcParams['text.usetex'] = True


# # # pulse params (raw data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# minimal T for CZ with Vnn=inf
params_CZ_T_4 = [7.61140652, 0.0, -0.07842706, 1.80300902, -0.61792703]
params_CZ_T_6 = [7.61130389, 0.0, -0.07643199, 2.09999741, -0.35463916, 0.0, 0.0, -0.62673274, -0.26439621]
params_CZ_T_8 = [7.61139780, 0.0, 0.07037713, 1.02561291, 0.33487771, 0.0, 0.0, -0.63848664, 0.16028477, 0.0, 0.0, -1.43129919, 0.13750356]
params_CZ_T_10 = [7.61143173, 0.0, -0.07799495, 1.76841406, -0.61812355, 0.0, 0.0, 0.40302377, -0.00008756, 0.0, 0.0, 0.09456914, 0.00098466, 0.0, 0.0, 0.24913223, 0.00155822]
params_CZ_T_12 = [7.61151965, 0.0, -0.07456469, 1.61769820, -0.62080959, 0.0, 0.0, 0.54933457, -0.00293115, 0.0, 0.0, 1.14423484, 0.05271353, 0.0, 0.0, 0.28483426, 0.20389992, 0.0, 0.0, -0.20439381, -0.25077449]
params_CZ_T_14 = [7.61179067, 0.0, 0.07951011, 1.12526283, 0.31200413, 0.0, 0.0, 2.73821828, 0.03154222, 0.0, 0.0, -1.83222807, 0.31699602, 0.0, 0.0, 0.97355238, -0.00031638, 0.0, 0.0, -2.26313790, -0.02787016, 0.0, 0.0, 0.37044390, 0.00116649]
params_CZ_T_16 = [7.61151872, 0.0, 0.02478701, -1.25168533, -0.34429190, 0.0, 0.0, -0.52003251, -0.55362641, 0.0, 0.0, 1.46481389, 0.03017798, 0.0, 0.0, -0.15873997, 0.00767558, 0.0, 0.0, -0.31986313, -0.04867624, 0.0, 0.0, 0.83953136, -0.00624906, 0.0, 0.0, -0.90142915, 0.01058957]
params_list_CZ_T = [params_CZ_T_4, params_CZ_T_6, params_CZ_T_8, params_CZ_T_10, params_CZ_T_12, params_CZ_T_14, params_CZ_T_16]

# minimal T for CZ with Vnn=inf, general ansatz:
params_CZ_T_4_gen = [7.61140408, 0.0, -0.07836346, 1.79873998, -0.61795448]
params_CZ_T_6_gen = [7.61138962, 0.0, 0.07908184, 1.86005558, 0.61736170, 1.82610992, -0.00000000]
params_CZ_T_8_gen = [7.61119336, 0.0, -0.05846416, 1.08517677, -0.60672350, 1.59656233, 0.00000000, 0.06872570, -0.04495429]
params_CZ_T_10_gen = [7.61139612, 0.0, -0.07660225, 1.71249683, -0.61955458, -2.17113209, 0.00000000, 1.74767461, -0.00301023, 1.32070648, 0.00000029]
params_CZ_T_12_gen = [7.61139171, 0.0, -0.07907924, 1.88455545, -0.61908942, -1.71230644, 0.00000000, 1.13031446, -0.01027778, 0.76377306, 0.00000003, -0.30116189, 0.00991275]
params_CZ_T_14_gen = [7.61146476, 0.0, 0.07288526, 1.56360383, 0.62280634, -1.24955727, 0.00007022, 1.37449304, 0.00770360, 0.83273728, -0.00039328, 0.74722071, -0.00075235, 1.58921442, 0.00103012]
params_CZ_T_16_gen = [7.61147916, 0.0, 0.06710152, 1.39886785, 0.62745385, 1.16721811, 0.00000152, 4.39161490, -0.16979166, 1.18446162, 0.02865185, -0.02926911, 0.18113673, -0.11362338, -0.02889928, 2.01482718, -0.00107830]
params_list_CZ_T_gen = [params_CZ_T_4_gen, params_CZ_T_6_gen, params_CZ_T_8_gen, params_CZ_T_10_gen, params_CZ_T_12_gen, params_CZ_T_14_gen, params_CZ_T_16_gen]

# minimal TR for CZ with Vnn=inf
params_CZ_TR_4 = [7.70165181, 0.0, -0.12062436, 0.16199017, 0.90407799]
params_CZ_TR_6 = [7.72506187, 0.0, 0.92491109, -0.89119131, -2.91001616, 0.0, 0.0, 0.63210387, -0.08132401]
params_CZ_TR_8 = [7.72399148, 0.0, -0.70056928, -0.38677147, 2.79016092, 0.0, 0.0, -1.48411655, -0.64661463, 0.0, 0.0, -0.36592091, 0.09921779]
params_CZ_TR_10 = [7.71964108, 0.0, -0.32331275, 0.08330442, 1.09251568, 0.0, 0.0, -1.13384773, 0.77701470, 0.0, 0.0, -1.73287209, -0.91221554, 0.0, 0.0, -1.97616094, 0.34517337]
params_CZ_TR_12 = [7.75174554, 0.0, 0.84045514, -1.30666713, -1.21149508, 0.0, 0.0, -2.00708857, -2.10011968, 0.0, 0.0, -1.92511213, 1.13930338, 0.0, 0.0, -1.46531793, -0.55038644, 0.0, 0.0, -2.13843331, 0.13051157]
params_CZ_TR_14 = [7.72953505, 0.0, -0.20939728, 0.54709675, -0.14129762, 0.0, 0.0, -2.04032046, 1.20927094, 0.0, 0.0, 0.97298080, 0.89752870, 0.0, 0.0, -0.23950462, 0.22556171, 0.0, 0.0, -0.34118410, -0.49700683, 0.0, 0.0, -0.84280584, -0.56423927]
params_CZ_TR_16 = [7.73104002, 0.0, -0.49921050, -0.43574431, 1.73133166, 0.0, 0.0, 1.05913764, 0.03635569, 0.0, 0.0, 0.69064806, -0.00694423, 0.0, 0.0, -1.10030748, 0.02372285, 0.0, 0.0, 0.99959916, -0.00584429, 0.0, 0.0, -1.11988727, 0.00612803, 0.0, 0.0, 1.17648502, 0.00996686]
params_list_CZ_TR = [params_CZ_TR_4, params_CZ_TR_6, params_CZ_TR_8, params_CZ_TR_10, params_CZ_TR_12, params_CZ_TR_14, params_CZ_TR_16]

# minimal TR for CZ with Vnn=inf, general ansatz:
params_CZ_TR_4_gen = [7.70165676, 0.0, -0.12063593, 0.16196119, 0.90409933]
params_CZ_TR_6_gen = [7.70163467, 0.0, 0.12058426, 0.16209068, -0.90400403, -1.48199740, 0.00000001]
params_CZ_TR_8_gen = [7.72150637, 0.0, 0.63394636, -0.58448231, -2.07764589, -2.85927057, 0.00000001, 0.68681771, -0.07060518]
params_CZ_TR_10_gen = [7.73024343, 0.0, 0.75979650, -0.70733507, -2.42169206, 2.67068622, 0.16616075, 0.79789195, -0.07966510, -0.55568176, -0.16616075]
params_CZ_TR_12_gen = [7.73526312, 0.0, 0.18543247, 2.30378529, 0.78792053, 0.10766754, 0.77505683, -1.07087065, -1.71779794, -1.79583783, -0.77505682, -0.29368939, -0.10597087]
params_CZ_TR_14_gen = [7.72522373, 0.0, -0.33337989, 1.61399894, -0.43595866, -1.86460265, 0.00265646, -2.20579254, 1.66199128, 1.13810465, 0.00373697, -0.41796223, 0.11617911, 1.18103065, -0.00227704]
params_CZ_TR_16_gen = [7.73863967, 0.0, 0.25388195, 0.06641124, -1.23775226, 1.12093002, 0.50541454, 0.10835437, 0.62200875, -1.25244595, -0.18579895, -0.54543764, -0.54284024, -2.81318833, -0.33920529, -0.83675813, -0.03059769]
params_list_CZ_TR_gen = [params_CZ_TR_4_gen, params_CZ_TR_6_gen, params_CZ_TR_8_gen, params_CZ_TR_10_gen, params_CZ_TR_12_gen, params_CZ_TR_14_gen, params_CZ_TR_16_gen]

# final choice (T-opt, TR-opt):
params_list_CZ = [params_CZ_T_4, params_CZ_TR_6]


# # # fidelities, Rydberg times (processed data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# CZ Vnn=inf, T-optimized
nParams_CZ_T = [4, 6, 8, 10, 12, 14, 16]
Ts_CZ_T = [7.611407, 7.611304, 7.611398, 7.611432, 7.611520, 7.611791, 7.611519]
infidelities_nodecay_CZ_T = [6.2101e-11, 7.5177e-07, 3.3307e-14, 3.4800e-10, 1.1102e-16, 2.1386e-08, 5.9412e-10]
infidelities_CZ_T = [2.9577e-04, 2.9640e-04, 2.9576e-04, 2.9574e-04, 2.9580e-04, 2.9583e-04, 2.9568e-04]
TRs_CZ_T = [2.957832, 2.956630, 2.957317, 2.957590, 2.958133, 2.957805, 2.956995]

# CZ Vnn=inf, T-optimized, general ansatz
nParams_CZ_T_gen = [4, 6, 8, 10, 12, 14, 16]
Ts_CZ_T_gen = [7.611404, 7.611390, 7.611193, 7.611396, 7.611392, 7.611465, 7.611479]
infidelities_nodecay_CZ_T_gen = [1.1050e-09, 7.8353e-08, 6.7053e-06, 2.6645e-15, 4.4409e-16, 3.2196e-15, 2.2204e-16]
infidelities_CZ_T_gen = [2.9576e-04, 2.9587e-04, 3.0226e-04, 2.9574e-04, 2.9577e-04, 2.9569e-04, 2.9583e-04]
TRs_CZ_T_gen = [2.957790, 2.957617, 2.955706, 2.957523, 2.957835, 2.956570, 2.958013]

# CZ Vnn=inf, TR-optimized
nParams_CZ_TR = [4, 6, 8, 10, 12, 14, 16]
Ts_CZ_TR = [7.701652, 7.725062, 7.723991, 7.719641, 7.751746, 7.729535, 7.731040]
infidelities_nodecay_CZ_TR = [4.2939e-09, 4.2211e-09, 4.2866e-09, 4.1711e-09, 4.0496e-09, 3.8738e-09, 4.1585e-09]
infidelities_CZ_TR = [2.9375e-04, 2.9354e-04, 2.9354e-04, 2.9354e-04, 2.9355e-04, 2.9355e-04, 2.9356e-04]
TRs_CZ_TR = [2.937154, 2.935531, 2.935004, 2.935044, 2.935633, 2.935135, 2.935187]

# CZ Vnn=inf, TR-optimized, general ansatz
nParams_CZ_TR_gen = [4, 6, 8, 10, 12, 14, 16]
Ts_CZ_TR_gen = [7.701657, 7.701635, 7.721506, 7.730243, 7.735263, 7.725224, 7.738640]
infidelities_nodecay_CZ_TR_gen = [4.2939e-09, 4.2937e-09, 4.1794e-09, 4.2404e-09, 4.0865e-09, 4.1053e-09, 4.1271e-09]
infidelities_CZ_TR_gen = [2.9375e-04, 2.9375e-04, 2.9354e-04, 2.9354e-04, 2.9354e-04, 2.9354e-04, 2.9355e-04]
TRs_CZ_TR_gen = [2.937154, 2.937687, 2.935569, 2.935552, 2.935594, 2.935034, 2.935623]

# final choice (T-opt, TR-opt):
nParams_CZ = [4, 6]
Ts_CZ = [7.611407, 7.725062]
infidelities_nodecay_CZ = [6.2101e-11, 4.2211e-09]
infidelities_CZ = [2.9577e-04, 2.9354e-04]
TRs_CZ = [2.957832, 2.935531]


# # # functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def plot_pulses(pulse, params_list):
    colors = ['tab:blue', 'tab:red']
    linestyles = ['solid', 'dotted']
    label_list = [r'$T$-opt.', r'$T_R$-opt.']
    if isinstance(pulse, str):
        pulse = pulses_qutip.get_pulse(pulse)
    fig, ax = plt.subplots(layout='constrained', figsize=(3.4, 1.4))
    for i, params in enumerate(params_list):
        params = np.array(params)
        T = params[0]
        Delta_of_t, phi_of_t = pulse(params)
        ts = np.linspace(0, T, 1000)
        phis = phi_of_t(ts)
        phis -= phis[0]
        ax.plot(ts, phis / (2 * np.pi), linewidth=2, color=colors[i], linestyle=linestyles[i], label=label_list[i])
    ax.set_xlabel(r'$\Omega_0 t$', fontsize=10)
    ax.set_ylabel(r'$\xi/(2\pi)$', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xlim(-0.3, 8.0)
    ax.set_ylim(-0.1, 0.31)
    ax.legend(fontsize=8, handlelength=2.2, loc='upper left', bbox_to_anchor=(0.53, 1.0))
    ax.grid()
    plt.show()
    fig.savefig('Figs/CZ_pulses.pdf', bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)

    # number of atoms participating in the gate (2, 3 or 4)
    n_atoms = 2

    # Rydberg interaction strengths
    Vnn = float("inf")
    Vnnn = float("inf")
    decay = 0.0001

    # target gate phases
    theta = np.pi
    eps = 0
    lamb = 0
    delta = 0
    kappa = 0

    # pulse type
    pulse = pulses_qutip.pulse_phase_sin_cos_crab_smooth

    # optimization parameters
    params_list = params_list_CZ

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # pulse_postprocessing.postprocess_pulses(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params_list, decay)
    plot_pulses(pulse, params_list_CZ)
