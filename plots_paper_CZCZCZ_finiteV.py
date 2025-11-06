import numpy as np
import matplotlib.pyplot as plt
import pulses_qutip
import pulse_postprocessing
from matplotlib.legend_handler import HandlerTuple


plt.rcParams['font.sans-serif'] = "Helvetica"
plt.rcParams['mathtext.fontset'] = "cm"
# plt.rcParams['text.usetex'] = True


# # # pulse params (raw data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# eps-CZ-CZ-CZ Vnn=32, Vnnn=[0.5, 1, 2, 4, 8, 16, 32], nParams=12, TR-optimized
params_epsCZCZCZ_TR_12_32_05 = [26.33930123, 0.0, 0.72339294, 1.17764694, 4.26310469, 0.0, 0.0, -0.87843102, -4.38863501, 0.0, 0.0, -0.91293983, -0.04443779, 0.0, 0.0, 2.46506652, -0.07381435, 0.0, 0.0, 0.87367360, 0.31824909]
params_epsCZCZCZ_TR_12_32_1 = [25.62323460, 0.0, 0.58931641, 0.63813759, -1.10554605, 0.0, 0.0, 2.15718052, -0.18753103, 0.0, 0.0, -1.12966547, 0.54860449, 0.0, 0.0, 0.14967435, 0.05761077, 0.0, 0.0, 0.81885942, 0.31364730]
params_epsCZCZCZ_TR_12_32_2 = [22.68049668, 0.0, 0.55116180, 0.19924726, 2.37683483, 0.0, 0.0, -1.05386013, -3.68057959, 0.0, 0.0, -1.82230006, 0.47872910, 0.0, 0.0, 1.37374013, -0.32808932, 0.0, 0.0, 1.61587465, 0.12842998]
params_epsCZCZCZ_TR_12_32_4 = [21.94065239, 0.0, 0.60052441, 0.15892761, -1.64354885, 0.0, 0.0, 1.48566021, -0.32689954, 0.0, 0.0, -0.18072491, 0.39423590, 0.0, 0.0, 1.09039727, -0.32269936, 0.0, 0.0, 1.90912185, 0.10450935]
params_epsCZCZCZ_TR_12_32_8 = [20.78367380, 0.0, -1.16009300, -1.16331923, 4.94902019, 0.0, 0.0, -1.23044108, 0.68381102, 0.0, 0.0, 1.00093455, 0.05977036, 0.0, 0.0, 1.21549623, 0.31220067, 0.0, 0.0, 0.90191209, -0.15665491]
params_epsCZCZCZ_TR_12_32_16 = [20.56907177, 0.0, -0.72712347, 0.12661388, 2.33841364, 0.0, 0.0, 1.11517453, -0.06872300, 0.0, 0.0, 1.20775442, 0.11789557, 0.0, 0.0, 2.21596453, 0.45830470, 0.0, 0.0, 0.68545315, -0.33046103]
params_epsCZCZCZ_TR_12_32_32 = [18.44225031, 0.0, -0.40260681, 1.58033144, 0.88621262, 0.0, 0.0, 0.53176586, -0.44758565, 0.0, 0.0, 3.26093558, -0.49338958, 0.0, 0.0, 1.17079694, 0.04730889, 0.0, 0.0, -0.56139440, 0.30767581]
params_list_CZCZCZ_TR_12_32 = [params_epsCZCZCZ_TR_12_32_05, params_epsCZCZCZ_TR_12_32_1, params_epsCZCZCZ_TR_12_32_2, params_epsCZCZCZ_TR_12_32_4, params_epsCZCZCZ_TR_12_32_8, params_epsCZCZCZ_TR_12_32_16, params_epsCZCZCZ_TR_12_32_32]


# # # fidelities, Rydberg times (processed data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# eps-CZ-CZ-CZ Vnn=32, Vnnn=[0.5, 1, 2, 4, 8, 16, 32], nParams=12, TR-optimized
Vnnns_CZCZCZ_TR_12_32 = [0.5, 1, 2, 4, 8, 16, 32]
Ts_CZCZCZ_TR_12_32 = [26.339301, 25.623235, 22.680497, 21.940652, 20.783674, 20.569072, 18.442250]
infidelities_nodecay_CZCZCZ_TR_12_32 = [1.3755e-06, 4.0219e-07, 8.9060e-07, 2.9989e-06, 1.6065e-06, 3.8701e-06, 2.4316e-07]
infidelities_CZCZCZ_TR_12_32 = [8.5849e-04, 8.0491e-04, 7.3209e-04, 6.7356e-04, 6.1689e-04, 6.1184e-04, 6.0192e-04]
TRs_CZCZCZ_TR_12_32 = [8.573747, 8.047100, 7.313582, 6.706817, 6.153731, 6.080565, 6.017614]
eps_CZCZCZ_TR_12_32 = [-0.73611, -1.30748, -1.78177, -2.43861, -2.70196, -3.00922, -3.14153]

# # # functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def plot_TR_vs_Vnnn(Vnnns_list, TRs_list, infidelities_list, eps_list):
    fig, ax = plt.subplots(layout='constrained', figsize=(3.4, 1.9))
    ax2 = ax.twinx()
    color = 'tab:blue'
    color2 = 'tab:red'
    infid, = ax.semilogx(Vnnns_list, infidelities_list, 'o', markersize=5, color=color, markerfacecolor='none', markeredgewidth=1.5)
    tr, = ax2.semilogx(Vnnns_list, TRs_list, '+', markersize=5, color=color2)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position("left")
    ax.set_xlabel(r'$V_{\mathrm{nnn}}/(\hbar \Omega_0)$', fontsize=10)
    ax.set_ylabel(r'$1-F$', fontsize=10, color=color)
    ax2.set_ylabel(r'$\Omega_0 T_R$', fontsize=10, color=color2)
    ax.tick_params(axis='both', labelsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax.set_xlim(0.4, 40)
    ax.set_ylim(0.00058, 0.0009)
    ax2.set_ylim(5.8, 9.0)
    ax.ticklabel_format(style='sci', scilimits=(-4, -4), axis='y', useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.legend([(tr, infid)], [r'CZ-CZ-CZ($\epsilon$)'], numpoints=1,
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=8, handlelength=2.2, loc='lower left')
    axin = ax.inset_axes([0.675, 0.6, 0.3, 0.35])
    eps_list = np.mod(np.array(eps_list), 2*np.pi)
    axin.semilogx(Vnnns_list, eps_list, 'x', markersize=4, color='gray', markerfacecolor='none')
    axin.set_xlabel(r'$V_{\mathrm{nnn}}/(\hbar \Omega_0)$', fontsize=10, labelpad=-1)
    axin.set_ylabel(r'$\epsilon$', fontsize=10, labelpad=-1)
    axin.tick_params(axis='both', labelsize=8)
    axin.set_xlim(0.32, 47)
    axin.set_ylim(0.8 * np.pi, 2 * np.pi)
    axin.set_xticks([1, 10])
    axin.set_xticklabels(['', ''])
    axin.set_yticks([np.pi, 2 * np.pi])
    axin.set_yticklabels([r'$\pi$', r'$2\pi$'])
    ax.grid()
    plt.show()
    fig.savefig('Figs/CZCZCZ_infidelity_and_TR_vs_Vnnn_params12.pdf', bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)

    # number of atoms participating in the gate (2, 3 or 4)
    n_atoms = 3

    # Rydberg interaction strengths
    Vnn = 32.0
    Vnnn = None
    decay = 0.0001

    # target gate phases
    theta = np.pi
    eps = None
    lamb = 0
    delta = 0
    kappa = 0

    # pulse type
    pulse = pulses_qutip.pulse_phase_sin_cos_crab_smooth

    # optimization parameters
    params_list = params_list_CZCZCZ_TR_12_32
    Vnnn_list = Vnnns_CZCZCZ_TR_12_32

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # pulse_postprocessing.postprocess_pulses(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params_list, decay, Vnnn_list)
    plot_TR_vs_Vnnn(Vnnns_CZCZCZ_TR_12_32, TRs_CZCZCZ_TR_12_32, infidelities_CZCZCZ_TR_12_32, eps_CZCZCZ_TR_12_32)
