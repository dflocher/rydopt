import numpy as np
import matplotlib.pyplot as plt
from rydopt import pulses_qutip
from rydopt import pulse_postprocessing
from matplotlib.legend_handler import HandlerTuple


plt.rcParams['font.sans-serif'] = "Helvetica"
plt.rcParams['mathtext.fontset'] = "cm"
# plt.rcParams['text.usetex'] = True


# # # pulse params (raw data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# minimal T for CCCZ' gates with Vnn=Vnnn=inf
params_CCCZ_T_8 = [14.14222223, 0.0, 0.16954382, 0.31647908, -0.75670150, 0.0, 0.0, 0.39053196, 1.30626162, 0.0, 0.0, 0.80503678, 0.57628361]
params_CCCZ_T_10 = [12.42032121, 0.0, 0.09371255, 1.02359835, -0.99945357, 0.0, 0.0, 1.96361795, -0.80000698, 0.0, 0.0, 0.74691915, -0.18123922, 0.0, 0.0, 0.70514304, 0.32656123]
params_CCCZ_T_12 = [12.42212917, 0.0, -0.13894700, 0.47510484, 1.02659969, 0.0, 0.0, 1.83680730, 0.22543606, 0.0, 0.0, 0.38868304, 0.39112548, 0.0, 0.0, 0.71932870, -0.31079664, 0.0, 0.0, -2.13803362, 0.45923459]
params_CCCZ_T_14 = [12.42026047, 0.0, -0.04232968, 0.39596865, 0.10377197, 0.0, 0.0, -0.39985662, 0.88202103, 0.0, 0.0, 2.82113660, -0.22136531, 0.0, 0.0, 0.34029841, 0.30629183, 0.0, 0.0, 0.09362225, -0.38687698, 0.0, 0.0, -1.72560345, 0.83072330]
params_CCCZ_T_16 = [12.42936966, 0.0, 0.41688619, -0.24302614, -1.88950979, 0.0, 0.0, 1.98365106, -0.81828647, 0.0, 0.0, -0.48980686, -0.24038075, 0.0, 0.0, -0.02566397, -0.28318901, 0.0, 0.0, -0.85496320, 0.11820058, 0.0, 0.0, -0.97737769, 0.05460315, 0.0, 0.0, -0.56889252, 0.34448845]
params_CCCZ_T_18 = [12.42537989, 0.0, 0.06111361, 1.74019966, -0.93828213, 0.0, 0.0, 1.90319392, -0.73834459, 0.0, 0.0, 0.72408178, -0.02906016, 0.0, 0.0, 0.93795983, 0.08238358, 0.0, 0.0, -0.47812107, -0.22489288, 0.0, 0.0, -0.86951121, 0.01459324, 0.0, 0.0, -0.62571135, 0.25908187, 0.0, 0.0, 1.13701314, -0.02230908]
params_CCCZ_T_20 = [12.46171139, 0.0, 0.01489662, 0.63472579, 0.07701216, 0.0, 0.0, -0.14406498, 0.88894165, 0.0, 0.0, 0.30224122, 0.52631501, 0.0, 0.0, -0.08944408, 0.08725757, 0.0, 0.0, -0.94641464, 0.19033835, 0.0, 0.0, -0.30582931, -0.30567209, 0.0, 0.0, -1.38794647, -0.04867837, 0.0, 0.0, 1.48608995, -0.05628373, 0.0, 0.0, 1.05062618, -0.00826354]
params_list_CCCZ_T = [params_CCCZ_T_8, params_CCCZ_T_10, params_CCCZ_T_12, params_CCCZ_T_14, params_CCCZ_T_16, params_CCCZ_T_18, params_CCCZ_T_20]

# minimal T for CCCZ' gates with Vnn=Vnnn=inf, general ansatz:
params_CCCZ_T_8_gen = [24.69240120, 0.0, 2.47621667, -0.21473245, 8.42977018, 0.98093871, 0.00000001, 4.22483949, 3.18755869]
params_CCCZ_T_10_gen = [19.05492122, 0.0, 0.93147016, -0.32095178, 9.32272812, -0.37899547, -0.00000003, 1.06903137, 1.45784218, 1.87763507, -0.00000008]
params_CCCZ_T_12_gen = [14.14350858, 0.0, 0.33563795, -0.42947587, -1.37783795, -1.14991018, 0.00000001, 0.43358038, 1.23786146, -1.92104945, 0.00000000, 0.81431624, 0.56886201]
params_CCCZ_T_14_gen = [12.90101399, 0.0, -0.72493392, 0.27702902, 0.27831625, 1.31260160, 2.48557524, 0.93608084, 1.09297468, 0.30089388, 0.38804592, 1.70877974, -0.22086581, 0.36557240, -0.42939585]
params_CCCZ_T_16_gen = [11.85491487, 0.0, 0.05145775, 1.24899612, -0.39873313, -0.45454278, -0.19956321, 1.91758258, -0.55811841, -0.18870230, 0.96109818, 1.93498434, 0.22047354, 0.27605323, -0.26273264, 3.92960225, -0.07183359]
params_CCCZ_T_18_gen = [11.80271325, 0.0, -0.13332200, 0.66983248, 1.31476492, 2.81147759, -1.58457206, -0.36210033, -1.57467284, -2.17781176, 1.09432979, 0.41649028, -0.64527764, 0.47012471, 0.19164886, 0.14593750, 0.51086808, 1.08196100, -0.06506773]
params_CCCZ_T_20_gen = [11.80656110, 0.0, 0.48163012, -0.87790444, -1.94864872, 1.39605145, 0.29969663, -0.77322549, 0.00168774, 0.15237996, -0.17407943, -0.13942584, -0.63142582, -1.24377526, -0.97589001, 0.39182751, 0.24188525, -0.35511263, 0.27333326, 0.25910999, -0.02733137]
params_list_CCCZ_T_gen = [params_CCCZ_T_8_gen, params_CCCZ_T_10_gen, params_CCCZ_T_12_gen, params_CCCZ_T_14_gen, params_CCCZ_T_16_gen, params_CCCZ_T_18_gen, params_CCCZ_T_20_gen]

# minimal TR for CCCZ' gates with Vnn=Vnnn=inf
params_CCCZ_TR_8 = [16.65290952, 0.0, 0.51149711, 0.39284022, 0.83438831, 0.0, 0.0, 2.18045988, -1.98696104, 0.0, 0.0, 1.46237604, -1.63719223]
params_CCCZ_TR_10 = [15.53330985, 0.0, 0.90293399, 1.31406533, -2.10907137, 0.0, 0.0, 0.67566147, -1.50262388, 0.0, 0.0, 4.33665536, -0.87190133, 0.0, 0.0, -0.15749697, 1.81642794]
params_CCCZ_TR_12 = [13.52993987, 0.0, 0.28667059, -0.77973140, -1.21127062, 0.0, 0.0, 0.18919636, 1.10162108, 0.0, 0.0, 0.57159922, -0.45684419, 0.0, 0.0, 1.49576081, -0.56710162, 0.0, 0.0, 0.17771504, 0.71128826]
params_CCCZ_TR_14 = [15.64882230, 0.0, 0.62344592, -1.21309950, 0.30395321, 0.0, 0.0, 3.65589621, 0.07086844, 0.0, 0.0, -0.90823921, -2.60857295, 0.0, 0.0, -0.12491227, 0.98704073, 0.0, 0.0, 0.31508103, 0.34077972, 0.0, 0.0, -0.35948449, -0.80274892]
params_CCCZ_TR_16 = [15.58452752, 0.0, -0.63607639, 0.13606902, -0.66968509, 0.0, 0.0, 0.42178041, 0.41789233, 0.0, 0.0, -1.19117687, 2.61595262, 0.0, 0.0, 0.44674778, 0.78937125, 0.0, 0.0, -0.57501510, -2.28901100, 0.0, 0.0, -1.07556933, 1.15399762, 0.0, 0.0, -0.36888388, -0.22300015]
params_CCCZ_TR_18 = [15.58154664, 0.0, 0.47769773, -0.66548956, 1.37783579, 0.0, 0.0, -1.03966930, -0.45271256, 0.0, 0.0, -0.99015533, -2.50756823, 0.0, 0.0, 2.24366829, 0.29266026, 0.0, 0.0, -0.08106135, -0.83510494, 0.0, 0.0, -0.94880800, 1.10052373, 0.0, 0.0, 0.82550139, 0.07192769, 0.0, 0.0, 0.79227309, -0.01741830]
params_CCCZ_TR_20 = [15.54738915, 0.0, -0.59738777, -2.33208426, -0.74123663, 0.0, 0.0, 0.46580393, 0.84247858, 0.0, 0.0, -1.23111096, 2.06009539, 0.0, 0.0, -0.64668969, 0.03711254, 0.0, 0.0, -0.67815650, -1.02612763, 0.0, 0.0, -0.94397072, -0.17354865, 0.0, 0.0, -0.67186841, 0.65167031, 0.0, 0.0, -0.56847506, -0.23194485, 0.0, 0.0, 1.98616762, -0.03381819]
params_list_CCCZ_TR = [params_CCCZ_TR_8, params_CCCZ_TR_10, params_CCCZ_TR_12, params_CCCZ_TR_14, params_CCCZ_TR_16, params_CCCZ_TR_18, params_CCCZ_TR_20]

# minimal TR for CCCZ' gates with Vnn=Vnnn=inf, general ansatz:
params_CCCZ_TR_8_gen = [13.39712895, 0.0, 1.31357055, 1.35271050, 4.65876106, 0.35549306, -0.00000001, 1.86624889, 1.67828811]
params_CCCZ_TR_10_gen = [16.45707518, 0.0, -1.43384498, -0.33316817, 7.21331668, 0.22639652, -4.08288342, 2.79105916, -0.50433499, -0.01913534, 2.81707519]
params_CCCZ_TR_12_gen = [15.62898356, 0.0, 0.46244783, 0.92761398, -0.51822062, -0.06460151, 2.17761799, -0.27113165, -0.78124219, 0.24690591, 0.49930696, 1.23157906, 0.87657606]
params_CCCZ_TR_14_gen = [14.21687617, 0.0, 0.15295947, 0.66796621, -1.34809266, -0.23174694, 0.27647582, 0.06929205, 0.55932007, 0.73868305, 0.84614403, 1.16644997, 0.42287888, 0.71489423, 0.39752556]
params_CCCZ_TR_16_gen = [12.46389302, 0.0, -0.15030744, 0.67620909, 1.16284016, -0.65119418, 0.00000137, 2.44758824, 0.60733473, -0.22922469, -0.00000477, 0.10939588, 0.30585253, -0.36314963, -0.00000140, 0.96960348, -0.31831616]
params_CCCZ_TR_18_gen = [12.46704527, 0.0, 0.11245508, 1.11709718, -1.13023953, 1.15630929, -0.11911817, 2.65986645, -1.11383094, -0.60235853, 0.18841184, -0.28143727, 0.27648408, -1.99099622, -0.07141728, 0.99414820, 0.31862150, -1.68168342, 0.00157473]
params_CCCZ_TR_20_gen = [12.53895000, 0.0, -0.40034499, -0.03928352, 1.89086202, 0.62441255, -0.00119158, 1.68910321, 0.90863396, -0.29015518, 0.00284550, 0.24063609, 0.06332820, 1.99322612, -0.00016889, 1.05581642, -0.34727004, 0.82055797, 0.00039086, 2.22237168, -0.04102438]
params_list_CCCZ_TR_gen = [params_CCCZ_TR_8_gen, params_CCCZ_TR_10_gen, params_CCCZ_TR_12_gen, params_CCCZ_TR_14_gen, params_CCCZ_TR_16_gen, params_CCCZ_TR_18_gen, params_CCCZ_TR_20_gen]

# final choice (T-opt, TR-opt, approx T-opt, min-parameter):
params_list_CCCZ = [params_CCCZ_T_18_gen, params_CCCZ_TR_14, params_CCCZ_T_10, params_CCCZ_T_8]


# # # fidelities, Rydberg times (processed data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# CCCZ' Vnn=inf, Vnnn=inf, T-optimized
nParams_CCCZ_T = [8, 10, 12, 14, 16, 18, 20]
Ts_CCCZ_T = [14.142222, 12.420321, 12.422129, 12.420260, 12.429370, 12.425380, 12.461711]
infidelities_nodecay_CCCZ_T = [1.5107e-08, 1.3046e-08, 5.4373e-09, 2.0084e-08, 1.3170e-08, 9.6985e-09, 2.5977e-08]
infidelities_CCCZ_T = [6.4059e-04, 4.9245e-04, 4.9263e-04, 4.9182e-04, 4.9256e-04, 4.9308e-04, 4.9526e-04]
TRs_CCCZ_T = [6.407276, 4.925657, 4.925549, 4.917287, 4.926673, 4.931915, 4.951645]

# CCCZ' Vnn=inf, Vnnn=inf, T-optimized, general ansatz
nParams_CCCZ_T_gen = [8, 10, 12, 14, 16, 18, 20]
Ts_CCCZ_T_gen = [24.692401, 19.054921, 14.143509, 12.901014, 11.854915, 11.802713, 11.806561]
infidelities_nodecay_CCCZ_T_gen = [7.5234e-04, 6.9885e-04, 4.4596e-08, 2.3032e-08, 5.0832e-08, 5.4317e-08, 1.0320e-08]
infidelities_CCCZ_T_gen = [2.5552e-03, 1.9708e-03, 6.4093e-04, 5.4277e-04, 5.3154e-04, 5.4934e-04, 5.4056e-04]
TRs_CCCZ_T_gen = [18.056899, 12.732982, 6.410328, 5.427917, 5.315628, 5.493425, 5.406144]

# CCCZ' Vnn=inf, Vnnn=inf, TR-optimized
nParams_CCCZ_TR = [8, 10, 12, 14, 16, 18, 20]
Ts_CCCZ_TR = [16.652910, 15.533310, 13.529940, 15.648822, 15.584528, 15.581547, 15.547389]
infidelities_nodecay_CCCZ_TR = [1.2164e-06, 2.2136e-06, 3.0506e-06, 5.1492e-06, 3.3817e-06, 4.5686e-06, 5.4057e-06]
infidelities_CCCZ_TR = [6.0541e-04, 4.8961e-04, 4.8585e-04, 4.8058e-04, 4.8144e-04, 4.8149e-04, 4.8156e-04]
TRs_CCCZ_TR = [6.042987, 4.874843, 4.827718, 4.755230, 4.780322, 4.770085, 4.761311]

# CCCZ' Vnn=inf, Vnnn=inf, TR-optimized, general ansatz
nParams_CCCZ_TR_gen = [8, 10, 12, 14, 16, 18, 20]
Ts_CCCZ_TR_gen = [13.397129, 16.457075, 15.628984, 14.216876, 12.463893, 12.467045, 12.538950]
infidelities_nodecay_CCCZ_TR_gen = [1.0453e-03, 8.4531e-04, 1.2383e-05, 3.4854e-06, 5.3810e-07, 9.8255e-07, 6.5782e-07]
infidelities_CCCZ_TR_gen = [1.5253e-03, 1.4903e-03, 5.6841e-04, 5.0082e-04, 4.8991e-04, 4.8999e-04, 4.8877e-04]
TRs_CCCZ_TR_gen = [None, None, 5.561810, 4.974815, 4.893025, 4.891283, 4.880423]  # 4.805441, 6.451009

# final choice (T-opt, TR-opt, approx T-opt, min-parameter):
nParams_CCCZ = [18, 14, 10, 8]
Ts_CCCZ = [11.802713, 15.648822, 12.420321, 14.142222]
infidelities_nodecay_CCCZ = [5.4317e-08, 5.1492e-06, 1.3046e-08, 1.5107e-08]
infidelities_CCCZ = [5.4934e-04, 4.8058e-04, 4.9245e-04, 6.4059e-04]
TRs_CCCZ = [5.493425, 4.755230, 4.925657, 6.407276]


# # # functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def plots_combined(nParams_list_T, Ts_list_T, nParams_list_T_gen, Ts_list_T_gen,
                   nParams_list_TR, TRs_list_TR, infidelities_list_TR, nParams_list_TR_gen, TRs_list_TR_gen, infidelities_list_TR_gen,
                   pulse, pulses_params_list):
    fig, ax = plt.subplots(1, 3, width_ratios=[0.6, 0.6, 1], figsize=(7.0, 1.8), layout='constrained')
    ax[0].plot(nParams_list_T_gen, Ts_list_T_gen, 'x', markersize=4, color='tab:olive', label=r'general ansatz')
    ax[0].plot(nParams_list_T, Ts_list_T, '+', markersize=5, color='tab:brown', label=r'antisym. ansatz')
    ax[0].set_xlabel(r'# parameters', fontsize=10)
    ax[0].set_ylabel(r'$\Omega_0 T$', fontsize=10)
    ax[0].tick_params(axis='both', labelsize=8)
    ax[0].set_xlim(7.3, 20.7)
    ax[0].set_ylim(11.0, 16.0)
    ax[0].set_xticks([8, 12, 16, 20])
    ax[0].set_yticks([12, 14, 16])
    ax[0].legend(fontsize=8, handlelength=0.8, handletextpad=0.8, borderpad=0.3, borderaxespad=0.2, loc='upper right')
    ax[0].grid()
    #
    ax1a = ax[1].twinx()
    color_fid = 'tab:blue'
    color_TR = 'tab:red'
    infid_gen, = ax[1].plot(nParams_list_TR_gen, infidelities_list_TR_gen, 'D', markersize=4, color='tab:cyan', markerfacecolor='none', markeredgewidth=1.5)
    tr_gen, = ax1a.plot(nParams_list_TR_gen, TRs_list_TR_gen, 'x', markersize=4, color='tab:pink')
    infid, = ax[1].plot(nParams_list_TR, infidelities_list_TR, 'o', markersize=5, color=color_fid, markerfacecolor='none', markeredgewidth=1.5)
    tr, = ax1a.plot(nParams_list_TR, TRs_list_TR, '+', markersize=5, color=color_TR)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax1a.yaxis.tick_left()
    ax1a.yaxis.set_label_position("left")
    ax[1].set_xlabel(r'# parameters', fontsize=10)
    ax[1].set_ylabel(r'$1-F$', fontsize=10, color=color_fid)
    ax1a.set_ylabel(r'$\Omega_0 T_R$', fontsize=10, color=color_TR)
    ax[1].tick_params(axis='both', labelsize=8)
    ax1a.tick_params(axis='both', labelsize=8)
    ax[1].set_xlim(7.3, 20.7)
    ax[1].set_ylim(0.00045, 0.00070)
    ax1a.set_ylim(4.5, 7.0)
    ax[1].set_xticks([8, 12, 16, 20])
    ax[1].set_yticks([0.0005, 0.0006, 0.0007])
    ax1a.set_yticks([5, 6, 7])
    ax[1].ticklabel_format(style='sci', scilimits=(-4, -4), axis='y', useMathText=True)
    ax[1].yaxis.get_offset_text().set_fontsize(8)
    ax[1].legend([(tr_gen, infid_gen), (tr, infid)], ['general ansatz', 'antisym. ansatz'], numpoints=1,
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=8, handlelength=1.8, handletextpad=0.8, borderpad=0.3, borderaxespad=0.2, loc='upper right')
    ax[1].grid()
    #
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    label_list = [r'$T$-opt. (1)', r'$T_R$-opt. (4)', r'$\approx T$-opt. (2)', r'min-param. (3)']
    if isinstance(pulse, str):
        pulse = pulses_qutip.get_pulse(pulse)
    for i, params in enumerate(pulses_params_list):
        params = np.array(params)
        T = params[0]
        Delta_of_t, phi_of_t = pulse(params)
        ts = np.linspace(0, T, 1000)
        phis = phi_of_t(ts)
        phis -= phis[0]
        ax[2].plot(ts, phis / (2 * np.pi), linewidth=2, color=colors[i], linestyle=linestyles[i], label=label_list[i])
    ax[2].set_xlabel(r'$\Omega_0 t$', fontsize=10)
    ax[2].set_ylabel(r'$\xi/(2\pi)$', fontsize=10)
    ax[2].tick_params(axis='both', labelsize=8)
    ax[2].set_xlim(-0.3, 16.0)
    ax[2].set_ylim(-0.6, 3.0)
    ax[2].legend(fontsize=8, ncol=2, handlelength=2.2, handletextpad=0.8, borderpad=0.3, borderaxespad=0.2, columnspacing=0.5, labelspacing=0.1, loc='upper left')
    ax[2].grid()
    fig.text(0.065, 0.93, '(a)', fontsize=10)
    fig.text(0.334, 0.93, '(b)', fontsize=10)
    fig.text(0.655, 0.93, '(c)', fontsize=10)
    plt.show()
    fig.savefig('Figs/CCCZ_combined_plot.pdf', bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)

    # number of atoms participating in the gate (2, 3 or 4)
    n_atoms = 4

    # Rydberg interaction strengths
    Vnn = float("inf")
    Vnnn = float("inf")
    decay = 0.0001

    # target gate phases
    theta = np.pi
    eps = np.pi
    lamb = np.pi
    delta = np.pi
    kappa = np.pi

    # pulse type
    pulse = pulses_qutip.pulse_phase_sin_cos_crab_smooth

    # optimization parameters
    params_list = params_list_CCCZ

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # pulse_postprocessing.postprocess_pulses(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params_list, decay)
    plots_combined(nParams_CCCZ_T, Ts_CCCZ_T, nParams_CCCZ_T_gen, Ts_CCCZ_T_gen, nParams_CCCZ_TR, TRs_CCCZ_TR, infidelities_CCCZ_TR, nParams_CCCZ_TR_gen, TRs_CCCZ_TR_gen, infidelities_CCCZ_TR_gen, pulse, params_list_CCCZ)
