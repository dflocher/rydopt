import numpy as np
import matplotlib.pyplot as plt
import pulses_qutip
import pulse_postprocessing
from matplotlib.legend_handler import HandlerTuple


plt.rcParams['font.sans-serif'] = "Helvetica"
plt.rcParams['mathtext.fontset'] = "cm"
# plt.rcParams['text.usetex'] = True


# # # pulse params (raw data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# minimal T for CZCZCZ gates with Vnn=Vnnn=inf
params_CZCZCZ_T_6 = [18.17182396, 0.0, 0.51050960, -1.03247663, 2.20024245, 0.0, 0.0, 0.74044080, -1.34735876]
params_CZCZCZ_T_8 = [16.36973127, 0.0, 0.28705772, -0.62187299, 2.40051297, 0.0, 0.0, 0.28403466, 0.45564291, 0.0, 0.0, 1.98974449, -0.20040935]
params_CZCZCZ_T_10 = [16.37358553, 0.0, 0.61432010, 0.09579554, -0.13626272, 0.0, 0.0, -0.94023826, 1.05826936, 0.0, 0.0, -0.34307055, 0.29954856, 0.0, 0.0, 0.22403604, -0.20584603]
params_CZCZCZ_T_12 = [16.36686730, 0.0, -0.91596238, -2.15802983, 3.01972826, 0.0, 0.0, -2.25787077, -1.84681230, 0.0, 0.0, 2.00193634, 0.20626402, 0.0, 0.0, -1.10072411, -0.31457866, 0.0, 0.0, 0.90682712, 0.02669395]
params_CZCZCZ_T_14 = [16.37275459, 0.0, 0.53738717, -0.43795663, 0.81365726, 0.0, 0.0, 0.74392986, 0.23345285, 0.0, 0.0, -1.65704238, 0.56589408, 0.0, 0.0, 0.21646962, -0.20648599, 0.0, 0.0, 1.56735486, -0.12305037, 0.0, 0.0, 0.49044389, 0.13027331]
params_CZCZCZ_T_16 = [16.36958798, 0.0, -0.62175535, 0.58806662, -0.91775891, 0.0, 0.0, 2.75063344, -0.03239631, 0.0, 0.0, 1.99513476, 0.20982055, 0.0, 0.0, -1.12872682, -0.27009915, 0.0, 0.0, 0.53689112, 0.46308826, 0.0, 0.0, 0.06239651, -0.51665186, 0.0, 0.0, -0.23037465, 0.08010553]
params_CZCZCZ_T_18 = [16.36710879, 0.0, -0.42728097, 1.64466664, -1.45223015, 0.0, 0.0, 0.63041986, -0.16586297, 0.0, 0.0, -0.41744184, 0.07474806, 0.0, 0.0, -0.03427191, -0.31044756, 0.0, 0.0, 0.25040124, -0.05104706, 0.0, 0.0, 0.39910099, -0.03348693, 0.0, 0.0, -0.64370646, -0.08854309, 0.0, 0.0, 2.16883007, -0.00214823]
params_CZCZCZ_T_20 = [16.36852674, 0.0, 0.94998523, -1.32300947, -2.13974333, 0.0, 0.0, -0.68587513, 1.34626329, 0.0, 0.0, 0.75012144, -0.32288801, 0.0, 0.0, 1.13338028, -0.05434556, 0.0, 0.0, -1.75231559, -0.20699584, 0.0, 0.0, -2.23068884, 0.48708192, 0.0, 0.0, 0.38109403, 0.00686881, 0.0, 0.0, -0.13440966, -0.02515267, 0.0, 0.0, -0.29703679, 0.01163954]
params_list_CZCZCZ_T = [params_CZCZCZ_T_6, params_CZCZCZ_T_8, params_CZCZCZ_T_10, params_CZCZCZ_T_12, params_CZCZCZ_T_14, params_CZCZCZ_T_16, params_CZCZCZ_T_18, params_CZCZCZ_T_20]

# minimal T for CZCZCZ gates with Vnn=Vnnn=inf, general ansatz:
params_CZCZCZ_T_6_gen = [18.89261615, 0.0, -1.05797018, 3.96584932, -5.91306624, 1.10495282, 0.00000004]
params_CZCZCZ_T_8_gen = [18.17455981, 0.0, -0.59737850, -0.70081556, -1.58331052, -1.07602239, 0.00000001, 0.73490606, 1.35163186]
params_CZCZCZ_T_10_gen = [18.08474633, 0.0, -0.13679508, 1.14491133, -1.02068230, 0.67767137, -2.77747645, 0.81863147, -0.99152384, 0.81645198, -0.72159646]
params_CZCZCZ_T_12_gen = [16.37171741, 0.0, -0.57718912, 0.28117341, -1.04598412, -1.92050678, 0.00000001, 0.40590937, -0.33897051, 0.05538608, 0.00000000, 2.18967395, 0.20565401]
params_CZCZCZ_T_14_gen = [16.38201986, 0.0, 0.49650044, -0.07734056, 1.18628446, 0.32784864, 1.01161842, 0.29541610, 0.38534795, -2.35997317, -0.72420889, 1.82536955, -0.21873915, -0.15390984, 0.10318699]
params_CZCZCZ_T_16_gen = [16.37164005, 0.0, 0.60171327, -0.07546039, -0.26245429, -1.63846507, -0.00000231, -1.15846941, 1.21107944, 2.08930444, -0.00000079, -0.38087467, 0.32369106, -1.91661312, -0.00000263, 0.22360925, -0.20517277]
params_CZCZCZ_T_18_gen = [16.37736305, 0.0, 0.78038427, -0.53531795, -0.97677653, 0.57586429, -0.40947220, -0.73506060, 1.25111614, -0.57887771, 2.03483531, 1.68281043, -0.22471345, -2.59750878, -1.81970357, -0.67163328, 0.17716258, -0.58108552, -0.03969609]
params_CZCZCZ_T_20_gen = [16.37486954, 0.0, -0.34559227, 0.48194485, -1.27004536, -0.81435551, 0.00133276, -0.92492024, -0.40917503, -1.13379096, 0.00154521, 1.18589755, -0.31349884, -0.33845676, 0.00171998, 1.44554571, -0.07222208, -1.07689949, -0.00175104, -2.02873580, -0.21272830]
params_list_CZCZCZ_T_gen = [params_CZCZCZ_T_6_gen, params_CZCZCZ_T_8_gen, params_CZCZCZ_T_10_gen, params_CZCZCZ_T_12_gen, params_CZCZCZ_T_14_gen, params_CZCZCZ_T_16_gen, params_CZCZCZ_T_18_gen, params_CZCZCZ_T_20_gen]

# minimal TR for CZCZCZ gates with Vnn=Vnnn=inf
params_CZCZCZ_TR_6 = [18.16902265, 0.0, -0.39852361, -1.60940640, -3.09436077, 0.0, 0.0, 0.74466587, 1.34438386]
params_CZCZCZ_TR_8 = [16.56456040, 0.0, 0.41971218, -0.04643639, -0.83527865, 0.0, 0.0, 1.34678303, 0.05257253, 0.0, 0.0, 0.81906055, -0.31596455]
params_CZCZCZ_TR_10 = [16.56612900, 0.0, -0.46182465, 0.17352921, 1.72417666, 0.0, 0.0, -0.83334660, -0.78176285, 0.0, 0.0, 0.93256224, 0.29826083, 0.0, 0.0, 1.11272319, -0.03210430]
params_CZCZCZ_TR_12 = [16.55801650, 0.0, -0.35165460, 1.65174219, 1.82317688, 0.0, 0.0, -0.46978813, -1.15922798, 0.0, 0.0, -0.96930039, -0.15860952, 0.0, 0.0, 1.91679465, -0.02199832, 0.0, 0.0, -0.40265708, 0.29943938]
params_CZCZCZ_TR_14 = [16.56714281, 0.0, 0.35061197, 0.48200046, 0.08821725, 0.0, 0.0, -0.58756564, -1.75767221, 0.0, 0.0, -1.50498871, 1.18901208, 0.0, 0.0, 2.74312691, 0.02821638, 0.0, 0.0, -0.40211555, -0.29905782, 0.0, 0.0, 0.64877035, 0.00300196]
params_CZCZCZ_TR_16 = [16.56394942, 0.0, -0.25657039, 1.23452536, -1.39394115, 0.0, 0.0, 0.60553533, -0.40090820, 0.0, 0.0, -1.78921420, 1.84186172, 0.0, 0.0, 1.01205922, 0.11554738, 0.0, 0.0, 0.33579460, -0.12015195, 0.0, 0.0, -0.99650991, 0.38447242, 0.0, 0.0, 0.22581605, 0.02251764]
params_CZCZCZ_TR_18 = [16.56962229, 0.0, -0.36148005, 2.07320717, 1.42508049, 0.0, 0.0, -0.22414655, -1.07574664, 0.0, 0.0, 0.92862197, 0.29420017, 0.0, 0.0, -1.32839843, 0.16429064, 0.0, 0.0, 1.56712244, -0.00237891, 0.0, 0.0, 2.28018400, -0.04724223, 0.0, 0.0, 0.56177372, 0.05877535, 0.0, 0.0, -0.63968438, -0.03408172]
params_CZCZCZ_TR_20 = [16.55190837, 0.0, -0.35177505, 1.08189868, -1.13436551, 0.0, 0.0, -0.61405018, 3.05551681, 0.0, 0.0, -1.62463544, -1.39299241, 0.0, 0.0, 1.00919891, -0.02675620, 0.0, 0.0, -0.39542807, 0.30115571, 0.0, 0.0, 1.03083593, -0.27598432, 0.0, 0.0, 0.53548216, 0.06600493, 0.0, 0.0, 0.04661408, 0.26705303, 0.0, 0.0, -0.21480159, -0.01926155]
params_list_CZCZCZ_TR = [params_CZCZCZ_TR_6, params_CZCZCZ_TR_8, params_CZCZCZ_TR_10, params_CZCZCZ_TR_12, params_CZCZCZ_TR_14, params_CZCZCZ_TR_16, params_CZCZCZ_TR_18, params_CZCZCZ_TR_20]

# minimal TR for CZCZCZ gates with Vnn=Vnnn=inf, general ansatz:
params_CZCZCZ_TR_6_gen = [24.67892419, 0.0, 1.19664793, 2.81453354, 10.31051221, 2.08455233, -0.00000015]
params_CZCZCZ_TR_8_gen = [18.16953853, 0.0, -0.36178363, -1.96387313, -3.40915303, -0.67071007, 0.00000000, 0.74559763, 1.34370802]
params_CZCZCZ_TR_10_gen = [19.55370648, 0.0, -0.72703397, 0.38104403, -3.96936317, 0.31688862, 1.76978464, 2.98177435, -0.40337869, 1.54793410, 1.19697174]
params_CZCZCZ_TR_12_gen = [16.56960668, 0.0, 0.44974849, -0.22227858, -0.96088741, 0.46734195, -1.15640939, 3.49792542, 0.05870216, -1.05043490, 1.15640986, 0.80281488, -0.32393523]
params_CZCZCZ_TR_14_gen = [16.56695805, 0.0, -0.33994921, 0.99596169, 0.63864846, 1.25265784, -0.00000422, 0.04636172, -0.19453294, 2.82709394, 0.00000001, 0.84471818, 0.30446916, 1.55418490, -0.00000038]
params_CZCZCZ_TR_16_gen = [16.59426873, 0.0, 0.86010399, -1.38549381, -3.49411409, 2.77925543, 0.97849245, -0.15552380, 0.00606725, -0.55443895, -0.97849249, 0.93914358, -0.29291029, -1.05522104, 0.00000000, 3.03506023, 0.04428670]
params_CZCZCZ_TR_18_gen = [16.58227321, 0.0, 0.58170165, -0.28596034, -1.99232845, 1.43277873, -0.07637758, -0.99138393, 0.47509060, 0.49023275, -1.04046106, 0.94668699, -0.29649353, -0.39602767, 1.06867922, 1.27990133, 0.03585982, -0.02199993, 0.01865970]
params_CZCZCZ_TR_20_gen = [16.55521682, 0.0, 0.81333254, -2.08444089, -3.32560474, -2.58021877, 0.02955223, -0.48555444, -0.06357177, -1.78366907, -0.02276215, 0.00254392, 0.05028527, 2.37560008, 0.00005897, 1.03596030, 0.02189556, 1.32994418, -0.00130956, -0.41665081, -0.32081890]
params_list_CZCZCZ_TR_gen = [params_CZCZCZ_TR_6_gen, params_CZCZCZ_TR_8_gen, params_CZCZCZ_TR_10_gen, params_CZCZCZ_TR_12_gen, params_CZCZCZ_TR_14_gen, params_CZCZCZ_TR_16_gen, params_CZCZCZ_TR_18_gen, params_CZCZCZ_TR_20_gen]

# final choice (T-opt, TR-opt, min-parameter):
params_list_CZCZCZ = [params_CZCZCZ_T_8, params_CZCZCZ_TR_8, params_CZCZCZ_T_6]


# # # fidelities, Rydberg times (processed data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# CZCZCZ Vnn=inf, Vnnn=inf, T-optimized
nParams_CZCZCZ_T = [6, 8, 10, 12, 14, 16, 18, 20]
Ts_CZCZCZ_T = [18.171824, 16.369731, 16.373586, 16.366867, 16.372755, 16.369588, 16.367109, 16.368527]
infidelities_nodecay_CZCZCZ_T = [5.9057e-09, 2.5739e-08, 2.5205e-12, 1.6690e-12, 2.1565e-10, 1.1049e-12, 1.2212e-14, 6.2172e-15]
infidelities_CZCZCZ_T = [7.8642e-04, 6.5399e-04, 6.5318e-04, 6.5759e-04, 6.5206e-04, 6.5624e-04, 7.1166e-04, 6.5260e-04]
TRs_CZCZCZ_T = [7.865886, 6.540339, 6.532555, 6.577555, 6.521284, 6.564097, 7.117694, 6.526722]

# CZCZCZ Vnn=inf, Vnnn=inf, T-optimized, general ansatz
nParams_CZCZCZ_T_gen = [6, 8, 10, 12, 14, 16, 18, 20]
Ts_CZCZCZ_T_gen = [None, 18.174560, 18.084746, 16.371717, 16.382020, 16.371640, 16.377363, 16.374870]  # 18.892616
infidelities_nodecay_CZCZCZ_T_gen = [9.3979e-02, 1.2801e-13, 1.4451e-08, 1.1990e-14, 4.9984e-11, 1.9255e-10, 1.5564e-12, 1.1481e-11]
infidelities_CZCZCZ_T_gen = [9.4807e-02, 7.8663e-04, 7.4913e-04, 6.5558e-04, 6.5064e-04, 6.5382e-04, 6.5491e-04, 7.1996e-04]
TRs_CZCZCZ_T_gen = [9.142719, 7.868133, 7.492426, 6.557368, 6.507029, 6.538946, 6.549781, 7.200705]

# CZCZCZ Vnn=inf, Vnnn=inf, TR-optimized
nParams_CZCZCZ_TR = [6, 8, 10, 12, 14, 16, 18, 20]
Ts_CZCZCZ_TR = [18.169023, 16.564560, 16.566129, 16.558017, 16.567143, 16.563949, 16.569622, 16.551908]
infidelities_nodecay_CZCZCZ_TR = [1.9777e-08, 5.0307e-08, 5.4313e-08, 5.8469e-08, 5.6055e-08, 9.4531e-08, 5.1590e-08, 8.0181e-08]
infidelities_CZCZCZ_TR = [7.8623e-04, 6.1040e-04, 6.0991e-04, 6.0995e-04, 6.0994e-04, 6.0960e-04, 6.0976e-04, 6.0990e-04]
TRs_CZCZCZ_TR = [7.863932, 6.104017, 6.099697, 6.100093, 6.099405, 6.096189, 6.098214, 6.099418]

# CZCZCZ Vnn=inf, Vnnn=inf, TR-optimized, general ansatz
nParams_CZCZCZ_TR_gen = [6, 8, 10, 12, 14, 16, 18, 20]
Ts_CZCZCZ_TR_gen = [24.678924, 18.169539, 19.553706, 16.569607, 16.566958, 16.594269, 16.582273, 16.555217]
infidelities_nodecay_CZCZCZ_TR_gen = [4.8166e-02, 2.1538e-08, 7.2107e-08, 4.8919e-08, 4.9892e-08, 6.1934e-08, 7.8611e-07, 6.2007e-08]
infidelities_CZCZCZ_TR_gen = [4.9311e-02, 7.8621e-04, 7.2962e-04, 6.1039e-04, 6.1041e-04, 6.1023e-04, 6.1077e-04, 6.1016e-04]
TRs_CZCZCZ_TR_gen = [12.018156, 7.863768, 7.296501, 6.103994, 6.104760, 6.102213, 6.100394, 6.101509]

# final choice (T-opt, TR-opt, min-parameter):
nParams_CZCZCZ = [8, 8, 6]
Ts_CZCZCZ = [16.369731, 16.564560, 18.171824]
infidelities_nodecay_CZCZCZ = [2.5739e-08, 5.0307e-08, 5.9057e-09]
infidelities_CZCZCZ = [6.5399e-04, 6.1040e-04, 7.8642e-04]
TRs_CZCZCZ = [6.540339, 6.104017, 7.865886]


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
    ax[0].set_xlim(5.3, 20.7)
    ax[0].set_ylim(16.0, 19.25)
    ax[0].set_xticks([6, 10, 14, 18])
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
    ax[1].set_xlim(5.3, 20.7)
    ax[1].set_ylim(0.000575, 0.00090)
    ax1a.set_ylim(5.75, 9.0)
    ax[1].set_xticks([6, 10, 14, 18])
    ax[1].ticklabel_format(style='sci', scilimits=(-4, -4), axis='y', useMathText=True)
    ax[1].yaxis.get_offset_text().set_fontsize(8)
    ax[1].legend([(tr_gen, infid_gen), (tr, infid)], ['general ansatz', 'antisym. ansatz'], numpoints=1,
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=8, handlelength=1.8, handletextpad=0.8, borderpad=0.3, borderaxespad=0.2, loc='upper right')
    ax[1].grid()
    #
    colors = ['tab:blue', 'tab:red', 'tab:orange']
    linestyles = ['solid', 'dotted', 'dashed']
    label_list = [r'$T$-opt. (1)', r'$T_R$-opt. (3)', r'min-param. (2)']
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
    ax[2].set_xlim(-0.3, 18.5)
    ax[2].set_ylim(-0.2, 2.0)
    ax[2].legend(fontsize=8, handlelength=2.2, handletextpad=0.8, borderpad=0.3, borderaxespad=0.2, loc='upper left')
    ax[2].grid()
    fig.text(0.065, 0.93, '(a)', fontsize=10)
    fig.text(0.331, 0.93, '(b)', fontsize=10)
    fig.text(0.66, 0.93, '(c)', fontsize=10)
    plt.show()
    fig.savefig('Figs/CZCZCZ_combined_plot.pdf', bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)

    # number of atoms participating in the gate (2, 3 or 4)
    n_atoms = 3

    # Rydberg interaction strengths
    Vnn = float("inf")
    Vnnn = float("inf")
    decay = 0.0001

    # target gate phases
    theta = np.pi
    eps = np.pi
    lamb = 0
    delta = 0
    kappa = 0

    # pulse type
    pulse = pulses_qutip.pulse_phase_sin_cos_crab_smooth

    # optimization parameters
    params_list = params_list_CZCZCZ

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # pulse_postprocessing.postprocess_pulses(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params_list, decay)
    plots_combined(nParams_CZCZCZ_T, Ts_CZCZCZ_T, nParams_CZCZCZ_T_gen, Ts_CZCZCZ_T_gen, nParams_CZCZCZ_TR, TRs_CZCZCZ_TR, infidelities_CZCZCZ_TR, nParams_CZCZCZ_TR_gen, TRs_CZCZCZ_TR_gen, infidelities_CZCZCZ_TR_gen, pulse, params_list_CZCZCZ)
