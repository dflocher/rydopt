import numpy as np
import matplotlib.pyplot as plt
from rydopt.pulses import pulses_qutip
from matplotlib.legend_handler import HandlerTuple


plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["mathtext.fontset"] = "cm"
# plt.rcParams['text.usetex'] = True


# # # pulse params (raw data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# minimal T for CCZ' gates with Vnn=Vnnn=inf
params_CCZ_T_6 = [
    12.24229503,
    0.0,
    0.73502207,
    3.00127734,
    2.80831494,
    0.0,
    0.0,
    3.45152105,
    1.08933841,
]
params_CCZ_T_8 = [
    10.97094681,
    0.0,
    0.19566367,
    0.43131090,
    -1.16460209,
    0.0,
    0.0,
    1.05669771,
    -0.70545851,
    0.0,
    0.0,
    0.88054914,
    -0.22756692,
]
params_CCZ_T_10 = [
    10.96884139,
    0.0,
    -0.07008709,
    2.43969907,
    0.98265246,
    0.0,
    0.0,
    2.32243142,
    0.63429561,
    0.0,
    0.0,
    1.15146035,
    0.17286608,
    0.0,
    0.0,
    0.88123862,
    0.01750192,
]
params_CCZ_T_12 = [
    10.97146357,
    0.0,
    0.05855260,
    2.24651001,
    -0.89077837,
    0.0,
    0.0,
    1.14713327,
    -0.54217475,
    0.0,
    0.0,
    -2.38294178,
    -0.03782761,
    0.0,
    0.0,
    -0.12522635,
    -0.26386991,
    0.0,
    0.0,
    -0.28610389,
    -0.01442632,
]
params_CCZ_T_14 = [
    10.98046742,
    0.0,
    0.06406242,
    0.26967787,
    -0.69114384,
    0.0,
    0.0,
    -0.84021858,
    0.22026745,
    0.0,
    0.0,
    -1.64908848,
    1.25113235,
    0.0,
    0.0,
    -0.43893892,
    0.52453165,
    0.0,
    0.0,
    0.02010211,
    0.06843022,
    0.0,
    0.0,
    -0.81154803,
    0.12952928,
]
params_CCZ_T_16 = [
    10.97699229,
    0.0,
    0.05202600,
    0.50204969,
    -0.41839833,
    0.0,
    0.0,
    1.41751260,
    -0.46352960,
    0.0,
    0.0,
    -1.11736106,
    -0.59110136,
    0.0,
    0.0,
    1.09914181,
    -0.06395958,
    0.0,
    0.0,
    -0.43331197,
    -0.38482962,
    0.0,
    0.0,
    -0.69312247,
    0.28465912,
    0.0,
    0.0,
    -1.39251668,
    -0.14601200,
]
params_CCZ_T_18 = [
    10.96504898,
    0.0,
    -0.09692630,
    0.16201188,
    0.15408660,
    0.0,
    0.0,
    -0.59675079,
    0.85570646,
    0.0,
    0.0,
    -0.05582021,
    0.76131022,
    0.0,
    0.0,
    -0.36871985,
    -0.21435129,
    0.0,
    0.0,
    -0.44066407,
    0.23128124,
    0.0,
    0.0,
    -0.92639357,
    0.06588520,
    0.0,
    0.0,
    -0.39429820,
    0.01963682,
    0.0,
    0.0,
    0.02287272,
    0.00539197,
]
params_CCZ_T_20 = [
    10.97109778,
    0.0,
    -0.00221211,
    0.65980724,
    0.11971514,
    0.0,
    0.0,
    -0.36409576,
    -1.01926936,
    0.0,
    0.0,
    1.81469123,
    0.35986169,
    0.0,
    0.0,
    0.84347657,
    -0.05791392,
    0.0,
    0.0,
    -1.18630963,
    -0.42454422,
    0.0,
    0.0,
    -0.97237904,
    -0.35375572,
    0.0,
    0.0,
    -1.41408328,
    0.23572069,
    0.0,
    0.0,
    -1.72542664,
    -0.50775527,
    0.0,
    0.0,
    0.34259385,
    -0.01100570,
]
params_list_CCZ_T = [
    params_CCZ_T_6,
    params_CCZ_T_8,
    params_CCZ_T_10,
    params_CCZ_T_12,
    params_CCZ_T_14,
    params_CCZ_T_16,
    params_CCZ_T_18,
    params_CCZ_T_20,
]

# minimal T for CCZ' gates with Vnn=Vnnn=inf, general ansatz:
params_CCZ_T_6_gen = [
    21.21672407,
    0.0,
    -0.47295335,
    -0.30531499,
    12.41595236,
    -0.08678792,
    -2.87313738,
]
params_CCZ_T_8_gen = [
    12.24163206,
    0.0,
    -0.73534871,
    3.06771049,
    -2.80804089,
    -0.12456725,
    -0.00000000,
    3.52935892,
    -1.08894706,
]
params_CCZ_T_10_gen = [
    12.24568874,
    0.0,
    0.73360313,
    2.78307636,
    2.80967626,
    0.46232503,
    -0.50615776,
    3.22070307,
    1.09135190,
    -1.05569683,
    0.50615776,
]
params_CCZ_T_12_gen = [
    10.96723134,
    0.0,
    0.10434435,
    0.94005560,
    -0.98105132,
    -2.44649653,
    -0.00055482,
    1.04370875,
    -0.62334614,
    1.08144355,
    0.00003447,
    0.70946634,
    -0.26119949,
]
params_CCZ_T_14_gen = [
    10.83264840,
    0.0,
    0.00808517,
    1.29967575,
    -0.15117887,
    0.26117336,
    0.32173635,
    -0.18593552,
    -0.37946965,
    -0.33014685,
    -1.01300629,
    0.07813142,
    -0.48186569,
    0.36779145,
    0.24847061,
]
params_CCZ_T_16_gen = [
    10.82524797,
    0.0,
    0.08367404,
    1.81084736,
    1.21067382,
    -0.87870536,
    0.17148690,
    -1.12761687,
    -0.97755532,
    -0.21182129,
    -0.86908356,
    0.08499797,
    0.47534799,
    0.31195592,
    0.26733244,
    0.65537471,
    -0.00096148,
]
params_CCZ_T_18_gen = [
    10.82275233,
    0.0,
    -0.08224065,
    1.54793197,
    -1.01828615,
    -0.61791398,
    0.71217468,
    -1.23828277,
    0.85286000,
    -0.56812468,
    -0.89058718,
    0.05717979,
    -0.45574283,
    0.34608422,
    0.27257727,
    -0.20242455,
    -0.00394941,
    -2.12874933,
    -0.27548407,
]
params_CCZ_T_20_gen = [
    10.83297191,
    0.0,
    0.05206165,
    -0.78402333,
    -0.34403124,
    2.20026764,
    0.04016910,
    -0.65528515,
    0.45866970,
    -0.08361566,
    -0.04807755,
    -0.05722618,
    -0.28863471,
    -1.00372618,
    -0.84336251,
    -0.21138632,
    0.05402187,
    -0.32868937,
    0.29485078,
    -1.19205494,
    0.75756327,
]
params_list_CCZ_T_gen = [
    params_CCZ_T_6_gen,
    params_CCZ_T_8_gen,
    params_CCZ_T_10_gen,
    params_CCZ_T_12_gen,
    params_CCZ_T_14_gen,
    params_CCZ_T_16_gen,
    params_CCZ_T_18_gen,
    params_CCZ_T_20_gen,
]

# minimal TR for CCZ' gates with Vnn=Vnnn=inf
params_CCZ_TR_6 = [
    12.23686569,
    0.0,
    0.73636271,
    3.29086312,
    2.80729332,
    0.0,
    0.0,
    3.80334079,
    1.08791348,
]
params_CCZ_TR_8 = [
    12.80428169,
    0.0,
    -0.15465767,
    -2.61106706,
    0.24591581,
    0.0,
    0.0,
    1.10146953,
    -1.54856790,
    0.0,
    0.0,
    0.24201368,
    1.22717433,
]
params_CCZ_TR_10 = [
    12.72911958,
    0.0,
    1.43281803,
    -1.98920752,
    -6.31464092,
    0.0,
    0.0,
    0.10028046,
    1.08234350,
    0.0,
    0.0,
    0.71976922,
    -0.43203706,
    0.0,
    0.0,
    0.65327507,
    0.20638712,
]
params_CCZ_TR_12 = [
    12.62662982,
    0.0,
    0.19223137,
    -0.07677351,
    -0.39326266,
    0.0,
    0.0,
    0.41796317,
    0.93797901,
    0.0,
    0.0,
    2.21165694,
    0.44761901,
    0.0,
    0.0,
    -0.05095336,
    -0.81126432,
    0.0,
    0.0,
    1.09518453,
    -0.05272573,
]
params_CCZ_TR_14 = [
    12.76395571,
    0.0,
    -0.35103176,
    -0.07162104,
    0.95020271,
    0.0,
    0.0,
    0.32774514,
    -1.05871720,
    0.0,
    0.0,
    0.90117158,
    -0.09977100,
    0.0,
    0.0,
    0.87288958,
    -0.14231086,
    0.0,
    0.0,
    -0.58431837,
    0.56716651,
    0.0,
    0.0,
    0.24114944,
    0.07520248,
]
params_CCZ_TR_16 = [
    12.71356955,
    0.0,
    0.08880897,
    -0.64457311,
    1.89239384,
    0.0,
    0.0,
    -1.96863791,
    -1.94155219,
    0.0,
    0.0,
    -0.59846030,
    1.19607266,
    0.0,
    0.0,
    0.89502662,
    0.14353159,
    0.0,
    0.0,
    0.69280665,
    -0.04673104,
    0.0,
    0.0,
    -1.02896041,
    -0.45836870,
    0.0,
    0.0,
    0.78823782,
    -0.02526929,
]
params_CCZ_TR_18 = [
    12.85451391,
    0.0,
    0.94755559,
    -0.78809643,
    -3.51691936,
    0.0,
    0.0,
    0.15222044,
    1.16225620,
    0.0,
    0.0,
    0.44092634,
    -0.39859735,
    0.0,
    0.0,
    1.35040829,
    -0.08806777,
    0.0,
    0.0,
    0.91026490,
    -0.44781920,
    0.0,
    0.0,
    0.30651986,
    0.20807062,
    0.0,
    0.0,
    -0.40549805,
    0.22342134,
    0.0,
    0.0,
    -0.15426775,
    0.05206450,
]
params_CCZ_TR_20 = [
    12.76782816,
    0.0,
    0.60359788,
    -0.66017348,
    -2.07901517,
    0.0,
    0.0,
    0.36491566,
    0.48769205,
    0.0,
    0.0,
    -0.58867832,
    0.63829705,
    0.0,
    0.0,
    1.20388067,
    0.16340470,
    0.0,
    0.0,
    -0.65548433,
    -0.48329569,
    0.0,
    0.0,
    0.23273637,
    -0.13129957,
    0.0,
    0.0,
    -0.70609665,
    0.01640657,
    0.0,
    0.0,
    1.82443324,
    0.07542600,
    0.0,
    0.0,
    0.83358457,
    -0.09341335,
]
params_list_CCZ_TR = [
    params_CCZ_TR_6,
    params_CCZ_TR_8,
    params_CCZ_TR_10,
    params_CCZ_TR_12,
    params_CCZ_TR_14,
    params_CCZ_TR_16,
    params_CCZ_TR_18,
    params_CCZ_TR_20,
]

# minimal TR for CCZ' gates with Vnn=Vnnn=inf, general ansatz:
params_CCZ_TR_6_gen = [
    21.20213908,
    0.0,
    0.49767700,
    -0.31672689,
    -12.58309179,
    -0.09069984,
    -2.87175343,
]
params_CCZ_TR_8_gen = [
    12.23634330,
    0.0,
    0.73638781,
    3.30920228,
    2.80703146,
    1.57912660,
    0.00000001,
    3.76931205,
    1.08745111,
]
params_CCZ_TR_10_gen = [
    12.24113130,
    0.0,
    -0.73443188,
    2.86943226,
    -2.80898553,
    0.43456379,
    -0.00000230,
    3.30880932,
    -1.09039593,
    0.23209528,
    -0.00000296,
]
params_CCZ_TR_12_gen = [
    12.85306446,
    0.0,
    -0.23074292,
    -0.67360332,
    0.51687498,
    0.05836869,
    -0.27072514,
    0.70160264,
    -1.09157799,
    -2.10686096,
    0.27072514,
    0.38987335,
    0.72810032,
]
params_CCZ_TR_14_gen = [
    11.01562149,
    0.0,
    -0.13530628,
    -0.45270640,
    0.26992599,
    -2.36169442,
    1.75594361,
    3.51264215,
    -0.34660790,
    -0.38248666,
    -1.16924637,
    1.58623743,
    0.08410157,
    0.46680547,
    0.24032935,
]
params_CCZ_TR_16_gen = [
    12.63464102,
    0.0,
    0.12704650,
    2.31829919,
    -0.26101900,
    1.67859441,
    -0.13553751,
    0.49351474,
    1.19163381,
    -0.59808574,
    0.13272988,
    0.30952296,
    -0.66610705,
    -1.73285594,
    0.00282516,
    0.76727456,
    0.15587785,
]
params_CCZ_TR_18_gen = [
    12.94195524,
    0.0,
    0.82139147,
    -1.78246316,
    -3.38576992,
    2.08830809,
    -0.11488769,
    0.42713982,
    1.10350811,
    -0.58917067,
    0.11614736,
    0.32363927,
    -0.54900569,
    1.07843025,
    -0.20945243,
    1.07797883,
    0.25002227,
    0.09511054,
    0.20852695,
]
params_CCZ_TR_20_gen = [
    12.74704031,
    0.0,
    -0.30634485,
    0.15262881,
    0.87965941,
    1.47106974,
    0.06563415,
    0.24020188,
    -1.09497925,
    -0.03744759,
    -0.09049481,
    0.54806306,
    0.46732390,
    -0.87105832,
    0.02420506,
    0.75699806,
    -0.14564537,
    1.27516995,
    0.02293226,
    1.56875568,
    0.00535058,
]
params_list_CCZ_TR_gen = [
    params_CCZ_TR_6_gen,
    params_CCZ_TR_8_gen,
    params_CCZ_TR_10_gen,
    params_CCZ_TR_12_gen,
    params_CCZ_TR_14_gen,
    params_CCZ_TR_16_gen,
    params_CCZ_TR_18_gen,
    params_CCZ_TR_20_gen,
]

# final choice (T-opt, TR-opt, approx T-opt, min-parameter):
params_list_CCZ = [
    params_CCZ_T_14_gen,
    params_CCZ_TR_10,
    params_CCZ_T_8,
    params_CCZ_T_6,
]


# # # fidelities, Rydberg times (processed data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# CCZ' Vnn=inf, Vnnn=inf, T-optimized
nParams_CCZ_T = [6, 8, 10, 12, 14, 16, 18, 20]
Ts_CCZ_T = [
    12.242295,
    10.970947,
    10.968841,
    10.971464,
    10.980467,
    10.976992,
    10.965049,
    10.971098,
]
infidelities_nodecay_CCZ_T = [
    1.8110e-09,
    5.0203e-09,
    2.8527e-08,
    7.6882e-09,
    2.5281e-08,
    1.9183e-09,
    1.1078e-08,
    9.4281e-08,
]
infidelities_CCZ_T = [
    4.4022e-04,
    4.1786e-04,
    4.1801e-04,
    4.1842e-04,
    4.1945e-04,
    4.1851e-04,
    4.1846e-04,
    4.1927e-04,
]
TRs_CCZ_T = [
    4.402245,
    4.178744,
    4.179821,
    4.184360,
    4.194319,
    4.185373,
    4.184514,
    4.191944,
]

# CCZ' Vnn=inf, Vnnn=inf, T-optimized, general ansatz
nParams_CCZ_T_gen = [6, 8, 10, 12, 14, 16, 18, 20]
Ts_CCZ_T_gen = [
    21.216724,
    12.241632,
    12.245689,
    10.967231,
    10.832648,
    10.825248,
    10.822752,
    10.832972,
]
infidelities_nodecay_CCZ_T_gen = [
    8.6732e-04,
    9.7524e-10,
    3.4033e-12,
    3.5230e-08,
    1.4322e-14,
    1.1102e-14,
    1.6653e-10,
    2.1160e-09,
]
infidelities_CCZ_T_gen = [
    1.8029e-03,
    4.4021e-04,
    4.4025e-04,
    4.1846e-04,
    4.9045e-04,
    4.7880e-04,
    4.6431e-04,
    4.6718e-04,
]
TRs_CCZ_T_gen = [
    9.354156,
    4.402419,
    4.402547,
    4.184380,
    4.905029,
    4.788176,
    4.643521,
    4.671918,
]

# CCZ' Vnn=inf, Vnnn=inf, TR-optimized
nParams_CCZ_TR = [6, 8, 10, 12, 14, 16, 18, 20]
Ts_CCZ_TR = [
    12.236866,
    12.804282,
    12.729120,
    12.626630,
    12.763956,
    12.713570,
    12.854514,
    12.767828,
]
infidelities_nodecay_CCZ_TR = [
    5.2656e-08,
    7.4964e-08,
    1.2399e-07,
    1.6861e-07,
    8.6725e-08,
    1.0916e-07,
    1.5214e-07,
    1.2021e-07,
]
infidelities_CCZ_TR = [
    4.4016e-04,
    4.0254e-04,
    3.9482e-04,
    3.9444e-04,
    3.9338e-04,
    3.9439e-04,
    3.9385e-04,
    3.9320e-04,
]
TRs_CCZ_TR = [
    4.401156,
    4.025036,
    3.946656,
    3.942452,
    3.933443,
    3.942557,
    3.936652,
    3.930496,
]

# CCZ' Vnn=inf, Vnnn=inf, TR-optimized, general ansatz
nParams_CCZ_TR_gen = [6, 8, 10, 12, 14, 16, 18, 20]
Ts_CCZ_TR_gen = [
    21.202139,
    12.236343,
    12.241131,
    12.853064,
    11.015621,
    12.634641,
    12.941955,
    12.747040,
]
infidelities_nodecay_CCZ_TR_gen = [
    9.4509e-04,
    5.9089e-08,
    4.1987e-08,
    8.0306e-08,
    1.2037e-07,
    1.5446e-07,
    1.1469e-07,
    4.2748e-07,
]
infidelities_CCZ_TR_gen = [
    1.8804e-03,
    4.4017e-04,
    4.4020e-04,
    4.0362e-04,
    3.9868e-04,
    3.9640e-04,
    3.9750e-04,
    3.9626e-04,
]
TRs_CCZ_TR_gen = [
    9.377646,
    4.401198,
    4.401837,
    4.035713,
    3.986112,
    3.962298,
    3.973471,
    3.958767,
]

# final choice (T-opt, TR-opt, approx T-opt, min-parameter):
nParams_CCZ = [14, 10, 8, 6]
Ts_CCZ = [10.832648, 12.729120, 10.970947, 12.242295]
infidelities_nodecay_CCZ = [1.4322e-14, 1.2399e-07, 5.0203e-09, 1.8110e-09]
infidelities_CCZ = [4.9045e-04, 3.9482e-04, 4.1786e-04, 4.4022e-04]
TRs_CCZ = [4.905029, 3.946656, 4.178744, 4.402245]


# # # functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def plots_combined(
    nParams_list_T,
    Ts_list_T,
    nParams_list_T_gen,
    Ts_list_T_gen,
    nParams_list_TR,
    TRs_list_TR,
    infidelities_list_TR,
    nParams_list_TR_gen,
    TRs_list_TR_gen,
    infidelities_list_TR_gen,
    pulse,
    pulses_params_list,
):
    fig, ax = plt.subplots(
        1, 3, width_ratios=[0.6, 0.6, 1], figsize=(7.0, 1.8), layout="constrained"
    )
    ax[0].plot(
        nParams_list_T_gen,
        Ts_list_T_gen,
        "x",
        markersize=4,
        color="tab:olive",
        label=r"general ansatz",
    )
    ax[0].plot(
        nParams_list_T,
        Ts_list_T,
        "+",
        markersize=5,
        color="tab:brown",
        label=r"antisym. ansatz",
    )
    ax[0].set_xlabel(r"# parameters", fontsize=10)
    ax[0].set_ylabel(r"$\Omega_0 T$", fontsize=10)
    ax[0].tick_params(axis="both", labelsize=8)
    ax[0].set_xlim(5.3, 20.7)
    ax[0].set_ylim(10.5, 13.125)
    ax[0].set_xticks([6, 10, 14, 18])
    ax[0].legend(
        fontsize=8,
        handlelength=0.8,
        handletextpad=0.8,
        borderpad=0.3,
        borderaxespad=0.2,
        loc="upper right",
    )
    ax[0].grid()
    #
    ax1a = ax[1].twinx()
    color_fid = "tab:blue"
    color_TR = "tab:red"
    (infid_gen,) = ax[1].plot(
        nParams_list_TR_gen,
        infidelities_list_TR_gen,
        "D",
        markersize=4,
        color="tab:cyan",
        markerfacecolor="none",
        markeredgewidth=1.5,
    )
    (tr_gen,) = ax1a.plot(
        nParams_list_TR_gen, TRs_list_TR_gen, "x", markersize=4, color="tab:pink"
    )
    (infid,) = ax[1].plot(
        nParams_list_TR,
        infidelities_list_TR,
        "o",
        markersize=5,
        color=color_fid,
        markerfacecolor="none",
        markeredgewidth=1.5,
    )
    (tr,) = ax1a.plot(nParams_list_TR, TRs_list_TR, "+", markersize=5, color=color_TR)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax1a.yaxis.tick_left()
    ax1a.yaxis.set_label_position("left")
    ax[1].set_xlabel(r"# parameters", fontsize=10)
    ax[1].set_ylabel(r"$1-F$", fontsize=10, color=color_fid)
    ax1a.set_ylabel(r"$\Omega_0 T_R$", fontsize=10, color=color_TR)
    ax[1].tick_params(axis="both", labelsize=8)
    ax1a.tick_params(axis="both", labelsize=8)
    ax[1].set_xlim(5.3, 20.7)
    ax[1].set_ylim(0.00038, 0.000485)
    ax1a.set_ylim(3.8, 4.85)
    ax[1].set_xticks([6, 10, 14, 18])
    ax[1].set_yticks([0.00040, 0.00044, 0.00048])
    ax1a.set_yticks([4.0, 4.4, 4.8])
    ax[1].ticklabel_format(style="sci", scilimits=(-4, -4), axis="y", useMathText=True)
    ax[1].yaxis.get_offset_text().set_fontsize(8)
    ax[1].legend(
        [(tr_gen, infid_gen), (tr, infid)],
        ["general ansatz", "antisym. ansatz"],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize=8,
        handlelength=1.8,
        handletextpad=0.8,
        borderpad=0.3,
        borderaxespad=0.2,
        loc="upper right",
    )
    ax[1].grid()
    #
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    label_list = [
        r"$T$-opt. (1)",
        r"$T_R$-opt. (4)",
        r"$\approx T$-opt. (2)",
        r"min-param. (3)",
    ]
    if isinstance(pulse, str):
        pulse = pulses_qutip.get_pulse(pulse)
    for i, params in enumerate(pulses_params_list):
        params = np.array(params)
        T = params[0]
        Delta_of_t, phi_of_t = pulse(params)
        ts = np.linspace(0, T, 1000)
        phis = phi_of_t(ts)
        phis -= phis[0]
        ax[2].plot(
            ts,
            phis / (2 * np.pi),
            linewidth=2,
            color=colors[i],
            linestyle=linestyles[i],
            label=label_list[i],
        )
    ax[2].set_xlabel(r"$\Omega_0 t$", fontsize=10)
    ax[2].set_ylabel(r"$\xi/(2\pi)$", fontsize=10)
    ax[2].tick_params(axis="both", labelsize=8)
    ax[2].set_xlim(-0.3, 13.1)
    ax[2].set_ylim(-0.5, 1.7)
    ax[2].legend(
        fontsize=8,
        ncol=2,
        handlelength=2.2,
        handletextpad=0.8,
        borderpad=0.3,
        borderaxespad=0.2,
        columnspacing=1.0,
        labelspacing=0.1,
        loc="upper right",
    )
    ax[2].grid()
    fig.text(0.065, 0.93, "(a)", fontsize=10)
    fig.text(0.332, 0.93, "(b)", fontsize=10)
    fig.text(0.673, 0.93, "(c)", fontsize=10)
    plt.show()
    fig.savefig("Figs/CCZ_combined_plot.pdf", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
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
    lamb = np.pi
    delta = 0
    kappa = 0

    # pulse type
    pulse = pulses_qutip.pulse_phase_sin_cos_crab_smooth

    # optimization parameters
    params_list = params_list_CCZ

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # pulses.postprocess_pulses(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params_list, decay)
    plots_combined(
        nParams_CCZ_T,
        Ts_CCZ_T,
        nParams_CCZ_T_gen,
        Ts_CCZ_T_gen,
        nParams_CCZ_TR,
        TRs_CCZ_TR,
        infidelities_CCZ_TR,
        nParams_CCZ_TR_gen,
        TRs_CCZ_TR_gen,
        infidelities_CCZ_TR_gen,
        pulse,
        params_list_CCZ,
    )
