import numpy as np
import matplotlib.pyplot as plt
import pulses_qutip
import pulse_verification
import pulse_postprocessing
from matplotlib.legend_handler import HandlerTuple


plt.rcParams['font.sans-serif'] = "Helvetica"
plt.rcParams['mathtext.fontset'] = "cm"
# plt.rcParams['text.usetex'] = True


# # # pulse params (raw data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# CCZ' Vnn=32, Vnnn=32
params_CCZ_TR_10_32_32 = [12.73103072, 0.0, 0.20481612, 0.17984404, -0.49973881, 0.0, 0.0, 0.36526123, 0.97167872, 0.0, 0.0, 0.62139120, -0.57510545, 0.0, 0.0, 0.42402089, 0.25587117]

# CCZ' Vnn=32, Vnnn=4
params_CCZ_TR_8_32_4 = [25.15702426, 0.0, -2.54459257, -0.03969836, -1.28435601, 0.0, 0.0, 0.03711036, 0.70426331, 0.0, 0.0, 1.03971858, 2.32167843]
params_CCZ_TR_10_32_4 = [19.57376025, 0.0, -1.36229563, -0.69677931, -7.86194430, 0.0, 0.0, 2.46432161, -0.63273799, 0.0, 0.0, 2.75885942, 0.78347987, 0.0, 0.0, 0.84702814, 1.06733851]
params_CCZ_TR_12_32_4 = [18.83374695, 0.0, -1.34304624, 0.41204883, 4.53813962, 0.0, 0.0, 2.52655604, -0.15931975, 0.0, 0.0, 0.10503397, -1.50791965, 0.0, 0.0, 0.72235774, 0.62894644, 0.0, 0.0, 1.09460236, -0.45586957]
params_CCZ_TR_14_32_4 = [13.05433360, 0.0, -0.85516750, 0.18802523, -0.01832473, 0.0, 0.0, -0.21264516, -0.94321238, 0.0, 0.0, 0.62721384, -0.68526583, 0.0, 0.0, 0.84841304, -0.56898339, 0.0, 0.0, 1.29266495, -0.31300030, 0.0, 0.0, 2.63091115, -0.75316982]
params_CCZ_TR_16_32_4 = [12.80914547, 0.0, -0.66904805, 0.60898590, -1.07749660, 0.0, 0.0, -1.32834774, -0.22544482, 0.0, 0.0, 0.18394220, -1.03172917, 0.0, 0.0, 0.53460908, -0.68199106, 0.0, 0.0, -0.79504635, 0.37951446, 0.0, 0.0, 1.57922015, -0.71325000, 0.0, 0.0, -0.08197756, -0.35493872]
params_CCZ_TR_18_32_4 = [14.02133290, 0.0, 0.23770216, 0.78821427, -1.47575981, 0.0, 0.0, -0.09879700, 1.58614387, 0.0, 0.0, -0.89156737, 0.11467345, 0.0, 0.0, 0.54412932, 0.30641617, 0.0, 0.0, 1.65119337, -0.08901811, 0.0, 0.0, -0.79489127, -0.48274224, 0.0, 0.0, 1.16449051, 0.44942959, 0.0, 0.0, 1.22877420, 0.52382607]
params_CCZ_TR_20_32_4 = [14.07398476, 0.0, 0.29031106, -0.71832208, -0.97988784, 0.0, 0.0, 2.68110899, -0.02390609, 0.0, 0.0, -0.35287082, 1.26692764, 0.0, 0.0, 1.24709712, 0.26109578, 0.0, 0.0, 0.71614189, -0.11447837, 0.0, 0.0, -1.49501634, -0.74618045, 0.0, 0.0, 1.39392175, 0.70998488, 0.0, 0.0, 0.75931407, -0.42381312, 0.0, 0.0, 0.56204234, 0.69361335]
params_list_CCZ_TR_32_4 = [params_CCZ_TR_8_32_4, params_CCZ_TR_10_32_4, params_CCZ_TR_12_32_4, params_CCZ_TR_14_32_4, params_CCZ_TR_16_32_4, params_CCZ_TR_18_32_4, params_CCZ_TR_20_32_4]

# eps-CCZ' Vnn=32, Vnnn=4
params_epsCCZ_TR_8_32_4 = [16.26547407, 0.0, -0.93373845, -0.29490584, 0.93338852, 0.0, 0.0, 2.38942089, 0.85760910, 0.0, 0.0, 0.38663762, -0.25533143]
params_epsCCZ_TR_10_32_4 = [14.02369187, 0.0, -1.10546750, -0.14842267, 4.08012347, 0.0, 0.0, 0.58624268, -0.11270254, 0.0, 0.0, 0.66021895, 0.86119552, 0.0, 0.0, 0.71275033, 0.30905395]
params_epsCCZ_TR_12_32_4 = [11.31844875, 0.0, -0.09607799, 1.40553502, 1.07291203, 0.0, 0.0, 2.02700124, -0.25448490, 0.0, 0.0, -0.02989352, 0.92557852, 0.0, 0.0, 0.95779995, -0.25737808, 0.0, 0.0, 0.68680772, -0.38473614]
params_epsCCZ_TR_14_32_4 = [14.15605566, 0.0, 0.03202490, 0.54696215, -0.32035084, 0.0, 0.0, -2.12995997, 0.57876309, 0.0, 0.0, 1.29275151, -1.26704797, 0.0, 0.0, 0.62870266, 0.54977527, 0.0, 0.0, 0.72378088, -0.30671976, 0.0, 0.0, -1.80291930, 0.96738121]
params_epsCCZ_TR_16_32_4 = [14.13299166, 0.0, -0.16665798, -0.91013984, 1.25260415, 0.0, 0.0, 0.18038350, -0.32302614, 0.0, 0.0, -0.00752275, 0.09099498, 0.0, 0.0, 0.14025430, -1.01617240, 0.0, 0.0, -1.17949746, 0.99318173, 0.0, 0.0, 0.15419564, -0.35336440, 0.0, 0.0, -0.50800862, 0.46032250]
params_epsCCZ_TR_18_32_4 = [12.52238851, 0.0, 0.61070441, 0.32368637, 0.94601455, 0.0, 0.0, -1.60275669, -2.78636140, 0.0, 0.0, 0.26140688, -0.65072538, 0.0, 0.0, -2.00087353, 1.10082381, 0.0, 0.0, -1.46870736, 0.30361845, 0.0, 0.0, 0.67498826, 0.26485312, 0.0, 0.0, -0.03587283, -0.18041541, 0.0, 0.0, 0.20015193, -0.06560793]
params_epsCCZ_TR_20_32_4 = [13.96412952, 0.0, -0.15112317, 3.06974041, -1.12704745, 0.0, 0.0, -1.60294537, 1.84391674, 0.0, 0.0, -0.02273785, 0.94424924, 0.0, 0.0, 0.06181891, -1.07166841, 0.0, 0.0, 0.14503440, 0.44409009, 0.0, 0.0, -1.46381749, 0.17083756, 0.0, 0.0, 0.43098004, -0.00011969, 0.0, 0.0, -0.42046077, -0.35522262, 0.0, 0.0, -0.17727205, 0.03327729]
params_list_epsCCZ_TR_32_4 = [params_epsCCZ_TR_8_32_4, params_epsCCZ_TR_10_32_4, params_epsCCZ_TR_12_32_4, params_epsCCZ_TR_14_32_4, params_epsCCZ_TR_16_32_4, params_epsCCZ_TR_18_32_4, params_epsCCZ_TR_20_32_4]

# CCZ' Vnn=32, Vnnn=0.5
params_CCZ_TR_8_32_05 = [26.32063881, 0.0, 0.49627810, -0.45633707, -10.48503079, 0.0, 0.0, -0.50462089, 3.56332654, 0.0, 0.0, 1.56906114, -0.83961124]
params_CCZ_TR_10_32_05 = [21.76924599, 0.0, 0.88976532, 0.96382294, -3.78317975, 0.0, 0.0, 0.78011122, -1.25198157, 0.0, 0.0, 1.06971407, -1.20271967, 0.0, 0.0, 1.50324801, 1.69445118]
params_CCZ_TR_12_32_05 = [17.21231500, 0.0, 0.26945124, 1.32220269, 3.01130028, 0.0, 0.0, 2.82178938, 0.49052818, 0.0, 0.0, 1.09951743, -0.47303594, 0.0, 0.0, -0.21096572, 1.51207710, 0.0, 0.0, 2.70493122, -0.37038808]
params_CCZ_TR_14_32_05 = [17.29056263, 0.0, 0.13136731, 0.51518347, 3.14519968, 0.0, 0.0, 0.17017720, 0.51830116, 0.0, 0.0, 1.06001814, -0.04243480, 0.0, 0.0, 0.33191145, -0.18451637, 0.0, 0.0, 3.27316141, -0.37585899, 0.0, 0.0, -1.30458992, 1.57944716]
params_CCZ_TR_16_32_05 = [17.12452016, 0.0, 0.30488665, 0.94418594, 1.64987984, 0.0, 0.0, -0.37041340, 1.35922297, 0.0, 0.0, 1.19535077, -0.68208015, 0.0, 0.0, 0.03352259, 0.38425023, 0.0, 0.0, -0.72412318, 1.65731539, 0.0, 0.0, 0.50229195, -0.36256618, 0.0, 0.0, 2.54254813, -0.01039096]
params_CCZ_TR_18_32_05 = [17.09341902, 0.0, 0.22790827, 0.99357893, 3.11030622, 0.0, 0.0, 2.28518476, 0.81549254, 0.0, 0.0, 1.11560505, -0.00996476, 0.0, 0.0, 0.10168435, -0.00391329, 0.0, 0.0, -0.12683984, -0.29515853, 0.0, 0.0, 0.49028122, -0.36319571, 0.0, 0.0, -2.03961653, 1.02862804, 0.0, 0.0, -0.69670505, 0.06292360]
params_CCZ_TR_20_32_05 = [17.05807000, 0.0, 0.28312859, 1.46450204, 2.95354098, 0.0, 0.0, -0.40873855, 0.06992277, 0.0, 0.0, 3.28816359, -1.37399474, 0.0, 0.0, 0.42206946, 0.48150305, 0.0, 0.0, 1.98792739, -0.34917942, 0.0, 0.0, -1.55574748, 1.36460042, 0.0, 0.0, 1.86206100, -0.00560477, 0.0, 0.0, -2.07817028, 1.05116154, 0.0, 0.0, 1.46583443, 0.07519329]
params_list_CCZ_TR_32_05 = [params_CCZ_TR_8_32_05, params_CCZ_TR_10_32_05, params_CCZ_TR_12_32_05, params_CCZ_TR_14_32_05, params_CCZ_TR_16_32_05, params_CCZ_TR_18_32_05, params_CCZ_TR_20_32_05]

# eps-CCZ' Vnn=32, Vnnn=0.5
params_epsCCZ_TR_8_32_05 = [19.54347130, 0.0, -0.92846312, 0.91934349, -0.80127308, 0.0, 0.0, 1.50298585, -1.63822626, 0.0, 0.0, 1.13128469, -0.96338565]
params_epsCCZ_TR_10_32_05 = [17.06613518, 0.0, -0.02827633, -0.02092351, -0.16613514, 0.0, 0.0, 0.18415403, -1.19968643, 0.0, 0.0, 3.68358995, -0.47570836, 0.0, 0.0, 0.56721489, -0.35827732]
params_epsCCZ_TR_12_32_05 = [17.03593069, 0.0, 0.26935234, -0.26867627, -1.75211679, 0.0, 0.0, 0.84437292, -1.24155666, 0.0, 0.0, 2.41298043, -1.00691922, 0.0, 0.0, 2.10490710, -0.08186184, 0.0, 0.0, -0.66188716, 0.65534436]
params_epsCCZ_TR_14_32_05 = [17.22072020, 0.0, 0.05859779, -0.06818385, -0.36649210, 0.0, 0.0, -2.09617076, -0.17740991, 0.0, 0.0, -0.48414449, -1.04674826, 0.0, 0.0, 0.33896992, -0.72402630, 0.0, 0.0, 2.38399033, -0.61114599, 0.0, 0.0, 0.66486914, 0.72656006]
params_epsCCZ_TR_16_32_05 = [17.39203493, 0.0, -0.01648847, -0.22146494, -0.00848886, 0.0, 0.0, 0.24371705, -1.14515518, 0.0, 0.0, 1.21187514, -0.20285513, 0.0, 0.0, 0.34607100, -0.48017014, 0.0, 0.0, 2.06848467, -0.43613503, 0.0, 0.0, 0.71911184, 0.58146211, 0.0, 0.0, 2.32022711, 0.16624854]
params_epsCCZ_TR_18_32_05 = [17.69888358, 0.0, -0.03672710, 0.61113721, 0.11807460, 0.0, 0.0, 0.43690683, -1.93864828, 0.0, 0.0, 1.44190528, -0.65210721, 0.0, 0.0, -1.31447368, -0.03781715, 0.0, 0.0, -1.56145949, 0.85188947, 0.0, 0.0, 0.99887414, 0.31226146, 0.0, 0.0, 2.28670350, 0.43589640, 0.0, 0.0, 0.89286733, -0.09426817]
params_epsCCZ_TR_20_32_05 = [17.23884714, 0.0, -0.06678283, -0.20467134, 0.18091951, 0.0, 0.0, 0.09307029, -1.14892564, 0.0, 0.0, 2.36821755, -0.75132049, 0.0, 0.0, 0.87039361, -0.02920037, 0.0, 0.0, -0.69746057, 0.22357348, 0.0, 0.0, -4.39668159, -0.29798247, 0.0, 0.0, 2.70013925, 0.22279590, 0.0, 0.0, 0.15935195, 0.14287704, 0.0, 0.0, 0.99458139, 0.10861701]
params_list_epsCCZ_TR_32_05 = [params_epsCCZ_TR_8_32_05, params_epsCCZ_TR_10_32_05, params_epsCCZ_TR_12_32_05, params_epsCCZ_TR_14_32_05, params_epsCCZ_TR_16_32_05, params_epsCCZ_TR_18_32_05, params_epsCCZ_TR_20_32_05]


# # # fidelities, Rydberg times (processed data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# CCZ' Vnn=32, Vnnn=4
nParams_CCZ_TR_32_4 = [8, 10, 12, 14, 16, 18, 20]
infidelities_nodecay_CCZ_TR_32_4 = [8.3665e-04, 1.7763e-05, 2.9809e-05, 8.9613e-07, 3.0022e-07, 5.6008e-06, 2.7303e-06]
infidelities_CCZ_TR_32_4 = [2.3166e-03, 9.1495e-04, 5.7196e-04, 5.2233e-04, 5.2388e-04, 5.1515e-04, 5.1737e-04]
TRs_CCZ_TR_32_4 = [14.828197, 8.975026, 5.422591, 5.214788, 5.236295, 5.095958, 5.146861]

# eps-CCZ' Vnn=32, Vnnn=4
nParams_epsCCZ_TR_32_4 = [8, 10, 12, 14, 16, 18, 20]
infidelities_nodecay_epsCCZ_TR_32_4 = [3.4878e-05, 1.8017e-05, 3.2920e-07, 1.5238e-06, 1.8802e-06, 1.9375e-06, 2.7120e-06]
infidelities_epsCCZ_TR_32_4 = [5.7586e-04, 4.8456e-04, 4.4029e-04, 3.9066e-04, 3.9065e-04, 3.8897e-04, 3.8771e-04]
TRs_epsCCZ_TR_32_4 = [5.410622, 4.665945, 4.399776, 3.891475, 3.887776, 3.870354, 3.850026]

# CCZ' Vnn=32, Vnnn=0.5
nParams_CCZ_TR_32_05 = [8, 10, 12, 14, 16, 18, 20]
infidelities_nodecay_CCZ_TR_32_05 = [1.9136e-04, 7.2420e-06, 7.0732e-06, 3.7913e-06, 6.1560e-07, 5.1396e-07, 1.1206e-06]
infidelities_CCZ_TR_32_05 = [1.4911e-03, 1.1841e-03, 9.4144e-04, 9.4012e-04, 9.3684e-04, 9.3642e-04, 9.3681e-04]
TRs_CCZ_TR_32_05 = [13.007814, 11.773931, 9.346825, 9.366416, 9.365385, 9.362225, 9.360043]

# eps-CCZ' Vnn=32, Vnnn=0.5
nParams_epsCCZ_TR_32_05 = [8, 10, 12, 14, 16, 18, 20]
infidelities_nodecay_epsCCZ_TR_32_05 = [4.6890e-05, 2.4097e-06, 1.6522e-06, 4.9635e-06, 9.0942e-07, 1.2098e-05, 1.0034e-06]
infidelities_epsCCZ_TR_32_05 = [9.5482e-04, 6.9891e-04, 6.9274e-04, 6.7353e-04, 6.5196e-04, 6.6428e-04, 6.6892e-04]
TRs_epsCCZ_TR_32_05 = [9.082740, 6.966381, 6.912261, 6.686881, 6.511659, 6.523039, 6.680401]


# # # fidelities as function of Rnnn (Vnnn) (processed data) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# CCZ' Vnn=32, Vnnn=32, 10 params
Rnnns_CCZ_32_32 = [0.9900, 0.9905, 0.9910, 0.9915, 0.9920, 0.9925, 0.9930, 0.9935, 0.9940, 0.9945, 0.9950, 0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000, 1.0005, 1.0010, 1.0015, 1.0020, 1.0025, 1.0030, 1.0035, 1.0040, 1.0045, 1.0050, 1.0055, 1.0060, 1.0065, 1.0070, 1.0075, 1.0080, 1.0085, 1.0090, 1.0095, 1.0100]
infids_CCZ_32_32 = [3.9799e-04, 3.9787e-04, 3.9767e-04, 3.9736e-04, 3.9707e-04, 3.9693e-04, 3.9683e-04, 3.9660e-04, 3.9626e-04, 3.9596e-04, 3.9581e-04, 3.9571e-04, 3.9549e-04, 3.9518e-04, 3.9496e-04, 3.9492e-04, 3.9496e-04, 3.9490e-04, 3.9476e-04, 3.9469e-04, 3.9482e-04, 3.9503e-04, 3.9516e-04, 3.9515e-04, 3.9517e-04, 3.9533e-04, 3.9559e-04, 3.9576e-04, 3.9576e-04, 3.9573e-04, 3.9584e-04, 3.9608e-04, 3.9631e-04, 3.9640e-04, 3.9641e-04, 3.9656e-04, 3.9690e-04, 3.9733e-04, 3.9767e-04, 3.9789e-04, 3.9816e-04]

# CCZ' Vnn=32, Vnnn=4, 14 params (params_CCZ_TR_14_32_4, params_epsCCZ_TR_14_32_4)
Rnnns_CCZ_32_4 = [0.9900, 0.9905, 0.9910, 0.9915, 0.9920, 0.9925, 0.9930, 0.9935, 0.9940, 0.9945, 0.9950, 0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000, 1.0005, 1.0010, 1.0015, 1.0020, 1.0025, 1.0030, 1.0035, 1.0040, 1.0045, 1.0050, 1.0055, 1.0060, 1.0065, 1.0070, 1.0075, 1.0080, 1.0085, 1.0090, 1.0095, 1.0100]
infids_CCZ_32_4 = [6.2021e-02, 5.3373e-02, 4.5567e-02, 3.8576e-02, 3.2365e-02, 2.6893e-02, 2.2114e-02, 1.7980e-02, 1.4440e-02, 1.1441e-02, 8.9315e-03, 6.8587e-03, 5.1727e-03, 3.8256e-03, 2.7720e-03, 1.9699e-03, 1.3808e-03, 9.7012e-04, 7.0721e-04, 5.6547e-04, 5.2233e-04, 5.5911e-04, 6.6081e-04, 8.1585e-04, 1.0158e-03, 1.2548e-03, 1.5297e-03, 1.8391e-03, 2.1834e-03, 2.5641e-03, 2.9838e-03, 3.4455e-03, 3.9525e-03, 4.5080e-03, 5.1146e-03, 5.7745e-03, 6.4891e-03, 7.2586e-03, 8.0825e-03, 8.9589e-03, 9.8851e-03]
infids_epsCCZ_32_4 = [6.5446e-04, 5.9933e-04, 5.6797e-04, 5.5391e-04, 5.5144e-04, 5.5562e-04, 5.6232e-04, 5.6823e-04, 5.7094e-04, 5.6890e-04, 5.6144e-04, 5.4864e-04, 5.3122e-04, 5.1031e-04, 4.8732e-04, 4.6378e-04, 4.4124e-04, 4.2123e-04, 4.0523e-04, 3.9463e-04, 3.9066e-04, 3.9431e-04, 4.0622e-04, 4.2661e-04, 4.5525e-04, 4.9149e-04, 5.3431e-04, 5.8248e-04, 6.3462e-04, 6.8932e-04, 7.4514e-04, 8.0065e-04, 8.5444e-04, 9.0514e-04, 9.5150e-04, 9.9248e-04, 1.0274e-03, 1.0559e-03, 1.0783e-03, 1.0954e-03, 1.1084e-03]

# CCZ' Vnn=32, Vnnn=0.5, 12 params (params_CCZ_TR_12_32_05, params_epsCCZ_TR_12_32_05)
Rnnns_CCZ_32_05 = [0.9900, 0.9905, 0.9910, 0.9915, 0.9920, 0.9925, 0.9930, 0.9935, 0.9940, 0.9945, 0.9950, 0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000, 1.0005, 1.0010, 1.0015, 1.0020, 1.0025, 1.0030, 1.0035, 1.0040, 1.0045, 1.0050, 1.0055, 1.0060, 1.0065, 1.0070, 1.0075, 1.0080, 1.0085, 1.0090, 1.0095, 1.0100]
infids_CCZ_32_05 = [6.2916e-03, 5.7508e-03, 5.2399e-03, 4.7591e-03, 4.3085e-03, 3.8870e-03, 3.4945e-03, 3.1315e-03, 2.7974e-03, 2.4913e-03, 2.2136e-03, 1.9644e-03, 1.7428e-03, 1.5484e-03, 1.3816e-03, 1.2422e-03, 1.1294e-03, 1.0428e-03, 9.8296e-04, 9.4951e-04, 9.4144e-04, 9.5882e-04, 1.0021e-03, 1.0706e-03, 1.1635e-03, 1.2810e-03, 1.4234e-03, 1.5900e-03, 1.7800e-03, 1.9937e-03, 2.2314e-03, 2.4922e-03, 2.7754e-03, 3.0815e-03, 3.4106e-03, 3.7619e-03, 4.1346e-03, 4.5291e-03, 4.9458e-03, 5.3837e-03, 5.8421e-03]
infids_epsCCZ_32_05 = [9.8749e-04, 9.5622e-04, 9.2695e-04, 8.9931e-04, 8.7359e-04, 8.5001e-04, 8.2811e-04, 8.0773e-04, 7.8931e-04, 7.7279e-04, 7.5767e-04, 7.4408e-04, 7.3241e-04, 7.2237e-04, 7.1357e-04, 7.0630e-04, 7.0084e-04, 6.9679e-04, 6.9386e-04, 6.9246e-04, 6.9274e-04, 6.9423e-04, 6.9677e-04, 7.0080e-04, 7.0638e-04, 7.1301e-04, 7.2062e-04, 7.2968e-04, 7.4017e-04, 7.5158e-04, 7.6390e-04, 7.7760e-04, 7.9263e-04, 8.0849e-04, 8.2517e-04, 8.4315e-04, 8.6239e-04, 8.8239e-04, 9.0312e-04, 9.2504e-04, 9.4818e-04]


# # # functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def postprocess_V_sensitivity(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params, decay, Rnnn_min, Rnnn_max, geometry):
    N = 41  # number of different interaction strengths to be tested
    Rnnn_array = np.linspace(Rnnn_min, Rnnn_max, N)
    if geometry == 'equilateral':
        Rnn_array = 0.5 * np.sqrt(3 + Rnnn_array**2)
    elif geometry == 'right':
        Rnn_array = (1/np.sqrt(2)) * np.sqrt(1 + Rnnn_array**2)
    elif geometry == 'line':
        Rnn_array = Rnnn_array.copy()
    else:
        Rnn_array = np.ones(N)
    Vnnn_array = Vnnn / Rnnn_array**6
    Vnn_array = Vnn / Rnn_array**6

    infidelities = []
    for Vnn_i, Vnnn_i in zip(Vnn_array, Vnnn_array):
        f, _ = pulse_verification.verify(n_atoms, Vnn_i, Vnnn_i, theta, eps, lamb, delta, kappa, pulse, params, decay)
        infidelities.append(1 - f)

    print()
    print('Rnnn  : [' + ', '.join('{rnnn:.8f}'.format(rnnn=rnnn) for rnnn in Rnnn_array) + ']')
    print('Rnn   : [' + ', '.join('{rnn:.8f}'.format(rnn=rnn) for rnn in Rnn_array) + ']')
    print('Vnnn  : [' + ', '.join('{vnnn:.8f}'.format(vnnn=vnnn) for vnnn in Vnnn_array) + ']')
    print('Vnn   : [' + ', '.join('{vnn:.8f}'.format(vnn=vnn) for vnn in Vnn_array) + ']')
    print('infid : [' + ', '.join('{infid:.4e}'.format(infid=infid) for infid in infidelities) + ']')
    return Rnnn_array, infidelities


def plots_combined_row1(nParams_list_A, TRs_list_A, eps_TRs_list_A, infids_list_A, eps_infids_list_A,
                   nParams_list_B, TRs_list_B, eps_TRs_list_B, infids_list_B, eps_infids_list_B):

    fig, ax = plt.subplots(1, 3, width_ratios=[1.2, 1, 1], figsize=(7.0, 1.7), layout='constrained')
    #
    ax[0].spines['bottom'].set_color('white')
    ax[0].spines['top'].set_color('white')
    ax[0].spines['right'].set_color('white')
    ax[0].spines['left'].set_color('white')
    ax[0].tick_params(axis='x', colors='white')
    ax[0].tick_params(axis='y', colors='white')
    ax[0].yaxis.label.set_color('white')
    ax[0].xaxis.label.set_color('white')
    #
    ax0a = ax[1].twinx()
    color = 'tab:blue'
    color2 = 'tab:red'
    ccz_infid_A, = ax[1].plot(nParams_list_A, infids_list_A, 'o', markersize=5, color=color, markerfacecolor='none', markeredgewidth=1.5)
    epsccz_infid_A, = ax[1].plot(nParams_list_A, eps_infids_list_A, 's', markersize=5, color=color, markerfacecolor='none', markeredgewidth=1.5)
    ccz_tr_A, = ax0a.plot(nParams_list_A, TRs_list_A, '+', markersize=5, color=color2)
    epsccz_tr_A, = ax0a.plot(nParams_list_A, eps_TRs_list_A, '^', markersize=3.5, color=color2)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax0a.yaxis.tick_left()
    ax0a.yaxis.set_label_position("left")
    ax[1].set_xlabel(r'# parameters', fontsize=10)
    ax0a.set_ylabel(r'$\Omega_0 T_R$', fontsize=10, color=color2)
    ax[1].tick_params(axis='both', labelsize=8)
    ax0a.tick_params(axis='both', labelsize=8)
    ax[1].set_xlim(7.5, 20.5)
    ax[1].set_ylim(0.00034, 0.00125)
    ax0a.set_ylim(3.0, 12.5)
    ax[1].set_xticks([8, 12, 16, 20])
    ax[1].set_yticks([0.0003, 0.0006, 0.0009, 0.0012])
    ax0a.set_yticks([3.0, 6.0, 9.0, 12.0])
    ax[1].set_yticklabels(['', '', '', ''])
    ax[1].yaxis.get_offset_text().set_fontsize(8)
    ax[1].legend([(ccz_tr_A, ccz_infid_A), (epsccz_tr_A, epsccz_infid_A)], ['CCZ\'', '$\epsilon$-CCZ\''], numpoints=1,
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=8, handlelength=1.8, handletextpad=0.8, borderpad=0.3, borderaxespad=0.2, loc='upper right')
    ax[1].grid()
    #
    ax1a = ax[2].twinx()
    ccz_infid_B, = ax[2].plot(nParams_list_B, infids_list_B, 'o', markersize=5, color=color, markerfacecolor='none', markeredgewidth=1.5)
    epsccz_infid_B, = ax[2].plot(nParams_list_B, eps_infids_list_B, 's', markersize=5, color=color, markerfacecolor='none', markeredgewidth=1.5)
    ccz_tr_B, = ax1a.plot(nParams_list_B, TRs_list_B, '+', markersize=5, color=color2)
    epsccz_tr_B, = ax1a.plot(nParams_list_B, eps_TRs_list_B, '^', markersize=3.5, color=color2)
    ax[2].yaxis.tick_right()
    ax[2].yaxis.set_label_position("right")
    ax1a.yaxis.tick_left()
    ax1a.yaxis.set_label_position("left")
    ax[2].set_xlabel(r'# parameters', fontsize=10)
    ax[2].set_ylabel(r'$1-F$', fontsize=10, color=color)
    ax[2].tick_params(axis='both', labelsize=8)
    ax1a.tick_params(axis='both', labelsize=8)
    ax[2].set_xlim(7.5, 20.5)
    ax[2].set_ylim(0.00030, 0.00125)
    ax1a.set_ylim(3.0, 12.5)
    ax[2].set_xticks([8, 12, 16, 20])
    ax[2].set_yticks([0.0003, 0.0006, 0.0009, 0.0012])
    ax1a.set_yticks([3.0, 6.0, 9.0, 12.0])
    ax1a.set_yticklabels(['', '', '', ''])
    ax[2].ticklabel_format(style='sci', scilimits=(-4, -4), axis='y', useMathText=True)
    ax[2].yaxis.get_offset_text().set_fontsize(8)
    ax[2].legend([(ccz_tr_B, ccz_infid_B), (epsccz_tr_B, epsccz_infid_B)], ['CCZ\'', '$\epsilon$-CCZ\''], numpoints=1,
                 handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=8, handlelength=1.8, handletextpad=0.8, borderpad=0.3, borderaxespad=0.2, loc='lower left')
    ax[2].grid()
    fig.text(0.005, 0.93, '(a)', fontsize=10)
    fig.text(0.365, 0.93, '(c)', fontsize=10)
    plt.show()
    fig.savefig('Figs/CCZ_finiteV_subplot1.pdf', bbox_inches='tight', pad_inches=0.02)


def plots_combined_row2(Vnnn0_B, dnnns_list_B, infids_list_B,
                        Vnnn0_C, dnnns_list_C, infids_list_C, eps_infids_list_C,
                        Vnnn0_D, dnnns_list_D, infids_list_D, eps_infids_list_D):

    def V2R_B(V):
        return (Vnnn0_B/V)**(1/6)

    def R2V_B(R):
        return Vnnn0_B/R**6

    def V2R_C(V):
        return (Vnnn0_C/V)**(1/6)

    def R2V_C(R):
        return Vnnn0_C/R**6

    def V2R_D(V):
        return (Vnnn0_D/V)**(1/6)

    def R2V_D(R):
        return Vnnn0_D/R**6

    fig, ax = plt.subplots(1, 3, sharey=True, width_ratios=[1, 1, 1], figsize=(7.0, 1.7), layout='constrained')
    #
    ax[0].semilogy(dnnns_list_B, infids_list_B, linestyle='solid', color='tab:orange', label='CCZ\'')
    ax[0].semilogy([], [], linestyle='dashed', color='tab:purple', label='$\epsilon$-CCZ\'')
    ax1a = ax[0].secondary_xaxis('top', functions=(R2V_B, V2R_B))
    ax[0].set_xlabel(r'$d_{\mathrm{nnn}}/d^0_{\mathrm{nnn}}$', fontsize=10)
    ax1a.set_xlabel(r'$V_{\mathrm{nnn}}/(\hbar\Omega_0)$', fontsize=10)
    ax[0].set_ylabel(r'$1-F$', fontsize=10)
    ax[0].tick_params(axis='both', which='both', labelsize=8)
    ax1a.tick_params(axis='both', labelsize=8, direction='in', pad=-15, zorder=10)
    ax1a.set_zorder(5)
    ax[0].set_xticks([0.99, 1, 1.01])
    ax1a.set_xticks([33.5, 32, 30.5])
    ax[0].legend(fontsize=8, loc='center', handlelength=1.8, handletextpad=0.8, borderpad=0.3, borderaxespad=0.2)
    ax[0].set_xlim(0.99, 1.01)
    ax[0].set_ylim(3e-4, 1e-1)
    ax[0].grid()
    #
    ax[1].semilogy(dnnns_list_C, infids_list_C, linestyle='solid', color='tab:orange', label='CCZ\'')
    ax[1].semilogy(dnnns_list_C, eps_infids_list_C, linestyle='dashed', color='tab:purple', label='$\epsilon$-CCZ\'')
    ax2a = ax[1].secondary_xaxis('top', functions=(R2V_C, V2R_C))
    ax[1].set_xlabel(r'$d_{\mathrm{nnn}}/d^0_{\mathrm{nnn}}$', fontsize=10)
    ax2a.set_xlabel(r'$V_{\mathrm{nnn}}/(\hbar\Omega_0)$', fontsize=10)
    ax[1].tick_params(axis='both', which='both', labelsize=8)
    ax2a.tick_params(axis='both', labelsize=8, direction='in', pad=-15, zorder=10)
    ax2a.set_zorder(5)
    ax[1].set_xticks([0.99, 1, 1.01])
    ax2a.set_xticks([4.15, 4, 3.85])
    ax[1].set_xlim(0.99, 1.01)
    ax[1].set_ylim(3e-4, 1e-1)
    ax[1].grid()
    #
    ax[2].semilogy(dnnns_list_D, infids_list_D, linestyle='solid', color='tab:orange', label='CCZ\'')
    ax[2].semilogy(dnnns_list_D, eps_infids_list_D, linestyle='dashed', color='tab:purple', label='$\epsilon$-CCZ\'')
    ax3a = ax[2].secondary_xaxis('top', functions=(R2V_D, V2R_D))
    ax[2].set_xlabel(r'$d_{\mathrm{nnn}}/d^0_{\mathrm{nnn}}$', fontsize=10)
    ax3a.set_xlabel(r'$V_{\mathrm{nnn}}/(\hbar\Omega_0)$', fontsize=10)
    ax[2].tick_params(axis='both', which='both', labelsize=8)
    ax3a.tick_params(axis='both', labelsize=8, direction='in', pad=-15, zorder=10)
    ax3a.set_zorder(5)
    ax3a.set_axisbelow(False)
    ax[2].set_xticks([0.99, 1, 1.01])
    ax[2].set_xlim(0.99, 1.01)
    ax[2].set_ylim(3e-4, 1e-1)
    ax[2].grid()
    fig.text(0.005, 0.93, '(b)', fontsize=10)
    plt.show()
    fig.savefig('Figs/CCZ_finiteV_subplot2.pdf', bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)

    # number of atoms participating in the gate (2, 3 or 4)
    n_atoms = 3

    # Rydberg interaction strengths
    Vnn = 32.0
    Vnnn = 32.0
    decay = 0.0001
    geometry = 'equilateral'  # 'right', 'line'

    # target gate phases
    theta = np.pi
    eps = np.pi
    lamb = np.pi
    delta = 0
    kappa = 0

    # pulse type
    pulse = pulses_qutip.pulse_phase_sin_cos_crab_smooth

    # optimization parameters
    params = params_CCZ_TR_10_32_32
    params_list = params_list_CCZ_TR_32_4

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # pulse_postprocessing.postprocess_pulses(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params_list, decay)
    # postprocess_V_sensitivity(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params, decay, 0.99, 1.01, geometry)

    plots_combined_row1(nParams_CCZ_TR_32_4, TRs_CCZ_TR_32_4, TRs_epsCCZ_TR_32_4, infidelities_CCZ_TR_32_4, infidelities_epsCCZ_TR_32_4, nParams_CCZ_TR_32_05, TRs_CCZ_TR_32_05, TRs_epsCCZ_TR_32_05, infidelities_CCZ_TR_32_05, infidelities_epsCCZ_TR_32_05)
    plots_combined_row2(32, Rnnns_CCZ_32_32, infids_CCZ_32_32, 4, Rnnns_CCZ_32_4, infids_CCZ_32_4, infids_epsCCZ_32_4, 0.5, Rnnns_CCZ_32_05, infids_CCZ_32_05, infids_epsCCZ_32_05)
