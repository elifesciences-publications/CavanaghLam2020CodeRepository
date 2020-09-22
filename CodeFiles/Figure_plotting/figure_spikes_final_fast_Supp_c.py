#!/usr/local/cluster/software/builds/njc2/python/Python-2.7.5/Installation/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
import analysis
import scipy.stats as sp_stat   # Coefficient of variation
import os
import copy
import matplotlib.cm as matplotlib_cm
import fig_funs
matplotlib.rc('xtick', labelsize=7)
matplotlib.rc('ytick', labelsize=7)
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm
font = FontProperties(family="Arial")
hfont = {'fontname':'Arial'}
import matplotlib as mpl
mpl.rc('font',family='Arial')
mpl.rcParams['lines.linewidth'] = 1.5
plt.rcParams["font.family"] = "Arial"
# mpl.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]                                                    # In order to use bold font in latex expressions
# matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
from mpl_toolkits.axes_grid1 import make_axes_locatable                                                                 # To make imshow 2D-plots to have color bars at the same height as the figure
from scipy.interpolate import UnivariateSpline
import MFAMPA_functions_WongWang2006_Hunt_Noisy_Input
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, IndexLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.mlab as mlab
import analysis
# mpl.style.use('classic')                                                                                              # One potential way to resolve figure differences across computers...

path_cwd = './'         #Current Directory
path_original = '../Figures_Biol_Psych/'         #Current Directory

#-----------------------------------------------------------------------------------------------------------------------### Initializtion
### Load spike rasters and make firing rate profiles ###

## First define paths to look into
path_joint = ''#''/'                                                                                                        # Joint between various folder sub-name strings
suffix_joint = "_"                                                                                                      # Joint between strings of suffix sub-parts
empty_joint = ""
foldername_joint = "_"

## Bar plot parameters
bar_width = 0.8
bar_opacity = 0.75

### --------------------------------------------------------------------------------------------------------------------
### Initialization, parameters

max1 = 8.5/2.54
max15 = 11.6/2.54
max2 = 17.6/2.54
fontsize_tick   = 7
fontsize_legend = 8
fontsize_fig_label = 10


colors_set1 = {'red':(228./255, 26./255, 28./255),
          'blue':(55./255, 126./255, 184./255),
          'green':(77./255, 175./255, 74./255),
          'purple':(152./255, 78./255, 163./255),
          'orange':(255./255, 127./255, 0./255),
          'yellow':(255./255, 255./255, 51./255),
          'brown':(166./255, 86./255, 40./255),
          'pink':(247./255, 129./255, 191./255),
          'gray':(153./255,153./255,153./255)}

colors_dark2 = {'teal':(27./255, 158./255, 119./255),
          'orange':(217./255, 95./255, 2./255),
          'purple':(117./255, 112./255, 179./255),
          'pink':(231./255, 41./255, 138./255),
          'green':(102./255, 166./255, 30./255),
          'yellow':(230./255, 171./255, 2./255),
          'brown':(166./255, 118./255, 29./255),
          'gray': (102./255, 102./255, 102./255)}

colors_set3 = {'cyan':(141./255, 211./255, 199./255),
          'yellow':(255./255, 255./255, 153./255),
          'purple':(190./255, 186./255, 218./255),
          'pink':(251./255, 128./255, 114./255),
          'blue':(128./255, 177./255, 211./255),
          'orange':(253./255, 180./255, 98./255),
          'green': (179./255, 222./255, 105./255)}

colors_set4 = {'cyan':(166./255, 206./255, 227./255),
          'blue':(31./255, 120./255, 180./255),
          'green_l':(178./255, 223./255, 138./255),
          'green_l':(178./255, 223./255, 138./255),
          'green':(51./255, 160./255, 44./255),
          'pink':(251./255, 154./255, 153./255),
          'red':(227./255, 26./255, 28./255),
          'orange_l':(253./255, 191./255, 111./255),
          'orange':(255./255, 127./255, 0./255),
          'purple':(202./255, 178./255, 214./255)}

colors_div1 = {'pink3':(197./255, 27./255, 125./255),
          'pink2':(233./255, 163./255, 201./255),
          'pink1':(253./255, 224./255, 239./255),
          'green0':(247./255, 247./255, 247./255),
          'green1':(230./255, 245./255, 208./255),
          'green2':(161./255, 215./255, 106./255),
          'green3':(77./255, 146./255, 33./255)}

colors_div2 = {'brown4':(84./255, 48./255, 5./255),
          'brown3':(140./255, 81./255, 10./255),
          'brown2':(216./255, 179./255, 101./255),
          'brown1':(246./255, 232./255, 195./255),
          'blue0':(245./255, 245./255, 245./255),
          'blue1':(199./255, 234./255, 229./255),
          'blue2':(90./255, 180./255, 172./255),
          'blue3':(1./255, 102./255, 94./255),
          'blue4':(0./255, 60./255, 48./255)}

colors_div3 = {'red5':(103./255, 0./255, 31./255),
          'red4':(178./255, 24./255, 43./255),
          'red3':(214./255, 96./255, 77./255),
          'red2':(244./255, 165./255, 130./255),
          'red1':(253./255, 219./255, 199./255),
          'blue0':(247./255, 247./255, 247./255),
          'blue1':(209./255, 229./255, 240./255),
          'blue2':(146./255, 197./255, 222./255),
          'blue3':(67./255, 147./255, 195./255),
          'blue4':(33./255, 102./255, 172./255),
          'blue5':(5./255, 48./255, 97./255)}

colors_div4 = {'red5':(158./255, 1./255, 66./255),
          'red4':(213./255, 62./255, 79./255),
          'red3':(244./255, 109./255, 67./255),
          'red2':(253./255, 174./255, 97./255),
          'red1':(254./255, 224./255, 139./255),
          'blue0':(255./255, 255./255, 191./255),
          'blue1':(230./255, 245./255, 152./255),
          'blue2':(171./255, 221./255, 164./255),
          'blue3':(102./255, 194./255, 165./255),
          'blue4':(50./255, 136./255, 189./255),
          'blue5':(94./255, 79./255, 162./255)}


label_list     = ['Control', 'Higher total evi', 'Lower total evi', 'Sensory Deficit']         #Manually used variable.
color_list = [colors_set1['green'], colors_set1['purple'], colors_set1['orange'], colors_set1['brown']]
label_list_expt     = ['Control', 'Lowered E/I', 'Elevated E/I']         #Manually used variable.
color_list_expt = [colors_div3['blue4'], colors_div3['red4']]
color_choice_bar = colors_dark2['gray']
color_mean_var_beta = [colors_div4['blue3'], colors_div1['pink2']]
color_bias_beta = colors_set1['gray']
color_mean_supp_choice = [colors_set1['blue'], colors_set3['blue']]
color_NB = [colors_div2['brown3'], colors_set3['blue']]
color_NBE = [colors_div2['brown3'], colors_set3['blue'], colors_dark2['gray']]

xbuf0 = 0.11
ybuf0 = 0.08
ybuf1 = 4.*ybuf0









########################################################################################################################
########################################################################################################################
### Figure 1: Conceptual/Schematics

## Example trial   (No values are <0.01 or >0.99, and both streams have SD within 0.04 tolerance of 0.12/0.24 (for narrow/broad). However, no constraints on the mean values here.)
eg_list_narrow = np.array([0.57609322, 0.69456108, 0.60317681, 0.42224414, 0.64392424, 0.47090883, 0.39879171, 0.30032174])
eg_list_broad  = np.array([0.28989753, 0.72108362, 0.5550954, 0.53767944, 0.30952643, 0.17715214, 0.78755439, 0.26728954])

# Psychometric function over correct/not (see figures below across narrow/broad), averaged across monkeys
# Log-Spaced # the two values right before and after the d_evidence=0 element encodes data with small but not exactly 0 evidence (narrow/broad for before/after). Their d_evidence value is wrong, and is chosen for the most suitable location on ax_0.
d_evidence_A_non_drug_list_corr =  100.*np.array([0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_A_list_non_drug_corr =  np.array([0.555555555555556, 0.540243196294152, 0.647308781869688, 0.649043869516310, 0.709251101321586, 0.786026200873362, 0.853876185164529, 0.913225613405147, 0.953953953953954, 0.973890339425587, 0.962264150943396])  # Log-Spaced.
ErrBar_P_corr_A_list_non_drug_corr = np.array([0.117121394821051, 0.0119925784122016, 0.0179825270396286, 0.0160070964633630, 0.0134791069063128, 0.0110638287214748, 0.00834195430689584, 0.00688646894089649, 0.00663097108024447, 0.00814809820978469, 0.0261749753572827])
n_dx_corr_A_list = np.array([18, 1727, 706, 889, 1135, 1374, 1793, 1671, 999, 383, 53])
d_evidence_H_non_drug_list_corr =  100.*np.array([0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_H_list_non_drug_corr =  np.array([0.409090909090909, 0.559055118110236, 0.602419354838710, 0.627653123104912, 0.687532603025561, 0.771630615640599, 0.815100154083205, 0.884286203355015, 0.923119777158774, 0.966772151898734, 0.961904761904762])  # Log-Spaced.
ErrBar_P_corr_H_list_non_drug_corr = np.array([0.104823561107156, 0.00899315307184037, 0.0138979656133209, 0.0119048349041193, 0.0105861476499721, 0.00856162958209245, 0.00681500865424198, 0.00591865761931849, 0.00628787816340555, 0.00712942714022833, 0.0186812844795725])
n_dx_corr_H_list = np.array([22, 3048, 1240, 1649, 1917, 2404, 3245, 2921, 1795, 632, 105])

## Separate by high/low context
d_evidence_A_non_drug_list_corr_chooseHigh =  100.*np.array([0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
d_evidence_A_non_drug_list_corr_chooseLow =  100.*np.array([0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_A_list_non_drug_corr_chooseHigh =  np.array([0.600000000000000, 0.532435740514076, 0.648437500000000, 0.650526315789474, 0.717277486910995, 0.791044776119403, 0.864130434782609, 0.920332936979786, 0.950592885375494, 0.975757575757576, 1])  # Log-Spaced.
P_corr_A_list_non_drug_corr_chooseLow =  np.array([0.500000000000000, 0.547252747252747, 0.645962732919255, 0.647342995169082, 0.701067615658363, 0.780219780219780, 0.843069873997709, 0.906024096385542, 0.957403651115619, 0.972477064220184, 0.952380952380952])  # Log-Spaced.
ErrBar_P_corr_A_list_non_drug_corr_chooseHigh = np.array([0.154919333848297, 0.0174559393832558, 0.0243651854489739, 0.0218772570326667, 0.0188124988009899, 0.0149759163644396, 0.0112968361512064, 0.00933714037361844, 0.00963422461287251, 0.0119733869310928, 0])
ErrBar_P_corr_A_list_non_drug_corr_chooseLow = np.array([0.176776695296637, 0.0165006552609505, 0.0266501885254739, 0.0234824449680562, 0.0193107116262321, 0.0164071467931687, 0.0123105604538809, 0.0101283589195277, 0.00909515841447438, 0.0110804861504227, 0.0328602647305883])
d_evidence_H_non_drug_list_corr_chooseHigh =  100.*np.array([0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
d_evidence_H_non_drug_list_corr_chooseLow =  100.*np.array([0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_H_list_non_drug_corr_chooseHigh =  np.array([0.444444444444444, 0.563380281690141, 0.597325408618128, 0.623921085080148, 0.715163934426230, 0.793929712460064, 0.839122486288848, 0.897214854111406, 0.937297297297297, 0.961538461538462, 1])  # Log-Spaced.
P_corr_H_list_non_drug_corr_chooseLow =  np.array([0.384615384615385, 0.555282555282555, 0.608465608465609, 0.631264916467780, 0.658873538788523, 0.747395833333333, 0.790523690773067, 0.870488322717622, 0.908045977011494, 0.971875000000000, 0.957446808510638])  # Log-Spaced.
ErrBar_P_corr_H_list_non_drug_corr_chooseHigh = np.array([0.165634664999984, 0.0131615889033115, 0.0189049300879554, 0.0170095920089641, 0.0144469238217731, 0.0114313244648240, 0.00906997441499408, 0.00782010771434524, 0.00797096581190586, 0.0108872791744673, 0])
ErrBar_P_corr_H_list_non_drug_corr_chooseLow = np.array([0.134932002970312, 0.0123160628790687, 0.0204979980000709, 0.0166663711005708, 0.0154548190941723, 0.0128017502889045, 0.0101606787276604, 0.00893233202273572, 0.00979669664268987, 0.00924222382177985, 0.0208189810548971])


# Psychophyical Kernal
i_PK_list = np.arange(1,8+1)
t_PK_list = 0.125 + 0.25*np.arange(8)
n_A_non_drug = 41.                                                                                                      # Alfie, 41 runs
n_H_non_drug = 63.                                                                                                      # Henry, 63 runs
PK_A_nondrug = np.array([3.97349317466863, 3.14577917590026, 3.14308934214697, 3.09270194108838, 2.65869626053626, 2.58228332627445, 2.26243327684102, 2.32548588191760])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_nondrug = np.array([3.24972294613704, 2.87215850443560, 2.41144038611466, 2.58619140237903, 2.25506241804606, 2.28328415494767, 1.97777876818272, 2.08155718671906])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_A_nondrug_errbar = np.array([0.114436877777907, 0.104349370934396, 0.107810651060901, 0.107321477770204, 0.103722082853755, 0.101065259589198, 0.101660772227339, 0.101973548343162])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data).
PK_H_nondrug_errbar = np.array([0.0777015532433701, 0.0738638886817392, 0.0738767825058227, 0.0749264426465390, 0.0723778004740631, 0.0716455370014343, 0.0717771595016059, 0.0713933395660157])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data).
## Separate by high/low context
PK_A_nondrug_chooseHigh = 100.*np.array([0.0415120827656887, 0.0369897067610839, 0.0308084653964484, 0.0316241837693435, 0.0281374033705074, 0.0269035170677060, 0.0234889175639483, 0.0232927424496404])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_A_nondrug_errbar_chooseHigh = 100.*np.array([0.00164266051629804, 0.00157521223993478, 0.00153591903506685, 0.00152395202625627, 0.00149456781408940, 0.00142116523940449, 0.00141956846980700, 0.00143158151035871])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data).
PK_A_nondrug_chooseLow = 100.*np.array([0.0385569291119254, 0.0264379245844889, 0.0326532501594194, 0.0307611439556908, 0.0250177597848587, 0.0245399084453607, 0.0221324030620475, 0.0237705581545699])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_A_nondrug_errbar_chooseLow = 100.*np.array([0.00160489849871266, 0.00140327121199686, 0.00153838231277541, 0.00152245369237610, 0.00144903458270546, 0.00144758064369922, 0.00146601943687889, 0.00147368042665260])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data).
PK_H_nondrug_chooseHigh = 100.*np.array([0.0309226848816748, 0.0310076507644510, 0.0256893894856451, 0.0286456751463619, 0.0249813791868224, 0.0272934332656132, 0.0219013641145492, 0.0237038873608619])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_nondrug_errbar_chooseHigh = 100.*np.array([0.00110133180656620, 0.00109407017208870, 0.00106171007528859, 0.00109140347226404, 0.00105274968428727, 0.00103566027251943, 0.00103675524981884, 0.00103512457641231])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data).
PK_H_nondrug_chooseLow = 100.*np.array([0.0348506398131391, 0.0263055027669680, 0.0226999787201196, 0.0232049427615248, 0.0202688567589456, 0.0181990597662468, 0.0178733955634692, 0.0182264920920492])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_nondrug_errbar_chooseLow = 100.*np.array([0.00110930150927949, 0.00101084676330797, 0.00103309764653882, 0.00103275807007954, 0.00100421641907450, 0.00100603040580200, 0.00100584746630927, 0.000995978965689708])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data).


def Psychometric_fit(params_pm, pm_fit2, x_list):
    prob_corr_fit = 0.5 + 0.5*(1. - np.exp(-(x_list/100/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    to_min = -sum(np.log(prob_corr_fit)*pm_fit2) - sum(np.log(1.-prob_corr_fit)*(1.-pm_fit2))                                                          # Maximum Likelihood Estimator
    return to_min
def Psychometric_function(params_pm, x_list):
    prob_corr_fit = 0.5 + 0.5*(1. - np.exp(-(x_list/100/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    return prob_corr_fit
## Fitting Psychometric Functions
x_list_psychometric = np.arange(1, 50, 1)                                                                        # See figure_psychometric_function_fit.py, esp lines 322-527
x0_psychometric = 0.
## non-binned MLE (i.e. done using literal net evidence, via matlab). See Psychometric_function_fit_NonDrugDays_NL.m. Also note there is no shift parameters.
psychometric_params_A_non_drug_corr = [0.064192932981096,1.091447981979582]
psychometric_params_H_non_drug_corr = [0.075250941223039,1.038829540311416]
psychometric_params_A_non_drug_corr_chooseHigh = [0.062331954080146,1.084217189531774]
psychometric_params_A_non_drug_corr_chooseLow = [0.066206626208686,1.103108981931701]
psychometric_params_H_non_drug_corr_chooseHigh = [0.068496360987983,1.027755441710905]
psychometric_params_H_non_drug_corr_chooseLow = [0.082734668492951,1.063458793512583]












## Define subfigure domain.
figsize = (max1,1.*max1)

width1_11 = 0.32; width1_12 = width1_11
width1_21 = width1_11; width1_22 = width1_21
x1_11 = 0.135; x1_12 = x1_11 + width1_11 + 1.7*xbuf0
x1_21 = x1_11; x1_22 = x1_12
height1_11 = 0.3; height1_12 = height1_11
height1_21= height1_11;  height1_22 = height1_21
y1_11 = 0.62; y1_12 = y1_11
y1_21 = y1_11 - height1_21 - 2.35*ybuf0; y1_22 = y1_21

rect1_11_0 = [x1_11, y1_11, width1_11*0.05, height1_11]
rect1_11 = [x1_11+width1_11*0.2, y1_11, width1_11*(1-0.2), height1_11]
rect1_12_0 = [x1_12, y1_12, width1_12*0.05, height1_12]
rect1_12 = [x1_12+width1_12*0.2, y1_12, width1_12*(1-0.2), height1_12]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]



##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.01, 0.915, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015+x1_12-x1_11, 0.915, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.01, 0.915 + y1_22 - y1_12, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015+x1_22-x1_21, 0.915 + y1_22 - y1_12, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.185, 0.96, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.695, 0.96, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
bar_width_compare3 = 1.



## rect1_11: Psychometric function (over dx_corr), monkey A
ax_0   = fig_temp.add_axes(rect1_11_0)
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
## Choose High
ax.errorbar( d_evidence_A_non_drug_list_corr_chooseHigh[2:],    P_corr_A_list_non_drug_corr_chooseHigh[2:], ErrBar_P_corr_A_list_non_drug_corr_chooseHigh[2:], color='k', markerfacecolor='red', ecolor='red', fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_A_non_drug_list_corr_chooseHigh[1], P_corr_A_list_non_drug_corr_chooseHigh[1], ErrBar_P_corr_A_list_non_drug_corr_chooseHigh[1], color='k', markerfacecolor='red', ecolor='red', marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(x_list_psychometric, Psychometric_function(psychometric_params_A_non_drug_corr_chooseHigh, x_list_psychometric), color='red', ls='-', clip_on=False, zorder=2, label='Choose Tall')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(x0_psychometric, Psychometric_function(psychometric_params_A_non_drug_corr_chooseHigh, x0_psychometric), s=15., color='red', marker='_', clip_on=False, zorder=2, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
## Choose Low
ax.errorbar( d_evidence_A_non_drug_list_corr_chooseLow[2:],    P_corr_A_list_non_drug_corr_chooseLow[2:], ErrBar_P_corr_A_list_non_drug_corr_chooseLow[2:], color='k', markerfacecolor='blue', ecolor='blue', fmt='.', zorder=3, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_A_non_drug_list_corr_chooseLow[1], P_corr_A_list_non_drug_corr_chooseLow[1], ErrBar_P_corr_A_list_non_drug_corr_chooseLow[1], color='k', markerfacecolor='blue', ecolor='blue', marker='.', zorder=3, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(x_list_psychometric, Psychometric_function(psychometric_params_A_non_drug_corr_chooseLow, x_list_psychometric), color='blue', ls='-', clip_on=False, zorder=1, label='Choose Short')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(x0_psychometric, Psychometric_function(psychometric_params_A_non_drug_corr_chooseLow, x0_psychometric), s=15., color='blue', marker='_', clip_on=False, zorder=1, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])

ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=-2.)
ax_0.set_ylim([0.48,1.])
ax.set_ylim([0.48,1.])
ax_0.set_xlim([-1.,1.])
ax.set_xlim([1,50])
ax_0.set_xticks([0.])
ax.xaxis.set_ticks([1, 10])
ax_0.set_yticks([0.5, 1.])
ax_0.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax_0.yaxis.set_minor_locator(minorLocator)
ax.set_yticks([])
ax_0.tick_params(direction='out', pad=1.5)
ax_0.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
## Add breakmark = wiggle
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.0968
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(-0.5,0.7), fontsize=fontsize_legend-2, frameon=False, ncol=1, markerscale=0., columnspacing=-0.1, handletextpad=0.)
for color,text,item in zip(['red','blue'], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)
ax.text(-20.5, 0.0105, r'$\times \mathregular{10^{-2}}$', fontsize=fontsize_tick-1.)


## rect1_12: Psychometric function (over dx_corr), monkey H
ax_0   = fig_temp.add_axes(rect1_12_0)
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
## Choose High
ax.errorbar( d_evidence_H_non_drug_list_corr_chooseHigh[2:],    P_corr_H_list_non_drug_corr_chooseHigh[2:], ErrBar_P_corr_H_list_non_drug_corr_chooseHigh[2:], color='k', markerfacecolor='red', ecolor='red', fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_H_non_drug_list_corr_chooseHigh[1], P_corr_H_list_non_drug_corr_chooseHigh[1], ErrBar_P_corr_H_list_non_drug_corr_chooseHigh[1], color='k', markerfacecolor='red', ecolor='red', marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(x_list_psychometric, Psychometric_function(psychometric_params_H_non_drug_corr_chooseHigh, x_list_psychometric), color='red', ls='-', clip_on=False, zorder=2, label='Choose Tall')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(x0_psychometric, Psychometric_function(psychometric_params_H_non_drug_corr_chooseHigh, x0_psychometric), s=15., color='red', marker='_', clip_on=False, zorder=2, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
## Choose Low
ax.errorbar( d_evidence_H_non_drug_list_corr_chooseLow[2:],    P_corr_H_list_non_drug_corr_chooseLow[2:], ErrBar_P_corr_H_list_non_drug_corr_chooseLow[2:], color='k', markerfacecolor='blue', ecolor='blue', fmt='.', zorder=3, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_H_non_drug_list_corr_chooseLow[1], P_corr_H_list_non_drug_corr_chooseLow[1], ErrBar_P_corr_H_list_non_drug_corr_chooseLow[1], color='k', markerfacecolor='blue', ecolor='blue', marker='.', zorder=3, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(x_list_psychometric, Psychometric_function(psychometric_params_H_non_drug_corr_chooseLow, x_list_psychometric), color='blue', ls='-', clip_on=False, zorder=1, label='Choose Short')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(x0_psychometric, Psychometric_function(psychometric_params_H_non_drug_corr_chooseLow, x0_psychometric), s=15., color='blue', marker='_', clip_on=False, zorder=1, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=-3.)
ax_0.set_ylim([0.48,1.])
ax.set_ylim([0.48,1.])
ax_0.set_xlim([-1,1])
ax.set_xlim([1,50])
ax_0.set_xticks([0.])
ax.xaxis.set_ticks([1, 10])
ax_0.set_yticks([0.5, 1.])
ax_0.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax_0.yaxis.set_minor_locator(minorLocator)
ax.set_yticks([])
ax_0.tick_params(direction='out', pad=1.5)
ax_0.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
## Add breakmark = wiggle
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.0968
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(-0.5,0.7), fontsize=fontsize_legend-2, frameon=False, ncol=1, markerscale=0., columnspacing=-0.1, handletextpad=0.)
for color,text,item in zip(['red','blue'], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)
ax.text(-20.5, 0.0105, r'$\times \mathregular{10^{-2}}$', fontsize=fontsize_tick-1.)


## rect1_21: Psychophysical Kernel, monkey A
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
tmp = ax.errorbar(i_PK_list, PK_A_nondrug_chooseHigh, PK_A_nondrug_errbar_chooseHigh, color='red', markerfacecolor='red', ecolor='red', markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1., label='Choose Tall')#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax.errorbar(i_PK_list, PK_A_nondrug_chooseLow, PK_A_nondrug_errbar_chooseLow, color='blue', markerfacecolor='blue', ecolor='blue', markeredgecolor='k', linestyle='-', marker='.', zorder=(3-2), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1., label='Choose Short')#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.set_xlabel('Sample Number', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylim([0.,4.35])
ax.set_xlim([1.,8.])
ax.set_xticks([1,8])
ax.set_yticks([0., 4.])
ax.text(0.1, 4.45, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
legend_elements= [Line2D([0], [0], color='red', lw=1, label='Choose Tall'), Line2D([0], [0], color='blue', lw=1, label='Choose Short')]
legend = ax.legend(handles=legend_elements, loc=(0.27,0.75), fontsize=fontsize_legend-2, frameon=False, ncol=1, markerscale=0., columnspacing=-0.1, handletextpad=0.)
for color,text,item in zip(['red','blue'], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

## rect1_22: Psychophysical Kernel, monkey H
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
tmp = ax.errorbar(i_PK_list, PK_H_nondrug_chooseHigh, PK_H_nondrug_errbar_chooseHigh, color='red', markerfacecolor='red', ecolor='red', markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1., label='Choose Tall')#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax.errorbar(i_PK_list, PK_H_nondrug_chooseLow, PK_H_nondrug_errbar_chooseLow, color='blue', markerfacecolor='blue', ecolor='blue', markeredgecolor='k', linestyle='-', marker='.', zorder=(3-2), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1., label='Choose Short')#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.set_xlabel('Sample Number', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylim([0.,4.35])
ax.set_xlim([1.,8.])
ax.set_xticks([1., 8.])
ax.set_yticks([0., 4.])
ax.text(0.1, 4.45, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
legend_elements= [Line2D([0], [0], color='red', lw=1, label='Choose Tall'), Line2D([0], [0], color='blue', lw=1, label='Choose Short')]
legend = ax.legend(handles=legend_elements, loc=(0.27,0.75), fontsize=fontsize_legend-2, frameon=False, ncol=1, markerscale=0., columnspacing=-0.1, handletextpad=0.)
for color,text,item in zip(['red','blue'], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

fig_temp.savefig(path_cwd+'Figure2S1.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
### Figure 1: Experimental non-drug data, Narrow-Broad Trials

## Schematics
x_schem = np.arange(1,99,1)
sigma_narrow = 12
sigma_broad = 24

## Define Data. Alternatively can also import.                                                                          # See MainAnalysisNonDrugDays.m: NarrowBroadTrialsCOL
n_A_non_drug = 41.                                                                                                      # Alfie, 41 runs
n_H_non_drug = 63.                                                                                                      # Henry, 63 runs
ENB_bars_A_non_drug_chooseHigh = np.array([0.618461538461539, 0.783050847457627, 0.909385113268608])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Alfie, 41 runs
ENB_bars_A_non_drug_chooseLow = np.array([0.687500000000000, 0.778597785977860, 0.921161825726141])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Alfie, 41 runs
ENB_bars_H_non_drug_chooseHigh = np.array([0.600000000000000, 0.756711409395973, 0.903398926654741])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Henry, 63 runs
ENB_bars_H_non_drug_chooseLow = np.array([0.539893617021277, 0.741007194244604, 0.851752021563342])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Henry, 63 runs

# Error bar data (of the old form).                                                                                     # See MainAnalysisNonDrugDays.m: NarrowBroadTrialsCOL_Errs)
ENB_bars_err_A_non_drug_chooseHigh = np.array([0.0269453500429937, 0.0239973316139149, 0.0163303110307984])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Alfie, 41 runs
ENB_bars_err_A_non_drug_chooseLow = np.array([0.0247052942200655, 0.0252210356799318, 0.0173591207388457])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Alfie, 41 runs
ENB_bars_err_H_non_drug_chooseHigh = np.array([0.0209849225034801, 0.0175753054851363, 0.0124946719216227])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Alfie, 41 runs
ENB_bars_err_H_non_drug_chooseLow = np.array([0.0181749952193854, 0.0185788003036698, 0.0184486257843724])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Alfie, 41 runs



## Define subfigure domain.
figsize = (max15, 1.2*max15)

# 4 rows (with sampled mean distribution)
width1_11=0.19; width1_12=width1_11; width1_13=width1_11; width1_11a=width1_11; width1_12a=width1_11a; width1_13a=width1_11a; width1_22=0.07; width1_21=width1_22*(1+bar_width)/bar_width; width1_23=width1_21; width1_24=width1_22
x1_11=0.115; x1_12 = x1_11 + width1_11 + xbuf0*1.3; x1_13 = x1_12 + width1_12 + xbuf0*1.3; x1_11a=x1_11; x1_12a = x1_11a + width1_11a + xbuf0*1.3; x1_13a = x1_12a + width1_12a + xbuf0*1.3; x1_21=0.105; x1_22 = x1_21 + width1_21 + xbuf0*1.1; x1_23 = x1_22 + width1_22 + xbuf0*1.4; x1_24 = x1_23 + width1_23 + xbuf0*1.1
height1_11=0.18; height1_12=height1_11; height1_13=height1_11; height1_11a=height1_11; height1_12a=height1_11a; height1_13a=height1_11a; height1_21=height1_11; height1_22 = height1_21; height1_23=height1_21; height1_24 = height1_21       # 4 rows
y1_11=0.76; y1_12=y1_11; y1_13=y1_11; y1_11a = y1_11 - height1_11a - 1.7*ybuf0; y1_12a=y1_11a; y1_13a=y1_11a; y1_21 = y1_11a - height1_21 - 2.*ybuf0; y1_22=y1_21; y1_23=y1_21; y1_24=y1_21

rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_11a = [x1_11a, y1_11a, width1_11a, height1_11a]
rect1_12a = [x1_12a, y1_12a, width1_12a, height1_12a]
rect1_13a = [x1_13a, y1_13a, width1_13a, height1_13a]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_24 = [x1_24, y1_24, width1_24, height1_24]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.21, 0.975 + y1_21 - y1_11, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.01, 0.93 + y1_21 - y1_11, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015 + x1_22 - x1_21, 0.935 + y1_21 - y1_11, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.715, 0.975 + y1_21 - y1_11, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.02 + x1_23 - x1_21, 0.93 + y1_21 - y1_11, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015 + x1_24 - x1_21, 0.935 + y1_21 - y1_11, 'D', fontsize=fontsize_fig_label, fontweight='bold')





## rect1_21: Accuracy with narrow/Broad Correct mean (monkey A)
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
bar_width=0.42
bar1 = ax.bar([0, 1], ENB_bars_A_non_drug_chooseHigh[1:], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_A_non_drug_chooseHigh[1:], ecolor='k', color='red', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
bar2 = ax.bar([0+bar_width, 1+bar_width], ENB_bars_A_non_drug_chooseLow[1:], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_A_non_drug_chooseLow[1:], ecolor='k', color='blue', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
for bar in bar1:
    bar.set_edgecolor("k")
for bar in bar2:
    bar.set_edgecolor("k")
ax.axhline(y=0.5, color='k', ls='--', lw=0.9)
ax.scatter([0.91], [1.01], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.42,1.42], [0.96,0.96], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Accuracy', fontsize=fontsize_legend)
ax.set_xlim([0,1+2.*bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width, 1+bar_width])
ax.xaxis.set_ticklabels(['Narrow\nCorrect', 'Broad\nCorrect'])
ax.set_yticks([0., 0.5, 1.])
ax.set_yticklabels([0, 0.5, 1])
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend = ax.legend([bar1, bar2], ('Choose Tall', 'Choose Short'), loc=(-0.9,1.02), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2, labelspacing=0.13)
for color,text,item in zip(['r','b'], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)
ax.spines['bottom'].set_position(('zero'))


## rect1_22: broad preference with equal mean (monkey A)
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar([0], ENB_bars_A_non_drug_chooseHigh[0], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_A_non_drug_chooseHigh[0], ecolor='k', color='red', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
bar2 = ax.bar([0+bar_width], ENB_bars_A_non_drug_chooseLow[0], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_A_non_drug_chooseLow[0], ecolor='k', color='blue', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
for bar in bar1:
    bar.set_edgecolor("k")
for bar in bar2:
    bar.set_edgecolor("k")
ax.axhline(y=0.5, color='k', ls='--', lw=0.9)
ax.set_ylabel('Broad Preference', fontsize=fontsize_legend)
ax.set_xlim([0,2.*bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width])
ax.xaxis.set_ticklabels(['Ambiguous'])
ax.set_yticks([0., 0.5, 1.])
ax.set_yticklabels([0, 0.5, 1])
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")


## rect1_23: Accuracy with narrow/Broad Correct mean (monkey H)
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar([0, 1], ENB_bars_H_non_drug_chooseHigh[1:], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_H_non_drug_chooseHigh[1:], ecolor='k', color='red', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
bar2 = ax.bar([0+bar_width, 1+bar_width], ENB_bars_H_non_drug_chooseLow[1:], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_H_non_drug_chooseLow[1:], ecolor='k', color='blue', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
for bar in bar1:
    bar.set_edgecolor("k")
for bar in bar2:
    bar.set_edgecolor("k")
ax.axhline(y=0.5, color='k', ls='--', lw=0.9)
ax.scatter([0.91], [1.01], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.42,1.42], [0.96,0.96], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Accuracy', fontsize=fontsize_legend)
ax.set_xlim([0,1+2.*bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width, 1+bar_width])
ax.xaxis.set_ticklabels(['Narrow\nCorrect', 'Broad\nCorrect'])
ax.set_yticks([0., 0.5, 1.])
ax.set_yticklabels([0, 0.5, 1])
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")


## rect1_24: broad preference with equal mean (monkey H)
ax   = fig_temp.add_axes(rect1_24)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar([0], ENB_bars_H_non_drug_chooseHigh[0], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_H_non_drug_chooseHigh[0], ecolor='k', color='red', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
bar2 = ax.bar([0+bar_width], ENB_bars_H_non_drug_chooseLow[0], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_H_non_drug_chooseLow[0], ecolor='k', color='blue', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
for bar in bar1:
    bar.set_edgecolor("k")
for bar in bar2:
    bar.set_edgecolor("k")
ax.axhline(y=0.5, color='k', ls='--', lw=0.9)
ax.set_ylabel('Broad Preference', fontsize=fontsize_legend)
ax.set_xlim([0,2*bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width])
ax.xaxis.set_ticklabels(['Ambiguous'])
ax.set_yticks([0., 0.5, 1.])
ax.set_yticklabels([0, 0.5, 1])
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")


fig_temp.savefig(path_cwd+'Figure3S2.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
### Figure 3: Experimental non-drug data, Regression Trials

## Define Data. Alternatively can also import.                                                                          # See MainAnalysisNonDrugDays.m: VarAndLocalWinsBetasCollapsed
n_A_non_drug = 41.                                                                                                      # Alfie, 41 runs
n_H_non_drug = 63.                                                                                                      # Henry, 63 runs
Reg_bars_A_non_drug = np.array([-0.864248625633257, 23.3670217599347, 8.30585175292630, -23.6882164391815, -2.43528720614255])    # [Bias, LeftVal, LeftVar, RightVal, RightVar]. Alfie regression Beta values.
Reg_bars_H_non_drug = np.array([-0.187958278434194, 19.3203042083793, 3.20022868283905, -19.4396340316793, -2.13857576352430])   # [Bias, LeftVal, LeftVar, RightVal, RightVar]. Henry regression Beta values.
Reg_bars_A_non_drug_rearanged = np.abs(np.array([Reg_bars_A_non_drug[1], Reg_bars_A_non_drug[3], Reg_bars_A_non_drug[2], Reg_bars_A_non_drug[4]]))
Reg_bars_H_non_drug_rearanged = np.abs(np.array([Reg_bars_H_non_drug[1], Reg_bars_H_non_drug[3], Reg_bars_H_non_drug[2], Reg_bars_H_non_drug[4]]))
Reg_bars_avg_non_drug = (n_A_non_drug*Reg_bars_A_non_drug + n_H_non_drug*Reg_bars_H_non_drug)/ (n_A_non_drug+n_H_non_drug)

Reg_bars_err_A_non_drug = np.array([0.327869337770797, 0.596275552574013, 0.453944518164700, 0.601947445646728, 0.446128695432194])  # [Bias, LeftVal, LeftVar, RightVal, RightVar]. Alfie regression Beta values.
Reg_bars_err_H_non_drug = np.array([0.232825707874531, 0.394577136784731, 0.312805845061314, 0.397988508899067, 0.318017800126500])  # [Bias, LeftVal, LeftVar, RightVal, RightVar]. Henry regression Beta values.
Reg_bars_err_A_non_drug_rearanged = np.abs(np.array([Reg_bars_err_A_non_drug[1], Reg_bars_err_A_non_drug[3], Reg_bars_err_A_non_drug[2], Reg_bars_err_A_non_drug[4]]))
Reg_bars_err_H_non_drug_rearanged = np.abs(np.array([Reg_bars_err_H_non_drug[1], Reg_bars_err_H_non_drug[3], Reg_bars_err_H_non_drug[2], Reg_bars_err_H_non_drug[4]]))

## Using L/R difference as regressors instead.
Reg_bars_mean_var_LRdiff_A_nondrug_chooseHigh = 100.*np.array([-0.00646169770645438, 0.232969321285219, 0.0300336369722477])        # Mean, SD, averaged over left and right
Reg_bars_mean_var_LRdiff_A_nondrug_chooseLow = 100.*np.array([0.0301991173841662, 0.236123330846453, 0.0791587001404509])        # Mean, SD, averaged over left and right
Reg_bars_mean_var_LRdiff_H_nondrug_chooseHigh = 100.*np.array([-0.127239912878490, 0.208551127970014, 0.0197912622145348])        # Mean, SD, averaged over left and right
Reg_bars_mean_var_LRdiff_H_nondrug_chooseLow = 100.*np.array([0.0105613482175904, 0.180955562609264, 0.0340351244238595])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_var_LRdiff_A_nondrug_chooseHigh = 100.*np.array([0.0382156839500196, 0.00710652074185305, 0.00441140634821305])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_var_LRdiff_A_nondrug_chooseLow = 100.*np.array([0.0396950304852793, 0.00735889181536205, 0.00480521038702039])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_var_LRdiff_H_nondrug_chooseHigh = 100.*np.array([0.0283020608410236, 0.00487754687870338, 0.00319602029160427])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_var_LRdiff_H_nondrug_chooseLow = 100.*np.array([0.0274556220320310, 0.00447526742685425, 0.00310227390833175])        # Mean, SD, averaged over left and right


bar_pos_2by2 = [0., 0.8, 1.8, 2.6]
mean_var_color_list_2by2 = [color_mean_var_beta[0], color_mean_var_beta[0], color_mean_var_beta[1], color_mean_var_beta[1]]
mean_supp_choice_color_list_2by2 = [color_mean_supp_choice[0], color_mean_supp_choice[0], color_mean_supp_choice[1], color_mean_supp_choice[1]]

## Psychometric function generated from all (regression) data                                                           # See MainAnalysisNonDrugDays.m: lines 80-104
d_evidence_A_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_A_list_chooseHigh =  np.array([0, 0.0291262135922330, 0.0341880341880342, 0.118243243243243, 0.153664302600473, 0.246153846153846, 0.296482412060302, 0.418478260869565, 0.414634146341463, 0.487500000000000, 0.800000000000000, 0.605633802816901, 0.698224852071006, 0.675000000000000, 0.808411214953271, 0.833910034602076, 0.876963350785340, 0.934984520123839, 0.948356807511737, 0.983870967741936, 1])  # Log-Spaced.
P_corr_A_list_chooseLow =  np.array([0.0833333333333333, 0.0181818181818182, 0.0640394088669951, 0.151202749140893, 0.225806451612903, 0.376383763837638, 0.390243902439024, 0.546762589928058, 0.430232558139535, 0.540166204986150, 0.125000000000000, 0.625000000000000, 0.703910614525140, 0.762430939226519, 0.811475409836066, 0.892307692307692, 0.920489296636086, 0.960493827160494, 0.974789915966387, 0.962962962962963, 1])  # Log-Spaced.
ErrBar_P_corr_A_list_chooseHigh = np.array([0, 0.0165693239996672, 0.0118788781353212, 0.0187679444466329, 0.0175342680161554, 0.0218128479064174, 0.0323750471099719, 0.0363672545951555, 0.0384701873033307, 0.0249921862789153, 0.126491106406735, 0.0289998482088330, 0.0353098631829212, 0.0370282831089966, 0.0269026040579412, 0.0218918455684897, 0.0168064616223518, 0.0137185855326794, 0.0151636105610860, 0.0159984306571507, 0])
ErrBar_P_corr_A_list_chooseLow = np.array([0.0564169333655275, 0.0127390736317340, 0.0171832148277772, 0.0210007754555835, 0.0208276573389929, 0.0294299484834445, 0.0340697705253453, 0.0422235622603213, 0.0533889310076761, 0.0262307402489587, 0.116926793336686, 0.0247052942200655, 0.0341227214463459, 0.0316341309956366, 0.0250395645181793, 0.0192248511603934, 0.0149605752451270, 0.00967948513413441, 0.0101614148457053, 0.0181723474615799, 0])
d_evidence_H_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_H_list_chooseHigh =  np.array([0, 0.0355029585798817, 0.0606060606060606, 0.115537848605578, 0.202898550724638, 0.222764227642276, 0.257309941520468, 0.439560439560440, 0.403908794788274, 0.481250000000000, 0.777777777777778, 0.626570915619390, 0.587360594795539, 0.650000000000000, 0.739247311827957, 0.822550831792976, 0.860497237569061, 0.894480519480519, 0.943089430894309, 0.958041958041958, 1])  # Log-Spaced.
P_corr_H_list_chooseLow =  np.array([0.0370370370370370, 0.0355029585798817, 0.115151515151515, 0.158813263525305, 0.249329758713137, 0.293436293436293, 0.386920980926431, 0.475177304964539, 0.502824858757062, 0.478040540540541, 0.636363636363636, 0.615611192930781, 0.699646643109541, 0.684713375796178, 0.702127659574468, 0.778251599147122, 0.834459459459459, 0.898333333333333, 0.938356164383562, 0.980132450331126, 0.950000000000000])  # Log-Spaced.
ErrBar_P_corr_H_list_chooseHigh = np.array([0, 0.0142344067101610, 0.0115200206819489, 0.0142675665224472, 0.0153098778778760, 0.0167788255315998, 0.0236384521412696, 0.0300394770500879, 0.0280045645581988, 0.0197503337590470, 0.138579903213850, 0.0204956449075949, 0.0300166079479759, 0.0295803989154981, 0.0227634353440828, 0.0164255425785944, 0.0128764712030265, 0.0123783115568065, 0.0120603446842029, 0.0167660922954863, 0])
ErrBar_P_corr_H_list_chooseLow = np.array([0.0256995802403226, 0.0142344067101610, 0.0175716406944952, 0.0152690593200639, 0.0158395283078441, 0.0200063423886897, 0.0254235583129075, 0.0297378519098109, 0.0375817015948284, 0.0205300448148992, 0.145040733675903, 0.0186682553702839, 0.0272497426249778, 0.0262205836876966, 0.0235846376046784, 0.0191824395292776, 0.0152754478623514, 0.0123376493949452, 0.0114918944183177, 0.0113560177876564, 0.0344601218802256])


# ## Combined Regression figure.                                                                                          # See MainAnalysisNonDrugDays.m: LongAvCOL, LongAvCOLSE
bar_pos_2by2_combined = np.array([0., 1.8, 3.6+0.5, 5.4+0.5, 7.2+0.5, 9.0+0.5, 10.8+0.5])
Reg_bar_pos_combined_model_control = np.array([0., 1., 2.+0.5, 3.+0.5, 4.+0.5, 5.0+0.5, 6.+0.5])
Reg_combined_color_list = [color_mean_var_beta[0], color_mean_var_beta[1], color_mean_var_beta[0], 'grey', 'grey', 'grey', 'grey']

Reg_bars_mean_var_combined_A_nondrug = np.array([0.5*(Reg_bars_A_non_drug[1]-Reg_bars_A_non_drug[3]), 0.5*(Reg_bars_A_non_drug[2]-Reg_bars_A_non_drug[4])])        # Mean, SD, averaged over left and right
Reg_bars_mean_var_combined_H_nondrug = np.array([0.5*(Reg_bars_H_non_drug[1]-Reg_bars_H_non_drug[3]), 0.5*(Reg_bars_H_non_drug[2]-Reg_bars_H_non_drug[4])])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_var_combined_A_nondrug = np.array([0.5*(Reg_bars_err_A_non_drug[1]**2.+Reg_bars_err_A_non_drug[3]**2.)**0.5, 0.5*(Reg_bars_err_A_non_drug[2]**2.+Reg_bars_err_A_non_drug[4]**2.)**0.5])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_var_combined_H_nondrug = np.array([0.5*(Reg_bars_err_A_non_drug[1]**2.+Reg_bars_err_A_non_drug[3]**2.)**0.5, 0.5*(Reg_bars_err_A_non_drug[2]**2.+Reg_bars_err_A_non_drug[4]**2.)**0.5])        # Mean, SD, averaged over left and right



## Combined Regression figure.                                                                                          # See MainAnalysisNonDrugDays.m: LongAvCOL, LongAvCOLSE
Reg_values_A_nondrug_chooseHigh = 100.*np.array([-1.34682828874276, 0.00577080782212465, -0.00553662028140105, 0.245043461188756, 0.0259435141937917, -0.0228566201777815, -0.0183648102073321, 0.00739989943306379, -0.204725107402377, -0.0130385627767056, -0.0228796709093002])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_A_nondrug_chooseHigh = 100.*np.array([0.489519164937712, 0.00227131978415561, 0.00226689792792362, 0.0117082779573794, 0.00472931313417923, 0.00510933794689973, 0.00232126081413155, 0.00231404373705673, 0.0109771736315664, 0.00483365273932856, 0.00473573372712063])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_H_nondrug_chooseHigh = 100.*np.array([-0.339948747040264, 0.00380716999009252, -0.00479961436948294, 0.206006370982769, 0.00894888456783106, 0.00373199355458185, -0.00534933423687899, 0.000361329224520820, -0.214142853131285, -0.00381896673080162, 0.0182343847392148])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_H_nondrug_chooseHigh = 100.*np.array([0.348911611430713, 0.00158931448942905, 0.00158850940401275, 0.00794706976290317, 0.00350889521504428, 0.00354066682046416, 0.00164990129408496, 0.00165786582941797, 0.00813070482080559, 0.00349844616446652, 0.00361428738928487])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_A_nondrug_chooseLow = 100.*np.array([-0.383274880938743, 0.0117675141788571, -0.000760321810322260, 0.236320341506066, 0.0243823364210443, -0.0411608492389197, -0.0109023876112093, 0.00704515648534199, -0.260820397314930, -0.00540777866669413, 0.0349943219126529])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_A_nondrug_chooseLow = 100.*np.array([0.490828878442514, 0.00236800566891351, 0.00248122198716349, 0.0117633666646555, 0.00498603984031467, 0.00472498894567768, 0.00233049077656228, 0.00251037289766586, 0.0119764426737758, 0.00506169712408824, 0.00520415906109678])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_H_nondrug_chooseLow = 100.*np.array([-0.0950217888459028, 0.0132574196112295, -0.00261972367117647, 0.157166266405944, 0.0260470784947470, -0.0112688041937741, -0.0116894697517847, 0.00471118452579611, -0.160929961763487, -0.0177211880271990, -0.00985568793950984])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_H_nondrug_chooseLow = 100.*np.array([0.337293407507428, 0.00164273824466824, 0.00167592371843453, 0.00746337732901167, 0.00332459246967667, 0.00324240884956248, 0.00159403503704819, 0.00168924130929422, 0.00760493178018702, 0.00345543439444969, 0.00353627204687867])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
bar_pos_2by2_combined = np.array([0., 1.8, 3.6+0.5, 5.4+0.5, 7.2+0.5, 9.0+0.5, 10.8+0.5])
Reg_bar_pos_combined_model_control = np.array([0., 1., 2.+0.5, 3.+0.5, 4.+0.5, 5.0+0.5, 6.+0.5])
Reg_combined_color_list = [color_mean_var_beta[0], color_mean_var_beta[1], color_mean_var_beta[0], 'grey', 'grey', 'grey', 'grey']

Reg_bars_LRsep_A_nondrug_chooseHigh = np.array([Reg_values_A_nondrug_chooseHigh[3], -Reg_values_A_nondrug_chooseHigh[8], Reg_values_A_nondrug_chooseHigh[4], -Reg_values_A_nondrug_chooseHigh[9], Reg_values_A_nondrug_chooseHigh[5], -Reg_values_A_nondrug_chooseHigh[10], Reg_values_A_nondrug_chooseHigh[1], -Reg_values_A_nondrug_chooseHigh[6], Reg_values_A_nondrug_chooseHigh[2], -Reg_values_A_nondrug_chooseHigh[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_err_LRsep_A_nondrug_chooseHigh = np.array([Reg_values_errbar_A_nondrug_chooseHigh[3], Reg_values_errbar_A_nondrug_chooseHigh[8], Reg_values_errbar_A_nondrug_chooseHigh[4], Reg_values_errbar_A_nondrug_chooseHigh[9], Reg_values_errbar_A_nondrug_chooseHigh[5], Reg_values_errbar_A_nondrug_chooseHigh[10], Reg_values_errbar_A_nondrug_chooseHigh[1], Reg_values_errbar_A_nondrug_chooseHigh[6], Reg_values_errbar_A_nondrug_chooseHigh[2], Reg_values_errbar_A_nondrug_chooseHigh[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_LRsep_H_nondrug_chooseHigh = np.array([Reg_values_H_nondrug_chooseHigh[3], -Reg_values_H_nondrug_chooseHigh[8], Reg_values_H_nondrug_chooseHigh[4], -Reg_values_H_nondrug_chooseHigh[9], Reg_values_H_nondrug_chooseHigh[5], -Reg_values_H_nondrug_chooseHigh[10], Reg_values_H_nondrug_chooseHigh[1], -Reg_values_H_nondrug_chooseHigh[6], Reg_values_H_nondrug_chooseHigh[2], -Reg_values_H_nondrug_chooseHigh[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_err_LRsep_H_nondrug_chooseHigh = np.array([Reg_values_errbar_H_nondrug_chooseHigh[3], Reg_values_errbar_H_nondrug_chooseHigh[8], Reg_values_errbar_H_nondrug_chooseHigh[4], Reg_values_errbar_H_nondrug_chooseHigh[9], Reg_values_errbar_H_nondrug_chooseHigh[5], Reg_values_errbar_H_nondrug_chooseHigh[10], Reg_values_errbar_H_nondrug_chooseHigh[1], Reg_values_errbar_H_nondrug_chooseHigh[6], Reg_values_errbar_H_nondrug_chooseHigh[2], Reg_values_errbar_H_nondrug_chooseHigh[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_LRsep_A_nondrug_chooseLow = np.array([Reg_values_A_nondrug_chooseLow[3], -Reg_values_A_nondrug_chooseLow[8], Reg_values_A_nondrug_chooseLow[4], -Reg_values_A_nondrug_chooseLow[9], Reg_values_A_nondrug_chooseLow[5], -Reg_values_A_nondrug_chooseLow[10], Reg_values_A_nondrug_chooseLow[1], -Reg_values_A_nondrug_chooseLow[6], Reg_values_A_nondrug_chooseLow[2], -Reg_values_A_nondrug_chooseLow[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_err_LRsep_A_nondrug_chooseLow = np.array([Reg_values_errbar_A_nondrug_chooseLow[3], Reg_values_errbar_A_nondrug_chooseLow[8], Reg_values_errbar_A_nondrug_chooseLow[4], Reg_values_errbar_A_nondrug_chooseLow[9], Reg_values_errbar_A_nondrug_chooseLow[5], Reg_values_errbar_A_nondrug_chooseLow[10], Reg_values_errbar_A_nondrug_chooseLow[1], Reg_values_errbar_A_nondrug_chooseLow[6], Reg_values_errbar_A_nondrug_chooseLow[2], Reg_values_errbar_A_nondrug_chooseLow[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_LRsep_H_nondrug_chooseLow = np.array([Reg_values_H_nondrug_chooseLow[3], -Reg_values_H_nondrug_chooseLow[8], Reg_values_H_nondrug_chooseLow[4], -Reg_values_H_nondrug_chooseLow[9], Reg_values_H_nondrug_chooseLow[5], -Reg_values_H_nondrug_chooseLow[10], Reg_values_H_nondrug_chooseLow[1], -Reg_values_H_nondrug_chooseLow[6], Reg_values_H_nondrug_chooseLow[2], -Reg_values_H_nondrug_chooseLow[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_err_LRsep_H_nondrug_chooseLow = np.array([Reg_values_errbar_H_nondrug_chooseLow[3], Reg_values_errbar_H_nondrug_chooseLow[8], Reg_values_errbar_H_nondrug_chooseLow[4], Reg_values_errbar_H_nondrug_chooseLow[9], Reg_values_errbar_H_nondrug_chooseLow[5], Reg_values_errbar_H_nondrug_chooseLow[10], Reg_values_errbar_H_nondrug_chooseLow[1], Reg_values_errbar_H_nondrug_chooseLow[6], Reg_values_errbar_H_nondrug_chooseLow[2], Reg_values_errbar_H_nondrug_chooseLow[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
x_LRsep_list = np.array([0., 0.4, 1., 1.4, 2., 2.4, 3., 3.4, 4., 4.4, 5., 5.4])
color_LRsep_list = [color_mean_var_beta[0], color_mean_var_beta[0], 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey']


### Psychometric function fit                                                                                           # See figure_psychometric_function_fit.py, esp lines 322-479
def Psychometric_fit_D(params_pm, pm_fit2, x_list):
    prob_corr_fit = 0.5 + 0.5*np.sign(x_list+params_pm[2])*(1. - np.exp(-(np.abs(x_list+params_pm[2])/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    to_min = -sum(np.log(prob_corr_fit)*pm_fit2) - sum(np.log(1.-prob_corr_fit)*(1.-pm_fit2))                                                          # Maximum Likelihood Estimator
    return to_min
def Psychometric_function_D(params_pm, x_list):
    prob_corr_fit = 0.5 + 0.5*np.sign(x_list+params_pm[2])*(1. - np.exp(-(np.abs(x_list+params_pm[2])/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    return prob_corr_fit

x_list_psychometric = np.arange(0.01, 0.5, 0.01)
x0_psychometric = 0.
## non-binned MLE (i.e. done using literal net evidence, via matlab). See Psychometric_function_fit_NonDrugDays_NL.m
psychometric_params_A_non_drug_chooseHigh = [0.060575520348707,1.004502972180624,0.008985918229141]
psychometric_params_A_non_drug_chooseLow = [0.066321527498328,1.253193264992802,0.023100231005683]
psychometric_params_H_non_drug_chooseHigh = [0.068415224565275,0.988385929198552,0.006171512207321]
psychometric_params_H_non_drug_chooseLow = [0.081958354715701,1.113598708298210,0.014108630055185]






bar_width = 0.8



## Define subfigure domain.
figsize = (max15,1.25*max15)

width1_11=0.13; width1_21=width1_11; width1_12=0.21; width1_22=width1_12; width1_14=0.3; width1_24=width1_14; width1_13=width1_14*1.8/4.8 *0.6; width1_23=width1_13
width1_32=width1_12; width1_34=width1_14; width1_33=width1_13; width1_42=width1_12; width1_44=width1_14; width1_43=width1_13
x1_11=0.065; x1_12=0.195; x1_13=x1_12 + width1_12 + 0.95*xbuf0; x1_14=x1_13 + width1_13 + 0.9*xbuf0; x1_21=x1_11; x1_22=x1_12; x1_23=x1_13; x1_24=x1_14
x1_32=x1_12; x1_33=x1_13; x1_34=x1_14; x1_42=x1_12; x1_43=x1_13; x1_44=x1_14
height1_11=0.35; height1_12=0.15; height1_13=height1_12+0.12*ybuf0; height1_14=height1_13; height1_21=height1_11; height1_22=height1_12; height1_23=height1_22+0.12*ybuf0; height1_24=height1_23
height1_32=height1_12; height1_33=height1_32+0.12*ybuf0; height1_34=height1_33; height1_42=height1_12; height1_43=height1_42+0.12*ybuf0; height1_44=height1_43
y1_11=0.8; y1_12=y1_11+0.4*ybuf0; y1_13=y1_12-0.212*ybuf0; y1_14=y1_13; y1_21 = y1_11 - height1_21 - 1.*ybuf0; y1_22=y1_12- height1_32 - 1.2*ybuf0; y1_23=y1_22-0.212*ybuf0; y1_24=y1_23
y1_32=y1_22 - height1_32 - 1.4*ybuf0; y1_33=y1_32-0.212*ybuf0; y1_34=y1_33; y1_42=y1_32 - height1_42 - 1.2*ybuf0; y1_43=y1_42-0.212*ybuf0; y1_44=y1_43

rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_12_0 = [x1_12, y1_12, width1_12*0.05, height1_12]
rect1_12 = [x1_12+width1_12*0.2, y1_12, width1_12*(1-0.2), height1_12]
rect1_22_0 = [x1_22, y1_22, width1_22*0.05, height1_22]
rect1_22 = [x1_22+width1_22*0.2, y1_22, width1_22*(1-0.2), height1_22]
rect1_32_0 = [x1_32, y1_32, width1_32*0.05, height1_32]
rect1_32 = [x1_32+width1_32*0.2, y1_32, width1_32*(1-0.2), height1_32]
rect1_42_0 = [x1_42, y1_42, width1_42*0.05, height1_42]
rect1_42 = [x1_42+width1_42*0.2, y1_42, width1_42*(1-0.2), height1_42]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_33 = [x1_33, y1_33, width1_33, height1_33]
rect1_43 = [x1_43, y1_43, width1_43, height1_43]
rect1_14 = [x1_14, y1_14, width1_14, height1_14]
rect1_24 = [x1_24, y1_24, width1_24, height1_24]
rect1_34 = [x1_34, y1_34, width1_34, height1_34]
rect1_44 = [x1_44, y1_44, width1_44, height1_44]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.06, 0.952, 'Choose Tall', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k')
fig_temp.text(-0.092 + x1_12, 0.977, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.067 + x1_13 , 0.977, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.067 + x1_14 , 0.977, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015, 0.815, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k')
fig_temp.text(0.06, 0.962+ y1_22 - y1_12, 'Choose Short', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k')
fig_temp.text(-0.092 + x1_12, 0.977 + y1_22 - y1_12, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.067 + x1_13, 0.977 + y1_22 - y1_12, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.067 + x1_14, 0.977 + y1_22 - y1_12, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.06, 0.952 + y1_32 - y1_12, 'Choose Tall', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k')
fig_temp.text(-0.092 + x1_12, 0.977 + y1_32 - y1_12, 'G', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.067 + x1_13, 0.977 + y1_32 - y1_12, 'H', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.067 + x1_14, 0.977 + y1_32 - y1_12, 'I', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015, 0.815 + y1_32 - y1_12, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k')
fig_temp.text(0.06, 0.962 + y1_42 - y1_12, 'Choose Short', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k')
fig_temp.text(-0.092 + x1_12, 0.977 + y1_42 - y1_12, 'J', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.067 + x1_13, 0.977 + y1_42 - y1_12, 'K', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.067 + x1_14, 0.977 + y1_42 - y1_12, 'L', fontsize=fontsize_fig_label, fontweight='bold')








## rect1_12: Psychometric function (over dx_broad, or dx_corr ?), Monkey A
ax_0   = fig_temp.add_axes(rect1_12_0)
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
ax.errorbar( d_evidence_A_list[12:],    P_corr_A_list_chooseHigh[12:], ErrBar_P_corr_A_list_chooseHigh[12:], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Corr.' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_A_list[1:9], 1.-P_corr_A_list_chooseHigh[1:9], ErrBar_P_corr_A_list_chooseHigh[1:9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], fmt='.', zorder=3, clip_on=False, label='Lower SD Corr.', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_A_list[11], P_corr_A_list_chooseHigh[11], ErrBar_P_corr_A_list_chooseHigh[11], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_A_list[9], 1.-P_corr_A_list_chooseHigh[9], ErrBar_P_corr_A_list_chooseHigh[9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], marker='.', zorder=3, clip_on=False                      , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_A_non_drug_chooseHigh, x_list_psychometric), color=color_NB[1], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_A_non_drug_chooseHigh, -x_list_psychometric), color=color_NB[0], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_A_non_drug_chooseHigh, x0_psychometric), s=15., color=color_NB[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_A_non_drug_chooseHigh, -x0_psychometric), s=15., color=color_NB[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=2.)
ax_0.set_ylim([0.4,1.])
ax.set_ylim([0.4,1.])
ax_0.set_xlim([-1,1])
ax.set_xlim([1,50])
ax_0.set_xticks([0.])
ax.xaxis.set_ticks([1, 10])
ax_0.set_yticks([0.5, 1.])
ax_0.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax_0.yaxis.set_minor_locator(minorLocator)
ax.set_yticks([])
ax_0.tick_params(direction='out', pad=1.5)
ax_0.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
## Add breakmark = wiggle
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.1135
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_NB[1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=color_NB[0], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.36,-0.15), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0., labelspacing=0.2)
for color,text,item in zip([color_NB[1], color_NB[0]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)


## rect1_22: Psychometric function (over dx_broad, or dx_corr ?), Monkey A
ax_0   = fig_temp.add_axes(rect1_22_0)
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
ax.errorbar( d_evidence_A_list[12:],    P_corr_A_list_chooseLow[12:], ErrBar_P_corr_A_list_chooseLow[12:], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Corr.' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_A_list[1:9], 1.-P_corr_A_list_chooseLow[1:9], ErrBar_P_corr_A_list_chooseLow[1:9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], fmt='.', zorder=3, clip_on=False, label='Lower SD Corr.', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_A_list[11], P_corr_A_list_chooseLow[11], ErrBar_P_corr_A_list_chooseLow[11], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_A_list[9], 1.-P_corr_A_list_chooseLow[9], ErrBar_P_corr_A_list_chooseLow[9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], marker='.', zorder=3, clip_on=False                      , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_A_non_drug_chooseLow, x_list_psychometric), color=color_NB[1], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_A_non_drug_chooseLow, -x_list_psychometric), color=color_NB[0], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_A_non_drug_chooseLow, x0_psychometric), s=15., color=color_NB[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_A_non_drug_chooseLow, -x0_psychometric), s=15., color=color_NB[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=2.)
ax_0.set_ylim([0.4,1.])
ax.set_ylim([0.4,1.])
ax_0.set_xlim([-1,1])
ax.set_xlim([1,50])
ax_0.set_xticks([0.])
ax.xaxis.set_ticks([1, 10])
ax_0.set_yticks([0.5, 1.])
ax_0.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax_0.yaxis.set_minor_locator(minorLocator)
ax.set_yticks([])
ax_0.tick_params(direction='out', pad=1.5)
ax_0.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
## Add breakmark = wiggle
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.1135
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))





## rect1_32: Psychometric function (over dx_broad, or dx_corr ?), Monkey A
ax_0   = fig_temp.add_axes(rect1_32_0)
ax   = fig_temp.add_axes(rect1_32)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
ax.errorbar( d_evidence_H_list[12:],    P_corr_H_list_chooseHigh[12:], ErrBar_P_corr_H_list_chooseHigh[12:], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Corr.' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_H_list[1:9], 1.-P_corr_H_list_chooseHigh[1:9], ErrBar_P_corr_H_list_chooseHigh[1:9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], fmt='.', zorder=3, clip_on=False, label='Lower SD Corr.', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_H_list[11], P_corr_H_list_chooseHigh[11], ErrBar_P_corr_H_list_chooseHigh[11], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_H_list[9], 1.-P_corr_H_list_chooseHigh[9], ErrBar_P_corr_H_list_chooseHigh[9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], marker='.', zorder=3, clip_on=False                      , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_H_non_drug_chooseHigh, x_list_psychometric), color=color_NB[1], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_H_non_drug_chooseHigh, -x_list_psychometric), color=color_NB[0], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_H_non_drug_chooseHigh, x0_psychometric), s=15., color=color_NB[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_H_non_drug_chooseHigh, -x0_psychometric), s=15., color=color_NB[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=2.)
ax_0.set_ylim([0.4,1.])
ax.set_ylim([0.4,1.])
ax_0.set_xlim([-1,1])
ax.set_xlim([1,50])
ax_0.set_xticks([0.])
ax.xaxis.set_ticks([1, 10])
ax_0.set_yticks([0.5, 1.])
ax_0.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax_0.yaxis.set_minor_locator(minorLocator)
ax.set_yticks([])
ax_0.tick_params(direction='out', pad=1.5)
ax_0.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
## Add breakmark = wiggle
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.1135
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))


## rect1_42: Psychometric function (over dx_broad, or dx_corr ?), Monkey A
ax_0   = fig_temp.add_axes(rect1_42_0)
ax   = fig_temp.add_axes(rect1_42)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
ax.errorbar( d_evidence_H_list[12:],    P_corr_H_list_chooseLow[12:], ErrBar_P_corr_H_list_chooseLow[12:], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Corr.' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_H_list[1:9], 1.-P_corr_H_list_chooseLow[1:9], ErrBar_P_corr_H_list_chooseLow[1:9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], fmt='.', zorder=3, clip_on=False, label='Lower SD Corr.', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_H_list[11], P_corr_H_list_chooseLow[11], ErrBar_P_corr_H_list_chooseLow[11], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_H_list[9], 1.-P_corr_H_list_chooseLow[9], ErrBar_P_corr_H_list_chooseLow[9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], marker='.', zorder=3, clip_on=False                      , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_H_non_drug_chooseLow, x_list_psychometric), color=color_NB[1], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_H_non_drug_chooseLow, -x_list_psychometric), color=color_NB[0], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_H_non_drug_chooseLow, x0_psychometric), s=15., color=color_NB[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_H_non_drug_chooseLow, -x0_psychometric), s=15., color=color_NB[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=2.)
ax_0.set_ylim([0.4,1.])
ax.set_ylim([0.4,1.])
ax_0.set_xlim([-1,1])
ax.set_xlim([1,50])
ax_0.set_xticks([0.])
ax.xaxis.set_ticks([1, 10])
ax_0.set_yticks([0.5, 1.])
ax_0.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax_0.yaxis.set_minor_locator(minorLocator)
ax.set_yticks([])
ax_0.tick_params(direction='out', pad=1.5)
ax_0.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
## Add breakmark = wiggle
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.1135
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))








### Mean and Var only. Averaged across L/R
## rect1_13: Regression Analysis, Monkey A
ax   = fig_temp.add_axes(rect1_13)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(np.arange(2), Reg_bars_mean_var_LRdiff_A_nondrug_chooseHigh[1:], bar_width, yerr=Reg_bars_Err_mean_var_LRdiff_A_nondrug_chooseHigh[1:], ecolor='k', alpha=1, color=Reg_combined_color_list[0:2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.scatter([0.4,1.4], [26.3,5.], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(Reg_bars_mean_var_LRdiff_A_nondrug_chooseHigh[1:])-1+bar_width])
ax.set_ylim([0.,27.5])
ax.set_xticks(np.arange(len(Reg_bars_mean_var_LRdiff_A_nondrug_chooseHigh[1:]))+bar_width/2. + [-0.2,0.2])
ax.xaxis.set_ticklabels(['Mean', 'Std'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0, 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))

### Mean and Var only. Averaged across L/R
## rect1_23: Regression Analysis, Monkey A
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(np.arange(2), Reg_bars_mean_var_LRdiff_A_nondrug_chooseLow[1:], bar_width, yerr=Reg_bars_Err_mean_var_LRdiff_A_nondrug_chooseLow[1:], ecolor='k', alpha=1, color=Reg_combined_color_list[0:2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.scatter([0.4,1.4], [26.5,10.], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(Reg_bars_mean_var_LRdiff_A_nondrug_chooseLow[1:])-1+bar_width])
ax.set_ylim([0.,27.5])
ax.set_xticks(np.arange(len(Reg_bars_mean_var_LRdiff_A_nondrug_chooseLow[1:]))+bar_width/2. + [-0.2,0.2])
ax.xaxis.set_ticklabels(['Mean', 'Std'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0, 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))

## rect1_33: Regression Analysis, Monkey H
ax   = fig_temp.add_axes(rect1_33)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(np.arange(len(Reg_bars_mean_var_LRdiff_H_nondrug_chooseHigh[1:])), Reg_bars_mean_var_LRdiff_H_nondrug_chooseHigh[1:], bar_width, yerr=Reg_bars_Err_mean_var_LRdiff_H_nondrug_chooseHigh[1:], ecolor='k', alpha=1, color=Reg_combined_color_list[0:2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
ax.scatter([0.4,1.4], [23.2,4.], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(Reg_bars_mean_var_LRdiff_H_nondrug_chooseHigh[1:])-1+bar_width])
ax.set_ylim([0.,27.5])
ax.set_xticks(np.arange(len(Reg_bars_mean_var_LRdiff_H_nondrug_chooseHigh[1:]))+bar_width/2. + [-0.2,0.2])
ax.xaxis.set_ticklabels(['Mean', 'Std'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0, 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))


## rect1_43: Regression Analysis, Monkey H
ax   = fig_temp.add_axes(rect1_43)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(np.arange(len(Reg_bars_mean_var_LRdiff_H_nondrug_chooseLow[1:])), Reg_bars_mean_var_LRdiff_H_nondrug_chooseLow[1:], bar_width, yerr=Reg_bars_Err_mean_var_LRdiff_H_nondrug_chooseLow[1:], ecolor='k', alpha=1, color=Reg_combined_color_list[0:2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
ax.scatter([0.4,1.4], [20.5,5.5], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(Reg_bars_mean_var_LRdiff_H_nondrug_chooseLow[1:])-1+bar_width])
ax.set_ylim([0.,27.5])
ax.set_xticks(np.arange(len(Reg_bars_mean_var_LRdiff_H_nondrug_chooseLow[1:]))+bar_width/2. + [-0.2,0.2])
ax.xaxis.set_ticklabels(['Mean', 'Std'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0, 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))


## rect1_14: Regression Analysis. Mean/Max/Min/First/Last model
ax   = fig_temp.add_axes(rect1_14)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(x_LRsep_list[:-2], Reg_bars_LRsep_A_nondrug_chooseHigh, bar_width/2., yerr=Reg_bars_err_LRsep_A_nondrug_chooseHigh, ecolor='k', alpha=1, color=color_LRsep_list, clip_on=False, align='edge', error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
for b in bar_temp.errorbar[1]:
    b.set_clip_on(False)
for b in bar_temp.errorbar[2]:
    b.set_clip_on(False)
for i_bar in range(len(bar_temp)):
    if (i_bar % 2)==1:
        bar_temp[i_bar].set_hatch('////')
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,0.5*len(Reg_bars_LRsep_A_nondrug_chooseHigh)-1+bar_width])
ax.set_ylim([0.,27.5])
ax.set_xticks(np.arange(len(Reg_bars_LRsep_A_nondrug_chooseHigh)/2)+bar_opacity/2.)
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0, 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(axis='x', direction='out', pad=7.5)
ax.tick_params(axis='y', direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend_bars = [ Patch(facecolor='grey', edgecolor='k', label='Left'), \
                Patch(facecolor='grey', edgecolor='k', hatch='////', label='Right')]
ax.legend(handles=legend_bars, loc=(0.65,0.5), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)

## rect1_24: Regression Analysis. Mean/Max/Min/First/Last model
ax   = fig_temp.add_axes(rect1_24)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(x_LRsep_list[:-2], Reg_bars_LRsep_A_nondrug_chooseLow, bar_width/2., yerr=Reg_bars_err_LRsep_A_nondrug_chooseLow, ecolor='k', alpha=1, color=color_LRsep_list, clip_on=False, align='edge', error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
for b in bar_temp.errorbar[1]:
    b.set_clip_on(False)
for b in bar_temp.errorbar[2]:
    b.set_clip_on(False)
for i_bar in range(len(bar_temp)):
    if (i_bar % 2)==1:
        bar_temp[i_bar].set_hatch('////')
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,0.5*len(Reg_bars_LRsep_A_nondrug_chooseLow)-1+bar_width])
ax.set_ylim([0.,27.5])
ax.set_xticks(np.arange(len(Reg_bars_LRsep_A_nondrug_chooseLow)/2)+bar_opacity/2.)
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0, 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(axis='x', direction='out', pad=14.5)
ax.tick_params(axis='y', direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend_bars = [ Patch(facecolor='grey', edgecolor='k', label='Left'), \
                Patch(facecolor='grey', edgecolor='k', hatch='////', label='Right')]
ax.legend(handles=legend_bars, loc=(0.65,0.5), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)


## rect1_34: Regression Analysis. Mean/Max/Min/First/Last model
ax   = fig_temp.add_axes(rect1_34)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(x_LRsep_list[:-2], Reg_bars_LRsep_H_nondrug_chooseHigh, bar_width/2., yerr=Reg_bars_err_LRsep_H_nondrug_chooseHigh, ecolor='k', alpha=1, color=color_LRsep_list, clip_on=False, align='edge', error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
for b in bar_temp.errorbar[1]:
    b.set_clip_on(False)
for b in bar_temp.errorbar[2]:
    b.set_clip_on(False)
for i_bar in range(len(bar_temp)):
    if (i_bar % 2)==1:
        bar_temp[i_bar].set_hatch('////')
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,0.5*len(Reg_bars_LRsep_H_nondrug_chooseHigh)-1+bar_width])
ax.set_ylim([0.,27.5])
ax.set_xticks(np.arange(len(Reg_bars_LRsep_H_nondrug_chooseHigh)/2)+bar_opacity/2.)
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0, 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(axis='x', direction='out', pad=6.)
ax.tick_params(axis='y', direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend_bars = [ Patch(facecolor='grey', edgecolor='k', label='Left'), \
                Patch(facecolor='grey', edgecolor='k', hatch='////', label='Right')]
ax.legend(handles=legend_bars, loc=(0.65,0.5), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)

## rect1_44: Regression Analysis. Mean/Max/Min/First/Last model
ax   = fig_temp.add_axes(rect1_44)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(x_LRsep_list[:-2], Reg_bars_LRsep_H_nondrug_chooseLow, bar_width/2., yerr=Reg_bars_err_LRsep_H_nondrug_chooseLow, ecolor='k', alpha=1, color=color_LRsep_list, clip_on=False, align='edge', error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
for b in bar_temp.errorbar[1]:
    b.set_clip_on(False)
for b in bar_temp.errorbar[2]:
    b.set_clip_on(False)
for i_bar in range(len(bar_temp)):
    if (i_bar % 2)==1:
        bar_temp[i_bar].set_hatch('////')
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,0.5*len(Reg_bars_LRsep_H_nondrug_chooseLow)-1+bar_width])
ax.set_ylim([0.,27.5])
ax.set_xticks(np.arange(len(Reg_bars_LRsep_H_nondrug_chooseLow)/2)+bar_opacity/2.)
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0, 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(axis='x', direction='out', pad=4.5)
ax.tick_params(axis='y', direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend_bars = [ Patch(facecolor='grey', edgecolor='k', label='Left'), \
                Patch(facecolor='grey', edgecolor='k', hatch='////', label='Right')]
ax.legend(handles=legend_bars, loc=(0.65,0.5), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)


fig_temp.savefig(path_cwd+'Figure4S2.pdf')    #Finally save fig


########################################################################################################################
########################################################################################################################
### Figure 1: Experimental non-drug data, Narrow-Broad Trials

## Schematics
x_schem = np.arange(1,99,1)
sigma_narrow = 12
sigma_broad = 24

## Define Data. Alternatively can also import.                                                                          # See MainAnalysisNonDrugDays.m: NarrowBroadTrialsCOL
Reg_bars_mean_localwins_A = np.array([21.857346345773642,0.039342495547732])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_localwins_A = np.array([0.627071626912533,0.031164428096925])        # Mean, SD, averaged over left and right
Reg_bars_mean_localwins_H = np.array([19.106491100134868,0.001084717528835])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_localwins_H = np.array([0.431052618823326,0.022407847288898])        # Mean, SD, averaged over left and right



## Define subfigure domain.
figsize = (max1, 1.2*max1)

# 4 rows (with sampled mean distribution)
width1_11=0.17; width1_12=width1_11; width1_13=width1_11; width1_11a=width1_11; width1_12a=width1_11a; width1_13a=width1_11a #; width1_22=0.07; width1_21=width1_22*(1+bar_width)/bar_width; width1_23=width1_21; width1_24=width1_22
width1_21= 0.32; width1_22=width1_21
x1_11=0.1; x1_12 = x1_11 + width1_11 + xbuf0*1.3; x1_13 = x1_12 + width1_12 + xbuf0*1.3; x1_11a=x1_11; x1_12a = x1_11a + width1_11a + xbuf0*1.3; x1_13a = x1_12a + width1_12a + xbuf0*1.3#; x1_21=0.105; x1_22 = x1_21 + width1_21 + xbuf0*1.1; x1_23 = x1_22 + width1_22 + xbuf0*1.4; x1_24 = x1_23 + width1_23 + xbuf0*1.1
x1_21=0.095; x1_22 = x1_21 + width1_21 + xbuf0*1.7
height1_11=0.18; height1_12=height1_11; height1_13=height1_11; height1_11a=height1_11; height1_12a=height1_11a; height1_13a=height1_11a; height1_21=0.3; height1_22 = height1_21; height1_23=height1_21; height1_24 = height1_21       # 4 rows
y1_11=0.9; y1_12=y1_11; y1_13=y1_11; y1_11a = y1_11 - height1_11a - 1.7*ybuf0; y1_12a=y1_11a; y1_13a=y1_11a; y1_21 = y1_11a - height1_21 - 2.*ybuf0; y1_22=y1_21; y1_23=y1_21; y1_24=y1_21

rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_11a = [x1_11a, y1_11a, width1_11a, height1_11a]
rect1_12a = [x1_12a, y1_12a, width1_12a, height1_12a]
rect1_13a = [x1_13a, y1_13a, width1_13a, height1_13a]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.1555, 1.215 + y1_21 - y1_11, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.005, 1.215 + y1_21 - y1_11, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.008 + x1_22 - x1_21, 1.215 + y1_21 - y1_11, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.1555 + x1_22 - x1_21, 1.215+ y1_21 - y1_11, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')





## rect1_21: Accuracy with narrow/Broad Correct mean (monkey A)
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_color(Reg_combined_color_list[0])
bar_temp = ax.bar([0], Reg_bars_mean_localwins_A[0], bar_width, yerr=Reg_bars_Err_mean_localwins_A[0], ecolor='k', alpha=1, color=Reg_combined_color_list[0], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
ax.scatter([0.4], [24.], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(Reg_bars_mean_localwins_A)-1+bar_width])
ax.set_ylim([0.,25.])
ax.set_xticks([bar_width/2., 1.+bar_width/2.])
ax.xaxis.set_ticklabels(['Mean\nEvidence', 'Local\nWins'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0, 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.spines['left'].set_position(('outward',0.918))
ax.tick_params(direction='out', pad=1., colors=Reg_combined_color_list[0])
ax.tick_params(which='minor',direction='out', colors=Reg_combined_color_list[0])
ax.tick_params(bottom="off", colors=Reg_combined_color_list[0])
ax.spines['bottom'].set_position(('zero'))
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(),[Reg_combined_color_list[0],'grey']):
    ticklabel.set_color(tickcolor)


## rect1_21 Right axis: FI'' curve
ax_twin = ax.twinx()
ax_twin.spines['top'].set_visible(False)
ax_twin.spines['left'].set_visible(False)
ax_twin.spines['right'].set_color('grey')
bar_temp = ax_twin.bar([1], Reg_bars_mean_localwins_A[1], bar_width, yerr=Reg_bars_Err_mean_localwins_A[1], ecolor='k', alpha=1, color='grey', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
for bar in bar_temp:
    bar.set_edgecolor("k")
ax_twin.set_xlim([0,len(Reg_bars_mean_localwins_A)-1+bar_width])
ax_twin.set_ylim([-0.025,0.075])
ax_twin.set_yticks([-0.025, 0., 0.075])
ax_twin.set_yticklabels([-2.5,0,7.5])
minorLocator = MultipleLocator(0.025)
ax_twin.yaxis.set_minor_locator(minorLocator)
ax_twin.spines['right'].set_position(('outward',0.918))
ax_twin.tick_params(direction='out', pad=1., colors='grey')
ax_twin.tick_params(which='minor',direction='out', colors='grey')
ax_twin.tick_params(bottom="off", colors='grey')
ax_twin.spines['bottom'].set_position(('zero'))
ax_twin.text(1.8, 0.079, r'$\times \mathregular{10^{-4}}$', fontsize=fontsize_tick-0.5, color='grey')




## rect1_22: Accuracy with narrow/Broad Correct mean (monkey A)
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_color(Reg_combined_color_list[0])
bar_temp = ax.bar([0], Reg_bars_mean_localwins_H[0], bar_width, yerr=Reg_bars_Err_mean_localwins_H[0], ecolor='k', alpha=1, color=Reg_combined_color_list[0], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
ax.scatter([0.4], [24.], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(Reg_bars_mean_localwins_H)-1+bar_width])
ax.set_ylim([0.,25.])
ax.set_xticks([bar_width/2., 1.+bar_width/2.])
ax.xaxis.set_ticklabels(['Mean\nEvidence', 'Local\nWins'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0, 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.spines['left'].set_position(('outward',0.918))
ax.tick_params(direction='out', pad=1., colors=Reg_combined_color_list[0])
ax.tick_params(which='minor',direction='out', colors=Reg_combined_color_list[0])
ax.tick_params(bottom="off", colors=Reg_combined_color_list[0])
ax.spines['bottom'].set_position(('zero'))
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(),[Reg_combined_color_list[0],'grey']):
    ticklabel.set_color(tickcolor)

## rect1_22 Right axis: FI'' curve
ax_twin = ax.twinx()
ax_twin.spines['top'].set_visible(False)
ax_twin.spines['left'].set_visible(False)
ax_twin.spines['right'].set_color('grey')
bar_temp = ax_twin.bar([1], Reg_bars_mean_localwins_H[1], bar_width, yerr=Reg_bars_Err_mean_localwins_H[1], ecolor='k', alpha=1, color='grey', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
for bar in bar_temp:
    bar.set_edgecolor("k")
ax_twin.set_xlim([0,len(Reg_bars_mean_localwins_H)-1+bar_width])
ax_twin.set_ylim([-0.025,0.075])
ax_twin.set_yticks([-0.025, 0., 0.075])
ax_twin.set_yticklabels([-2.5,0,7.5])
minorLocator = MultipleLocator(0.025)
ax_twin.yaxis.set_minor_locator(minorLocator)
ax_twin.spines['right'].set_position(('outward',0.918))
ax_twin.tick_params(direction='out', pad=1., colors='grey')
ax_twin.tick_params(which='minor',direction='out', colors='grey')
ax_twin.tick_params(bottom="off", colors='grey')
ax_twin.spines['bottom'].set_position(('zero'))
ax_twin.text(1.8, 0.079, r'$\times \mathregular{10^{-4}}$', fontsize=fontsize_tick-0.5, color='grey')






fig_temp.savefig(path_cwd+'Figure4S3.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
### Figure 1: Experimental non-drug data, Narrow-Broad Trials

## Schematics
x_schem = np.arange(1,99,1)
sigma_narrow = 12
sigma_broad = 24

## Define Data. Alternatively can also import.                                                                          # See MainAnalysisNonDrugDays.m: NarrowBroadTrialsCOL
Reg_bars_lapse_param_saline_ket_A = np.array([0.0000, 0.1321])        # Mean, SD, averaged over left and right
Reg_bars_Err_lapse_param_saline_ket_A = np.array([0.0022, 0.0299])        # Mean, SD, averaged over left and right
Reg_bars_lapse_param_saline_ket_H = np.array([0.0163, 0.0930])        # Mean, SD, averaged over left and right
Reg_bars_Err_lapse_param_saline_ket_H = np.array([0.0094, 0.0435])        # Mean, SD, averaged over left and right


## Define subfigure domain.
figsize = (max1, 1.2*max1)

# 4 rows (with sampled mean distribution)
width1_11=0.19; width1_12=width1_11; width1_13=width1_11; width1_11a=width1_11; width1_12a=width1_11a; width1_13a=width1_11a #; width1_22=0.07; width1_21=width1_22*(1+bar_width)/bar_width; width1_23=width1_21; width1_24=width1_22
width1_21= 0.35; width1_22=width1_21
x1_11=0.115; x1_12 = x1_11 + width1_11 + xbuf0*1.3; x1_13 = x1_12 + width1_12 + xbuf0*1.3; x1_11a=x1_11; x1_12a = x1_11a + width1_11a + xbuf0*1.3; x1_13a = x1_12a + width1_12a + xbuf0*1.3#; x1_21=0.105; x1_22 = x1_21 + width1_21 + xbuf0*1.1; x1_23 = x1_22 + width1_22 + xbuf0*1.4; x1_24 = x1_23 + width1_23 + xbuf0*1.1
x1_21=0.14; x1_22 = x1_21 + width1_21 + xbuf0*1.3
height1_11=0.18; height1_12=height1_11; height1_13=height1_11; height1_11a=height1_11; height1_12a=height1_11a; height1_13a=height1_11a; height1_21=0.3; height1_22 = height1_21; height1_23=height1_21; height1_24 = height1_21       # 4 rows
y1_11=0.9; y1_12=y1_11; y1_13=y1_11; y1_11a = y1_11 - height1_11a - 1.7*ybuf0; y1_12a=y1_11a; y1_13a=y1_11a; y1_21 = y1_11a - height1_21 - 2.*ybuf0; y1_22=y1_21; y1_23=y1_21; y1_24=y1_21

rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_11a = [x1_11a, y1_11a, width1_11a, height1_11a]
rect1_12a = [x1_12a, y1_12a, width1_12a, height1_12a]
rect1_13a = [x1_13a, y1_13a, width1_13a, height1_13a]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.21, 1.215 + y1_21 - y1_11, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.04, 1.2 + y1_21 - y1_11, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.05 + x1_22 - x1_21, 1.2 + y1_21 - y1_11, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.21 + x1_22 - x1_21, 1.215+ y1_21 - y1_11, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')





## rect1_21: Accuracy with narrow/Broad Correct mean (monkey A)
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(np.arange(len(Reg_bars_lapse_param_saline_ket_A)), Reg_bars_lapse_param_saline_ket_A, bar_width, yerr=Reg_bars_Err_lapse_param_saline_ket_A, ecolor='k', alpha=bar_opacity, color=color_list_expt[0:2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
ax.scatter([0.4], [24.], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.set_ylabel('Lapse rate', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(Reg_bars_lapse_param_saline_ket_A)-1+bar_width])
ax.set_ylim([0.,0.17])
ax.set_xticks(np.arange(len(Reg_bars_lapse_param_saline_ket_A))+bar_width/2.)
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 0.15])
ax.set_yticklabels([0, 0.15])
minorLocator = MultipleLocator(0.05)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))


## rect1_22: Accuracy with narrow/Broad Correct mean (monkey A)
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(np.arange(len(Reg_bars_lapse_param_saline_ket_H)), Reg_bars_lapse_param_saline_ket_H, bar_width, yerr=Reg_bars_Err_lapse_param_saline_ket_H, ecolor='k', alpha=bar_opacity, color=color_list_expt[0:2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
ax.scatter([0.4], [21.2], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.set_ylabel('Lapse rate', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(Reg_bars_lapse_param_saline_ket_H)-1+bar_width])
ax.set_ylim([0.,0.17])
ax.set_xticks(np.arange(len(Reg_bars_lapse_param_saline_ket_H))+bar_width/2.)
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 0.15])
ax.set_yticklabels([0, 0.15])
minorLocator = MultipleLocator(0.05)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))






fig_temp.savefig(path_cwd+'Figure8S2.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
### Figure 1: Conceptual/Schematics

## PVB index & lapse rate vs time                                                                                                   # See DrugDayModellingScript.m: DrugDayFigs_TimeCourseAnal
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
t_list_PVB_lapse = np.arange(-20, 61)

PVBIndex_t_mean_list_ketamine_A   = np.array([0.357483795010433, 0.324142901029718, 0.226811424410733, 0.201146176832663, 0.198202448743170, 0.174776912371063, 0.150279485652396, 0.176817490911140, 0.191391754050108, 0.202944500118437, 0.225657964663654, 0.144631630831733, 0.231367852595150, 0.193834492629662, 0.0996570312530812, 0.114419135904181, 0.182860127075701, 0.205521390323677, 0.209433930282980, 0.274836325271661, 0.368174501113594, 0.369805826929246, 0.413131900699504, 0.458792093838905, 0.505203366888092, 0.485169754660403, 0.771873285859257, 0.704541875614230, 0.803113376893058, 0.703708800343094, 0.794846190217767, 0.667930737189954, 0.495487137605619, 0.550140061587012, 0.692390641181355, 0.786799433294895, 0.561160528254663, 0.555419943372197, 0.531461887972395, 0.530781731550108, 0.408861581804610, 0.345635721355878, 0.427587913966307, 0.465718962055659, 0.387003845842942, 0.325612582196837, 0.322537003975779, 0.340356429234869, 0.276769825092293, 0.239059670054173, 0.274787368017037, 0.330390602443091, 0.311532735306195, 0.288483695717313, 0.328176101915054, 0.316294385916040, 0.309161247858417, 0.239810461442150, 0.279193379005947, 0.234496789965291, 0.251898984441871, 0.251942266542231, 0.279282821198909, 0.284315572623443, 0.218442550618599, 0.278407296712731, 0.289243172568412, 0.317882854678131, 0.268813054726404, 0.337724222228366, 0.360978812601452, 0.351291130547263, 0.320827451128364, 0.246194603263157, 0.231146290120133, 0.182149932005441, 0.190532621666215, 0.207241591727215, 0.186414468035614, 0.202533243003821, 0.202633652932940])
PVBIndex_t_mean_list_saline_A   = np.array([0.348605009689304, 0.315470809622112, 0.241616348546411, 0.143082959615261, 0.253374311992654, 0.246742952922485, 0.306902404690545, 0.296967772406900, 0.319035929042285, 0.362852535063040, 0.296343709856336, 0.227467328258162, 0.180461647423033, 0.103176785535771, 0.118018125445720, 0.109018412085104, 0.129843221241711, 0.173669501434966, 0.211484001778795, 0.226875113940091, 0.210033043281251, 0.218592894890730, 0.206989856562512, 0.186529287601572, 0.166517611611529, 0.161309572236918, 0.239422355090303, 0.207107666298840, 0.171698561688780, 0.171970477426520, 0.195549498686815, 0.231881590826926, 0.236540959120783, 0.160543000052549, 0.170509039365559, 0.183723406413464, 0.272238448252072, 0.246676568798276, 0.220222392818596, 0.258173352284528, 0.289951032889827, 0.222699434867816, 0.250298941376622, 0.257997570370235, 0.205729438367559, 0.189403071658796, 0.203345923319391, 0.261063516084620, 0.228309565540958, 0.222101234235681, 0.318209678792626, 0.302486397091743, 0.279838229825568, 0.324557480966533, 0.390009624338769, 0.401917174320855, 0.317856334092470, 0.353048381634363, 0.371135374987180, 0.326509316596467, 0.292923206530343, 0.305241311158825, 0.315087085475913, 0.284696783271980, 0.237115549485711, 0.307056834873513, 0.305439905979312, 0.257389487119335, 0.329205878429139, 0.256767249200903, 0.301505236921445, 0.238630786994046, 0.202756709103288, 0.228242113848665, 0.205260374103405, 0.179220089917755, 0.154876454505393, 0.232475079100569, 0.276939361917257, 0.230209897035546, 0.275717437834976])
PVBIndex_t_mean_list_ketamine_H = np.array([0.261333522107880, 0.219971329159294, 0.210200429942294, 0.227423762723594, 0.205110898847336, 0.148032485699207, 0.108293198662712, 0.0968406614237538, 0.130202282805773, 0.126116387335760, 0.108254609179270, 0.131773699205270, 0.150893868163790, 0.173072003502041, 0.148372603355980, 0.140177643133172, 0.178148739498668, 0.130542330345551, 0.105302686644215, 0.209204578673188, 0.122289308480708, 0.227581971978190, 0.298046750926128, 0.374240397397390, 0.284445933328089, 0.124169480060876, 0.127900490546221, 0.0707824410578505, 0.00961431022030389, 0.0445591469336688, 0.161026563235880, 0.372968563747302, 0.418445725069208, 0.436072857481909, 0.592742979217349, 0.523228173880493, 0.464153765376181, 0.418700272913010, 0.528738508475630, 0.468420654325248, 0.414760458303072, 0.497920278185999, 0.469670833437858, 0.419563552124192, 0.174638110824104, 0.194866559063252, 0.232472965186889, 0.177634141479676, 0.233177419166761, 0.170082135776800, 0.222656455010802, 0.300149525761836, 0.232419079687293, 0.161364815965919, 0.114725562655866, 0.146786927098623, 0.157794983368284, 0.119890455273159, 0.0681510683114965, 0.0666613746104966, 0.100238583137104, 0.173611370682432, 0.187774947179968, 0.190634519986891, 0.240586981201784, 0.257805606304876, 0.296631930376439, 0.276614070620743, 0.201052111466689, 0.182972179761319, 0.275213555624028, 0.212530175046083, 0.233571376512280, 0.223905347198948, 0.266641325916128, 0.218598644869545, 0.134167049004824, 0.185612514921609, 0.136838669627024, 0.185862418769983, 0.170319816066870])
PVBIndex_t_mean_list_saline_H   = np.array([0.118887484721580, 0.0601905789017422, 0.0679799066574797, 0.108493993164067, 0.0391743306491747, 0.0860111467685841, 0.0261897129973755, 0.0495782070345769, 0.0522239545488608, 0.0446910393759709, 0.0430112716002934, 0.0384286499299962, -0.0405200135828429, -0.0544326676491263, -0.0158868214236316, 0.00820397862502858, -0.00217880864538592, 0.0271900362688319, 0.0901665898391893, 0.122757757375993, 0.0857713360070095, 0.111649121978269, 0.116088413988935, 0.0945405786076776, 0.143471374188222, 0.153111389069787, 0.130550086148804, 0.154333494856364, 0.132864570733820, 0.137888367267856, 0.0941968276229960, 0.0465852611401115, 0.0504973899780842, 0.0132137092874287, 0.0650057590506659, 0.0953225267281366, 0.0919691742223526, 0.132119107522570, 0.160400827786992, 0.195128825936538, 0.190025829154907, 0.183863251054306, 0.214767043864418, 0.200570532780622, 0.185556025601119, 0.191449060975254, 0.164476981883094, 0.160794418292064, 0.168834980948783, 0.187037666059978, 0.176902836108158, 0.137286485091623, 0.0983886426637359, 0.107107453223993, 0.124181392727003, 0.142447921300081, 0.143674353301627, 0.182692820290161, 0.230070019384837, 0.195079320430168, 0.143874661434073, 0.145157844091083, 0.134571957203401, 0.0719813275260212, 0.100723192973294, 0.0744168560142847, 0.119020345833975, 0.164961244099272, 0.160524751590331, 0.257534661303375, 0.254616092093376, 0.195998968130867, 0.167696552694730, 0.0782922634656670, 0.0855241208883064, 0.102580134320586, 0.133551527343375, 0.129942067378081, 0.142467842798862, 0.121307385805447, 0.129271659015578])


PVB_sign_range_A = [21,40]; # in matlab -> -1
PVB_sign_range_H = [32,44]; # in matlab -> -1








## Define subfigure domain.
figsize = (max15,0.6*max15)

width1_11 = 0.32; width1_12 = width1_11
width1_21 = width1_11; width1_22 = width1_21
x1_11 = 0.135; x1_12 = x1_11 + width1_11 + 1.7*xbuf0
x1_21 = x1_11; x1_22 = x1_12
height1_11 = 0.3; height1_12 = height1_11
height1_21= height1_11;  height1_22 = height1_21
y1_11 = 0.62; y1_12 = y1_11
y1_21 = y1_11 - height1_21 - 2.35*ybuf0; y1_22 = y1_21

rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]



##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.04, 0.905, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.042+x1_12-x1_11, 0.905, 'B', fontsize=fontsize_fig_label, fontweight='bold')
# fig_temp.text(0.04, 0.912 + y1_22 - y1_12, 'C', fontsize=fontsize_fig_label, fontweight='bold')
# fig_temp.text(0.042+x1_22-x1_21, 0.912 + y1_22 - y1_12, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.2118, 0.945, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.7185, 0.945, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
bar_width_compare3 = 1.



## rect1_11: Correct Probability vs time, Both Monkeys
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax)
# ax_log.spines['left'].set_visible(False)
ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_saline_A, color=color_list_expt[0], linestyle='-', zorder=3, clip_on=False, label='Saline', linewidth=1.)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
# ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_saline_A + PVBIndex_t_se_list_saline_A, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
# ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_saline_A - PVBIndex_t_se_list_saline_A, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_ketamine_A, color=color_list_expt[1], linestyle='-', zorder=3, clip_on=False, label='Ketamine', linewidth=1.)#, linestyle=linestyle_list[i_var_a])
# ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_ketamine_A + PVBIndex_t_se_list_ketamine_A, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=True, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
# ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_ketamine_A - PVBIndex_t_se_list_ketamine_A, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=True, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
ax.plot([t_list_PVB_lapse[PVB_sign_range_A[0]-1],t_list_PVB_lapse[PVB_sign_range_A[1]-1]], [0.95,0.95], color='k', linestyle='-', zorder=3, clip_on=False, linewidth=1., solid_capstyle='projecting')#, linestyle=linestyle_list[i_var_a])
ax.fill_between([5., 30.], y1=1.1,y2=-0.15, lw=0, color='k', alpha=0.2, zorder=0)
ax.set_xlabel('Time (mins)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('PVB Index', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([-20, 60])
ax.set_ylim([-0.15,1.1])
ax.set_xticks([-20, 0, 20, 40, 60])
ax.set_yticks([0., 1.])
# ax.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.25)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
# ax.tick_params(bottom="off")
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
legend = ax.legend(loc=(0.48,0.6), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)


## rect1_12: Correct Probability vs time, Both Monkeys
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax)
# ax_log.spines['left'].set_visible(False)
ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_saline_H, color=color_list_expt[0], linestyle='-', zorder=3, clip_on=False, label='Saline', linewidth=1.)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
# ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_saline_H + PVBIndex_t_se_list_saline_H, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
# ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_saline_H - PVBIndex_t_se_list_saline_H, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_ketamine_H, color=color_list_expt[1], linestyle='-', zorder=3, clip_on=False, label='Ketamine', linewidth=1.)#, linestyle=linestyle_list[i_var_a])
# ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_ketamine_H + PVBIndex_t_se_list_ketamine_H, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
# ax.plot(t_list_PVB_lapse, PVBIndex_t_mean_list_ketamine_H - PVBIndex_t_se_list_ketamine_H, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
ax.plot([t_list_PVB_lapse[PVB_sign_range_H[0]-1],t_list_PVB_lapse[PVB_sign_range_H[1]-1]], [0.95,0.95], color='k', linestyle='-', zorder=3, clip_on=False, linewidth=1., solid_capstyle='projecting')#, linestyle=linestyle_list[i_var_a])
ax.fill_between([5., 30.], y1=1.1,y2=-0.15, lw=0, color='k', alpha=0.2, zorder=0)
ax.set_xlabel('Time (mins)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('PVB Index', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([-20, 60])
ax.set_ylim([-0.15,1.1])
ax.set_xticks([-20, 0, 20, 40, 60])
ax.set_yticks([0., 1.])
# ax.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.25)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
# ax.tick_params(bottom="off")
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
legend = ax.legend(loc=(0.48,0.6), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)





fig_temp.savefig(path_cwd+'Figure8S3.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
### Figure 1: Conceptual/Schematics

## Psychometric function, drug days                                                                                     # See DrugDayModellingScript.m:  line 581-605
d_evidence_A_ket_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_A_ket_list =  np.array([0.250000000000000, 0.273809523809524, 0.315217391304348, 0.381355932203390, 0.418326693227092, 0.401746724890830, 0.438095238095238, 0.728395061728395, 0.651982378854626, 0.733333333333333, 0.773504273504274, 0.773809523809524, 0.727272727272727, 0.740740740740741])  # Log-Spaced.
ErrBar_P_corr_A_ket_list = np.array([0.0968245836551854, 0.0486530315798862, 0.0342509431418564, 0.0316176565920964, 0.0311358334295023, 0.0323967148807855, 0.0484195749920879, 0.0494208053076868, 0.0316159049722523, 0.0276926801084766, 0.0273623526671680, 0.0322774748849353, 0.0507536842035530, 0.0843370433412313])
d_evidence_A_saline_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_A_saline_list =  np.array([0, 0.0253164556962025, 0.0335570469798658, 0.0578034682080925, 0.184834123222749, 0.275862068965517, 0.371794871794872, 0.767857142857143, 0.835164835164835, 0.936781609195402, 0.948113207547170, 0.977941176470588, 1, 1])  # Log-Spaced.
ErrBar_P_corr_A_saline_list = np.array([0, 0.0176733843968922, 0.0147532209150465, 0.0177429017429878, 0.0267222390045845, 0.0371169807659960, 0.0547211398120702, 0.0564188024351951, 0.0275027004936727, 0.0184487213942955, 0.0152331802112380, 0.0125944174823581, 0, 0])
d_evidence_H_ket_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_H_ket_list =  np.array([0.125000000000000, 0.180722891566265, 0.225352112676056, 0.299465240641711, 0.392857142857143, 0.428571428571429, 0.450704225352113, 0.551724137931035, 0.710144927536232, 0.726415094339623, 0.780000000000000, 0.788732394366197, 0.902439024390244, 0.782608695652174])  # Log-Spaced.
ErrBar_P_corr_H_ket_list = np.array([0.116926793336686, 0.0422360161545184, 0.0350621719425284, 0.0334940181656808, 0.0307653954338870, 0.0353479756646710, 0.0590499820040121, 0.0653009760653792, 0.0315339790226193, 0.0306175591696263, 0.0292916370317536, 0.0342560372435389, 0.0327672560474013, 0.0860061487037983])
d_evidence_H_saline_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_H_saline_list =  np.array([0, 0.0161290322580645, 0.0537190082644628, 0.103448275862069, 0.229651162790698, 0.311475409836066, 0.261261261261261, 0.674698795180723, 0.847826086956522, 0.867132867132867, 0.918918918918919, 0.942408376963351, 0.979381443298969, 0.968750000000000])  # Log-Spaced.
ErrBar_P_corr_H_saline_list = np.array([0, 0.0113125988060140, 0.0144932695145458, 0.0170511612676205, 0.0226776902342200, 0.0296467231279874, 0.0416985983545450, 0.0514231595382512, 0.0216206601771226, 0.0200709805014482, 0.0158654540752180, 0.0168570876222110, 0.0144284340623192, 0.0307578432578586])

## Psychometric function for large sensory deficit
d_evidence_model_sensory_deficit =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_sensory_deficit_xpt5 =  np.array([0.503448275862069, 0.458837772397094, 0.487288135593220, 0.479456824512535, 0.490707531790023, 0.493689680772086, 0.508113120074177, 0.497897897897898, 0.497564935064935, 0.498440424204616, 0.507073954983923, 0.506318449873631, 0.519526627218935, 0.498178506375228, 0.532883043014575, 0.528481012658228, 0.533425223983460, 0.540540540540541, 0.589435774309724, 0.616883116883117])  # Log-Spaced.
ErrBar_P_corr_model_sensory_deficit_xpt5 = np.array([0.0415217524540122, 0.0173381715447929, 0.0108455691114776, 0.00932203894792067, 0.00902688897116495, 0.00963244678846483, 0.0107643485396003, 0.0122534687413911, 0.0142449031229686, 0.00883051703498844, 0.00896491839271647, 0.0145114217917772, 0.0121533279059287, 0.0106696691932136, 0.00940684310977362, 0.00888015908024706, 0.00926080411746290, 0.0112537933900905, 0.0170445797192969, 0.0391747945226784])
P_corr_model_sensory_deficit_xpt6 =  np.array([0.220689655172414, 0.372881355932203, 0.427966101694915, 0.454735376044568, 0.477339419628301, 0.499257609502598, 0.493277700509968, 0.476876876876877, 0.497564935064935, 0.494697442295696, 0.514469453376206, 0.535804549283909, 0.543786982248521, 0.546448087431694, 0.543192321365091, 0.567088607594937, 0.580634045485872, 0.654767975522693, 0.707082833133253, 0.740259740259740])  # Log-Spaced.
ErrBar_P_corr_model_sensory_deficit_xpt6 = np.array([0.0344399258351854, 0.0168255847762684, 0.0107358964000982, 0.00929160660377463, 0.00901917130123077, 0.00963320339208613, 0.0107647928577845, 0.0122404665649618, 0.0142449031229686, 0.00883006339806144, 0.00896206070768849, 0.0144753235177678, 0.0121158778943675, 0.0106236019153873, 0.00939201208459560, 0.00881417009905742, 0.00916007710935929, 0.0107364438143640, 0.0157683066961621, 0.0353346944409579])



def Psychometric_fit(params_pm, pm_fit2, x_list):
    prob_corr_fit = 0.5 + 0.5*(1. - np.exp(-(x_list/100/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    to_min = -sum(np.log(prob_corr_fit)*pm_fit2) - sum(np.log(1.-prob_corr_fit)*(1.-pm_fit2))                                                          # Maximum Likelihood Estimator
    return to_min
def Psychometric_function(params_pm, x_list):
    prob_corr_fit = 0.5 + 0.5*(1. - np.exp(-(x_list/100/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    return prob_corr_fit
## Fitting Psychometric Functions
x_list_psychometric = np.arange(1, 50, 1)                                                                        # See figure_psychometric_function_fit.py, esp lines 322-527
x0_psychometric = 0.
## non-binned MLE (i.e. done using literal net evidence, via matlab). See Psychometric_function_fit_NonDrugDays_NL.m. Also note there is no shift parameters.
psychometric_params_A_ketamine_all      = [0.333901061072675, 0.615907828200203, 0.0380758582398804]
psychometric_params_H_ketamine_all      = [0.171475348179639, 0.998334934315150, 0.0252640762972983]
psychometric_params_model_sensory_coeff_xpt5      = [0.841516605388732, 1.50264133959267, 0.0423356135008768]
psychometric_params_model_sensory_coeff_xpt6      = [0.339047696453902, 1.65744413483465, 0.0289649178312268]


### Rescale Psychometric function based on lapse rate.
lapse_rate_ket_A_H = np.array([0.13212969, 0.09301522])
P_corr_model_sensory_deficit_xpt5_A = lapse_rate_ket_A_H[0] + (1.-2.*lapse_rate_ket_A_H[0])*P_corr_model_sensory_deficit_xpt5
P_corr_model_sensory_deficit_xpt5_H = lapse_rate_ket_A_H[1] + (1.-2.*lapse_rate_ket_A_H[1])*P_corr_model_sensory_deficit_xpt5
P_corr_model_sensory_deficit_xpt6_A = lapse_rate_ket_A_H[0] + (1.-2.*lapse_rate_ket_A_H[0])*P_corr_model_sensory_deficit_xpt6
P_corr_model_sensory_deficit_xpt6_H = lapse_rate_ket_A_H[1] + (1.-2.*lapse_rate_ket_A_H[1])*P_corr_model_sensory_deficit_xpt6
ErrBar_P_corr_model_sensory_deficit_xpt5_A = (1.-2.*lapse_rate_ket_A_H[0])*ErrBar_P_corr_model_sensory_deficit_xpt5
ErrBar_P_corr_model_sensory_deficit_xpt5_H = (1.-2.*lapse_rate_ket_A_H[1])*ErrBar_P_corr_model_sensory_deficit_xpt5
ErrBar_P_corr_model_sensory_deficit_xpt6_A = (1.-2.*lapse_rate_ket_A_H[0])*ErrBar_P_corr_model_sensory_deficit_xpt6
ErrBar_P_corr_model_sensory_deficit_xpt6_H = (1.-2.*lapse_rate_ket_A_H[1])*ErrBar_P_corr_model_sensory_deficit_xpt6









## Define subfigure domain.
figsize = (max1,1.*max1)

width1_11 = 0.3; width1_12 = width1_11
width1_21 = width1_11; width1_22 = width1_21
x1_11 = 0.21; x1_12 = x1_11 + width1_11 + 1.4*xbuf0
x1_21 = x1_11; x1_22 = x1_12
height1_11 = 0.3; height1_12 = height1_11
height1_21= height1_11;  height1_22 = height1_21
y1_11 = 0.62; y1_12 = y1_11
y1_21 = y1_11 - height1_21 - 2.35*ybuf0; y1_22 = y1_21

rect1_11_0 = [x1_11, y1_11, width1_11*0.05, height1_11]
rect1_11 = [x1_11+width1_11*0.2, y1_11, width1_11*(1-0.2), height1_11]
rect1_12_0 = [x1_12, y1_12, width1_12*0.05, height1_12]
rect1_12 = [x1_12+width1_12*0.2, y1_12, width1_12*(1-0.2), height1_12]
rect1_21_0 = [x1_21, y1_21, width1_21*0.05, height1_21]
rect1_21 = [x1_21+width1_21*0.2, y1_21, width1_21*(1-0.2), height1_21]
rect1_22_0 = [x1_22, y1_22, width1_22*0.05, height1_22]
rect1_22 = [x1_22+width1_22*0.2, y1_22, width1_22*(1-0.2), height1_22]



##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.11, 0.915, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.115+x1_12-x1_11, 0.915, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.11, 0.915 + y1_22 - y1_12, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.115+x1_22-x1_21, 0.915 + y1_22 - y1_12, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.253, 0.96, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.71, 0.96, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.03, 0.882, 'Ketamine data', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k')
fig_temp.text(0.055, 0.36, 'Sensory Deficit\nModel', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k', horizontalalignment='center')
bar_width_compare3 = 1.



## rect1_11: Psychometric function (over dx_corr), monkey A
ax_0   = fig_temp.add_axes(rect1_11_0)
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
ax.errorbar( d_evidence_A_ket_list[6:],    P_corr_A_ket_list[6:], ErrBar_P_corr_A_ket_list[6:], color=color_list_expt[1], markerfacecolor=color_list_expt[1], ecolor=color_list_expt[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_A_ket_list[:6], 1.-P_corr_A_ket_list[:6], ErrBar_P_corr_A_ket_list[:6], color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], markerfacecolor=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[1]], fmt='.', zorder=4, clip_on=False, label='Lower SD Correct', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(x_list_psychometric, Psychometric_function_D(psychometric_params_A_ketamine_all, 0.01*x_list_psychometric), color=color_list_expt[1], ls='-', clip_on=False, zorder=3)#, linestyle=linestyle_list[i_var_a])
ax.plot(x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_A_ketamine_all, -0.01*x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ls='-', clip_on=False, zorder=2)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_A_ketamine_all, x0_psychometric), s=15., color=color_list_expt[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_A_ketamine_all, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=-5.)
ax_0.set_ylim([0.4,1.])
ax.set_ylim([0.4,1.])
ax_0.set_xlim([-1,1])
ax.set_xlim([1,50])
ax_0.set_xticks([0.])
ax.xaxis.set_ticks([1, 10])
ax_0.set_yticks([0.5, 1.])
ax_0.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax_0.yaxis.set_minor_locator(minorLocator)
ax.set_yticks([])
ax_0.tick_params(direction='out', pad=1.5)
ax_0.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
## Add breakmark = wiggle
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.0968
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_list_expt[1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.6,0.7), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0.)
for color,text,item in zip([color_list_expt[1], [1-(1-ci)*0.5 for ci in color_list_expt[1]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)


## rect1_12: Psychometric function (over dx_corr), monkey H
ax_0   = fig_temp.add_axes(rect1_12_0)
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_H_ket_list[6:],    P_corr_H_ket_list[6:], ErrBar_P_corr_H_ket_list[6:], color=color_list_expt[1], markerfacecolor=color_list_expt[1], ecolor=color_list_expt[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_H_ket_list[:6], 1.-P_corr_H_ket_list[:6], ErrBar_P_corr_H_ket_list[:6], color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], markerfacecolor=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[1]], fmt='.', zorder=4, clip_on=False, label='Lower SD Correct', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(x_list_psychometric, Psychometric_function_D(psychometric_params_H_ketamine_all, 0.01*x_list_psychometric), color=color_list_expt[1], ls='-', clip_on=False, zorder=3)#, linestyle=linestyle_list[i_var_a])
ax.plot(x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_H_ketamine_all, -0.01*x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ls='-', clip_on=False, zorder=2)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_H_ketamine_all, x0_psychometric), s=15., color=color_list_expt[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_H_ketamine_all, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=-5.)
ax_0.set_ylim([0.4,1.])
ax.set_ylim([0.4,1.])
ax_0.set_xlim([-1,1])
ax.set_xlim([1,50])
ax_0.set_xticks([0.])
ax.xaxis.set_ticks([1, 10])
ax_0.set_yticks([0.5, 1.])
ax_0.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax_0.yaxis.set_minor_locator(minorLocator)
ax.set_yticks([])
ax_0.tick_params(direction='out', pad=1.5)
ax_0.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
## Add breakmark = wiggle
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.0968
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_list_expt[1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.6,0.7), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0.)
for color,text,item in zip([color_list_expt[1], [1-(1-ci)*0.5 for ci in color_list_expt[1]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

## rect1_21: Psychometric function (over dx_corr), monkey A
ax_0   = fig_temp.add_axes(rect1_21_0)
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
ax.errorbar( d_evidence_model_sensory_deficit[6:],    P_corr_model_sensory_deficit_xpt6_A[6:], ErrBar_P_corr_model_sensory_deficit_xpt6_A[6:], color=color_list[-1], markerfacecolor=color_list[-1], ecolor=color_list[-1], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_model_sensory_deficit[:6], 1.-P_corr_model_sensory_deficit_xpt6_A[:6], ErrBar_P_corr_model_sensory_deficit_xpt6_A[:6], color=[1-(1-ci)*0.5 for ci in color_list[-1]], markerfacecolor=[1-(1-ci)*0.5 for ci in color_list[-1]], ecolor=[1-(1-ci)*0.5 for ci in color_list[-1]], fmt='.', zorder=4, clip_on=False, label='Lower SD Correct', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(x_list_psychometric, lapse_rate_ket_A_H[0] + (1.-2.*lapse_rate_ket_A_H[0])*Psychometric_function_D(psychometric_params_model_sensory_coeff_xpt6, 0.01*x_list_psychometric), color=color_list[-1], ls='-', clip_on=False, zorder=3)#, linestyle=linestyle_list[i_var_a])
ax.plot(x_list_psychometric, 1.-lapse_rate_ket_A_H[0] - (1.-2.*lapse_rate_ket_A_H[0])*Psychometric_function_D(psychometric_params_model_sensory_coeff_xpt6, -0.01*x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list[-1]], ls='-', clip_on=False, zorder=2)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, lapse_rate_ket_A_H[0] + (1.-2.*lapse_rate_ket_A_H[0])*Psychometric_function_D(psychometric_params_model_sensory_coeff_xpt6, x0_psychometric), s=15., color=color_list[-1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-lapse_rate_ket_A_H[0] - (1.-2.*lapse_rate_ket_A_H[0])*Psychometric_function_D(psychometric_params_model_sensory_coeff_xpt6, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list[-1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=-5.)
ax_0.set_ylim([0.4,1.])
ax.set_ylim([0.4,1.])
ax_0.set_xlim([-1,1])
ax.set_xlim([1,50])
ax_0.set_xticks([0.])
ax.xaxis.set_ticks([1, 10])
ax_0.set_yticks([0.5, 1.])
ax_0.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax_0.yaxis.set_minor_locator(minorLocator)
ax.set_yticks([])
ax_0.tick_params(direction='out', pad=1.5)
ax_0.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
## Add breakmark = wiggle
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.0968
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_list[-1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=[1-(1-ci)*0.5 for ci in color_list[-1]], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.6,0.7), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0.)
for color,text,item in zip([color_list[-1], [1-(1-ci)*0.5 for ci in color_list[-1]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)


## rect1_22: Psychometric function (over dx_corr), monkey A
ax_0   = fig_temp.add_axes(rect1_22_0)
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
ax.errorbar( d_evidence_model_sensory_deficit[6:],    P_corr_model_sensory_deficit_xpt6_H[6:], ErrBar_P_corr_model_sensory_deficit_xpt6_H[6:], color=color_list[-1], markerfacecolor=color_list[-1], ecolor=color_list[-1], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_model_sensory_deficit[:6], 1.-P_corr_model_sensory_deficit_xpt6_H[:6], ErrBar_P_corr_model_sensory_deficit_xpt6_H[:6], color=[1-(1-ci)*0.5 for ci in color_list[-1]], markerfacecolor=[1-(1-ci)*0.5 for ci in color_list[-1]], ecolor=[1-(1-ci)*0.5 for ci in color_list[-1]], fmt='.', zorder=4, clip_on=False, label='Lower SD Correct', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(x_list_psychometric, lapse_rate_ket_A_H[1] + (1.-2.*lapse_rate_ket_A_H[1])*Psychometric_function_D(psychometric_params_model_sensory_coeff_xpt6, 0.01*x_list_psychometric), color=color_list[-1], ls='-', clip_on=False, zorder=3)#, linestyle=linestyle_list[i_var_a])
ax.plot(x_list_psychometric, 1.-lapse_rate_ket_A_H[1] - (1.-2.*lapse_rate_ket_A_H[1])*Psychometric_function_D(psychometric_params_model_sensory_coeff_xpt6, -0.01*x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list[-1]], ls='-', clip_on=False, zorder=2)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, lapse_rate_ket_A_H[1] + (1.-2.*lapse_rate_ket_A_H[1])*Psychometric_function_D(psychometric_params_model_sensory_coeff_xpt6, x0_psychometric), s=15., color=color_list[-1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-lapse_rate_ket_A_H[1] - (1.-2.*lapse_rate_ket_A_H[1])*Psychometric_function_D(psychometric_params_model_sensory_coeff_xpt6, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list[-1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=-5.)
ax_0.set_ylim([0.4,1.])
ax.set_ylim([0.4,1.])
ax_0.set_xlim([-1,1])
ax.set_xlim([1,50])
ax_0.set_xticks([0.])
ax.xaxis.set_ticks([1, 10])
ax_0.set_yticks([0.5, 1.])
ax_0.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax_0.yaxis.set_minor_locator(minorLocator)
ax.set_yticks([])
ax_0.tick_params(direction='out', pad=1.5)
ax_0.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
## Add breakmark = wiggle
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.0968
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_list[-1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=[1-(1-ci)*0.5 for ci in color_list[-1]], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.6,0.7), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0.)
for color,text,item in zip([color_list[-1], [1-(1-ci)*0.5 for ci in color_list[-1]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)



fig_temp.savefig(path_cwd+'Figure8S7.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
### Figure 8: Ketamine Data
## Mean/Variance Regression model                                                                                       # See DrugDayModellingScript.m: line 434-460
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
Reg_bars_A_ketamine = np.array([0.875741073112642, 11.528262865855169, 5.233422331691739])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on ketamine.
Reg_bars_H_ketamine = np.array([-0.054946117132906, 11.298580759143913, 3.616936576677555])  # [Bias, Val diff , Std diff]. Henry regression Beta values on ketamine.
Reg_bars_A_saline = np.array([0.123672806343438, 25.565433692118590, 5.762095065977849])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on saline.
Reg_bars_H_saline = np.array([-0.015314326931326, 22.141685413467790, 3.313026659905981])  # [Bias, Val diff , Std diff]. Henry regression Beta values on saline.
Reg_bars_err_low_A_ketamine = np.array([0.047236118202221, 8.30532974619988, 3.42216791743146])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on ketamine.
Reg_bars_err_low_H_ketamine = np.array([0.051815368521218, 8.631952828223486,1.974824680486972])  # [Bias, Val diff , Std diff]. Henry regression Beta values on ketamine.
Reg_bars_err_low_A_saline = np.array([0.080275332862987, 23.5954694311769, 4.00138687044306])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on saline.
Reg_bars_err_low_H_saline = np.array([0.057234482845660, 20.149389511091050,1.817929049926931])  # [Bias, Val diff , Std diff]. Henry regression Beta values on saline.
Reg_bars_err_up_A_ketamine = np.array([0.047236118202221, 15.0229631401706, 7.31212628604223])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on ketamine.
Reg_bars_err_up_H_ketamine = np.array([0.051815368521218, 15.041377610498401,5.814631823925737])  # [Bias, Val diff , Std diff]. Henry regression Beta values on ketamine.
Reg_bars_err_up_A_saline = np.array([0.080275332862987, 27.7782041609912, 7.58936157838872])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on saline.
Reg_bars_err_up_H_saline = np.array([0.057234482845660, 24.672555881182920,5.034848487306757])  # [Bias, Val diff , Std diff]. Henry regression Beta values on saline.

PVBindex_err_lh_A_saline = [0.158529070056875,0.290987785859371];
PVBindex_err_lh_A_ketamine = [0.314567528447066,0.604359636340356];
PVBindex_err_lh_H_saline = [0.084186275972941,0.216805836292652];
PVBindex_err_lh_H_ketamine = [0.188612245498817,0.457690756672795];


Reg_bars_A_pre_ketamine = np.array([-0.018486360992728, 25.066552469697804, 5.476637879268277])  # [Bias, Val diff , Std diff]. Alfie regression Beta values pre ketamine.
Reg_bars_H_pre_ketamine = np.array([-0.088145566508677, 20.242263778486283, 2.989324913443232])  # [Bias, Val diff , Std diff]. Henry regression Beta values pre ketamine.
Reg_bars_A_pre_saline = np.array([0.025344551399459, 25.080324535558805, 7.222049938157925])  # [Bias, Val diff , Std diff]. Alfie regression Beta values pre saline.
Reg_bars_H_pre_saline = np.array([0.031205437334164, 20.469342955391840, 1.421425556593712])  # [Bias, Val diff , Std diff]. Henry regression Beta values pre saline.
Reg_bars_err_A_pre_ketamine = np.array([0.070992670850458, 0.996841131251648, 0.822609782456692])  # [Bias, Val diff , Std diff]. Alfie regression Beta values pre ketamine.
Reg_bars_err_H_pre_ketamine = np.array([0.066413408732252, 0.809835262014914, 0.774257285106152])  # [Bias, Val diff , Std diff]. Henry regression Beta values pre ketamine.
Reg_bars_err_A_pre_saline = np.array([0.077956130762707, 1.075750024238907, 0.992814700495137])  # [Bias, Val diff , Std diff]. Alfie regression Beta values pre saline.
Reg_bars_err_H_pre_saline = np.array([0.056119259718531, 0.682254829601997, 0.663911849258801])  # [Bias, Val diff , Std diff]. Henry regression Beta values pre saline.


## Using across-session analysis                                                                                        # See DrugDayModellingScript.m: DrugDayFigs_ProVarEffects (lines 294-297)
mean_effect_list_A    = np.array([Reg_bars_A_saline[1], Reg_bars_A_ketamine[1]])             # Saline/ketamine. Mean Regressor
var_effect_list_A     = np.array([Reg_bars_A_saline[2], Reg_bars_A_ketamine[2]])              # Saline/ketamine. Variance Regressor
var_mean_ratio_list_A = var_effect_list_A/mean_effect_list_A            # Saline/ketamine. Variance Regressor/ Mean Regressor
mean_effect_list_H    = np.array([Reg_bars_H_saline[1], Reg_bars_H_ketamine[1]])             # Saline/ketamine. Mean Regressor
var_effect_list_H     = np.array([Reg_bars_H_saline[2], Reg_bars_H_ketamine[2]])              # Saline/ketamine. Variance Regressor
var_mean_ratio_list_H = var_effect_list_H/mean_effect_list_H            # Saline/ketamine. Variance Regressor/ Mean Regressor
mean_effect_list_A_preSK    = np.array([Reg_bars_A_pre_saline[1], Reg_bars_A_pre_ketamine[1]])             # Saline/ketamine. Mean Regressor
var_effect_list_A_preSK     = np.array([Reg_bars_A_pre_saline[2], Reg_bars_A_pre_ketamine[2]])              # Saline/ketamine. Variance Regressor
var_mean_ratio_list_A_preSK = var_effect_list_A_preSK/mean_effect_list_A_preSK            # Saline/ketamine. Variance Regressor/ Mean Regressor
mean_effect_list_H_preSK    = np.array([Reg_bars_H_pre_saline[1], Reg_bars_H_pre_ketamine[1]])             # Saline/ketamine. Mean Regressor
var_effect_list_H_preSK     = np.array([Reg_bars_H_pre_saline[2], Reg_bars_H_pre_ketamine[2]])              # Saline/ketamine. Variance Regressor
var_mean_ratio_list_H_preSK = var_effect_list_H_preSK/mean_effect_list_H_preSK            # Saline/ketamine. Variance Regressor/ Mean Regressor




Mean_reg_err_bars_A_v2  = np.abs(np.array([[Reg_bars_err_low_A_saline[1] - Reg_bars_A_saline[1], Reg_bars_err_low_A_ketamine[1] - Reg_bars_A_ketamine[1]], [Reg_bars_err_up_A_saline[1] - Reg_bars_A_saline[1], Reg_bars_err_up_A_ketamine[1] - Reg_bars_A_ketamine[1]]]))
Mean_reg_err_bars_H_v2  = np.abs(np.array([[Reg_bars_err_low_H_saline[1] - Reg_bars_H_saline[1], Reg_bars_err_low_H_ketamine[1] - Reg_bars_H_ketamine[1]], [Reg_bars_err_up_H_saline[1] - Reg_bars_H_saline[1], Reg_bars_err_up_H_ketamine[1] - Reg_bars_H_ketamine[1]]]))
Var_reg_err_bars_A_v2  = np.abs(np.array([[Reg_bars_err_low_A_saline[2] - Reg_bars_A_saline[2], Reg_bars_err_low_A_ketamine[2] - Reg_bars_A_ketamine[2]], [Reg_bars_err_up_A_saline[2] - Reg_bars_A_saline[2], Reg_bars_err_up_A_ketamine[2] - Reg_bars_A_ketamine[2]]]))
Var_reg_err_bars_H_v2  = np.abs(np.array([[Reg_bars_err_low_H_saline[2] - Reg_bars_H_saline[2], Reg_bars_err_low_H_ketamine[2] - Reg_bars_H_ketamine[2]], [Reg_bars_err_up_H_saline[2] - Reg_bars_H_saline[2], Reg_bars_err_up_H_ketamine[2] - Reg_bars_H_ketamine[2]]]))
PVBindex_err_bars_A_v2  = np.abs(np.array([[PVBindex_err_lh_A_saline[0] - var_mean_ratio_list_A[0], PVBindex_err_lh_A_ketamine[0] - var_mean_ratio_list_A[1]], [PVBindex_err_lh_A_saline[1] - var_mean_ratio_list_A[0], PVBindex_err_lh_A_ketamine[1] - var_mean_ratio_list_A[1]]]))
PVBindex_err_bars_H_v2  = np.abs(np.array([[PVBindex_err_lh_H_saline[0] - var_mean_ratio_list_H[0], PVBindex_err_lh_H_ketamine[0] - var_mean_ratio_list_H[1]], [PVBindex_err_lh_H_saline[1] - var_mean_ratio_list_H[0], PVBindex_err_lh_H_ketamine[1] - var_mean_ratio_list_H[1]]]))


### PK                                                                                                                    # See DrugDayModellingScript.m: end of DrugDayFigs_PsychKernel.m (For old, unpaired method in see lines 275-430).
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
i_PK_list_6 = np.arange(1,6+1)
t_PK_list_6 = 0.125 + 0.25*np.arange(6)
PK_A_ketamine = np.array([3.907325777201067,2.821637146141091,2.857311283544315,2.186128890636230,2.014742285904774,2.218665930934278])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_ketamine = np.array([2.276538239865711,2.263225994878025,2.241860915854905,2.712738372931044,1.58972093951529,1.498378289313458])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_A_saline = np.array([4.797902007903518,4.034137010062050,4.291776493769314,3.711598738218275,3.512305575973259,5.018266914136376])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_saline = np.array([4.056775507537500,3.524791991643180,3.366469345834473,3.518023517012175,3.760479001226668,3.667192910652407])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.

# PK_A_ketamine_errbar = np.array([0.181909966668632, 0.176946758892851, 0.177572620178772, 0.178282426674973, 0.184364067419049, 0.176687477165004])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
# PK_H_ketamine_errbar = np.array([0.159372647178722, 0.159925072505669, 0.160508358698636, 0.163262950152970, 0.161709018929227, 0.161379976272803])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
# PK_A_saline_errbar = np.array([0.355754158014346, 0.330038122089591, 0.337514707486557, 0.329125923448195, 0.315738162506590, 0.352890409481412])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
# PK_H_saline_errbar = np.array([0.234501270200309, 0.219481714028684, 0.211731025923342, 0.220078112403546, 0.222437119478033, 0.222560460414643])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_A_ketamine_errbar_low = np.array([2.23629133945989, 1.59886714706573, 1.80939584762806, 1.39016526456575, 1.08371259111269, 1.26874461246663])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_A_ketamine_errbar_high = np.array([6.26861764348127, 4.43616191589453, 4.16972526535744, 3.16516288119788, 3.32936171712675, 3.42385053029132])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_A_saline_errbar_low = np.array([4.18443929210583, 3.45904274624742, 3.73549700672343, 3.18345485579270, 2.95142870034307, 4.43544070610893])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_A_saline_errbar_high = np.array([6.06180279527001, 4.99365958165957, 5.31968463487817, 4.58545428540351, 4.33347220074346, 6.02043684514872])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_ketamine_errbar_low = np.array([1.48208218814063, 1.46827851839214, 1.34042040779829, 1.78994926419599, 0.985307577505681, 0.903064758786001])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_ketamine_errbar_high = np.array([3.76842982459975, 3.76456325364197, 3.91845062799255, 4.42514469797282, 2.56692678724885, 2.39462426415374])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_saline_errbar_low = np.array([3.49785007939408, 3.01510008771437, 2.89132005888288, 3.00011107313006, 3.26833191549070, 3.14838247173371])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_saline_errbar_high = np.array([4.85309672511362, 4.19386120316926, 4.01517733129787, 4.21666705789109, 4.40483750907540, 4.33651461806905])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.









## Regression analysis, Experiments                                                                                     # See DrugDayModellingScript.m: line227-261, DrugRegStrat
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
Reg_values_A_ketamine = np.array([1.592681059366568, 0.462825580482115, -0.059602876258340, 11.650201509774572, 1.353304468101836, -2.442057469018322, -1.373779812274950, 0.301250018907861, -10.245358390440714, -2.250702877822999, 0.960700464922197, 0.121478336581161])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_A_saline = np.array([-0.122055702111979, 1.390122169259026, 1.409042367426404, 23.079397952352977, 1.727573630925311, -2.097889286602293, -0.824712158316410, -1.079694766083464, -22.665467138971223, -2.186871692387939, 1.570767790499428, 1.896491234487414e-07])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_H_ketamine = np.array([-1.387231777467695, 0.500273506165217, -0.549182545909488, 9.490841069472282, 3.265848349920347, -0.004152320848420, 0.411102145140890, 0.602966043355566, -12.071384031861697, -0.760022147172587, 1.110242494866628, 0.075970303899447])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_H_saline = np.array([-0.116971115692486, 1.112601182765259, 0.107184845046210, 18.074079452914660, 2.720967638654494, -0.199201619304004, 0.123258768189988, -0.333289161826822, -22.952636275847220, -0.248489447621877, 1.081266137135686, 0.010807242389919])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)

Reg_values_A_ketamine_lrmean = 0.5*(Reg_values_A_ketamine[[3,4,5,1,2]]-Reg_values_A_ketamine[[8,9,10,6,7]])
Reg_values_A_saline_lrmean = 0.5*(Reg_values_A_saline[[3,4,5,1,2]]-Reg_values_A_saline[[8,9,10,6,7]])
Reg_values_H_ketamine_lrmean = 0.5*(Reg_values_H_ketamine[[3,4,5,1,2]]-Reg_values_H_ketamine[[8,9,10,6,7]])
Reg_values_H_saline_lrmean = 0.5*(Reg_values_H_saline[[3,4,5,1,2]]-Reg_values_H_saline[[8,9,10,6,7]])

Reg_values_errbar_low_A_ketamine = np.abs(np.array([5.249042371483373,0.093028858973010,-3.984516410838989,-0.272510487523184,-0.990212199958535]) - Reg_values_A_ketamine_lrmean) # Error bars for Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_up_A_ketamine = np.abs(np.array([15.409707567179229,3.968243877201712,0.775870630344921,2.572699781859696,0.728744267394364]) - Reg_values_A_ketamine_lrmean) # Error bars for Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_low_A_saline = np.abs(np.array([19.798541177845177,0.566990933491439,-3.578000722425427,0.180613931110705,0.327643795104687]) - Reg_values_A_saline_lrmean)  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_up_A_saline = np.abs(np.array([26.865820642325240,3.780550359314451,-0.038276785775205,1.947469475908966,2.070072481059210]) - Reg_values_A_saline_lrmean)  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_low_H_ketamine = np.abs(np.array([7.104045370435397,-0.358430286526839,-2.306590559806222,-1.121055605180648,-1.548903464831781]) - Reg_values_H_ketamine_lrmean) # Error bars for Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_up_H_ketamine = np.abs(np.array([15.389487380072383,5.074895214695435,1.280952375287867,1.146388264924713,0.235883149830632]) - Reg_values_H_ketamine_lrmean) # Error bars for Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_low_H_saline = np.abs(np.array([15.769542190124387,-0.353201065221144,-2.269329593902764,-0.651990522485495,-0.496065017129729]) - Reg_values_H_saline_lrmean)  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_up_H_saline = np.abs(np.array([25.657797657833243,4.082120438573978,0.984246177090373,1.775933660973312,0.956620790186758]) - Reg_values_H_saline_lrmean)  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)

print(Reg_values_errbar_low_A_ketamine)
print(Reg_values_errbar_up_A_ketamine)




## Define subfigure domain.
figsize = (max2,1.2*max2)
width1_11 = 0.37; width1_12 = width1_11
width1_21 = 0.15; width1_22 = 0.15; width1_23 = width1_21; width1_24 = width1_22
width1_31 = 0.08; width1_32 = width1_31; width1_33 = width1_31; width1_34 = width1_31; width1_35 = width1_31; width1_36 = width1_31
width1_41 = 0.19; width1_42 = 0.12; width1_43 = width1_41; width1_44 = width1_42
x1_11 = 0.09; x1_12 = x1_11 + width1_11 + 1.2*xbuf0; x1_13 = x1_12 + width1_12 + xbuf0
x1_21 = 0.09; x1_22 = x1_21 + width1_21 + 0.7*xbuf0; x1_23 = x1_22 + width1_22 + 1.15*xbuf0; x1_24 = x1_23 + width1_23 + 0.7*xbuf0
x1_31 = 0.08; x1_32 = x1_31 + width1_31 + 0.62*xbuf0; x1_33 = x1_32 + width1_32 + 0.7*xbuf0; x1_34 = x1_33 + width1_33 + 1.07*xbuf0; x1_35 = x1_34 + width1_34 + 0.62*xbuf0; x1_36 = x1_35 + width1_35 + 0.7*xbuf0
x1_41 = 0.08; x1_42 = x1_41 + width1_41 + 0.7*xbuf0; x1_43 = x1_42 + width1_42 + 1.05*xbuf0; x1_44 = x1_43 + width1_43 + 0.7*xbuf0
height1_11 = 0.17; height1_12 = height1_11; height1_13 = height1_12
height1_21= 0.17;  height1_22 = height1_21;  height1_23 = height1_21;  height1_24 = height1_21
height1_31= 0.17;  height1_32 = height1_31;  height1_33 = height1_31; height1_34 = height1_31; height1_35 = height1_31; height1_36 = height1_31
height1_41= 0.17;  height1_42 = height1_41;  height1_43 = height1_41;  height1_44 = height1_41
y1_11 = 0.79; y1_12 = y1_11; y1_13=y1_12
y1_21 = y1_11 - height1_21 - 0.9*ybuf0; y1_22 = y1_21; y1_23 = y1_21; y1_24 = y1_21
y1_31 = y1_21 - height1_31 - 1.*ybuf0; y1_32 = y1_31; y1_33 = y1_31;  y1_34 = y1_31; y1_35 = y1_31; y1_36 = y1_31;
y1_41 = y1_31 - height1_41 - 0.85*ybuf0; y1_42 = y1_41; y1_43 = y1_41; y1_44 = y1_41

rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_21_0 = [x1_21, y1_21, width1_21*0.05, height1_21]
rect1_21 = [x1_21+width1_21*0.2, y1_21, width1_21*(1-0.2), height1_21]
rect1_22_0 = [x1_22, y1_22, width1_22*0.05, height1_22]
rect1_22 = [x1_22+width1_22*0.2, y1_22, width1_22*(1-0.2), height1_22]
rect1_23_0 = [x1_23, y1_23, width1_23*0.05, height1_23]
rect1_23 = [x1_23+width1_23*0.2, y1_23, width1_23*(1-0.2), height1_23]
rect1_24_0 = [x1_24, y1_24, width1_24*0.05, height1_24]
rect1_24 = [x1_24+width1_24*0.2, y1_24, width1_24*(1-0.2), height1_24]
rect1_31 = [x1_31, y1_31, width1_31, height1_31]
rect1_32 = [x1_32, y1_32, width1_32, height1_32]
rect1_33 = [x1_33, y1_33, width1_33, height1_33]
rect1_34 = [x1_34, y1_34, width1_34, height1_34]
rect1_35 = [x1_35, y1_35, width1_35, height1_35]
rect1_36 = [x1_36, y1_36, width1_36, height1_36]
rect1_41 = [x1_41, y1_41, width1_41, height1_41]
rect1_42 = [x1_42, y1_42, width1_42, height1_42]
rect1_43 = [x1_43, y1_43, width1_43, height1_43]
rect1_44 = [x1_44, y1_44, width1_44, height1_44]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.2162, 0.975+y1_31-y1_11, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.718, 0.975+y1_31-y1_11, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.015, 0.955 + y1_31 - y1_11, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.03+x1_34-x1_31, 0.955 + y1_31 - y1_11, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015, 0.955 + y1_41 - y1_11, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015+x1_42-x1_41, 0.955 + y1_41 - y1_11, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.027+x1_43-x1_41, 0.955 + y1_41 - y1_11, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.02+x1_44-x1_41, 0.955 + y1_41 - y1_11, 'F', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.


## rect1_31: Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_31)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar(np.arange(len(mean_effect_list_A)), mean_effect_list_A, bar_width_compare3, alpha=bar_opacity, yerr=Mean_reg_err_bars_A_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar1.errorbar[1]:
    b.set_clip_on(False)
for b in bar1.errorbar[2]:
    b.set_clip_on(False)
for bar in bar1:
    bar.set_edgecolor("k")
    bar.set_clip_on(False)
ax.scatter([1.], [30.2], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [28.9,28.9], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Mean Evidence Beta', fontsize=fontsize_legend, labelpad=1.5)
ax.set_xlim([0,len(mean_effect_list_A)-1+bar_width_compare3])
ax.set_ylim([0.,30.])
ax.set_xticks([0., 1.])
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
ax.set_yticks([0., 30.])
ax.set_yticklabels([0., 0.3])
minorLocator = MultipleLocator(10.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")

## rect1_32: Variance Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_32)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar(np.arange(len(var_effect_list_A)), var_effect_list_A, bar_width_compare3, alpha=bar_opacity, yerr=Var_reg_err_bars_A_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for bar in bar1:
    bar.set_edgecolor("k")
    bar.set_clip_on(False)
ax.set_ylabel('SD Evidence Beta', fontsize=fontsize_legend, labelpad=1.5)
ax.set_xlim([0,len(var_effect_list_A)-1+bar_width_compare3])
ax.set_ylim([0.,8])
ax.set_xticks([0., 1.])
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
ax.set_yticks([0., 8.])
ax.set_yticklabels([0., 0.08])
minorLocator = MultipleLocator(2.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")

## rect1_33: Variance Beta/ Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_33)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar(np.arange(len(var_mean_ratio_list_A)), var_mean_ratio_list_A, bar_width_compare3, alpha=bar_opacity, yerr=PVBindex_err_bars_A_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for bar in bar1:
    bar.set_edgecolor("k")
    bar.set_clip_on(False)
ax.scatter([1.], [0.65], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [0.625,0.625], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('PVB Index', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(var_mean_ratio_list_A)-1+bar_width_compare3])
ax.set_ylim([0.,0.65])
ax.set_xticks([0., 1.])
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
ax.set_yticks([0., 0.6])
ax.yaxis.set_ticklabels([0, 0.6])
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")


## rect1_34: Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_34)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar(np.arange(len(mean_effect_list_H)), mean_effect_list_H, bar_width_compare3, alpha=bar_opacity, yerr=Mean_reg_err_bars_H_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for bar in bar1:
    bar.set_edgecolor("k")
    bar.set_clip_on(False)
ax.scatter([1.], [27.], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [25.7,25.7], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Mean Evidence Beta', fontsize=fontsize_legend, labelpad=1.5)
ax.set_xlim([0,len(mean_effect_list_H)-1+bar_width_compare3])
ax.set_ylim([0.,30.])
ax.set_xticks([0., 1.])
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
ax.set_yticks([0., 30.])
ax.set_yticklabels([0., 0.3])
minorLocator = MultipleLocator(10.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")

## rect1_35: Variance Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_35)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar(np.arange(len(var_effect_list_H)), var_effect_list_H, bar_width_compare3, alpha=bar_opacity, yerr=Var_reg_err_bars_H_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for bar in bar1:
    bar.set_edgecolor("k")
    bar.set_clip_on(False)
ax.set_ylabel('SD Evidence Beta', fontsize=fontsize_legend, labelpad=1.5)
ax.set_xlim([0,len(var_effect_list_H)-1+bar_width_compare3])
ax.set_ylim([0.,8.])
ax.set_xticks([0., 1.])
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
ax.set_yticks([0., 8.])
ax.set_yticklabels([0., 0.08])
minorLocator = MultipleLocator(2.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")

## rect1_36: Variance Beta/ Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_36)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar(np.arange(len(var_mean_ratio_list_H)), var_mean_ratio_list_H, bar_width_compare3, alpha=bar_opacity, yerr=PVBindex_err_bars_A_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for bar in bar1:
    bar.set_edgecolor("k")
    bar.set_clip_on(False)
ax.scatter([1.], [0.53], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [0.5,0.5], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('PVB Index', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(var_mean_ratio_list_H)-1+bar_width_compare3])
ax.set_ylim([0.,0.65])
ax.set_xticks([0., 1.])
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
ax.set_yticks([0., 0.6])
ax.yaxis.set_ticklabels([0, 0.6])
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")



## rect1_41: Monkey A: Regression Values
bar_width_figS1 = 0.42
ax   = fig_temp.add_axes(rect1_41)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar(np.arange(5)                , 0.5*(Reg_values_A_saline[[ 3,4,5,1,2]]-Reg_values_A_saline[[ 8,9,10,6,7]]), bar_width_figS1, yerr=[Reg_values_errbar_low_A_saline, Reg_values_errbar_up_A_saline], ecolor='k', alpha=bar_opacity, color=color_list_expt[0], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar1.errorbar[1]:
    b.set_clip_on(False)
for b in bar1.errorbar[2]:
    b.set_clip_on(False)
for bar in bar1:
    bar.set_edgecolor("k")
    bar.set_clip_on(False)
bar3 = ax.bar(np.arange(5)+bar_width_figS1, 0.5*(Reg_values_A_ketamine[[3,4,5,1,2]]-Reg_values_A_ketamine[[8,9,10,6,7]]), bar_width_figS1, yerr=[Reg_values_errbar_low_A_ketamine, Reg_values_errbar_up_A_ketamine], ecolor='k', alpha=bar_opacity, color=color_list_expt[1], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar3.errorbar[1]:
    b.set_clip_on(False)
for b in bar3.errorbar[2]:
    b.set_clip_on(False)
for bar in bar3:
    bar.set_edgecolor("k")
    bar.set_clip_on(False)
ax.scatter([bar_width_figS1], [29.6], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([bar_width_figS1*0.5,bar_width_figS1*1.5], [28.2,28.2], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.scatter([4+bar_width_figS1], [4.4], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([4+bar_width_figS1*0.5,4+bar_width_figS1*1.5], [3.,3.], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Beta', fontsize=fontsize_legend)
ax.set_xlim([0.,4.+2.*bar_width_figS1])
ax.set_ylim([0.,30.])
ax.set_xticks([bar_width_figS1, 1.+bar_width_figS1, 2.+bar_width_figS1, 3.+bar_width_figS1, 4.+bar_width_figS1])
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 30.])
ax.set_yticklabels([0., 0.3])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=14.5, axis='x')
ax.tick_params(direction='out', pad=1., axis='y')
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend = ax.legend([bar1, bar3], ('Saline', 'Ketamine'), loc=(0.25,0.75), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)
ax.spines['bottom'].set_position(('zero'))


## rect1_42: Psychophysical Kernel, Monkey A
ax   = fig_temp.add_axes(rect1_42)
fig_funs.remove_topright_spines(ax)
tmp = ax.errorbar(i_PK_list_6, PK_A_ketamine, np.abs([PK_A_ketamine_errbar_low-PK_A_ketamine,PK_A_ketamine_errbar_high-PK_A_ketamine]), color=color_list_expt[1], linestyle='-', marker='.', zorder=(3-1), clip_on=False, markeredgecolor='k', elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
tmp = ax.errorbar(i_PK_list_6, PK_A_saline, np.abs([PK_A_saline_errbar_low-PK_A_saline,PK_A_saline_errbar_high-PK_A_saline]), color=color_list_expt[0], linestyle='-', marker='.', zorder=(3-1), clip_on=False, markeredgecolor='k', elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
ax.scatter([6], [6.3], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_xlabel('Sample Number', fontsize=fontsize_legend)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend)
ax.set_ylim([0.,6.4])
ax.set_yticks([0., 6.])
ax.text(0., 6.6, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
ax.set_xlim([1,6])
ax.set_xticks([1,6])
minorLocator = MultipleLocator(2.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))



## rect1_43: Monkey H: Regression Values
bar_width_figS1 = 0.42
ax   = fig_temp.add_axes(rect1_43)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar(np.arange(5)                , 0.5*(Reg_values_H_saline[[ 3,4,5,1,2]]-Reg_values_H_saline[[ 8,9,10,6,7]]), bar_width_figS1, yerr=[Reg_values_errbar_low_H_saline, Reg_values_errbar_up_H_saline], ecolor='k', alpha=bar_opacity, color=color_list_expt[0], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar1.errorbar[1]:
    b.set_clip_on(False)
for b in bar1.errorbar[2]:
    b.set_clip_on(False)
for bar in bar1:
    bar.set_edgecolor("k")
    bar.set_clip_on(False)
bar3 = ax.bar(np.arange(5)+bar_width_figS1, 0.5*(Reg_values_H_ketamine[[3,4,5,1,2]]-Reg_values_H_ketamine[[8,9,10,6,7]]), bar_width_figS1, yerr=[Reg_values_errbar_low_H_ketamine, Reg_values_errbar_up_H_ketamine], ecolor='k', alpha=bar_opacity, color=color_list_expt[1], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar3.errorbar[1]:
    b.set_clip_on(False)
for b in bar3.errorbar[2]:
    b.set_clip_on(False)
for bar in bar3:
    bar.set_edgecolor("k")
    bar.set_clip_on(False)
ax.scatter([bar_width_figS1], [28.4], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([bar_width_figS1*0.5,bar_width_figS1*1.5], [27.,27.], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Beta', fontsize=fontsize_legend)
ax.set_xlim([0.,4.+2.*bar_width_figS1])
ax.set_ylim([0.,30.])
ax.set_xticks([bar_width_figS1, 1.+bar_width_figS1, 2.+bar_width_figS1, 3.+bar_width_figS1, 4.+bar_width_figS1])
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 30.])
ax.set_yticklabels([0., 0.3])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=8.5, axis='x')
ax.tick_params(direction='out', pad=1., axis='y')
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend = ax.legend([bar1, bar3], ('Saline', 'Ketamine'), loc=(0.25,0.75), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)
ax.spines['bottom'].set_position(('zero'))


## rect1_44: Psychophysical Kernel, Monkey H
ax   = fig_temp.add_axes(rect1_44)
fig_funs.remove_topright_spines(ax)
tmp = ax.errorbar(i_PK_list_6, PK_H_ketamine, np.abs([PK_H_ketamine_errbar_low-PK_H_ketamine,PK_H_ketamine_errbar_high-PK_H_ketamine]), color=color_list_expt[1], linestyle='-', marker='.', zorder=(3-1), clip_on=False, markeredgecolor='k', elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
tmp = ax.errorbar(i_PK_list_6, PK_H_saline, np.abs([PK_H_saline_errbar_low-PK_H_saline,PK_H_saline_errbar_high-PK_H_saline]), color=color_list_expt[0], linestyle='-', marker='.', zorder=(3-1), clip_on=False, markeredgecolor='k', elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
ax.scatter([[1,2,5,6]], [5.3,5.3,5.3,5.3], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_xlabel('Sample Number', fontsize=fontsize_legend)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend)
ax.set_ylim([0.,6.4])
ax.set_yticks([0., 6.])
ax.text(0., 6.6, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
ax.set_xlim([1,6])
ax.set_xticks([1,6])
minorLocator = MultipleLocator(2.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

fig_temp.savefig(path_cwd+'Figure8S2.pdf')    #Finally save fig

########################################################################################################################
