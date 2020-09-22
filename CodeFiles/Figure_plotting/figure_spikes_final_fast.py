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


label_list     = ['Control', 'Lowered E/I', 'Elevated E/I', 'Sensory Deficit']         #Manually used variable.
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
########################################################################################################################
########################################################################################################################
### Figure 2: General Task performance.

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

# Psychophyical Kernal
i_PK_list = np.arange(1,8+1)
t_PK_list = 0.125 + 0.25*np.arange(8)
n_A_non_drug = 41.                                                                                                      # Alfie, 41 runs
n_H_non_drug = 63.                                                                                                      # Henry, 63 runs
PK_A_nondrug = np.array([3.97349317466863, 3.14577917590026, 3.14308934214697, 3.09270194108838, 2.65869626053626, 2.58228332627445, 2.26243327684102, 2.32548588191760])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_nondrug = np.array([3.24972294613704, 2.87215850443560, 2.41144038611466, 2.58619140237903, 2.25506241804606, 2.28328415494767, 1.97777876818272, 2.08155718671906])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_A_nondrug_errbar = np.array([0.114436877777907, 0.104349370934396, 0.107810651060901, 0.107321477770204, 0.103722082853755, 0.101065259589198, 0.101660772227339, 0.101973548343162])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data).
PK_H_nondrug_errbar = np.array([0.0777015532433701, 0.0738638886817392, 0.0738767825058227, 0.0749264426465390, 0.0723778004740631, 0.0716455370014343, 0.0717771595016059, 0.0713933395660157])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data).


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
ax.errorbar( d_evidence_A_non_drug_list_corr[2:],    P_corr_A_list_non_drug_corr[2:], ErrBar_P_corr_A_list_non_drug_corr[2:], color='k', markerfacecolor='grey', ecolor='grey', fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_A_non_drug_list_corr[1], P_corr_A_list_non_drug_corr[1], ErrBar_P_corr_A_list_non_drug_corr[1], color='k', markerfacecolor='grey', ecolor='grey', marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(x_list_psychometric, Psychometric_function(psychometric_params_A_non_drug_corr, x_list_psychometric), color='k', ls='-', clip_on=False, zorder=2)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(x0_psychometric, Psychometric_function(psychometric_params_A_non_drug_corr, x0_psychometric), s=15., color='k', marker='_', clip_on=False, zorder=2, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
y_shift_spines = -0.0968
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))


## rect1_12: Psychometric function (over dx_corr), monkey H
ax_0   = fig_temp.add_axes(rect1_12_0)
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
ax.errorbar( d_evidence_H_non_drug_list_corr[2:],    P_corr_H_list_non_drug_corr[2:], ErrBar_P_corr_H_list_non_drug_corr[2:], color='k', markerfacecolor='grey', ecolor='grey', fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_H_non_drug_list_corr[1], P_corr_H_list_non_drug_corr[1], ErrBar_P_corr_H_list_non_drug_corr[1], color='k', markerfacecolor='grey', ecolor='grey', marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(x_list_psychometric, Psychometric_function(psychometric_params_H_non_drug_corr, x_list_psychometric), color='k', ls='-', clip_on=False, zorder=2)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(x0_psychometric, Psychometric_function(psychometric_params_H_non_drug_corr, x0_psychometric), s=15., color='k', marker='_', clip_on=False, zorder=2, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
y_shift_spines = -0.0968
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))


## rect1_21: Psychophysical Kernel, monkey A
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
ax.errorbar(i_PK_list, PK_A_nondrug, PK_A_nondrug_errbar, color='k', markerfacecolor='grey', ecolor='grey', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.set_xlabel('Sample Number', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylim([0.,4.05])
ax.set_xlim([1.,8.])
ax.set_xticks([1,8])
ax.set_yticks([0., 4.])
ax.text(0.1, 4.2, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_22: Psychophysical Kernel, monkey H
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
ax.errorbar(i_PK_list, PK_H_nondrug, PK_H_nondrug_errbar, color='k', markerfacecolor='grey', ecolor='grey', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.set_xlabel('Sample Number', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylim([0.,4.05])
ax.set_xlim([1.,8.])
ax.set_xticks([1., 8.])
ax.set_yticks([0., 4.])
ax.text(0.1, 4.2, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

fig_temp.savefig(path_cwd+'Figure2.pdf')    #Finally save fig

########################################################################################################################
################################################################################################################################################################################################################################################
########################################################################################################################
### Figure 3: Experimental non-drug data, Narrow-Broad Trials

## Schematics
x_schem = np.arange(1,99,1)
sigma_narrow = 12
sigma_broad = 24

## Ambiguous (Equal)/ Narrow-Correct/ Broad-Correct bars
ENB_bars_avg_non_drug = np.array([0.595744680851064, 0.759604190919674, 0.894594594594595])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Henry, 63 runs
ENB_bars_err_avg_non_drug = np.array([0.0110454746698929, 0.0103096884725492, 0.00798203667012338])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Alfie, 41 runs

ENB_bars_avg_Tsetsos = np.array([0.623, 0.671, 0.781])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Tsetsos 2012, PNAS
ENB_bars_err_avg_Tsetsos = np.array([0.09598/2., 0.065/2., 0.08979/2.])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Error bars are guessed right now.

## Extract number distribution of stimuli, for Narrow-Broad trials.                                                     # See MainAnalysisNonDrugDays.m: lines 137-187)
n_distribution_narrow_high_avg = np.loadtxt('Data/Stim_Distribution/dx=0.05/n_distribution_narrow_high_avg.txt', delimiter=',')              # axis is (narrow, broad)
n_distribution_broad_high_avg = np.loadtxt('Data/Stim_Distribution/dx=0.05/n_distribution_broad_high_avg.txt', delimiter=',')              # axis is (narrow, broad)
n_distribution_NB_balanced_avg = np.loadtxt('Data/Stim_Distribution/dx=0.05/n_distribution_NB_balanced_avg.txt', delimiter=',')              # axis is (narrow, broad)
density_distribution_narrow_high_all = (n_distribution_narrow_high_avg) / np.sum(n_distribution_narrow_high_avg)
density_distribution_broad_high_all = (n_distribution_broad_high_avg) / np.sum(n_distribution_broad_high_avg)
density_distribution_NB_balanced_all = (n_distribution_NB_balanced_avg) / np.sum(n_distribution_NB_balanced_avg)
density_distribution_net_narrow_high_all = np.zeros(len(density_distribution_narrow_high_all))
density_distribution_net_broad_high_all  = np.zeros(len(density_distribution_broad_high_all))
density_distribution_net_NB_balanced_all = np.zeros(len(density_distribution_NB_balanced_all))
for i in range(len(density_distribution_net_narrow_high_all)):
    density_distribution_net_narrow_high_all[i] = np.sum(density_distribution_narrow_high_all.diagonal(i - int((len(density_distribution_narrow_high_all)-1.)/2.)))
    density_distribution_net_broad_high_all[ i] = np.sum(density_distribution_broad_high_all.diagonal( i - int((len(density_distribution_broad_high_all )-1.)/2.)))
    density_distribution_net_NB_balanced_all[i] = np.sum(density_distribution_NB_balanced_all.diagonal(i - int((len(density_distribution_NB_balanced_all)-1.)/2.)))

dx_density=0.05
length_density = int(100/dx_density) +1
x_pm = 2
n_x_smooth=40
density_distribution_net_narrow_high_all_smooth_2p = analysis.sliding_win_on_lin_data(density_distribution_net_narrow_high_all[ (int(52/dx_density) +1):(int(64/dx_density) +2)], n_x_smooth)
density_distribution_net_narrow_high_all_smooth_n2m = analysis.sliding_win_on_lin_data(density_distribution_net_narrow_high_all[(int(36/dx_density) +1):(int(48/dx_density) +2)], n_x_smooth)
density_distribution_net_narrow_high_all_smooth = np.zeros(length_density)
density_distribution_net_narrow_high_all_smooth[(int(52/dx_density) +1):(int(64/dx_density) +2)] = density_distribution_net_narrow_high_all_smooth_2p
density_distribution_net_narrow_high_all_smooth[(int(36/dx_density) +1):(int(48/dx_density) +2)] = density_distribution_net_narrow_high_all_smooth_n2m
density_distribution_net_broad_high_all_smooth_2p = analysis.sliding_win_on_lin_data(density_distribution_net_broad_high_all[(int(52/dx_density) +1):(int(64/dx_density) +2)], n_x_smooth)
density_distribution_net_broad_high_all_smooth_n2m = analysis.sliding_win_on_lin_data(density_distribution_net_broad_high_all[(int(36/dx_density) +1):(int(48/dx_density) +2)], n_x_smooth)
density_distribution_net_broad_high_all_smooth = np.zeros(length_density)
density_distribution_net_broad_high_all_smooth[(int(52/dx_density) +1):(int(64/dx_density) +2)] = density_distribution_net_broad_high_all_smooth_2p
density_distribution_net_broad_high_all_smooth[(int(36/dx_density) +1):(int(48/dx_density) +2)] = density_distribution_net_broad_high_all_smooth_n2m
density_distribution_net_NB_balanced_all_smooth_pn4 = analysis.sliding_win_on_lin_data(density_distribution_net_NB_balanced_all[(int(46/dx_density) +1):(int(54/dx_density) +2)], n_x_smooth)
density_distribution_net_NB_balanced_all_smooth = np.zeros(length_density)
density_distribution_net_NB_balanced_all_smooth[(int(46/dx_density) +1):(int(54/dx_density) +2)] = density_distribution_net_NB_balanced_all_smooth_pn4


## Define subfigure domain.
figsize = (max1, 1.6*max1)

# ## 3 rows (no sampled mean distribution)
width1_11=0.8; width1_22=0.185; width1_21=width1_22*(1+bar_width)/bar_width; width1_31 = width1_21; width1_32 = width1_22
x1_11=0.14; x1_21 = x1_11-0.18*xbuf0; x1_22 = x1_21 + width1_21 + 1.8*xbuf0; x1_31 = x1_21; x1_32 = x1_22
height1_11=0.22; height1_21=height1_11; height1_22 = height1_21; height1_31=height1_21; height1_32=height1_22
y1_11=0.725; y1_21=y1_11 - height1_21 - 1.5*ybuf0; y1_22 = y1_21; y1_31=y1_21 - height1_31 - 1.25*ybuf0; y1_32 = y1_31

rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_31 = [x1_31, y1_31, width1_31, height1_31]
rect1_32 = [x1_32, y1_32, width1_32, height1_32]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.5, 0.97, 'Narrow-Broad Trials', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(0.01, 0.945, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.5, 0.97 +y1_21 - y1_11, 'Monkey (current experiment)', fontsize=fontsize_fig_label-1, fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(0.01, 0.94 +y1_21 - y1_11, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.01 + x1_22 - x1_21, 0.94 +y1_21 - y1_11, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.5, 0.97 +y1_31 - y1_11, 'Human (Tsetsos et al., ' + r'$\bf{\it{PNAS}}$'+' 2012)', fontsize=fontsize_fig_label-1, fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(0.01, 0.94 + y1_31 - y1_11, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.01 + x1_32 - x1_31, 0.94 + y1_31 - y1_11, 'E', fontsize=fontsize_fig_label, fontweight='bold')





## rect1_11: schematics, Narrow Correct
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax)
dist_narrow = mlab.normpdf(x_schem, 54., sigma_narrow)
dist_broad = mlab.normpdf(x_schem, 54.-8., sigma_broad)
x_net_list_temp = 100./len(density_distribution_net_narrow_high_all_smooth)*np.arange(-int((len(density_distribution_net_narrow_high_all_smooth)-1.)/2.), len(density_distribution_net_narrow_high_all_smooth) - int((len(density_distribution_net_narrow_high_all_smooth)-1.)/2.))
ax.plot(x_net_list_temp, density_distribution_net_narrow_high_all_smooth, color=color_NBE[0], ls='-', clip_on=True, zorder=10, label='Narrow Correct')#, linestyle=linestyle_list[i_var_a])
ax.plot(x_net_list_temp, density_distribution_net_NB_balanced_all_smooth, color=color_NBE[2], ls='-', clip_on=True, zorder=11, label='Ambiguous')#, linestyle=linestyle_list[i_var_a])
ax.plot(x_net_list_temp, density_distribution_net_broad_high_all_smooth, color=color_NBE[1], ls='-', clip_on=True, zorder=10, label='Broad Correct')#, linestyle=linestyle_list[i_var_a])
ax.set_xlabel('Evidence Strength (Broad minus Narrow)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Probability Density', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([-15.,15.])
ax.set_ylim([0.,0.01])
ax.set_xticks([-15, 0, 15])
minorLocator = MultipleLocator(5)
ax.xaxis.set_minor_locator(minorLocator)
ax.set_yticks([0, 0.01])
ax.yaxis.set_ticklabels([0, 1])
minorLocator = MultipleLocator(0.002)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
legend = ax.legend(loc=(0.03,0.87), fontsize=fontsize_legend-2, frameon=False, ncol=3, markerscale=0., columnspacing=1.5, handletextpad=0.)
for color,text,item in zip([color_NBE[0],color_NBE[2],color_NBE[1]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)
ax.text(-16.5, 0.0105, r'$\times \mathregular{10^{-2}}$', fontsize=fontsize_tick-1.)



### Distribution of stimuli conditions

## rect1_21: Accuracy with narrow/Broad Correct mean (monkey A)
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
ax.bar([0, 1], ENB_bars_avg_non_drug[1:], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_avg_non_drug[1:], ecolor='k', color=color_NBE, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(7.,3.5))
ax.scatter([0.4,1.4, 0.9], [0.535,0.535,1.01], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.4,1.4], [0.96,0.96], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0,1+bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width/2. -0., 1+bar_width/2. +0.])
ax.xaxis.set_ticklabels(['Narrow Correct', 'Broad Correct'])
ax.set_yticks([0., 0.5, 1.])
ax.yaxis.set_ticklabels([0, 0.5, 1])
minorLocator = MultipleLocator(0.25)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")



## rect1_22: broad preference with equal mean (monkey A)
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
ax.bar([0], ENB_bars_avg_non_drug[0], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_avg_non_drug[0], ecolor='k', color=color_NBE[2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(7.5,3.75))
ax.scatter(0.4, 0.535, s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Broad Preference', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0,bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width/2.-0.])
ax.xaxis.set_ticklabels(['Ambiguous'])
ax.set_yticks([0., 0.5, 1.])
ax.yaxis.set_ticklabels([0, 0.5, 1])
minorLocator = MultipleLocator(0.25)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")


## rect1_31: Accuracy with narrow/Broad Correct mean (monkey A)
ax   = fig_temp.add_axes(rect1_31)
fig_funs.remove_topright_spines(ax)
ax.bar([0, 1], ENB_bars_avg_Tsetsos[1:], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_avg_Tsetsos[1:], ecolor='k', color=color_NBE, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(7.,3.5))
ax.scatter([0.4,1.4, 0.9], [0.535,0.535,1.01], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.4,1.4], [0.96,0.96], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0,1+bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width/2. -0., 1+bar_width/2. +0.])
ax.xaxis.set_ticklabels(['Narrow Correct', 'Broad Correct'])
ax.set_yticks([0., 0.5, 1.])
ax.yaxis.set_ticklabels([0, 0.5, 1])
minorLocator = MultipleLocator(0.25)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")



## rect1_32: broad preference with equal mean (monkey A)
ax   = fig_temp.add_axes(rect1_32)
fig_funs.remove_topright_spines(ax)
ax.bar([0], ENB_bars_avg_Tsetsos[0], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_avg_Tsetsos[0], ecolor='k', color=color_NBE[2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(7.5,3.75))
ax.scatter(0.4, 0.535, s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Broad Preference', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0,bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width/2.-0.])
ax.xaxis.set_ticklabels(['Ambiguous'])
ax.set_yticks([0., 0.5, 1.])
ax.yaxis.set_ticklabels([0, 0.5, 1])
minorLocator = MultipleLocator(0.25)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")



fig_temp.savefig(path_cwd+'Figure3.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
### Figure 4: Experimental non-drug data, Regression Trials

## Psychometric function generated from all (regression) data                                                           # See MainAnalysisNonDrugDays.m: lines 80-104
# # Log-Spaced # the two values right before and after the d_evidence=0 element encodes data with small but not exactly 0 evidence (narrow/broad for before/after). Their d_evidence value is wrong, and is chosen for the most suitable location on ax_0.
d_evidence_avg_list =  100*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_avg_list =  np.array([0.0481927710843374, 0.0308529945553539, 0.0710702341137124, 0.137184115523466, 0.213085764809903, 0.271460423634337, 0.331536388140162, 0.463553530751708, 0.433242506811989, 0.492222779729052, 0.605263157894737, 0.619222689075630, 0.666666666666667, 0.688524590163934, 0.754560530679934, 0.822963438101347, 0.865679012345679, 0.916152263374486, 0.948330683624801, 0.969827586206897, 0.973333333333333])  # Log-Spaced.
ErrBar_P_corr_avg_list = np.array([0.0235085803099071, 0.00736661021541921, 0.00742967339590677, 0.00843908711405435, 0.00860983085129568, 0.0104994999624684, 0.0141109711632133, 0.0168292934923764, 0.0182901023360976, 0.0111986020480162, 0.0792928727782124, 0.0111282145391145, 0.0157134840263677, 0.0153095019925757, 0.0123921194204745, 0.00966715532170431, 0.00757770804561883, 0.00628610123762718, 0.00624102523326547, 0.00794133110625836, 0.0186030662546279])

bar_pos_2by2 = [0., 0.8, 1.8, 2.6]
mean_var_color_list_2by2 = [color_mean_var_beta[0], color_mean_var_beta[0], color_mean_var_beta[1], color_mean_var_beta[1]]
mean_supp_choice_color_list_2by2 = [color_mean_supp_choice[0], color_mean_supp_choice[0], color_mean_supp_choice[1], color_mean_supp_choice[1]]


## Using L/R difference as regressors instead.
Reg_bars_mean_var_LRdiff_avg_nondrug = np.array([-0.0393515374644066, 20.6286415933539, 3.57390218117472])        # Bias, Mean, SD, averaged over left and right
Reg_bars_Err_mean_var_LRdiff_avg_nondrug = np.array([0.0159281352724043, 0.275866230528743, 0.181885413917953])        # Bias, Mean, SD, averaged over left and right

bar_pos_2by2_combined = np.array([0., 1.8, 3.6+0.5, 5.4+0.5, 7.2+0.5, 9.0+0.5, 10.8+0.5])
Reg_bar_pos_combined_model_control = np.array([0., 1., 2.+0.5, 3.+0.5, 4.+0.5, 5.0+0.5, 6.+0.5])
Reg_combined_color_list = [color_mean_var_beta[0], color_mean_var_beta[1], color_mean_var_beta[0], 'grey', 'grey', 'grey', 'grey']


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
psychometric_params_avg_non_drug = [0.070543470362281,1.055070714685022,0.011921638129677]                                            # Combine both monkeys pre analysis.


## Extract number distribution of stimuli, for Standard/Regression trials.                                              # See MainAnalysisNonDrugDays.m: lines 137-187
dx_Reg_density = 0.1
n_x_Reg_smooth = 20

n_distribution_Regression_monkey_avg = np.loadtxt('Data/Stim_Distribution/dx=2/n_distribution_regression_avg.txt', delimiter=',')              # axis is (narrow, broad)
density_distribution_Regression_all = n_distribution_Regression_monkey_avg / np.sum(n_distribution_Regression_monkey_avg)
density_distribution_net_Regression_all = np.zeros(len(density_distribution_Regression_all))
for i in range(len(density_distribution_net_Regression_all)):
    density_distribution_net_Regression_all[i] = np.sum(density_distribution_Regression_all.diagonal(i - int((len(density_distribution_Regression_all)-1.)/2.)))
density_distribution_net_Regression_all_smooth = analysis.sliding_win_on_lin_data(density_distribution_net_Regression_all, n_x_Reg_smooth)

n_SD_distribution_Regression_monkey_avg = np.loadtxt('Data/Stim_Distribution/dx=1/n_SD_distribution_regression_avg.txt', delimiter=',')              # axis is (narrow, broad)
density_SD_distribution_Regression_all = n_SD_distribution_Regression_monkey_avg / np.sum(n_SD_distribution_Regression_monkey_avg)
density_SD_distribution_net_Regression_all = np.zeros((2*len(density_SD_distribution_Regression_all)-1))
for i in range(2*len(density_SD_distribution_Regression_all)-1):
    density_SD_distribution_net_Regression_all[i] = np.sum(density_SD_distribution_Regression_all.diagonal(i - int((len(density_SD_distribution_Regression_all)-1.))))
density_SD_distribution_net_Regression_all_smooth_0_20 = analysis.sliding_win_on_lin_data(density_SD_distribution_net_Regression_all[(int(30/dx_Reg_density) +1):(int(50/dx_Reg_density) +2)], n_x_Reg_smooth)
density_SD_distribution_net_Regression_all_smooth = np.zeros(len(density_SD_distribution_net_Regression_all))
density_SD_distribution_net_Regression_all_smooth[(int(30/dx_Reg_density) +1):(int(50/dx_Reg_density) +2)] = density_SD_distribution_net_Regression_all_smooth_0_20








## Define subfigure domain.
figsize = (max1,1.2*max1)

width1_11=0.3; width1_12=0.25; width1_21=0.3; width1_22=0.25
x1_11=0.15; x1_12=x1_11 + width1_12 + 2.6*xbuf0; x1_21=x1_11; x1_22=x1_12-0.021
height1_11=0.3; height1_12=0.24; height1_21=height1_11; height1_22=0.28
y1_11=0.59; y1_12=y1_11+0.038; y1_21 = y1_11 - height1_21 - 2.4*ybuf0; y1_22 = y1_21+0.013


rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12_0 = [x1_12, y1_12, width1_12*0.05, height1_12]
rect1_12 = [x1_12+width1_12*0.2, y1_12, width1_12*(1-0.2), height1_12]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.025, 0.9, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.5, 0.965, 'Regular Trials', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(0.025 + x1_12 - x1_11, 0.9, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.025, 0.9 + y1_21 - y1_11, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.05 + x1_22 - x1_21, 0.9  + y1_21 - y1_11, 'D', fontsize=fontsize_fig_label, fontweight='bold')



### Distribution of stimuli conditions
## rect1_11: Stimuli Distribution for narrow-high trials.
ax = fig_temp.add_axes(rect1_11)
aspect_ratio=1.
plt.imshow(density_distribution_Regression_all, extent=(0.,100.,0.,100.), interpolation='nearest', cmap='BuPu', aspect=aspect_ratio, origin='lower', vmin=0., vmax=np.max(density_distribution_Regression_all))
ax.plot([0,100], [0,100], color='k', alpha=0.8, ls='--', lw=1.)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([25., 50., 75.])
ax.set_yticks([25., 50., 75.])
ax.set_xlim([25., 75.])
ax.set_ylim([25., 75.])
ax.tick_params(direction='out', pad=0.75)
ax.set_xlabel('Mean Evidence\n(Higher SD)', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylabel('Mean Evidence\n(Lower SD)', fontsize=fontsize_legend, labelpad=1.)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cb_temp = plt.colorbar(ticks=[0., 0.01, 0.02], cax=cax_scale_bar_size, orientation='horizontal')
cb_temp.set_ticklabels((0, 1, 2))
cb_temp.ax.xaxis.set_tick_params(pad=1.)
cax_scale_bar_size.xaxis.set_ticks_position("top")
ax.set_title("Trial Frequency", fontsize=fontsize_legend, x=0.49, y=1.2)
ax.text(76.5, 77.8, r'$\times \mathregular{10^{-2}}$', fontsize=fontsize_tick-1.)


### Distribution of stimuli conditions
## rect1_21: Stimuli Distribution for narrow-high trials.
ax = fig_temp.add_axes(rect1_21)
aspect_ratio=1.
plt.imshow(density_SD_distribution_Regression_all, extent=(0.,30.,0.,30.), interpolation='nearest', cmap='BuPu', aspect=aspect_ratio, origin='lower', vmin=0., vmax=np.max(density_SD_distribution_Regression_all))
ax.plot([0,30], [0,30], color='k', alpha=0.8, ls='--', lw=1.)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([12., 24.])
ax.set_yticks([12., 24.])
ax.set_xlim([8., 28.])
ax.set_ylim([8., 28.])
ax.tick_params(direction='out', pad=0.75)
ax.set_xlabel('Evidence SD\n(Higher SD)', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylabel('Evidence SD\n(Lower SD)', fontsize=fontsize_legend, labelpad=0.)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cb_temp = plt.colorbar(ticks=[0., 0.01, 0.02], cax=cax_scale_bar_size, orientation='horizontal')
cb_temp.set_ticklabels((0, 1, 2))
cb_temp.ax.xaxis.set_tick_params(pad=1.)
cax_scale_bar_size.xaxis.set_ticks_position("top")
ax.set_title("Trial Frequency", fontsize=fontsize_legend, x=0.49, y=1.2)
ax.text(28.5, 29., r'$\times \mathregular{10^{-3}}$', fontsize=fontsize_tick-1.)




## rect1_12: Psychometric function (over dx_broad, or dx_corr ?), Monkey A
ax_0   = fig_temp.add_axes(rect1_12_0)
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
ax.errorbar( d_evidence_avg_list[12:],    P_corr_avg_list[12:], ErrBar_P_corr_avg_list[12:], color=color_NB[1], ecolor=color_NB[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_avg_list[1:9], 1.-P_corr_avg_list[1:9], ErrBar_P_corr_avg_list[1:9], color=color_NB[0], ecolor=color_NB[0], fmt='.', zorder=3, clip_on=False, label='Lower SD Correct', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_avg_list[11], P_corr_avg_list[11], ErrBar_P_corr_avg_list[11], color=color_NB[1], ecolor=color_NB[1], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_avg_list[9], 1.-P_corr_avg_list[9], ErrBar_P_corr_avg_list[9], color=color_NB[0], ecolor=color_NB[0], marker='.', zorder=3, clip_on=False                      , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_avg_non_drug, x_list_psychometric), color=color_NB[1], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_avg_non_drug, -x_list_psychometric), color=color_NB[0], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_avg_non_drug, x0_psychometric), s=15., color=color_NB[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_avg_non_drug, -x0_psychometric), s=15., color=color_NB[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
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
ax_0.tick_params(direction='out', pad=2.7)
ax_0.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
## Add breakmark = wiggle
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
y_shift_spines = -0.1009
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_NB[1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=color_NB[0], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.35,-0.12), fontsize=fontsize_legend-2, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0., labelspacing=0.3)
for color,text,item in zip([color_NB[1], color_NB[0]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)


### Mean and Var only. L/R difference
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(Reg_bars_mean_var_LRdiff_avg_nondrug[1:])), Reg_bars_mean_var_LRdiff_avg_nondrug[1:], bar_width, yerr=Reg_bars_Err_mean_var_LRdiff_avg_nondrug[1:], ecolor='k', alpha=1, color=Reg_combined_color_list[0:2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.scatter([0.4,1.4], [22.5,5.], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0,len(Reg_bars_mean_var_LRdiff_avg_nondrug[1:])-1+bar_width])
ax.set_ylim([0.,23.5])
ax.set_xticks(np.arange(len(Reg_bars_mean_var_LRdiff_avg_nondrug[1:]))+bar_width/2.)
ax.xaxis.set_ticklabels(['Mean\nEvidence', 'Evidence\nSD'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 20.])
ax.set_yticklabels([0, 0.2])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))


fig_temp.savefig(path_cwd+'Figure4.pdf')    #Finally save fig


########################################################################################################################
### Figure 5: Model non-drug day data (NB, reg)

## Define Data. Alternatively can also import.                                                                          # See MainAnalysisNonDrugDays_NL.m: VarAndLocalWinsBetasCollapsed
ENB_bars_model_control = np.array([0.635873749037721, 0.627619288058571, 0.851181402439024])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is Correct, Accuracy when broad is correct].
ENB_bars_err_model_control = np.array([0.00384069051639553, 0.00385404023634202, 0.00283649567866252])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is Correct, Accuracy when broad is correct].

## Regression model: L/R differences of mean and SD.
Reg_bars_LRdiff_model_control = np.array([0.0105109456397199, 14.6886128805792, 5.22524834398918])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_control = np.array([0.00931913283747808, 0.106165115318636, 0.103654118038022])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.

## Psychometric function generated from model control data                                                              # See MainAnalysisNonDrugDays_NL.m: line 80-104
# Log-Spaced # the two values right between the d_evidence=-0.02 and 0.02 elements encode data with small but not exactly 0 evidence (narrow/broad for before/after). Their d_evidence value is wrong, and is chosen for the most suitable location on ax_0. d_evidence=0 encode exactly 0 net evidence, where there's none in all models.
d_evidence_model_control_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.0168, 0.0168, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_control = np.array([0.0272479564032698, 0.100597152044097, 0.166030187306783, 0.243478260869565, 0.330932703659976, 0.390492815899020, 0.436106857240088, 0.487804878048781, 0.520646319569120, 0.547033611549741, 0.620918017812286, 0.670053795576808, 0.689487632508834, 0.712544438801422, 0.762169491525424, 0.820525059665871, 0.876367752483964, 0.925233644859813, 0.964055299539171, 0.976878612716763])#, 1])                              # Log-Spaced.
ErrBar_P_corr_model_control = np.array([0.00849835779538010, 0.00644674894101349, 0.00501795583317643, 0.00481800599533429, 0.00511284985938039, 0.00565334203537729, 0.00646870013112883, 0.00734361057188574, 0.00864164082570305, 0.00528659609615176, 0.00518419384604154, 0.00812855663311837, 0.00687621555470198, 0.00588854298351436, 0.00495768067306082, 0.00419204392088500, 0.00369146027949865, 0.00359585294053457, 0.00399612017883492, 0.00807958762707065])       # Log-Spaced.

## Regression model using first/last/mean/Max/Min:                                                                   # See MainAnalysisNonDrugDays_NL.m: LongAvCOL, LongAvCOLSE
Reg_values_control = np.array([-0.563702490200418, 0.566544476994033, -1.79664446212739, 16.3607962599429, 2.35054221646502, -1.13961920728613, -0.544626694721213, 1.68487206917613, -15.2989874253047, -2.31071785001726, 1.25431512790789])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_control = np.array([0.0943938239603895, 0.0482762181928213, 0.0492635194160946, 0.229530157469093, 0.0994137270628964, 0.0987580748646614, 0.0353934238842000, 0.0364433340211602, 0.173398020098838, 0.0838030187014566, 0.0851095486830076])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_bar_pos_combined_model_control = np.array([0., 1., 2.+0.5, 3.+0.5, 4.+0.5, 5.0+0.5, 6.+0.5])

x_list_psychometric = np.arange(0.01, 0.5, 0.01)
x0_psychometric = 0.
## non-binned MLE (i.e. done using literal net evidence, via matlab). See Psychometric_function_fit_model_NL.m.
psychometric_params_model_control = [0.102757281562668, 1.17220614175807, 0.0259375390410745]


# PK                                                                                                                    # See MainAnalysisNonDrugDays_NL.m: lines 130-136.
t_PK_list = 0.125 + 0.25*np.arange(8)
PK_paired_model_control = np.array([2.75487907905315, 3.44357338089850, 3.11267511076330, 2.45350841044723, 1.86190163844012, 1.27762823633582, 0.845223913838725, 0.390593860452421])    # Paired ({(A-B)_PK}). Model Control
PK_paired_err_model_control = np.array([0.0303843679658410, 0.0318002722755726, 0.0310587499419505, 0.0297577409568170, 0.0289151237458699, 0.0282338609833591, 0.0278005645123162, 0.0276824703010797])    # Paired ({(A-B)_PK}). Model Control

## Define subfigure domain.
figsize = (max15,0.75*max15)

width1_11=0.25; width1_12=0.2; width1_14=0.05; width1_13=width1_14*(1+bar_width)/bar_width; width1_21=0.255; width1_22=width1_13; width1_23=0.255; width1_31=0.11; width1_32=0.26; width1_33=0.27;
x1_11=0.04; x1_12=x1_11 + width1_11 + xbuf0; x1_13 = x1_12 + width1_12 + xbuf0; x1_14 = x1_13 + width1_13 + 0.9*xbuf0; x1_21=0.11; x1_22=x1_21 + width1_21 + 1.*xbuf0; x1_23=x1_22 + width1_22 + 1.1*xbuf0; x1_31=0.09; x1_32=x1_31 + width1_31 + 1.05*xbuf0; x1_33=x1_32 + width1_32 + 0.95*xbuf0
height1_11=0.3; height1_12=0.27; height1_13 = height1_11; height1_14 = height1_13; height1_21=0.3; height1_22=0.328; height1_23=0.308; height1_31=height1_22; height1_32=height1_31; height1_33=height1_31
y1_11=0.59; y1_12=y1_11+0.04; y1_13=y1_12-0.0205; y1_14=y1_13; y1_21 = y1_11 - height1_21 - 1.98*ybuf0; y1_22=y1_21 - 0.351*ybuf0; y1_23=y1_22 + 0.244*ybuf0; y1_31 = y1_21 - height1_31 - 1.55*ybuf0; y1_32=y1_31; y1_33=y1_31+0.01

rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_14 = [x1_14, y1_14, width1_14, height1_14]
rect1_21_0 = [x1_21, y1_21, width1_21*0.05, height1_21]
rect1_21 = [x1_21+width1_21*0.2, y1_21, width1_21*(1-0.2), height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_31 = [x1_31, y1_31, width1_31, height1_31]
rect1_32 = [x1_32, y1_32, width1_32, height1_32]

rect1_33 = [x1_33, y1_33, width1_33, height1_33]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.015, 0.955, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.05+x1_12-x1_11, 0.955, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.67, 0.95, 'Narrow-Broad Trials', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(-0.04+x1_13-x1_11, 0.895, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.038+x1_14-x1_11, 0.895, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.397, 0.465, 'Regular Trials', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.015, 0.875 + y1_21 - y1_11, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.037 + x1_22 - x1_21, 0.875 + y1_21 - y1_11, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.029 + x1_23 - x1_21, 0.875 + y1_21 - y1_11, 'G', fontsize=fontsize_fig_label, fontweight='bold')

## rect1_11: schematics

## rect1_12: Firing Rate Trajectories
r_E_win_model_control  = np.loadtxt( 'Data/Model_Control/median_Set_18206/r_smooth_E2.txt')                                                                          #Reload everytime, just to make sure I don't mess anything up..
r_E_loss_model_control = np.loadtxt( 'Data/Model_Control/median_Set_18206/r_smooth_E1.txt')                                                                          #Reload everytime, just to make sure I don't mess anything up..
ax   = fig_temp.add_axes(rect1_12)
t_sim         = 5.          # Total Simulation time.
dt=0.001
tpts_0 = int(np.round((t_sim-0.)/dt))+1
t_vec_list = np.linspace(0.-1., t_sim-1., tpts_0)
ax = ax
fig_funs.remove_topright_spines(ax)
ax.plot(t_vec_list, r_E_win_model_control, color=colors_dark2['teal'], label='Winner', zorder=4)#, linestyle=linestyle_list[0])
ax.plot(t_vec_list, r_E_loss_model_control, color=colors_dark2['orange'], label='Loser', zorder=2)#, linestyle=linestyle_list[0])
ax.axhline(y=69, xmin=0.5/3., xmax=2.5/3., linewidth=1., color = 'k', clip_on=False)
ax.set_xlim([-0.5,2.5])
ax.set_ylim([0.,65.])
ax.set_xticks([0., 1., 2.])
ax.set_yticks([0., 60.])
minorLocator = MultipleLocator(20.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.set_xlabel('Time (s)', fontsize=fontsize_legend, labelpad=1.2)
ax.set_ylabel('Firing Rate (spikes/s)', fontsize=fontsize_legend, labelpad=2.)
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.text(1., 72., "Stimulus\nPresentation", fontsize=fontsize_legend, color='k', ha='center')#, fontweight='bold')


## rect1_13: Accuracy with narrow/Broad Correct mean
ax   = fig_temp.add_axes(rect1_13)
fig_funs.remove_topright_spines(ax)
ax.bar([0, 1], ENB_bars_model_control[1:], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_model_control[1:], ecolor='k', color=color_NBE, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(6.8,3.4))
ax.scatter([0.4,1.4, 0.9], [0.55,0.55,1.], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.4,1.4], [0.95,0.95], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0,1+bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width/2.-0.15, 1+bar_width/2.+0.07])
ax.xaxis.set_ticklabels(['Narrow\nCorrect', 'Broad\nCorrect'])
ax.set_yticks([0., 0.5, 1.])
ax.yaxis.set_ticklabels([0, 0.5, 1])
minorLocator = MultipleLocator(0.25)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")


## rect1_14: broad preference with equal mean
ax   = fig_temp.add_axes(rect1_14)
fig_funs.remove_topright_spines(ax)
ax.bar([0], ENB_bars_model_control[0], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_model_control[0], ecolor='k', color=color_NBE[2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(6.8,3.4))
ax.scatter([0.4], [0.55], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Broad Preference', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0,bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width/2.-0.15])
ax.xaxis.set_ticklabels(['Ambiguous'])
ax.set_yticks([0., 0.5, 1.])
ax.yaxis.set_ticklabels([0, 0.5, 1])
minorLocator = MultipleLocator(0.25)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")


## rect1_21: Psychometric function, Model
ax_0   = fig_temp.add_axes(rect1_21_0)
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_model_control_list[11:],  P_corr_model_control[11:], ErrBar_P_corr_model_control[11:], color=color_NB[1], ecolor=color_NB[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_model_control_list[:9], 1.-P_corr_model_control[:9], ErrBar_P_corr_model_control[:9] , color=color_NB[0], ecolor=color_NB[0], fmt='.', zorder=3, clip_on=False, label='Lower SD Correct', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_model_control_list[10],     P_corr_model_control[10], ErrBar_P_corr_model_control[10]  , color=color_NB[1], ecolor=color_NB[1], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_model_control_list[9], 1.-P_corr_model_control[9] , ErrBar_P_corr_model_control[9]  , color=color_NB[0], ecolor=color_NB[0], marker='.', zorder=3, clip_on=False                      , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_model_control, x_list_psychometric), color=color_NB[1], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_model_control, -x_list_psychometric), color=color_NB[0], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_model_control, x0_psychometric), s=15., color=color_NB[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_model_control, -x0_psychometric), s=15., color=color_NB[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.39, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=2.)
ax_0.set_ylim([0.4,1.])
ax.set_ylim([0.4,1.])
ax_0.set_xlim([0.,1])
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
y_shift_spines = -0.0946
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_NB[1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=color_NB[0], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.54,0.725), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0.)
for color,text,item in zip([color_NB[1], color_NB[0]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)


## rect1_22: Model Control, Model and perturbations
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(Reg_bars_LRdiff_model_control[1:])), Reg_bars_LRdiff_model_control[1:], bar_width, yerr=Reg_bars_err_LRdiff_model_control[1:], ecolor='k', alpha=1, color=Reg_combined_color_list[0:2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.scatter([0.4,1.4], [15.8,6.3], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(Reg_bars_LRdiff_model_control[1:])-1+bar_width])
ax.set_ylim([0.,16.5])
ax.set_xticks(np.arange(len(Reg_bars_LRdiff_model_control[1:]))+bar_width/2. + [-0.1,0.15])
ax.xaxis.set_ticklabels(['Mean\nEvidence', 'Evidence\nSD'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 15.])
ax.set_yticklabels([0, 0.15])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))


## rect1_23: Psychophysical Kernel, Model Control
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
ax.errorbar( i_PK_list, PK_paired_model_control, PK_paired_err_model_control, color='k', markerfacecolor='grey', ecolor='grey', fmt='.', zorder=4, clip_on=False, label=label_list[0] , markeredgecolor='k', linewidth=1., ls='-', elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.set_xlabel('Sample Number', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylim([0.,4.05])
ax.set_xlim([1.,8.])
ax.set_xticks([1., 8.])
ax.set_yticks([0., 4.])
ax.text(0.1, 4.2, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

fig_temp.savefig(path_cwd+'Figure5.pdf')    #Finally save fig

########################################################################################################################
### Figure 6: Mean-Field Model
execfile('MFAMPA_functions_WongWang2006_Hunt_Noisy_Input.py')

Reg_bars_LRdiff_modelMF_control = np.array([-0.00105387664058295, 19.3627664172908, 3.63209828716278])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_modelMF_control = np.array([0.0106846338483828, 0.134907614298566, 0.120139261669664])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.


## Regression model using first/last/mean/Max/Min:                                                                   # See MainAnalysisNonDrugDays_NL.m: LongAvCOL, LongAvCOLSE
Reg_values_modelMF_control = np.array([-0.824720688164682, 5.10483829134541, -2.66591540321193, 24.1873730068665, 2.08891331546669, -1.01219322534966, -5.01144047063643, 2.51430404713043, -22.9666587623899, -1.89846228936625, 1.26737380871806])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_modelMF_control = np.array([0.121747922543159, 0.0697668932702089, 0.0658972219338625, 0.315322166453290, 0.131146427525639, 0.130202504082441, 0.0529733080300937, 0.0491463731005067, 0.244017265520443, 0.108318615358950, 0.108897678051422])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_bars_LRsep_modelMF_control = np.array([Reg_values_modelMF_control[3], -Reg_values_modelMF_control[8], Reg_values_modelMF_control[4], -Reg_values_modelMF_control[9], Reg_values_modelMF_control[5], -Reg_values_modelMF_control[10], Reg_values_modelMF_control[1], -Reg_values_modelMF_control[6], Reg_values_modelMF_control[2], -Reg_values_modelMF_control[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_err_LRsep_modelMF_control = np.array([Reg_values_err_modelMF_control[3], Reg_values_err_modelMF_control[8], Reg_values_err_modelMF_control[4], Reg_values_err_modelMF_control[9], Reg_values_err_modelMF_control[5], Reg_values_err_modelMF_control[10], Reg_values_err_modelMF_control[1], Reg_values_err_modelMF_control[6], Reg_values_err_modelMF_control[2], Reg_values_err_modelMF_control[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_combined_modelMF_control = np.array([Reg_bars_LRdiff_modelMF_control[1], Reg_bars_LRdiff_modelMF_control[2], 0.5*(Reg_values_modelMF_control[3]-Reg_values_modelMF_control[8]), 0.5*(Reg_values_modelMF_control[4]-Reg_values_modelMF_control[9]), 0.5*(Reg_values_modelMF_control[5]-Reg_values_modelMF_control[10]), 0.5*(Reg_values_modelMF_control[1]-Reg_values_modelMF_control[6]), 0.5*(Reg_values_modelMF_control[2]-Reg_values_modelMF_control[7])])        # Mean, SD, Mean, Max, Min, First, Last), averaged over left and right
Reg_bars_err_combined_modelMF_control = np.array([Reg_bars_err_LRdiff_modelMF_control[1], Reg_bars_err_LRdiff_modelMF_control[2], 0.5*(Reg_values_err_modelMF_control[3]**2+Reg_values_err_modelMF_control[8]**2)**0.5, 0.5*(Reg_values_err_modelMF_control[4]**2+Reg_values_err_modelMF_control[9]**2)**0.5, 0.5*(Reg_values_err_modelMF_control[5]**2+Reg_values_err_modelMF_control[10]**2)**0.5, 0.5*(Reg_values_err_control[1]**2+Reg_values_err_modelMF_control[6]**2)**0.5, 0.5*(Reg_values_err_modelMF_control[2]**2+Reg_values_err_modelMF_control[7]**2)**0.5])        # Mean, SD, Mean, Max, Min, First, Last), averaged over left and right
Reg_bar_pos_combined_modelMF_control = np.array([0., 1., 2.+0.5, 3.+0.5, 4.+0.5, 5.0+0.5, 6.+0.5])

## Psychometric function generated from model control data                                                              # See MainAnalysisNonDrugDays_NL.m: line 80-104
# Log-Spaced # the two values right between the d_evidence=-0.02 and 0.02 elements encode data with small but not exactly 0 evidence (narrow/broad for before/after). Their d_evidence value is wrong, and is chosen for the most suitable location on ax_0. d_evidence=0 encode exactly 0 net evidence, where there's none in all models.
d_evidence_modelMF_control_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.017, 0.017, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_modelMF_control = np.array([0.00315457413249211, 0.0399369416710457, 0.0916470839564196, 0.153868360277136, 0.233310486634681, 0.295009185548071, 0.361904761904762, 0.414446721311475, 0.462533156498674, 0.521306252489048, 0.610743582958455, 0.660639777468707, 0.697329376854599, 0.742445190598459, 0.800773395204950, 0.855595172792101, 0.910848855395990, 0.948663101604278, 0.973340303188709, 0.993670886075949])                              # Log-Spaced.
ErrBar_P_corr_modelMF_control = np.array([0.00314959453328271, 0.00448867216490562, 0.00421713313895920, 0.00433500587993096, 0.00495181308603866, 0.00564269284526271, 0.00663224023367842, 0.00788429123544134, 0.00907886598492878, 0.00575561002918264, 0.00560846356556093, 0.00882913966339246, 0.00722434608238319, 0.00614558270288382, 0.00496757256831573, 0.00411624947599421, 0.00339794473765873, 0.00322760225431342, 0.00368300638131680, 0.00446117436466950])       # Log-Spaced.
## non-binned MLE (i.e. done using literal net evidence, via matlab). See Psychometric_function_fit_model_NL.m.
psychometric_params_modelMF_control = [0.0758377793514703, 1.09105880486443, 0.0137420202192980]



## Define subfigure domain.
figsize = (max15,0.6*max15)

width1_11 = 0.17; width1_12 = 0.15; width1_13 = 0.1; width1_15 = 0.25; width1_21 = 0.15; width1_22 = width1_21; width1_23 = width1_21; width1_24 = width1_21
x1_11 = 0.03; x1_12 = x1_11 + width1_11 + 1.*xbuf0; x1_13 = x1_12 + width1_12 + 0.8*xbuf0; x1_14 = x1_13 + width1_13 + 1.1*xbuf0; x1_21 = 0.065; x1_22 = x1_21 + width1_21 + 0.75*xbuf0; x1_23 = x1_22 + width1_22 + 0.7*xbuf0; x1_24 = x1_23 + width1_23 + 1.4*xbuf0
height1_11 = 0.28; height1_12 = height1_11; height1_13 = 0.315; height1_14 = height1_11; height1_21= 0.38; height1_22 = height1_21; height1_23 = height1_21; height1_24 = height1_21*0.6
y1_11 = 0.68; y1_12=y1_11; y1_13=y1_12 - 0.44*ybuf0; y1_14=y1_12 - 0.13*ybuf0; y1_21 = y1_11 - height1_21 - 2.5*ybuf0; y1_22=y1_21; y1_23=y1_21; y1_24=y1_21+1.133*ybuf0


rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12_0 = [x1_12, y1_12, width1_12*0.05, height1_12]
rect1_12 = [x1_12+width1_12*0.2, y1_12, width1_12*(1-0.2), height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_14 = [x1_14, y1_14, width1_14, height1_14]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_24 = [x1_24, y1_24, width1_24, height1_24]





##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.002, 0.96, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.067+x1_12-x1_11, 0.96, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.035+x1_13-x1_11, 0.96, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.085+x1_14-x1_11, 0.96, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.002, 1.03 + y1_21 - y1_11, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.007+x1_22-x1_21, 1.03 + y1_21 - y1_11, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.0035+x1_23-x1_21, 1.03 + y1_21 - y1_11, 'G', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.035+x1_24-x1_21, 1.03 + y1_21 - y1_11, 'H', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.

## rect1_11: Mean-Field model schematics

## rect1_12: Psychometric function (over dx_broad, or dx_corr ?), Model
ax_0   = fig_temp.add_axes(rect1_12_0)
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_modelMF_control_list[11:],  P_corr_modelMF_control[11:], ErrBar_P_corr_modelMF_control[11:], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_modelMF_control_list[:9], 1.-P_corr_modelMF_control[:9], ErrBar_P_corr_modelMF_control[:9] , color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], fmt='.', zorder=3, clip_on=False, label='Lower SD Correct', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_modelMF_control_list[10],     P_corr_modelMF_control[10], ErrBar_P_corr_modelMF_control[10]  , color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_modelMF_control_list[9], 1.-P_corr_modelMF_control[9] , ErrBar_P_corr_modelMF_control[9]  , color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], marker='.', zorder=3, clip_on=False                      , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_modelMF_control, x_list_psychometric), color=color_NB[1], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_modelMF_control, -x_list_psychometric), color=color_NB[0], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_modelMF_control, x0_psychometric), s=15., color=color_NB[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_modelMF_control, -x0_psychometric), s=15., color=color_NB[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.39, labelpad=1.)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=2.)
ax_0.set_ylim([0.4,1.])
ax.set_ylim([0.4,1.])
ax_0.set_xlim([0.,1])
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
y_shift_spines = -0.1267
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_NB[1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=color_NB[0], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.41,-0.16), fontsize=fontsize_legend-2.2, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0., labelspacing=0.15)
for color,text,item in zip([color_NB[1], color_NB[0]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

## rect1_13: ModelMF Control, Model and perturbations
### Mean and Var only
ax   = fig_temp.add_axes(rect1_13)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(Reg_bars_LRdiff_modelMF_control[1:])), Reg_bars_LRdiff_modelMF_control[1:], bar_width, yerr=Reg_bars_err_LRdiff_modelMF_control[1:], ecolor='k', alpha=1, color=Reg_combined_color_list[0:2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.scatter([0.4,1.4], [21.,5.5], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-2.)
ax.set_xlim([0,len(Reg_bars_LRdiff_modelMF_control[1:])-1+bar_width])
ax.set_ylim([0.,22.])
ax.set_xticks(np.arange(len(Reg_bars_LRdiff_modelMF_control[1:]))+bar_width/2.+[-0.25,0.15])
ax.xaxis.set_ticklabels(['Mean\nEvidence', 'Evidence\nSD'])
ax.set_yticks([0., 20.])
ax.set_yticklabels([0, 0.2])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))

## rect1_14: FI curve
ax   = fig_temp.add_axes(rect1_14)
fig_funs.remove_topright_spines(ax)
fluc_Input1_0 = 0.
fluc_Input2_0 = 0.
pars = default_pars()
pars['mu0'] = 30.
pars['mu0_slope'] = 50.
pars['coh'] = 0.
I0_list = np.arange(0., 0.6, 0.01)
F_list = F(I0_list,0,pars)
ax.plot(I0_list, F_list, color='k', linestyle='-', zorder=(3-0), clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.set_xlabel('Input Current (nA)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Firing Rate (sp/s)', fontsize=fontsize_legend, labelpad=1.)
ax.set_xlim([0.,0.55])
ax.set_xticks([0.,0.5])
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
ax.set_yticks([0., 100.])
minorLocator = MultipleLocator(20.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_14 Right axis: FI'' curve
ax_twin = ax.twinx()
ax_twin.spines['top'].set_visible(False)
ax_twin.spines['right'].set_color('grey')
Fpp_list = Fpp_x20(I0_list,pars)
ax_twin.plot(I0_list, Fpp_list, color='grey', linestyle='-', zorder=0, clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax_twin.tick_params(axis='y', labelcolor='grey')
ax_twin.set_ylabel(r'$\mathbf{\frac{d^2F}{dI^2} (\frac{sp}{s \ nA^2})}$', fontsize=fontsize_legend, color='grey', labelpad=-10.)
ax_twin.set_ylim([0.,5400.])
ax_twin.set_yticks([0., 5000.])
minorLocator = MultipleLocator(1000.)
ax_twin.yaxis.set_minor_locator(minorLocator)
ax_twin.tick_params(direction='out', pad=1.5)
ax_twin.tick_params(which='minor',direction='out', colors='grey')
ax_twin.tick_params(axis='y', colors='grey')
ax_twin.spines['left'].set_position(('outward',5))
ax_twin.spines['right'].set_position(('outward',5))
ax_twin.spines['bottom'].set_position(('outward',5))
ax.set_zorder(ax_twin.get_zorder()+1) # put ax in front of ax2
ax.patch.set_visible(False) # hide the 'canvas'


## rect1_21: Vector Field: 0% coherence
ax   = fig_temp.add_axes(rect1_21)
fluc_Input1_0 = 40.
fluc_Input2_0 = 0.
pars = default_pars()
pars['mu0'] = 30.
pars['mu0_slope'] = 50.
pars['coh'] = 0.
pars['fluc_Input1'] = fluc_Input1_0
pars['fluc_Input2'] = fluc_Input2_0
plot_phase_plane(pars, fluc_Input1_0, fluc_Input2_0,expt='DM', ax=ax)
ax.set_xticks([0., 0.8])
ax.xaxis.set_ticklabels([0, 0.8])
minorLocator = MultipleLocator(0.2)
ax.xaxis.set_minor_locator(minorLocator)
ax.set_yticks([0., 0.8])
ax.yaxis.set_ticklabels([0, 0.8])
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(which='minor',direction='out')
ax.set_title(r'$I_B=I_N+\Delta$', fontsize=fontsize_legend-1.)
ax.xaxis.labelpad = 0.
ax.yaxis.labelpad = -3.
ax.plot([0,0.8], [0,0.8], color='k', lw=1., ls='--', dashes=(3.5,1.5))
ax.text(0.46, 0.63, r"$S_B=S_N$", fontsize=fontsize_legend-1.5, color='k', rotation=45)

## rect1_22: Vector Field: Positive Fluctuation
ax   = fig_temp.add_axes(rect1_22)
fluc_Input1_0 = -40.
fluc_Input2_0 = 0.
pars = default_pars()
pars['mu0'] = 30.
pars['mu0_slope'] = 50.
pars['coh'] = 0.
pars['fluc_Input1'] = fluc_Input1_0
pars['fluc_Input2'] = fluc_Input2_0
plot_phase_plane(pars, fluc_Input1_0, fluc_Input2_0,expt='DM', ax=ax)
ax.set_xticks([0., 0.8])
ax.xaxis.set_ticklabels([0, 0.8])
minorLocator = MultipleLocator(0.2)
ax.xaxis.set_minor_locator(minorLocator)
ax.set_yticks([0., 0.8])
ax.yaxis.set_ticklabels([0, 0.8])
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(which='minor',direction='out')
ax.set_title(r'$I_B=I_N-\Delta$', fontsize=fontsize_legend-1.)
ax.xaxis.labelpad = 0.
ax.yaxis.labelpad = -3.
ax.plot([0,0.8], [0,0.8], color='k', lw=1., ls='--', dashes=(3.5,1.5))
ax.text(0.46, 0.63, r"$S_B=S_N$", fontsize=fontsize_legend-1.5, color='k', rotation=45)

## rect1_23: Vector Field: Positive Fluctuation - Negative Fluctuation
ax   = fig_temp.add_axes(rect1_23)
fluc_Input1_2Ddiff_list = [-40., 0, 40.]
fluc_Input2_0 = 0.
# Make 2D grid of (S1,S2) values
S_vec = np.linspace(0.001,0.999,200) # things break down at S=0 or S=1
S1_2Ddiff, S2_2Ddiff = np.meshgrid(S_vec,S_vec)
dS1dt_2Ddiff_each = np.zeros((len(S1_2Ddiff), len(S1_2Ddiff[0]), len(fluc_Input1_2Ddiff_list)))
dS2dt_2Ddiff_each = np.zeros((len(S1_2Ddiff), len(S1_2Ddiff[0]), len(fluc_Input1_2Ddiff_list)))

for i_fluc_2Ddiff in range(len(fluc_Input1_2Ddiff_list)):
    fluc_Input1_2Ddiff_temp = fluc_Input1_2Ddiff_list[i_fluc_2Ddiff]
    I1_2Ddiff, I2_2Ddiff = currents_DM(S1_2Ddiff,S2_2Ddiff,pars, fluc_Input1_2Ddiff_temp, fluc_Input2_0)
    dS1dt_2Ddiff_temp, dS2dt_2Ddiff_temp = Sderivs(S1_2Ddiff, S2_2Ddiff, I1_2Ddiff, I2_2Ddiff, pars)
    dS1dt_2Ddiff_each[:,:,i_fluc_2Ddiff] = dS1dt_2Ddiff_temp
    dS2dt_2Ddiff_each[:,:,i_fluc_2Ddiff] = dS2dt_2Ddiff_temp
dS1dt_2Ddiff = dS1dt_2Ddiff_each[:,:,2] + dS1dt_2Ddiff_each[:,:,0] - 2.*dS1dt_2Ddiff_each[:,:,1]
dS2dt_2Ddiff = dS2dt_2Ddiff_each[:,:,2] + dS2dt_2Ddiff_each[:,:,0] - 2.*dS2dt_2Ddiff_each[:,:,1]

n_skip=20
scale=0.6
facecolor='gray'
ax.quiver(S2_2Ddiff[::n_skip,::n_skip], S1_2Ddiff[::n_skip,::n_skip],
          dS2dt_2Ddiff[::n_skip,::n_skip], dS1dt_2Ddiff[::n_skip,::n_skip],
          angles='xy', scale_units='xy', scale=scale,facecolor=facecolor, width=0.01, headwidth=5., alpha=0.6)
ax.set_xlabel('$S_N$', labelpad=-2., fontsize=8.)
ax.set_ylabel('$S_B$', labelpad=-6., fontsize=8.)
ax.set_xlim(0,0.8)
ax.set_ylim(0,0.8)
ax.set_xticks([0., 0.8])
ax.xaxis.set_ticklabels([0, 0.8])
minorLocator = MultipleLocator(0.2)
ax.xaxis.set_minor_locator(minorLocator)
ax.set_yticks([0., 0.8])
ax.yaxis.set_ticklabels([0, 0.8])
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(which='minor',direction='out')
ax.set_aspect('equal')
ax.set_title(r'(E) $+$ (F)'+'\n'+r'$-$ $2$($I_B=I_N$)', fontsize=fontsize_legend-1., y=0.95)
ax.xaxis.labelpad = 0.
ax.yaxis.labelpad = -3.
ax.plot([0,0.8], [0,0.8], color='k', lw=1., ls='--', dashes=(3.5,1.5))
ax.text(0.46, 0.63, r"$S_B=S_N$", fontsize=fontsize_legend-1.5, color='k', rotation=45)

## rect1_24: Absolute dot product with (1,-1). I.e. Absolute driving force to choice
ax   = fig_temp.add_axes(rect1_24)
fig_funs.remove_topright_spines(ax)
pars = default_pars()
pars['mu0'] = 30.
pars['mu0_slope'] = 50.
pars['coh'] = 0.
fluc_Input1_list = [-40., 0, 40.]
fluc_Input2_0 = 0.
S12_list = np.arange(0., 1.00001, 0.01)
dS1dt_list = np.zeros((len(S12_list), len(fluc_Input1_list)))
dS2dt_list = np.zeros((len(S12_list), len(fluc_Input1_list)))
negdiag_dot_prod_list = np.zeros((len(S12_list), len(fluc_Input1_list)))
for i_temp in range(len(fluc_Input1_list)):
    I1_temp, I2_temp = currents_DM(S12_list,S12_list, pars, fluc_Input1_list[i_temp], fluc_Input2_0)
    dS1_temp, dS2_temp = Sderivs(S12_list, S12_list, I1_temp, I2_temp, pars)
    dS1dt_list[:,i_temp] = dS1_temp
    dS2dt_list[:,i_temp] = dS2_temp
for i_temp in range(len(fluc_Input1_list)):
    for j_temp in range(len(S12_list)):
        negdiag_dot_prod_list[j_temp, i_temp] = np.dot([dS1dt_list[j_temp,i_temp], dS2dt_list[j_temp,i_temp]], [1,-1]) #/2**0.5 /(dS1dt_list[j_temp,i_temp]**2 + dS2dt_list[j_temp,i_temp]**2)**0.5
ax.plot(S12_list, np.abs(negdiag_dot_prod_list[:,2]-negdiag_dot_prod_list[:,1]) , color=colors_div4['red4'], label=r'$I_B=I_N+\Delta$', lw=1., zorder=2)        # Positive/negative driving force to option 1. Using dot product with   (S1,S2)=(1,-1)
ax.plot(S12_list, np.abs(negdiag_dot_prod_list[:,0]-negdiag_dot_prod_list[:,1]) , color=colors_div4['blue4'], label=r'$I_B=I_N-\Delta$', lw=1., zorder=1)        # Positive/negative driving force to option 1. Using dot product with   (S1,S2)=(1,-1)
ax.set_xlabel(r"$S_B=S_N$", fontsize=fontsize_legend, labelpad=-1.5)
ax.set_ylabel('Driving force to\noptions (1/s)', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0.,0.8])
ax.set_ylim([0.15,0.53])
ax.set_xticks([0., 0.8])
ax.xaxis.set_ticklabels([0, 0.8])
minorLocator = MultipleLocator(0.2)
ax.xaxis.set_minor_locator(minorLocator)
ax.set_yticks([0.2, 0.5])
ax.yaxis.set_ticklabels([0.2, 0.5])
minorLocator = MultipleLocator(0.1)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(which='minor',direction='out')
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
legend = ax.legend(loc=(-0.12,-0.13), fontsize=fontsize_legend-2.2, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip([colors_div4['red4'], colors_div4['blue4']], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

fig_temp.savefig(path_cwd+'Figure6.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
### Figure 5: Model E/I Perturbsation & Upstream Sensory Deficit

## Probability Correct, using regression trials                                                                         # See MainAnalysisNonDrugDays_NL.m: line 80-104
d_evidence_model_control_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_control = np.array([0.0137931034482759, 0.0847457627118644, 0.178436911487759, 0.240250696378830, 0.318552331268340, 0.386414253897550, 0.445526193787668, 0.473273273273273, 0.489448051948052, 0.544291952588896, 0.608038585209003, 0.646166807076664, 0.697041420118343, 0.717213114754098, 0.776039815143974, 0.818037974683544, 0.879048931771192, 0.929117797042325, 0.962785114045618, 0.993506493506494])#, 1])                              # Log-Spaced.
ErrBar_P_corr_model_control = np.array([0.00968569999803871, 0.00969035855516513, 0.00830779080125371, 0.00797214639414235, 0.00841297814930442, 0.00938135141509752, 0.0107016825967615, 0.0122360586128648, 0.0142418995087430, 0.00879584457366940, 0.00875400990536126, 0.0138786188464826, 0.0111783465332580, 0.00961031395087781, 0.00786036194383691, 0.00686330931588882, 0.00605288852117157, 0.00579515686333411, 0.00655843968591512, 0.00647238934329532])       # Log-Spaced.
P_corr_model_reduced_gEE = np.array([0.0413793103448276, 0.164648910411622, 0.259887005649718, 0.348189415041783, 0.419954352787740, 0.448775055679287, 0.475660639777469, 0.495495495495496, 0.504870129870130, 0.538677479725515, 0.574919614147910, 0.620893007582140, 0.628994082840237, 0.658014571948998, 0.692143618912193, 0.740506329113924, 0.812198483804273, 0.886282508924018, 0.953181272509004, 0.987012987012987])
ErrBar_P_corr_model_reduced_gEE = np.array([0.0165398292573879, 0.0129039873481690, 0.00951621052129289, 0.00888947819199700, 0.00891200130820058, 0.00958252564452817, 0.0107530029561445, 0.0122530797609062, 0.0142443963074892, 0.00880410024284789, 0.00886459505399765, 0.0140819864594248, 0.0117508787684031, 0.0101229102962096, 0.00870336753029191, 0.00779801795768645, 0.00724989856414016, 0.00716903873750661, 0.00731939276858023, 0.00912337887421051])
d_evidence_model_reduced_gEI_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_reduced_gEI = np.array([0.144827586206897, 0.158595641646489, 0.213747645951036, 0.272284122562674, 0.347244864688621, 0.396807720861173, 0.426981919332406, 0.452252252252252, 0.476461038961039, 0.507174048658765, 0.563987138263666, 0.582982308340354, 0.610650887573965, 0.627049180327869, 0.663348738002133, 0.727848101265823, 0.768435561681599, 0.824579296277409, 0.885954381752701, 0.935064935064935])
ErrBar_P_corr_model_reduced_gEI = np.array([0.0292259398810998, 0.0127103637384885, 0.00889517288532411, 0.00830615184564554, 0.00859678719789298, 0.00942582013402603, 0.0106503488398540, 0.0121975764979538, 0.0142292774054587, 0.00882965098063060, 0.00889209421867734, 0.0143113159020974, 0.0118610391264521, 0.0103195418669318, 0.00890996936801899, 0.00791739877748742, 0.00783052636048486, 0.00858851029820107, 0.0110134270995753, 0.0198563840819854])
d_evidence_model_upstream_deficit_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_upstream_deficit = np.array([0.103448275862069, 0.163438256658596, 0.250470809792844, 0.316155988857939, 0.398761004238670, 0.427988121752042, 0.470097357440890, 0.458858858858859, 0.486201298701299, 0.517779164067374, 0.582636655948553, 0.642796967144061, 0.607692307692308, 0.646174863387978, 0.681478848204764, 0.724367088607595, 0.797381116471399, 0.862825089240184, 0.932773109243698, 0.967532467532468])#, 1])                              # Log-Spaced.
ErrBar_P_corr_model_upstream_deficit = np.array([0.0252909592792303, 0.0128657715895482, 0.00940146598824545, 0.00867634899498222, 0.00884144038373232, 0.00953278034133173, 0.0107464958433210, 0.0122120259367905, 0.0142396463822987, 0.00882497556941132, 0.00884251610272178, 0.0139081420873604, 0.0118771414603255, 0.0102035947921272, 0.00878436682302336, 0.00794879597501741, 0.00746146709674289, 0.00776890711130964, 0.00867634915805814, 0.0142822639541446])       # Log-Spaced.

## non-binned MLE (i.e. done using literal net evidence, via matlab). See Psychometric_function_fit_model_NL.m.
x_list_psychometric = np.arange(0.01, 0.5, 0.01)                                                                        # See figure_psychometric_function_fit.py, esp lines 633-687
x0_psychometric = 0.
psychometric_params_model_control = [0.101086226880943, 1.21003801562976, 0.0249992602771917]
psychometric_params_model_reduced_gEE = [0.140804363508584, 1.51912611575380, 0.0336836858081337]
psychometric_params_model_reduced_gEI = [0.152156466017638, 1.07458939999049, 0.0133138564266258]
psychometric_params_model_upstream_deficit = [0.145996442400798, 1.31352896831548, 0.0263962064412712]

##  Mean & Variance, LR differences, Constrained across-trials
Reg_bars_LRdiff_model_control = np.array([-0.0234540455791373, 14.9203952955068, 5.19551385674560])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_control = np.array([0.0155158561968261, 0.175706516543730, 0.178905605928961])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_lowered_EI = np.array([-0.0402458902571528, 10.1662968567125, 4.40546748298841])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_lowered_EI = np.array([0.0146114202415381, 0.146766745583410, 0.167399601883599])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_elevated_EI = np.array([-0.00230531310095626, 10.0719048208088, 1.87343690160535])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_elevated_EI = np.array([0.0145697421075110, 0.144813449154007, 0.163959952570434])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_upstream_deficit = np.array([-0.0147899377590499, 10.0012805935718, 3.56242118133917])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model upstream_deficit regression Beta values.
Reg_bars_err_LRdiff_model_upstream_deficit = np.array([0.0145717046349854, 0.145214668563661, 0.165602653567603])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model upstream_deficit regression Beta values.

Reg_mean_models = np.array([Reg_bars_LRdiff_model_control[1], Reg_bars_LRdiff_model_lowered_EI[1], Reg_bars_LRdiff_model_elevated_EI[1], Reg_bars_LRdiff_model_upstream_deficit[1]])
Reg_std_models = np.array([Reg_bars_LRdiff_model_control[2], Reg_bars_LRdiff_model_lowered_EI[2], Reg_bars_LRdiff_model_elevated_EI[2], Reg_bars_LRdiff_model_upstream_deficit[2]])
Reg_ratio_models = Reg_std_models / Reg_mean_models
Reg_err_mean_models = np.array([Reg_bars_err_LRdiff_model_control[1], Reg_bars_err_LRdiff_model_lowered_EI[1], Reg_bars_err_LRdiff_model_elevated_EI[1], Reg_bars_err_LRdiff_model_upstream_deficit[1]])
Reg_err_std_models = np.array([Reg_bars_err_LRdiff_model_control[2], Reg_bars_err_LRdiff_model_lowered_EI[2], Reg_bars_err_LRdiff_model_elevated_EI[2], Reg_bars_err_LRdiff_model_upstream_deficit[2]])
Reg_err_ratio_models = Reg_ratio_models *( (Reg_err_mean_models/Reg_mean_models)**2 + (Reg_err_std_models/Reg_std_models)**2)**0.5


## First, Last, Mean, Max, Min                                                                                          # See MainAnalysisNonDrugDays_NL.m: LongAvCOL, LongAvCOLSE
Reg_values_control = np.array([-0.799975795441590, 0.496091788783012, -1.95147482506435, 16.6435510742991, 2.51869160833734, -0.775652462213961, -0.563937819902963, 1.72726445867963, -15.5439816070891, -2.21963076055435, 1.23313129184529])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_lowered_EI = np.array([0.190003506885178, -0.235599553981222, -1.07015666540429, 10.7143801356569, 1.73536901552416, -1.05480870621277, 0.306620167816446, 0.963593249348418, -11.2866203206244, -1.66782027032213, 1.00893647865141])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_elevated_EI = np.array([-0.317237265125046, 3.57025700473867, -1.38041746171278, 9.42971561503303, 1.13803974828794, -0.250209186493841, -3.37711415320973, 1.32572992661215, -10.3266097437763, -0.449125721821575, 0.906379587234502])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_upstream_deficit = np.array([0.0560311909515569, -0.160674814456601, -0.881991504397344, 10.7179400646621, 1.29444976079686, -0.873468733846600, 0.220992880834769, 0.986593266733566, -11.0766687429408, -1.30012991374204, 0.875693989956842])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)

Reg_values_err_control = np.array([0.155259207050030, 0.0814567425789707, 0.0833418952000110, 0.380405186170427, 0.167947586495689, 0.165399951383634, 0.0599810684092175, 0.0618760698960268, 0.290581409807592, 0.139654535033320, 0.142078616907945])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_lowered_EI = np.array([0.143136035928441, 0.0746855406809858, 0.0753280128921357, 0.336090493254340, 0.153561120959519, 0.151917942379994, 0.0553331500596443, 0.0560195709313174, 0.256989632519849, 0.129163640092481, 0.131170489417240])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_elevated_EI = np.array([0.153826504023430, 0.0842572769780524, 0.0817096933761490, 0.359559773477624, 0.165851584561131, 0.164103971568001, 0.0627951316515934, 0.0602985276382010, 0.273416473281369, 0.138401242084907, 0.139809194108627])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_upstream_deficit = np.array([0.142186429790058, 0.0742174710133451, 0.0747323951036313, 0.334075001924850, 0.152391678290145, 0.150983011904768, 0.0547667934757646, 0.0555336023725431, 0.254172384516366, 0.128201539812090, 0.129835392722852])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)



# PK                                                                                                                    # See MainAnalysisNonDrugDays_NL.m: lines 130-136.
t_PK_list = 0.125 + 0.25*np.arange(8)
PK_paired_model_control = np.array([2.77143624709664, 3.37910108153866, 3.20459845348331, 2.55186969761736, 2.01761332882084, 1.35656124714966, 0.812710966520876, 0.338036433865493])    # Paired ({(A-B)_PK}). Model Control
PK_paired_model_reduced_gEE = np.array([1.14973955457240, 1.68235451560962, 1.75520058261858, 1.70131922168596, 1.52745932131311, 1.13048459306852, 0.840126292745054, 0.438581531163469])    # Paired ({(A-B)_PK}). Model gEE x0.9825
PK_paired_model_reduced_gEI = np.array([5.75208846588290, 4.80925712698461, 2.60561800628665, 1.10211367035136, 0.460044336209576, 0.0117438582356510, 0.187487499834329, -0.0755308781434812])    # Paired ({(A-B)_PK}). Model gEI x0.965
PK_paired_model_upstream_deficit = np.array([1.20186301100136, 1.64970481199342, 1.80089369767290, 1.71507750311849, 1.37175704396749, 1.13456044723117, 0.791523219150575, 0.457464377784191])    # Paired ({(A-B)_PK}). Model upstream_deficit
PK_paired_err_model_control = np.array([0.0511740706795857, 0.0529491850769526, 0.0525299602938788, 0.0502514722788041, 0.0489525869304274, 0.0477087501140784, 0.0470069892321134, 0.0462720158954127])    # Paired ({(A-B)_PK}). Model Control
PK_paired_err_model_reduced_gEE = np.array([0.0425986860963261, 0.0431912189188233, 0.0435161982774777, 0.0432456831224134, 0.0430396907062545, 0.0425120607168643, 0.0421152705790713, 0.0414994690036606])    # Paired ({(A-B)_PK}). Model gEE x0.9825
PK_paired_err_model_reduced_gEI = np.array([0.0671775229443579, 0.0622992301684584, 0.0536999851239659, 0.0501760005362226, 0.0496430731564513, 0.0496339570451881, 0.0495978291432362, 0.0494006653142745])    # Paired ({(A-B)_PK}). Model gEI x0.965
PK_paired_err_model_upstream_deficit = np.array([0.0424771207321716, 0.0429410107058938, 0.0434305766600649, 0.0430919321129183, 0.0425986524756611, 0.0423325202917773, 0.0418856597464239, 0.0413344295534633])    # Paired ({(A-B)_PK}). Model Control





## Define subfigure domain.
figsize = (max2,1.*max2)

width1_11 = 0.09; width1_12 = 0.135; width1_13 = width1_12; width1_14 = width1_12; width1_15 = width1_12; width1_21 = 0.18; width1_22 = width1_21; width1_23 = width1_21; width1_24 = 0.21; width1_31 = 0.4; width1_32 = 0.35      # v3
x1_11 = 0.07; x1_12 = x1_11 + width1_11 + xbuf0; x1_13 = x1_12 + width1_12 + 0.5*xbuf0; x1_14 = x1_13 + width1_13 + 0.5*xbuf0; x1_15 = x1_14 + width1_14 + 0.5*xbuf0; x1_21 = 0.05; x1_22 = x1_21 + width1_21 + 0.45*xbuf0; x1_23 = x1_22 + width1_22 + 0.45*xbuf0; x1_24 = x1_23 + width1_23 + 0.8*xbuf0; x1_31=0.07; x1_32 = x1_31 + width1_31 + xbuf0; x1_33 = x1_32 + width1_32 + xbuf0
height1_11 = 0.22; height1_12 = 0.22; height1_13 = height1_12; height1_14 = height1_12; height1_15 = height1_12; height1_21= 0.2;  height1_22 = height1_21;  height1_23 = height1_21;  height1_24 = height1_21;  height1_31 = height1_21;  height1_32 = height1_31;  height1_33 = height1_31
y1_11 = 0.7; y1_12 = y1_11+0.02; y1_13 = y1_12; y1_14 = y1_12; y1_15 = y1_12; y1_21 = y1_11 - height1_22 - 1.45*ybuf0; y1_22 = y1_21; y1_23 = y1_21; y1_24 = y1_23+0.125*ybuf0; y1_31 = y1_21 - height1_31 - ybuf0; y1_32=y1_31; y1_33=y1_31




rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12_0 = [x1_12, y1_12, width1_12*0.05, height1_12]
rect1_12 = [x1_12+width1_12*0.2, y1_12, width1_12*(1-0.2), height1_12]
rect1_13_0 = [x1_13, y1_13, width1_13*0.05, height1_13]
rect1_13 = [x1_13+width1_13*0.2, y1_13, width1_13*(1-0.2), height1_13]
rect1_14_0 = [x1_14, y1_14, width1_14*0.05, height1_14]
rect1_14 = [x1_14+width1_14*0.2, y1_14, width1_14*(1-0.2), height1_14]
rect1_15_0 = [x1_15, y1_15, width1_15*0.05, height1_15]
rect1_15 = [x1_15+width1_15*0.2, y1_15, width1_15*(1-0.2), height1_15]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_24 = [x1_24, y1_24, width1_24, height1_24]
rect1_31 = [x1_31, y1_31, width1_31, height1_31]
rect1_32 = [x1_32, y1_32, width1_32, height1_32]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.01, 0.975, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.02+x1_12-x1_11, 0.975, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.01, 0.92 + y1_21 - y1_11, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.0478, 0.915 + y1_21 - y1_11, 'Mean Evidence Beta', fontsize=fontsize_fig_label, rotation='horizontal', color='k')
fig_temp.text(0.025+x1_22-x1_21, 0.92 + y1_21 - y1_11, 'G', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.288, 0.915 + y1_21 - y1_11, 'SD Evidence Beta', fontsize=fontsize_fig_label, rotation='horizontal', color='k')
fig_temp.text(0.023+x1_23-x1_21, 0.92 + y1_21 - y1_11, 'H', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.599, 0.915 + y1_21 - y1_11, 'PVB Index', fontsize=fontsize_fig_label, rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(-0.005+x1_24-x1_21, 0.92 + y1_21 - y1_11, 'I', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.295, 0.96, 'Control', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.275+x1_13-x1_12, 0.96, 'Lowered E/I', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.275+x1_14-x1_12, 0.96, 'Elevated E/I', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.335+x1_15-x1_12, 0.95, 'Sensory\nDeficit', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(0.032+x1_13-x1_11, 0.975, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.032+x1_14-x1_11, 0.975, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.032+x1_15-x1_11, 0.975, 'E', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.

## rect1_11: E/I perturbation schematics

## rect1_12: Psychometric function. Control model.
ax_0   = fig_temp.add_axes(rect1_12_0)
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_model_control_list[11:],    P_corr_model_control[11:], ErrBar_P_corr_model_control[11:], color=color_list[0], ecolor=color_list[0], fmt='.', zorder=4, clip_on=False , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_model_control_list[:9], 1.-P_corr_model_control[:9], ErrBar_P_corr_model_control[:9], color=[1-(1-ci)*0.5 for ci in color_list[0]], ecolor=[1-(1-ci)*0.5 for ci in color_list[0]], fmt='.', zorder=3, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_model_control_list[10], P_corr_model_control[10], ErrBar_P_corr_model_control[10], color=color_list[0], ecolor=color_list[0], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_model_control_list[9], 1.-P_corr_model_control[9], ErrBar_P_corr_model_control[9], color=[1-(1-ci)*0.5 for ci in color_list[0]], ecolor=[1-(1-ci)*0.5 for ci in color_list[0]], marker='.', zorder=3, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_model_control, x_list_psychometric), color=color_list[0], ls='-', clip_on=False, label='Higher SD Correct')#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_model_control, -x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list[0]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_model_control, x0_psychometric), s=15., color=color_list[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_model_control, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list[0]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4)
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
y_shift_spines = -0.064
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(-0.595,0.86), fontsize=fontsize_legend-1.5, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0., labelspacing=0.2)
for color,text,item in zip([color_list[0], [1-(1-ci)*0.5 for ci in color_list[0]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

## rect1_13: Psychometric function. Reduced gEE model.
ax_0   = fig_temp.add_axes(rect1_13_0)
ax   = fig_temp.add_axes(rect1_13)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_model_reduced_gEE_list[11:],    P_corr_model_reduced_gEE[11:], ErrBar_P_corr_model_reduced_gEE[11:], color=color_list[1], ecolor=color_list[1], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_model_reduced_gEE_list[:9], 1.-P_corr_model_reduced_gEE[:9], ErrBar_P_corr_model_reduced_gEE[:9], color=[1-(1-ci)*0.5 for ci in color_list[1]], ecolor=[1-(1-ci)*0.5 for ci in color_list[1]], fmt='.', zorder=3, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_model_reduced_gEE_list[10], P_corr_model_reduced_gEE[10], ErrBar_P_corr_model_reduced_gEE[10], color=color_list[1], ecolor=color_list[1], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_model_reduced_gEE_list[9], 1.-P_corr_model_reduced_gEE[9], ErrBar_P_corr_model_reduced_gEE[9], color=[1-(1-ci)*0.5 for ci in color_list[1]], ecolor=[1-(1-ci)*0.5 for ci in color_list[1]], marker='.', zorder=3, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_model_reduced_gEE, x_list_psychometric), color=color_list[1], ls='-', clip_on=False, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_model_reduced_gEE, -x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list[1]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_model_reduced_gEE, x0_psychometric), s=15., color=color_list[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_model_reduced_gEE, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list[1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4)
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
y_shift_spines = -0.064
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(-0.59,0.84), fontsize=fontsize_legend-1.5, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0., labelspacing=0.3)
for color,text,item in zip([color_list[1], [1-(1-ci)*0.5 for ci in color_list[1]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

## rect1_14: Psychometric function. Reduced gEI
ax_0   = fig_temp.add_axes(rect1_14_0)
ax   = fig_temp.add_axes(rect1_14)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_model_reduced_gEI_list[11:],    P_corr_model_reduced_gEI[11:], ErrBar_P_corr_model_reduced_gEI[11:], color=color_list[2], ecolor=color_list[2], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_model_reduced_gEI_list[:9], 1.-P_corr_model_reduced_gEI[:9], ErrBar_P_corr_model_reduced_gEI[:9], color=[1-(1-ci)*0.5 for ci in color_list[2]], ecolor=[1-(1-ci)*0.5 for ci in color_list[2]], fmt='.', zorder=3, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_model_reduced_gEI_list[10], P_corr_model_reduced_gEI[10], ErrBar_P_corr_model_reduced_gEI[10], color=color_list[2], ecolor=color_list[2], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_model_reduced_gEI_list[9], 1.-P_corr_model_reduced_gEI[9], ErrBar_P_corr_model_reduced_gEI[9], color=[1-(1-ci)*0.5 for ci in color_list[2]], ecolor=[1-(1-ci)*0.5 for ci in color_list[2]], marker='.', zorder=3, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_model_reduced_gEI, x_list_psychometric), color=color_list[2], ls='-', clip_on=False, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_model_reduced_gEI, -x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list[2]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_model_reduced_gEI, x0_psychometric), s=15., color=color_list[2], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_model_reduced_gEI, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list[2]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4)
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
y_shift_spines = -0.064
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(-0.59,0.84), fontsize=fontsize_legend-1.5, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0., labelspacing=0.3)
for color,text,item in zip([color_list[2], [1-(1-ci)*0.5 for ci in color_list[2]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

## rect1_15: Psychometric function. Sensory Deficit Model.
ax_0   = fig_temp.add_axes(rect1_15_0)
ax   = fig_temp.add_axes(rect1_15)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_model_upstream_deficit_list[11:],    P_corr_model_upstream_deficit[11:], ErrBar_P_corr_model_upstream_deficit[11:], color=color_list[3], ecolor=color_list[3], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_model_upstream_deficit_list[:9], 1.-P_corr_model_upstream_deficit[:9], ErrBar_P_corr_model_upstream_deficit[:9], color=[1-(1-ci)*0.5 for ci in color_list[3]], ecolor=[1-(1-ci)*0.5 for ci in color_list[3]], fmt='.', zorder=3, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_model_upstream_deficit_list[10], P_corr_model_upstream_deficit[10], ErrBar_P_corr_model_upstream_deficit[10], color=color_list[3], ecolor=color_list[3], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_model_upstream_deficit_list[9], 1.-P_corr_model_upstream_deficit[9], ErrBar_P_corr_model_upstream_deficit[9], color=[1-(1-ci)*0.5 for ci in color_list[3]], ecolor=[1-(1-ci)*0.5 for ci in color_list[3]], marker='.', zorder=3, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_model_upstream_deficit, x_list_psychometric), color=color_list[3], ls='-', clip_on=False, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_model_upstream_deficit, -x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list[3]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_model_upstream_deficit, x0_psychometric), s=15., color=color_list[3], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_model_upstream_deficit, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list[3]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4)
ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=-5.)
ax_0.set_ylim([0.4,1.])
ax.set_ylim([0.4,1.])
ax_0.set_xlim([-1,1])
ax.set_xlim([1,50])
ax_0.set_xticks([0.])
ax.xaxis.set_ticks([1, 10])
ax_0.yaxis.set_ticks([0.5, 1])
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
y_shift_spines = -0.064
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(-0.59,0.84), fontsize=fontsize_legend-1.5, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0., labelspacing=0.3)
for color,text,item in zip([color_list[3], [1-(1-ci)*0.5 for ci in color_list[3]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)



## rect1_21: Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(Reg_mean_models)), Reg_mean_models   , bar_width_compare3, alpha=bar_opacity, yerr=Reg_err_mean_models , ecolor='k', color=color_list[0:3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.bar(3                              , Reg_mean_models[3], bar_width_compare3, alpha=bar_opacity, yerr=Reg_err_mean_models[3], ecolor='k', color=color_list[3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.set_xlim([0,len(Reg_mean_models)-1+bar_width_compare3])
ax.set_ylim([0.,15.2])
ax.set_xticks([0., 1., 2., 3.])
ax.xaxis.set_ticklabels(['Control', 'Lowered E/I', 'Elevated E/I', 'Sensory Deficit'], rotation=30)
ax.set_yticks([0., 15.])
ax.yaxis.set_ticklabels([0, 0.15])
minorLocator = MultipleLocator(5)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=-0.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")

## rect1_22: Variance Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(Reg_std_models)), Reg_std_models, bar_width_compare3, alpha=bar_opacity, yerr=Reg_err_std_models, ecolor='k', color=color_list[0:3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.bar(3, Reg_std_models[3], bar_width_compare3, alpha=bar_opacity, yerr=Reg_err_std_models[3], ecolor='k', color=color_list[3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.set_xlim([0,len(Reg_std_models)-1+bar_width_compare3])
ax.set_ylim([0.,5.4])
ax.set_xticks([0., 1., 2., 3.])
ax.xaxis.set_ticklabels(['Control', 'Lowered E/I', 'Elevated E/I', 'Sensory Deficit'], rotation=30)
ax.set_yticks([0., 5.])
ax.yaxis.set_ticklabels([0, 0.05])
minorLocator = MultipleLocator(1)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=-0.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")

## rect1_23: Variance Beta/ Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(Reg_ratio_models)), Reg_ratio_models, bar_width_compare3, alpha=bar_opacity, yerr=Reg_err_ratio_models, ecolor='k', color=color_list[0:3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.bar(3, Reg_ratio_models[3], bar_width_compare3, alpha=bar_opacity, yerr=Reg_err_ratio_models[3], ecolor='k', color=color_list[3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,4.*bar_width_compare3], [Reg_ratio_models[0], Reg_ratio_models[0]], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1., 1.5], [0.4765, 0.513], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [0.46,0.46], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,2.5], [0.4965,0.4965], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_xlim([0,len(Reg_ratio_models)-1+bar_width_compare3])
ax.set_ylim([0.,0.5])
ax.set_xticks([0., 1., 2., 3.])
ax.xaxis.set_ticklabels(['Control', 'Lowered E/I', 'Elevated E/I', 'Sensory Deficit'], rotation=30)
ax.set_yticks([0., 0.5])
ax.yaxis.set_ticklabels([0, 0.5])
minorLocator = MultipleLocator(0.1)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=-0.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")

## rect1_24: Psychophysical Kernel, Model and perturbations
ax   = fig_temp.add_axes(rect1_24)
fig_funs.remove_topright_spines(ax)
ax.errorbar( i_PK_list, PK_paired_model_control, PK_paired_err_model_control, color=color_list[0], ecolor=color_list[0], marker='.', zorder=4, clip_on=False, markerfacecolor=color_list[0], markeredgecolor='k', linewidth=1., ls='-', elinewidth=0.6, markeredgewidth=0.6, markersize=5., capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar( i_PK_list, PK_paired_model_reduced_gEE, PK_paired_err_model_reduced_gEE, color=color_list[1], ecolor=color_list[1], marker='^', zorder=3, clip_on=False, markerfacecolor=color_list[1], markeredgecolor='k', linewidth=1., ls='-', elinewidth=0.6, markersize=2.5, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax.errorbar(i_PK_list, PK_paired_model_reduced_gEI, PK_paired_err_model_reduced_gEI, color=color_list[2], ecolor=color_list[2], marker='s', zorder=2, clip_on=False, markerfacecolor=color_list[2], markeredgecolor='k', linewidth=1., ls='-', elinewidth=0.6, markeredgewidth=0.6, markersize=2.5, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
ax.errorbar( i_PK_list, PK_paired_model_upstream_deficit, PK_paired_err_model_upstream_deficit, color=color_list[3], ecolor=color_list[3], marker='x', zorder=1, clip_on=False, markerfacecolor=color_list[3], markeredgecolor=color_list[3], linewidth=1., ls='-', elinewidth=0.6, markeredgewidth=0.6, markersize=3., capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.set_xlabel('Sample Number', fontsize=fontsize_legend)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend)
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.set_xlim([1,8.])
ax.set_ylim([0.,6.])
ax.set_xticks([1., 8.])
ax.set_yticks([0., 6.])
ax.text(0.1, 6.2, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(2.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
tmp1 = ax.scatter( i_PK_list, PK_paired_model_control         , color=color_list[0], marker='.', zorder=4, clip_on=False, facecolors=color_list[0], edgecolors='k', linewidths=0.6, s=12., label=label_list[0])#, linestyle=linestyle_list[i_var_a])
tmp2 = ax.scatter( i_PK_list, PK_paired_model_reduced_gEE     , color=color_list[1], marker='^', zorder=3, clip_on=False, facecolors=color_list[1], edgecolors='k', linewidths=0.6, s=4., label=label_list[1])#, linestyle=linestyle_list[i_var_a])
tmp3 = ax.scatter(i_PK_list, PK_paired_model_reduced_gEI      , color=color_list[2], marker='s', zorder=2, clip_on=False, facecolors=color_list[2], edgecolors='k', linewidths=0.6, s=3.5, label=label_list[2])#, linestyle=linestyle_list[i_var_a])
tmp4 = ax.scatter( i_PK_list, PK_paired_model_upstream_deficit, color=color_list[3], marker='x', zorder=1, clip_on=False, facecolors=color_list[3], edgecolors=color_list[3], linewidths=0.6, s=5., label=label_list[3])#, linestyle=linestyle_list[i_var_a])
legend = ax.legend(loc=(0.3,0.55), fontsize=fontsize_legend-1.5, frameon=False, ncol=1, columnspacing=1., handletextpad=0., scatterpoints=1)
for color,text,item in zip(color_list, legend.get_texts(), legend.legendHandles):
    text.set_color(color)

fig_temp.savefig(path_cwd+'Figure7.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
### Figure 8: Ketamine Data
## Mean/Variance Regression model                                                                                       # See DrugDayModellingScript.m: line 434-460
Reg_bars_avg_ketamine = np.array([0.301477882491371, 8.03170979301053, 3.07957706837800])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on ketamine.
Reg_bars_avg_saline = np.array([0.0105951281051839, 22.6330120449559, 3.98202205297213])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on saline.
Reg_bars_avg_pre_ketamine = np.array([-0.0550793702096746, 22.3906178260106, 4.21128772397676])  # [Bias, Val diff , Std diff]. Alfie regression Beta values pre ketamine.
Reg_bars_avg_pre_saline = np.array([0.0332565168334564, 21.8556360024285, 3.31761090955798])  # [Bias, Val diff , Std diff]. Alfie regression Beta values pre saline.
Reg_bars_err_avg_ketamine = np.array([0.0344479757147984, 0.342404202559497, 0.401989098892436])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on ketamine.
Reg_bars_err_avg_saline = np.array([0.0464483774377630, 0.599023332528761, 0.567865643238342])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on saline.
Reg_bars_err_avg_pre_ketamine = np.array([0.0482865079714177, 0.628273965090647, 0.560898605791729])  # [Bias, Val diff , Std diff]. Alfie regression Beta values pre ketamine.
Reg_bars_err_avg_pre_saline = np.array([0.0453384567091918, 0.570764012214166, 0.546373045671303])  # [Bias, Val diff , Std diff]. Alfie regression Beta values pre saline.

mean_effect_list_avg    = np.array([Reg_bars_avg_saline[1], Reg_bars_avg_ketamine[1]])             # Saline/ketamine. Mean Regressor
var_effect_list_avg     = np.array([Reg_bars_avg_saline[2], Reg_bars_avg_ketamine[2]])              # Saline/ketamine. Variance Regressor
var_mean_ratio_list_avg = var_effect_list_avg/mean_effect_list_avg            # Saline/ketamine. Variance Regressor/ Mean Regressor
mean_effect_list_avg_preSK    = np.array([Reg_bars_avg_pre_saline[1], Reg_bars_avg_pre_ketamine[1]])             # Saline/ketamine. Mean Regressor
var_effect_list_avg_preSK     = np.array([Reg_bars_avg_pre_saline[2], Reg_bars_avg_pre_ketamine[2]])              # Saline/ketamine. Variance Regressor
var_mean_ratio_list_avg_preSK = var_effect_list_avg_preSK/mean_effect_list_avg_preSK            # Saline/ketamine. Variance Regressor/ Mean Regressor

Mean_reg_err_bars_avg_v2  = np.abs([Reg_bars_err_avg_saline[1], Reg_bars_err_avg_ketamine[1]])
Var_reg_err_bars_avg_v2  = np.abs([Reg_bars_err_avg_saline[2], Reg_bars_err_avg_ketamine[2]])
Var_mean_ratio_err_Reg_bars_avg_v2  = var_mean_ratio_list_avg*((Var_reg_err_bars_avg_v2/var_effect_list_avg)**2 + (Mean_reg_err_bars_avg_v2/mean_effect_list_avg)**2)**0.5

### PK                                                                                                                    # See DrugDayModellingScript.m: end of DrugDayFigs_PsychKernel.m (For old, unpaired method in see lines 275-430).
i_PK_list_6 = np.arange(1,6+1)
t_PK_list_6 = 0.125 + 0.25*np.arange(6)
PK_avg_ketamine = np.array([1.54344385843845, 1.35716344743175, 1.43387415617364, 1.45289852782578, 1.03978819671103, 1.07870494126977])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_avg_saline = np.array([4.18181966637721, 3.55509157907307, 3.53233603357858, 3.44221565600612, 3.53852570816032, 4.01180355888058])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_avg_ketamine_errbar = np.array([0.129324410912331, 0.128967918975535, 0.128371981115283, 0.131530797417894, 0.131345398035829, 0.127652509702890])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_avg_saline_errbar = np.array([0.200802746872777, 0.186249782799301, 0.182811811844770, 0.185909222851744, 0.186982246146777, 0.192706657490924])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.



## Pcorr & RT vs time                                                                                                   # See DrugDayModellingScript.m: DrugDayFigs_TimeCourseAnal
t_list_Pcorr_RT = np.arange(-20, 61)
Pcorr_t_mean_list_ketamine_avg = np.array([0.853424224135278, 0.866062950123339, 0.867860389063784, 0.865256389683876, 0.854709347878012, 0.864004384919810, 0.851511277189839, 0.844034555282928, 0.850709544077447, 0.852139024681164, 0.860721863014822, 0.868516044113901, 0.880513057272557, 0.880637063318617, 0.879475616453274, 0.872784485976770, 0.875854940509344, 0.869811808579097, 0.865166636493611, 0.847617727267153, 0.827700426793940, 0.795301791292255, 0.747130362330291, 0.687449879578272, 0.650106714514782, 0.631454263458375, 0.597781703623970, 0.578133948331541, 0.584707269405427, 0.599485142177337, 0.606908175486982, 0.607308152302572, 0.636790179798571, 0.646269505291055, 0.633323941992166, 0.631481233711707, 0.655285811314900, 0.671764730020737, 0.674979965075746, 0.694676325996089, 0.719097019583916, 0.721015361061535, 0.729047704697712, 0.739495195439528, 0.755990215208679, 0.772061118502966, 0.777009147424545, 0.795643103788181, 0.809585010882080, 0.825646981033650, 0.831403313312581, 0.826857468228732, 0.831233098566522, 0.832457588161480, 0.831132981985795, 0.820418431151541, 0.810265173256729, 0.812770297620254, 0.811909658336082, 0.823416638117737, 0.821589940598725, 0.815688427263022, 0.820763102780976, 0.827252711871524, 0.836315908832532, 0.856711585055734, 0.852376640175876, 0.851853925662496, 0.847180003251683, 0.842029164499347, 0.828549507833659, 0.831165655412017, 0.833727387153424, 0.816053536965676, 0.824824054535658, 0.827041443364114, 0.832387368980002, 0.830274847574665, 0.827185770608855, 0.841671506555382, 0.842025440489822])
Pcorr_t_se_list_ketamine_avg   = np.array([0.0134754536722323, 0.0129272002894246, 0.0129187865351855, 0.0177045288226560, 0.0180027205220793, 0.0149118610536051, 0.0127688507493567, 0.0118659512220985, 0.0111577855268612, 0.0120176129127855, 0.00948368574893723, 0.00964166431631806, 0.0121999696050822, 0.0127023993007378, 0.0121536610874225, 0.0116296287194642, 0.0103749544537303, 0.0106229245535866, 0.0116221016671551, 0.0158604706751390, 0.0178196115267471, 0.0209021633066683, 0.0246822074169426, 0.0263872027726200, 0.0251692588119893, 0.0229187003249104, 0.0213377023731106, 0.0223598274149528, 0.0208107701429526, 0.0194153630158397, 0.0202007122509156, 0.0222227406954684, 0.0183974531253689, 0.0187329339295052, 0.0210498436675537, 0.0225659469217256, 0.0233427053966200, 0.0241774875421121, 0.0194131786207001, 0.0172840355701620, 0.0146096381351686, 0.0141338021438404, 0.0141277342458984, 0.0163493006286821, 0.0159212620233197, 0.0170984276569845, 0.0135574578657653, 0.0123104960774650, 0.0141142134733332, 0.0143538779170259, 0.0127844729107005, 0.0140954606465289, 0.0141015773924770, 0.0135594882569432, 0.0141375176560166, 0.0153575593987532, 0.0167228866031091, 0.0159814830496596, 0.0182887500905595, 0.0198887454817218, 0.0204943588331689, 0.0352238955230879, 0.0356398467124452, 0.0363415225709289, 0.0351357268175821, 0.0132135925222927, 0.0141501707110350, 0.0124029057834276, 0.0133557106692394, 0.0130157748810284, 0.0145976189866492, 0.0152494903418007, 0.0158923647524939, 0.0198127295988518, 0.0155460458073380, 0.0147414886025043, 0.0134178203870482, 0.0124791139273467, 0.0114227927375121, 0.0116882612285543, 0.0119871896150782])
Pcorr_t_mean_list_saline_avg = np.array([0.848857637146466, 0.854002571210600, 0.855000299786298, 0.853624721721915, 0.852028343315863, 0.859816148297194, 0.868743422012807, 0.866756949138544, 0.866553821541771, 0.873626663236312, 0.880429059289940, 0.877860620009917, 0.881864374360389, 0.885746043653791, 0.884433414095375, 0.888669902041409, 0.887029269881040, 0.885555272439850, 0.880590866926983, 0.874290015979847, 0.874345799789814, 0.869859940137557, 0.871737479258346, 0.876322559710813, 0.880417360216343, 0.888433979197166, 0.881915862410222, 0.875814236474055, 0.876365155087876, 0.874725323635650, 0.878075681946344, 0.867899028296989, 0.874362030107879, 0.878137966155691, 0.871596402003433, 0.877725262723117, 0.870874952214329, 0.878311236719616, 0.872341308204148, 0.874011316110839, 0.884715959001642, 0.875916976347480, 0.870037917745871, 0.859165245002216, 0.865997180099948, 0.858574146649431, 0.846753698045958, 0.848548006197340, 0.852424397116881, 0.853740905629304, 0.852418047475977, 0.856377804370167, 0.857070391275833, 0.850257492963939, 0.850508178327832, 0.861850969760097, 0.859200864068422, 0.860938248561843, 0.863133207368750, 0.864335352529900, 0.865907519151203, 0.870142314573412, 0.874118497550266, 0.877156520686173, 0.881186187695262, 0.878462897838302, 0.875316440110923, 0.862679444775853, 0.855022369258257, 0.849461069310932, 0.845483240873336, 0.858798537781125, 0.857706688361579, 0.861817589234997, 0.866908391739364, 0.879446387076234, 0.879625902563498, 0.869251808067098, 0.874329718269915, 0.875456798685543, 0.880803673796730])
Pcorr_t_se_list_saline_avg   = np.array([0.0183931655631401, 0.0147861554609992, 0.0140775545891902, 0.0158579274150793, 0.0161666790706485, 0.0152377127826169, 0.0142077056361510, 0.0141903482265833, 0.0150914769501323, 0.0133240758111552, 0.0121270219215984, 0.0116300200141136, 0.0114777096672419, 0.0103018529433321, 0.00965233945890063, 0.0111308983534501, 0.0108728001146828, 0.0108424940190862, 0.0117519383844001, 0.0119666243916425, 0.0126086653153199, 0.0130499699659537, 0.0129346757166396, 0.0113972729487351, 0.0105785409476471, 0.0117380652351315, 0.0123423821027967, 0.0129543864768992, 0.0119295186790432, 0.0110585076954663, 0.0112845808074465, 0.0118399864859387, 0.0116907733951720, 0.0100549141305751, 0.0118225742737052, 0.0137579273464834, 0.0136047756822321, 0.0123049611644541, 0.0119619392239452, 0.0109985665630805, 0.00956670770061928, 0.00966826913368934, 0.0110462786401385, 0.0126833504625770, 0.0103959788586307, 0.0102732078204860, 0.00929690842015938, 0.0100606043009623, 0.00949394685771559, 0.0105365000394903, 0.0117263058197227, 0.0126971043931647, 0.0121942186938221, 0.0113149016676658, 0.0107431582935099, 0.0120956397385124, 0.0141817255984835, 0.0115992172429651, 0.0114014607000965, 0.00969827034356536, 0.00888938303910926, 0.0104200905280017, 0.0106037376903559, 0.0106777952360654, 0.0121941747805425, 0.0131949976197984, 0.0130599853192774, 0.0132378485647113, 0.0129595219151639, 0.0131450610505754, 0.0127334861864246, 0.0103932843642995, 0.0112625361306602, 0.0117879284619030, 0.0131121080149667, 0.0117152696883296, 0.0122287318440754, 0.0116540478897034, 0.00972553575451297, 0.00953176182639584, 0.00963779115770253])

## Psychometric function, drug days                                                                                     # See DrugDayModellingScript.m:  line 581-605
d_evidence_avg_ket_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_avg_ket_list =  np.array([0.214285714285714, 0.227544910179641, 0.276073619631902, 0.345153664302600, 0.405566600397614, 0.414117647058824, 0.443181818181818, 0.654676258992806, 0.679723502304148, 0.730192719486081, 0.776497695852535, 0.780645161290323, 0.817610062893082, 0.760000000000000])  # Log-Spaced.
ErrBar_P_corr_avg_ket_list = np.array([0.0775443069059728, 0.0324423232138953, 0.0247600123006130, 0.0231156314392388, 0.0218926684642180, 0.0238931061410307, 0.0374447846695392, 0.0403291542530031, 0.0223966887117801, 0.0205393710767981, 0.0199970633897213, 0.0235027996578259, 0.0306249218147176, 0.0603986754821660])
d_evidence_avg_saline_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_avg_saline_list =  np.array([0, 0.0197044334975369, 0.0460358056265985, 0.0873983739837398, 0.212612612612613, 0.298200514138818, 0.306878306878307, 0.712230215827338, 0.842794759825328, 0.893478260869565, 0.931102362204724, 0.957186544342508, 0.988372093023256, 0.978260869565217])  # Log-Spaced.
ErrBar_P_corr_avg_saline_list = np.array([0, 0.00975466764565723, 0.0105980394638833, 0.0127323844681742, 0.0173676975487633, 0.0231945481309261, 0.0335472510205536, 0.0383994843612226, 0.0170083303912917, 0.0143840837630455, 0.0112374823886864, 0.0111947540783662, 0.00817422877241312, 0.0215015371759496])



## Regression analysis, Experiments                                                                                     # See DrugDayModellingScript.m: line227-261, DrugRegStrat
Reg_values_avg_ketamine = np.array([-0.174395803703761, 0.104110887712269, -0.313605602255712, 7.83370238384515, 1.67529576738009, -0.969703512156112, -0.257158543370569, 0.242863229505471, -7.13913845630237, -1.17360996418811, 0.478464138982874])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_avg_ketamine = np.array([0.352930238120823, 0.212063514040423, 0.207866319751090, 0.881717895145497, 0.424149856415424, 0.429111478818009, 0.210678987432390, 0.204700832397445, 0.848719333494373, 0.415126185377429, 0.419914852352852]) # Error bars for Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_avg_saline = np.array([-0.0981113552951833, 0.994158388688079, 0.488084558896090, 20.4918034565360, 1.95550378806422, -1.26339159294571, -0.231531657133946, -0.498397038365780, -23.0031799495554, -0.690649929786167, 1.50571679149191])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_avg_saline = np.array([0.470630543630353, 0.304631692836395, 0.285292540728253, 1.27271321051066, 0.587390333461395, 0.595939727242080, 0.288813006397683, 0.285224475035092, 1.31393932569555, 0.597720796478704, 0.603440930043411])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)



## Fitting Psychometric Functions
x_list_psychometric = np.arange(0.01, 0.5, 0.01)                                                                        # See figure_psychometric_function_fit.py, esp lines 322-527
x0_psychometric = 0.
## non-binned MLE (i.e. done using literal net evidence, via matlab). See Psychometric_function_fit_DrugDays_NL.m.
psychometric_params_avg_ketamine_all      = [0.237752612052054, 0.738629708456002, 0.0354392999982641]
psychometric_params_avg_saline_all        = [0.0643105028665596, 1.12858456632950, 0.0133052082986780]



x1_31 = 0.071; x1_32 = x1_31 + width1_31 + 0.7*xbuf0; x1_33 = x1_32 + width1_32 + 1.15*xbuf0; x1_34 = x1_33 + width1_33 + 0.7*xbuf0

## Define subfigure domain.
figsize = (max15,0.8*max15)
width1_11 = 0.22; width1_12 = 0.2; width1_13 = width1_12
width1_21 = 0.1; width1_22 = width1_21; width1_23 = width1_21; width1_24 = width1_11
x1_11 = 0.098; x1_12 = x1_11 + width1_11 + 1.25*xbuf0; x1_13 = x1_12 + width1_12 + 1.15*xbuf0
x1_21 = 0.0825; x1_22 = x1_21 + width1_21 + 0.8*xbuf0; x1_23 = x1_22 + width1_22 + 1.1*xbuf0; x1_24 = x1_23 + width1_23 + 1.55*xbuf0 #x1_24 = x1_23 + width1_23 + 1.25*xbuf0
height1_11 = 0.3; height1_12 = height1_11; height1_13 = height1_12
height1_21= 0.27;  height1_22 = height1_21;  height1_23 = height1_21;  height1_24 = height1_21
y1_11 = 0.63; y1_12 = y1_11; y1_13=y1_12
y1_21 = y1_11 - height1_21 - 3.1*ybuf0; y1_22 = y1_21; y1_23 = y1_22; y1_24 = y1_23 + 0.23*ybuf0


rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12_0 = [x1_12, y1_12, width1_12*0.05, height1_12]
rect1_12 = [x1_12+width1_12*0.2, y1_12, width1_12*(1-0.2), height1_12]
rect1_13_0 = [x1_13, y1_13, width1_13*0.05, height1_13]
rect1_13 = [x1_13+width1_13*0.2, y1_13, width1_13*(1-0.2), height1_13]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_24 = [x1_24, y1_24, width1_24, height1_24]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.005, 0.925, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.007+x1_12-x1_11, 0.925, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.007+x1_13-x1_11, 0.925, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.005, 0.975 + y1_21 - y1_11, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.047+x1_22-x1_21, 0.975 + y1_21 - y1_11, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.049+x1_23-x1_21, 0.975 + y1_21 - y1_11, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.001+x1_24-x1_21, 0.975 + y1_21 - y1_11, 'G', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.
fig_temp.text(0.495, 0.95, 'Saline', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.475+x1_13-x1_12, 0.95, 'Ketamine', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.14-x1_11+x1_21, 0.933 + y1_21 - y1_11, 'Mean Evidence\nBeta', fontsize=fontsize_fig_label-1, rotation='horizontal', color='k', va='center', horizontalalignment='center')
fig_temp.text(0.3305-x1_11+x1_21, 0.933 + y1_21 - y1_11, 'SD Evidence\nBeta', fontsize=fontsize_fig_label-1, rotation='horizontal', color='k', va='center', horizontalalignment='center')
fig_temp.text(0.5535-x1_11+x1_21, 0.933 + y1_21 - y1_11, 'PVB Index', fontsize=fontsize_fig_label-1, rotation='horizontal', color='k', ha='center', va='center')


## rect1_11: Correct Probability vs time, Both Monkeys
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax)
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_avg, color=color_list_expt[0], linestyle='-', zorder=3, clip_on=False, label='Saline', linewidth=1.)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_avg + Pcorr_t_se_list_saline_avg, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_avg - Pcorr_t_se_list_saline_avg, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_avg, color=color_list_expt[1], linestyle='-', zorder=3, clip_on=False, label='Ketamine', linewidth=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_avg + Pcorr_t_se_list_ketamine_avg, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_avg - Pcorr_t_se_list_ketamine_avg, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
ax.fill_between([5., 30.], 1., lw=0, color='k', alpha=0.2, zorder=0)
ax.set_xlabel('Time (mins)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Correct Probability', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([-20, 60])
ax.set_ylim([0.5,1.])
ax.set_xticks([-20, 0, 20, 40, 60])
ax.set_yticks([0.5, 1.])
ax.yaxis.set_ticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
legend = ax.legend(loc=(0.39,-0.02), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)



##### Psychometric functions.
## rect1_12: Psychometric function. Saline.
ax_0   = fig_temp.add_axes(rect1_12_0)
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_avg_saline_list[:],    P_corr_avg_saline_list[:], ErrBar_P_corr_avg_saline_list[:], color=color_list_expt[0], ecolor=color_list_expt[0], fmt='.', zorder=4, clip_on=False , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_avg_saline_list[:], 1.-P_corr_avg_saline_list[:], ErrBar_P_corr_avg_saline_list[:], color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[0]], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_avg_saline_all, x_list_psychometric), color=color_list_expt[0], ls='-', clip_on=False, zorder=3, label='Higher SD Correct')#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_avg_saline_all, -x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], ls='-', clip_on=False, zorder=2, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_avg_saline_all, x0_psychometric), s=15., color=color_list_expt[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_avg_saline_all, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False)
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
y_shift_spines = -0.08864
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(-0.34,-0.12), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0., labelspacing=0.3)
for color,text,item in zip([color_list_expt[0], [1-(1-ci)*0.5 for ci in color_list_expt[0]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

## rect1_13: Psychometric function. Ketamine.
ax_0   = fig_temp.add_axes(rect1_13_0)
ax   = fig_temp.add_axes(rect1_13)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_avg_ket_list[:],    P_corr_avg_ket_list[:], ErrBar_P_corr_avg_ket_list[:], color=color_list_expt[1], ecolor=color_list_expt[1], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_avg_ket_list[:], 1.-P_corr_avg_ket_list[:], ErrBar_P_corr_avg_ket_list[:], color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[1]], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_avg_ketamine_all, x_list_psychometric), color=color_list_expt[1], ls='-', clip_on=False, zorder=3, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_avg_ketamine_all, -x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ls='-', clip_on=False, zorder=2, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_avg_ketamine_all, x0_psychometric), s=15., color=color_list_expt[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_avg_ketamine_all, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False)
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
y_shift_spines = -0.08864
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(-0.6,0.74), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0.)
for color,text,item in zip([color_list_expt[1], [1-(1-ci)*0.5 for ci in color_list_expt[1]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)



## rect1_21: Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(mean_effect_list_avg)), mean_effect_list_avg, bar_width_compare3, alpha=bar_opacity, yerr=Mean_reg_err_bars_avg_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(mean_effect_list_avg_preSK[0]+mean_effect_list_avg_preSK[1]), 0.5*(mean_effect_list_avg_preSK[0]+mean_effect_list_avg_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.], [25.2], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [24., 24.], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_xlim([0,len(mean_effect_list_avg)-1+bar_width_compare3])
ax.set_ylim([0.,26.])
ax.set_xticks([0., 1.])
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
ax.set_yticks([0., 25.])
ax.set_yticklabels([0., 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")

## rect1_22: Variance Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(var_effect_list_avg)), var_effect_list_avg, bar_width_compare3, alpha=bar_opacity, yerr=Var_reg_err_bars_avg_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(var_effect_list_avg_preSK[0]+var_effect_list_avg_preSK[1]), 0.5*(var_effect_list_avg_preSK[0]+var_effect_list_avg_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.set_xlim([0,len(var_effect_list_avg)-1+bar_width_compare3])
ax.set_ylim([0.,5.])
ax.set_xticks([0., 1.])
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
ax.set_yticks([0., 5.])
ax.set_yticklabels([0., 0.05])
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")

## rect1_23: Variance Beta/ Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(var_mean_ratio_list_avg)), var_mean_ratio_list_avg, bar_width_compare3, alpha=bar_opacity, yerr=Var_mean_ratio_err_Reg_bars_avg_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(var_mean_ratio_list_avg_preSK[0]+var_mean_ratio_list_avg_preSK[1]), 0.5*(var_mean_ratio_list_avg_preSK[0]+var_mean_ratio_list_avg_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.], [0.49], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [0.46,0.46], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_xlim([0,len(var_mean_ratio_list_avg)-1+bar_width_compare3])
ax.set_ylim([0.,0.5])
ax.set_xticks([0., 1.])
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
ax.set_yticks([0., 0.5])
ax.yaxis.set_ticklabels([0, 0.5])
minorLocator = MultipleLocator(0.1)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")



## rect1_24: Psychophysical Kernel, Monkey A
ax   = fig_temp.add_axes(rect1_24)
fig_funs.remove_topright_spines(ax)
tmp = ax.errorbar(i_PK_list_6, PK_avg_ketamine, PK_avg_ketamine_errbar, color=color_list_expt[1], linestyle='-', marker='.', zorder=(3-1), clip_on=False, markeredgecolor='k', elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
tmp = ax.errorbar(i_PK_list_6, PK_avg_saline, PK_avg_saline_errbar, color=color_list_expt[0], linestyle='-', marker='.', zorder=(3-1), clip_on=False, markeredgecolor='k', elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
ax.set_xlabel('Sample Number', fontsize=fontsize_legend)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend)
ax.set_ylim([0.,4.35])
ax.set_xlim([1,6])
ax.set_xticks([1,6])
ax.set_yticks([0., 4.])
ax.text(0.1, 4.5, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.plot(i_PK_list_6, PK_avg_saline, label='Saline', color=color_list_expt[0], linestyle='-', zorder=0, clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(i_PK_list_6, PK_avg_ketamine, label='Ketamine', color=color_list_expt[1], linestyle='-', zorder=0, clip_on=False)#, linestyle=linestyle_list[i_var_a])
legend = ax.legend(loc=(0.,0.4), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0.)
for color,text,item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

fig_temp.savefig(path_cwd+'Figure8.pdf')    #Finally save fig
