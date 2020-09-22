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
mpl.rcParams['hatch.linewidth'] = 1.5             # need mpl 2.0
mpl.rcParams['axes.linewidth'] = 1.
print(mpl.__version__)
plt.rcParams["font.family"] = "Arial"
# mpl.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]                                                    # In order to use bold font in latex expressions
# matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
from mpl_toolkits.axes_grid1 import make_axes_locatable                                                                 # To make imshow 2D-plots to have color bars at the same height as the figure
from scipy.interpolate import UnivariateSpline
import MFAMPA_functions_WongWang2006_Hunt_Noisy_Input
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.mlab as mlab
# mpl.style.use('classic')                                                                                              # One potential way to resolve figure differences across computers...

path_cwd = './'         #Current Directory

#-----------------------------------------------------------------------------------------------------------------------### Initializtion
### Load spike rasters and make firing rate profiles ###
### Flags for what to compute and what not to.

## First define paths to look into
path_joint = ''#''/'                                                                                                        # Joint between various folder sub-name strings
suffix_joint = "_"                                                                                                      # Joint between strings of suffix sub-parts
empty_joint = ""
foldername_joint = "_"

### Information for the indices and savepath of the simulation.

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
### Figure 3S1: Experimental non-drug data, Narrow-Broad Trials

## Schematics
x_schem = np.arange(1,99,1)
sigma_narrow = 12
sigma_broad = 24

## Define Data. Alternatively can also import.                                                                          # See MainAnalysisNonDrugDays.m: NarrowBroadTrialsCOL
n_A_non_drug = 41.                                                                                                      # Alfie, 41 runs
n_H_non_drug = 63.                                                                                                      # Henry, 63 runs
ENB_bars_A_non_drug = np.array([0.654357459379616, 0.780918727915194, 0.914545454545455, ])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Alfie, 41 runs
ENB_bars_H_non_drug = np.array([0.565150346954510, 0.749131944444444, 0.882795698924731])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Henry, 63 runs
ENB_bars_avg_non_drug = (n_A_non_drug*ENB_bars_A_non_drug + n_H_non_drug*ENB_bars_H_non_drug)/ (n_A_non_drug+n_H_non_drug)

# Error bar data (of the old form).                                                                                     # See MainAnalysisNonDrugDays.m: NarrowBroadTrialsCOL_Errs)
ENB_bars_err_A_non_drug = np.array([0.0182779207004317, 0.0173859061308377, 0.0119203467090900])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Alfie, 41 runs
ENB_bars_err_H_non_drug = np.array([0.0137651698694795, 0.0127724908609410, 0.0105477640134060])                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Alfie, 41 runs
ENB_bars_err_avg_non_drug = ((ENB_bars_err_A_non_drug*n_A_non_drug/(n_A_non_drug+n_H_non_drug))**2 + (ENB_bars_err_H_non_drug*n_H_non_drug/(n_H_non_drug+n_H_non_drug))**2)**0.5                                                                         # [Broad probability when means are equal, Accuracy(correct probability) when narrow is correct, Accuracy when broad is correct]. Alfie, 41 runs

## Extract number distribution of stimuli, for Narrow-Broad trials.                                                     # See MainAnalysisNonDrugDays.m: lines 137-187)
n_distribution_narrow_high_monkey_A = np.loadtxt('Data/Stim_Distribution/dx=1/n_distribution_narrow_high_monkey_A.txt', delimiter=',')              # axis is (narrow, broad)
n_distribution_narrow_high_monkey_H = np.loadtxt('Data/Stim_Distribution/dx=1/n_distribution_narrow_high_monkey_H.txt', delimiter=',')              # axis is (narrow, broad)
n_distribution_broad_high_monkey_A = np.loadtxt('Data/Stim_Distribution/dx=1/n_distribution_broad_high_monkey_A.txt', delimiter=',')              # axis is (narrow, broad)
n_distribution_broad_high_monkey_H = np.loadtxt('Data/Stim_Distribution/dx=1/n_distribution_broad_high_monkey_H.txt', delimiter=',')              # axis is (narrow, broad)
n_distribution_NB_balanced_monkey_A = np.loadtxt('Data/Stim_Distribution/dx=1/n_distribution_NB_balanced_monkey_A.txt', delimiter=',')              # axis is (narrow, broad)
n_distribution_NB_balanced_monkey_H = np.loadtxt('Data/Stim_Distribution/dx=1/n_distribution_NB_balanced_monkey_H.txt', delimiter=',')              # axis is (narrow, broad)
density_distribution_narrow_high_all = (n_distribution_narrow_high_monkey_A + n_distribution_narrow_high_monkey_H) / np.sum(n_distribution_narrow_high_monkey_A + n_distribution_narrow_high_monkey_H)
density_distribution_broad_high_all = (n_distribution_broad_high_monkey_A + n_distribution_broad_high_monkey_H) / np.sum(n_distribution_broad_high_monkey_A + n_distribution_broad_high_monkey_H)
density_distribution_NB_balanced_all = (n_distribution_NB_balanced_monkey_A + n_distribution_NB_balanced_monkey_H) / np.sum(n_distribution_NB_balanced_monkey_A + n_distribution_NB_balanced_monkey_H)


## Define subfigure domain.
figsize = (max15, 1.2*max15)

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
fig_temp.text(0.006, 0.945, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.09, 0.97, 'Narrow Correct', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.008 + x1_12 - x1_11, 0.945, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.433, 0.97, 'Broad Correct', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.008 + x1_13 - x1_11, 0.945, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.785, 0.97, 'Ambiguous', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.01, 0.93 + y1_11a - y1_11, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.02 + x1_12 - x1_11, 0.93 + y1_12a - y1_12, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.02 + x1_13 - x1_11, 0.93 + y1_13a - y1_13, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.21, 0.975 + y1_21 - y1_11, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.01, 0.94 + y1_21 - y1_11, 'G', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015 + x1_22 - x1_21, 0.94 + y1_21 - y1_11, 'H', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.715, 0.975 + y1_21 - y1_11, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.02 + x1_23 - x1_21, 0.94 + y1_21 - y1_11, 'I', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015 + x1_24 - x1_21, 0.94 + y1_21 - y1_11, 'J', fontsize=fontsize_fig_label, fontweight='bold')





## rect1_11: schematics, Narrow Correct
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax)
dist_narrow = mlab.normpdf(x_schem, 54., sigma_narrow)
dist_broad = mlab.normpdf(x_schem, 54.-8., sigma_broad)
ax.plot(x_schem, dist_narrow, color=color_NB[0], ls='-', clip_on=False, zorder=10, label='Narrow')#, linestyle=linestyle_list[i_var_a])
ax.plot(x_schem, dist_broad, color=color_NB[1], ls='-', clip_on=False, zorder=9, label='Broad')#, linestyle=linestyle_list[i_var_a])
ax.plot([48,48,60,60], [0.038,0.039,0.039,0.038], ls='-', lw=1., color=color_NB[0], clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.text(23.5, 0.0375, r'$\mu_N\in$', fontsize=fontsize_legend-1, fontweight='bold')
ax.text(56, 0.0345, r'$\mu_B=\mu_N-8$', fontsize=fontsize_legend-1, fontweight='bold')
ax.plot([46,54], [0.0355,0.0355], ls='-', lw=1., color=color_NB[1], clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.plot([46,46], [0.035,0.036], ls='-', lw=1., color=color_NB[1], clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.plot([54,54], [0.035,0.036], ls='-', lw=1., color=color_NB[1], clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.plot([46,46], [mlab.normpdf(46., 46., sigma_broad),0.035], ls='--', lw=0.8, color='k', clip_on=False, zorder=0, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot([54,54], [mlab.normpdf(54., 54., sigma_narrow),0.039], ls='--', lw=0.8, color='k', clip_on=False, zorder=0, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.set_xlabel('Evidence Strength', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Probability Density', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0.,100.])
ax.set_xticks([0, 50, 100])
ax.xaxis.set_ticklabels([0, 50, 100])
ax.set_yticks([0., 0.04])
ax.yaxis.set_ticklabels([0, 0.04])
minorLocator = MultipleLocator(0.01)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
legend = ax.legend(loc=(-0.26,0.58), fontsize=fontsize_legend-2, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip(color_NB, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

## rect1_12: schematics, Broad Correct
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax)
dist_narrow = mlab.normpdf(x_schem, 54.-8., sigma_narrow)
dist_broad = mlab.normpdf(x_schem, 54., sigma_broad)
ax.plot(x_schem, dist_narrow, color=color_NB[0], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(x_schem, dist_broad, color=color_NB[1], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot([48,48,60,60], [0.038,0.039,0.039,0.038], ls='-', lw=1., color=color_NB[1], clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.text(23.5, 0.0375, r'$\mu_B\in$', fontsize=fontsize_legend-1, fontweight='bold')
ax.text(56, 0.034, r'$\mu_N=\mu_B-8$', fontsize=fontsize_legend-1, fontweight='bold')
ax.plot([46,54], [0.0355,0.0355], ls='-', lw=1., color=color_NB[0], clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.plot([46,46], [0.035,0.036], ls='-', lw=1., color=color_NB[0], clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.plot([54,54], [0.035,0.036], ls='-', lw=1., color=color_NB[0], clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.plot([46,46], [mlab.normpdf(46., 46., sigma_narrow),0.035], ls='--', lw=0.8, color='k', clip_on=False, zorder=0, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot([54,54], [mlab.normpdf(54., 54., sigma_broad),0.039], ls='--', lw=0.8, color='k', clip_on=False, zorder=0, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.set_xlabel('Evidence Strength', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Probability Density', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0.,100.])
ax.set_xticks([0, 50, 100])
ax.xaxis.set_ticklabels([0, 50, 100])
ax.set_yticks([0., 0.04])
ax.yaxis.set_ticklabels([0, 0.04])
minorLocator = MultipleLocator(0.01)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_13: schematics, Equal Means
ax   = fig_temp.add_axes(rect1_13)
fig_funs.remove_topright_spines(ax)
dist_narrow = mlab.normpdf(x_schem, 52., sigma_narrow)
dist_broad = mlab.normpdf(x_schem, 52., sigma_broad)
ax.plot(x_schem, dist_narrow, color=color_NB[0], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(x_schem, dist_broad, color=color_NB[1], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot([44,44,56,56], [0.038,0.039,0.039,0.038], ls='-', lw=1., color=color_NB[1], clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.text(19, 0.0375, r'$\mu_B\in$', fontsize=fontsize_legend-1, fontweight='bold')
ax.text(56, 0.0345, r'$\mu_N=\mu_B$', fontsize=fontsize_legend-1, fontweight='bold')
ax.plot([52,52], [mlab.normpdf(52., 52., sigma_broad),0.039], ls='--', lw=0.8, color='k', clip_on=False, zorder=0, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.set_xlabel('Evidence Strength', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Probability Density', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0.,100.])
ax.set_xticks([0, 50, 100])
ax.xaxis.set_ticklabels([0, 50, 100])
ax.set_yticks([0., 0.04])
ax.yaxis.set_ticklabels([0, 0.04])
minorLocator = MultipleLocator(0.01)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

### Distribution of stimuli conditions
## rect1_11a: Stimuli Distribution for narrow-high trials.
ax = fig_temp.add_axes(rect1_11a)
aspect_ratio=1.
plt.imshow(density_distribution_narrow_high_all, extent=(0.,100.,0.,100.), interpolation='nearest', cmap='BuPu', aspect=aspect_ratio, origin='lower', vmin=0., vmax=np.max(density_distribution_narrow_high_all))
ax.plot([0,100], [0,100], color='k', alpha=0.8, ls='--', lw=1.)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([25., 50., 75.])
ax.set_yticks([25., 50., 75.])
ax.set_xlim([25., 75.])
ax.set_ylim([25., 75.])
ax.tick_params(direction='out', pad=0.75)
ax.set_xlabel('Mean Broad\nEvidence', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylabel('Mean Narrow\nEvidence', fontsize=fontsize_legend, labelpad=1.)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cb_temp = plt.colorbar(ticks=[0., 0.01, 0.02], cax=cax_scale_bar_size, orientation='horizontal')
cb_temp.set_ticklabels((0, 1, 2))
cb_temp.ax.xaxis.set_tick_params(pad=1.)
cax_scale_bar_size.xaxis.set_ticks_position("top")
ax.set_title("Trial Frequency", fontsize=fontsize_legend, x=0.49, y=1.2)
ax.text(69, 83, r'$\times \mathregular{10^{-2}}$', fontsize=fontsize_tick-1.)


### Distribution of stimuli conditions
## rect1_12a: Stimuli Distribution for broad-high trials.
ax = fig_temp.add_axes(rect1_12a)
aspect_ratio=1.
plt.imshow(density_distribution_broad_high_all, extent=(0.,100.,0.,100.), interpolation='nearest', cmap='BuPu', aspect=aspect_ratio, origin='lower', vmin=0., vmax=np.max(density_distribution_broad_high_all))
ax.plot([0,100], [0,100], color='k', alpha=0.8, ls='--', lw=1.)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([25., 50., 75.])
ax.set_yticks([25., 50., 75.])
ax.set_xlim([25., 75.])
ax.set_ylim([25., 75.])
ax.tick_params(direction='out', pad=0.75)
ax.set_xlabel('Mean Broad\nEvidence', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylabel('Mean Narrow\nEvidence', fontsize=fontsize_legend, labelpad=1.)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cb_temp = plt.colorbar(ticks=[0., 0.01, 0.02], cax=cax_scale_bar_size, orientation='horizontal')
cb_temp.set_ticklabels((0, 1, 2))
cb_temp.ax.xaxis.set_tick_params(pad=1.)
cax_scale_bar_size.xaxis.set_ticks_position("top")
ax.set_title("Trial Frequency", fontsize=fontsize_legend, x=0.49, y=1.2)
ax.text(73, 83, r'$\times \mathregular{10^{-2}}$', fontsize=fontsize_tick-1.)


### Distribution of stimuli conditions
## rect1_13a: Stimuli Distribution for balanced/ equal-mean trials.
ax = fig_temp.add_axes(rect1_13a)
aspect_ratio=1.
plt.imshow(density_distribution_NB_balanced_all, extent=(0.,100.,0.,100.), interpolation='nearest', cmap='BuPu', aspect=aspect_ratio, origin='lower', vmin=0., vmax=np.max(density_distribution_NB_balanced_all))
ax.plot([0,100], [0,100], color='k', alpha=0.8, ls='--', lw=1.)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([25., 50., 75.])
ax.set_yticks([25., 50., 75.])
ax.set_xlim([25., 75.])
ax.set_ylim([25., 75.])
ax.tick_params(direction='out', pad=0.75)
ax.set_xlabel('Mean Broad\nEvidence', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylabel('Mean Narrow\nEvidence', fontsize=fontsize_legend, labelpad=1.)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cb_temp = plt.colorbar(ticks=[0., 0.01, 0.02], cax=cax_scale_bar_size, orientation='horizontal')
cb_temp.set_ticklabels((0, 1, 2))
cb_temp.ax.xaxis.set_tick_params(pad=1.)
cax_scale_bar_size.xaxis.set_ticks_position("top")
ax.set_title("Trial Frequency", fontsize=fontsize_legend, x=0.49, y=1.2)
ax.text(69, 83, r'$\times \mathregular{10^{-2}}$', fontsize=fontsize_tick-1.)


## rect1_21: Accuracy with narrow/Broad Correct mean (monkey A)
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
ax.bar([0, 1], ENB_bars_A_non_drug[1:], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_A_non_drug[1:], ecolor='k', color=color_choice_bar, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(9.4,4.75))
ax.scatter([0.4,1.4, 0.9], [0.54,0.54,1.01], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.4,1.4], [0.96,0.96], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Accuracy', fontsize=fontsize_legend)
ax.set_xlim([0,1+bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width/2.-0.0, 1+bar_width/2.+0.0])
ax.xaxis.set_ticklabels(['Narrow\nCorrect', 'Broad\nCorrect'])
ax.set_yticks([0., 0.5, 1.])
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")


## rect1_22: broad preference with equal mean (monkey A)
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
ax.bar([0], ENB_bars_A_non_drug[0], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_A_non_drug[0], ecolor='k', color=color_choice_bar, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(9.4,4.75))
ax.scatter(0.4, 0.54, s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Broad Preference', fontsize=fontsize_legend)
ax.set_xlim([0,bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width/2.])
ax.xaxis.set_ticklabels(['Ambiguous'])
ax.set_yticks([0., 0.5, 1.])
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")


## rect1_23: Accuracy with narrow/Broad Correct mean (monkey H)
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
ax.bar([0, 1], ENB_bars_H_non_drug[1:], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_H_non_drug[1:], ecolor='k', color=color_choice_bar, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(9.4,4.75))
ax.scatter([0.4,1.4, 0.9], [0.54,0.54,0.97], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.4,1.4], [0.92,0.92], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Accuracy', fontsize=fontsize_legend)
ax.set_xlim([0,1+bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width/2.-0.0, 1+bar_width/2.+0.0])
ax.xaxis.set_ticklabels(['Narrow\nCorrect', 'Broad\nCorrect'])
ax.set_yticks([0., 0.5, 1.])
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")


## rect1_24: broad preference with equal mean (monkey H)
ax   = fig_temp.add_axes(rect1_24)
fig_funs.remove_topright_spines(ax)
ax.bar([0], ENB_bars_H_non_drug[0], bar_width, alpha=bar_opacity, yerr=ENB_bars_err_H_non_drug[0], ecolor='k', color=color_choice_bar, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(9.4,4.75))
ax.scatter(0.4, 0.51, s=16., color='k', marker=(5,2), zorder=10, clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Broad Preference', fontsize=fontsize_legend)
ax.set_xlim([0,bar_width])
ax.set_ylim([0.,1.])
ax.set_xticks([bar_width/2.])
ax.xaxis.set_ticklabels(['Ambiguous'])
ax.set_yticks([0., 0.5, 1.])
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")


fig_temp.savefig(path_cwd+'Figure3S1.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
### Figure 4S1: Experimental non-drug data, Regression Trials

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
Reg_bars_mean_var_LRdiff_A_nondrug = np.array([0.00149873713482237, 23.2706129763586, 5.36000877341553])        # Mean, SD, averaged over left and right
Reg_bars_mean_var_LRdiff_H_nondrug = np.array([-0.0606870575047338, 19.3770229683117, 2.67903293994933])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_var_LRdiff_A_nondrug = np.array([0.0273828124050631, 0.507005913709698, 0.321307940864474])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_var_LRdiff_H_nondrug = np.array([0.0196355231057010, 0.329083068643739, 0.221699309692895])        # Mean, SD, averaged over left and right


bar_pos_2by2 = [0., 0.8, 1.8, 2.6]
mean_var_color_list_2by2 = [color_mean_var_beta[0], color_mean_var_beta[0], color_mean_var_beta[1], color_mean_var_beta[1]]
mean_supp_choice_color_list_2by2 = [color_mean_supp_choice[0], color_mean_supp_choice[0], color_mean_supp_choice[1], color_mean_supp_choice[1]]

## Psychometric function generated from all (regression) data                                                           # See MainAnalysisNonDrugDays.m: lines 80-104
## Log-Spaced # the two values right before and after the d_evidence=0 element encodes data with small but not exactly 0 evidence (narrow/broad for before/after). Their d_evidence value is wrong, and is chosen for the most suitable location on ax_0.
d_evidence_A_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_A_list =  np.array([0.0800000000000000, 0.0234741784037559, 0.0480549199084668, 0.134582623509370, 0.188861985472155, 0.299546142208775, 0.344059405940594, 0.473684210526316, 0.420000000000000, 0.512483574244415, 0.500000000000000, 0.616766467065868, 0.701149425287356, 0.721407624633431, 0.810043668122271, 0.861566484517304, 0.897038081805360, 0.949175824175824, 0.962305986696231, 0.970588235294118, 1])  # Log-Spaced.
ErrBar_P_corr_A_list = np.array([0.0542586398650021, 0.0103740243847018, 0.0102313786402976, 0.0140860183532181, 0.0136185111129751, 0.0178164388427016, 0.0236351460022049, 0.0277821845951758, 0.0312153808242027, 0.0181193357771934, 0.117851130197758, 0.0188106540779356, 0.0245382008484921, 0.0242771548088617, 0.0183294048637110, 0.0147393675842240, 0.0114135470936323, 0.00814034510162708, 0.00896818405248063, 0.0129584659592208, 0])
d_evidence_H_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_H_list =  np.array([0.0344827586206897, 0.0355029585798817, 0.0843214756258235, 0.138604651162791, 0.227019498607242, 0.255075022065313, 0.324400564174894, 0.457657657657658, 0.440082644628099, 0.479707792207792, 0.700000000000000, 0.620550161812298, 0.644927536231884, 0.668989547038328, 0.720588235294118, 0.801980198019802, 0.848784194528875, 0.896381578947369, 0.940520446096654, 0.969387755102041, 0.957446808510638])  # Log-Spaced.
ErrBar_P_corr_H_list = np.array([0.0239589080613673, 0.0100652455109222, 0.0100860176877973, 0.0105386730482314, 0.0110544879232228, 0.0129501643449445, 0.0175817589686430, 0.0211475778028564, 0.0225634967454675, 0.0142333357307799, 0.102469507659596, 0.0138024568215189, 0.0203678138942123, 0.0196414965062009, 0.0164064692491009, 0.0125393621833121, 0.00987573188572657, 0.00873973288007844, 0.00832590027140452, 0.0100466827639142, 0.0294424853626240])


# ## Combined Regression figure.                                                                                          # See MainAnalysisNonDrugDays.m: LongAvCOL, LongAvCOLSE
bar_pos_2by2_combined = np.array([0., 1.8, 3.6+0.5, 5.4+0.5, 7.2+0.5, 9.0+0.5, 10.8+0.5])
Reg_bar_pos_combined_model_control = np.array([0., 1., 2.+0.5, 3.+0.5, 4.+0.5, 5.0+0.5, 6.+0.5])
Reg_combined_color_list = [color_mean_var_beta[0], color_mean_var_beta[1], color_mean_var_beta[0], 'grey', 'grey', 'grey', 'grey']

Reg_bars_mean_var_combined_A_nondrug = np.array([0.5*(Reg_bars_A_non_drug[1]-Reg_bars_A_non_drug[3]), 0.5*(Reg_bars_A_non_drug[2]-Reg_bars_A_non_drug[4])])        # Mean, SD, averaged over left and right
Reg_bars_mean_var_combined_H_nondrug = np.array([0.5*(Reg_bars_H_non_drug[1]-Reg_bars_H_non_drug[3]), 0.5*(Reg_bars_H_non_drug[2]-Reg_bars_H_non_drug[4])])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_var_combined_A_nondrug = np.array([0.5*(Reg_bars_err_A_non_drug[1]**2.+Reg_bars_err_A_non_drug[3]**2.)**0.5, 0.5*(Reg_bars_err_A_non_drug[2]**2.+Reg_bars_err_A_non_drug[4]**2.)**0.5])        # Mean, SD, averaged over left and right
Reg_bars_Err_mean_var_combined_H_nondrug = np.array([0.5*(Reg_bars_err_A_non_drug[1]**2.+Reg_bars_err_A_non_drug[3]**2.)**0.5, 0.5*(Reg_bars_err_A_non_drug[2]**2.+Reg_bars_err_A_non_drug[4]**2.)**0.5])        # Mean, SD, averaged over left and right



## Combined Regression figure.                                                                                          # See MainAnalysisNonDrugDays.m: LongAvCOL, LongAvCOLSE
Reg_values_A_nondrug = np.array([-0.683515760487767, 0.857750257881972, -0.362048010685892, 23.9059324499611, 2.31888740296024, -3.39090669582641, -1.44847211860585, 0.630255180927178, -22.6289253237029, -1.07429141923381, 0.428325277524458])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_A_nondrug = np.array([0.340091706079433, 0.161916759821178, 0.164125073839184, 0.821056692528016, 0.336519259278951, 0.340519605884390, 0.162373945530126, 0.166905816093258, 0.796315325484657, 0.345967119320088, 0.344950736338556])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_H_nondrug = np.array([-0.239322222821878, 0.829103333194566, -0.383911778486991, 18.0439214688905, 1.70032339566670, -0.408721514401331, -0.768486234789177, 0.282507719580400, -18.7029029533569, -1.01972144983357, 0.471733204034387])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_H_nondrug = np.array([0.241190422148207, 0.112814374931893, 0.113919221581905, 0.538679697853817, 0.238343030325939, 0.236658514189981, 0.113454542428770, 0.117509382517541, 0.550572103603052, 0.243597748074078, 0.249246891028432])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_bars_L_combined_A_nondrug = np.array([Reg_bars_A_non_drug_rearanged[0], Reg_bars_A_non_drug_rearanged[2], Reg_values_A_nondrug[3], Reg_values_A_nondrug[4], Reg_values_A_nondrug[5], Reg_values_A_nondrug[1], Reg_values_A_nondrug[2]])        # Mean, SD, Mean, Max, Min, First, Last, averaged over left and right
Reg_bars_R_combined_A_nondrug = np.array([Reg_bars_A_non_drug_rearanged[1], Reg_bars_A_non_drug_rearanged[3], -Reg_values_A_nondrug[8], -Reg_values_A_nondrug[9], -Reg_values_A_nondrug[10], -Reg_values_A_nondrug[6], -Reg_values_A_nondrug[7]])        # Mean, SD, Mean, Max, Min, First, Last, averaged over left and right
Reg_bars_L_combined_H_nondrug = np.array([Reg_bars_H_non_drug_rearanged[0], Reg_bars_H_non_drug_rearanged[2], Reg_values_H_nondrug[3], Reg_values_H_nondrug[4], Reg_values_H_nondrug[5], Reg_values_H_nondrug[1], Reg_values_H_nondrug[2]])        # Mean, SD, Mean, Max, Min, First, Last, averaged over left and right
Reg_bars_R_combined_H_nondrug = np.array([Reg_bars_H_non_drug_rearanged[1], Reg_bars_H_non_drug_rearanged[3], -Reg_values_H_nondrug[8], -Reg_values_H_nondrug[9], -Reg_values_H_nondrug[10], -Reg_values_H_nondrug[6], -Reg_values_H_nondrug[7]])        # Mean, SD, Mean, Max, Min, First, Last, averaged over left and right
Reg_bars_combined_H_nondrug = np.array([Reg_bars_H_non_drug_rearanged[0], Reg_bars_H_non_drug_rearanged[1], Reg_bars_H_non_drug_rearanged[2], Reg_bars_H_non_drug_rearanged[3], Reg_values_H_nondrug[3], Reg_values_H_nondrug[8], Reg_values_H_nondrug[4], Reg_values_H_nondrug[9], Reg_values_H_nondrug[5], Reg_values_H_nondrug[10], Reg_values_H_nondrug[1], Reg_values_H_nondrug[6], Reg_values_H_nondrug[2], Reg_values_H_nondrug[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_Err_L_combined_A_nondrug = np.array([Reg_bars_err_A_non_drug_rearanged[0], Reg_bars_err_A_non_drug_rearanged[2], Reg_values_errbar_A_nondrug[3], Reg_values_errbar_A_nondrug[4], Reg_values_errbar_A_nondrug[5], Reg_values_errbar_A_nondrug[1], Reg_values_errbar_A_nondrug[2]])        # Mean, SD, Mean, Max, Min, First, Last, averaged over left and right
Reg_bars_Err_R_combined_A_nondrug = np.array([Reg_bars_err_A_non_drug_rearanged[1], Reg_bars_err_A_non_drug_rearanged[3], Reg_values_errbar_A_nondrug[8], Reg_values_errbar_A_nondrug[9], Reg_values_errbar_A_nondrug[10], Reg_values_errbar_A_nondrug[6], Reg_values_errbar_A_nondrug[7]])        # Mean, SD, Mean, Max, Min, First, Last, averaged over left and right
Reg_bars_Err_L_combined_H_nondrug = np.array([Reg_bars_err_H_non_drug_rearanged[0], Reg_bars_err_H_non_drug_rearanged[2], Reg_values_errbar_H_nondrug[3], Reg_values_errbar_H_nondrug[4], Reg_values_errbar_H_nondrug[5], Reg_values_errbar_H_nondrug[1], Reg_values_errbar_H_nondrug[2]])        # Mean, SD, Mean, Max, Min, First, Last, averaged over left and right
Reg_bars_Err_R_combined_H_nondrug = np.array([Reg_bars_err_H_non_drug_rearanged[1], Reg_bars_err_H_non_drug_rearanged[3], Reg_values_errbar_H_nondrug[8], Reg_values_errbar_H_nondrug[9], Reg_values_errbar_H_nondrug[10], Reg_values_errbar_H_nondrug[6], Reg_values_errbar_H_nondrug[7]])        # Mean, SD, Mean, Max, Min, First, Last, averaged over left and right
bar_pos_2by2_combined = np.array([0., 1.8, 3.6+0.5, 5.4+0.5, 7.2+0.5, 9.0+0.5, 10.8+0.5])
Reg_bar_pos_combined_model_control = np.array([0., 1., 2.+0.5, 3.+0.5, 4.+0.5, 5.0+0.5, 6.+0.5])
Reg_combined_color_list = [color_mean_var_beta[0], color_mean_var_beta[1], color_mean_var_beta[0], 'grey', 'grey', 'grey', 'grey']

Reg_bars_LRsep_A_nondrug = np.array([Reg_values_A_nondrug[3], -Reg_values_A_nondrug[8], Reg_values_A_nondrug[4], -Reg_values_A_nondrug[9], Reg_values_A_nondrug[5], -Reg_values_A_nondrug[10], Reg_values_A_nondrug[1], -Reg_values_A_nondrug[6], Reg_values_A_nondrug[2], -Reg_values_A_nondrug[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_err_LRsep_A_nondrug = np.array([Reg_values_errbar_A_nondrug[3], Reg_values_errbar_A_nondrug[8], Reg_values_errbar_A_nondrug[4], Reg_values_errbar_A_nondrug[9], Reg_values_errbar_A_nondrug[5], Reg_values_errbar_A_nondrug[10], Reg_values_errbar_A_nondrug[1], Reg_values_errbar_A_nondrug[6], Reg_values_errbar_A_nondrug[2], Reg_values_errbar_A_nondrug[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_LRsep_H_nondrug = np.array([Reg_values_H_nondrug[3], -Reg_values_H_nondrug[8], Reg_values_H_nondrug[4], -Reg_values_H_nondrug[9], Reg_values_H_nondrug[5], -Reg_values_H_nondrug[10], Reg_values_H_nondrug[1], -Reg_values_H_nondrug[6], Reg_values_H_nondrug[2], -Reg_values_H_nondrug[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_err_LRsep_H_nondrug = np.array([Reg_values_errbar_H_nondrug[3], Reg_values_errbar_H_nondrug[8], Reg_values_errbar_H_nondrug[4], Reg_values_errbar_H_nondrug[9], Reg_values_errbar_H_nondrug[5], Reg_values_errbar_H_nondrug[10], Reg_values_errbar_H_nondrug[1], Reg_values_errbar_H_nondrug[6], Reg_values_errbar_H_nondrug[2], Reg_values_errbar_H_nondrug[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
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
psychometric_params_A_non_drug = [0.063355560693220,1.115972329440189,0.015468311403056]
psychometric_params_H_non_drug = [0.075084388209508,1.036168328976971,0.009766466546951]


## Extract number distribution of stimuli, for Standard/Regression trials.                                              # See MainAnalysisNonDrugDays.m: lines 137-187
n_distribution_Regression_monkey_avg = np.loadtxt('Data/Stim_Distribution/dx=2/n_distribution_regression_avg.txt', delimiter=',')              # axis is (narrow, broad)
density_distribution_Regression_all = n_distribution_Regression_monkey_avg / np.sum(n_distribution_Regression_monkey_avg)
n_SD_distribution_Regression_monkey_avg = np.loadtxt('Data/Stim_Distribution/dx=1/n_SD_distribution_regression_avg.txt', delimiter=',')              # axis is (narrow, broad)
density_SD_distribution_Regression_all = n_SD_distribution_Regression_monkey_avg / np.sum(n_SD_distribution_Regression_monkey_avg)










## Define subfigure domain.
figsize = (max2,0.55*max2)

width1_11=0.13; width1_21=width1_11; width1_12=0.13; width1_22=width1_12; width1_14=0.28; width1_24=width1_14; width1_13=width1_14*1.8/4.8 *0.8; width1_23=width1_13
x1_11=0.065; x1_12=x1_11 + width1_12 + 1.25*xbuf0; x1_13=x1_12 + width1_13 + 1.15*xbuf0; x1_14=x1_13 + width1_13 + 0.7*xbuf0; x1_21=x1_11; x1_22=x1_12; x1_23=x1_13; x1_24=x1_14
height1_11=0.35; height1_12=0.31; height1_13=height1_12+0.25*ybuf0; height1_14=height1_13; height1_21=height1_11; height1_22=height1_12; height1_23=height1_22+0.25*ybuf0; height1_24=height1_23
y1_11=0.57; y1_12=y1_11+0.872*ybuf0; y1_13=y1_12-0.32*ybuf0; y1_14=y1_13; y1_21 = y1_11 - height1_21 - 1.8*ybuf0; y1_22=y1_21+0.875*ybuf0; y1_23=y1_22-0.32*ybuf0; y1_24=y1_23


rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_12_0 = [x1_12, y1_12, width1_12*0.05, height1_12]
rect1_12 = [x1_12+width1_12*0.2, y1_12, width1_12*(1-0.2), height1_12]
rect1_22_0 = [x1_22, y1_22, width1_22*0.05, height1_22]
rect1_22 = [x1_22+width1_22*0.2, y1_22, width1_22*(1-0.2), height1_22]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_14 = [x1_14, y1_14, width1_14, height1_14]
rect1_24 = [x1_24, y1_24, width1_24, height1_24]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.01, 0.87, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.062, 0.96, 'Regular Trials', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.005 + x1_12 - x1_11, 0.94, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.009 + x1_13 - x1_11, 0.94, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.005 + x1_14 - x1_11, 0.94, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.02 + x1_12 - x1_11, 0.85, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k')
fig_temp.text(-0.02 + x1_12 - x1_11, 0.85 + y1_21 - y1_11, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k')
fig_temp.text(0.01, 0.87 + y1_21 - y1_11, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.005 + x1_22 - x1_21, 0.94 + y1_21 - y1_11, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.009 + x1_23 - x1_21, 0.94 + y1_21 - y1_11, 'G', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.005 + x1_24 - x1_21, 0.94 + y1_21 - y1_11, 'H', fontsize=fontsize_fig_label, fontweight='bold')




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
ax.text(76., 77.8, r'$\times \mathregular{10^{-3}}$', fontsize=fontsize_tick-1.)



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
ax.set_ylabel('Evidence SD\n(Lower SD)', fontsize=fontsize_legend, labelpad=1.)
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
ax.errorbar( d_evidence_A_list[12:],    P_corr_A_list[12:], ErrBar_P_corr_A_list[12:], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Corr.' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_A_list[1:9], 1.-P_corr_A_list[1:9], ErrBar_P_corr_A_list[1:9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], fmt='.', zorder=3, clip_on=False, label='Lower SD Corr.', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_A_list[11], P_corr_A_list[11], ErrBar_P_corr_A_list[11], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_A_list[9], 1.-P_corr_A_list[9], ErrBar_P_corr_A_list[9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], marker='.', zorder=3, clip_on=False                      , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_A_non_drug, x_list_psychometric), color=color_NB[1], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_A_non_drug, -x_list_psychometric), color=color_NB[0], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_A_non_drug, x0_psychometric), s=15., color=color_NB[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_A_non_drug, -x0_psychometric), s=15., color=color_NB[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
y_shift_spines = -0.0823
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_NB[1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=color_NB[0], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.37,-0.11), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0., labelspacing=0.2)
for color,text,item in zip([color_NB[1], color_NB[0]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)


## rect1_22: Psychometric function (over dx_broad, or dx_corr ?), Monkey H
ax_0   = fig_temp.add_axes(rect1_22_0)
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
# Log-Spaced
ax.errorbar( d_evidence_H_list[12:],    P_corr_H_list[12:], ErrBar_P_corr_H_list[12:], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Corr.' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_H_list[:9], 1.-P_corr_H_list[:9], ErrBar_P_corr_H_list[:9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], fmt='.', zorder=3, clip_on=False, label='Lower SD Corr.', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax_0.errorbar(d_evidence_H_list[11], P_corr_H_list[11], ErrBar_P_corr_H_list[11], color=color_NB[1], markerfacecolor=color_NB[1], ecolor=color_NB[1], marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
tmp = ax_0.errorbar(-d_evidence_H_list[9], 1.-P_corr_H_list[9], ErrBar_P_corr_H_list[9], color=color_NB[0], markerfacecolor=color_NB[0], ecolor=color_NB[0], marker='.', zorder=3, clip_on=False                      , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_H_non_drug, x_list_psychometric), color=color_NB[1], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_H_non_drug, -x_list_psychometric), color=color_NB[0], ls='-', clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_H_non_drug, x0_psychometric), s=15., color=color_NB[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_H_non_drug, -x0_psychometric), s=15., color=color_NB[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
y_shift_spines = -0.0823
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_NB[1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=color_NB[0], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.37,-0.1), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0., labelspacing=0.2)
for color,text,item in zip([color_NB[1], color_NB[0]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)



### Mean and Var only. Averaged across L/R
## rect1_13: Regression Analysis, Monkey A
ax   = fig_temp.add_axes(rect1_13)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(np.arange(2), Reg_bars_mean_var_LRdiff_A_nondrug[1:], bar_width, yerr=Reg_bars_Err_mean_var_LRdiff_A_nondrug[1:], ecolor='k', alpha=1, color=Reg_combined_color_list[0:2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.scatter([0.4,1.4], [24.8,7.], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(Reg_bars_mean_var_LRdiff_A_nondrug[1:])-1+bar_width])
ax.set_ylim([0.,25.])
ax.set_xticks(np.arange(len(Reg_bars_mean_var_LRdiff_A_nondrug[1:]))+bar_width/2. + [-0.1,0.1])
ax.xaxis.set_ticklabels(['Mean\nEvidence', 'Evidence\nSD'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0., 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))

## rect1_23: Regression Analysis, Monkey H
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(np.arange(len(Reg_bars_mean_var_LRdiff_H_nondrug[1:])), Reg_bars_mean_var_LRdiff_H_nondrug[1:], bar_width, yerr=Reg_bars_Err_mean_var_LRdiff_H_nondrug[1:], ecolor='k', alpha=1, color=Reg_combined_color_list[0:2], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
ax.scatter([0.4,1.4], [21.2,4.2], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
for bar in bar_temp:
    bar.set_edgecolor("k")
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(Reg_bars_mean_var_LRdiff_H_nondrug[1:])-1+bar_width])
ax.set_ylim([0.,25.])
ax.set_xticks(np.arange(len(Reg_bars_mean_var_LRdiff_H_nondrug[1:]))+bar_width/2. + [-0.1,0.1])
ax.xaxis.set_ticklabels(['Mean\nEvidence', 'Evidence\nSD'])#, 'Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0., 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))


## rect1_14: Regression Analysis. Mean/Max/Min/First/Last model
ax   = fig_temp.add_axes(rect1_14)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(x_LRsep_list[:-2], Reg_bars_LRsep_A_nondrug, bar_width/2., yerr=Reg_bars_err_LRsep_A_nondrug, ecolor='k', alpha=1, color=color_LRsep_list, clip_on=False, align='edge', error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
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
ax.set_xlim([0,0.5*len(Reg_bars_LRsep_A_nondrug)-1+bar_width])
ax.set_ylim([0.,25.])
ax.set_xticks(np.arange(len(Reg_bars_LRsep_A_nondrug)/2)+bar_opacity/2.)
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0., 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(axis='x', direction='out', pad=12.5)
ax.tick_params(axis='y', direction='out')
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend_bars = [ Patch(facecolor='grey', edgecolor='k', label='Left'), \
                Patch(facecolor='grey', edgecolor='k', hatch='////', label='Right')]
ax.legend(handles=legend_bars, loc=(0.65,0.5), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)


## rect1_24: Regression Analysis. Mean/Max/Min/First/Last model
ax   = fig_temp.add_axes(rect1_24)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(x_LRsep_list[:-2], Reg_bars_LRsep_H_nondrug, bar_width/2., yerr=Reg_bars_err_LRsep_H_nondrug, ecolor='k', alpha=1, color=color_LRsep_list, clip_on=False, align='edge', error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2., lw=1)
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
ax.set_xlim([0,0.5*len(Reg_bars_LRsep_H_nondrug)-1+bar_width])
ax.set_ylim([0.,25.])
ax.set_xticks(np.arange(len(Reg_bars_LRsep_H_nondrug)/2)+bar_opacity/2.)
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0., 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(axis='x', direction='out', pad=2.)
ax.tick_params(axis='y', direction='out')
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend_bars = [ Patch(facecolor='grey', edgecolor='k', label='Left'), \
                Patch(facecolor='grey', edgecolor='k', hatch='////', label='Right')]
ax.legend(handles=legend_bars, loc=(0.65,0.5), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)


fig_temp.savefig(path_cwd+'Figure4S1.pdf')    #Finally save fig


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
### Figure 5S1: Model non-drug day data (NB, reg)

## Regression model using first/last/mean/Max/Min:                                                                   # See MainAnalysisNonDrugDays_NL.m: LongAvCOL, LongAvCOLSE
Reg_values_control = np.array([-0.563702490200418, 0.566544476994033, -1.79664446212739, 16.3607962599429, 2.35054221646502, -1.13961920728613, -0.544626694721213, 1.68487206917613, -15.2989874253047, -2.31071785001726, 1.25431512790789])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_control = np.array([0.0943938239603895, 0.0482762181928213, 0.0492635194160946, 0.229530157469093, 0.0994137270628964, 0.0987580748646614, 0.0353934238842000, 0.0364433340211602, 0.173398020098838, 0.0838030187014566, 0.0851095486830076])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_bar_pos_combined_model_control = np.array([0., 1., 2.+0.5, 3.+0.5, 4.+0.5, 5.0+0.5, 6.+0.5])

Reg_bars_LRsep_model_control = np.array([Reg_values_control[3], -Reg_values_control[8], Reg_values_control[4], -Reg_values_control[9], Reg_values_control[5], -Reg_values_control[10], Reg_values_control[1], -Reg_values_control[6], Reg_values_control[2], -Reg_values_control[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_err_LRsep_model_control = np.array([Reg_values_err_control[3], Reg_values_err_control[8], Reg_values_err_control[4], Reg_values_err_control[9], Reg_values_err_control[5], Reg_values_err_control[10], Reg_values_err_control[1], Reg_values_err_control[6], Reg_values_err_control[2], Reg_values_err_control[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right

### Psychometric function fit                                                                                           # See figure_psychometric_function_fit.py, esp lines 533-638
def Psychometric_fit_D(params_pm, pm_fit2, x_list):
    prob_corr_fit = 0.5 + 0.5*np.sign(x_list+params_pm[2])*(1. - np.exp(-(np.abs(x_list+params_pm[2])/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    to_min = -sum(np.log(prob_corr_fit)*pm_fit2) - sum(np.log(1.-prob_corr_fit)*(1.-pm_fit2))                                                          # Maximum Likelihood Estimator
    return to_min
def Psychometric_function_D(params_pm, x_list):
    prob_corr_fit = 0.5 + 0.5*np.sign(x_list+params_pm[2])*(1. - np.exp(-(np.abs(x_list+params_pm[2])/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    return prob_corr_fit



## Define subfigure domain.
figsize = (max15,0.4*max15)

width1_11 = 0.17; width1_12 = 0.6
x1_11 = 0.07; x1_12 = x1_11 + width1_11 + 1.*xbuf0
height1_11 = 0.7; height1_12 = height1_11
y1_11 = 0.2; y1_12 = y1_11




rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.01, 0.92, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.005+x1_12-x1_11, 0.92, 'B', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.

## rect1_11: E/I perturbation schematics

## rect1_12: Regression Analysis. Mean/SD/Max/Min/First/Last model
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(x_LRsep_list[:-2], Reg_bars_LRsep_model_control, bar_width/2., yerr=Reg_bars_err_LRsep_model_control, ecolor='k', alpha=1, color=color_LRsep_list, clip_on=False, align='edge', error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
for b in bar_temp.errorbar[1]:
    b.set_clip_on(False)
for b in bar_temp.errorbar[2]:
    b.set_clip_on(False)
for bar in bar_temp:
    bar.set_edgecolor("k")
for i_bar in range(len(bar_temp)):
    if (i_bar % 2)==1:
        bar_temp[i_bar].set_hatch('////')
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,0.5*len(Reg_bars_LRsep_model_control)-1+bar_width])
ax.set_ylim([0.,17.])
ax.set_xticks(np.arange(len(Reg_bars_LRsep_model_control)/2)+bar_opacity/2.)
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 15.])
ax.set_yticklabels([0, 0.15])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(axis='x', direction='out', pad=10.5)
ax.tick_params(axis='y', direction='out')
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend_bars = [ Patch(facecolor='grey', edgecolor='k', label='Left'), \
                Patch(facecolor='grey', edgecolor='k', hatch='////', label='Right')]
ax.legend(handles=legend_bars, loc=(0.65,0.5), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)

fig_temp.savefig(path_cwd+'Figure5S1.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
### Figure 6S1: Mean-Field Model

## Regression model using first/last/mean/Max/Min:                                                                   # See MainAnalysisNonDrugDays_NL.m: LongAvCOL, LongAvCOLSE
Reg_values_modelMF_control = np.array([-0.824720688164682, 5.10483829134541, -2.66591540321193, 24.1873730068665, 2.08891331546669, -1.01219322534966, -5.01144047063643, 2.51430404713043, -22.9666587623899, -1.89846228936625, 1.26737380871806])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_modelMF_control = np.array([0.121747922543159, 0.0697668932702089, 0.0658972219338625, 0.315322166453290, 0.131146427525639, 0.130202504082441, 0.0529733080300937, 0.0491463731005067, 0.244017265520443, 0.108318615358950, 0.108897678051422])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_bars_LRsep_modelMF_control = np.array([Reg_values_modelMF_control[3], -Reg_values_modelMF_control[8], Reg_values_modelMF_control[4], -Reg_values_modelMF_control[9], Reg_values_modelMF_control[5], -Reg_values_modelMF_control[10], Reg_values_modelMF_control[1], -Reg_values_modelMF_control[6], Reg_values_modelMF_control[2], -Reg_values_modelMF_control[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_err_LRsep_modelMF_control = np.array([Reg_values_err_modelMF_control[3], Reg_values_err_modelMF_control[8], Reg_values_err_modelMF_control[4], Reg_values_err_modelMF_control[9], Reg_values_err_modelMF_control[5], Reg_values_err_modelMF_control[10], Reg_values_err_modelMF_control[1], Reg_values_err_modelMF_control[6], Reg_values_err_modelMF_control[2], Reg_values_err_modelMF_control[7]])        # Mean, SD, First, Last, Max, Min), averaged over left and right
Reg_bars_combined_modelMF_control = np.array([Reg_bars_LRdiff_modelMF_control[1], Reg_bars_LRdiff_modelMF_control[2], 0.5*(Reg_values_modelMF_control[3]-Reg_values_modelMF_control[8]), 0.5*(Reg_values_modelMF_control[4]-Reg_values_modelMF_control[9]), 0.5*(Reg_values_modelMF_control[5]-Reg_values_modelMF_control[10]), 0.5*(Reg_values_modelMF_control[1]-Reg_values_modelMF_control[6]), 0.5*(Reg_values_modelMF_control[2]-Reg_values_modelMF_control[7])])        # Mean, SD, Mean, Max, Min, First, Last), averaged over left and right
Reg_bars_err_combined_modelMF_control = np.array([Reg_bars_err_LRdiff_modelMF_control[1], Reg_bars_err_LRdiff_modelMF_control[2], 0.5*(Reg_values_err_modelMF_control[3]**2+Reg_values_err_modelMF_control[8]**2)**0.5, 0.5*(Reg_values_err_modelMF_control[4]**2+Reg_values_err_modelMF_control[9]**2)**0.5, 0.5*(Reg_values_err_modelMF_control[5]**2+Reg_values_err_modelMF_control[10]**2)**0.5, 0.5*(Reg_values_err_control[1]**2+Reg_values_err_modelMF_control[6]**2)**0.5, 0.5*(Reg_values_err_modelMF_control[2]**2+Reg_values_err_modelMF_control[7]**2)**0.5])        # Mean, SD, Mean, Max, Min, First, Last), averaged over left and right
Reg_bar_pos_combined_modelMF_control = np.array([0., 1., 2.+0.5, 3.+0.5, 4.+0.5, 5.0+0.5, 6.+0.5])


## Define subfigure domain.
figsize = (max15,0.4*max15)

width1_11 = 0.17; width1_12 = 0.6
x1_11 = 0.07; x1_12 = x1_11 + width1_11 + 1.*xbuf0
height1_11 = 0.7; height1_12 = height1_11
y1_11 = 0.2; y1_12 = y1_11


rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.01, 0.92, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.02+x1_12-x1_11, 0.92, 'B', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.

## rect1_11: Mean-Field model schematics


## rect1_12: Regression Analysis. Mean/SD/Max/Min/First/Last model
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax)
bar_temp = ax.bar(x_LRsep_list[:-2], Reg_bars_LRsep_modelMF_control, bar_width/2., yerr=Reg_bars_err_LRsep_modelMF_control, ecolor='k', alpha=1, color=color_LRsep_list, clip_on=False, align='edge', error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
for b in bar_temp.errorbar[1]:
    b.set_clip_on(False)
for b in bar_temp.errorbar[2]:
    b.set_clip_on(False)
for bar in bar_temp:
    bar.set_edgecolor("k")
for i_bar in range(len(bar_temp)):
    if (i_bar % 2)==1:
        bar_temp[i_bar].set_hatch('////')
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,0.5*len(Reg_bars_LRsep_modelMF_control)-1+bar_width])
ax.set_ylim([0.,25.])
ax.set_xticks(np.arange(len(Reg_bars_LRsep_modelMF_control)/2)+bar_opacity/2.)
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0., 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(axis='x', direction='out', pad=9.5)
ax.tick_params(axis='y', direction='out')
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.spines['bottom'].set_position(('zero'))
legend_bars = [ Patch(facecolor='grey', edgecolor='k', label='Left'), \
                Patch(facecolor='grey', edgecolor='k', hatch='////', label='Right')]
ax.legend(handles=legend_bars, loc=(0.65,0.5), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)

fig_temp.savefig(path_cwd+'Figure6S1.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
########################################################################################################################
### Figure 7: Model E/I Perturbsation & Upstream Sensory Deficit

### Regression models                                                                                                   # See MainAnalysisNonDrugDays_NL.m: VarAndLocalWinsBetasCollapsed, VarAndLocalWinsErrCollapsed
## First, Last, Mean, Max, Min                                                                                          # See MainAnalysisNonDrugDays_NL.m: LongAvCOL, LongAvCOLSE
Reg_values_control = np.array([-0.799975795441590, 0.496091788783012, -1.95147482506435, 16.6435510742991, 2.51869160833734, -0.775652462213961, -0.563937819902963, 1.72726445867963, -15.5439816070891, -2.21963076055435, 1.23313129184529])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_lowered_EI = np.array([0.190003506885178, -0.235599553981222, -1.07015666540429, 10.7143801356569, 1.73536901552416, -1.05480870621277, 0.306620167816446, 0.963593249348418, -11.2866203206244, -1.66782027032213, 1.00893647865141])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_elevated_EI = np.array([-0.317237265125046, 3.57025700473867, -1.38041746171278, 9.42971561503303, 1.13803974828794, -0.250209186493841, -3.37711415320973, 1.32572992661215, -10.3266097437763, -0.449125721821575, 0.906379587234502])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_upstream_deficit = np.array([0.0560311909515569, -0.160674814456601, -0.881991504397344, 10.7179400646621, 1.29444976079686, -0.873468733846600, 0.220992880834769, 0.986593266733566, -11.0766687429408, -1.30012991374204, 0.875693989956842])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)

Reg_values_err_control = np.array([0.155259207050030, 0.0814567425789707, 0.0833418952000110, 0.380405186170427, 0.167947586495689, 0.165399951383634, 0.0599810684092175, 0.0618760698960268, 0.290581409807592, 0.139654535033320, 0.142078616907945])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_lowered_EI = np.array([0.143136035928441, 0.0746855406809858, 0.0753280128921357, 0.336090493254340, 0.153561120959519, 0.151917942379994, 0.0553331500596443, 0.0560195709313174, 0.256989632519849, 0.129163640092481, 0.131170489417240])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_elevated_EI = np.array([0.153826504023430, 0.0842572769780524, 0.0817096933761490, 0.359559773477624, 0.165851584561131, 0.164103971568001, 0.0627951316515934, 0.0602985276382010, 0.273416473281369, 0.138401242084907, 0.139809194108627])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_upstream_deficit = np.array([0.142186429790058, 0.0742174710133451, 0.0747323951036313, 0.334075001924850, 0.152391678290145, 0.150983011904768, 0.0547667934757646, 0.0555336023725431, 0.254172384516366, 0.128201539812090, 0.129835392722852])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)



#### Fitted to monkey A lapse rate.
## First, Last, Mean, Max, Min                                                                                          # See MainAnalysisNonDrugDays_NL.m: LongAvCOL, LongAvCOLSE
Reg_values_control_fitted_A_lapse = np.array([-0.524060393439731, 0.293449791740516, -1.19352596986781, 10.7009366319403, 1.68018051780375, -0.572509326240899, -0.354405524626893, 1.06738931534072, -9.78946300604857, -1.54672683384728, 0.722610973866920])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_lowered_EI_fitted_A_lapse = np.array([0.118055157236285, -0.159108711460770, -0.738474205855703, 7.45178178491451, 1.23107582687227, -0.736691032589832, 0.214224628884342, 0.667653292381897, -7.83403163204941, -1.18194675059403, 0.700624628801509])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_elevated_EI_fitted_A_lapse = np.array([-0.190775603346266, 2.31396485549950, -0.897155350847555, 6.11482923258001, 0.743673357855439, -0.172061543739482, -2.21559999585935, 0.856217970434154, -6.70273053282033, -0.286017284486571, 0.584284551366613])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_upstream_deficit_fitted_A_lapse = np.array([5.97569267393207e-05, -0.132872322584030, -0.610622977306144, 7.90829531366675, 0.845397891602157, -0.723334350750050, 0.173652253955829, 0.669154358541933, -7.84874607097209, -0.950561544630342, 0.625649878160126])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)

Reg_values_err_control_fitted_A_lapse = np.array([0.142847161624535, 0.0742469947935042, 0.0752165015578474, 0.335241932485171, 0.152598531940596, 0.151050912830013, 0.0546271072712374, 0.0557551340508917, 0.251611408734459, 0.128689916108949, 0.130017972824943])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_lowered_EI_fitted_A_lapse = np.array([0.137060587264945, 0.0712391922245377, 0.0716123418580269, 0.315323697388666, 0.146405446451021, 0.145102008186877, 0.0527133949828616, 0.0531453685708457, 0.238032121608974, 0.124002109056757, 0.125183256381088])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_elevated_EI_fitted_A_lapse = np.array([0.142065343947464, 0.0755278022978541, 0.0746371091593121, 0.326029430941723, 0.152264844773893, 0.150920940863554, 0.0558250820199932, 0.0550628641849008, 0.244909944703269, 0.128304206875966, 0.129080797268048])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_upstream_deficit_fitted_A_lapse = np.array([0.136810029453187, 0.0711582742745153, 0.0714703309026419, 0.315694664988331, 0.146082199646810, 0.144961233460030, 0.0525073485198762, 0.0529656099926241, 0.237209031737837, 0.123706795404299, 0.124659459580132])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)


#### Fitted to monkey H lapse rate.
## First, Last, Mean, Max, Min                                                                                          # See MainAnalysisNonDrugDays_NL.m: LongAvCOL, LongAvCOLSE
Reg_values_control_fitted_H_lapse = np.array([-0.627862266598890, 0.358270945283961, -1.42748028308022, 12.9203111968442, 1.98851485201843, -0.700941049117172, -0.430766093034292, 1.28231805887081, -11.8052181533779, -1.84575732810298, 0.871677888402864])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_lowered_EI_fitted_H_lapse = np.array([0.156973474131819, -0.190517956573965, -0.873127531778522, 8.72489985018739, 1.42548217749694, -0.863601274038313, 0.247942429512494, 0.781458934337603, -9.16942843660793, -1.37706397137125, 0.800662497428994])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_elevated_EI_fitted_H_lapse = np.array([-0.247607883150358, 2.78569949253798, -1.06842432131352, 7.32396509956650, 0.887046570346137, -0.197010565931470, -2.64303840944531, 1.02993620125854, -8.01234360283582, -0.352514307197746, 0.702600738530501])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_upstream_deficit_fitted_H_lapse = np.array([-0.0121736957391923, -0.133830809324166, -0.758976615479105, 9.21321803012014, 0.918814666123908, -0.872167489234618, 0.213842293619169, 0.776991171815005, -8.99149895045293, -1.10197869755712, 0.713643925711240])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)

Reg_values_err_control_fitted_H_lapse = np.array([0.147310057837789, 0.0767149793754011, 0.0779442254222985, 0.350741124594866, 0.157686968349207, 0.155976076891348, 0.0564273497279154, 0.0578567340641490, 0.264444043353272, 0.132514754479815, 0.134158951088744])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_lowered_EI_fitted_H_lapse = np.array([0.139297924897953, 0.0725124858752832, 0.0729881289889842, 0.322941251038777, 0.149042823075995, 0.147618911871303, 0.0536776434522146, 0.0541993358970680, 0.244877068349006, 0.125890028688845, 0.127374572258127])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_elevated_EI_fitted_H_lapse = np.array([0.146204508793301, 0.0785437262287538, 0.0771269832656815, 0.337804925178140, 0.157082912135844, 0.155584289476061, 0.0581703015331351, 0.0568916829323742, 0.254726370589544, 0.131840236276928, 0.132843331051045])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_upstream_deficit_fitted_H_lapse = np.array([0.138710401193466, 0.0722492493950381, 0.0726806060942897, 0.322758267192545, 0.148333058363656, 0.147151666317276, 0.0533086376480508, 0.0538525459439464, 0.242942922654811, 0.125284602487200, 0.126460244203009])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)


## Define subfigure domain.
figsize = (max15,1.*max15)

width1_11 = 0.17; width1_12 = 0.6; width1_22 = width1_12; width1_32 = width1_12
x1_11 = 0.05; x1_12 = x1_11 + width1_11 + 1.*xbuf0; x1_22 = x1_12; x1_32 = x1_12
height1_11 = 0.7; height1_12 = 0.22; height1_22 = height1_12; height1_32 = height1_12
y1_11 = 0.7; y1_12 = y1_11; y1_22 = y1_12 - height1_22 - 1.*ybuf0; y1_32 = y1_22 - height1_32 - 1.*ybuf0




rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_32 = [x1_32, y1_32, width1_32, height1_32]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.01, 0.92, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.01+x1_12-x1_11, 0.92, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.01+x1_12-x1_11, 0.92+y1_22-y1_12, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.01+x1_12-x1_11, 0.92+y1_32-y1_12, 'D', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.

## rect1_11: E/I perturbation schematics


## rect1_12: no-SD Regression Model
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax)
bar_width_figS1 = 0.21
bar1 = ax.bar(np.arange(5)                   , 0.5*(Reg_values_control[[ 3,4,5,1,2]]-Reg_values_control[[ 8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_control[[ 3,4,5,1,2]]**2+Reg_values_err_control[[ 8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[0], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar1.errorbar[1]:
    b.set_clip_on(False)
for b in bar1.errorbar[2]:
    b.set_clip_on(False)
bar2 = ax.bar(np.arange(5)+   bar_width_figS1, 0.5*(Reg_values_lowered_EI[[3,4,5,1,2]]-Reg_values_lowered_EI[[8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_lowered_EI[[3,4,5,1,2]]**2+Reg_values_err_lowered_EI[[8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[1], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar2.errorbar[1]:
    b.set_clip_on(False)
for b in bar2.errorbar[2]:
    b.set_clip_on(False)
bar3 = ax.bar(np.arange(5)+2.*bar_width_figS1, 0.5*(Reg_values_elevated_EI[[3,4,5,1,2]]-Reg_values_elevated_EI[[8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_elevated_EI[[3,4,5,1,2]]**2+Reg_values_err_elevated_EI[[8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[2], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar3.errorbar[1]:
    b.set_clip_on(False)
for b in bar3.errorbar[2]:
    b.set_clip_on(False)
bar4 = ax.bar(np.arange(5)+3.*bar_width_figS1, 0.5*(Reg_values_upstream_deficit[[3,4,5,1,2]]-Reg_values_upstream_deficit[[8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_upstream_deficit[[3,4,5,1,2]]**2+Reg_values_err_upstream_deficit[[8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar4.errorbar[1]:
    b.set_clip_on(False)
for b in bar4.errorbar[2]:
    b.set_clip_on(False)
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0.,4.+3.*bar_width_figS1])
ax.set_ylim([0.,16.3])
ax.set_xticks([bar_width_figS1*2., 1.+bar_width_figS1*2., 2.+bar_width_figS1*2., 3.+bar_width_figS1*2., 4.+bar_width_figS1*2.])
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 15.])
ax.set_yticklabels([0., 0.15])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=9., axis='x')
ax.tick_params(direction='out', pad=1., axis='y')
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend = ax.legend([bar1, bar2, bar3, bar4], (label_list[0], label_list[1], label_list[2], label_list[3]), loc=(0.6,0.25), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip(color_list, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)
ax.spines['bottom'].set_position(('zero'))


## rect1_22: no-SD Regression Model + monkey A lapse
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
bar_width_figS1 = 0.21
bar1 = ax.bar(np.arange(5)                   , 0.5*(Reg_values_control_fitted_A_lapse[[ 3,4,5,1,2]]-Reg_values_control_fitted_A_lapse[[ 8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_control_fitted_A_lapse[[ 3,4,5,1,2]]**2+Reg_values_err_control_fitted_A_lapse[[ 8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[0], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar1.errorbar[1]:
    b.set_clip_on(False)
for b in bar1.errorbar[2]:
    b.set_clip_on(False)
bar2 = ax.bar(np.arange(5)+   bar_width_figS1, 0.5*(Reg_values_lowered_EI_fitted_A_lapse[[3,4,5,1,2]]-Reg_values_lowered_EI_fitted_A_lapse[[8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_lowered_EI_fitted_A_lapse[[3,4,5,1,2]]**2+Reg_values_err_lowered_EI_fitted_A_lapse[[8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[1], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar2.errorbar[1]:
    b.set_clip_on(False)
for b in bar2.errorbar[2]:
    b.set_clip_on(False)
bar3 = ax.bar(np.arange(5)+2.*bar_width_figS1, 0.5*(Reg_values_elevated_EI_fitted_A_lapse[[3,4,5,1,2]]-Reg_values_elevated_EI_fitted_A_lapse[[8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_elevated_EI_fitted_A_lapse[[3,4,5,1,2]]**2+Reg_values_err_elevated_EI_fitted_A_lapse[[8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[2], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar3.errorbar[1]:
    b.set_clip_on(False)
for b in bar3.errorbar[2]:
    b.set_clip_on(False)
bar4 = ax.bar(np.arange(5)+3.*bar_width_figS1, 0.5*(Reg_values_upstream_deficit_fitted_A_lapse[[3,4,5,1,2]]-Reg_values_upstream_deficit_fitted_A_lapse[[8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_upstream_deficit_fitted_A_lapse[[3,4,5,1,2]]**2+Reg_values_err_upstream_deficit_fitted_A_lapse[[8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar4.errorbar[1]:
    b.set_clip_on(False)
for b in bar4.errorbar[2]:
    b.set_clip_on(False)
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0.,4.+3.*bar_width_figS1])
ax.set_ylim([0.,16.3])
ax.set_xticks([bar_width_figS1*2., 1.+bar_width_figS1*2., 2.+bar_width_figS1*2., 3.+bar_width_figS1*2., 4.+bar_width_figS1*2.])
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 15.])
ax.set_yticklabels([0., 0.15])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=9., axis='x')
ax.tick_params(direction='out', pad=1., axis='y')
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend = ax.legend([bar1, bar2, bar3, bar4], (label_list[0], label_list[1], label_list[2], label_list[3]), loc=(0.6,0.18), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip(color_list, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)
ax.spines['bottom'].set_position(('zero'))


## rect1_32: no-SD Regression Model + monkey H lapse
ax   = fig_temp.add_axes(rect1_32)
fig_funs.remove_topright_spines(ax)
bar_width_figS1 = 0.21
bar1 = ax.bar(np.arange(5)                   , 0.5*(Reg_values_control_fitted_H_lapse[[ 3,4,5,1,2]]-Reg_values_control_fitted_H_lapse[[ 8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_control_fitted_H_lapse[[ 3,4,5,1,2]]**2+Reg_values_err_control_fitted_H_lapse[[ 8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[0], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar1.errorbar[1]:
    b.set_clip_on(False)
for b in bar1.errorbar[2]:
    b.set_clip_on(False)
bar2 = ax.bar(np.arange(5)+   bar_width_figS1, 0.5*(Reg_values_lowered_EI_fitted_H_lapse[[3,4,5,1,2]]-Reg_values_lowered_EI_fitted_H_lapse[[8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_lowered_EI_fitted_H_lapse[[3,4,5,1,2]]**2+Reg_values_err_lowered_EI_fitted_H_lapse[[8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[1], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar2.errorbar[1]:
    b.set_clip_on(False)
for b in bar2.errorbar[2]:
    b.set_clip_on(False)
bar3 = ax.bar(np.arange(5)+2.*bar_width_figS1, 0.5*(Reg_values_elevated_EI_fitted_H_lapse[[3,4,5,1,2]]-Reg_values_elevated_EI_fitted_H_lapse[[8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_elevated_EI_fitted_H_lapse[[3,4,5,1,2]]**2+Reg_values_err_elevated_EI_fitted_H_lapse[[8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[2], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar3.errorbar[1]:
    b.set_clip_on(False)
for b in bar3.errorbar[2]:
    b.set_clip_on(False)
bar4 = ax.bar(np.arange(5)+3.*bar_width_figS1, 0.5*(Reg_values_upstream_deficit_fitted_H_lapse[[3,4,5,1,2]]-Reg_values_upstream_deficit_fitted_H_lapse[[8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_err_upstream_deficit_fitted_H_lapse[[3,4,5,1,2]]**2+Reg_values_err_upstream_deficit_fitted_H_lapse[[8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list[3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar4.errorbar[1]:
    b.set_clip_on(False)
for b in bar4.errorbar[2]:
    b.set_clip_on(False)
ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0.,4.+3.*bar_width_figS1])
ax.set_ylim([0.,16.3])
ax.set_xticks([bar_width_figS1*2., 1.+bar_width_figS1*2., 2.+bar_width_figS1*2., 3.+bar_width_figS1*2., 4.+bar_width_figS1*2.])
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 15.])
ax.set_yticklabels([0., 0.15])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=9., axis='x')
ax.tick_params(direction='out', pad=1., axis='y')
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
legend = ax.legend([bar1, bar2, bar3, bar4], (label_list[0], label_list[1], label_list[2], label_list[3]), loc=(0.6,0.18), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip(color_list, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)
ax.spines['bottom'].set_position(('zero'))



fig_temp.savefig(path_cwd+'Figure7S1.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
########################################################################################################################
# ### Figure 8S1: Ketamine Data
## Mean/Variance Regression model                                                                                       # See DrugDayModellingScript.m: line 434-460
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
Reg_bars_A_ketamine = np.array([0.579709987966915, 7.197909585639628, 3.372968517421578])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on ketamine.
Reg_bars_H_ketamine = np.array([-0.041342389429829, 9.127575307913101, 2.830329249052940])  # [Bias, Val diff , Std diff]. Henry regression Beta values on ketamine.
Reg_bars_A_saline = np.array([0.067202587850084, 26.069967029926556, 5.902075606627891])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on saline.
Reg_bars_H_saline = np.array([-0.012952340744864, 20.950515203294927, 2.979811722986580])  # [Bias, Val diff , Std diff]. Henry regression Beta values on saline.
Reg_bars_A_pre_ketamine = np.array([-0.018486360992728, 25.066552469697804, 5.476637879268277])  # [Bias, Val diff , Std diff]. Alfie regression Beta values pre ketamine.
Reg_bars_H_pre_ketamine = np.array([-0.088145566508677, 20.242263778486283, 2.989324913443232])  # [Bias, Val diff , Std diff]. Henry regression Beta values pre ketamine.
Reg_bars_A_pre_saline = np.array([0.025344551399459, 25.080324535558805, 7.222049938157925])  # [Bias, Val diff , Std diff]. Alfie regression Beta values pre saline.
Reg_bars_H_pre_saline = np.array([0.031205437334164, 20.469342955391840, 1.421425556593712])  # [Bias, Val diff , Std diff]. Henry regression Beta values pre saline.
Reg_bars_err_A_ketamine = np.array([0.047236118202221, 0.464666834175333, 0.538572459402258])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on ketamine.
Reg_bars_err_H_ketamine = np.array([0.051815368521218, 0.521360485218823, 0.623558968474620])  # [Bias, Val diff , Std diff]. Henry regression Beta values on ketamine.
Reg_bars_err_A_saline = np.array([0.080275332862987, 1.122992936724174, 0.984051716756847])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on saline.
Reg_bars_err_H_saline = np.array([0.057234482845660, 0.711194176843697, 0.701774559140750])  # [Bias, Val diff , Std diff]. Henry regression Beta values on saline.
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

Mean_reg_err_bars_A_v2  = np.abs([Reg_bars_err_A_saline[1], Reg_bars_err_A_ketamine[1]])
Mean_reg_err_bars_H_v2  = np.abs([Reg_bars_err_H_saline[1], Reg_bars_err_H_ketamine[1]])
Var_reg_err_bars_A_v2  = np.abs([Reg_bars_err_A_saline[2], Reg_bars_err_A_ketamine[2]])
Var_reg_err_bars_H_v2  = np.abs([Reg_bars_err_H_saline[2], Reg_bars_err_H_ketamine[2]])
Var_mean_ratio_err_Reg_bars_A_v2  = var_mean_ratio_list_A*((Var_reg_err_bars_A_v2/var_effect_list_A)**2 + (Mean_reg_err_bars_A_v2/mean_effect_list_A)**2)**0.5
Var_mean_ratio_err_Reg_bars_H_v2  = var_mean_ratio_list_H*((Var_reg_err_bars_H_v2/var_effect_list_H)**2 + (Mean_reg_err_bars_H_v2/mean_effect_list_H)**2)**0.5


### PK                                                                                                                    # See DrugDayModellingScript.m: end of DrugDayFigs_PsychKernel.m (For old, unpaired method in see lines 275-430).
i_PK_list_6 = np.arange(1,6+1)
t_PK_list_6 = 0.125 + 0.25*np.arange(6)
PK_A_ketamine = np.array([1.588076560991160,1.172131014527615,1.348232083830557,1.130033677949196,0.865410950858707,0.983196410835269])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_ketamine = np.array([1.618410540743415,1.625929373302735,1.539405099097833,1.930377748043883,1.190732825227660,1.135496780528839])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_A_saline = np.array([4.803126510079912,4.039796862653806,4.300008382210277,3.719457819997655,3.525180800769672,5.024612665261444])    # [{A&B_PK}]. Alfie. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_saline = np.array([3.853809934503830,3.338190357430990,3.195019612163858,3.318322799202861,3.558361021587738,3.492684990900345])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.

PK_A_ketamine_errbar = np.array([0.177265348015414,0.172413647578042,0.173208472412933,0.175110800690537,0.178692279224766,0.172827355387130])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_ketamine_errbar = np.array([0.196746922522314,0.201666875273074,0.197385551271038,0.206332861449933,0.199344435697785,0.195232307765412])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_A_saline_errbar = np.array([0.355312183824056,0.329217592139402,0.334326051364771,0.324841845308493,0.313840995472368,0.351034890835890])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.
PK_H_saline_errbar = np.array([0.245070726563752,0.227332600986689,0.220146551460523,0.227963136258739,0.233726860570945,0.232278503931951])    # [{A&B_PK}]. Henry. Paired (check with Sean whether I am using the right data). Note that ketamine/ drug day data only has 6 instead of 8 samples.



## Pcorr & RT vs time                                                                                                   # See DrugDayModellingScript.m: DrugDayFigs_TimeCourseAnal
t_list_Pcorr_RT = np.arange(-20, 61)
Pcorr_t_mean_list_ketamine_A = np.array([0.883703811287736, 0.885788786215132, 0.890558328737963, 0.882539386539387, 0.870627709339829, 0.878191913451078, 0.862232327762563, 0.849034367114978, 0.856096202335329, 0.859266759842611, 0.866779722749154, 0.869106230634987, 0.881086223708648, 0.883318846311240, 0.884458676167278, 0.881009679899088, 0.883925536802851, 0.872667542938258, 0.869556026657616, 0.858742940651543, 0.821580870022723, 0.782780327378029, 0.724611595523175, 0.658889811382683, 0.605550768327394, 0.579741786050376, 0.550391468415288, 0.524164630520298, 0.526823896421921, 0.548938622861374, 0.564529470711261, 0.572112433774473, 0.609316654608728, 0.625921489392345, 0.618254686829248, 0.613341993065150, 0.642583109317284, 0.669398536353813, 0.675667427747138, 0.691093192468766, 0.719297344800441, 0.733689838780684, 0.736517129411476, 0.739029543407134, 0.751531498280121, 0.770308733561342, 0.785442750662867, 0.789983206431372, 0.802998324774641, 0.809899288177624, 0.815750646074119, 0.812488411420932, 0.805573531541024, 0.811213521724429, 0.825047798704246, 0.817550473324423, 0.816005334879174, 0.819024080006864, 0.824640159446112, 0.839630196723451, 0.823289809899717, 0.842740472583397, 0.837837667392363, 0.841735177072146, 0.861198362160380, 0.845557092912960, 0.841993566970253, 0.840966757114533, 0.851519718713331, 0.847257318516097, 0.836667215957881, 0.839635165108561, 0.848457314395777, 0.828687544496657, 0.824554251082143, 0.835885023914495, 0.832556362340661, 0.834458283275636, 0.829058755772804, 0.848423064989327, 0.850195241031287])
Pcorr_t_se_list_ketamine_A   = np.array([0.0148775240678694, 0.0171264136441060, 0.0173379326228600, 0.0289809247526689, 0.0280390943184773, 0.0210999441985796, 0.0213602406297143, 0.0198993159892748, 0.0193172277731075, 0.0195749436482875, 0.0152387489945035, 0.0150751410929515, 0.0175580276807106, 0.0171041552642837, 0.0192772024754729, 0.0184467080616490, 0.0153419872249229, 0.0147544608408925, 0.0149574458594036, 0.0214086380356508, 0.0260928129849735, 0.0294694730348198, 0.0324106642768447, 0.0341884976879969, 0.0310211119425426, 0.0297004509947635, 0.0263182895736251, 0.0263999818776431, 0.0234952745891776, 0.0173662135042439, 0.0201373836739187, 0.0223842416516911, 0.0158349815860392, 0.0215123274351815, 0.0263789256718572, 0.0299085910467981, 0.0301308691584528, 0.0307580823839286, 0.0253751083024743, 0.0262841643256258, 0.0208154422470270, 0.0173650640431121, 0.0188376146004632, 0.0203489722403446, 0.0207845028409843, 0.0221331567329062, 0.0184667949889641, 0.0178763796227513, 0.0199592981399063, 0.0193113723954141, 0.0182999543952710, 0.0205195062977608, 0.0212727869924992, 0.0196010189490558, 0.0196597198645704, 0.0208587664001593, 0.0246882925180981, 0.0227885772494417, 0.0240163129453466, 0.0251249747181711, 0.0278045496406203, 0.0270846025478693, 0.0269744459813218, 0.0271971367591389, 0.0215307611464390, 0.0198602444683479, 0.0219601308134300, 0.0180363142177232, 0.0200873378593112, 0.0194480748168347, 0.0216774085943442, 0.0193379738421115, 0.0154733409989961, 0.0232891174616129, 0.0198832103135806, 0.0172369569010172, 0.0195230172248424, 0.0175029029359645, 0.0159160282089218, 0.0177866524153166, 0.0182736200361430])
Pcorr_t_mean_list_ketamine_H = np.array([0.815574740194704, 0.841405655008596, 0.839487964471061, 0.843652643614487, 0.834811396050741, 0.846269974255725, 0.838109963973935, 0.837784790492865, 0.843976221255093, 0.843229355729356, 0.853149538346907, 0.867778310962544, 0.879796599227444, 0.877284834577839, 0.873246791810768, 0.862502993573873, 0.865766695142460, 0.866242140630146, 0.859679898788605, 0.833711210536666, 0.835349872757961, 0.810953621185039, 0.775278820839186, 0.723149964822759, 0.705801647249016, 0.696094860218375, 0.657019497634823, 0.645595595595596, 0.657061485634809, 0.662668291322291, 0.659881556456634, 0.651302800462695, 0.671132086285875, 0.671704525164443, 0.652160510945813, 0.654155284519903, 0.671164188811920, 0.674722472104393, 0.674120636736505, 0.699155242905243, 0.718846613063259, 0.705172263912599, 0.719710923805507, 0.740077260480021, 0.761563611369378, 0.774251599679996, 0.766467143376643, 0.802717975484193, 0.817818368516379, 0.845331597103683, 0.850969147360659, 0.844818789238483, 0.863307557348395, 0.859012671207793, 0.838739461087731, 0.824003378435437, 0.803089971228672, 0.804953069636992, 0.795996531948545, 0.803149689860594, 0.819465103972485, 0.781873370612552, 0.799419897016741, 0.809149630370747, 0.805212842172723, 0.871922256159516, 0.866535376365361, 0.866700064591535, 0.841262209440346, 0.834899863567415, 0.817479905846083, 0.819616324007639, 0.813641122732032, 0.798825344877977, 0.825191968335907, 0.814982015340867, 0.832156923488193, 0.824570162527887, 0.824631699930743, 0.832464835963640, 0.830884803387822])
Pcorr_t_se_list_ketamine_H   = np.array([0.0194720697382234, 0.0179613674785784, 0.0166618827962992, 0.0159504933736019, 0.0202006286974772, 0.0205625874636626, 0.0105753933730602, 0.0106037767890490, 0.00781141999155896, 0.0120613446635660, 0.00997476068232163, 0.0115216451585273, 0.0173498670989751, 0.0197694457602505, 0.0137116414505682, 0.0126736266159868, 0.0134659179706839, 0.0158657384297721, 0.0189074455503961, 0.0240209466149276, 0.0244132409254016, 0.0300300446352687, 0.0379350703226034, 0.0404335757205955, 0.0364758671767230, 0.0265773883326211, 0.0272981527894058, 0.0283790157923347, 0.0238402440007850, 0.0295241075229138, 0.0326474308156858, 0.0388912642847307, 0.0347580618007616, 0.0320026108550268, 0.0345162764518930, 0.0346540240632789, 0.0375790770695594, 0.0400277602392230, 0.0313012803515695, 0.0220537657119233, 0.0211139124785474, 0.0232718385558216, 0.0219695908773195, 0.0276175287389271, 0.0256074030074463, 0.0278328805096034, 0.0204153192303470, 0.0170109400506299, 0.0203610331400942, 0.0209528632116823, 0.0165468359871923, 0.0182752289187231, 0.0130101326817449, 0.0158486716490398, 0.0209423117510605, 0.0236649341568076, 0.0225644272572446, 0.0228498520043674, 0.0286316227385408, 0.0321274645530194, 0.0316713843960534, 0.0724170281072056, 0.0743170732571218, 0.0761141148219325, 0.0752966969632244, 0.0159720688995715, 0.0158269851182452, 0.0161553379360402, 0.0170782927761820, 0.0168213946138871, 0.0188897260874452, 0.0251487126522854, 0.0306201458820907, 0.0348099308607285, 0.0258375175844614, 0.0260638204097840, 0.0185928862501358, 0.0182944114573743, 0.0170177758586953, 0.0141899713743388, 0.0142892315048022])
Pcorr_t_mean_list_saline_A = np.array([0.856191017883950, 0.863598724638769, 0.866978668550575, 0.870528930378429, 0.873077222072724, 0.879283750255617, 0.884249935912340, 0.868695257906088, 0.867479160720583, 0.876061648465495, 0.891528662274539, 0.895214703205532, 0.907934477613651, 0.907944871585279, 0.905631366881549, 0.906965556455941, 0.910817300712317, 0.901404776999014, 0.890677828826310, 0.899150942787574, 0.904935103256389, 0.898412381687091, 0.889822341875162, 0.903684179068795, 0.901785560239398, 0.908214323835848, 0.903495714885568, 0.886084189811351, 0.885367620005137, 0.877359669897534, 0.889025107522182, 0.882116262849344, 0.889501606576165, 0.904270297856050, 0.905805257258575, 0.910789163837077, 0.902250853009295, 0.901205326094029, 0.892658734881932, 0.903369359839948, 0.912433699120295, 0.908090296799974, 0.900870456362135, 0.885494642556002, 0.889017859923033, 0.872982804600842, 0.844506820521738, 0.836564975553809, 0.842113733205701, 0.840955050102070, 0.840703910735218, 0.850037670393621, 0.862026825370479, 0.858104224903339, 0.858036222706935, 0.870186791733979, 0.870817829700394, 0.869640050243499, 0.861236299457551, 0.873216469424362, 0.874672067524797, 0.875127447455000, 0.881910039832079, 0.892384965602664, 0.905941979466062, 0.898340054432294, 0.891884074666028, 0.882533122732225, 0.876314196771939, 0.869178252796978, 0.861463960467831, 0.876955211420970, 0.875791155342945, 0.873639623434717, 0.876551687061228, 0.881613922012977, 0.889607884826756, 0.875581811978871, 0.873656204906205, 0.885212310354625, 0.884995234931848])
Pcorr_t_se_list_saline_A   = np.array([0.0184628557035720, 0.0187600208678747, 0.0130858723346604, 0.0162390688870086, 0.0198151627399836, 0.0238544586203958, 0.0223051947362850, 0.0245398364403513, 0.0259750954134834, 0.0231118437807392, 0.0204793601742768, 0.0170854014500821, 0.0152121472859935, 0.0169323696430357, 0.0138005659794993, 0.0147545552085968, 0.0118478380351701, 0.0123382603518487, 0.0142608199431453, 0.0135232761777189, 0.0154872156912983, 0.0158540415490361, 0.0164502562467616, 0.0115662602686174, 0.0121866195104809, 0.0152825586570352, 0.0168745053228276, 0.0203371219960262, 0.0203096388689099, 0.0188657381896483, 0.0147453215630216, 0.0153586589909849, 0.0155865858391511, 0.0124975967538607, 0.0135541745566610, 0.0165348503375073, 0.0211045688110486, 0.0197842162464969, 0.0218652177860984, 0.0168733893597222, 0.0138732406580304, 0.0146679905967230, 0.0178852749366856, 0.0226824721168942, 0.0183704064416448, 0.0184362476006140, 0.0165333837789186, 0.0193234664292653, 0.0173849906926658, 0.0193620508821180, 0.0200823071824641, 0.0194119801453625, 0.0151343993827135, 0.0165799311798246, 0.0180244836366626, 0.0208103401385456, 0.0251286492711336, 0.0225582196154867, 0.0214596583505101, 0.0182821720030386, 0.0141465289459643, 0.0168700725942459, 0.0171786283602510, 0.0138744071219503, 0.0172623879558153, 0.0195140252269775, 0.0180714141206285, 0.0151548792496281, 0.0175393589140480, 0.0227154021711341, 0.0196225860799105, 0.0185322975022841, 0.0194311389424473, 0.0225893544341283, 0.0237291958168351, 0.0219497580267990, 0.0206210009823052, 0.0225627606712048, 0.0195979734193280, 0.0191398289500788, 0.0168349062105887])
Pcorr_t_mean_list_saline_H = np.array([0.843249757758979, 0.846664336236119, 0.845840370731263, 0.840697973925758, 0.835932141913557, 0.844929158564283, 0.856885499619047, 0.865274713022186, 0.865846209228562, 0.871764615708114, 0.871941127595835, 0.864589850507387, 0.861928413049071, 0.868770469353241, 0.868223214905948, 0.874679107489121, 0.868838422774770, 0.873435063071078, 0.872877307827498, 0.855278719009232, 0.850953979491844, 0.848025720129090, 0.857907878433722, 0.855398968437062, 0.864076971963419, 0.873307833296998, 0.865413622282016, 0.867960742745535, 0.869480917209970, 0.872710823553031, 0.869702591800115, 0.857027025404012, 0.862784706926248, 0.858154418384827, 0.845436689161265, 0.852441103047735, 0.846881616312297, 0.860803991903889, 0.856804452509372, 0.851561047376814, 0.863520040087378, 0.851313848942632, 0.846460094098140, 0.839030999814027, 0.848393130823472, 0.847555761157176, 0.848471898505657, 0.857711500218863, 0.860309022460725, 0.863518324561895, 0.861375916748323, 0.861226142116937, 0.853280176968163, 0.844257050892632, 0.844751438508518, 0.855476517662423, 0.850317302114561, 0.854283929628812, 0.864583784006726, 0.857543910198841, 0.859205217453748, 0.866330154134550, 0.868160259334763, 0.865511239279446, 0.862255288105827, 0.863262719266426, 0.862647072509961, 0.847497220456274, 0.838740383512501, 0.834383223115720, 0.833262690595192, 0.844914022644773, 0.843877390081711, 0.852777210141093, 0.859534107081468, 0.877788860359900, 0.871992622009243, 0.864411216840449, 0.874844757900987, 0.867996701526834, 0.877598362340464])
Pcorr_t_se_list_saline_H   = np.array([0.0296993432731170, 0.0221287583698986, 0.0228815189850695, 0.0250799030964718, 0.0239460454052467, 0.0196045045407078, 0.0184398469848613, 0.0172735832799920, 0.0184780388722357, 0.0161549713012411, 0.0148141452309982, 0.0154856123697293, 0.0152093304501387, 0.0115523701635686, 0.0122828331860118, 0.0155948783570728, 0.0158078791973533, 0.0163560614444277, 0.0177910609892250, 0.0173497551224638, 0.0170884225543725, 0.0182153362868936, 0.0187708295211848, 0.0166015796699532, 0.0153050756781380, 0.0165453534245483, 0.0168717713913510, 0.0170486987332612, 0.0145529352345288, 0.0136541715006666, 0.0165074287655612, 0.0172060937176041, 0.0166880292477339, 0.0132739794393794, 0.0155884905647181, 0.0188788420099832, 0.0159227241336218, 0.0146833431891990, 0.0121925113661034, 0.0122856217947944, 0.0108421391228616, 0.00939596702859465, 0.0112535867529830, 0.0126882580712372, 0.0103652285863630, 0.0111856485386571, 0.0109204054207013, 0.00987303331088234, 0.0102951818327738, 0.0112381389375424, 0.0140335432079205, 0.0172088651737720, 0.0184980839745982, 0.0157049415662850, 0.0133435568006771, 0.0146191688496659, 0.0164242377605355, 0.0114932899202948, 0.0123030044598685, 0.0101239637838612, 0.0114671391947024, 0.0134895775488366, 0.0136113344421723, 0.0153205052761101, 0.0159065797055786, 0.0174690353200025, 0.0183057265146228, 0.0198882988619475, 0.0179588452008759, 0.0149652105112753, 0.0166204714868636, 0.0109874980563734, 0.0127116280695991, 0.0118323600040281, 0.0148030932497095, 0.0127386119666029, 0.0150343202151782, 0.0117925108345454, 0.00906733096245799, 0.00851299451411291, 0.0115357586266575])


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







## Regression analysis, Experiments                                                                                     # See DrugDayModellingScript.m: line227-261, DrugRegStrat
Reg_values_A_ketamine = np.array([0.654017871440318, 0.181195216732370, -0.225326176642387, 7.63031172852905, 1.03979879333475, -1.53517578956187, -0.769768604559133, 0.107092723605785, -5.32677983136311, -1.78756772176576, 0.315170701505100])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_A_ketamine = np.array([0.490048328944004, 0.287057949410812, 0.282965538754991, 1.22124475156854, 0.588501834915180, 0.602608055418754, 0.280961026860069, 0.269634368382140, 1.12477353027445, 0.568165891973565, 0.560812598695400]) # Error bars for Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_H_ketamine = np.array([-0.991860342637263, 0.263177081137356, -0.489283776566774, 8.29061685234432, 2.17514421296188, -0.362498712722909, 0.338679402146550, 0.479564884442233, -9.95199266224805, -0.405867747655713, 1.05429830932184])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_H_ketamine = np.array([0.526354948785369, 0.327060108096409, 0.317548133422190, 1.32077514433373, 0.636869759192090, 0.631518473695472, 0.330045477559998, 0.322441149171933, 1.33841571187787, 0.626575145610808, 0.659560148522371]) # Error bars for Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_A_saline = np.array([-0.103029802616500, 1.04521176509978, 1.14830714774286, 25.2915896364690, 1.50284376638223, -2.61313222381824, -0.757815204640959, -1.04094505155285, -24.2662508207248, -1.98631141461863, 1.89564405116018])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_A_saline = np.array([0.831423955681739, 0.526834545841784, 0.488180774535345, 2.21667538453773, 0.998248793961396, 1.02857749181223, 0.484254786737343, 0.483040502621355, 2.21732973775158, 1.01254558149924, 1.03292332364698])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_H_saline = np.array([-0.119302793575504, 0.980848277928142, 0.138043505072248, 18.0136308724504, 2.26309306020433, -0.433329332310245, 0.129293265064817, -0.216802459208964, -22.3947581623215, -0.131388636926997, 1.27532692218506])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_errbar_H_saline = np.array([0.576631048009977, 0.376253337905010, 0.356070193160662, 1.57830439726393, 0.734334147525212, 0.737123697927465, 0.363375582166555, 0.358701719850583, 1.64629996256491, 0.748512663663330, 0.752499893626100])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)







## Fitting Psychometric Functions
x_list_psychometric = np.arange(0.01, 0.5, 0.01)                                                                        # See figure_psychometric_function_fit.py, esp lines 322-527
x0_psychometric = 0.
## non-binned MLE (i.e. done using literal net evidence, via matlab). See Psychometric_function_fit_DrugDays_NL.m.
psychometric_params_A_saline_all        = [0.0595776866237313, 1.26810162179331, 0.0138702695806634]
psychometric_params_H_saline_all        = [0.0681413425053521, 1.07582639210372, 0.0123764957351213]
psychometric_params_A_ketamine_all      = [0.164267968758472, 0.732705192383852, 0.0377990600679478]
psychometric_params_H_ketamine_all      = [0.130851990508893, 1.16584379279672, 0.0238689326833176]






x1_31 = 0.071; x1_32 = x1_31 + width1_31 + 0.7*xbuf0; x1_33 = x1_32 + width1_32 + 1.15*xbuf0; x1_34 = x1_33 + width1_33 + 0.7*xbuf0

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
fig_temp.text(0.01, 0.955, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.2162, 0.975, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.012+x1_12-x1_11, 0.955, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.718, 0.975, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.015, 0.955 + y1_21 - y1_11, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015+x1_23-x1_21, 0.955 + y1_21 - y1_11, 'G', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015, 0.955 + y1_31 - y1_11, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.03+x1_34-x1_31, 0.955 + y1_31 - y1_11, 'H', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015, 0.955 + y1_41 - y1_11, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.015+x1_42-x1_41, 0.955 + y1_41 - y1_11, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.027+x1_43-x1_41, 0.955 + y1_41 - y1_11, 'I', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.02+x1_44-x1_41, 0.955 + y1_41 - y1_11, 'J', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.
fig_temp.text(0.125, 0.97+y1_21-y1_11, 'Saline', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.115+x1_22-x1_21, 0.97+y1_21-y1_11, 'Ketamine', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.125+x1_23-x1_21, 0.97+y1_21-y1_11, 'Saline', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.16+x1_24-x1_21, 0.97+y1_21-y1_11, 'Ketamine', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')


## rect1_11: Correct Probability vs time, Monkey A
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax)
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_A, color=color_list_expt[0], linestyle='-', zorder=3, clip_on=False, label='Saline', linewidth=1.)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_A + Pcorr_t_se_list_saline_A, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_A - Pcorr_t_se_list_saline_A, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_A, color=color_list_expt[1], linestyle='-', zorder=3, clip_on=False, label='Ketamine', linewidth=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_A + Pcorr_t_se_list_ketamine_A, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_A - Pcorr_t_se_list_ketamine_A, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
ax.fill_between([5., 30.], 1., lw=0, color='k', alpha=0.2, zorder=0)
ax.set_xlabel('Time (mins)', fontsize=fontsize_legend)
ax.set_ylabel('Correct Probability', fontsize=fontsize_legend)
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
legend = ax.legend(loc=(0.65,-0.02), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

## rect1_12: Correct Probability vs time, Monkey H
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax)
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_H, color=color_list_expt[0], linestyle='-', zorder=3, clip_on=False, label='Saline', linewidth=1.)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_H + Pcorr_t_se_list_saline_H, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_H - Pcorr_t_se_list_saline_H, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_H, color=color_list_expt[1], linestyle='-', zorder=3, clip_on=False, label='Ketamine', linewidth=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_H + Pcorr_t_se_list_ketamine_H, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_H - Pcorr_t_se_list_ketamine_H, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
ax.fill_between([5., 30.], 1., lw=0, color='k', alpha=0.2, zorder=0)
ax.set_xlabel('Time (mins)', fontsize=fontsize_legend)
ax.set_ylabel('Correct Probability', fontsize=fontsize_legend)
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
legend = ax.legend(loc=(0.65,-0.02), fontsize=fontsize_legend, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
for color,text,item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)




##### Psychometric functions.
## rect1_21: Psychometric function: Monkey A Saline
ax_0   = fig_temp.add_axes(rect1_21_0)
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_A_saline_list[6:],    P_corr_A_saline_list[6:], ErrBar_P_corr_A_saline_list[6:], color=color_list_expt[0], markerfacecolor=color_list_expt[0], ecolor=color_list_expt[0], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_A_saline_list[:6], 1.-P_corr_A_saline_list[:6], ErrBar_P_corr_A_saline_list[:6], color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], markerfacecolor=[1-(1-ci)*0.5 for ci in color_list_expt[0]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[0]], fmt='.', zorder=4, clip_on=False, label='Lower SD Correct', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_A_saline_all, x_list_psychometric), color=color_list_expt[0], ls='-', clip_on=False, zorder=3)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_A_saline_all, -x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], ls='-', clip_on=False, zorder=2)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_A_saline_all, x0_psychometric), s=15., color=color_list_expt[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_A_saline_all, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
y_shift_spines = -0.0688
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_list_expt[0], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.2,-0.08), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0.)
for color,text,item in zip([color_list_expt[0], [1-(1-ci)*0.5 for ci in color_list_expt[0]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

## rect1_22: Psychometric function: Monkey A ketamine.
ax_0   = fig_temp.add_axes(rect1_22_0)
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_A_ket_list[6:],    P_corr_A_ket_list[6:], ErrBar_P_corr_A_ket_list[6:], color=color_list_expt[1], markerfacecolor=color_list_expt[1], ecolor=color_list_expt[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_A_ket_list[:6], 1.-P_corr_A_ket_list[:6], ErrBar_P_corr_A_ket_list[:6], color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], markerfacecolor=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[1]], fmt='.', zorder=4, clip_on=False, label='Lower SD Correct', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_A_ketamine_all, x_list_psychometric, 0.118), color=color_list_expt[1], ls='-', clip_on=False, zorder=3)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_A_ketamine_all, -x_list_psychometric, 0.118), color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ls='-', clip_on=False, zorder=2)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_A_ketamine_all, x0_psychometric, 0.118), s=15., color=color_list_expt[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_A_ketamine_all, -x0_psychometric, 0.118), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
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
y_shift_spines = -0.0688
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_list_expt[1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.55,0.75), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0.)
for color,text,item in zip([color_list_expt[1], [1-(1-ci)*0.5 for ci in color_list_expt[1]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)



## rect1_23: Psychometric function: Monkey H Saline.
ax_0   = fig_temp.add_axes(rect1_23_0)
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_H_saline_list[6:],    P_corr_H_saline_list[6:], ErrBar_P_corr_H_saline_list[6:], color=color_list_expt[0], markerfacecolor=color_list_expt[0], ecolor=color_list_expt[0], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_H_saline_list[:6], 1.-P_corr_H_saline_list[:6], ErrBar_P_corr_H_saline_list[:6], color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], markerfacecolor=[1-(1-ci)*0.5 for ci in color_list_expt[0]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[0]], fmt='.', zorder=4, clip_on=False, label='Lower SD Correct', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_H_saline_all, x_list_psychometric), color=color_list_expt[0], ls='-', clip_on=False, zorder=3)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_H_saline_all, -x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], ls='-', clip_on=False, zorder=2)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_H_saline_all, x0_psychometric), s=15., color=color_list_expt[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_H_saline_all, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False)
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
y_shift_spines = -0.0688
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_list_expt[0], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.2,-0.08), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0.)
for color,text,item in zip([color_list_expt[0], [1-(1-ci)*0.5 for ci in color_list_expt[0]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)

## rect1_24: Psychometric function: Monkey H Ketamine.
ax_0   = fig_temp.add_axes(rect1_24_0)
ax   = fig_temp.add_axes(rect1_24)
fig_funs.remove_topright_spines(ax_0)
fig_funs.remove_topright_spines(ax)
ax.spines['left'].set_visible(False)
fig_funs.remove_topright_spines(ax)
ax.errorbar( d_evidence_H_ket_list[6:],    P_corr_H_ket_list[6:], ErrBar_P_corr_H_ket_list[6:], color=color_list_expt[1], markerfacecolor=color_list_expt[1], ecolor=color_list_expt[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct' , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_H_ket_list[:6], 1.-P_corr_H_ket_list[:6], ErrBar_P_corr_H_ket_list[:6], color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], markerfacecolor=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[1]], fmt='.', zorder=4, clip_on=False, label='Lower SD Correct', markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_H_ketamine_all, x_list_psychometric, 0.0684), color=color_list_expt[1], ls='-', clip_on=False, zorder=3)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_H_ketamine_all, -x_list_psychometric, 0.0684), color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ls='-', clip_on=False, zorder=2)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_H_ketamine_all, x0_psychometric, 0.0684), s=15., color=color_list_expt[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_H_ketamine_all, -x0_psychometric, 0.0684), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.3, 50], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
ax.set_xscale('log')
ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
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
y_shift_spines = -0.0688
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend_bars = [Line2D([0] , [0], color=color_list_expt[1], alpha=1., label='Higher SD Correct'),
                Line2D([0], [0], color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], alpha=1., label='Lower SD Correct')]
legend = ax.legend(handles=legend_bars, loc=(-0.55,0.75), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=0.5, handletextpad=0.)
for color,text,item in zip([color_list_expt[1], [1-(1-ci)*0.5 for ci in color_list_expt[1]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)



## rect1_31: Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_31)
fig_funs.remove_topright_spines(ax)
bar1 = ax.bar(np.arange(len(mean_effect_list_A)), mean_effect_list_A, bar_width_compare3, alpha=bar_opacity, yerr=Mean_reg_err_bars_A_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar1.errorbar[1]:
    b.set_clip_on(False)
for b in bar1.errorbar[2]:
    b.set_clip_on(False)
ax.plot([0,2.*bar_width_compare3], [0.5*(mean_effect_list_A_preSK[0]+mean_effect_list_A_preSK[1]), 0.5*(mean_effect_list_A_preSK[0]+mean_effect_list_A_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.], [29.5], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [28.2,28.2], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
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
ax.bar(np.arange(len(var_effect_list_A)), var_effect_list_A, bar_width_compare3, alpha=bar_opacity, yerr=Var_reg_err_bars_A_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(var_effect_list_A_preSK[0]+var_effect_list_A_preSK[1]), 0.5*(var_effect_list_A_preSK[0]+var_effect_list_A_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.], [7.4], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [7.1,7.1], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('SD Evidence Beta', fontsize=fontsize_legend, labelpad=1.5)
ax.set_xlim([0,len(var_effect_list_A)-1+bar_width_compare3])
ax.set_ylim([0.,7.5])
ax.set_xticks([0., 1.])
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
ax.set_yticks([0., 6.])
ax.set_yticklabels([0., 0.06])
minorLocator = MultipleLocator(2.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")

## rect1_33: Variance Beta/ Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_33)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(var_mean_ratio_list_A)), var_mean_ratio_list_A, bar_width_compare3, alpha=bar_opacity, yerr=Var_mean_ratio_err_Reg_bars_A_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(var_mean_ratio_list_A_preSK[0]+var_mean_ratio_list_A_preSK[1]), 0.5*(var_mean_ratio_list_A_preSK[0]+var_mean_ratio_list_A_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.], [0.59], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [0.565,0.565], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('PVB Index', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(var_mean_ratio_list_A)-1+bar_width_compare3])
ax.set_ylim([0.,0.6])
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
ax.bar(np.arange(len(mean_effect_list_H)), mean_effect_list_H, bar_width_compare3, alpha=bar_opacity, yerr=Mean_reg_err_bars_H_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(mean_effect_list_H_preSK[0]+mean_effect_list_H_preSK[1]), 0.5*(mean_effect_list_H_preSK[0]+mean_effect_list_H_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.], [24.3], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [23,23], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
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
ax.bar(np.arange(len(var_effect_list_H)), var_effect_list_H, bar_width_compare3, alpha=bar_opacity, yerr=Var_reg_err_bars_H_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(var_effect_list_H_preSK[0]+var_effect_list_H_preSK[1]), 0.5*(var_effect_list_H_preSK[0]+var_effect_list_H_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.set_ylabel('SD Evidence Beta', fontsize=fontsize_legend, labelpad=1.5)
ax.set_xlim([0,len(var_effect_list_H)-1+bar_width_compare3])
ax.set_ylim([0.,7.1])
ax.set_xticks([0., 1.])
ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
ax.set_yticks([0., 6.])
ax.set_yticklabels([0., 0.06])
minorLocator = MultipleLocator(2.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")

## rect1_36: Variance Beta/ Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_36)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(var_mean_ratio_list_H)), var_mean_ratio_list_H, bar_width_compare3, alpha=bar_opacity, yerr=Var_mean_ratio_err_Reg_bars_H_v2, ecolor='k', color=color_list_expt, clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(var_mean_ratio_list_H_preSK[0]+var_mean_ratio_list_H_preSK[1]), 0.5*(var_mean_ratio_list_H_preSK[0]+var_mean_ratio_list_H_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.], [0.435], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [0.41,0.41], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('PVB Index', fontsize=fontsize_legend, labelpad=-5.)
ax.set_xlim([0,len(var_mean_ratio_list_H)-1+bar_width_compare3])
ax.set_ylim([0.,0.6])
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
bar1 = ax.bar(np.arange(5)                , 0.5*(Reg_values_A_saline[[ 3,4,5,1,2]]-Reg_values_A_saline[[ 8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_errbar_A_saline[[ 3,4,5,1,2]]**2+Reg_values_errbar_A_saline[[ 8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list_expt[0], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar1.errorbar[1]:
    b.set_clip_on(False)
for b in bar1.errorbar[2]:
    b.set_clip_on(False)
bar3 = ax.bar(np.arange(5)+bar_width_figS1, 0.5*(Reg_values_A_ketamine[[3,4,5,1,2]]-Reg_values_A_ketamine[[8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_errbar_A_ketamine[[3,4,5,1,2]]**2+Reg_values_errbar_A_ketamine[[8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list_expt[1], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar3.errorbar[1]:
    b.set_clip_on(False)
for b in bar3.errorbar[2]:
    b.set_clip_on(False)
ax.set_ylabel('Beta', fontsize=fontsize_legend)
ax.set_xlim([0.,4.+2.*bar_width_figS1])
ax.set_ylim([0.,27.])
ax.set_xticks([bar_width_figS1, 1.+bar_width_figS1, 2.+bar_width_figS1, 3.+bar_width_figS1, 4.+bar_width_figS1])
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0., 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=9.5, axis='x')
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
tmp = ax.errorbar(i_PK_list_6, PK_A_ketamine, PK_A_ketamine_errbar, color=color_list_expt[1], linestyle='-', marker='.', zorder=(3-1), clip_on=False, markeredgecolor='k', elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
tmp = ax.errorbar(i_PK_list_6, PK_A_saline, PK_A_saline_errbar, color=color_list_expt[0], linestyle='-', marker='.', zorder=(3-1), clip_on=False, markeredgecolor='k', elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
ax.set_xlabel('Sample Number', fontsize=fontsize_legend)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend)
ax.set_ylim([0.,5.4])
ax.set_yticks([0., 5.])
ax.text(0., 5.6, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
ax.set_xlim([1,6])
ax.set_xticks([1,6])
minorLocator = MultipleLocator(1.)
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
bar1 = ax.bar(np.arange(5)                , 0.5*(Reg_values_H_saline[[ 3,4,5,1,2]]-Reg_values_H_saline[[ 8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_errbar_H_saline[[ 3,4,5,1,2]]**2+Reg_values_errbar_H_saline[[ 8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list_expt[0], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar1.errorbar[1]:
    b.set_clip_on(False)
for b in bar1.errorbar[2]:
    b.set_clip_on(False)
bar3 = ax.bar(np.arange(5)+bar_width_figS1, 0.5*(Reg_values_H_ketamine[[3,4,5,1,2]]-Reg_values_H_ketamine[[8,9,10,6,7]]), bar_width_figS1, yerr=0.5*(Reg_values_errbar_H_ketamine[[3,4,5,1,2]]**2+Reg_values_errbar_H_ketamine[[8,9,10,6,7]]**2)**0.5, ecolor='k', alpha=bar_opacity, color=color_list_expt[1], clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
for b in bar3.errorbar[1]:
    b.set_clip_on(False)
for b in bar3.errorbar[2]:
    b.set_clip_on(False)
ax.set_ylabel('Beta', fontsize=fontsize_legend)
ax.set_xlim([0.,4.+2.*bar_width_figS1])
ax.set_ylim([0.,27.])
ax.set_xticks([bar_width_figS1, 1.+bar_width_figS1, 2.+bar_width_figS1, 3.+bar_width_figS1, 4.+bar_width_figS1])
ax.xaxis.set_ticklabels(['Mean', 'Max', 'Min', 'First', 'Last'])
ax.set_yticks([0., 25.])
ax.set_yticklabels([0., 0.25])
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=3.5, axis='x')
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
tmp = ax.errorbar(i_PK_list_6, PK_H_ketamine, PK_H_ketamine_errbar, color=color_list_expt[1], linestyle='-', marker='.', zorder=(3-1), clip_on=False, markeredgecolor='k', elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
tmp = ax.errorbar(i_PK_list_6, PK_H_saline, PK_H_saline_errbar, color=color_list_expt[0], linestyle='-', marker='.', zorder=(3-1), clip_on=False, markeredgecolor='k', elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
ax.set_xlabel('Sample Number', fontsize=fontsize_legend)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend)
ax.set_ylim([0.,5.4])
ax.set_yticks([0., 5.])
ax.text(0., 5.6, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
ax.set_xlim([1,6])
ax.set_xticks([1,6])
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

fig_temp.savefig(path_cwd+'Figure8S1.pdf')    #Finally save fig

