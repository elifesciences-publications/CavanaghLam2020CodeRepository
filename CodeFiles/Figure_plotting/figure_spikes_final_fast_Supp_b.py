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


label_list     = ['Control', 'More total evidence', 'Less total evidence', 'Sensory Deficit']         #Manually used variable.
color_list = [colors_set1['green'], colors_set1['purple'], colors_set1['orange'], colors_set1['brown']]
label_list_expt     = ['Control', 'Lowered E/I', 'Elevated E/I']         #Manually used variable.
color_list_expt = [colors_div3['blue4'], colors_div3['red4']]
color_list_all_high_low_total = [(0.3,0.3,0.3), colors_dark2['pink'], colors_dark2['purple']]
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
########################################################################################################################
## 2D plot: vary E/I: mean/SD regression model

gE_pert_list = np.arange(0.,1.51,0.25)
gI_pert_list = np.arange(0.,1.51,0.25)
gE_pert_mesh, gI_pert_mesh = np.meshgrid(gE_pert_list, gI_pert_list)


### beta_mesh[i,j] => gI_NMDA*(1-i*0.025), gE_NMDA*(1-j*0.025)
beta_mean_mesh_mean_std = np.array([[14.8283142997145,14.0857177629188,12.0028592401592,10.2014389538047,8.40836473267439,6.22680327581640,4.39046088248553], [13.5954473978963,14.8023330625905,14.5954754850656,13.5692622916881,11.3467229411639,8.26610271990784,6.67155161324750], [11.7081192707300,12.8320162950264,14.1430465950555,14.6775807166403,14.7754734630303,12.6536488008520,10.2610469638212], [9.73101722264681,10.6177309254892,12.0533003145445,13.7135796769792,14.1773265831697,15.1616941633943,13.6961087076376], [8.39642656668856,9.23036455584402,10.2213277755599,11.2300146529774,12.2919877727683,14.2559220475422,14.8728136676597], [7.18970781059219,8.13235529669250,8.78194819351606,9.49570988184582,10.5193405861620,11.9853840299753,13.4563837473795], [5.67429252850205,6.41436736427050,7.72862851275279,8.44784421142950,9.26980205064090,9.64035143355079,11.1248900394200]])      # i = which g_NMDA_i, j = which g_NMDA_e
beta_std_mesh_mean_std = np.array([[5.20867479806350,5.14497097350535,5.47818659513605,4.62890872918725,3.29286214365509,2.89152936619150,2.25695129737829], [4.23934278604735,4.51423385107720,5.53121145439314,5.58127586055453,5.27835135976882,4.47903173337584,3.11912354175296], [3.16160848903534,3.10962725463417,4.54755291226089,5.18682114469207,5.63830449979491,5.47765477060461,4.30500827626971], [1.40084187005132,2.46969370351230,2.89247763586771,4.46028159722481,5.21370216754289,5.22934058898037,5.34490804781560], [1.47275328583221,1.46586249656699,2.44181028136418,2.60904407847027,3.20184738167340,4.36471790055063,4.63463730040570], [0.0238612621255343,0.799090126537122,1.07938994197327,2.19170757530445,2.17997006748756,3.03824240511528,3.70333526641306], [0.308280405518696,0.160254909764317,1.13418071460817,1.01905802302903,1.17227845891242,1.96922285077769,2.69650365670580]])      # i = which g_NMDA_i, j = which g_NMDA_e
beta_ratio_mesh_mean_std = beta_std_mesh_mean_std / beta_mean_mesh_mean_std


figsize = (max2, max1)

width1_11 = 0.22; width1_12 = width1_11; width1_13 = width1_11
x1_11 = 0.09; x1_12 = x1_11 + width1_11 + xbuf0; x1_13 = x1_12 + width1_12 + xbuf0
height1_11 = 0.5; height1_12 = height1_11; height1_13 = height1_11
y1_11 = 0.2; y1_12 = y1_11; y1_13 = y1_11

# First column, upper rows
rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.025, 0.78, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.028 + x1_12 - x1_11, 0.78, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.028 + x1_13 - x1_11, 0.78, 'C', fontsize=fontsize_fig_label, fontweight='bold')



ax   = fig_temp.add_axes(rect1_11)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((beta_mean_mesh_mean_std))
vmin_pscan = np.min((beta_mean_mesh_mean_std))
plt.imshow(beta_mean_mesh_mean_std, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=vmax_pscan)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(ticks=[6,14], cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
cbar_temp.ax.set_xticklabels([0.06, 0.14])
ax.set_title("Mean evidence beta", fontsize=fontsize_legend, rotation=0, y=1.2)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)


ax   = fig_temp.add_axes(rect1_12)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((beta_std_mesh_mean_std))
vmin_pscan = np.min((beta_std_mesh_mean_std))
plt.imshow(beta_std_mesh_mean_std, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', aspect=aspect_ratio, origin='lower', vmin=0, vmax=vmax_pscan)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(ticks=[0,5], cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
cbar_temp.ax.set_xticklabels([0, 0.05])
ax.set_title("Evidence SD beta", fontsize=fontsize_legend, rotation=0, y=1.2)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)


ax   = fig_temp.add_axes(rect1_13)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((beta_ratio_mesh_mean_std))
vmin_pscan = np.min((beta_ratio_mesh_mean_std))
plt.imshow(beta_ratio_mesh_mean_std, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', aspect=aspect_ratio, origin='lower', vmin=0, vmax=vmax_pscan)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(ticks=[0,0.5], cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
cbar_temp.ax.set_xticklabels([0, 0.5])
ax.set_title("PVB Index", fontsize=fontsize_legend, rotation=0, y=1.2)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)









fig_temp.savefig(path_cwd+'Figure7S2.pdf')    #Finally save fig


########################################################################################################################
## 2D plot: vary E/I: mean/max/min/first/last regression model

gE_pert_list = np.arange(0.,1.51,0.25)
gI_pert_list = np.arange(0.,1.51,0.25)
gE_pert_mesh, gI_pert_mesh = np.meshgrid(gE_pert_list, gI_pert_list)


### beta_mesh[i,j] => gI_NMDA*(1-i*0.025), gE_NMDA*(1-j*0.025)
beta_mean_mesh = np.array([[15.2808800802566,15.1089923373834,12.9324940424945,11.0310748213719,8.93969913151857,6.90925733709290,5.02694370918977], [14.5332397121459,15.8203738131458,15.4432193644125,14.1179200438727,12.2214710926565,8.49399512198536,7.65924658572471], [12.7071695384148,13.7524431782645,14.4523695301061,15.7263780128816,15.1887371670932,14.0041259446284,11.3731521221003], [9.26343214493093,11.1824464939927,12.8222755493647,14.0710185652737,15.0881813364881,15.8099140475945,14.7740467643004], [7.47227852744405,8.51079275463486,10.1673400161660,11.6428590112856,13.0592641815801,15.0266971176842,16.2356865404010], [5.82333104890806,6.76860185668610,8.20782622015763,8.81188095769173,11.1821749521649,12.6948891876506,13.8029820511111], [3.19433562130724,4.13071250662488,6.08748766326578,7.02005314508497,8.87677361113406,9.42371428106552,11.4658434834046]])      # i = which g_NMDA_i, j = which g_NMDA_e
beta_max_mesh = np.array([[2.64266281815507,2.31641455831394,2.29275456173527,1.90863413488131,1.24007200092814,0.901085019515838,0.666347594435441], [2.04652455361425,2.09233411844685,2.56065668990047,2.59131050422467,1.92355406185920,1.92371639559342,0.876648102083406], [1.06635407263204,1.24217660028163,2.36966296028424,2.41763589855272,2.66767790311553,1.95999737594688,1.38909457433702], [0.734633497721363,0.827354192945468,1.20648916833963,2.06412083772845,2.21847868007646,2.45693805296588,2.13473417484434], [0.640855952274743,0.777747037101865,1.13117425825770,1.10335688945447,1.36611562332700,2.15838863139217,1.99587659626841], [-0.0325662306537675,0.550451959999890,0.405382214614287,1.27647035084153,0.657872222478472,1.41023513962512,2.06539171163333], [0.160411389511217,0.220186417694389,0.655763783068797,0.708598598753505,0.483259954997051,1.00226167364255,1.12265974040445]])      # i = which g_NMDA_i, j = which g_NMDA_e
beta_min_mesh = np.array([[-0.867257188731699,-1.03823007958043,-1.17308630680344,-0.983216802352010,-0.736799221420543,-0.826277056804132,-0.619969316116230], [-0.884020203460980,-1.01926245764136,-1.08057467361635,-0.930800226960613,-1.46455089872909,-0.885652411353143,-0.992309806350971], [-1.29583160694725,-0.788491447808945,-0.713012754887706,-1.10467190436349,-0.931638816346254,-1.45986445472059,-1.24598170139174], [-0.318462699790988,-0.967891900004270,-0.776915984070619,-0.888061752547590,-1.16583157000090,-0.998024985686695,-1.23240205023721], [-0.514702162565050,-0.375044383559815,-0.738860203726320,-0.728029394411471,-0.936541647242399,-0.948466245586362,-1.13928224036985], [-0.0668191968748618,-0.255240803993945,-0.510226820264416,-0.448381965749539,-0.876367094988256,-0.782807282152182,-0.548241904714330], [-0.165635249343181,-0.0216866205358192,-0.411700688691221,-0.0821416492707638,-0.546297982662083,-0.522658458334731,-0.777354957144393]])      # i = which g_NMDA_i, j = which g_NMDA_e
beta_first_mesh = np.array([[0.699831295477069,0.0729763006765326,-0.120234264560301,-0.312879036344445,-0.246745665018456,-0.327960065828938,-0.319759581361665], [1.27600376509008,0.908595562814069,0.395580128172490,-0.0308805221194074,-0.108850576779828,-0.287970351782447,-0.468626973853277], [2.54994162607873,1.93441927096578,1.29692479966289,0.433446659724211,0.211978150656592,-0.144231286000952,-0.177508572108263], [3.41329988070359,2.90484282022546,2.22285993644890,1.53023706054440,0.926716348990673,0.496148934395411,-0.0385458779011503], [4.40170426505498,3.93341996413341,3.09087978712269,2.53485193983119,1.99380291243638,1.36946068662856,0.708525132059156], [5.02408015177072,4.73483816706423,4.23598357913582,3.63765105884416,3.17225708325699,2.31424102822684,1.50759461251207], [4.56455420341898,4.99513545720077,4.98429388188063,4.64846200791663,4.07014553138605,3.30571597954657,2.68446039093910]])      # i = which g_NMDA_i, j = which g_NMDA_e
beta_last_mesh = np.array([[-1.70861296085815,-1.52476721246436,-1.33471948210232,--1.09729537835168,0.769094840345615,-0.546884094684754,-0.489861824968435], [-1.86635311443413,-1.73354917939534,-1.69362253465121,-1.44208863197444,-1.00636503484767,-0.756612055441834,-0.577492455118083], [-1.66334151585869,-1.82049994522454,-1.74382567804092,-1.67364909812762,-1.43123222768633,-1.32125231832606,-0.968877428638084], [-1.32476630088847,-1.52216683018604,-1.73043301225348,-1.64614363240937,-1.72497185973710,-1.60022149848286,-1.39199590207461], [-1.15205125661711,-1.22918540765899,-1.52278689634111,-1.66289405923624,-1.67778595762477,-1.82957327334128,-1.86251403698885], [-0.926846215184039,-0.972625954723274,-1.10359228431385,-1.41672446129241,-1.52552675326041,-1.79688530646694,-1.79798454640719], [-0.389444396091605,-0.577885336223001,-0.834152799939436,-0.918219142313886,-1.27511683446360,-1.48320682766477,-1.61842267443676]])      # i = which g_NMDA_i, j = which g_NMDA_e


figsize = (max2, 0.8*max2)

width1_11 = 0.22; width1_12 = width1_11; width1_13 = width1_11; width1_22 = width1_12; width1_23 = width1_13
x1_11 = 0.09; x1_12 = x1_11 + width1_11 + xbuf0; x1_13 = x1_12 + width1_12 + xbuf0; x1_22 = x1_12; x1_23 = x1_13
height1_11 = 0.3; height1_12 = height1_11; height1_13 = height1_11; height1_22 = height1_12; height1_23 = height1_13
y1_11 = 0.61; y1_12 = y1_11; y1_13 = y1_11; y1_22 = y1_12 - width1_22 - 3.4*ybuf0; y1_23 = y1_22

# First column, upper rows
rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.025, 0.965, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.028 + x1_12 - x1_11, 0.965, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.028 + x1_13 - x1_11, 0.965, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.028 + x1_12 - x1_11, 0.965-y1_12+y1_22, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.028 + x1_13 - x1_11, 0.965-y1_13+y1_23, 'E', fontsize=fontsize_fig_label, fontweight='bold')


ax   = fig_temp.add_axes(rect1_11)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((beta_mean_mesh))
vmin_pscan = np.min((beta_mean_mesh))
plt.imshow(beta_mean_mesh, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=vmax_pscan)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(ticks=[4,16],cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
cbar_temp.ax.set_xticklabels([0.04, 0.16])
ax.set_title("Mean evidence beta", fontsize=fontsize_legend, rotation=0, y=1.2)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)


ax   = fig_temp.add_axes(rect1_12)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((beta_max_mesh))
vmin_pscan = np.min((beta_max_mesh))
plt.imshow(beta_max_mesh, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=vmax_pscan)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(ticks=[0,2.5], cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
cbar_temp.ax.set_xticklabels([0, 0.025])
ax.set_title("Max evidence beta", fontsize=fontsize_legend, rotation=0, y=1.2)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)


ax   = fig_temp.add_axes(rect1_22)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((beta_min_mesh))
vmin_pscan = np.min((beta_min_mesh))
plt.imshow(beta_min_mesh, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=0)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(ticks=[-1.4,0], cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
cbar_temp.ax.set_xticklabels([-0.014, 0])
ax.set_title("Min evidence beta", fontsize=fontsize_legend, rotation=0, y=1.2)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)


ax   = fig_temp.add_axes(rect1_13)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((beta_first_mesh))
vmin_pscan = np.min((beta_first_mesh))
plt.imshow(beta_first_mesh, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=vmax_pscan)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(ticks=[0,5], cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
cbar_temp.ax.set_xticklabels([0, 0.05])
ax.set_title("First evidence beta", fontsize=fontsize_legend, rotation=0, y=1.2)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)


ax   = fig_temp.add_axes(rect1_23)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((beta_last_mesh))
vmin_pscan = np.min((beta_last_mesh))
plt.imshow(beta_last_mesh, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=vmax_pscan)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(ticks=[-1.5,1], cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
cbar_temp.ax.set_xticklabels([-0.015, 0.01])
ax.set_title("Last evidence beta", fontsize=fontsize_legend, rotation=0, y=1.2)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)








fig_temp.savefig(path_cwd+'Figure7S3.pdf')    #Finally save fig

################################################################################################################################################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## 1D plot: vary Sensory Deficit: mean/SD regression model

sensory_coeff_list = np.arange(0.5,1.01,0.05)
## mean/std
beta_mean_sensory_list_mean_std = np.array([0.806663984366803, 1.96698254088940, 2.95172716009682, 4.30855608909215, 6.46780264614419, 7.65579260874250, 10.0452561592003, 11.7095124776270, 13.1572358484712, 14.3570046822795, 14.5248182711777])
beta_std_sensory_list_mean_std = np.array([0.765138296645126, 0.700225882146885, 1.71955014960258, 1.91460863050387, 2.14978840008073, 3.23706201293493, 3.70771121053318, 4.30660210297212, 5.51267599665361, 5.11321404085230, 5.16925860864037])
beta_ratio_sensory_list_mean_std = beta_std_sensory_list_mean_std/beta_mean_sensory_list_mean_std
beta_err_mean_1D_list = np.array([0.211203063376827, 0.213261961054265, 0.216711423846099, 0.222645232150933, 0.235990688915342, 0.245760370224420, 0.267177407533421, 0.285300729960002, 0.301638526559982, 0.315990596108146, 0.318106104089955])
beta_err_std_1D_list = np.array([0.278902680455928, 0.279764062159353, 0.281828529838596, 0.284333660853194, 0.289950438957634, 0.295288268928176, 0.304288583698380, 0.312996229822155, 0.321150142842190, 0.325953516357008, 0.326883145824454])
beta_err_ratio_1D_list = beta_ratio_sensory_list_mean_std *( (beta_err_mean_1D_list/beta_mean_sensory_list_mean_std)**2 + (beta_err_std_1D_list/beta_std_sensory_list_mean_std)**2)**0.5
## mean/max/min/first/last
beta_mean5_1D_list = np.array([0.713427654443829, 2.29319513690491, 2.97731917234549, 5.28802259874386, 6.79387505428861, 8.08447541010282, 10.8930388985162, 12.7860568973184, 14.6090147416713, 15.5153657100341, 15.5278077836702])
beta_max_1D_list = np.array([0.325568324712177, 0.00298269813491191, 0.616814660941421, 0.414835319423735, 0.982534985987869, 1.27520271338407, 1.26590898734673, 1.43798336812816, 2.04785212889133, 2.11598966560393, 2.30816947531805])
beta_min_1D_list = np.array([-0.143676889716145, -0.318316080303628, -0.486130126219293, -0.729830832061071, -0.408383824774330, -0.735065402034368, -0.943757872142494, -1.27954350412993, -1.47758040686613, -1.27028744387289, -1.06863019293379])
beta_first_1D_list = np.array([0.0630787105912960, -0.0812220041869995, -0.0715541458466962, -0.301664176497266, -0.367828501298767, -0.286531908608680, -0.284177431248157, -0.0812262581204547, 0.0396368154694454, 0.365752988094059, 0.522646827103653])
beta_last_1D_list = np.array([-0.143170800528189, -0.0654275755809390, -0.148018519455433, -0.513405135487748, -0.530065825470294, -0.656347669928916, -0.800443150775371, -1.01778072892580, -1.53640940300070, -1.55611313939169, -1.73111224846866])

beta_err_mean5_1D_list = np.array([0.334940652700139, 0.336779112816374, 0.339802711817324, 0.347350734797669, 0.357717691904035, 0.366547298408930, 0.386080275348329, 0.403517111892725, 0.421647086811546, 0.433250060432904, 0.436218484922253])
beta_err_max_1D_list = np.array([0.167611924119035, 0.168042868583514, 0.169163177970550, 0.171113599074801, 0.174750794021354, 0.177567357494457, 0.183093917736637, 0.188564889274540, 0.194257229467401, 0.198324462077186, 0.199996250512873])
beta_err_min_1D_list = np.array([0.167465797137579, 0.167940326521538, 0.169145102992984, 0.171335099358934, 0.174773417769279, 0.177732844024780, 0.183538619151276, 0.189121747885944, 0.194743425614851, 0.198579423722022, 0.199763806760975])
beta_err_first_1D_list = np.array([0.0765431696614665, 0.0767921333637020, 0.0773340618243902, 0.0784112520024727, 0.0801948925480768, 0.0815148674663486, 0.0842628347290575, 0.0867415533743931, 0.0892507312820966, 0.0911111050414426, 0.0917892880093069])
beta_err_last_1D_list = np.array([0.0766700689027265, 0.0769059472566093, 0.0774546406020449, 0.0786291275842147, 0.0803319574532951, 0.0818174501126638, 0.0846642770166747, 0.0875092805145178, 0.0909345567432287, 0.0928033869932776, 0.0937868926834865])


## Monkey data, with lapse (beta0, mean_beta, std_beta, lapse rate)
Reg_bars_A_ketamine = np.array([0.875741073112642, 11.528262865855169, 5.233422331691739])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on ketamine.
Reg_bars_H_ketamine = np.array([-0.054946117132906, 11.298580759143913, 3.616936576677555])  # [Bias, Val diff , Std diff]. Henry regression Beta values on ketamine.
Reg_bars_A_saline = np.array([0.123672806343438, 25.565433692118590, 5.762095065977849])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on saline.
Reg_bars_H_saline = np.array([-0.015314326931326, 22.141685413467790, 3.313026659905981])  # [Bias, Val diff , Std diff]. Henry regression Beta values on saline.




## Define subfigure domain.
figsize = (max15,0.7*max15)

width1_11 = 0.2; width1_12 = width1_11; width1_13 = width1_11
width1_21 = width1_11; width1_22 = width1_21; width1_23 = width1_21
x1_11 = 0.1; x1_12 = x1_11 + width1_11 + 1.25*xbuf0; x1_13 = x1_12 + width1_12 + 1.25*xbuf0
x1_21 = x1_11; x1_22 = x1_12; x1_23 = x1_13
height1_11 = 0.3; height1_12 = height1_11; height1_13 = height1_11
height1_21= height1_11;  height1_22 = height1_21;  height1_23 = height1_21
y1_11 = 0.62; y1_12 = y1_11; y1_13 = y1_11
y1_21 = y1_11 - height1_21 - 2.35*ybuf0; y1_22 = y1_21; y1_23 = y1_21

rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]



##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.01, 0.935, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.01+x1_12-x1_11, 0.935, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.005+x1_13-x1_11, 0.935, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.01, 0.935 + y1_22 - y1_12, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.005+x1_22-x1_21, 0.935 + y1_22 - y1_12, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(-0.005+x1_23-x1_21, 0.935 + y1_22 - y1_12, 'F', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.




## rect1_11: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax)
tmp = ax.errorbar(1.-sensory_coeff_list, beta_mean_sensory_list_mean_std, beta_err_mean_1D_list, color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', ecolor=color_list[3], linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.axvline(1.-sensory_coeff_list[-5], color='k', linestyle=":", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Mean evidence beta', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylim([0.,15.1])
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,50])
ax.set_yticks([0., 15.])
ax.text(0.43-0.5, 15.5, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
# ax.tick_params(bottom="off")
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_12: Evidence Std beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax)
tmp = ax.errorbar(1.-sensory_coeff_list, beta_std_sensory_list_mean_std, beta_err_std_1D_list, color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', ecolor=color_list[3], linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.axvline(1.-sensory_coeff_list[-5], color='k', linestyle=":", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Evidence std beta', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylim([0.,6.05])
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,50])
ax.set_yticks([0., 6.])
ax.text(0.43-0.5, 6.2, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(2.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_13: PVB Index vs sensory coefficient
ax   = fig_temp.add_axes(rect1_13)
fig_funs.remove_topright_spines(ax)
tmp = ax.errorbar(1.-sensory_coeff_list, beta_ratio_sensory_list_mean_std, beta_err_ratio_1D_list, color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', ecolor=color_list[3], linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    print b
    b.set_clip_on(False)
ax.axvline(1.-sensory_coeff_list[-5], color='k', linestyle=":", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('PVB index', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylim([0.28,1.2])
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,50])
ax.set_yticks([0.2, 1.4])
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_21: Mean (in mean/max/min/first/last) Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
tmp = ax.errorbar(1.-sensory_coeff_list, beta_mean5_1D_list, beta_err_mean5_1D_list, color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', ecolor=color_list[3], linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.axvline(1.-sensory_coeff_list[-5], color='k', linestyle=":", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Mean evidence beta', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylim([0.,16.])
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,50])
ax.set_yticks([0., 15.])
ax.text(0.43-0.5, 16.5, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(5.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_22: Max/Min Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
tmp = ax.errorbar(1.-sensory_coeff_list, beta_max_1D_list, beta_err_max_1D_list, color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', ecolor=color_list[3], linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax.errorbar(1.-sensory_coeff_list, beta_min_1D_list, beta_err_min_1D_list, color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', ecolor=color_list[3], linestyle='--', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.axvline(1.-sensory_coeff_list[-5], color='k', linestyle=":", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Max/min evidence beta', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylim([-3.,3.])
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,50])
ax.set_yticks([-3., 0., 3.])
ax.text(0.43-0.5, 3.15, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.text(0.95-0.63, 1.2, 'Max', fontsize=fontsize_tick)
ax.text(0.95-0.63, -1.63, 'Min', fontsize=fontsize_tick)

## rect1_23: First/Last Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
tmp = ax.errorbar(1.-sensory_coeff_list, beta_first_1D_list, beta_err_first_1D_list, color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', ecolor=color_list[3], linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
tmp = ax.errorbar(1.-sensory_coeff_list, beta_last_1D_list, beta_err_last_1D_list, color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', ecolor=color_list[3], linestyle='--', marker='.', zorder=(3-1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
for b in tmp[2]:
    b.set_clip_on(False)
ax.axvline(1.-sensory_coeff_list[-5], color='k', linestyle=":", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('First/last evidence beta', fontsize=fontsize_legend, labelpad=2.)
ax.set_ylim([-2.,2.])
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,50])
ax.set_yticks([-2.,0.,2.])
ax.text(0.43-0.5, 2.1, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.text(0.05, 0.55, 'First', fontsize=fontsize_tick)
ax.text(0.05, -2.05, 'Last', fontsize=fontsize_tick)





fig_temp.savefig(path_cwd+'Figure7S4.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## 1D plot: vary Sensory Deficit: mean/SD regression model


sensory_coeff_list = np.arange(0,1.01,0.1)

KL_A_ket_sensory_list = np.array([1.14690924198911, 1.16304293814147, 1.07754593099122, 1.10804162572097, 1.02426354856181, 0.746390759018919, 0.477599196391312, 0.123225290239455, 0.0413560959209905, 0.433302511674715, 0.425582213746040])
KL_A_saline_sensory_list = np.array([3.15872185172653, 3.25579588800771, 3.18992729066383, 3.15628653519065, 3.10938064060767, 2.63792379102361, 1.87836704778202, 1.00358646191499, 0.410381938415099, 0.0928781050199246, -0.0218626940710304])
KL_H_ket_sensory_list = np.array([1.22423335780349, 1.26753228782944, 1.20706169699907, 1.21359973015495, 1.13666029459817, 0.797057465921836, 0.362024651664349, -0.133262903337299, -0.351926058527127, -0.215872973257652, -0.226158167049481])
KL_H_saline_sensory_list = np.array([2.66374579394794, 2.76081142835735, 2.69624937603612, 2.66448699027334, 2.62050156765695, 2.16316438607728, 1.43631966216555, 0.605806047608025, 0.0581046960741349, -0.227837064253640, -0.330047133002144])

KL_A_ket_gNMDAE_list = KL_A_ket_matrix[0,:]
KL_A_saline_gNMDAE_list = KL_A_saline_matrix[0,:]
KL_H_ket_gNMDAE_list = KL_H_ket_matrix[0,:]
KL_H_saline_gNMDAE_list = KL_H_saline_matrix[0,:]

KL_A_ket_gNMDAI_list = KL_A_ket_matrix[:,0]
KL_A_saline_gNMDAI_list = KL_A_saline_matrix[:,0]
KL_H_ket_gNMDAI_list = KL_H_ket_matrix[:,0]
KL_H_saline_gNMDAI_list = KL_H_saline_matrix[:,0]

## P_corr with lapse rate fitted to monkey data. (N.B.: All KL_sensory data is using 20/10k/10k data)
KL_A_ket_sensory_list = np.array([1.14824049871608, 1.16153808081683, 1.09820556863969, 1.12144295289860, 1.05947490763764, 0.849130868568747, 0.608805815985327, 0.269475473085581, 0.107368461951745, 0.133937327520534, 0.118784635479593])
KL_A_saline_sensory_list = np.array([3.15872185172653, 3.25579588800771, 3.18992729066383, 3.15628653519065, 3.10938064060767, 2.63792379102361, 1.87836704778202, 1.00358646191499, 0.410381938415099, 0.0928781050199246, -0.0218626940710304])
KL_H_ket_sensory_list = np.array([2.66616701449489, 2.76033600370706, 2.69785123995055, 2.66720719092989, 2.62459442021816, 2.18097954569267, 1.47359107492903, 0.661945871298781, 0.124155388026141, -0.158913994097337, -0.261197114804899])
KL_H_saline_sensory_list = np.array([1.23060061731249, 1.26694081288943, 1.21748606750782, 1.22331151934656, 1.16042327741282, 0.878992377638849, 0.498049313634094, 0.0432686051461402, -0.199311701835304, -0.231128543398768, -0.253425700301691])

# KL Divergence with monkey-fitted lapse rate to P_corr.
KL_A_ket_matrix = np.array([[0.0612315946710040, 0.0726028995577782, -0.0251084899858666, 0.00534924491674638, 0.136754661192464, 0.135594076138256, 0.307478725566939], [0.167529755210966, 0.203611788373683, 0.174528037093989, 0.0365866508445205, 0.0485325659747548, -0.0726795268458215, 0.0721596102977752], [0.217906113922143, 0.318143365778266, 0.230416622277383, 0.143275913377823, 0.146194683336421, -0.0557653552566165, -0.0539437042477977], [0.368158699801681, 0.292793758102799, 0.255330384794741, 0.214261784853862, 0.194716972590461, 0.119945861456634, -0.0578696096685738], [0.397659534593832, 0.406683260271066, 0.361618175848413, 0.327811391415628, 0.246244356202888, 0.222623638880258, 0.112460672112397], [0.569963791838774, 0.401899966871877, 0.346895442968578, 0.294846746923008, 0.260666996586305, 0.200090549184159, 0.279148059645666], [0.660960343637186, 0.504132582573355, 0.391599683913586, 0.420482997433917, 0.415114577720821, 0.322828710324859, 0.166144658888200]])
KL_A_saline_matrix = np.array([[-0.0266687577981276, -0.0400026829194878, 0.0405090993669350, 0.187237643544449, 0.498389795541426, 0.854787449635861, 1.28564122743175], [0.151400403748287, 0.109962253082262, -0.00451387886298954, -0.0318162936378029, 0.196645455982128, 0.401129323482682, 0.711096265252436], [0.373931700411290, 0.325205658467795, 0.190369324103699, 0.0393344197159249, -0.0279859867429781, 0.00535878164557060, 0.200045808300191], [0.745740936412951, 0.516587236815852, 0.382314548116739, 0.192893371286148, 0.0855663261959165, -0.0239104948800873, -0.0919111376246004], [0.914683012211360, 0.835447532590657, 0.673704387595393, 0.537640571612821, 0.336621887336944, 0.0707035385576934, -0.0185436795627039], [1.32153380683123, 1.01243676403186, 0.862980535886011, 0.693550635948604, 0.514751834829099, 0.315708661308230, 0.240353682033922], [1.56882889383798, 1.32780577102178, 1.01883475646907, 0.932998259589351, 0.871837119338533, 0.657994964048574, 0.349555915233894]])
KL_H_ket_matrix = np.array([[-0.292651281803015, -0.303949276985044, -0.374017964328443, -0.342783942553787, -0.184615455238794, -0.106118241849367, 0.120265757244502], [-0.172169829542740, -0.152369760300412, -0.217323862650282, -0.326325710222028, -0.302280689847145, -0.345972009412987, -0.191219578313784], [-0.0993458419261488, -0.0266929148409120, -0.121714588806339, -0.217460571638196, -0.239763128715429, -0.398000304132147, -0.378560720032967], [0.106351592987056, -0.0144155946515693, -0.0561848307017724, -0.129312606942955, -0.175380544695030, -0.245989392180559, -0.415031271719360], [0.155620947698366, 0.137952342576492, 0.0719762465472689, 0.0234622928108007, -0.0869970752872093, -0.153359075820873, -0.257517743945902], [0.375185790342860, 0.182599161989311, 0.107890600795115, 0.0298361272311121, -0.0370375326760692, -0.119933514750083, -0.0634482930559727], [0.494520338367837, 0.321587342251002, 0.169693921858130, 0.173328114463513, 0.158470030134335, 0.0372443205665730, -0.149589233217907]])
KL_H_saline_matrix = np.array([[-0.253886197994285, -0.279176821541068, -0.220270529272230, -0.103687933292307, 0.184913489641061, 0.517239510973172, 0.906895935662233], [-0.0967009518847820, -0.138632488532080, -0.245870477592834, -0.283020131811403, -0.0811809296165914, 0.0973815250394336, 0.382665582489545], [0.111055825388297, 0.0706759111511026, -0.0611797953092704, -0.196779594695172, -0.265425885430623, -0.248706914528925, -0.0892691329559575], [0.449039699308101, 0.231144846437257, 0.125179634495036, -0.0589064063325154, -0.153469790550522, -0.261194245613257, -0.333313524251169], [0.600619239135627, 0.532722800309865, 0.385563135151303, 0.268615284111922, 0.0766787515176806, -0.169170234233506, -0.236134461619567], [0.977456271083573, 0.684933790267839, 0.565283685523998, 0.403528112440717, 0.239288109193070, 0.0614545460570743, -0.00911997854998435], [1.20097211315899, 0.977167267912175, 0.701609272321701, 0.623616933537866, 0.572904986592398, 0.362440125677467, 0.0843658272447415]])




## Define subfigure domain.
figsize = (max2,1.*max2)


### 4x3
width1_11 = 0.25; width1_12 = 0.16; width1_13 = 0.13
width1_21 = width1_11; width1_22 = width1_12; width1_23 = width1_13
width1_31 = width1_11; width1_32 = width1_12; width1_33 = width1_13
width1_41 = width1_11; width1_42 = width1_12; width1_43 = width1_13
x1_11 = 0.13; x1_12 = x1_11 + width1_11 + 1.2*xbuf0; x1_13 = x1_12 + width1_12 + 1.*xbuf0
x1_21 = x1_11; x1_22 = x1_12; x1_23 = x1_13
x1_31 = x1_11; x1_32 = x1_12; x1_33 = x1_13
x1_41 = x1_11; x1_42 = x1_12; x1_43 = x1_13
height1_11 = 0.15; height1_12 = height1_11; height1_13 = height1_11
height1_21= height1_11;  height1_22 = height1_21;  height1_23 = height1_21
height1_31= height1_11;  height1_32 = height1_21;  height1_33 = height1_21
height1_41= height1_11;  height1_42 = height1_21;  height1_43 = height1_21
y1_11 = 0.8; y1_12 = y1_11; y1_13 = y1_11
y1_21 = y1_11 - height1_21 - 1.1*ybuf0; y1_22 = y1_21; y1_23 = y1_21
y1_31 = y1_21 - height1_31 - 1.1*ybuf0; y1_32 = y1_31; y1_33 = y1_31
y1_41 = y1_31 - height1_41 - 1.1*ybuf0; y1_42 = y1_41; y1_43 = y1_41


rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_31 = [x1_31, y1_31, width1_31, height1_31]
rect1_32 = [x1_32, y1_32, width1_32, height1_32]
rect1_33 = [x1_33, y1_33, width1_33, height1_33]
rect1_41 = [x1_41, y1_41, width1_41, height1_41]
rect1_42 = [x1_42, y1_42, width1_42, height1_42]
rect1_43 = [x1_43, y1_43, width1_43, height1_43]



##### Plotting  (4x3)
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.098, 0.95, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.09-0.015+x1_12-x1_11, 0.95, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.098-0.015+x1_13-x1_11, 0.95, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.098, 0.95 + y1_22 - y1_12, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.083-0.015+x1_22-x1_21, 0.95 + y1_22 - y1_12, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.098-0.015+x1_23-x1_21, 0.95 + y1_22 - y1_12, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.098, 0.95 + y1_32 - y1_12, 'G', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.083-0.015+x1_32-x1_31, 0.95 + y1_32 - y1_12, 'H', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.098-0.015+x1_33-x1_31, 0.95 + y1_32 - y1_12, 'I', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.098, 0.95 + y1_42 - y1_12, 'J', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.083-0.015+x1_42-x1_41, 0.95 + y1_42 - y1_12, 'K', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.098-0.015+x1_43-x1_41, 0.95 + y1_42 - y1_12, 'L', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.05, 0.888, 'Monkey A\nsaline', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k', horizontalalignment='center')
fig_temp.text(0.05, 0.651, 'Monkey H\nsaline', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k', horizontalalignment='center')
fig_temp.text(0.05, 0.412, 'Monkey A\nketamine', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k', horizontalalignment='center')
fig_temp.text(0.05, 0.175, 'Monkey H\nketamine', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k', horizontalalignment='center')
bar_width_compare3 = 1.


KL_min_monkey_A_saline = np.min(np.array([np.min(KL_A_saline_gNMDAE_list), np.min(KL_A_saline_gNMDAI_list), np.min(KL_A_saline_sensory_list)]))
KL_max_monkey_A_saline = np.max(np.array([np.max(KL_A_saline_gNMDAE_list), np.max(KL_A_saline_gNMDAI_list), np.max(KL_A_saline_sensory_list)]))



ax   = fig_temp.add_axes(rect1_11)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((KL_A_saline_matrix))
vmin_pscan = np.min((KL_A_saline_matrix))
plt.imshow(KL_A_saline_matrix, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', cmap=matplotlib_cm.viridis_r, aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=vmax_pscan)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("right", size="5%", pad=0.07)
cbar_temp = plt.colorbar(cax=cax_scale_bar_size)
cbar_temp.ax.tick_params(axis='y', direction='out')
for label in cbar_temp.ax.yaxis.get_ticklabels()[1:-1]:
    label.set_visible(False)
ax.set_title("KL divergence", fontsize=fontsize_legend, rotation=90, x=1.47,y=0.72)


ax   = fig_temp.add_axes(rect1_31)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((KL_A_ket_matrix))
vmin_pscan = np.min((KL_A_ket_matrix))
plt.imshow(KL_A_ket_matrix, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', cmap=matplotlib_cm.viridis_r, aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=1.0)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("right", size="5%", pad=0.07)
cbar_temp = plt.colorbar(cax=cax_scale_bar_size)
cbar_temp.ax.tick_params(axis='y', direction='out')
for label in cbar_temp.ax.yaxis.get_ticklabels()[1:-1]:
    label.set_visible(False)
ax.set_title("KL divergence", fontsize=fontsize_legend, rotation=90, x=1.47,y=0.72)

ax   = fig_temp.add_axes(rect1_21)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((KL_H_saline_matrix))
vmin_pscan = np.min((KL_H_saline_matrix))
plt.imshow(KL_H_saline_matrix, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', cmap=matplotlib_cm.viridis_r, aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=vmax_pscan)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("right", size="5%", pad=0.07)
cbar_temp = plt.colorbar(cax=cax_scale_bar_size)
cbar_temp.ax.tick_params(axis='y', direction='out')
for label in cbar_temp.ax.yaxis.get_ticklabels()[1:-1]:
    label.set_visible(False)
ax.set_title("KL divergence", fontsize=fontsize_legend, rotation=90, x=1.47,y=0.77)

ax   = fig_temp.add_axes(rect1_41)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((KL_H_ket_matrix))
vmin_pscan = np.min((KL_H_ket_matrix))
plt.imshow(KL_H_ket_matrix, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', cmap=matplotlib_cm.viridis_r, aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=-vmin_pscan)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("right", size="5%", pad=0.07)
cbar_temp = plt.colorbar(cax=cax_scale_bar_size)
cbar_temp.ax.tick_params(axis='y', direction='out')
for label in cbar_temp.ax.yaxis.get_ticklabels()[1:-1]:
    label.set_visible(False)
ax.set_title("KL divergence", fontsize=fontsize_legend, rotation=90, x=1.47,y=0.72)




## rect1_13: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(1.-sensory_coeff_list[5:], KL_A_saline_sensory_list[5:], color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.axhline(KL_A_saline_sensory_list[-1], color=color_list[0], linestyle="--", zorder=1, lw=1, clip_on=False)
ax.set_xlabel('Sensory deficit', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('KL divergence', fontsize=fontsize_legend, labelpad=2.)
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,0.5])
ax.set_ylim([0,3])
ax.set_yticks([0., 3.])
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))


## rect1_23: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(1.-sensory_coeff_list[5:], KL_H_saline_sensory_list[5:], color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.axhline(KL_H_saline_sensory_list[-1], color=color_list[0], linestyle="--", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('KL divergence', fontsize=fontsize_legend, labelpad=-2.)
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,0.5])
ax.set_ylim([-0.5,2.])
ax.set_yticks([-0.5, 2.])
ax.set_yticklabels([-0.5, 2])
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.5)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_33: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_32)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(1.-sensory_coeff_list[5:], KL_A_ket_sensory_list[5:], color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.axhline(np.min(KL_A_ket_gNMDAE_list), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('KL divergence', fontsize=fontsize_legend, labelpad=-4.)
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,0.5])
ax.set_ylim([-0.1,0.9])
ax.set_yticks([-0.1,0.9])
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))



## rect1_43: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_42)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(1.-sensory_coeff_list[5:], KL_H_ket_sensory_list[5:], color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.axhline(np.min(KL_H_ket_gNMDAE_list), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('KL divergence', fontsize=fontsize_legend, labelpad=-4.)
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,0.5])
ax.set_ylim([-0.5,2.5])
ax.set_yticks([-0.5, 2.5])
ax.set_yticklabels([-0.5, 2.5])
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.5)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))




ax   = fig_temp.add_axes(rect1_13)
fig_funs.remove_topright_spines(ax)
ax.spines['bottom'].set_visible(False)
ax.plot(0.5, KL_A_saline_matrix[0,0]      , color=color_list[0], markerfacecolor=color_list[0], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_A_saline_matrix[0,3]      , color=color_list[1], markerfacecolor=color_list[1], markeredgecolor='k', linestyle='-', marker='.', zorder=(4-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_A_saline_matrix[3,0]      , color=color_list[2], markerfacecolor=color_list[2], markeredgecolor='k', linestyle='-', marker='.', zorder=(1-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_A_saline_sensory_list[8]  , color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(2-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('KL divergence', fontsize=fontsize_legend, labelpad=-4.)
ax.set_xlim([0.,1.5])
ax.set_ylim([-0.1, 0.8])
ax.set_yticks([0., 0.8])
ax.yaxis.set_ticklabels([0, 0.8])
ax.set_xticks([])
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.tick_params(axis='x', direction='out', pad=2.)
ax.tick_params(axis='y', direction='out', pad=1.)
ax.spines['bottom'].set_position(('zero'))
ax.text(0.8, KL_A_saline_matrix[0,0]-0.008, 'Control', fontsize=fontsize_legend, color=color_list[0], verticalalignment='center')
ax.text(0.8, KL_A_saline_matrix[0,3]-0.008, 'Lowered E/I', fontsize=fontsize_legend, color=color_list[1], verticalalignment='center')
ax.text(0.8, KL_A_saline_matrix[3,0]-0.008, 'Elevated E/I', fontsize=fontsize_legend, color=color_list[2], verticalalignment='center')
ax.text(0.8, KL_A_saline_sensory_list[8]-0.008, 'Sensory Deficit', fontsize=fontsize_legend, color=color_list[3], verticalalignment='center')

ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
ax.spines['bottom'].set_visible(False)
ax.plot(0.5, KL_H_saline_matrix[0,0]      , color=color_list[0], markerfacecolor=color_list[0], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_H_saline_matrix[0,3]      , color=color_list[1], markerfacecolor=color_list[1], markeredgecolor='k', linestyle='-', marker='.', zorder=(4-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_H_saline_matrix[3,0]      , color=color_list[2], markerfacecolor=color_list[2], markeredgecolor='k', linestyle='-', marker='.', zorder=(1-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_H_saline_sensory_list[8]  , color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(2-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('KL divergence', fontsize=fontsize_legend, labelpad=-4.)
ax.set_xlim([0.,1.5])
ax.set_ylim([-0.3, 0.5])
ax.set_yticks([-0.2, 0.4])
ax.yaxis.set_ticklabels([-0.2, 0.4])
ax.set_xticks([])
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.tick_params(axis='x', direction='out', pad=2.)
ax.tick_params(axis='y', direction='out', pad=1.)
ax.spines['bottom'].set_position(('zero'))
ax.text(0.8, KL_H_saline_matrix[0,0]-0.02, 'Control', fontsize=fontsize_legend, color=color_list[0], verticalalignment='center')
ax.text(0.8, KL_H_saline_matrix[0,3]-0.006, 'Lowered E/I', fontsize=fontsize_legend, color=color_list[1], verticalalignment='center')
ax.text(0.8, KL_H_saline_matrix[3,0]-0.008, 'Elevated E/I', fontsize=fontsize_legend, color=color_list[2], verticalalignment='center')
ax.text(0.8, KL_H_saline_sensory_list[8]+0.002, 'Sensory Deficit', fontsize=fontsize_legend, color=color_list[3], verticalalignment='center')

ax   = fig_temp.add_axes(rect1_33)
fig_funs.remove_topright_spines(ax)
ax.spines['bottom'].set_visible(False)
ax.plot(0.5, KL_A_ket_matrix[0,0]      , color=color_list[0], markerfacecolor=color_list[0], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_A_ket_matrix[0,3]      , color=color_list[1], markerfacecolor=color_list[1], markeredgecolor='k', linestyle='-', marker='.', zorder=(4-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_A_ket_matrix[3,0]      , color=color_list[2], markerfacecolor=color_list[2], markeredgecolor='k', linestyle='-', marker='.', zorder=(1-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_A_ket_sensory_list[8]  , color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(2-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('KL divergence', fontsize=fontsize_legend, labelpad=-4.)
ax.set_xlim([0.,1.5])
ax.set_ylim([-0.02, 0.4])
ax.set_yticks([0., 0.4])
ax.yaxis.set_ticklabels([0., 0.4])
ax.set_xticks([])
minorLocator = MultipleLocator(0.1)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.tick_params(axis='x', direction='out', pad=2.)
ax.tick_params(axis='y', direction='out', pad=1.)
ax.spines['bottom'].set_position(('zero'))
ax.text(0.8, KL_A_ket_matrix[0,0]-0.004, 'Control', fontsize=fontsize_legend, color=color_list[0], verticalalignment='center')
ax.text(0.8, KL_A_ket_matrix[0,3]-0.004, 'Lowered E/I', fontsize=fontsize_legend, color=color_list[1], verticalalignment='center')
ax.text(0.8, KL_A_ket_matrix[3,0]-0.004, 'Elevated E/I', fontsize=fontsize_legend, color=color_list[2], verticalalignment='center')
ax.text(0.8, KL_A_ket_sensory_list[8]-0.004, 'Sensory Deficit', fontsize=fontsize_legend, color=color_list[3], verticalalignment='center')

ax   = fig_temp.add_axes(rect1_43)
fig_funs.remove_topright_spines(ax)
ax.spines['bottom'].set_visible(False)
ax.plot(0.5, KL_H_ket_matrix[0,0]      , color=color_list[0], markerfacecolor=color_list[0], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_H_ket_matrix[0,3]      , color=color_list[1], markerfacecolor=color_list[1], markeredgecolor='k', linestyle='-', marker='.', zorder=(4-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_H_ket_matrix[3,0]      , color=color_list[2], markerfacecolor=color_list[2], markeredgecolor='k', linestyle='-', marker='.', zorder=(1-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.plot(0.5, KL_H_ket_sensory_list[8]  , color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(2-1), clip_on=False, alpha=1., markeredgewidth=0.6, markersize=12)#, linestyle=linestyle_list[i_var_a])
ax.set_ylabel('KL divergence', fontsize=fontsize_legend, labelpad=-4.)
ax.set_xlim([0.,1.5])
ax.set_ylim([-0.4, 0.2])
ax.set_yticks([-0.4, 0.2])
ax.yaxis.set_ticklabels([-0.4, 0.2])
ax.set_xticks([])
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
ax.tick_params(axis='x', direction='out', pad=2.)
ax.tick_params(axis='y', direction='out', pad=1.)
ax.spines['bottom'].set_position(('zero'))
ax.text(0.8, KL_H_ket_matrix[0,0]-0.00, 'Control', fontsize=fontsize_legend, color=color_list[0], verticalalignment='center')
ax.text(0.8, KL_H_ket_matrix[0,3]-0.01, 'Lowered E/I', fontsize=fontsize_legend, color=color_list[1], verticalalignment='center')
ax.text(0.8, KL_H_ket_matrix[3,0]-0.03, 'Elevated E/I', fontsize=fontsize_legend, color=color_list[2], verticalalignment='center')
ax.text(0.8, KL_H_ket_sensory_list[8]+0.015, 'Sensory Deficit', fontsize=fontsize_legend, color=color_list[3], verticalalignment='center')


fig_temp.savefig(path_cwd+'Figure8S6.pdf')    #Finally save fig

########################################################################################################################
################################################################################################################################################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
## 2D plot: 2D E/I pscan goodness of fit, KL divergence

gE_pert_list = np.arange(0.,1.51,0.25)
gI_pert_list = np.arange(0.,1.51,0.25)
gE_pert_mesh, gI_pert_mesh = np.meshgrid(gE_pert_list, gI_pert_list)



## Monkey data, with lapse (beta0, mean_beta, std_beta, lapse rate)
Reg_bars_A_ketamine = np.array([0.875741073112642, 11.528262865855169, 5.233422331691739])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on ketamine.
Reg_bars_H_ketamine = np.array([-0.054946117132906, 11.298580759143913, 3.616936576677555])  # [Bias, Val diff , Std diff]. Henry regression Beta values on ketamine.
Reg_bars_A_saline = np.array([0.123672806343438, 25.565433692118590, 5.762095065977849])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on saline.
Reg_bars_H_saline = np.array([-0.015314326931326, 22.141685413467790, 3.313026659905981])  # [Bias, Val diff , Std diff]. Henry regression Beta values on saline.

## KL Divergence with monkey-fitted lapse rate to P_corr.
cos_sim_A_ket_2D_EI_matrix = ((beta_mean_mesh_mean_std - beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0]*(Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1] + (beta_std_mesh_mean_std - beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0]*(Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2]) / ((((beta_mean_mesh_mean_std - beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0])**2 + ((beta_std_mesh_mean_std - beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0])**2)**0.5 * (((Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1])**2 + ((Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2])**2)**0.5)
cos_sim_H_ket_2D_EI_matrix = ((beta_mean_mesh_mean_std - beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0]*(Reg_bars_H_ketamine[1]-Reg_bars_H_saline[1])/Reg_bars_H_saline[1] + (beta_std_mesh_mean_std - beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0]*(Reg_bars_H_ketamine[2]-Reg_bars_H_saline[2])/Reg_bars_H_saline[2]) / ((((beta_mean_mesh_mean_std - beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0])**2 + ((beta_std_mesh_mean_std - beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0])**2)**0.5 * (((Reg_bars_H_ketamine[1]-Reg_bars_H_saline[1])/Reg_bars_H_saline[1])**2 + ((Reg_bars_H_ketamine[2]-Reg_bars_H_saline[2])/Reg_bars_H_saline[2])**2)**0.5)
cos_sim_A_ket_2D_EI_matrix[0,0] = 2
cos_sim_H_ket_2D_EI_matrix[0,0] = 2


cos_sim_A_ket_gNMDAE_list = cos_sim_A_ket_2D_EI_matrix[0,:]
cos_sim_H_ket_gNMDAE_list = cos_sim_H_ket_2D_EI_matrix[0,:]

cos_sim_A_ket_gNMDAI_list = cos_sim_A_ket_2D_EI_matrix[:,0]
cos_sim_H_ket_gNMDAI_list = cos_sim_H_ket_2D_EI_matrix[:,0]

## P_corr with lapse rate fitted to monkey data. (N.B.: All cos_sim_sensory data is using 20/10k/10k data)
cos_sim_A_ket_sensory_list = ((beta_mean_sensory_list_mean_std - beta_mean_sensory_list_mean_std[-1])/beta_mean_sensory_list_mean_std[-1]*(Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1] + (beta_std_sensory_list_mean_std - beta_std_sensory_list_mean_std[-1])/beta_std_sensory_list_mean_std[-1]*(Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2])  / ((((beta_mean_sensory_list_mean_std - beta_mean_sensory_list_mean_std[-1])/beta_mean_sensory_list_mean_std[-1])**2 + ((beta_std_sensory_list_mean_std - beta_std_sensory_list_mean_std[-1])/beta_std_sensory_list_mean_std[-1])**2)**0.5 * (((Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1])**2 + ((Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2])**2)**0.5)
cos_sim_H_ket_sensory_list = ((beta_mean_sensory_list_mean_std - beta_mean_sensory_list_mean_std[-1])/beta_mean_sensory_list_mean_std[-1]*(Reg_bars_H_ketamine[1]-Reg_bars_H_saline[1])/Reg_bars_H_saline[1] + (beta_std_sensory_list_mean_std - beta_std_sensory_list_mean_std[-1])/beta_std_sensory_list_mean_std[-1]*(Reg_bars_H_ketamine[2]-Reg_bars_H_saline[2])/Reg_bars_H_saline[2])  / ((((beta_mean_sensory_list_mean_std - beta_mean_sensory_list_mean_std[-1])/beta_mean_sensory_list_mean_std[-1])**2 + ((beta_std_sensory_list_mean_std - beta_std_sensory_list_mean_std[-1])/beta_std_sensory_list_mean_std[-1])**2)**0.5 * (((Reg_bars_H_ketamine[1]-Reg_bars_H_saline[1])/Reg_bars_H_saline[1])**2 + ((Reg_bars_H_ketamine[2]-Reg_bars_H_saline[2])/Reg_bars_H_saline[2])**2)**0.5)

print(np.max(cos_sim_A_ket_gNMDAE_list[1:]))
print(np.max(cos_sim_H_ket_gNMDAE_list[1:]))



figsize = (max2, 1.*max2)

width1_11 = 0.22; width1_12 = width1_11; width1_13 = width1_11; width1_21 = 0.19; width1_22 = width1_21; width1_23 = width1_21; width1_31 = width1_21; width1_32 = width1_22; width1_33 = width1_23
x1_11 = 0.09; x1_12 = x1_11 + width1_11 + xbuf0; x1_13 = x1_12 + width1_12 + xbuf0; x1_21 = 0.15; x1_22 = x1_21 + width1_21 + xbuf0; x1_23 = x1_22 + width1_22 + xbuf0; x1_31 = x1_21; x1_32 = x1_22; x1_33 = x1_23
height1_11 = 0.23; height1_12 = height1_11; height1_13 = height1_11; height1_21 = 0.2; height1_22 = height1_21; height1_23 = height1_21; height1_31 = height1_21; height1_32 = height1_21; height1_33 = height1_21
y1_11 = 0.7; y1_12 = y1_11; y1_13 = y1_11; y1_22 = y1_12 - height1_22 - 1.65*ybuf0; y1_21 = y1_22; y1_23 = y1_22; y1_32 = y1_22 - height1_32 - 1.1*ybuf0; y1_31 = y1_32; y1_33 = y1_32

# First column, upper rows
rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_31 = [x1_31, y1_31, width1_31, height1_31]
rect1_32 = [x1_32, y1_32, width1_32, height1_32]
rect1_33 = [x1_33, y1_33, width1_33, height1_33]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.02, 0.95, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.028 + x1_12 - x1_11, 0.95, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.028 + x1_13 - x1_11, 0.95, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.098, 0.897 - y1_11 + y1_21, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.0685 + x1_12 - x1_11, 0.897 - y1_12 + y1_22, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.04 + x1_13 - x1_11, 0.897 - y1_13 + y1_23, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.098, 0.897 - y1_11 + y1_31, 'G', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.0685 + x1_12 - x1_11, 0.897 - y1_12 + y1_32, 'H', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.04 + x1_13 - x1_11, 0.897 - y1_13 + y1_33, 'I', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.484, 0.975, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.815, 0.975, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.055, 0.502, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k', horizontalalignment='center')
fig_temp.text(0.055, 0.215, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k', horizontalalignment='center')
fig_temp.text(0.185, 0.59, 'Lowered E/I', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.485, 0.59, 'Elevated E/I', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.77, 0.59, 'Sensory deficit', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')






## rect1_11: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax)
ax.arrow(0,0,(Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1],(Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2]                     , color=color_list_expt[1], head_width=0.02, head_length=0.03, label='Monkey A', zorder=4)
ax.arrow(0,0,(Reg_bars_H_ketamine[1]-Reg_bars_H_saline[1])/Reg_bars_H_saline[1],(Reg_bars_H_ketamine[2]-Reg_bars_H_saline[2])/Reg_bars_H_saline[2]                     , color=color_list_expt[1], head_width=0.02, head_length=0.03, label='Monkey H', zorder=2)
ax.arrow(0,0,(beta_mean_mesh_mean_std[0,3]-beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0],(beta_std_mesh_mean_std[0,3]-beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0], color='k', head_width=0.02, head_length=0.03, label='Model', zorder=5)
arc1 = matplotlib.patches.Arc([0,0],0.3,0.15, angle= 180.-0.5*90.*np.abs((beta_std_mesh_mean_std[0,2]-beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0])/np.abs((beta_mean_mesh_mean_std[0,2]-beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0])+0.5*90.*np.abs((Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2])/np.abs((Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1]),theta1=-12,theta2=28, color='k' , lw=1, fill=False, zorder=1)
arc2 = matplotlib.patches.Arc([0,0],0.56,0.25, angle= 180.-0.5*90.*np.abs((beta_std_mesh_mean_std[0,2]-beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0])/np.abs((beta_mean_mesh_mean_std[0,2]-beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0])+0.5*90.*np.abs((Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2])/np.abs((Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1]),theta1=30,theta2=45, color='k' , lw=1, fill=False, zorder=1)
ax.add_patch(arc1)
ax.add_patch(arc2)
ax.set_xlabel('Relative change\nin mean evidence beta', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Relative change\nin evidence SD beta', fontsize=fontsize_legend, labelpad=1.)
ax.set_xlim([-0.6,0.])
ax.set_xticks([-0.6,0])
ax.set_xticklabels([-0.6,0])
ax.set_ylim([-0.3, 0.3])
ax.set_yticks([-0.3,0,0.3])
ax.set_yticklabels([-0.3,0,0.3])
minorLocator = MultipleLocator(0.3)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.text(-0.615, -0.062, 'Monkey A', fontsize=fontsize_tick, color=color_list_expt[1])
ax.text(-0.56, 0.115, 'Monkey H', fontsize=fontsize_tick, color=color_list_expt[1])
ax.text(-0.11, -0.14, 'Example\nmodel', fontsize=fontsize_tick, horizontalalignment='center')
ax.text(-0.25, -0.01, r'$20.1^o$', fontsize=fontsize_tick-0.5)
ax.text(-0.33, -0.088, r'$10.1^o$', fontsize=fontsize_tick-0.5)




ax   = fig_temp.add_axes(rect1_12)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
viridis_w = copy.copy(matplotlib_cm.viridis)
viridis_w.set_over((1, 1, 1, 1))
vmax_pscan = np.max((cos_sim_A_ket_2D_EI_matrix))
vmin_pscan = np.min((cos_sim_A_ket_2D_EI_matrix))
plt.imshow(cos_sim_A_ket_2D_EI_matrix, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', cmap=viridis_w, aspect=aspect_ratio, origin='lower', vmin=0., vmax=1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(ticks=[0, 1], cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
cbar_temp.ax.set_xticklabels([0, 1])
ax.set_title("Cosine similarity", fontsize=fontsize_legend, rotation=0, y=1.12)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)



ax   = fig_temp.add_axes(rect1_13)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
viridis_w = copy.copy(matplotlib_cm.viridis)
viridis_w.set_over((1, 1, 1, 1))
vmax_pscan = np.max((cos_sim_H_ket_2D_EI_matrix))
vmin_pscan = np.min((cos_sim_H_ket_2D_EI_matrix))
plt.imshow(cos_sim_H_ket_2D_EI_matrix, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', cmap=viridis_w, aspect=aspect_ratio, origin='lower', vmin=0., vmax=1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(ticks=[0,1], cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
cbar_temp.ax.set_xticklabels([0, 1])
ax.set_title("Cosine similarity", fontsize=fontsize_legend, rotation=0, y=1.12)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)




## rect1_21: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(gE_pert_list[1:], cos_sim_A_ket_gNMDAE_list[1:], color=color_list[1], markerfacecolor=color_list[1], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(gE_pert_list[3], cos_sim_A_ket_gNMDAE_list[3], color=color_list[1], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.axhline(np.max(cos_sim_A_ket_gNMDAE_list[1:]), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Cosine similarity', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,1.5])
ax.set_xticks([0.,1.5])
ax.set_xticklabels([0,1.5])
ax.set_ylim([0.45, 1.0])
ax.set_yticks([0.5, 1.0])
ax.set_yticklabels([0.5, 1])
minorLocator = MultipleLocator(0.25)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.1)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_31: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_31)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(gE_pert_list[1:], cos_sim_H_ket_gNMDAE_list[1:], color=color_list[1], markerfacecolor=color_list[1], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(gE_pert_list[3], cos_sim_H_ket_gNMDAE_list[3], color=color_list[1], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.axhline(np.max(cos_sim_H_ket_gNMDAE_list[1:]), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Cosine similarity', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,1.5])
ax.set_xticks([0.,1.5])
ax.set_xticklabels([0,1.5])
ax.set_ylim([0.45, 1.0])
ax.set_yticks([0.5, 1.0])
ax.set_yticklabels([0.5, 1])
minorLocator = MultipleLocator(0.25)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.1)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))


## rect1_22: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(gI_pert_list[1:], cos_sim_A_ket_gNMDAI_list[1:], color=color_list[2], markerfacecolor=color_list[2], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(gI_pert_list[3], cos_sim_A_ket_gNMDAI_list[3], color=color_list[2], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.set_xlabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend, labelpad=1.)
ax.axhline(np.max(cos_sim_A_ket_gNMDAE_list[1:]), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_ylabel('Cosine similarity', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,1.5])
ax.set_xticks([0.,1.5])
ax.set_xticklabels([0,1.5])
ax.set_ylim([0.45, 1.0])
ax.set_yticks([0.5, 1.0])
ax.set_yticklabels([0.5, 1])
minorLocator = MultipleLocator(0.25)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.1)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_32: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_32)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(gI_pert_list[1:], cos_sim_H_ket_gNMDAI_list[1:], color=color_list[2], markerfacecolor=color_list[2], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(gI_pert_list[3], cos_sim_H_ket_gNMDAI_list[3], color=color_list[2], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.axhline(np.max(cos_sim_H_ket_gNMDAE_list[1:]), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Cosine similarity', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,1.5])
ax.set_xticks([0.,1.5])
ax.set_xticklabels([0,1.5])
ax.set_ylim([0., 1.0])
ax.set_yticks([0, 1])
minorLocator = MultipleLocator(0.25)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.25)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_23: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(1.-sensory_coeff_list[:-1], cos_sim_A_ket_sensory_list[:-1], color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(1.-sensory_coeff_list[-5], cos_sim_A_ket_sensory_list[-5], color=color_list[3], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.axhline(np.max(cos_sim_A_ket_gNMDAE_list[1:]), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Cosine similarity', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,50])
ax.set_ylim([0.45, 1.0])
ax.set_yticks([0.5, 1.0])
ax.set_yticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.1)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))



## rect1_33: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_33)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(1.-sensory_coeff_list[:-1], cos_sim_H_ket_sensory_list[:-1], color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(1.-sensory_coeff_list[-5], cos_sim_H_ket_sensory_list[-5], color=color_list[3], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.axhline(np.max(cos_sim_H_ket_gNMDAE_list[1:]), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Cosine similarity', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,50])
ax.set_ylim([0.45, 1.0])
ax.set_yticks([0.5, 1.0])
ax.set_yticklabels([0.5, 1])
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.1)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))







fig_temp.savefig(path_cwd+'Figure8S4.pdf')    #Finally save fig

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
## 2D plot: 2D E/I pscan goodness of fit, KL divergence

gE_pert_list = np.arange(0.,1.51,0.25)
gI_pert_list = np.arange(0.,1.51,0.25)
gE_pert_mesh, gI_pert_mesh = np.meshgrid(gE_pert_list, gI_pert_list)



## Monkey data, with lapse (beta0, mean_beta, std_beta, lapse rate)
Reg_bars_A_ketamine = np.array([0.875741073112642, 11.528262865855169, 5.233422331691739])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on ketamine.
Reg_bars_H_ketamine = np.array([-0.054946117132906, 11.298580759143913, 3.616936576677555])  # [Bias, Val diff , Std diff]. Henry regression Beta values on ketamine.
Reg_bars_A_saline = np.array([0.123672806343438, 25.565433692118590, 5.762095065977849])  # [Bias, Val diff , Std diff]. Alfie regression Beta values on saline.
Reg_bars_H_saline = np.array([-0.015314326931326, 22.141685413467790, 3.313026659905981])  # [Bias, Val diff , Std diff]. Henry regression Beta values on saline.

## KL Divergence with monkey-fitted lapse rate to P_corr.
eucl_dist_A_ket_2D_EI_matrix = (((beta_mean_mesh_mean_std - beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0] - (Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1])**2 + ((beta_std_mesh_mean_std - beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0] - (Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2])**2)**0.5
eucl_dist_H_ket_2D_EI_matrix = (((beta_mean_mesh_mean_std - beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0] - (Reg_bars_H_ketamine[1]-Reg_bars_H_saline[1])/Reg_bars_H_saline[1])**2 + ((beta_std_mesh_mean_std - beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0] - (Reg_bars_H_ketamine[2]-Reg_bars_H_saline[2])/Reg_bars_H_saline[2])**2)**0.5

eucl_dist_A_ket_gNMDAE_list = eucl_dist_A_ket_2D_EI_matrix[0,:]
eucl_dist_H_ket_gNMDAE_list = eucl_dist_H_ket_2D_EI_matrix[0,:]

eucl_dist_A_ket_gNMDAI_list = eucl_dist_A_ket_2D_EI_matrix[:,0]
eucl_dist_H_ket_gNMDAI_list = eucl_dist_H_ket_2D_EI_matrix[:,0]

eucl_dist_A_ket_sensory_list = (((beta_mean_sensory_list_mean_std - beta_mean_sensory_list_mean_std[-1])/beta_mean_sensory_list_mean_std[-1] - (Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1])**2 + ((beta_std_sensory_list_mean_std - beta_std_sensory_list_mean_std[-1])/beta_std_sensory_list_mean_std[-1] - (Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2])**2)**0.5
eucl_dist_H_ket_sensory_list = (((beta_mean_sensory_list_mean_std - beta_mean_sensory_list_mean_std[-1])/beta_mean_sensory_list_mean_std[-1] - (Reg_bars_H_ketamine[1]-Reg_bars_H_saline[1])/Reg_bars_H_saline[1])**2 + ((beta_std_sensory_list_mean_std - beta_std_sensory_list_mean_std[-1])/beta_std_sensory_list_mean_std[-1] - (Reg_bars_H_ketamine[2]-Reg_bars_H_saline[2])/Reg_bars_H_saline[2])**2)**0.5

eucl_dist_bet_monkeys_ket_sensory_list = (((Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1] - (Reg_bars_H_ketamine[1]-Reg_bars_H_saline[1])/Reg_bars_H_saline[1])**2 + ((Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2] - (Reg_bars_H_ketamine[2]-Reg_bars_H_saline[2])/Reg_bars_H_saline[2])**2)**0.5

figsize = (max2, 1.*max2)

width1_11 = 0.22; width1_12 = width1_11; width1_13 = width1_11; width1_21 = 0.19; width1_22 = width1_21; width1_23 = width1_21; width1_31 = width1_21; width1_32 = width1_22; width1_33 = width1_23
x1_11 = 0.09; x1_12 = x1_11 + width1_11 + xbuf0; x1_13 = x1_12 + width1_12 + xbuf0; x1_21 = 0.15; x1_22 = x1_21 + width1_21 + xbuf0; x1_23 = x1_22 + width1_22 + xbuf0; x1_31 = x1_21; x1_32 = x1_22; x1_33 = x1_23
height1_11 = 0.23; height1_12 = height1_11; height1_13 = height1_11; height1_21 = 0.2; height1_22 = height1_21; height1_23 = height1_21; height1_31 = height1_21; height1_32 = height1_21; height1_33 = height1_21
y1_11 = 0.7; y1_12 = y1_11; y1_13 = y1_11; y1_22 = y1_12 - height1_22 - 1.65*ybuf0; y1_21 = y1_22; y1_23 = y1_22; y1_32 = y1_22 - height1_32 - 1.1*ybuf0; y1_31 = y1_32; y1_33 = y1_32

# First column, upper rows
rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_31 = [x1_31, y1_31, width1_31, height1_31]
rect1_32 = [x1_32, y1_32, width1_32, height1_32]
rect1_33 = [x1_33, y1_33, width1_33, height1_33]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.02, 0.95, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.028 + x1_12 - x1_11, 0.95, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.028 + x1_13 - x1_11, 0.95, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.09, 0.897 - y1_11 + y1_21, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.06+ x1_12 - x1_11, 0.897 - y1_12 + y1_22, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.04 + x1_13 - x1_11, 0.897 - y1_13 + y1_23, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.09, 0.897 - y1_11 + y1_31, 'G', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.06+ x1_12 - x1_11, 0.897 - y1_12 + y1_32, 'H', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.04 + x1_13 - x1_11, 0.897 - y1_13 + y1_33, 'I', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.484, 0.975, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.815, 0.975, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.055, 0.502, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k', horizontalalignment='center')
fig_temp.text(0.055, 0.215, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k', horizontalalignment='center')
fig_temp.text(0.185, 0.59, 'Lowered E/I', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.485, 0.59, 'Elevated E/I', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.77, 0.59, 'Sensory deficit', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')







## rect1_11: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax)
ax.arrow(0,0,(Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1],(Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2]                     , color=color_list_expt[1], head_width=0.02, head_length=0.03, label='Monkey A', zorder=4)
ax.arrow(0,0,(Reg_bars_H_ketamine[1]-Reg_bars_H_saline[1])/Reg_bars_H_saline[1],(Reg_bars_H_ketamine[2]-Reg_bars_H_saline[2])/Reg_bars_H_saline[2]                     , color=color_list_expt[1], head_width=0.02, head_length=0.03, label='Monkey H', zorder=2)
ax.arrow(0,0,(beta_mean_mesh_mean_std[0,3]-beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0],(beta_std_mesh_mean_std[0,3]-beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0], color='k', head_width=0.02, head_length=0.03, label='Model', zorder=5)
ax.plot([(beta_mean_mesh_mean_std[0,3]-beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0], (Reg_bars_A_ketamine[1]-Reg_bars_A_saline[1])/Reg_bars_A_saline[1]], [(beta_std_mesh_mean_std[0,3]-beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0], (Reg_bars_A_ketamine[2]-Reg_bars_A_saline[2])/Reg_bars_A_saline[2]], color='grey', ls=':',lw=1, zorder=0)
ax.plot([(beta_mean_mesh_mean_std[0,3]-beta_mean_mesh_mean_std[0,0])/beta_mean_mesh_mean_std[0,0], (Reg_bars_H_ketamine[1]-Reg_bars_H_saline[1])/Reg_bars_H_saline[1]], [(beta_std_mesh_mean_std[0,3]-beta_std_mesh_mean_std[0,0])/beta_std_mesh_mean_std[0,0], (Reg_bars_H_ketamine[2]-Reg_bars_H_saline[2])/Reg_bars_H_saline[2]], color='grey', ls=':',lw=1, zorder=0)
ax.set_xlabel('Relative change\nin mean evidence beta', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Relative change\nin evidence SD beta', fontsize=fontsize_legend, labelpad=1.)
ax.set_xlim([-0.6,0.])
ax.set_xticks([-0.6,0])
ax.set_xticklabels([-0.6,0])
ax.set_ylim([-0.3, 0.3])
ax.set_yticks([-0.3,0,0.3])
ax.set_yticklabels([-0.3,0,0.3])
minorLocator = MultipleLocator(0.3)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.text(-0.615, -0.062, 'Monkey A', fontsize=fontsize_tick, color=color_list_expt[1])
ax.text(-0.56, 0.115, 'Monkey H', fontsize=fontsize_tick, color=color_list_expt[1])
ax.text(-0.11, -0.14, 'Example\nmodel', fontsize=fontsize_tick, horizontalalignment='center')




ax   = fig_temp.add_axes(rect1_12)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((eucl_dist_A_ket_2D_EI_matrix))
vmin_pscan = np.min((eucl_dist_A_ket_2D_EI_matrix))
plt.imshow(eucl_dist_A_ket_2D_EI_matrix, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', cmap=matplotlib_cm.viridis_r, aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=vmax_pscan)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
for label in cbar_temp.ax.xaxis.get_ticklabels()[1:-1]:
    label.set_visible(False)
ax.set_title("Euclidean distance", fontsize=fontsize_legend, rotation=0, y=1.12)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)



ax   = fig_temp.add_axes(rect1_13)
aspect_ratio = (100.*gI_pert_list[-1]-100.*gI_pert_list[0])/(100.*gE_pert_list[-1]-100.*gE_pert_list[0])
## Label unstable mem/spont states black/white
cmap_jet_bw = copy.copy(matplotlib_cm.jet)
cmap_jet_bw.set_over((1, 1, 1, 1))
cmap_jet_bw.set_under((0, 0, 0, 1))
vmax_pscan = np.max((eucl_dist_H_ket_2D_EI_matrix))
vmin_pscan = np.min((eucl_dist_H_ket_2D_EI_matrix))
plt.imshow(eucl_dist_H_ket_2D_EI_matrix, extent=(gI_pert_list[0], gI_pert_list[-1], gE_pert_list[0], gE_pert_list[-1]), interpolation='nearest', cmap=matplotlib_cm.viridis_r, aspect=aspect_ratio, origin='lower', vmin=vmin_pscan, vmax=vmax_pscan)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_yticks([1.5*0.5/7., 1.5*2.5/7., 1.5*4.5/7., 1.5*6.5/7])
ax.set_xticklabels([0, '', '', 2.625])
ax.set_yticklabels([0, '', '', 5.25])
ax.tick_params(direction='out', pad=1.5)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend)
ax.set_ylabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend)
divider = make_axes_locatable(ax)
cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
cbar_temp = plt.colorbar(cax=cax_scale_bar_size, orientation="horizontal")
cbar_temp.ax.tick_params(axis='x', direction='out')
cbar_temp.ax.xaxis.set_label_position('top')
cbar_temp.ax.xaxis.set_ticks_position('top')
for label in cbar_temp.ax.xaxis.get_ticklabels()[1:-1]:
    label.set_visible(False)
ax.set_title("Euclidean distance", fontsize=fontsize_legend, rotation=0, y=1.12)
ax.scatter(1.5*0.5/7., 1.5*0.5/7., color=color_list[0], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*3.5/7., 1.5*0.5/7., color=color_list[1], s=20, edgecolors='k', linewidth=0.5)
ax.scatter(1.5*0.5/7., 1.5*3.5/7., color=color_list[2], s=20, edgecolors='k', linewidth=0.5)




## rect1_21: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(gE_pert_list, eucl_dist_A_ket_gNMDAE_list, color=color_list[1], markerfacecolor=color_list[1], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(gE_pert_list[3], eucl_dist_A_ket_gNMDAE_list[3], color=color_list[1], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.axhline(np.min(eucl_dist_A_ket_gNMDAE_list), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Euclidean distance', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,1.5])
ax.set_xticks([0.,1.5])
ax.set_xticklabels([0,1.5])
ax.set_ylim([0.2, 1.0])
ax.set_yticks([0.2, 1.0])
ax.set_yticklabels([0.2, 1])
minorLocator = MultipleLocator(0.25)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_31: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_31)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(gE_pert_list, eucl_dist_H_ket_gNMDAE_list, color=color_list[1], markerfacecolor=color_list[1], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(gE_pert_list[3], eucl_dist_H_ket_gNMDAE_list[3], color=color_list[1], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.axhline(np.min(eucl_dist_H_ket_gNMDAE_list), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel(r'$G_{E\rightarrow E}$' +' reduction (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Euclidean distance', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,1.5])
ax.set_xticks([0.,1.5])
ax.set_xticklabels([0,1.5])
ax.set_ylim([0.26, 1.1])
ax.set_yticks([0.3, 1.1])
ax.set_yticklabels([0.3, 1.1])
minorLocator = MultipleLocator(0.25)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))


## rect1_22: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(gI_pert_list, eucl_dist_A_ket_gNMDAI_list, color=color_list[2], markerfacecolor=color_list[2], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(gI_pert_list[3], eucl_dist_A_ket_gNMDAI_list[3], color=color_list[2], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.set_xlabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend, labelpad=1.)
ax.axhline(np.min(eucl_dist_A_ket_gNMDAE_list), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_ylabel('Euclidean distance', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,1.5])
ax.set_xticks([0.,1.5])
ax.set_xticklabels([0,1.5])
ax.set_ylim([0.2, 1.0])
ax.set_yticks([0.2, 1.0])
ax.set_yticklabels([0.2, 1])
minorLocator = MultipleLocator(0.25)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_32: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_32)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(gI_pert_list, eucl_dist_H_ket_gNMDAI_list, color=color_list[2], markerfacecolor=color_list[2], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(gI_pert_list[3], eucl_dist_H_ket_gNMDAI_list[3], color=color_list[2], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.axhline(np.min(eucl_dist_H_ket_gNMDAE_list), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel(r'$G_{E\rightarrow I}$' +' reduction (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Euclidean distance', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,1.5])
ax.set_xticks([0.,1.5])
ax.set_xticklabels([0,1.5])
ax.set_ylim([0.26, 1.1])
ax.set_yticks([0.3, 1.1])
ax.set_yticklabels([0.3, 1.1])
minorLocator = MultipleLocator(0.25)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))

## rect1_23: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(1.-sensory_coeff_list, eucl_dist_A_ket_sensory_list, color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(1.-sensory_coeff_list[-5], eucl_dist_A_ket_sensory_list[-5], color=color_list[3], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.axhline(np.min(eucl_dist_A_ket_gNMDAE_list), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Euclidean distance', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,50])
ax.set_ylim([0.2, 1.])
ax.set_yticks([0.2, 1.])
ax.set_yticklabels([0.2, 1])
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))



## rect1_33: Mean Evidence beta vs sensory coefficient
ax   = fig_temp.add_axes(rect1_33)
fig_funs.remove_topright_spines(ax)
tmp = ax.plot(1.-sensory_coeff_list, eucl_dist_H_ket_sensory_list, color=color_list[3], markerfacecolor=color_list[3], markeredgecolor='k', linestyle='-', marker='.', zorder=(3-1), clip_on=False, alpha=1., markeredgewidth=0.6)#, linestyle=linestyle_list[i_var_a])
ax.scatter(1.-sensory_coeff_list[-5], eucl_dist_H_ket_sensory_list[-5], color=color_list[3], edgecolors='k', zorder=5, clip_on=False, alpha=1., linewidths=0.6, s=25)
ax.axhline(np.min(eucl_dist_H_ket_gNMDAE_list), color=color_list[1], linestyle="--", zorder=1, lw=1)
ax.set_xlabel('Sensory deficit (%)', fontsize=fontsize_legend, labelpad=1.)
ax.set_ylabel('Euclidean distance', fontsize=fontsize_legend, labelpad=-3.)
ax.set_xlim([0.,0.5])
ax.set_xticks([0.,0.5])
ax.set_xticklabels([0,50])
ax.set_ylim([0.26, 1.1])
ax.set_yticks([0.3, 1.1])
ax.set_yticklabels([0.3, 1.1])
minorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(0.2)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))







fig_temp.savefig(path_cwd+'Figure8S5.pdf')    #Finally save fig

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
### Figure 5: Model E/I Perturbsation & Upstream Sensory Deficit

## Probability Correct, using regression trials                                                                         # See MainAnalysisNonDrugDays_NL.m: line 80-104
d_evidence_model_control_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_control = np.array([0.0299251870324190, 0.102455546147333, 0.165787738958471, 0.242960579243765, 0.327673293927996, 0.393847644422914, 0.434475341834383, 0.487556339408191, 0.520799567801189, 0.548271630190223, 0.621996685998343, 0.672585418348130, 0.690380761523046, 0.710402429764617, 0.761951458690856, 0.821070774534095, 0.875543353923587, 0.924349881796690, 0.961880559085133, 0.978891820580475])#, 1])                              # Log-Spaced.
ErrBar_P_corr_model_control = np.array([0.00850841892200621, 0.00623958597960317, 0.00477410209902654, 0.00459772053896107, 0.00486578052700240, 0.00537704761750850, 0.00614399995684410, 0.00699717413511669, 0.00821061523806815, 0.00503282180572321, 0.00493450065010101, 0.00769708799958044, 0.00654497567157159, 0.00558948488802517, 0.00471524826792922, 0.00397820137135327, 0.00353055086983611, 0.00343628201397335, 0.00394081312756613, 0.00738368453494984])       # Log-Spaced.
d_evidence_model_reduced_gEE_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_reduced_gEE = np.array([0.0769230769230769, 0.226502963590178, 0.325937132291141, 0.388312274368231, 0.439564032697548, 0.467936117936118, 0.487738004569688, 0.507159904534606, 0.519804666304938, 0.534309145645178, 0.571769342460071, 0.582397003745318, 0.592909535452323, 0.618070122864849, 0.659605374651979, 0.699967876646322, 0.771115627822945, 0.856283422459893, 0.916977999169780, 0.972222222222222])
ErrBar_P_corr_model_reduced_gEE = np.array([0.0139667892154339, 0.00861244228906594, 0.00607708257373520, 0.00517655185166871, 0.00518168535233430, 0.00553048184325286, 0.00616910272271109, 0.00705062859025434, 0.00822908198645589, 0.00508234775950867, 0.00503924817219099, 0.00806625078699140, 0.00701273463568272, 0.00594726911297190, 0.00521335295868287, 0.00474212224051575, 0.00446425525893815, 0.00453488636369150, 0.00562157042567251, 0.00825817208348005])
d_evidence_model_reduced_gEI_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_reduced_gEI = np.array([0.116710875331565, 0.185714285714286, 0.254110612855007, 0.298013245033113, 0.360124811706477, 0.396219556524900, 0.436341720656923, 0.460926341562946, 0.472214542438485, 0.509094710432783, 0.543428452405749, 0.563240562138330, 0.593497576736672, 0.612824798662817, 0.650646057239464, 0.686163522012579, 0.738836265223275, 0.790869419268211, 0.846314907872697, 0.904000000000000])
ErrBar_P_corr_model_reduced_gEI = np.array([0.0165362271628761, 0.00797117943562455, 0.00561066452695734, 0.00488742123914029, 0.00497935193967755, 0.00538395662310134, 0.00608744999006230, 0.00712028649298353, 0.00830087997353963, 0.00511131868910556, 0.00508328848898850, 0.00823331301360864, 0.00697992452350820, 0.00600448807484708, 0.00523918619872408, 0.00475106898565844, 0.00466463530778136, 0.00526879029894549, 0.00738014178097813, 0.0152126263347260])
d_evidence_model_upstream_deficit_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_upstream_deficit = np.array([0.116352201257862, 0.224633749321758, 0.301235084781243, 0.365974985787379, 0.412243209706328, 0.447145714725342, 0.473461538461539, 0.487860976233069, 0.504231311706629, 0.529051987767584, 0.551042081508031, 0.580163043478261, 0.585148514851485, 0.604416892677257, 0.640930805979350, 0.682332345812012, 0.737902057374674, 0.819282136894825, 0.891484942886812, 0.949843260188088])
ErrBar_P_corr_model_upstream_deficit = np.array([0.0179809793389682, 0.00972138811396821, 0.00663805277996999, 0.00574269805493527, 0.00577985235079000, 0.00616746378665013, 0.00692397882028006, 0.00799074093789279, 0.00938861087194533, 0.00575569622994858, 0.00573074675516848, 0.00909591639853725, 0.00775155254999240, 0.00680578406435674, 0.00595532900732226, 0.00540265213610709, 0.00529350940567843, 0.00555851624917067, 0.00708718658224564, 0.0122206774336945])




### Psychometric function fit                                                                                           # See figure_psychometric_function_fit.py, esp lines 533-638
def Psychometric_fit_D(params_pm, pm_fit2, x_list):
    prob_corr_fit = 0.5 + 0.5*np.sign(x_list+params_pm[2])*(1. - np.exp(-(np.abs(x_list+params_pm[2])/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    to_min = -sum(np.log(prob_corr_fit)*pm_fit2) - sum(np.log(1.-prob_corr_fit)*(1.-pm_fit2))                                                          # Maximum Likelihood Estimator
    return to_min
def Psychometric_function_D(params_pm, x_list):
    prob_corr_fit = 0.5 + 0.5*np.sign(x_list+params_pm[2])*(1. - np.exp(-(np.abs(x_list+params_pm[2])/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    return prob_corr_fit
def Psychometric_function_D_lapse(params_pm, x_list, lapse_temp):
    prob_corr_fit = 0.5 + 0.5*(1.-2.*lapse_temp)*np.sign(x_list+params_pm[2])*(1. - np.exp(-(np.abs(x_list+params_pm[2])/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    return prob_corr_fit


x_list_psychometric = np.arange(0.01, 0.5, 0.01)                                                                        # See figure_psychometric_function_fit.py, esp lines 633-687
x0_psychometric = 0.
## non-binned MLE (i.e. done using literal net evidence, via matlab). See Psychometric_function_fit_model_NL.m.
psychometric_params_model_control = [0.102757281562668, 1.17220614175807, 0.0259375390410745]
psychometric_params_model_control_higher_total = [0.102757281562668, 1.17220614175807, 0.0259375390410745]
psychometric_params_model_control_lower_total = [0.102757281562668, 1.17220614175807, 0.0259375390410745]


### Regression models                                                                                                   # See MainAnalysisNonDrugDays_NL.m: VarAndLocalWinsBetasCollapsed, VarAndLocalWinsErrCollapsed
##  Mean & Variance, LR differences, Constrained across-trials
Reg_bars_LRdiff_model_control = np.array([0.00394437983254986, 14.6627950683381, 5.13134552497242])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_control = np.array([0.00931236025330767, 0.106074853388491, 0.103503029289104])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_lowered_EI = np.array([-0.00830447006144746, 8.03754426615278, 3.86692416574574])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_lowered_EI = np.array([0.00856943866173542, 0.0830155816504511, 0.0944878874910878])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_elevated_EI = np.array([0.00317979473767400, 8.69251280135024, 1.36235744031899])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_elevated_EI = np.array([0.00863879436121764, 0.0840814140111880, 0.0936521774572359])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_upstream_deficit = np.array([-0.0256895987381230, 7.76699115829589, 2.75339437429155])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_upstream_deficit = np.array([0.00924865197027314, 0.0876192364533815, 0.103558537734544])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.



Reg_mean_models = np.array([Reg_bars_LRdiff_model_control[1], Reg_bars_LRdiff_model_lowered_EI[1], Reg_bars_LRdiff_model_elevated_EI[1], Reg_bars_LRdiff_model_upstream_deficit[1]])
Reg_std_models = np.array([Reg_bars_LRdiff_model_control[2], Reg_bars_LRdiff_model_lowered_EI[2], Reg_bars_LRdiff_model_elevated_EI[2], Reg_bars_LRdiff_model_upstream_deficit[2]])
Reg_ratio_models = Reg_std_models / Reg_mean_models
Reg_err_mean_models = np.array([Reg_bars_err_LRdiff_model_control[1], Reg_bars_err_LRdiff_model_lowered_EI[1], Reg_bars_err_LRdiff_model_elevated_EI[1], Reg_bars_err_LRdiff_model_upstream_deficit[1]])
Reg_err_std_models = np.array([Reg_bars_err_LRdiff_model_control[2], Reg_bars_err_LRdiff_model_lowered_EI[2], Reg_bars_err_LRdiff_model_elevated_EI[2], Reg_bars_err_LRdiff_model_upstream_deficit[2]])
Reg_err_ratio_models = Reg_ratio_models *( (Reg_err_mean_models/Reg_mean_models)**2 + (Reg_err_std_models/Reg_std_models)**2)**0.5


## First, Last, Mean, Max, Min                                                                                          # See MainAnalysisNonDrugDays_NL.m: LongAvCOL, LongAvCOLSE
Reg_values_control = np.array([-0.594952801979088, 0.584953790623774, -1.90488425835368, 16.4148876696628, 2.22696356532233, -0.866395115195592, -0.556850665791866, 1.62758056315876, -15.1776923252959, -2.27985172996483, 1.23919132522955])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_lowered_EI = np.array([0.567486855116527, -0.218079897735126, -0.741571022606320, 8.50344386339854, 1.09850660321577, -1.06674130962764, 0.339287804652993, 0.741915791434785, -9.36691707524605, -1.43194897044197, 1.10885070023942])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_elevated_EI = np.array([-0.200653967667183, 4.43432571437492, -1.00917756542300, 7.71007571087563, 0.613461667714435, -0.355698901261671, -4.34569289397638, 0.982021336246365, -7.52812241844965, -0.584509496643205, 0.495264262238854])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_upstream_deficit = np.array([0.429346798674262, -0.276006539689926, -0.800896570423972, 8.51765281872633, 0.780428255775727, -0.890199022672271, 0.337928852004486, 0.678777999841433, -8.81232267822642, -1.09593783476854, 0.636474713701459])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)

Reg_values_err_control = np.array([0.0901060871206523, 0.0481681362329720, 0.0490458147648183, 0.223526535823836, 0.0987535748895928, 0.0973008705653459, 0.0327404026174387, 0.0337102745323623, 0.162776998403304, 0.0821474467756892, 0.0838500349573119])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_lowered_EI = np.array([0.0811274986259432, 0.0430172025261177, 0.0431946763902718, 0.190557524705992, 0.0880115253517955, 0.0872886423650272, 0.0295333375234778, 0.0297344466223492, 0.138283780232219, 0.0740489361198712, 0.0754751706474502])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_elevated_EI = np.array([0.0913940790077220, 0.0525212832153103, 0.0494382123647683, 0.218466618372610, 0.100467303955254, 0.100244645123161, 0.0368602823355416, 0.0333753687633678, 0.154120836599403, 0.0831634756187457, 0.0836807366813228])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_upstream_deficit = np.array([0.0899274453739104, 0.0465067475428660, 0.0468312940564403, 0.208686734881739, 0.0948333315622302, 0.0946232823536707, 0.0343600419669564, 0.0344945845546153, 0.156820558880769, 0.0807080286133876, 0.0813655080342015])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)



# PK                                                                                                                    # See MainAnalysisNonDrugDays_NL.m: lines 130-136.
t_PK_list = 0.125 + 0.25*np.arange(8)
PK_paired_model_control = np.array([2.79732602861567, 3.48690000546767, 3.15360801061874, 2.47121934471340, 1.86191966272277, 1.28808284169562, 0.857792125030154, 0.395494173666558])    # Paired ({(A-B)_PK}). Model Control
PK_paired_model_reduced_gEE = np.array([0.839162067892684, 1.25128821826160, 1.37192902230126, 1.31022736452434, 1.16873984116527, 1.02363235689304, 0.718459772395972, 0.401012775579593])    # Paired ({(A-B)_PK}). Model gEE x0.9825
PK_paired_model_reduced_gEI = np.array([6.32987119940692, 4.27435772736543, 1.70483062095945, 0.586145842249525, 0.239476053733595, 0.0473133664099220, -0.00517930502853544, -0.00245120766544880])    # Paired ({(A-B)_PK}). Model gEI x0.965
PK_paired_model_upstream_deficit = np.array([0.788471821945004, 1.23932473638417, 1.31716847118481, 1.27718065685540, 1.10707198030494, 0.934238318453053, 0.760765353967356, 0.385192700524176])    # Paired ({(A-B)_PK}). Upstream Deficit
PK_paired_err_model_control = np.array([0.0290555435019742, 0.0304695002971457, 0.0297121772518990, 0.0284328596355590, 0.0275728730157040, 0.0269646886112568, 0.0265499263004365, 0.0264176606778744])    # Paired ({(A-B)_PK}). Model Control
PK_paired_err_model_reduced_gEE = np.array([0.0230691990360291, 0.0234429995116957, 0.0235205682684045, 0.0234387717369943, 0.0232677220576122, 0.0231819624838383, 0.0231004559293285, 0.0229411845625228])    # Paired ({(A-B)_PK}). Model gEE x0.9825
PK_paired_err_model_reduced_gEI = np.array([0.0401820801556948, 0.0341378527984556, 0.0290762782212247, 0.0283521286699686, 0.0282377369100764, 0.0280807582515457, 0.0282602329250464, 0.0280915814786799])    # Paired ({(A-B)_PK}). Model gEI x0.965
PK_paired_err_model_upstream_deficit = np.array([0.0261305032161871, 0.0263733043340413, 0.0265645566214206, 0.0265556633913962, 0.0264475473881296, 0.0262539167987318, 0.0260768107038687, 0.0260013914922905])    # Paired ({(A-B)_PK}). Upstream Deficit



### Temporary data: cutting data into higher/Less total evidence halves.
d_evidence_model_control_list_higher_total_evidence =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_control_higher_total_evidence = np.array([0.0156250000000000, 0.111111111111111, 0.129310344827586, 0.218016654049962, 0.305849189570120, 0.365815931941222, 0.430924062214090, 0.455797933409874, 0.467669172932331, 0.538932146829811, 0.582753079807177, 0.628378378378378, 0.727884615384615, 0.710382513661202, 0.760000000000000, 0.829186602870813, 0.878060969515242, 0.933333333333333, 0.972222222222222, 1])#, 1])                              # Log-Spaced.
ErrBar_P_corr_model_control_higher_total_evidence = np.array([0.0155024490882691, 0.0185185185185185, 0.0117752547768321, 0.0113603622027140, 0.0122317722698210, 0.0133949072024792, 0.0149787441594950, 0.0168755210765886, 0.0193485915875740, 0.0117558658811594, 0.0114121297205032, 0.0177641832167433, 0.0138003832886782, 0.0126731319405099, 0.0102092395127439, 0.00823216298500351, 0.00731492977176231, 0.00640864835546787, 0.00664287087018000, 0])       # Log-Spaced.
Reg_bars_LRdiff_model_control_higher_total_evidence = np.array([-0.00321185718029206, 15.2319444534763, 4.52193985289268])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_control_higher_total_evidence = np.array([0.0207257193067125, 0.244472143786507, 0.235673859131335])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
PK_paired_model_control_higher_total_evidence = np.array([3.62752939462287, 4.28886282386122, 3.65193633896198, 2.67639063078019, 1.98068137092902, 1.24142925554281, 0.801505305908611, 0.481121327689241])    # Paired ({(A-B)_PK}). Model Control
PK_paired_err_model_control_higher_total_evidence = np.array([0.0747892643936510, 0.0783605977598930, 0.0749973433404257, 0.0707649326848713, 0.0672893241310554, 0.0659642296762005, 0.0654574587810397, 0.0639857855517271])    # Paired ({(A-B)_PK}). Model Control

d_evidence_model_control_list_lower_total_evidence =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_control_lower_total_evidence = np.array([0.0212765957446809, 0.0729166666666667, 0.180739706908583, 0.259464916708733, 0.354046242774567, 0.412500000000000, 0.469072164948454, 0.459979736575481, 0.510389610389610, 0.576756756756757, 0.602610966057441, 0.653374233128834, 0.683122847301952, 0.707317073170732, 0.759189797449362, 0.839363241678726, 0.868562644119908, 0.926582278481013, 0.958860759493671, 1])#, 1])                              # Log-Spaced.
ErrBar_P_corr_model_control_lower_total_evidence = np.array([0.0148839132845392, 0.0108333194221777, 0.0101651699724532, 0.00984849577903004, 0.0104958359655848, 0.0120105050744279, 0.0135421522964807, 0.0158641132648909, 0.0180148588020301, 0.0114869702987526, 0.0111825813272637, 0.0186374986724906, 0.0157647029324635, 0.0131951653853771, 0.0117110961962213, 0.00987741764346357, 0.00936744888761405, 0.00927959239651740, 0.0111728087927467, 0])       # Log-Spaced.
Reg_bars_LRdiff_model_control_lower_total_evidence = np.array([-0.0212643567484262, 14.8023071215506, 5.44863532513846])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_control_lower_total_evidence = np.array([0.0205834357752314, 0.236210014527515, 0.234083560605284])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
PK_paired_model_control_lower_total_evidence = np.array([2.10086530172427, 2.83952723813410, 2.72308974896793, 2.45266512366364, 1.87934385334721, 1.48141125938634, 0.985019548669577, 0.409692411519303])    # Paired ({(A-B)_PK}). Model Control
PK_paired_err_model_control_lower_total_evidence = np.array([0.0622134098848053, 0.0649441682429119, 0.0645684838772082, 0.0631696961062822, 0.0618884609515847, 0.0606473058993085, 0.0593202143170495, 0.0584211325711178])    # Paired ({(A-B)_PK}). Model Control

Reg_mean_models = np.array([Reg_bars_LRdiff_model_control[1], Reg_bars_LRdiff_model_control_higher_total_evidence[1], Reg_bars_LRdiff_model_control_lower_total_evidence[1]])/100.
Reg_std_models = np.array([Reg_bars_LRdiff_model_control[2], Reg_bars_LRdiff_model_control_higher_total_evidence[2], Reg_bars_LRdiff_model_control_lower_total_evidence[2]])/100.
Reg_ratio_models = Reg_std_models / Reg_mean_models
Reg_err_mean_models = np.array([Reg_bars_err_LRdiff_model_control[1], Reg_bars_err_LRdiff_model_control_higher_total_evidence[1], Reg_bars_err_LRdiff_model_control_lower_total_evidence[1]])/100.
Reg_err_std_models = np.array([Reg_bars_err_LRdiff_model_control[2], Reg_bars_err_LRdiff_model_control_higher_total_evidence[2], Reg_bars_err_LRdiff_model_control_lower_total_evidence[2]])/100.
Reg_err_ratio_models = Reg_ratio_models *( (Reg_err_mean_models/Reg_mean_models)**2 + (Reg_err_std_models/Reg_std_models)**2)**0.5


### Monkey data
Reg_bars_LRdiff_nondrug = np.array([-0.0393515374644066, 20.6286415933539, 3.57390218117472])/100.        # Bias, Mean, SD, averaged over left and right
Reg_bars_err_LRdiff_nondrug = np.array([0.0159281352724043, 0.275866230528743, 0.181885413917953])/100.        # Bias, Mean, SD, averaged over left and right
PK_paired_nondrug = np.array([0.0340538054476688, 0.0291771221741694, 0.0264718269621357, 0.0272377717657362, 0.0235251838041248, 0.0238119541959954, 0.0199176312765762, 0.0221322453862373])*100.    # Paired ({(A-B)_PK}). Model Control
PK_paired_err_nondrug = np.array([0.000705868677546016, 0.000659132170488234, 0.000676902641325257, 0.000683148475080474, 0.000656662191699381, 0.000647916388860366, 0.000643487730352078, 0.000653624405106187])*100.    # Paired ({(A-B)_PK}). Model Control
## Temporary data: cutting data into higher/Less total evidence halves.
d_evidence_nondrug_list_higher_total_evidence =  100*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_nondrug_higher_total_evidence = np.array([0.0294117647058824, 0.0314136125654450, 0.0422960725075529, 0.138059701492537, 0.218875502008032, 0.255913978494624, 0.309262166405024, 0.417948717948718, 0.392550143266476, 0.492554410080183, 0.615384615384615, 0.623203285420945, 0.657971014492754, 0.695390781563126, 0.752475247524753, 0.841880341880342, 0.856399583766910, 0.925866236905721, 0.957189901207464, 0.969230769230769, 0.962264150943396])#, 1])                              # Log-Spaced.
ErrBar_P_corr_nondrug_higher_total_evidence = np.array([0.0289760107689347, 0.0126215135688953, 0.0110624585967769, 0.0149001164673106, 0.0131017418378438, 0.0143092576284401, 0.0183126123781463, 0.0249752486250165, 0.0261390664979961, 0.0169205597298126, 0.134932002970312, 0.0155270496721377, 0.0255402585632691, 0.0206032490812909, 0.0162310165206818, 0.0119255987609620, 0.0113123902626960, 0.00743697387314321, 0.00670676332081229, 0.00957922193838962, 0.0261749753572827])       # Log-Spaced.
Reg_bars_LRdiff_nondrug_higher_total_evidence = np.array([-0.0193451030632044, 0.209355573053493, 0.0343752462039092])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_nondrug_higher_total_evidence = np.array([0.0230000872900402, 0.00406119085680255, 0.00263396373153637])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
PK_paired_nondrug_higher_total_evidence = np.array([0.0400400978073793, 0.0326149745132541, 0.0293387702901905, 0.0312620770475189, 0.0257678505650124, 0.0241980556686970, 0.0203912141757260, 0.0214083037226623])*100.    # Paired ({(A-B)_PK}). Model Control
PK_paired_err_nondrug_higher_total_evidence = np.array([0.00105313848851938, 0.000983157113542651, 0.00101764534545527, 0.00103934103945311, 0.000958721184543967, 0.000970996648122634, 0.000942929042958415, 0.000984256603797720])*100.    # Paired ({(A-B)_PK}). Model Control

d_evidence_nondrug_list_lower_total_evidence =  100*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0., 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])  # Log-Spaced.
P_corr_nondrug_lower_total_evidence = np.array([0.0652173913043478, 0.0305555555555556, 0.0804733727810651, 0.142325581395349, 0.208530805687204, 0.288372093023256, 0.361344537815126, 0.500000000000000, 0.468750000000000, 0.505097312326228, 0.600000000000000, 0.611542730299667, 0.672072072072072, 0.680851063829787, 0.757515030060120, 0.797406807131280, 0.873814041745731, 0.899713467048711, 0.925072046109510, 0.970802919708029, 1])#, 1])                              # Log-Spaced.
ErrBar_P_corr_nondrug_lower_total_evidence = np.array([0.0364047545795288, 0.00907100288162580, 0.00935793237254460, 0.0106561047150655, 0.0114178745143513, 0.0154473460123091, 0.0220186448307179, 0.0226339365106296, 0.0254656343953228, 0.0152207731267363, 0.0979795897113271, 0.0162376297769691, 0.0199273998526644, 0.0240396886325971, 0.0191861469570204, 0.0161811659467742, 0.0102280942017433, 0.0113696136691415, 0.0141333455737635, 0.0143838440377443, 0])       # Log-Spaced.
Reg_bars_LRdiff_nondrug_lower_total_evidence = np.array([-0.0581269902183543, 0.203504681593724, 0.0366280858018433])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_nondrug_lower_total_evidence = np.array([0.0223100998577842, 0.00390953471819450, 0.00261884514769658])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
PK_paired_nondrug_lower_total_evidence = np.array([0.0281876891865518, 0.0258160179904016, 0.0243981503027384, 0.0240266065872021, 0.0213784432695346, 0.0236627661512188, 0.0200546610814298, 0.0231221822747845])*100.    # Paired ({(A-B)_PK}). Model Control
PK_paired_err_nondrug_lower_total_evidence = np.array([0.000965596505680379, 0.000900362275396354, 0.000918220975797547, 0.000919337037239795, 0.000918294289456877, 0.000883582583538285, 0.000894982705493643, 0.000887198569032057])*100.    # Paired ({(A-B)_PK}). Model Control

Reg_mean_models_nondrug = np.array([Reg_bars_LRdiff_nondrug[1], Reg_bars_LRdiff_nondrug_higher_total_evidence[1], Reg_bars_LRdiff_nondrug_lower_total_evidence[1]])
Reg_std_models_nondrug = np.array([Reg_bars_LRdiff_nondrug[2], Reg_bars_LRdiff_nondrug_higher_total_evidence[2], Reg_bars_LRdiff_nondrug_lower_total_evidence[2]])
Reg_ratio_models_nondrug = Reg_std_models_nondrug / Reg_mean_models_nondrug
Reg_err_mean_models_nondrug = np.array([Reg_bars_err_LRdiff_nondrug[1], Reg_bars_err_LRdiff_nondrug_higher_total_evidence[1], Reg_bars_err_LRdiff_nondrug_lower_total_evidence[1]])
Reg_err_std_models_nondrug = np.array([Reg_bars_err_LRdiff_nondrug[2], Reg_bars_err_LRdiff_nondrug_higher_total_evidence[2], Reg_bars_err_LRdiff_nondrug_lower_total_evidence[2]])
Reg_err_ratio_models_nondrug = Reg_ratio_models *( (Reg_err_mean_models/Reg_mean_models)**2 + (Reg_err_std_models/Reg_std_models)**2)**0.5


## Define subfigure domain.
figsize = (max2,1.*max2)

width1_21 = 0.15; width1_22 = width1_21; width1_23 = width1_21; width1_24 = 0.21; width1_11 = width1_21; width1_12 = width1_22; width1_13 = width1_23; width1_14 = width1_24
x1_21 = 0.1; x1_22 = x1_21 + width1_21 + 0.6*xbuf0; x1_23 = x1_22 + width1_22 + 0.45*xbuf0; x1_24 = x1_23 + width1_23 + 0.8*xbuf0; x1_11 = x1_21; x1_12 = x1_22; x1_13 = x1_23; x1_14 = x1_24
height1_11 = 0.2; height1_12 = height1_11; height1_13 = height1_12; height1_14 = height1_12; height1_21= height1_11;  height1_22 = height1_21;  height1_23 = height1_21;  height1_24 = height1_21
y1_11 = 0.72; y1_12 = y1_11; y1_13 = y1_12; y1_14 = y1_13; y1_21 = y1_11 - height1_21 - 1.5*ybuf0; y1_22 = y1_21; y1_23 = y1_21; y1_24 = y1_23; y1_31 = y1_21 - height1_31 - ybuf0



rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12 = [x1_12, y1_12, width1_12, height1_12]
rect1_13 = [x1_13, y1_13, width1_13, height1_13]
rect1_14 = [x1_14, y1_14, width1_14, height1_14]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_24 = [x1_24, y1_24, width1_24, height1_24]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.174, 0.95, 'Mean\nEvidence Beta', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(0.391, 0.95, 'SD\nEvidence Beta', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(0.592, 0.95, 'PVB Index', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')

fig_temp.text(0.027, 0.918, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.027+0.01+x1_12-x1_11, 0.918, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.027+0.018+x1_13-x1_11, 0.918, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.027+0.019+x1_14-x1_11, 0.918, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.027, 0.918 + y1_22 - y1_12, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.027+0.01+x1_22-x1_21, 0.918 + y1_22 - y1_12, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.027+0.018+x1_23-x1_21, 0.918 + y1_22 - y1_12, 'G', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.027+0.019+x1_24-x1_21, 0.918 + y1_22 - y1_12, 'H', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.02, 0.83, 'Model', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k', horizontalalignment='center')
fig_temp.text(0.02, 0.528, 'Monkey', fontsize=fontsize_fig_label, fontweight='bold', rotation='vertical', color='k', horizontalalignment='center')


bar_width_compare3 = 1.







## rect1_11: Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax)
ax.errorbar(0.25, Reg_mean_models[0], alpha=1, yerr=Reg_err_mean_models[0], clip_on=False, color=color_list_all_high_low_total[0], markerfacecolor=color_list_all_high_low_total[0], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(1.25, Reg_mean_models[1], alpha=1, yerr=Reg_err_mean_models[1], clip_on=False, color=color_list_all_high_low_total[1], markerfacecolor=color_list_all_high_low_total[1], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(2.25, Reg_mean_models[2], alpha=1, yerr=Reg_err_mean_models[2], clip_on=False, color=color_list_all_high_low_total[2], markerfacecolor=color_list_all_high_low_total[2], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.set_xlim([0,2.5])
ax.set_ylim([0.145,0.155])
ax.set_xticks([0., 1., 2.])
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.xaxis.set_ticklabels(['All trials', 'More total\nevidence', 'Less total\nevidence'], rotation=45)
ax.set_yticks([0.145, 0.155])
ax.yaxis.set_ticklabels([0.145, 0.155])
minorLocator = MultipleLocator(0.0025)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(),color_list_all_high_low_total):
    ticklabel.set_color(tickcolor)

## rect1_12: Variance Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_12)
fig_funs.remove_topright_spines(ax)
ax.errorbar(0.25, Reg_std_models[0], alpha=1, yerr=Reg_err_std_models[0], clip_on=False, color=color_list_all_high_low_total[0], markerfacecolor=color_list_all_high_low_total[0], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(1.25, Reg_std_models[1], alpha=1, yerr=Reg_err_std_models[1], clip_on=False, color=color_list_all_high_low_total[1], markerfacecolor=color_list_all_high_low_total[1], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(2.25, Reg_std_models[2], alpha=1, yerr=Reg_err_std_models[2], clip_on=False, color=color_list_all_high_low_total[2], markerfacecolor=color_list_all_high_low_total[2], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.set_xlim([0,2.5])
ax.set_ylim([0.04,0.06])
ax.set_xticks([0., 1., 2.])
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.xaxis.set_ticklabels(['All trials', 'More total\nevidence', 'Less total\nevidence'], rotation=45)
ax.set_yticks([0.04, 0.06])
ax.yaxis.set_ticklabels([0.04, 0.06])
minorLocator = MultipleLocator(0.005)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(),color_list_all_high_low_total):
    ticklabel.set_color(tickcolor)

## rect1_13: Variance Beta/ Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_13)
fig_funs.remove_topright_spines(ax)
ax.errorbar(0.25, Reg_ratio_models[0], alpha=1, yerr=Reg_err_ratio_models[0], clip_on=False, color=color_list_all_high_low_total[0], markerfacecolor=color_list_all_high_low_total[0], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', markersize=10., linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(1.25, Reg_ratio_models[1], alpha=1, yerr=Reg_err_ratio_models[1], clip_on=False, color=color_list_all_high_low_total[1], markerfacecolor=color_list_all_high_low_total[1], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', markersize=10., linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(2.25, Reg_ratio_models[2], alpha=1, yerr=Reg_err_ratio_models[2], clip_on=False, color=color_list_all_high_low_total[2], markerfacecolor=color_list_all_high_low_total[2], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', markersize=10., linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.plot([0,2.5], [Reg_ratio_models[0], Reg_ratio_models[0]], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.75], [0.4], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([1.25,2.25], [0.392,0.392], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_xlim([0,2.5])
ax.set_ylim([0.25,0.4])
ax.set_xticks([0., 1., 2.])
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.xaxis.set_ticklabels(['All trials', 'More total\nevidence', 'Less total\nevidence'], rotation=45)
ax.set_yticks([0.25, 0.4])
ax.yaxis.set_ticklabels([0.25, 0.4])
minorLocator = MultipleLocator(0.05)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(),color_list_all_high_low_total):
    ticklabel.set_color(tickcolor)

## rect1_14: Psychophysical Kernel, Model and perturbations
ax   = fig_temp.add_axes(rect1_14)
fig_funs.remove_topright_spines(ax)
i_PK_list = np.arange(1,8+1)
t_PK_list = 0.125 + 0.25*np.arange(8)
ax.errorbar( i_PK_list, PK_paired_model_control, PK_paired_err_model_control, color=color_list_all_high_low_total[0], ecolor=color_list_all_high_low_total[0], marker='.', zorder=4, clip_on=False, markerfacecolor=color_list_all_high_low_total[0], markeredgecolor='k', linewidth=1., ls='-', elinewidth=0.6, markeredgewidth=0.6, markersize=5., capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar( i_PK_list, PK_paired_model_control_higher_total_evidence, PK_paired_err_model_control_higher_total_evidence, color=color_list_all_high_low_total[1], ecolor=color_list_all_high_low_total[1], marker='^', zorder=3, clip_on=False, markerfacecolor=color_list_all_high_low_total[1], markeredgecolor='k', linewidth=1., ls='-', elinewidth=0.6, markersize=2.5, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax.errorbar(i_PK_list, PK_paired_model_control_lower_total_evidence, PK_paired_err_model_control_lower_total_evidence, color=color_list_all_high_low_total[2], ecolor=color_list_all_high_low_total[2], marker='s', zorder=2, clip_on=False, markerfacecolor=color_list_all_high_low_total[2], markeredgecolor='k', linewidth=1., ls='-', elinewidth=0.6, markeredgewidth=0.6, markersize=2.5, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
ax.set_xlabel('Sample Number', fontsize=fontsize_legend)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend)
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.set_xlim([1,8.])
ax.set_ylim([0.,4.5])
ax.set_xticks([1., 8.])
ax.set_yticks([0., 4.])
ax.text(0.1, 4.6, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
tmp1 = ax.scatter( i_PK_list, PK_paired_model_control         , color=color_list_all_high_low_total[0], marker='.', zorder=4, clip_on=False, facecolors=color_list_all_high_low_total[0], edgecolors='k', linewidths=0.6, s=12., label=label_list[0])#, linestyle=linestyle_list[i_var_a])
tmp2 = ax.scatter( i_PK_list, PK_paired_model_control_higher_total_evidence     , color=color_list_all_high_low_total[1], marker='^', zorder=3, clip_on=False, facecolors=color_list_all_high_low_total[1], edgecolors='k', linewidths=0.6, s=4., label=label_list[1])#, linestyle=linestyle_list[i_var_a])
tmp3 = ax.scatter(i_PK_list, PK_paired_model_control_lower_total_evidence      , color=color_list_all_high_low_total[2], marker='s', zorder=2, clip_on=False, facecolors=color_list_all_high_low_total[2], edgecolors='k', linewidths=0.6, s=3.5, label=label_list[2])#, linestyle=linestyle_list[i_var_a])
legend = ax.legend(loc=(-0.07,-0.0), fontsize=fontsize_legend-1., frameon=False, ncol=1, columnspacing=1., handletextpad=0., scatterpoints=1)
for color,text,item in zip(color_list_all_high_low_total, legend.get_texts(), legend.legendHandles):
    text.set_color(color)

## rect1_21: Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
ax.errorbar(0.25, Reg_mean_models_nondrug[0], alpha=1, yerr=Reg_err_mean_models_nondrug[0], clip_on=False, color=color_list_all_high_low_total[0], markerfacecolor=color_list_all_high_low_total[0], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(1.25, Reg_mean_models_nondrug[1], alpha=1, yerr=Reg_err_mean_models_nondrug[1], clip_on=False, color=color_list_all_high_low_total[1], markerfacecolor=color_list_all_high_low_total[1], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(2.25, Reg_mean_models_nondrug[2], alpha=1, yerr=Reg_err_mean_models_nondrug[2], clip_on=False, color=color_list_all_high_low_total[2], markerfacecolor=color_list_all_high_low_total[2], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.set_xlim([0,2.5])
ax.set_ylim([0.199,0.215])
ax.set_xticks([0., 1., 2.])
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.xaxis.set_ticklabels(['All trials', 'More total\nevidence', 'Less total\nevidence'], rotation=45)
ax.set_yticks([0.2, 0.215])
ax.yaxis.set_ticklabels([0.2, 0.215])
minorLocator = MultipleLocator(0.005)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(),color_list_all_high_low_total):
    ticklabel.set_color(tickcolor)

## rect1_22: Variance Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
ax.errorbar(0.25, Reg_std_models_nondrug[0], alpha=1, yerr=Reg_err_std_models_nondrug[0], clip_on=False, color=color_list_all_high_low_total[0], markerfacecolor=color_list_all_high_low_total[0], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(1.25, Reg_std_models_nondrug[1], alpha=1, yerr=Reg_err_std_models_nondrug[1], clip_on=False, color=color_list_all_high_low_total[1], markerfacecolor=color_list_all_high_low_total[1], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(2.25, Reg_std_models_nondrug[2], alpha=1, yerr=Reg_err_std_models_nondrug[2], clip_on=False, color=color_list_all_high_low_total[2], markerfacecolor=color_list_all_high_low_total[2], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', linewidth=0.3, markersize=10., elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.set_xlim([0,2.5])
ax.set_ylim([0.03, 0.04])
ax.set_xticks([0., 1., 2.])
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.xaxis.set_ticklabels(['All trials', 'More total\nevidence', 'Less total\nevidence'], rotation=45)
ax.set_yticks([0.03, 0.04])
ax.yaxis.set_ticklabels([0.03, 0.04])
minorLocator = MultipleLocator(0.005)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(),color_list_all_high_low_total):
    ticklabel.set_color(tickcolor)

## rect1_23: Variance Beta/ Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
ax.errorbar(0.25, Reg_ratio_models_nondrug[0], alpha=1, yerr=Reg_err_ratio_models_nondrug[0], clip_on=False, color=color_list_all_high_low_total[0], markerfacecolor=color_list_all_high_low_total[0], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', markersize=10., linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(1.25, Reg_ratio_models_nondrug[1], alpha=1, yerr=Reg_err_ratio_models_nondrug[1], clip_on=False, color=color_list_all_high_low_total[1], markerfacecolor=color_list_all_high_low_total[1], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', markersize=10., linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.errorbar(2.25, Reg_ratio_models_nondrug[2], alpha=1, yerr=Reg_err_ratio_models_nondrug[2], clip_on=False, color=color_list_all_high_low_total[2], markerfacecolor=color_list_all_high_low_total[2], ecolor='grey', fmt='.', zorder=4, markeredgecolor='k', markersize=10., linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)
ax.plot([0,2.5], [Reg_ratio_models_nondrug[0], Reg_ratio_models_nondrug[0]], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.set_xlim([0,2.5])
ax.set_ylim([0.147,0.2])
ax.set_xticks([0., 1., 2.])
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.xaxis.set_ticklabels(['All trials', 'More total\nevidence', 'Less total\nevidence'], rotation=45)
ax.set_yticks([0.15, 0.2])
ax.yaxis.set_ticklabels([0.15, 0.2])
minorLocator = MultipleLocator(0.01)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=0.5)
ax.tick_params(which='minor',direction='out')
ax.tick_params(bottom="off")
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(),color_list_all_high_low_total):
    ticklabel.set_color(tickcolor)

## rect1_24: Psychophysical Kernel, Model and perturbations
ax   = fig_temp.add_axes(rect1_24)
fig_funs.remove_topright_spines(ax)
i_PK_list = np.arange(1,8+1)
t_PK_list = 0.125 + 0.25*np.arange(8)
ax.errorbar( i_PK_list, PK_paired_nondrug, PK_paired_err_nondrug, color=color_list_all_high_low_total[0], ecolor=color_list_all_high_low_total[0], marker='.', zorder=4, clip_on=False, markerfacecolor=color_list_all_high_low_total[0], markeredgecolor='k', linewidth=1., ls='-', elinewidth=0.6, markeredgewidth=0.6, markersize=5., capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar( i_PK_list, PK_paired_nondrug_higher_total_evidence, PK_paired_err_nondrug_higher_total_evidence, color=color_list_all_high_low_total[1], ecolor=color_list_all_high_low_total[1], marker='^', zorder=3, clip_on=False, markerfacecolor=color_list_all_high_low_total[1], markeredgecolor='k', linewidth=1., ls='-', elinewidth=0.6, markersize=2.5, capsize=1.)#, linestyle=linestyle_list[i_var_a])
tmp = ax.errorbar(i_PK_list, PK_paired_nondrug_lower_total_evidence, PK_paired_err_nondrug_lower_total_evidence, color=color_list_all_high_low_total[2], ecolor=color_list_all_high_low_total[2], marker='s', zorder=2, clip_on=False, markerfacecolor=color_list_all_high_low_total[2], markeredgecolor='k', linewidth=1., ls='-', elinewidth=0.6, markeredgewidth=0.6, markersize=2.5, capsize=1.)#, linestyle=linestyle_list[i_var_a])
for b in tmp[1]:
    b.set_clip_on(False)
ax.set_xlabel('Sample Number', fontsize=fontsize_legend)
ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend)
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.set_xlim([1,8.])
ax.set_ylim([0.,4.5])
ax.set_xticks([1., 8.])
ax.set_yticks([0., 4.])
ax.text(0.1, 4.6, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
tmp1 = ax.scatter( i_PK_list, PK_paired_nondrug         , color=color_list_all_high_low_total[0], marker='.', zorder=4, clip_on=False, facecolors=color_list_all_high_low_total[0], edgecolors='k', linewidths=0.6, s=12., label=label_list[0])#, linestyle=linestyle_list[i_var_a])
tmp2 = ax.scatter( i_PK_list, PK_paired_nondrug_higher_total_evidence     , color=color_list_all_high_low_total[1], marker='^', zorder=3, clip_on=False, facecolors=color_list_all_high_low_total[1], edgecolors='k', linewidths=0.6, s=4., label=label_list[1])#, linestyle=linestyle_list[i_var_a])
tmp3 = ax.scatter(i_PK_list, PK_paired_nondrug_lower_total_evidence      , color=color_list_all_high_low_total[2], marker='s', zorder=2, clip_on=False, facecolors=color_list_all_high_low_total[2], edgecolors='k', linewidths=0.6, s=3.5, label=label_list[2])#, linestyle=linestyle_list[i_var_a])
legend = ax.legend(loc=(-0.07,0.), fontsize=fontsize_legend-1., frameon=False, ncol=1, columnspacing=1., handletextpad=0., scatterpoints=1)
for color,text,item in zip(color_list_all_high_low_total, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    # item.set_visible(False)

fig_temp.savefig(path_cwd+'Figure5S2.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
### Figure 7 but with built in lapse (monkey A lapse rate=0.118).



####################################################### Monkey H data
### Figure 6: Ketamine Data
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
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
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
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
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
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
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
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
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
# ## Without Lapse
psychometric_params_A_saline_all        = [0.0595776866237313, 1.26810162179331, 0.0138702695806634]
psychometric_params_H_saline_all        = [0.0681413425053521, 1.07582639210372, 0.0123764957351213]
## With Lapse
psychometric_params_A_ketamine_all      = [0.164267968758472, 0.732705192383852, 0.0377990600679478]
psychometric_params_H_ketamine_all      = [0.130851990508893, 1.16584379279672, 0.0238689326833176]

####################################################### Monkey H data

## Data from Figure5_PNAS_temp_built_in_monkey_A_lapse
## Probability Correct, using regression trials                                                                         # See MainAnalysisNonDrugDays_NL.m: line 80-104
d_evidence_model_control_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_control = np.array([0.138070175438596, 0.190468948035488, 0.238779692594318, 0.301846466012851, 0.367532765399738, 0.407035010940919, 0.455293031841255, 0.487371601208459, 0.514835605453088, 0.538653230966091, 0.599510400000000, 0.610382003395585, 0.634113001215067, 0.670527264533573, 0.709375227025064, 0.746972650110028, 0.790241212956582, 0.831014713343481, 0.851728538283064, 0.871273885350318])                              # Log-Spaced.
ErrBar_P_corr_model_control = np.array([0.0321775202031413, 0.0139749645869194, 0.00919979727878892, 0.00844134113694555, 0.00872673085104690, 0.00938139866256666, 0.0106968468765313, 0.0122852897420790, 0.0141505963544799, 0.00891528268556049, 0.00876484177397858, 0.0142066044631151, 0.0118705681848353, 0.00997676836121672, 0.00865292576448037, 0.00770761366769286, 0.00755711821133638, 0.00843962216228882, 0.0120971525976719, 0.0266556955449550])       # Log-Spaced.
d_evidence_model_reduced_gEE_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_reduced_gEE = np.array([0.145517241379310, 0.245121065375303, 0.316676082862524, 0.384742339832869, 0.439135963482230, 0.461172976985895, 0.481863699582754, 0.496168168168168, 0.503774350649351, 0.529887710542733, 0.556832797427653, 0.593984835720303, 0.599514792899408, 0.621211293260473, 0.647049413437611, 0.684775316455696, 0.738662991040662, 0.795374808771035, 0.844465786314526, 0.866428571428572])
ErrBar_P_corr_model_reduced_gEE = np.array([0.0291812362447513, 0.0149624044065771, 0.0100924197031324, 0.00907799546023679, 0.00896074993563987, 0.00960331498974515, 0.0107575567973046, 0.0122515182335940, 0.0142418283529507, 0.00881432006125775, 0.00890722021178660, 0.0142514696256663, 0.0119174276237139, 0.0103503464578540, 0.00900965668603211, 0.00826430293372605, 0.00815521255259096, 0.00910892029990550, 0.0125513531697682, 0.0273256339746427])
d_evidence_model_reduced_gEI_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_reduced_gEI = np.array([0.231724137931035, 0.241053268765133, 0.280729755178908, 0.325208913649025, 0.382536680795566, 0.420475129918337, 0.444381084840056, 0.463339339339339, 0.482426948051948, 0.505505302557704, 0.549209003215434, 0.561752316764954, 0.584804733727811, 0.595842440801457, 0.623700675435478, 0.673607594936709, 0.705861474844934, 0.749408465068842, 0.794093637454982, 0.832727272727273])
ErrBar_P_corr_model_reduced_gEI = np.array([0.0349593697030429, 0.0148777958686459, 0.00974891242485061, 0.00874065034665859, 0.00877517529162317, 0.00950982125489488, 0.0106979812897534, 0.0122192336105989, 0.0142341305385188, 0.00882940955283825, 0.00892143637926130, 0.0143994509631108, 0.0119850911779985, 0.0104708652417126, 0.00913357887623009, 0.00834055692967491, 0.00845763304702873, 0.00978468946796526, 0.0140040237466158, 0.0299831217000101])
d_evidence_model_upstream_deficit_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_upstream_deficit = np.array([0.207793103448276, 0.240556900726392, 0.308008474576271, 0.358575905292479, 0.420697750244539, 0.439350408314774, 0.475739452943903, 0.469231231231231, 0.482110389610390, 0.519953212726138, 0.557742765273312, 0.590042122999157, 0.589881656804734, 0.612017304189435, 0.628929968005688, 0.673813291139241, 0.728900758097864, 0.780402855685875, 0.828847539015606, 0.852662337662338])
ErrBar_P_corr_model_upstream_deficit = np.array([0.0336200280431258, 0.0148678820521776, 0.0100162490334040, 0.00894819711588319, 0.00891358065658223, 0.00956130235213085, 0.0107518842355878, 0.0122289487278733, 0.0142328461627293, 0.00882282227690159, 0.00890525439727986, 0.0142729635473629, 0.0119629856475488, 0.0103976801057960, 0.00910773673329094, 0.00833908281981799, 0.00825114901538065, 0.00934672678035318, 0.0130443081252453, 0.0284827911808200])

## non-binned MLE (i.e. done using literal net evidence, via matlab). See Psychometric_function_fit_model_NL.m.
x_list_psychometric = np.arange(0.01, 0.5, 0.01)                                                                        # See figure_psychometric_function_fit.py, esp lines 633-687
x0_psychometric = 0.
psychometric_params_model_control = [0.100329415689634, 1.16978014238109, 0.0251088975429423]
psychometric_params_model_reduced_gEE = [0.142105471808786, 1.47907851571470, 0.0335042791254340]
psychometric_params_model_reduced_gEI = [0.151654534746071, 1.07984959314117, 0.0130560286699992]
psychometric_params_model_upstream_deficit = [0.143632586727113, 1.30582415990816, 0.0255444947057615]

##  Mean & Variance, LR differences, Constrained across-trials
Reg_bars_LRdiff_model_control = np.array([0.00440745707719049, 9.72340738863521, 3.46559656385882])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_control = np.array([0.0145206430604281, 0.143544741518788, 0.164676852603497])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_lowered_EI = np.array([-0.0105701923647212, 7.17613381133086, 3.24939847545677])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_lowered_EI = np.array([0.0141148663344087, 0.131719296562874, 0.160047204418128])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_elevated_EI = np.array([-0.00309113137298573, 7.06218641017103, 1.11990928870965])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_elevated_EI = np.array([0.0140817847489371, 0.130382940378342, 0.158030787172467])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_upstream_deficit = np.array([-0.0194951721493130, 7.16264389107799, 2.42803311259080])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_upstream_deficit = np.array([0.0141046802725938, 0.131270693995090, 0.159165225354544])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.

Reg_mean_models = np.array([Reg_bars_LRdiff_model_control[1], Reg_bars_LRdiff_model_lowered_EI[1], Reg_bars_LRdiff_model_elevated_EI[1], Reg_bars_LRdiff_model_upstream_deficit[1]])
Reg_std_models = np.array([Reg_bars_LRdiff_model_control[2], Reg_bars_LRdiff_model_lowered_EI[2], Reg_bars_LRdiff_model_elevated_EI[2], Reg_bars_LRdiff_model_upstream_deficit[2]])
Reg_ratio_models = Reg_std_models / Reg_mean_models
Reg_err_mean_models = np.array([Reg_bars_err_LRdiff_model_control[1], Reg_bars_err_LRdiff_model_lowered_EI[1], Reg_bars_err_LRdiff_model_elevated_EI[1], Reg_bars_err_LRdiff_model_upstream_deficit[1]])
Reg_err_std_models = np.array([Reg_bars_err_LRdiff_model_control[2], Reg_bars_err_LRdiff_model_lowered_EI[2], Reg_bars_err_LRdiff_model_elevated_EI[2], Reg_bars_err_LRdiff_model_upstream_deficit[2]])
Reg_err_ratio_models = Reg_ratio_models *( (Reg_err_mean_models/Reg_mean_models)**2 + (Reg_err_std_models/Reg_std_models)**2)**0.5


## First, Last, Mean, Max, Min                                                                                          # See MainAnalysisNonDrugDays_NL.m: LongAvCOL, LongAvCOLSE
Reg_values_control = np.array([-0.524060393439731, 0.293449791740516, -1.19352596986781, 10.7009366319403, 1.68018051780375, -0.572509326240899, -0.354405524626893, 1.06738931534072, -9.78946300604857, -1.54672683384728, 0.722610973866920])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_lowered_EI = np.array([0.118055157236285, -0.159108711460770, -0.738474205855703, 7.45178178491451, 1.23107582687227, -0.736691032589832, 0.214224628884342, 0.667653292381897, -7.83403163204941, -1.18194675059403, 0.700624628801509])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_elevated_EI = np.array([-0.190775603346266, 2.31396485549950, -0.897155350847555, 6.11482923258001, 0.743673357855439, -0.172061543739482, -2.21559999585935, 0.856217970434154, -6.70273053282033, -0.286017284486571, 0.584284551366613])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_upstream_deficit = np.array([5.97569267393207e-05, -0.132872322584030, -0.610622977306144, 7.90829531366675, 0.845397891602157, -0.723334350750050, 0.173652253955829, 0.669154358541933, -7.84874607097209, -0.950561544630342, 0.625649878160126])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)

Reg_values_err_control = np.array([0.142847161624535, 0.0742469947935042, 0.0752165015578474, 0.335241932485171, 0.152598531940596, 0.151050912830013, 0.0546271072712374, 0.0557551340508917, 0.251611408734459, 0.128689916108949, 0.130017972824943])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_lowered_EI = np.array([0.137060587264945, 0.0712391922245377, 0.0716123418580269, 0.315323697388666, 0.146405446451021, 0.145102008186877, 0.0527133949828616, 0.0531453685708457, 0.238032121608974, 0.124002109056757, 0.125183256381088])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_elevated_EI = np.array([0.142065343947464, 0.0755278022978541, 0.0746371091593121, 0.326029430941723, 0.152264844773893, 0.150920940863554, 0.0558250820199932, 0.0550628641849008, 0.244909944703269, 0.128304206875966, 0.129080797268048])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_upstream_deficit = np.array([0.136810029453187, 0.0711582742745153, 0.0714703309026419, 0.315694664988331, 0.146082199646810, 0.144961233460030, 0.0525073485198762, 0.0529656099926241, 0.237209031737837, 0.123706795404299, 0.124659459580132])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)

# PK                                                                                                                    # See MainAnalysisNonDrugDays_NL.m: lines 130-136.
t_PK_list = 0.125 + 0.25*np.arange(8)
PK_paired_model_control = np.array([1.70665963843400, 2.09595090238125, 1.99552615195149, 1.45981546432681, 1.18795456176529, 0.883677031907113, 0.506068484016206, 0.278666830942131])    # Paired ({(A-B)_PK}). Model Control
PK_paired_model_reduced_gEE = np.array([0.827025563023372, 1.20535506292282, 1.20942153636047, 1.19144294385502, 1.07782127468347, 0.789219077236436, 0.581056377325586, 0.306326579599523])    # Paired ({(A-B)_PK}). Model gEE x0.9825
PK_paired_model_reduced_gEI = np.array([3.26391521924413, 2.74335173906269, 1.51683520600862, 0.628244266403643, 0.215271080207100, 0.00680136174149171, 0.123109128370268, -0.0550089552969320])    # Paired ({(A-B)_PK}). Model gEI x0.965
PK_paired_model_upstream_deficit = np.array([0.828031787266345, 1.17737201231846, 1.33587634186930, 1.21346932799728, 1.01117650859519, 0.780827400263766, 0.539018005380821, 0.328725043874354])    # Paired ({(A-B)_PK}). Upstream Deficit
PK_paired_err_model_control = np.array([0.0435424957151598, 0.0441661792412214, 0.0441950685360023, 0.0431389910519280, 0.0425994942984684, 0.0424270882795912, 0.0420077616398932, 0.0417431540324621])    # Paired ({(A-B)_PK}). Model Control
PK_paired_err_model_reduced_gEE = np.array([0.0403357753321098, 0.0404955511537648, 0.0406904845891699, 0.0404976533050921, 0.0404884526780699, 0.0402405846100832, 0.0400103174422955, 0.0395997935066256])    # Paired ({(A-B)_PK}). Model gEE x0.9825
PK_paired_err_model_reduced_gEI = np.array([0.0490049599720236, 0.0470901353077473, 0.0443191253875975, 0.0429892385600982, 0.0429013077006419, 0.0429918164812708, 0.0429279850838066, 0.0426954376539772])    # Paired ({(A-B)_PK}). Model gEI x0.965
PK_paired_err_model_upstream_deficit = np.array([0.0403141014478188, 0.0404373506235992, 0.0408671271086513, 0.0405139624271616, 0.0403901003174124, 0.0402115627286864, 0.0399640839317250, 0.0395920798354227])    # Paired ({(A-B)_PK}). Upstream Deficit







## Define subfigure domain.
figsize = (max2,1.3*max2)

width1_11 = 0.22; width1_12 = 0.2; width1_13 = width1_12
width1_21 = 0.1; width1_22 = width1_21; width1_23 = width1_21; width1_24 = width1_11
x1_11 = 0.098; x1_12 = x1_11 + width1_11 + 1.25*xbuf0; x1_13 = x1_12 + width1_12 + 1.15*xbuf0
x1_21 = 0.0825; x1_22 = x1_21 + width1_21 + 0.8*xbuf0; x1_23 = x1_22 + width1_22 + 1.1*xbuf0; x1_24 = x1_23 + width1_23 + 1.55*xbuf0 #x1_24 = x1_23 + width1_23 + 1.25*xbuf0
height1_11 = 0.15; height1_12 = height1_11; height1_13 = height1_12
height1_21= 0.15;  height1_22 = height1_21;  height1_23 = height1_21;  height1_24 = height1_21
y1_11 = 0.8; y1_12 = y1_11; y1_13=y1_12
y1_21 = y1_11 - height1_21 - 0.95*ybuf0; y1_22 = y1_21; y1_23 = y1_22; y1_24 = y1_23 + 0.07*ybuf0



## Define subfigure domain.

model_width1_11 = 0.09; model_width1_12 = 0.135; model_width1_13 = model_width1_12; model_width1_14 = model_width1_12; model_width1_15 = model_width1_12; model_width1_21 = 0.18; model_width1_22 = model_width1_21; model_width1_23 = model_width1_21; model_width1_24 = 0.21; model_width1_31 = 0.4; model_width1_32 = 0.35      # v3
model_x1_11 = 0.07; model_x1_12 = model_x1_11 + model_width1_11 + xbuf0; model_x1_13 = model_x1_12 + model_width1_12 + 0.5*xbuf0; model_x1_14 = model_x1_13 + model_width1_13 + 0.5*xbuf0; model_x1_15 = model_x1_14 + model_width1_14 + 0.5*xbuf0; model_x1_21 = 0.05; model_x1_22 = model_x1_21 + model_width1_21 + 0.45*xbuf0; model_x1_23 = model_x1_22 + model_width1_22 + 0.45*xbuf0; model_x1_24 = model_x1_23 + model_width1_23 + 0.8*xbuf0; model_x1_31=0.07; model_x1_32 = model_x1_31 + model_width1_31 + xbuf0; model_x1_33 = model_x1_32 + model_width1_32 + xbuf0
model_height1_11 = 0.18; model_height1_12 = 0.18; model_height1_13 = model_height1_12; model_height1_14 = model_height1_12; model_height1_15 = model_height1_12; model_height1_21= 0.15;  model_height1_22 = model_height1_21;  model_height1_23 = model_height1_21;  model_height1_24 = model_height1_21;  model_height1_31 = model_height1_21;  model_height1_32 = model_height1_31;  model_height1_33 = model_height1_31
model_y1_11 = 0.29; model_y1_12 = model_y1_11+0.01; model_y1_13 = model_y1_12; model_y1_14 = model_y1_12; model_y1_15 = model_y1_12; model_y1_21 = model_y1_11 - model_height1_22 - 1.*ybuf0; model_y1_22 = model_y1_21; model_y1_23 = model_y1_21; model_y1_24 = model_y1_23+0.05*ybuf0; model_y1_31 = model_y1_21 - model_height1_31 - ybuf0; model_y1_32=model_y1_31; model_y1_33=model_y1_31



rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12_0 = [x1_12, y1_12, width1_12*0.05, height1_12]
rect1_12 = [x1_12+width1_12*0.2, y1_12, width1_12*(1-0.2), height1_12]
rect1_13_0 = [x1_13, y1_13, width1_13*0.05, height1_13]
rect1_13 = [x1_13+width1_13*0.2, y1_13, width1_13*(1-0.2), height1_13]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_24 = [x1_24, y1_24, width1_24, height1_24]

model_rect1_11 = [model_x1_11, model_y1_11, model_width1_11, model_height1_11]
model_rect1_12_0 = [model_x1_12, model_y1_12, model_width1_12*0.05, model_height1_12]
model_rect1_12 = [model_x1_12+model_width1_12*0.2, model_y1_12, model_width1_12*(1-0.2), model_height1_12]
model_rect1_13_0 = [model_x1_13, model_y1_13, model_width1_13*0.05, model_height1_13]
model_rect1_13 = [model_x1_13+model_width1_13*0.2, model_y1_13, model_width1_13*(1-0.2), model_height1_13]
model_rect1_14_0 = [model_x1_14, model_y1_14, model_width1_14*0.05, model_height1_14]
model_rect1_14 = [model_x1_14+model_width1_14*0.2, model_y1_14, model_width1_14*(1-0.2), model_height1_14]
model_rect1_15_0 = [model_x1_15, model_y1_15, model_width1_15*0.05, model_height1_15]
model_rect1_15 = [model_x1_15+model_width1_15*0.2, model_y1_15, model_width1_15*(1-0.2), model_height1_15]
model_rect1_21 = [model_x1_21, model_y1_21, model_width1_21, model_height1_21]
model_rect1_22 = [model_x1_22, model_y1_22, model_width1_22, model_height1_22]
model_rect1_23 = [model_x1_23, model_y1_23, model_width1_23, model_height1_23]
model_rect1_24 = [model_x1_24, model_y1_24, model_width1_24, model_height1_24]
model_rect1_31 = [model_x1_31, model_y1_31, model_width1_31, model_height1_31]
model_rect1_32 = [model_x1_32, model_y1_32, model_width1_32, model_height1_32]


##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.01, 0.952, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.03+x1_12-x1_11, 0.952, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.032+x1_13-x1_11, 0.952, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.01, 0.965 + y1_21 - y1_11, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.047+x1_22-x1_21, 0.965 + y1_21 - y1_11, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.049+x1_23-x1_21, 0.965 + y1_21 - y1_11, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.02+x1_24-x1_21, 0.965 + y1_21 - y1_11, 'G', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.
fig_temp.text(0.525, 0.96, 'Saline', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.51+x1_13-x1_12, 0.96, 'Ketamine', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.144-x1_11+x1_21, 0.963 + y1_21 - y1_11, 'Mean Evidence\nBeta', fontsize=fontsize_fig_label-1, rotation='horizontal', color='k', va='center', horizontalalignment='center')
fig_temp.text(0.3325-x1_11+x1_21, 0.963 + y1_21 - y1_11, 'SD Evidence\nBeta', fontsize=fontsize_fig_label-1, rotation='horizontal', color='k', va='center', horizontalalignment='center')
fig_temp.text(0.5565-x1_11+x1_21, 0.963 + y1_21 - y1_11, 'PVB Index', fontsize=fontsize_fig_label-1, rotation='horizontal', color='k', ha='center', va='center')

fig_temp.text(0.01, 1.002 + model_y1_11 - y1_11, 'H', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.02+model_x1_12-model_x1_11, 1.002 + model_y1_11 - y1_11, 'I', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.01, 0.962 + model_y1_21 - y1_11, 'M', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.0478, 0.96 + model_y1_21 - y1_11, 'Mean Evidence Beta', fontsize=fontsize_fig_label, rotation='horizontal', color='k')
fig_temp.text(0.025+model_x1_22-model_x1_21, 0.962 + model_y1_21 - y1_11, 'N', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.288, 0.96 + model_y1_21 - y1_11, 'SD Evidence Beta', fontsize=fontsize_fig_label, rotation='horizontal', color='k')
fig_temp.text(0.023+model_x1_23-model_x1_21, 0.962 + model_y1_21 - y1_11, 'O', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.599, 0.96 + model_y1_21 - y1_11, 'PVB Index', fontsize=fontsize_fig_label, rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(-0.005+model_x1_24-model_x1_21, 0.962 + model_y1_21 - y1_11, 'P', fontsize=fontsize_fig_label, fontweight='bold')

fig_temp.text(0.295, 1. + model_y1_11 - y1_11, 'Control', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.275+model_x1_13-model_x1_12, 1. + model_y1_11 - y1_11, 'Lowered E/I', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.275+model_x1_14-model_x1_12, 1. + model_y1_11 - y1_11, 'Elevated E/I', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.335+model_x1_15-model_x1_12, 0.993 + model_y1_11 - y1_11, 'Sensory\nDeficit', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(0.032+model_x1_13-model_x1_11, 1.002 + model_y1_11 - y1_11, 'J', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.032+model_x1_14-model_x1_11, 1.002 + model_y1_11 - y1_11, 'K', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.032+model_x1_15-model_x1_11, 1.002 + model_y1_11 - y1_11, 'L', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.



############### Plotting monkey H data


## rect1_11: Correct Probability vs time, Both Monkeys
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax)
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_A, color=color_list_expt[0], linestyle='-', zorder=3, clip_on=False, label='Saline', linewidth=1.)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_A + Pcorr_t_se_list_saline_A, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_A - Pcorr_t_se_list_saline_A, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_A, color=color_list_expt[1], linestyle='-', zorder=3, clip_on=False, label='Ketamine', linewidth=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_A + Pcorr_t_se_list_ketamine_A, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_A - Pcorr_t_se_list_ketamine_A, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
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
legend = ax.legend(loc=(0.5,-0.03), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
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
ax.errorbar( d_evidence_A_saline_list[:],    P_corr_A_saline_list[:], ErrBar_P_corr_A_saline_list[:], color=color_list_expt[0], ecolor=color_list_expt[0], fmt='.', zorder=4, clip_on=False , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_A_saline_list[:], 1.-P_corr_A_saline_list[:], ErrBar_P_corr_A_saline_list[:], color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[0]], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_A_saline_all, x_list_psychometric), color=color_list_expt[0], ls='-', clip_on=False, zorder=3, label='Higher SD Correct')#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_A_saline_all, -x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], ls='-', clip_on=False, zorder=2, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_A_saline_all, x0_psychometric), s=15., color=color_list_expt[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_A_saline_all, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.072
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(0.01,-0.075), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0., labelspacing=0.3)
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
ax.errorbar( d_evidence_A_ket_list[:],    P_corr_A_ket_list[:], ErrBar_P_corr_A_ket_list[:], color=color_list_expt[1], ecolor=color_list_expt[1], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_A_ket_list[:], 1.-P_corr_A_ket_list[:], ErrBar_P_corr_A_ket_list[:], color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[1]], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_A_ketamine_all, x_list_psychometric, 0.118), color=color_list_expt[1], ls='-', clip_on=False, zorder=3, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_A_ketamine_all, -x_list_psychometric, 0.118), color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ls='-', clip_on=False, zorder=2, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_A_ketamine_all, x0_psychometric, 0.118), s=15., color=color_list_expt[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_A_ketamine_all, -x0_psychometric, 0.118), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.072
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(-0.45,0.74), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0.)
for color,text,item in zip([color_list_expt[1], [1-(1-ci)*0.5 for ci in color_list_expt[1]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)



## rect1_21: Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(mean_effect_list_A)), mean_effect_list_A, bar_width_compare3, alpha=bar_opacity, yerr=Mean_reg_err_bars_A_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(mean_effect_list_A_preSK[0]+mean_effect_list_A_preSK[1]), 0.5*(mean_effect_list_A_preSK[0]+mean_effect_list_A_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.], [29.], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [27.8, 27.8], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
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

## rect1_22: Variance Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(var_effect_list_A)), var_effect_list_A, bar_width_compare3, alpha=bar_opacity, yerr=Var_reg_err_bars_A_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(var_effect_list_A_preSK[0]+var_effect_list_A_preSK[1]), 0.5*(var_effect_list_A_preSK[0]+var_effect_list_A_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.set_xlim([0,len(var_effect_list_A)-1+bar_width_compare3])
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

## rect1_23: Variance Beta/ Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(var_mean_ratio_list_A)), var_mean_ratio_list_A, bar_width_compare3, alpha=bar_opacity, yerr=Var_mean_ratio_err_Reg_bars_A_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(var_mean_ratio_list_A_preSK[0]+var_mean_ratio_list_A_preSK[1]), 0.5*(var_mean_ratio_list_A_preSK[0]+var_mean_ratio_list_A_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.], [0.595], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [0.57,0.57], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
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



## rect1_24: Psychophysical Kernel, Monkey A
ax   = fig_temp.add_axes(rect1_24)
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
ax.set_xlim([1,6])
ax.set_xticks([1,6])
ax.set_yticks([0., 5.])
ax.text(0.15, 5.5, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.plot(i_PK_list_6, PK_A_saline, label='Saline', color=color_list_expt[0], linestyle='-', zorder=0, clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(i_PK_list_6, PK_A_ketamine, label='Ketamine', color=color_list_expt[1], linestyle='-', zorder=0, clip_on=False)#, linestyle=linestyle_list[i_var_a])
legend = ax.legend(loc=(-0.05,0.4), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0.)
for color,text,item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)
##################### Plotting monkey H data

## rect1_11: E/I perturbation schematics

## rect1_12: Psychometric function. Control model.
ax_0   = fig_temp.add_axes(model_rect1_12_0)
ax   = fig_temp.add_axes(model_rect1_12)
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
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_model_control, x_list_psychometric, 0.118), color=color_list[0], ls='-', clip_on=False, label='Higher SD Correct')#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_control, -x_list_psychometric, 0.118), color=[1-(1-ci)*0.5 for ci in color_list[0]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_model_control, x0_psychometric, 0.118), s=15., color=color_list[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_control, -x0_psychometric, 0.118), s=15., color=[1-(1-ci)*0.5 for ci in color_list[0]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
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
y_shift_spines = -0.06
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
ax_0   = fig_temp.add_axes(model_rect1_13_0)
ax   = fig_temp.add_axes(model_rect1_13)
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
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_model_reduced_gEE, x_list_psychometric, 0.118), color=color_list[1], ls='-', clip_on=False, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_reduced_gEE, -x_list_psychometric, 0.118), color=[1-(1-ci)*0.5 for ci in color_list[1]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_model_reduced_gEE, x0_psychometric, 0.118), s=15., color=color_list[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_reduced_gEE, -x0_psychometric, 0.118), s=15., color=[1-(1-ci)*0.5 for ci in color_list[1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
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
y_shift_spines = -0.06
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
ax_0   = fig_temp.add_axes(model_rect1_14_0)
ax   = fig_temp.add_axes(model_rect1_14)
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
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_model_reduced_gEI, x_list_psychometric, 0.118), color=color_list[2], ls='-', clip_on=False, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_reduced_gEI, -x_list_psychometric, 0.118), color=[1-(1-ci)*0.5 for ci in color_list[2]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_model_reduced_gEI, x0_psychometric, 0.118), s=15., color=color_list[2], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_reduced_gEI, -x0_psychometric, 0.118), s=15., color=[1-(1-ci)*0.5 for ci in color_list[2]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
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
y_shift_spines = -0.06
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
ax_0   = fig_temp.add_axes(model_rect1_15_0)
ax   = fig_temp.add_axes(model_rect1_15)
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
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_model_upstream_deficit, x_list_psychometric, 0.118), color=color_list[3], ls='-', clip_on=False, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_upstream_deficit, -x_list_psychometric, 0.118), color=[1-(1-ci)*0.5 for ci in color_list[3]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_model_upstream_deficit, x0_psychometric, 0.118), s=15., color=color_list[3], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_upstream_deficit, -x0_psychometric, 0.118), s=15., color=[1-(1-ci)*0.5 for ci in color_list[3]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
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
y_shift_spines = -0.06
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
ax   = fig_temp.add_axes(model_rect1_21)
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
ax   = fig_temp.add_axes(model_rect1_22)
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
ax   = fig_temp.add_axes(model_rect1_23)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(Reg_ratio_models)), Reg_ratio_models, bar_width_compare3, alpha=bar_opacity, yerr=Reg_err_ratio_models, ecolor='k', color=color_list[0:3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.bar(3, Reg_ratio_models[3], bar_width_compare3, alpha=bar_opacity, yerr=Reg_err_ratio_models[3], ecolor='k', color=color_list[3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,4.*bar_width_compare3], [Reg_ratio_models[0], Reg_ratio_models[0]], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1., 1.5], [0.508, 0.546], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [0.49,0.49], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,2.5], [0.528,0.528], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_xlim([0,len(Reg_ratio_models)-1+bar_width_compare3])
ax.set_ylim([0.,0.54])
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
ax   = fig_temp.add_axes(model_rect1_24)
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
tmp2 = ax.scatter( i_PK_list, PK_paired_model_reduced_gEE     , color=color_list[1], marker='^', zorder=3, clip_on=False, facecolors=color_list[1], edgecolors='k', linewidths=0.6, s=4., label=label_list_expt[1])#, linestyle=linestyle_list[i_var_a])
tmp3 = ax.scatter(i_PK_list, PK_paired_model_reduced_gEI      , color=color_list[2], marker='s', zorder=2, clip_on=False, facecolors=color_list[2], edgecolors='k', linewidths=0.6, s=3.5, label=label_list_expt[2])#, linestyle=linestyle_list[i_var_a])
tmp4 = ax.scatter( i_PK_list, PK_paired_model_upstream_deficit, color=color_list[3], marker='x', zorder=1, clip_on=False, facecolors=color_list[3], edgecolors=color_list[3], linewidths=0.6, s=5., label=label_list[3])#, linestyle=linestyle_list[i_var_a])
legend = ax.legend(loc=(0.3,0.55), fontsize=fontsize_legend-1.5, frameon=False, ncol=1, columnspacing=1., handletextpad=0., scatterpoints=1)
for color,text,item in zip(color_list, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    # item.set_visible(False)

fig_temp.savefig(path_cwd+'Figure8S8.pdf')    #Finally save fig

########################################################################################################################
########################################################################################################################
########################################################################################################################
### Figure 7 but with built in lapse (monkey H lapse rate=0.0684).



####################################################### Monkey H data
### Figure 6: Ketamine Data
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
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
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
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
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
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
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
## Combining across monkeys (n_A=n_H=16). Using regular, narrow-broad, and half-half trials (no control-non-integrating trials).
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
# ## Without Lapse
psychometric_params_A_saline_all        = [0.0595776866237313, 1.26810162179331, 0.0138702695806634]
psychometric_params_H_saline_all        = [0.0681413425053521, 1.07582639210372, 0.0123764957351213]
## With Lapse
psychometric_params_A_ketamine_all      = [0.164267968758472, 0.732705192383852, 0.0377990600679478]
psychometric_params_H_ketamine_all      = [0.130851990508893, 1.16584379279672, 0.0238689326833176]



####################################################### Monkey H data


## Probability Correct, using regression trials                                                                         # See MainAnalysisNonDrugDays_NL.m: line 80-104
d_evidence_model_control_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_control = np.array([0.0927192982456140, 0.151698352344740, 0.205910572892408, 0.275546161650321, 0.349400393184797, 0.394795769511306, 0.448809413936317, 0.485969788519637, 0.517137129109864, 0.543479373201151, 0.610768000000000, 0.624668930390493, 0.652772036474164, 0.693208652546192, 0.737635306937886, 0.779317824583464, 0.828690558235700, 0.874956874682902, 0.898689095127609, 0.920191082802548])                              # Log-Spaced.
ErrBar_P_corr_model_control = np.array([0.0270849181720042, 0.0127671247436430, 0.00872621538316132, 0.00821582871028207, 0.00862988281486234, 0.00933432680366009, 0.0106837655049761, 0.0122847824617512, 0.0141494880493981, 0.00890729231272248, 0.00872159659394540, 0.0141062928135158, 0.0117375001726336, 0.00978910912137285, 0.00838388647739514, 0.00735247714217562, 0.00699364561211562, 0.00744936694475905, 0.0102732590916659, 0.0215678112946541])       # Log-Spaced.
d_evidence_model_reduced_gEE_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_reduced_gEE = np.array([0.104758620689655, 0.211585956416465, 0.291563088512241, 0.368743036211699, 0.431059667427454, 0.455649591685226, 0.479647658785350, 0.495885885885886, 0.504001623376623, 0.533424828446662, 0.564540192926045, 0.605315922493682, 0.611295857988166, 0.636375227686703, 0.665858514041948, 0.707515822784810, 0.769441764300482, 0.833559408465069, 0.890240096038416, 0.920129870129870])
ErrBar_P_corr_model_reduced_gEE = np.array([0.0253764150755095, 0.0142076902415485, 0.00986076162581143, 0.00900226426525581, 0.00894186896926475, 0.00959483164035966, 0.0107561805552604, 0.0122525807869502, 0.0142430816942549, 0.00881055602847032, 0.00889035737148525, 0.0141854331773540, 0.0118567532719241, 0.0102646110858199, 0.00889314804863894, 0.00809200965506889, 0.00781816017000590, 0.00841040697042555, 0.0108265438680730, 0.0217714584565852])
d_evidence_model_reduced_gEI_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_reduced_gEI = np.array([0.193034482758621, 0.205690072639225, 0.253206214689266, 0.303910167130919, 0.368444734268014, 0.411388270230141, 0.436198423736671, 0.458438438438438, 0.478733766233766, 0.506718652526513, 0.555649517684888, 0.570724515585510, 0.597349112426036, 0.610414389799636, 0.640991823675791, 0.696781645569620, 0.731467953135768, 0.780489546149924, 0.833781512605042, 0.872012987012987])
ErrBar_P_corr_model_reduced_gEI = np.array([0.0327355698360997, 0.0140622610938410, 0.00943476384121710, 0.00858213354346191, 0.00870993298406751, 0.00948028783430796, 0.0106771789312913, 0.0122101766886540, 0.0142308523985623, 0.00882942280240997, 0.00890965345155703, 0.0143652287705804, 0.0119290833164658, 0.0104058280000830, 0.00904432235123828, 0.00817640611719357, 0.00822668294121627, 0.00934608661578220, 0.0128954254162662, 0.0268677923376304])
d_evidence_model_upstream_deficit_list =  100.*np.array([-0.286748465774755, -0.205561706560439, -0.147361259945616, -0.105639038010100, -0.0757295801882930, -0.0542883523318981, -0.0389177543515278, -0.0278990158792484, -0.0200000000000000, -0.024, 0.024, 0.0200000000000000, 0.0278990158792484, 0.0389177543515278, 0.0542883523318981, 0.0757295801882930, 0.105639038010100, 0.147361259945616, 0.205561706560439, 0.286748465774755])#, 0.500000000000000])  # Log-Spaced.
P_corr_model_upstream_deficit = np.array([0.174000000000000, 0.209079903147700, 0.283865348399247, 0.338951949860724, 0.409752200847734, 0.433051224944320, 0.472012053778396, 0.466018018018018, 0.481534090909091, 0.524606986899563, 0.565344051446945, 0.601710193765796, 0.600562130177515, 0.625582877959927, 0.647063633131888, 0.695281645569620, 0.758480358373536, 0.817032126466089, 0.870900360144058, 0.897532467532468])
ErrBar_P_corr_model_upstream_deficit = np.array([0.0314309984456560, 0.0141459572683668, 0.00978241379524509, 0.00883222429220661, 0.00887981485340860, 0.00954599114630286, 0.0107481724967842, 0.0122242773397859, 0.0142339396835127, 0.00881958695676621, 0.00888849055172082, 0.0142076608176763, 0.0119132775279769, 0.0103270511987895, 0.00900980138868110, 0.00818782273154843, 0.00794468735380149, 0.00873024454639892, 0.0116134000975043, 0.0243811862273613])

## non-binned MLE (i.e. done using literal net evidence, via matlab). See Psychometric_function_fit_model_NL.m.
x_list_psychometric = np.arange(0.01, 0.5, 0.01)                                                                        # See figure_psychometric_function_fit.py, esp lines 633-687
x0_psychometric = 0.
psychometric_params_model_control = [0.100059921579512, 1.17296720944422, 0.0252108221439562]
psychometric_params_model_reduced_gEE = [0.141640569695278, 1.48996062603167, 0.0333574127517644]
psychometric_params_model_reduced_gEI = [0.152194541525120, 1.07396300894210, 0.0133434710517555]
psychometric_params_model_upstream_deficit = [0.146057325838743, 1.30154857708728, 0.0252605465246046]

##  Mean & Variance, LR differences, Constrained across-trials
Reg_bars_LRdiff_model_control = np.array([-0.00954709079332105, 11.7803724129835, 4.11464026510856])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_control = np.array([0.0148972366676718, 0.155160289387647, 0.169874336437954])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_lowered_EI = np.array([-0.0334935825856715, 8.31775930321168, 3.61174101024210])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_lowered_EI = np.array([0.0142929677510426, 0.137057097991299, 0.162639305874748])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_elevated_EI = np.array([-0.00240748358997683, 8.30457386861440, 1.54801768626000])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_elevated_EI = np.array([0.0142726162142577, 0.135941277489426, 0.160386155919356])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_LRdiff_model_upstream_deficit = np.array([-0.00427719843657431, 8.24342372935718, 2.97670542327232])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.
Reg_bars_err_LRdiff_model_upstream_deficit = np.array([0.0142739726210132, 0.136211257956017, 0.161508031372483])  # [Bias, LeftVal-RightVal, LeftVar-RightVar]. Model Control regression Beta values.


Reg_mean_models = np.array([Reg_bars_LRdiff_model_control[1], Reg_bars_LRdiff_model_lowered_EI[1], Reg_bars_LRdiff_model_elevated_EI[1], Reg_bars_LRdiff_model_upstream_deficit[1]])
Reg_std_models = np.array([Reg_bars_LRdiff_model_control[2], Reg_bars_LRdiff_model_lowered_EI[2], Reg_bars_LRdiff_model_elevated_EI[2], Reg_bars_LRdiff_model_upstream_deficit[2]])
Reg_ratio_models = Reg_std_models / Reg_mean_models
Reg_err_mean_models = np.array([Reg_bars_err_LRdiff_model_control[1], Reg_bars_err_LRdiff_model_lowered_EI[1], Reg_bars_err_LRdiff_model_elevated_EI[1], Reg_bars_err_LRdiff_model_upstream_deficit[1]])
Reg_err_std_models = np.array([Reg_bars_err_LRdiff_model_control[2], Reg_bars_err_LRdiff_model_lowered_EI[2], Reg_bars_err_LRdiff_model_elevated_EI[2], Reg_bars_err_LRdiff_model_upstream_deficit[2]])
Reg_err_ratio_models = Reg_ratio_models *( (Reg_err_mean_models/Reg_mean_models)**2 + (Reg_err_std_models/Reg_std_models)**2)**0.5


## First, Last, Mean, Max, Min                                                                                          # See MainAnalysisNonDrugDays_NL.m: LongAvCOL, LongAvCOLSE
Reg_values_control = np.array([-0.627862266598890, 0.358270945283961, -1.42748028308022, 12.9203111968442, 1.98851485201843, -0.700941049117172, -0.430766093034292, 1.28231805887081, -11.8052181533779, -1.84575732810298, 0.871677888402864])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_lowered_EI = np.array([0.156973474131819, -0.190517956573965, -0.873127531778522, 8.72489985018739, 1.42548217749694, -0.863601274038313, 0.247942429512494, 0.781458934337603, -9.16942843660793, -1.37706397137125, 0.800662497428994])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_elevated_EI = np.array([-0.247607883150358, 2.78569949253798, -1.06842432131352, 7.32396509956650, 0.887046570346137, -0.197010565931470, -2.64303840944531, 1.02993620125854, -8.01234360283582, -0.352514307197746, 0.702600738530501])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_upstream_deficit = np.array([-0.0121736957391923, -0.133830809324166, -0.758976615479105, 9.21321803012014, 0.918814666123908, -0.872167489234618, 0.213842293619169, 0.776991171815005, -8.99149895045293, -1.10197869755712, 0.713643925711240])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)

Reg_values_err_control = np.array([0.147310057837789, 0.0767149793754011, 0.0779442254222985, 0.350741124594866, 0.157686968349207, 0.155976076891348, 0.0564273497279154, 0.0578567340641490, 0.264444043353272, 0.132514754479815, 0.134158951088744])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_lowered_EI = np.array([0.139297924897953, 0.0725124858752832, 0.0729881289889842, 0.322941251038777, 0.149042823075995, 0.147618911871303, 0.0536776434522146, 0.0541993358970680, 0.244877068349006, 0.125890028688845, 0.127374572258127])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_elevated_EI = np.array([0.146204508793301, 0.0785437262287538, 0.0771269832656815, 0.337804925178140, 0.157082912135844, 0.155584289476061, 0.0581703015331351, 0.0568916829323742, 0.254726370589544, 0.131840236276928, 0.132843331051045])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)
Reg_values_err_upstream_deficit = np.array([0.138710401193466, 0.0722492493950381, 0.0726806060942897, 0.322758267192545, 0.148333058363656, 0.147151666317276, 0.0533086376480508, 0.0538525459439464, 0.242942922654811, 0.125284602487200, 0.126460244203009])  # Bias, Left: first/last/average/max/min, Right: first/last/average/max/min (no L, R)

# PK                                                                                                                    # See MainAnalysisNonDrugDays_NL.m: lines 130-136.
t_PK_list = 0.125 + 0.25*np.arange(8)
PK_paired_model_control = np.array([2.08830763632344, 2.59216495211791, 2.43527596256224, 1.88247557003053, 1.44330992992352, 1.07602583502591, 0.658766442697616, 0.306016390926839])    # Paired ({(A-B)_PK}). Model Control
PK_paired_model_reduced_gEE = np.array([0.939941782583566, 1.37307218449301, 1.42965647022848, 1.38752465302606, 1.24678590701743, 0.922620198346854, 0.688116367748709, 0.359286813802251])    # Paired ({(A-B)_PK}). Model gEE x0.9825
PK_paired_model_reduced_gEI = np.array([4.14369648830275, 3.45607222655065, 1.86952468234253, 0.788575107102589, 0.323472374048294, 0.00574050876971110, 0.135809352352413, -0.0566354076656522])    # Paired ({(A-B)_PK}). Model gEI x0.965
PK_paired_model_upstream_deficit = np.array([0.966108694713330, 1.36659451532731, 1.47792658791781, 1.38196937700048, 1.14774516313742, 0.914703064970518, 0.663583155524308, 0.385421519922818])    # Paired ({(A-B)_PK}). Upstream Deficit

PK_paired_err_model_control = np.array([0.0460811126696094, 0.0472028415553903, 0.0469714372923288, 0.0456739255856088, 0.0446914636232892, 0.0443511052582633, 0.0438106167994995, 0.0434162156831403])    # Paired ({(A-B)_PK}). Model Control
PK_paired_err_model_reduced_gEE = np.array([0.0411463547779118, 0.0414442789292361, 0.0417246315738134, 0.0414874178840956, 0.0414040736109774, 0.0410681698595270, 0.0407840487547582, 0.0402967289027768])    # Paired ({(A-B)_PK}). Model gEE x0.9825
PK_paired_err_model_reduced_gEI = np.array([0.0547052508355091, 0.0518192527695535, 0.0473077093396017, 0.0453996267044160, 0.0451944391043623, 0.0452602246429945, 0.0451982591931595, 0.0449767749505234])    # Paired ({(A-B)_PK}). Model gEI x0.965
PK_paired_err_model_upstream_deficit = np.array([0.0410437373186125, 0.0413079154055605, 0.0416851846442376, 0.0413544157334006, 0.0411436515621505, 0.0409337593835532, 0.0406403678031242, 0.0401901261604235])    # Paired ({(A-B)_PK}). Upstream Deficit







## Define subfigure domain.
figsize = (max2,1.3*max2)

width1_11 = 0.22; width1_12 = 0.2; width1_13 = width1_12
width1_21 = 0.1; width1_22 = width1_21; width1_23 = width1_21; width1_24 = width1_11
x1_11 = 0.098; x1_12 = x1_11 + width1_11 + 1.25*xbuf0; x1_13 = x1_12 + width1_12 + 1.15*xbuf0
x1_21 = 0.0825; x1_22 = x1_21 + width1_21 + 0.8*xbuf0; x1_23 = x1_22 + width1_22 + 1.1*xbuf0; x1_24 = x1_23 + width1_23 + 1.55*xbuf0 #x1_24 = x1_23 + width1_23 + 1.25*xbuf0
height1_11 = 0.15; height1_12 = height1_11; height1_13 = height1_12
height1_21= 0.15;  height1_22 = height1_21;  height1_23 = height1_21;  height1_24 = height1_21
y1_11 = 0.8; y1_12 = y1_11; y1_13=y1_12
y1_21 = y1_11 - height1_21 - 0.95*ybuf0; y1_22 = y1_21; y1_23 = y1_22; y1_24 = y1_23 + 0.07*ybuf0



## Define subfigure domain.

model_width1_11 = 0.09; model_width1_12 = 0.135; model_width1_13 = model_width1_12; model_width1_14 = model_width1_12; model_width1_15 = model_width1_12; model_width1_21 = 0.18; model_width1_22 = model_width1_21; model_width1_23 = model_width1_21; model_width1_24 = 0.21; model_width1_31 = 0.4; model_width1_32 = 0.35      # v3
model_x1_11 = 0.07; model_x1_12 = model_x1_11 + model_width1_11 + xbuf0; model_x1_13 = model_x1_12 + model_width1_12 + 0.5*xbuf0; model_x1_14 = model_x1_13 + model_width1_13 + 0.5*xbuf0; model_x1_15 = model_x1_14 + model_width1_14 + 0.5*xbuf0; model_x1_21 = 0.05; model_x1_22 = model_x1_21 + model_width1_21 + 0.45*xbuf0; model_x1_23 = model_x1_22 + model_width1_22 + 0.45*xbuf0; model_x1_24 = model_x1_23 + model_width1_23 + 0.8*xbuf0; model_x1_31=0.07; model_x1_32 = model_x1_31 + model_width1_31 + xbuf0; model_x1_33 = model_x1_32 + model_width1_32 + xbuf0
model_height1_11 = 0.18; model_height1_12 = 0.18; model_height1_13 = model_height1_12; model_height1_14 = model_height1_12; model_height1_15 = model_height1_12; model_height1_21= 0.15;  model_height1_22 = model_height1_21;  model_height1_23 = model_height1_21;  model_height1_24 = model_height1_21;  model_height1_31 = model_height1_21;  model_height1_32 = model_height1_31;  model_height1_33 = model_height1_31
model_y1_11 = 0.29; model_y1_12 = model_y1_11+0.01; model_y1_13 = model_y1_12; model_y1_14 = model_y1_12; model_y1_15 = model_y1_12; model_y1_21 = model_y1_11 - model_height1_22 - 1.*ybuf0; model_y1_22 = model_y1_21; model_y1_23 = model_y1_21; model_y1_24 = model_y1_23+0.05*ybuf0; model_y1_31 = model_y1_21 - model_height1_31 - ybuf0; model_y1_32=model_y1_31; model_y1_33=model_y1_31



rect1_11 = [x1_11, y1_11, width1_11, height1_11]
rect1_12_0 = [x1_12, y1_12, width1_12*0.05, height1_12]
rect1_12 = [x1_12+width1_12*0.2, y1_12, width1_12*(1-0.2), height1_12]
rect1_13_0 = [x1_13, y1_13, width1_13*0.05, height1_13]
rect1_13 = [x1_13+width1_13*0.2, y1_13, width1_13*(1-0.2), height1_13]
rect1_21 = [x1_21, y1_21, width1_21, height1_21]
rect1_22 = [x1_22, y1_22, width1_22, height1_22]
rect1_23 = [x1_23, y1_23, width1_23, height1_23]
rect1_24 = [x1_24, y1_24, width1_24, height1_24]

model_rect1_11 = [model_x1_11, model_y1_11, model_width1_11, model_height1_11]
model_rect1_12_0 = [model_x1_12, model_y1_12, model_width1_12*0.05, model_height1_12]
model_rect1_12 = [model_x1_12+model_width1_12*0.2, model_y1_12, model_width1_12*(1-0.2), model_height1_12]
model_rect1_13_0 = [model_x1_13, model_y1_13, model_width1_13*0.05, model_height1_13]
model_rect1_13 = [model_x1_13+model_width1_13*0.2, model_y1_13, model_width1_13*(1-0.2), model_height1_13]
model_rect1_14_0 = [model_x1_14, model_y1_14, model_width1_14*0.05, model_height1_14]
model_rect1_14 = [model_x1_14+model_width1_14*0.2, model_y1_14, model_width1_14*(1-0.2), model_height1_14]
model_rect1_15_0 = [model_x1_15, model_y1_15, model_width1_15*0.05, model_height1_15]
model_rect1_15 = [model_x1_15+model_width1_15*0.2, model_y1_15, model_width1_15*(1-0.2), model_height1_15]
model_rect1_21 = [model_x1_21, model_y1_21, model_width1_21, model_height1_21]
model_rect1_22 = [model_x1_22, model_y1_22, model_width1_22, model_height1_22]
model_rect1_23 = [model_x1_23, model_y1_23, model_width1_23, model_height1_23]
model_rect1_24 = [model_x1_24, model_y1_24, model_width1_24, model_height1_24]
model_rect1_31 = [model_x1_31, model_y1_31, model_width1_31, model_height1_31]
model_rect1_32 = [model_x1_32, model_y1_32, model_width1_32, model_height1_32]


##### Plotting
##### Plotting
fig_temp = plt.figure(figsize=figsize)
fig_temp.text(0.01, 0.952, 'A', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.03+x1_12-x1_11, 0.952, 'B', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.032+x1_13-x1_11, 0.952, 'C', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.01, 0.965 + y1_21 - y1_11, 'D', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.047+x1_22-x1_21, 0.965 + y1_21 - y1_11, 'E', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.049+x1_23-x1_21, 0.965 + y1_21 - y1_11, 'F', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.02+x1_24-x1_21, 0.965 + y1_21 - y1_11, 'G', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.
fig_temp.text(0.525, 0.96, 'Saline', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.51+x1_13-x1_12, 0.96, 'Ketamine', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.144-x1_11+x1_21, 0.963 + y1_21 - y1_11, 'Mean Evidence\nBeta', fontsize=fontsize_fig_label-1, rotation='horizontal', color='k', va='center', horizontalalignment='center')
fig_temp.text(0.3325-x1_11+x1_21, 0.963 + y1_21 - y1_11, 'SD Evidence\nBeta', fontsize=fontsize_fig_label-1, rotation='horizontal', color='k', va='center', horizontalalignment='center')
fig_temp.text(0.5565-x1_11+x1_21, 0.963 + y1_21 - y1_11, 'PVB Index', fontsize=fontsize_fig_label-1, rotation='horizontal', color='k', ha='center', va='center')

fig_temp.text(0.01, 1.002 + model_y1_11 - y1_11, 'H', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.02+model_x1_12-model_x1_11, 1.002 + model_y1_11 - y1_11, 'I', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.01, 0.962 + model_y1_21 - y1_11, 'M', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.0478, 0.96 + model_y1_21 - y1_11, 'Mean Evidence Beta', fontsize=fontsize_fig_label, rotation='horizontal', color='k')
fig_temp.text(0.025+model_x1_22-model_x1_21, 0.962 + model_y1_21 - y1_11, 'N', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.288, 0.96 + model_y1_21 - y1_11, 'SD Evidence Beta', fontsize=fontsize_fig_label, rotation='horizontal', color='k')
fig_temp.text(0.023+model_x1_23-model_x1_21, 0.962 + model_y1_21 - y1_11, 'O', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.599, 0.96 + model_y1_21 - y1_11, 'PVB Index', fontsize=fontsize_fig_label, rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(-0.005+model_x1_24-model_x1_21, 0.962 + model_y1_21 - y1_11, 'P', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.295, 1. + model_y1_11 - y1_11, 'Control', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.275+model_x1_13-model_x1_12, 1. + model_y1_11 - y1_11, 'Lowered E/I', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.275+model_x1_14-model_x1_12, 1. + model_y1_11 - y1_11, 'Elevated E/I', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k')
fig_temp.text(0.335+model_x1_15-model_x1_12, 0.993 + model_y1_11 - y1_11, 'Sensory\nDeficit', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')
fig_temp.text(0.032+model_x1_13-model_x1_11, 1.002 + model_y1_11 - y1_11, 'J', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.032+model_x1_14-model_x1_11, 1.002 + model_y1_11 - y1_11, 'K', fontsize=fontsize_fig_label, fontweight='bold')
fig_temp.text(0.032+model_x1_15-model_x1_11, 1.002 + model_y1_11 - y1_11, 'L', fontsize=fontsize_fig_label, fontweight='bold')
bar_width_compare3 = 1.



############### Plotting monkey H data


## rect1_11: Correct Probability vs time, Both Monkeys
# TODO: fix colors.
ax   = fig_temp.add_axes(rect1_11)
fig_funs.remove_topright_spines(ax)
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_H, color=color_list_expt[0], linestyle='-', zorder=3, clip_on=False, label='Saline', linewidth=1.)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_H + Pcorr_t_se_list_saline_H, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_H - Pcorr_t_se_list_saline_H, color=color_list_expt[0], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_H, color=color_list_expt[1], linestyle='-', zorder=3, clip_on=False, label='Ketamine', linewidth=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_H + Pcorr_t_se_list_ketamine_H, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_H - Pcorr_t_se_list_ketamine_H, color=color_list_expt[1], linestyle='-', zorder=2, clip_on=False, linewidth=0.5)#, linestyle=linestyle_list[i_var_a])
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
legend = ax.legend(loc=(0.5,-0.03), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=-1., columnspacing=1., handletextpad=0.2)
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
ax.errorbar( d_evidence_H_saline_list[:],    P_corr_H_saline_list[:], ErrBar_P_corr_H_saline_list[:], color=color_list_expt[0], ecolor=color_list_expt[0], fmt='.', zorder=4, clip_on=False , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_H_saline_list[:], 1.-P_corr_H_saline_list[:], ErrBar_P_corr_H_saline_list[:], color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[0]], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, Psychometric_function_D(psychometric_params_H_saline_all, x_list_psychometric), color=color_list_expt[0], ls='-', clip_on=False, zorder=3, label='Higher SD Correct')#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D(psychometric_params_H_saline_all, -x_list_psychometric), color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], ls='-', clip_on=False, zorder=2, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D(psychometric_params_H_saline_all, x0_psychometric), s=15., color=color_list_expt[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D(psychometric_params_H_saline_all, -x0_psychometric), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[0]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.072
# y_shift_spines = -0.08864
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(0.01,-0.075), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0., labelspacing=0.3)
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
ax.errorbar( d_evidence_H_ket_list[:],    P_corr_H_ket_list[:], ErrBar_P_corr_H_ket_list[:], color=color_list_expt[1], ecolor=color_list_expt[1], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.errorbar(-d_evidence_H_ket_list[:], 1.-P_corr_H_ket_list[:], ErrBar_P_corr_H_ket_list[:], color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ecolor=[1-(1-ci)*0.5 for ci in color_list_expt[1]], fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_H_ketamine_all, x_list_psychometric, 0.0684), color=color_list_expt[1], ls='-', clip_on=False, zorder=3, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_H_ketamine_all, -x_list_psychometric, 0.0684), color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], ls='-', clip_on=False, zorder=2, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_H_ketamine_all, x0_psychometric, 0.0684), s=15., color=color_list_expt[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_H_ketamine_all, -x0_psychometric, 0.0684), s=15., color=[1-(1-ci)*0.5 for ci in color_list_expt[1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
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
kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=0.8, clip_on=False)
y_shift_spines = -0.072
# y_shift_spines = -0.08864
ax_0.plot((1      , 1+2./3.), (y_shift_spines+0.  ,y_shift_spines+0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+2./3., 1+4./3,), (y_shift_spines+0.05,y_shift_spines-0.05), **kwargs)        # top-left diagonal
ax_0.plot((1+4./3., 1+6./3.), (y_shift_spines-0.05,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.plot((1+6./3., 1+9./3.), (y_shift_spines+0.  ,y_shift_spines+0.)  , **kwargs)        # top-left diagonal
ax_0.spines['left'].set_position(('outward',5))
ax_0.spines['bottom'].set_position(('outward',7))
ax.spines['bottom'].set_position(('outward',7))
legend = ax.legend(loc=(-0.45,0.74), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0.)
for color,text,item in zip([color_list_expt[1], [1-(1-ci)*0.5 for ci in color_list_expt[1]]], legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)



## rect1_21: Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_21)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(mean_effect_list_H)), mean_effect_list_H, bar_width_compare3, alpha=bar_opacity, yerr=Mean_reg_err_bars_H_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(mean_effect_list_H_preSK[0]+mean_effect_list_H_preSK[1]), 0.5*(mean_effect_list_H_preSK[0]+mean_effect_list_H_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.], [25.2], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [24., 24.], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
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

## rect1_22: Variance Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_22)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(var_effect_list_H)), var_effect_list_H, bar_width_compare3, alpha=bar_opacity, yerr=Var_reg_err_bars_H_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(var_effect_list_H_preSK[0]+var_effect_list_H_preSK[1]), 0.5*(var_effect_list_H_preSK[0]+var_effect_list_H_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
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

## rect1_23: Variance Beta/ Mean Beta, Model and perturbations
ax   = fig_temp.add_axes(rect1_23)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(var_mean_ratio_list_H)), var_mean_ratio_list_H, bar_width_compare3, alpha=bar_opacity, yerr=Var_mean_ratio_err_Reg_bars_H_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,2.*bar_width_compare3], [0.5*(var_mean_ratio_list_H_preSK[0]+var_mean_ratio_list_H_preSK[1]), 0.5*(var_mean_ratio_list_H_preSK[0]+var_mean_ratio_list_H_preSK[1])], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1.], [0.49], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [0.46,0.46], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
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



## rect1_24: Psychophysical Kernel, Monkey A
ax   = fig_temp.add_axes(rect1_24)
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
ax.set_xlim([1,6])
ax.set_xticks([1,6])
ax.set_yticks([0., 5.])
ax.text(0.15, 5.5, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
minorLocator = MultipleLocator(1.)
ax.yaxis.set_minor_locator(minorLocator)
minorLocator = MultipleLocator(1.)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(direction='out', pad=1.5)
ax.tick_params(which='minor',direction='out')
ax.spines['left'].set_position(('outward',5))
ax.spines['bottom'].set_position(('outward',5))
ax.plot(i_PK_list_6, PK_H_saline, label='Saline', color=color_list_expt[0], linestyle='-', zorder=0, clip_on=False)#, linestyle=linestyle_list[i_var_a])
ax.plot(i_PK_list_6, PK_H_ketamine, label='Ketamine', color=color_list_expt[1], linestyle='-', zorder=0, clip_on=False)#, linestyle=linestyle_list[i_var_a])
legend = ax.legend(loc=(-0.05,0.33), fontsize=fontsize_legend-1, frameon=False, ncol=1, markerscale=0., columnspacing=1., handletextpad=0.)
for color,text,item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    item.set_visible(False)
##################### Plotting monkey H data

## rect1_11: E/I perturbation schematics

## rect1_12: Psychometric function. Control model.
ax_0   = fig_temp.add_axes(model_rect1_12_0)
ax   = fig_temp.add_axes(model_rect1_12)
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
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_model_control, x_list_psychometric, 0.0684), color=color_list[0], ls='-', clip_on=False, label='Higher SD Correct')#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_control, -x_list_psychometric, 0.0684), color=[1-(1-ci)*0.5 for ci in color_list[0]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_model_control, x0_psychometric, 0.0684), s=15., color=color_list[0], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_control, -x0_psychometric, 0.0684), s=15., color=[1-(1-ci)*0.5 for ci in color_list[0]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
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
y_shift_spines = -0.06
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
ax_0   = fig_temp.add_axes(model_rect1_13_0)
ax   = fig_temp.add_axes(model_rect1_13)
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
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_model_reduced_gEE, x_list_psychometric, 0.0684), color=color_list[1], ls='-', clip_on=False, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_reduced_gEE, -x_list_psychometric, 0.0684), color=[1-(1-ci)*0.5 for ci in color_list[1]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_model_reduced_gEE, x0_psychometric, 0.0684), s=15., color=color_list[1], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_reduced_gEE, -x0_psychometric, 0.0684), s=15., color=[1-(1-ci)*0.5 for ci in color_list[1]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
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
y_shift_spines = -0.06
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
ax_0   = fig_temp.add_axes(model_rect1_14_0)
ax   = fig_temp.add_axes(model_rect1_14)
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
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_model_reduced_gEI, x_list_psychometric, 0.0684), color=color_list[2], ls='-', clip_on=False, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_reduced_gEI, -x_list_psychometric, 0.0684), color=[1-(1-ci)*0.5 for ci in color_list[2]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_model_reduced_gEI, x0_psychometric, 0.0684), s=15., color=color_list[2], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_reduced_gEI, -x0_psychometric, 0.0684), s=15., color=[1-(1-ci)*0.5 for ci in color_list[2]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
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
y_shift_spines = -0.06
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
ax_0   = fig_temp.add_axes(model_rect1_15_0)
ax   = fig_temp.add_axes(model_rect1_15)
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
ax.plot(100.*x_list_psychometric, Psychometric_function_D_lapse(psychometric_params_model_upstream_deficit, x_list_psychometric, 0.0684), color=color_list[3], ls='-', clip_on=False, label='Higher SD Correct' )#, linestyle=linestyle_list[i_var_a])
ax.plot(100.*x_list_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_upstream_deficit, -x_list_psychometric, 0.0684), color=[1-(1-ci)*0.5 for ci in color_list[3]], ls='-', clip_on=False, label='Lower SD Correct')#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, Psychometric_function_D_lapse(psychometric_params_model_upstream_deficit, x0_psychometric, 0.0684), s=15., color=color_list[3], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])
ax_0.scatter(100.*x0_psychometric, 1.-Psychometric_function_D_lapse(psychometric_params_model_upstream_deficit, -x0_psychometric, 0.0684), s=15., color=[1-(1-ci)*0.5 for ci in color_list[3]], marker='_', clip_on=False, linewidth=1.305)#, linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
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
y_shift_spines = -0.06
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
ax   = fig_temp.add_axes(model_rect1_21)
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
ax   = fig_temp.add_axes(model_rect1_22)
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
ax   = fig_temp.add_axes(model_rect1_23)
fig_funs.remove_topright_spines(ax)
ax.bar(np.arange(len(Reg_ratio_models)), Reg_ratio_models, bar_width_compare3, alpha=bar_opacity, yerr=Reg_err_ratio_models, ecolor='k', color=color_list[0:3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.bar(3, Reg_ratio_models[3], bar_width_compare3, alpha=bar_opacity, yerr=Reg_err_ratio_models[3], ecolor='k', color=color_list[3], edgecolor='k', clip_on=False, align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
ax.plot([0,4.*bar_width_compare3], [Reg_ratio_models[0], Reg_ratio_models[0]], ls='--', color='k', clip_on=False, lw=0.8) # Pre saline/ketamine values
ax.scatter([1., 1.5], [0.508, 0.546], s=16., color='k', marker=(5,2), clip_on=False, zorder=10)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,1.5], [0.49,0.49], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.plot([0.5,2.5], [0.528,0.528], ls='-', lw=1., color='k', clip_on=False, zorder=9)#, linestyle=linestyle_list[i_var_a])
ax.set_xlim([0,len(Reg_ratio_models)-1+bar_width_compare3])
ax.set_ylim([0.,0.54])
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
ax   = fig_temp.add_axes(model_rect1_24)
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
tmp2 = ax.scatter( i_PK_list, PK_paired_model_reduced_gEE     , color=color_list[1], marker='^', zorder=3, clip_on=False, facecolors=color_list[1], edgecolors='k', linewidths=0.6, s=4., label=label_list_expt[1])#, linestyle=linestyle_list[i_var_a])
tmp3 = ax.scatter(i_PK_list, PK_paired_model_reduced_gEI      , color=color_list[2], marker='s', zorder=2, clip_on=False, facecolors=color_list[2], edgecolors='k', linewidths=0.6, s=3.5, label=label_list_expt[2])#, linestyle=linestyle_list[i_var_a])
tmp4 = ax.scatter( i_PK_list, PK_paired_model_upstream_deficit, color=color_list[3], marker='x', zorder=1, clip_on=False, facecolors=color_list[3], edgecolors=color_list[3], linewidths=0.6, s=5., label=label_list[3])#, linestyle=linestyle_list[i_var_a])
legend = ax.legend(loc=(0.3,0.55), fontsize=fontsize_legend-1.5, frameon=False, ncol=1, columnspacing=1., handletextpad=0., scatterpoints=1)
for color,text,item in zip(color_list, legend.get_texts(), legend.legendHandles):
    text.set_color(color)
    # item.set_visible(False)

fig_temp.savefig(path_cwd+'Figure8S9.pdf')    #Finally save fig

########################################################################################################################


