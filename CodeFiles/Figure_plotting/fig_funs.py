import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def remove_topright_spines(ax):
    # hide the top and right spines
    [spin.set_visible(False) for spin in ax.spines['top'],ax.spines['right']]

    #hide the right and top tick marks
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    return None

def remove_all_but_left_spines(axs):
    if type(axs) is list:
        for ax in axs:
            # hide the top and right spines
            [spin.set_visible(False) for spin in ax.spines['top'],ax.spines['right'],ax.spines['bottom']]

            #hide the right and top tick marks
            ax.yaxis.tick_left()
            ax.xaxis.tick_bottom()
            ax.set_xticks([])
    else:
        # hide the top and right spines
        [spin.set_visible(False) for spin in axs.spines['top'],axs.spines['right'],axs.spines['bottom']]

        #hide the right and top tick marks
        axs.yaxis.tick_left()
        axs.xaxis.tick_bottom()
        axs.set_xticks([])
    return None

def remove_all_spines(axs):
    if type(axs) is list:
        for ax in axs:
            # hide the top and right spines
            [spin.set_visible(False) for spin in ax.spines['top'],ax.spines['right'],ax.spines['bottom'],ax.spines['left']]

            #hide the right and top tick marks
            ax.yaxis.tick_left()
            ax.xaxis.tick_bottom()
            ax.set_xticks([])
    else:
        # hide the top and right spines
        [spin.set_visible(False) for spin in axs.spines['top'],axs.spines['right'],axs.spines['bottom'],axs.spines['left']]

        #hide the right and top tick marks
        axs.yaxis.tick_left()
        axs.xaxis.tick_bottom()
        axs.set_xticks([])
    return None

def remove_spines(ax,sps):
    for spi in sps:
        [spin.set_visible(False) for spin in ax.spines[spi]]
    return None

def remove_all_spines2(ax):
    # hide the top and right spines
    [spin.set_visible(False) for spin in ax.spines['top'],ax.spines['right']]
    [spin.set_visible(False) for spin in ax.spines['bottom'],ax.spines['left']]

    #hide the right and top tick marks
    #ax.yaxis.tick_left()
    #ax.xaxis.tick_bottom()
    #ax.yaxis.tick_right()
    #ax.xaxis.tick_top()
    return None

def add_significance(ax, xs=(0,1), y=0, dy = 1,sigtext='*',lw=1,ts=3,text_disp=0.,fs=10):
    ax.plot([xs[0],xs[0]],[y-dy,y],lw=lw,color='k')
    ax.plot([xs[1],xs[1]],[y-dy,y],lw=lw,color='k')
    ax.plot([xs[0],xs[1]],[y,y],lw=lw,color='k')
    #ax.axhline(y=y,xmin=xs[0],xmax=xs[1],linewidth=lw,color='k',clip_on=False)
    #ax.axvline(x=xs[0],ymin=y-dy,ymax=y,linewidth=lw,color='k',clip_on=False)
    #ax.axvline(x=xs[1],ymin=y-dy,ymax=y,linewidth=lw,color='k',clip_on=False)
    ax.annotate(sigtext,(np.mean(xs),y+text_disp),textcoords='data',ha='center',fontsize=fs)
    return 0


def add_scalebar(ax, pt=(0,0), xlen=0, ylen=0, xtext='', ytext='',lw=3,ts=3,xtext_disp=0.,ytext_disp=0.,fs=10,c='k'):

    xbounds = ax.get_xbound()
    ybounds = ax.get_ybound()

    xpt0 = xbounds[0] + pt[0]*(xbounds[1]-xbounds[0])
    xpt1 = pt[0]
    xpt2 = pt[0]+xlen/(xbounds[1]-xbounds[0])

    ypt0 = ybounds[0] + pt[1]*(ybounds[1]-ybounds[0])
    ypt1 = pt[1]
    ypt2 = pt[1]+ylen/(ybounds[1]-ybounds[0])

    if xlen != 0:
        ax.axhline(y=ypt0,xmin=xpt1,xmax=xpt2,linewidth=lw,color=c,clip_on=False,solid_capstyle='butt')
    if ylen != 0:
        ax.axvline(x=xpt0,ymin=ypt1,ymax=ypt2,linewidth=lw,color=c,clip_on=False,solid_capstyle='butt')

    ax.text(xpt1+ytext_disp,(ypt1+ypt2)/2,ytext,va='center',transform=ax.transAxes,fontsize=fs,clip_on=False,rotation='vertical')
    ax.text((xpt1+xpt2)/2.,ypt1-xtext_disp,xtext,ha='center',transform=ax.transAxes,fontsize=fs,clip_on=False)


if __name__ == '__main__':

    x = np.arange(0,20,0.001)
    y = 4*np.sin(x)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    i1 = ax1.plot(x,y)
    pt = (0.2,0.5)
    xlen = 5
    ylen = 1.5
    add_scalebar(ax=ax1,pt= pt,xlen=-xlen,ylen=-ylen,lw=5)
    xbounds = ax1.get_xbound()
    print xbounds
    ybounds = ax1.get_ybound()
    print ybounds
    print pt[1]*(ybounds[1]-ybounds[0])
    ax1.axhline(y=pt[1]*(ybounds[1]-ybounds[0]),xmin=pt[0],xmax=pt[0]+xlen/(xbounds[1]-xbounds[0]))
    fig.savefig('fig1.png',bbox_inches='tight')
