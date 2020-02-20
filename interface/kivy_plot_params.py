### Author: Devin Whitten
## interface for matplotlib figures in CCS_kivy

import kivy
import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
import matplotlib.pyplot as plt
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
matplotlib.style.use("dark_background")



color_frame = {"GI":'blue', 'GII':"green", 'GIII':"orange"}

def build_spec_axis():
    fig = plt.figure()
    axTOP = fig.add_axes([0.1, 0.6, 0.8, 0.3])
    axBOT = fig.add_axes([0.1, 0.1, 0.8, 0.3])
    axBAR = fig.add_axes([0.1, 0.5, 0.8, 0.03])

    fig.suptitle("Spectrum", fontname='Times New Roman', fontsize=20)

    spec_line, = axTOP.plot([],[], color='white', linewidth=0.50)
    cont_line, = axTOP.plot([], [])
            ## norm flux
    norm_line, = axBOT.plot([], [], color='white', linewidth=0.50)


    axBOT.axhline(1.0, linestyle='--', color='red')


    [plt.setp(axis.get_yticklabels(), visible=False) for axis in [axTOP, axBAR]]
    plt.setp(axBAR.get_xticklabels(), visible=False)
    [axTOP.set_xlabel(r"$\lambda$ [$\AA$]", fontname='Times New Roman') for axis in [axTOP, axBOT]]
    #### Plot params
    [ax.tick_params(direction="in", top=True, right=True) for ax in [axTOP, axBOT]]
    axBAR.tick_params(length=0)

    return fig, (axTOP, axBOT, axBAR), {"spec_line":spec_line, "cont_line":cont_line, "norm_line":norm_line}


def build_class_axis():
    fig = plt.figure()
    fig.suptitle(r"Archetype $\chi^2$", fontname="Times New Roman", fontsize=20)
    chi_lines = []
    #axes = [fig.add_subplot(3,2,i+1) for i in range(6)]

    a1 = fig.add_subplot(3,2,1)
    a2 = fig.add_subplot(3,2,2, sharey=a1)
    a3 = fig.add_subplot(3,2,3)
    a4 = fig.add_subplot(3,2,4, sharey=a3)
    a5 = fig.add_subplot(3,2,5)
    a6 = fig.add_subplot(3,2,6, sharey=a5)

    axes = [a1, a2, a3, a4, a5, a6]


    [ax.tick_params(direction="in", top=True, right=True) for ax in axes]
    [ax.text(0.5, 0.85, GROUP, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes) for GROUP, ax in zip(['GI', 'GI', 'GII', 'GII', 'GIII', 'GIII'], axes)]

    a1.set_title("Dwarf")
    a2.set_title("Giant")


    for cur, group in zip(axes, ['GI', 'GI', 'GII', 'GII', 'GIII', 'GIII']):
        cur, = cur.plot([], [], color=color_frame[group], linewidth=2.)

        chi_lines.append(cur)



    return fig, axes, chi_lines


def build_custom_axis():
    ## I'm gonna try and use this axis for everything else I need.
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
    line = ax.plot([], [])
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax, line
