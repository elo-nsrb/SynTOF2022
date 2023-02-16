import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import scanpy as sc
#import squidpy as sq
import pandas as pd
import skimage.io as io
import seaborn as sns
import skimage


plt_dir = "/home/eloiseb/stanford_drive/experiences/aec_joe/"
list_br = ['BA9-ctx', 'DLCau-str', 'Hipp']
for br in list_br:
    df = pd.read_csv(plt_dir + "pseudo_bulk_positive_events_%s_mean.csv"%br, index_col=0).T
    df_std = pd.read_csv(plt_dir + "pseudo_bulk_positive_events_%s_std.csv"%br, index_col=0).T
    
    title = 'Species mean positive events in %s (%%)'%br
    savename = "species_positive_events_%s"%br

    list_cond = df.columns
    print(list_cond)
    df.reset_index(inplace=True)
    df_std.reset_index(inplace=True)
    df.columns = ["condition"] + list_cond.tolist()
    df_std.columns = ["condition"] + list_cond.tolist()

    df_freq = pd.melt(df, value_vars=list_cond, id_vars="condition")
    df_freq_std = pd.melt(df_std, value_vars=list_cond, id_vars="condition")
    df_freq["log(value +1)"] = np.log(10*df_freq["value"] + 10)
    key_to_plot = "value" #"value"


    #list_order = [ 'CD56','CD47','Synaptophysin', 'TotalTau', 'MAP2', 'Calbindin', 'Calretinin','PV', 'SERT',  'TH','PanGAD',  'VGAT', 'VGLUT1', 'VGLUT2','PSD95', 'PHF1Tau', 'Abeta','PanAPOE','ApoE', 'HH3', 'MAG',]
    list_order1 = ["AS" , "Calreticulin", "BIN1", "ApoE", "DJ1" , "GAMT", "SLC6A8"]
    list_order2 = ["CD56", "SNAP25", "VGLUT", "Tau","APP", "GAD65",
            "Synaptobrevin2", "VMAT2", "LRRK2",
            "Parkin","CD47", "GATM","TMEM230","GBA1"]
    list_order3 = ["DAT"]
    list_order = list_order1 + list_order2 + list_order3 #+ list_order[0]
    colors_label1 = ["#F7A072"] * (len(list_order1))
    colors_label2 = ["#EDDEA4"] * (len(list_order2))
    colors_label3 = ["#B5E2FA"] * (len(list_order3))
    colors_label = colors_label1 + colors_label2 + colors_label3 + colors_label1[0:1]
    textsubaxescolors = "#5A5A5A"
    S1 = df_freq[key_to_plot].min()#quantile(0.2) 
    S2 = df_freq[key_to_plot].mean() 
    S3 = df_freq[key_to_plot].max() 
    if "log" in key_to_plot:
        LS1 = str(np.round(np.exp(S1)/10. - 1, 2))
        LS2 = str(np.round(np.exp(S2)/10. - 1, 2))
        LS3 = str(np.round(np.exp(S3)/10. -1, 2))
    else:
        LS1 = str(np.round(S1*100,0).astype(int)) + "%"
        LS2 = str(np.round(S2*100,0).astype(int)) +"%"
        LS3 = str(np.round(S3*100,0).astype(int)) + "%"
    conds = df_freq.condition.unique().tolist()
    cond0 = conds[0]#"Technical Control 1_cohort_1"
    categories = df_freq[df_freq.condition == cond0 ]["variable"].values.tolist()

    idx = np.asarray([categories.index(it) for it in list_order]).astype(int)
    categories = np.asarray(categories)[idx]
    list_to_plots = []
    list_to_plots_std = []
    for cd in conds:
        pt = df_freq[df_freq.condition == cd][key_to_plot].values.tolist()
        pt = np.asarray(pt)[idx]
        pt = [*pt, pt[0]]
        list_to_plots.append(pt)
        pt = df_freq_std[df_freq_std.condition == cd][key_to_plot].values.tolist()
        pt = np.asarray(pt)[idx]
        pt = [*pt, pt[0]]
        list_to_plots_std.append(pt)

    categories = [*categories, categories[0]]

    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(pt))

    plt.figure(figsize=(10, 10))
    plt.subplot(polar=True)
    color_species={"Hu":"#BFD7EA", "Mi":"#FF5A5F", "Ma":"#087E8B"}
    for jj, (it, name) in enumerate(list(zip(list_to_plots,conds))):
        #plt.plot(label_loc, it, label=name, color=color_species[name])
        plt.errorbar(label_loc, it, list_to_plots_std[jj],
                        #linestyle='None', 
                        label=name, 
                        color=color_species[name])
                        #uplims=True, lolims=True,)
        plt.fill(label_loc, it, color=color_species[name], alpha=0.2)

    # Remove lines for radial axis (y)
    plt.yticks([])
    #plt.grid(False)
    #plt.axis.XAxis.grid(False)

    ## Remove spines
    #plt.spines["start"].set_color("none")
    #plt.spines["polar"].set_color("none")

    #
    # Used for the equivalent of horizontal lines in cartesian coordinates plots 
    # The last one is also used to add a fill which acts a background color.
    H0 = np.ones(len(pt))*S1 
    H1 = np.ones(len(pt))*S2
    H2 = np.ones(len(pt))*S3

    # Add custom lines for radial axis (y) at 0, 0.5 and 1.
    plt.plot(label_loc,H0, ls=(0, (6, 6)), c=textsubaxescolors)
    plt.plot(label_loc, H1, ls=(0, (6, 6)), c=textsubaxescolors)
    plt.plot(label_loc, H2, ls=(0, (6, 6)), c=textsubaxescolors)

    # Add levels -----------------------------------------------------
    # These labels indicate the values of the radial axis
    PAD = 0.06
    alp = 0.8
    plt.text(-0.2, S1 + PAD, LS1, size=16, c=textsubaxescolors, alpha=alp)#, fontname="Roboto")
    plt.text(-0.2, S2 + PAD, LS2, size=16,c=textsubaxescolors, alpha=alp)#, fontname="Roboto")
    plt.text(-0.2, S3 + PAD, LS3, size=16, c=textsubaxescolors, alpha=alp)#, fontname="Roboto")


    plt.title(title, size=20, y=1.05)

    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories, size=16)
    for ticklabel, tickcolor in zip(labels, colors_label):
        ticklabel.set_color(tickcolor)
    #plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(plt_dir+ savename + br + "error_bar.png", bbox_inches="tight")
    plt.show()
