import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as transforms
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from statannot import add_stat_annotation
import scanpy as sc
import anndata
import scipy.stats as stats

import umap
import sys
import matplotlib
sys.path.append('..')
from list_feat_params import map_feat_spec, list_feat_surface, list_feat_activity
from load_data import load_data
import params_visu

### PATHS to edit
SAVEPATH = "/home/eloiseb/stanford_drive/experiences/aec_joe/MARKER_20" 
sc.settings.autosave = True
sc.settings.autoshow = False
sc.settings.figdir = SAVEPATH

font = {'size'   : 22}
matplotlib.rc('font', **font)
sns.set(font_scale=1.6)
sns.set_style("white")

list_feat_all = list_feat_surface 
params = params_visu.params_hipp
region = "hipp"
df, save_dir = load_data(params, region, list_feat_all) 
savename = "to_save_"

mean_df = df.groupby(["Species", "Sample_num"]).mean().reset_index()
#print(mean_df)
all_pp = []
pairs_p = [("Hu", "Mu")]
#list_m = ["ApoE",  "Calreticulin", "CD47","GAD65", "GAMT","SLC6A8", "Synaptobrevin2",  "VGLUT"]#, "VMAT2"]
save_df_hu_mi = pd.DataFrame(columns=["marker", "mean_primate", "mean_mouse", "ratio_mean", "pvalue"])

x='Species'
for i, mark in enumerate(list_feat_all):
    y=mark
    pvalues = []
    for pair in pairs_p:
        data1 = mean_df.groupby(x)[y].get_group(pair[0])
        print(pair)
        data2 = mean_df.groupby(x)[y].get_group(pair[1])
        
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        print(mean1)
        print(mean2)
        stat, p = stats.mannwhitneyu(data1, data2)
        save_df_hu_mi.loc[mark] = [mark, mean1, mean2, np.log2(mean2/mean1), p]
        print("Mann statistical test for equal variances on pair:",
                  pair, "stat={:.2e} p-value={:.2e}".format(stat, p))
        pvalues.append(p)
        print("pvalues:", pvalues)
        all_pp.append(p)

import statsmodels.stats.multitest as ssm
_,corrected_pp,_,_ = ssm.multipletests(all_pp,  method='fdr_bh')
corrected_pp
save_df_hu_mi["corrected_p"] = corrected_pp
save_df_hu_mi["qvalue"] =-np.log(corrected_pp)

lst_tmp = (save_df_hu_mi["ratio_mean"]>0).values.astype(int)
list_color = ["#F46197" if it else "#246B6B" for it in lst_tmp]
print(list_color)
dico_color = {mkk:ccl for mkk,ccl in zip(save_df_hu_mi.index.tolist(), list_color)}
print(dico_color)

if True:
    fig, ax =plt.subplots()
    x="ratio_mean"
    xlim = -np.log(0.05)
    ylim = 0.5#std

    jj=save_df_hu_mi
    jj_s = jj[((jj["qvalue"]>xlim) & (jj[x] >ylim)) |((jj["qvalue"]>xlim) & (jj[x] <-ylim)) ]
    g =sns.scatterplot(ax=ax, data=jj_s, y="qvalue", x=x, hue="marker",  palette=dico_color)
    for line in range(0,jj_s.shape[0]):
         plt.text(jj_s[x][line]+0.2, jj_s["qvalue"][line], jj_s["marker"][line], horizontalalignment='left', size='12', color='black')
    #jj_s = jj[((jj["qvalue_with_mouse"]>xlim) & (jj[x] <xlim)) & (jj["ratio_mean_mouse"]<-0.5 ) ]
    #g =sns.scatterplot(ax=ax, data=jj_s, y="qvalue_with_mouse", x=x, hue="marker", size="abs_ratio_mean_mouse")

    #print(jj_s[jj_s.marker == "Tau"])
    jj_ns = jj[~(((jj["qvalue"]>xlim) & (jj[x] >ylim)) |((jj["qvalue"]>xlim) & (jj[x] <-ylim)))]
    g =sns.scatterplot(ax=ax, data=jj_ns, y="qvalue", x=x, color="grey", s=12)
    
    ax.axhline(y=xlim, color= 'k', linestyle='--', linewidth=0.8)
    ax.axvline(x=-ylim,color= 'k', linestyle='--', linewidth=0.8)
    ax.axvline(x=ylim, color= 'k', linestyle='--', linewidth=0.8)
    #ax.vlines(x=ylim,ymin=0.0,  ymax =17.5, color= 'k', linestyle='--', linewidth=0.8)
    h,l = g.axes.get_legend_handles_labels()
    g.axes.legend_.remove()
    ax.set_xlabel("Qvalue Mu/Hu")
    ax.set_ylabel("log2(mean Ratio)")
    g.legend(h,l, bbox_to_anchor=(1.18, 1),
    borderaxespad=0, ncol=1)
    plt.savefig(save_dir + savename + "pseudo_bulk_analysis_hipp.svg")
    plt.show()
