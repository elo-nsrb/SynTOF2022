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
SAVEPATH = "/home/eloiseb/stanford_drive/experiences/ae_joe/" 
DATASET_PATH = "/home/eloiseb/stanford_drive/data/csv/PreSyn_Single_evt_CG2Eloi_Prem_35Ch_MultiSpecies_13Apr2021_spill_not_applied_scaled_events_train_LowNo.csv"

sc.settings.autosave = True
sc.settings.autoshow = False
sc.settings.figdir = SAVEPATH

color_species={"hu":"#84BC9C", "mouse":"#F46197", "mk":"#007991"}
color_brain_area={"BA9-ctx":"#104D3F", "DLCau-str":"#6D9DB3", "Hipp":"#800E1F"}
color_map = {** color_species, ** color_brain_area}
font = {'size'   : 22}
matplotlib.rc('font', **font)
sns.set(font_scale=1.6)
sns.set_style("white")
df = pd.read_csv(DATASET_PATH)
list_feat_all = list_feat_surface 
if "PrP" in list_feat_all:
        list_feat_all.remove("PrP")
        
params = params_visu.params_ctx
region = "ctx"
df, save_dir = load_data(params, region, list_feat_all) 
savename = "test"

scaler = StandardScaler()

#df["aec"].replace(mapping, inplace=True)
df["Specie"].replace({"hu":"Hu", "mk":"Mo", "mouse":"Mu"}, inplace=True)
df["Sample_num"] = df["Sample_num"].astype(str)
cd = df[list_feat_all + ["Specie", "aec","Sample_num"]].groupby(["Specie","aec","Sample_num"]).mean()
df_mE_ = cd.T.corr()
df_mE_.columns=df_mE_.columns.map('|'.join).str.strip('|').str.strip('|')
df_mE_.index = df_mE_.index.map('|'.join).str.strip('|').str.strip('|')
links = df_mE_.stack().reset_index()
links[["from_Specie", "from_cluster","from_sample"]] = links["level_0"].str.split("|", expand=True)
links[["to_Specie", "to_cluster","to_sample"]] = links["level_1"].str.split("|", expand=True)
tmp = links[(links.to_cluster == links.from_cluster)]# & (links.to_Specie == links.from_Specie)]
tmp[["from_Specie", "to_Specie"]] = tmp[["from_Specie", "to_Specie"]].replace({"Hu":"primate","Mo":"primate"})
tmp["group"] = tmp["from_Specie"] + "-" + tmp["to_Specie"]
tmp.group.unique()
color_map={"primate-primate":"#246B6B", "primate-Mu":"#F7F740", "Mu-Mu":"#F46197"}
tmp[2] = (tmp[0] - tmp[0].min()) / (tmp[0].max() - tmp[0].min())
tmp_2 = tmp[tmp.group.isin(["primate-primate","primate-Mu", "Mu-Mu"])]
ax = sns.kdeplot(data=tmp_2,x=2,  hue="group" , fill=True, common_norm=False,alpha=.8, linewidth=0,legend=False, palette=color_map)
ax.set_xlabel("")
ax.set_ylabel("")
plt.savefig(save_dir + savename + "_inter-specie_correlation.svg")
plt.show()




df["Specie_2"] = df["Specie"].copy()
df["Specie_2"].replace({ "Hu":"Primate", "Mo":"Primate", "mouse":"Mu"}, inplace=True)
print(df["Specie_2"])
mean_df_2 = df.groupby(["Specie_2", "Sample_num"]).mean().reset_index()

all_pp = []
pairs_p = [("Primate", "Mu")]
#list_m = ["ApoE",  "Calreticulin", "CD47","GAD65", "GAMT","SLC6A8", "Synaptobrevin2",  "VGLUT"]#, "VMAT2"]
list_m =list_feat_all
save_df_primate = pd.DataFrame(columns=["marker", "mean_primate", "mean_mouse", "ratio_mean", "pvalue"])

x='Specie_2'
for i, mark in enumerate(list_m):
    y=mark
    pvaluesp = []
    for pair in pairs_p:
        data1 = mean_df_2.groupby(x)[y].get_group(pair[0])
        
        data2 = mean_df_2.groupby(x)[y].get_group(pair[1])
       
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        stat, p = stats.ttest_ind(data1, data2)
        print("Mann statistical test for equal variances on pair:",
                  pair, "stat={:.2e} p-value={:.2e}".format(stat, p))
        save_df_primate.loc[mark] = [mark, mean1, mean2, np.log2(mean2/mean1), p]

        pvaluesp.append(p)
        print("pvalues:", pvaluesp)
        all_pp.append(p)
        
mean_df = df.groupby(["Specie", "Sample_num"]).mean().reset_index()
all_ppp = []
pairs_ppp = [("Hu", "Mo")]
list_m =list_feat_all
save_df_hu_ppp = pd.DataFrame(columns=["marker", "mean_primate", "mean_mouse", "ratio_mean", "pvalue"])

x='Specie'
for i, mark in enumerate(list_m):
    y=mark
    pvaluesppp = []
    for pair in pairs_ppp:
        data1 = mean_df.groupby(x)[y].get_group(pair[0])
      
        data2 = mean_df.groupby(x)[y].get_group(pair[1])
    
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        stat, p = stats.ttest_ind(data1, data2)
        print("Mann statistical test for equal variances on pair:",
                  pair, "stat={:.2e} p-value={:.2e}".format(stat, p))
        save_df_hu_ppp.loc[mark] = [mark, mean1, mean2, np.log2(mean2/mean1), p]

        pvaluesppp.append(p)
        print("pvalues:", pvaluesppp)
        all_ppp.append(p)



import statsmodels.stats.multitest as ssm
_,corrected_pp,_,_ = ssm.multipletests(all_pp,  method='fdr_bh')
corrected_pp
save_df_primate["corrected_p"] = corrected_pp
save_df_primate["qvalue"] =-np.log(corrected_pp)

_,corrected_ppp,_,_ = ssm.multipletests(all_ppp,  method='fdr_bh')
corrected_pp
save_df_hu_ppp["corrected_p"] = corrected_ppp
save_df_hu_ppp["qvalue"] =-np.log(corrected_ppp)

save_df_hu_ppp["qvalue_with_mouse"]=save_df_primate["qvalue"].values
save_df_hu_ppp["ratio_mean_mouse"]= save_df_primate["ratio_mean"].values
save_df_hu_ppp["abs_ratio_mean_mouse"]= np.abs(save_df_primate["ratio_mean"].values)

lst_tmp = (save_df_hu_ppp["ratio_mean_mouse"]>0).values.astype(int)
list_color = ["#F46197" if it else "#246B6B" for it in lst_tmp]
print(list_color)
dico_color = {mkk:ccl for mkk,ccl in zip(save_df_hu_ppp.index.tolist(), list_color)}
print(dico_color)

if True:
    sns.set(font_scale=1)
    sns.set_style("white")
    fig, ax =plt.subplots(figsize=(8,3))
    x="qvalue"
    xlim = -np.log(0.05)
    
    ylim = 0.5#std

    jj=save_df_hu_ppp
    jj_s = jj[((jj["qvalue_with_mouse"]>xlim) & (jj[x] <xlim)) & (jj["abs_ratio_mean_mouse"]>0.5 ) ]
    g =sns.scatterplot(ax=ax, data=jj_s, y="qvalue_with_mouse", x=x, hue="marker", size="abs_ratio_mean_mouse", palette=dico_color)
    # add annotations one by one with a loop
    for line in range(0,jj_s.shape[0]):
         plt.text(jj_s[x][line]+0.2, jj_s["qvalue_with_mouse"][line], jj_s["marker"][line], horizontalalignment='left', size='12', color='black')

    jj_ns = jj[~((jj["qvalue_with_mouse"]>xlim) & (jj[x] <xlim)& (jj["abs_ratio_mean_mouse"]>0.5))]
    g =sns.scatterplot(ax=ax, data=jj_ns, y="qvalue_with_mouse", x=x, color="grey", s=12)
    
    ax.hlines(y=xlim,xmin=0,  xmax =13, color= 'k', linestyle='--', linewidth=0.8)
    ax.vlines(x=xlim,ymin=0.0,  ymax =26, color= 'k', linestyle='--', linewidth=0.8)
    #ax.vlines(x=ylim,ymin=0.0,  ymax =17.5, color= 'k', linestyle='--', linewidth=0.8)
    h,l = g.axes.get_legend_handles_labels()
    g.axes.legend_.remove()
    ax.set_xlabel("Qvalue Mo/Hu")
    ax.set_ylabel("Qvalue Mu/primate")
    g.legend(h,l, bbox_to_anchor=(1.48, 1),
    borderaxespad=0, ncol=1)
    plt.savefig(save_dir + savename + "_log_log_mouse_primate_vs_primate_primate.svg",bbox_inches='tight')
    plt.show()
