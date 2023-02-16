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
import umap
import sys
import matplotlib
sys.path.append('..')
from list_feat_params import map_feat_spec, list_feat_surface, list_feat_activity


### PATHS to edit
SAVEPATH = "/home/eloiseb/stanford_drive/experiences/aec_joe/MARKER_20/" 
DATASET_PATH = "/home/eloiseb/stanford_drive/data/SynTOF/csv/PreSyn_Single_evt_CG2Eloi_Prem_35Ch_MultiSpecies_13Apr2021_spill_not_applied_scaled_events_train_LowNo_map.csv"


sc.settings.autosave = True
sc.settings.autoshow = False
sc.settings.figdir = SAVEPATH

#color_species={"hu":"#84BC9C", "Mu":"#F46197", "NHP":"#007991"}
color_species={"Hu":"#87C38F", 
                        "Mu":"#DA2C38", "NHP":"#126D80"}
color_brain_area={"CTX":"#104D3F", "NSTR":"#6D9DB3", "HIPP":"#800E1F"}
color_map = {** color_species, ** color_brain_area}
font = {'size'   : 22}
matplotlib.rc('font', **font)
sns.set(font_scale=1.6)
sns.set_style("white")
df = pd.read_csv(DATASET_PATH)
df["Species"] = df["Species"].replace({"Ma":"NHP", "Mi":"Mu"})



#### Bar Plot comparing marker reactivity
list_feat = list_feat_surface #+ list_feat_activity
target="Species"
bas = df.Brain_area.unique()
for ba in bas:
    df_ = df[df.Brain_area == ba]
    for i, it in enumerate(map_feat_spec.keys()): 
        if it != "NHP-only":
            new_list = [mm for mm in map_feat_spec[it] if mm in list_feat_surface]
            if len(new_list)>0:
                df_hu = df_[df_.Species == 'Hu']
                pd_hu_marker_type = df_hu[new_list].mean() #map_feat_spec[it]].mean()
                if ba in ["CTX", "NSTR"]:
                    pd_mo_marker_type = df_[df_.Species == "NHP"][new_list].mean()
                    fvalue, pvalue = stats.f_oneway(pd_hu_marker_type, pd_mo_marker_type)
                    print("hu, NHP " +it + "fvalue, pvalue : " + str(fvalue) + ' ' + str(pvalue))
                if ba in ["CTX", "HIPP"]:
                    df_mu = df_[df_.Species == 'Mu']
                    pd_mu_marker_type = df_mu[new_list].mean() #map_feat_spec[it]].mean()
                    fvalue, pvalue = stats.f_oneway(pd_hu_marker_type, pd_mu_marker_type)
                    print("hu, Mu " +it + "fvalue, pvalue : " + str(fvalue) + ' ' + str(pvalue))
                if ba in ["CTX"]:
                    fvalue, pvalue = stats.f_oneway(pd_mo_marker_type, pd_mu_marker_type)
                    print("NHP, Mu " +it + "fvalue, pvalue : " + str(fvalue) + ' ' + str(pvalue))
                    fvalue, pvalue = stats.f_oneway(pd_hu_marker_type, pd_mo_marker_type, pd_mu_marker_type)
                    print("NHP, Mu, hu " +it + "fvalue, pvalue : " + str(fvalue) + ' ' + str(pvalue))
                    pd_combine = pd.concat([pd_hu_marker_type,pd_mo_marker_type, pd_mu_marker_type], keys=['Hu',"NHP", 'Mu'],
                                    axis=0)
                elif ba in ["HIPP"]:
                    pd_combine = pd.concat([pd_hu_marker_type, pd_mu_marker_type], keys=['Hu', 'Mu'],
                                    axis=0)
                elif ba in ["NSTR"]:
                    pd_combine = pd.concat([pd_hu_marker_type,pd_mo_marker_type], keys=['Hu',"NHP"],
                                    axis=0)
                color_map['Hu'] = color_map['Hu']
                color_map['NHP'] = color_map['NHP']

                
                kk = pd_combine.reset_index(level=0)
                kk.rename(columns={'level_0': 'Species', 0:'mean'}, inplace=True)
                fig, ax = plt.subplots(figsize=(17,10), sharex=True)
                ax = sns.barplot(x=kk.index, y='mean', hue='Species',
                        data = kk, palette=color_map)
                pd_hu_m = pd_hu_marker_type.mean()
                pd_mo_m = pd_mo_marker_type.mean()
                pd_mu_m = pd_mu_marker_type.mean()
                ll = pd_hu_marker_type.shape[0]
                ax.axhline(y=pd_hu_m, color='k', linewidth=1)
                ax.axhline(y=pd_mo_m, color='k', linewidth=1)
                ax.axhline(y=pd_mu_m, color='k', linewidth=1)
                font = {'size':'15', 'color':'r'}
                level_t = round(pd_hu_m, 3)
                tt=0
                if ba=="NSTR":
                        if it=='Hu-only':
                            tt = -0.017
                        elif it == 'Hu-NHP':
                            tt = -0.05# -6.5
                        else:
                            tt = 0
                if ba=="HIPP":
                        if it=='Hu-only':
                            tt = 0
                        elif it == 'Hu-NHP':
                            tt = -0.05# -6.5
                        else:
                            tt = 0
                if ba=="CTX":
                        if it=='Hu-only':
                            tt = 0
                        elif it == 'Hu-NHP':
                            tt= 0.04
                trans = transforms.blended_transform_factory(
                            ax.get_yticklabels()[0].get_transform(), ax.transData)
                ax.text(0,pd_hu_m, "{:.3f}".format(pd_hu_m),
                            color=color_map['Hu'],
                                transform=trans, 
                                ha="right", va="center")
                ax.text(0,pd_mo_m, "{:.3f}".format(pd_mo_m),
                            color=color_map['NHP'],
                                transform=trans, 
                                ha="right", va="center")
                ax.text(0,pd_mu_m, "{:.3f}".format(pd_mu_m),
                            color=color_map['Mu'],
                                transform=trans, 
                                ha="right", va="center")
            #ax.text(-0.1, level_t, 'Mean NHPuse:'  + str(round(pd_mu_m, 3)), fontdict=font) 
            ax.set_xticklabels(ax.get_xticklabels(), rotation=75, fontsize=22)
            print(ax.get_xticks())
            #ax.set_xlim(-1,14)
            ax.set_xlabel("Mean marker value")
            ax.tick_params(labelsize=22)
            ax.xaxis.label.set_size(22)
            ax.yaxis.label.set_size(22)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.title('Marker reactivity ' + it + " (Pvalue: %.2f)"%pvalue) 
            plt.tight_layout()
            plt.savefig(SAVEPATH + ba +'REVISIONmarker_reactivity_bar_plot_' + it + '.svg')
            plt.show()
            plt.close('all')
            plt.clf()
