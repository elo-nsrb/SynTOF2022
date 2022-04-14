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




#### Bar Plot comparing marker reactivity
list_feat = list_feat_surface #+ list_feat_activity
if "PrP" in list_feat:
    list_feat.remove("PrP")
target="Specie"
bas = df.Brain_area.unique()
for ba in bas:
    df_ = df[df.Brain_area == ba]
    for i, it in enumerate(map_feat_spec.keys()): 
        if it != "Mo-only":
            new_list = [mm for mm in map_feat_spec[it] if mm in list_feat_surface]
            df_hu = df_[df_.Specie == 'hu']
            pd_hu_marker_type = df_hu[new_list].mean() #map_feat_spec[it]].mean()
            if ba in ["BA9-ctx", "DLCau-str"]:
                pd_mo_marker_type = df_[df_.Specie == "mk"][new_list].mean()
                fvalue, pvalue = stats.f_oneway(pd_hu_marker_type, pd_mo_marker_type)
                print("hu, Mo " +it + "fvalue, pvalue : " + str(fvalue) + ' ' + str(pvalue))
            if ba in ["BA9-ctx", "Hipp"]:
                df_mu = df_[df_.Specie == 'mouse']
                pd_mu_marker_type = df_mu[new_list].mean() #map_feat_spec[it]].mean()
                fvalue, pvalue = stats.f_oneway(pd_hu_marker_type, pd_mu_marker_type)
                print("hu, Mu " +it + "fvalue, pvalue : " + str(fvalue) + ' ' + str(pvalue))
            if ba in ["BA9-ctx"]:
                fvalue, pvalue = stats.f_oneway(pd_mo_marker_type, pd_mu_marker_type)
                print("Mo, Mu " +it + "fvalue, pvalue : " + str(fvalue) + ' ' + str(pvalue))
                fvalue, pvalue = stats.f_oneway(pd_hu_marker_type, pd_mo_marker_type, pd_mu_marker_type)
                print("Mo, Mu, hu " +it + "fvalue, pvalue : " + str(fvalue) + ' ' + str(pvalue))
                pd_combine = pd.concat([pd_hu_marker_type,pd_mo_marker_type, pd_mu_marker_type], keys=['human',"monkey", 'mouse'],
                                axis=0)
            elif ba in ["Hipp"]:
                pd_combine = pd.concat([pd_hu_marker_type, pd_mu_marker_type], keys=['human', 'mouse'],
                                axis=0)
            elif ba in ["DLCau-str"]:
                pd_combine = pd.concat([pd_hu_marker_type,pd_mo_marker_type], keys=['human',"monkey"],
                                axis=0)
            color_map['human'] = color_map['hu']
            color_map['monkey'] = color_map['mk']

            
            kk = pd_combine.reset_index(level=0)
            kk.rename(columns={'level_0': 'Specie', 0:'mean'}, inplace=True)
            fig, ax = plt.subplots(figsize=(12,8), sharex=True)
            ax = sns.barplot(x=kk.index, y='mean', hue='Specie',
                    data = kk, palette=color_map)
            pd_hu_m = pd_hu_marker_type.mean()
            pd_mo_m = pd_mo_marker_type.mean()
            pd_mu_m = pd_mu_marker_type.mean()
            ll = pd_hu_marker_type.shape[0]
            ax.hlines(y=pd_hu_m, xmin=-2.0,  xmax =ll + 0.2 , color='k', linewidth=1)
            font = {'size':'15', 'color':'r'}
            level_t = round(pd_hu_m, 3)
            tt=0
            if ba=="DLCau-str":
                    if it=='Hu-only':
                        tt = -0.017
                    elif it == 'Hu-Mo':
                        tt = -0.05# -6.5
                    else:
                        tt = 0
            if ba=="Hipp":
                    if it=='Hu-only':
                        tt = 0
                    elif it == 'Hu-Mo':
                        tt = -0.05# -6.5
                    else:
                        tt = 0
            if ba=="BA9-ctx":
                    if it=='Hu-only':
                        tt = 0
                    elif it == 'Hu-Mo':
                        tt= 0.04
            trans = transforms.blended_transform_factory(
                        ax.get_yticklabels()[0].get_transform(), ax.transData)
            ax.text(0,pd_hu_m + tt, "{:.3f}".format(pd_hu_m),
                        color=color_map['hu'],
                            transform=trans, 
                            ha="right", va="center")
            #ax.text(-0.1,level_t, 'Mean Human:' + str(level_t), fontdict=font ) 
            if ba in ["BA9-ctx", "Hipp"]:
                ttu = 0
                if ba=="Hipp":
                    if it == 'Hu-Mo':
                        ttu = 0.05
                    if it =='Hu-only':
                        ttu = -0.02
                level_t = round(pd_mu_m, 3)
                ax.hlines(y=pd_mu_m,xmin=-2.0,  xmax =ll+0.2, color= 'k', linewidth=1)
                trans = transforms.blended_transform_factory(
                            ax.get_yticklabels()[0].get_transform(), ax.transData)
                ax.text(0,pd_mu_m +ttu, "{:.3f}".format(pd_mu_m), color=color_map['mouse'],
                                transform=trans, 
                                ha="right", va="center")
            if ba in ["BA9-ctx", "DLCau-str"]:
                if ba=="DLCau-str":
                    if it=='Hu-only':
                        ttm = 0.02
                    elif it == 'Hu-Mo':
                        ttm = +0.05# -6.5
                    else:
                        ttm = 0
                        #tt =0.02
                if ba=="BA9-ctx":
                    if it=='Hu-only':
                        ttm = -0.013
                    elif it=='Hu-Mo':
                        ttm = +0.00
                    else:
                        ttm=0
                ax.hlines(y=pd_mo_m,xmin=-2.0,  xmax =ll+0.2, color= 'k', linewidth=1)
                trans = transforms.blended_transform_factory(
                            ax.get_yticklabels()[0].get_transform(), ax.transData)
                ax.text(0,pd_mo_m +ttm, "{:.3f}".format(pd_mo_m), color=color_map['monkey'],
                                transform=trans, 
                                ha="right", va="center")
            #ax.text(-0.1, level_t, 'Mean Mouse:'  + str(round(pd_mu_m, 3)), fontdict=font) 
            ax.set_xticklabels(ax.get_xticklabels(), rotation=75, fontsize=22)
            print(ax.get_xticks())
            ax.set_xlim(-1,14)
            ax.set_xlabel("Mean marker value")
            ax.tick_params(labelsize=22)
            ax.xaxis.label.set_size(22)
            ax.yaxis.label.set_size(22)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.title('Marker reactivity ' + it + " (Pvalue: %.2f)"%pvalue) 
            plt.tight_layout()
            plt.savefig(SAVEPATH + ba +'marker_reactivity_bar_plot_' + it + '.svg')
            plt.show()
            plt.close('all')
            plt.clf()
 
        
