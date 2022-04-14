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
list_feat = list_feat_surface #+ list_feat_activity
if "PrP" in list_feat:
    list_feat.remove("PrP")

#define function to calculate cv
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 
mean = df.groupby(["Brain_area", "Specie"])[list_feat].apply(np.mean)
std = df.groupby(["Brain_area", "Specie"])[list_feat].apply(np.std)

cv = mean.div(std)
cv

cv.reset_index(1,inplace=True)

cv_hu = cv[cv.Specie == "hu"]
cv_mk = cv[cv.Specie == "mk"]
cv_mouse = cv[cv.Specie == "mouse"]

cv_hu.reset_index(inplace=True)


cv_hu_melt = pd.melt(cv_hu,  id_vars=['Brain_area','Specie'], 
                value_vars=list_feat, var_name='markers', value_name="CV")

mean_mix = df.groupby(["Brain_area"])[list_feat].apply(np.mean)
std_mix = df.groupby(["Brain_area"])[list_feat].apply(np.std)

cv_mix = mean_mix.div(std_mix)
cv_mix.reset_index(inplace=True)
cv_mix["Specie"] = "Multi-specie"

cv_mix_melt = pd.melt(cv_mix,  id_vars=['Brain_area','Specie'], value_vars=list_feat, var_name='markers', value_name="CV")
cv_melt_comb = pd.concat([cv_hu_melt,cv_mix_melt], axis=0)

for sp in cv_melt_comb.Specie.unique():
    for sp2 in cv_melt_comb.Specie.unique():
        if sp!=sp2:
            for ba in cv_melt_comb.Brain_area.unique():
                x = cv_hu[(cv_hu.Brain_area == ba)][list_feat].values.squeeze()
                print(x.shape)
                y = cv_mix[ (cv_mix.Brain_area == ba)][list_feat].values.squeeze()
                st, pval = stats.ttest_ind(x, y)
                print("Hu, Multispecie " +ba + "ttest ind fvalue, pvalue : " + str(st) + ' ' + str(pval))

                fvalue, pvalue = stats.f_oneway(x,y)
                print("Hu, Multispecie " +ba + "f_oneway fvalue, pvalue : " + str(fvalue) + ' ' + str(pvalue))

   
ctx_hu = cv_melt_comb[(cv_melt_comb.Brain_area =="BA9-ctx")&(cv_melt_comb.Specie=="hu")]["CV"].values.tolist()
ctx_multi = cv_melt_comb[(cv_melt_comb.Brain_area =="BA9-ctx")&(cv_melt_comb.Specie=="Multi-specie")]["CV"].values.tolist()
str_hu = cv_melt_comb[(cv_melt_comb.Brain_area =="DLCau-str")&(cv_melt_comb.Specie=="hu")]["CV"].values.tolist()
str_multi = cv_melt_comb[(cv_melt_comb.Brain_area =="DLCau-str")&(cv_melt_comb.Specie=="Multi-specie")]["CV"].values.tolist()
hipp_hu = cv_melt_comb[(cv_melt_comb.Brain_area =="Hipp")&(cv_melt_comb.Specie=="hu")]["CV"].values.tolist()
hipp_multi = cv_melt_comb[(cv_melt_comb.Brain_area =="Hipp")&(cv_melt_comb.Specie=="Multi-specie")]["CV"].values.tolist()

  
n=cv_melt_comb.markers.nunique()
r = np.arange(n)
width = 0.25
  
fig, ax = plt.subplots(figsize=(17, 5))  
ax.bar(r-width, ctx_hu, color = '#e1c847',
        width = width, edgecolor = '#e1c847',
        label='CTX/Mono')
ax.bar(r-width, ctx_multi, color = '#e1c847',
        width = width, edgecolor = 'black',hatch="x",alpha=0.2,
        label='CTX/Multi')
ax.bar(r , str_hu, color = '#4A4A74',
        width = width, edgecolor = '#4A4A74',
        label='STR/Mono')
ax.bar(r , str_multi, color = '#4A4A74',
        width = width, edgecolor = 'black',hatch="x",alpha=0.2,
        label='STR/Multi')
ax.bar(r + width, hipp_hu, color = '#a7d0bb',
        width = width, edgecolor = '#a7d0bb',
        label='HIPP/Mono')
ax.bar(r + width, hipp_multi, color = '#a7d0bb',
        width = width, edgecolor = 'black',hatch="x",alpha=0.2,
        label='HIPP/Multi')
ax.set_xlabel("Brain Regions")
ax.set_ylabel("Coefficient of Variation (CV)")
#plt.title("Number of people voted in each year")
  
# plt.grid(linestyle='--')
plt.xticks(r + width/2,cv_melt_comb[(cv_melt_comb.Brain_area =="BA9-ctx")&(cv_melt_comb.Specie=="hu")]["markers"].values.tolist())
plt.xticks(rotation=75, fontsize=30)
plt.legend()
plt.tight_layout()
savepath = (SAVEPATH + 'barplot_cv_comparison_' +'.svg')
plt.savefig(savepath, )

plt.show()
