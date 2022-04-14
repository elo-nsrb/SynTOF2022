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
#from statannot import add_stat_annotation


from visualization_func import *
import umap.umap_ as umap
#import openTSNE as tsne
from sklearn.manifold import TSNE
import sys
import params_visu
sys.path.append('../')
from list_feat_params import map_feat_spec, list_feat_surface, list_feat_activity

PATH_AEC = "/home/eloiseb/stanford_drive/experiences/aec_joe/" 
DATASET_PATH = "/home/eloiseb/stanford_drive/data/csv/PreSyn_Single_evt_CG2Eloi_Prem_35Ch_MultiSpecies_13Apr2021_spill_not_applied_scaled_events_train_LowNo.csv"

def load_data(params, region, list_feat_all):
    brain_region = params["brain_region"]

    df = pd.read_csv(DATASET_PATH)
    save_dir = PATH_AEC + params["model_dir"]
    df_hidden = pd.read_csv(save_dir + "/hidden_ae_" + region + "_no_prp_sess_1.csv")
    main_path = save_dir + 'mcResultsDWH' + '.csv' 
    df = df[df.Brain_area == brain_region]
    df_aec = pd.read_csv(main_path)
    df_aec.columns
    df['aec'] = df_aec['mc'].values

    savename = "hidden_aec_0"
    col_hidden = ["hidden_0_%s"%str(k) for k in range(15)]
    df[col_hidden] = df_hidden[col_hidden].values

    list_no_mouse = params["list_no_mouse"] 
    list_no_mk = params["list_no_mo"]
    list_no_hu = params["list_no_hu"]
    df.drop(df[(df["Specie"]=="hu") & (df["aec"].isin(list_no_hu))][list_feat_all].index,axis=0,inplace=True)
    df.drop(df[(df["Specie"]=="mk") & (df["aec"].isin(list_no_mk))][list_feat_all].index,axis=0,inplace=True)
    df.drop(df[(df["Specie"]=="mouse") & (df["aec"].isin(list_no_mouse))][list_feat_all].index, axis=0, inplace=True)
    order_t = params["order"] 
    mapping=params["mapping"] 
    order = [mapping[it] for it in order_t]
    df.loc[:,"aec"].replace(mapping, inplace=True)
    return df, save_dir

