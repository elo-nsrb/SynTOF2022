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
from load_data import load_data


def plotTSNEs(df, save_dir, params,
                savename,
                list_feat_all,
                list_feat_c=None):
    """
    list_feat_c list of markers for enrichent plot
    """
    n=15000
    df["Sample_num"] = df["Sample_num"].astype(str)
    obs = "Specie"
    sub_sample = df.groupby([obs, 
                        "Brain_area"]).sample(n,
                                replace=False,
                                random_state=1)
    col_hidden = ["hidden_0_%s"%str(k) for k in range(15)]

    ####Plot TSNEs
    model_tsne = TSNE(n_jobs=30, perplexity=30, early_exaggeration=30)#n_neighbors=50, min_dist=0.01, random_state=1)
    res= model_tsne.fit_transform(sub_sample[col_hidden].values)
    method_name="tsne"
    col_umap = ["tsne_%d"%d for d in range(res.shape[1])]
    sub_sample[col_umap] = res
    
    savename = "TEST_REDO"
    list_obs_value = [[sp] for sp in df["Specie"].unique()]

    plot_fancy_scatter_clustering(sub_sample,
                    dico_color = params["dico_color"],
                    save_dir=save_dir, savename="TSNE" + savename,
                    dico_offset_x=params["dico_offset_x"],
                    dico_offset_y=params["dico_offset_x"],
                    cluster_key="aec", method_name="tsne")
    mmmpp = {"hu":"Human", "mk":"Monkey","mouse":"Mouse"}

    color_map_species=params["color_map_species"] #{"hu":"#84BC9C", "mouse":"#F46197", "mk":"#007991"}

    plot_fancy_scatter(sub_sample, color_map_species,
                    save_dir, savename="TSNE" + savename,
                    dico_offset_x=params["dico_offset_x"],
                    dico_offset_y=params["dico_offset_x"],
                    mapping_label=mmmpp,
                    color_key="Specie", method_name="tsne")
    list_Colors = params["color_batch_effect"] #["#ffd8b1", "#e6194B", "#2EC785", "#ffe119", "#4363d8", "#911eb4"]
    mapping = {k:col for k,col in zip(range(6), list_Colors)}
    plot_fancy_scatter_bacth_effect(sub_sample, mapping,
                    save_dir, savename="TSNE" + savename,
                    dico_offset_x=params["dico_offset_x"],
                    dico_offset_y=params["dico_offset_x"],
                    color_key="Sample_num",
                    stratified_key="Specie",
                    mapping_label=mmmpp,
                    method_name="tsne")
    plot_scatter_marker_enrichment(sub_sample, list_feat_all,                 
                    save_dir, savename="TSNE_ALL" + savename,
                    nb_rows=5,
                    nb_cols=5,
                    figsize=(28,28),
                    method_name="tsne")

    if list_feat_c is not None:
        plot_scatter_marker_enrichment(sub_sample, 
                    list_feat_c,
                    save_dir, savename="TSNE" + savename,
                    method_name="tsne")



def plotVolcanoes(df, save_dir, params, savename, list_feat_all):
    sig_mm = create_signature_matrix(df, list_feat_all,
                                    cluster_method = 'aec')
    (df_mouse_sort, 
     df_mk_sort,
     df_p_value) = ttest_per_cluster_per_marker(sig_mm, list_feat_all)
    x="log2_mean_ratio"

    std_med_ratio = 0.5
    volcano_plot(df_mk_sort,
            save_dir, 
            list_feat_all,
            savename="HU_MO_" + savename,
            x="log2_mean_ratio",
            clusters=params["clusters_mo"],
            xlim = -np.log(0.05),
            ylim = 0.5,
            title="Human vs Monkey",
            side_1="Human",
            side_2="Monkey"
            )
    volcano_plot(df_mouse_sort,
                save_dir,
                list_feat_all,
                savename="HU_MU_" +savename,
                x="log2_mean_ratio",
                clusters=params["clusters_mu"],
                xlim = -np.log(0.05),
                ylim = 0.5,
                title="Human vs Mouse",
                side_2="Mouse"
                )
    volcanot_plot_reverse(df_mouse_sort,save_dir,
                list_feat_all,
                savename="HU_MU_" +savename,
                x="log2_mean_ratio",
                clusters=["cluster_mu"],
                title="Human vs Mouse",
                xlim = -np.log(0.05),
                ylim = 0.5)




def main():
    region = "hipp"
    if region == "ctx":
        params = params_visu.params_ctx
    elif region == "str":
        params = params_visu.params_str
    elif region == "hipp":
        params = params_visu.params_hipp
    list_feat_all = list_feat_surface
    if "PrP" in list_feat_all:
        list_feat_all.remove("PrP")
    df, save_dir = load_data(params, region, list_feat_all) 
    cluster_method = "aec"
    savename = "TEST"

    ####boxplot frq
    cluster_method = 'aec'
    mm = get_freq_matrix(df, cluster_method)
    boxplot_freq(mm, params["mapping"],
                 save_dir, savename,
                 params["color_map_species"],
                 params["order"],
                 params["list_no_hu"],
                 params["list_no_mo"],
                 params["list_no_mouse"])


    ####Heatmap
    heatmap(df,
            list_feat_all + [cluster_method, 'Sample_num'],
            list_feat_all,
            save_dir, savename,
            cluster_method = "aec")
    #### Plot TSNEs
    list_feat_c=["SNAP25","GAD65","VGLUT","VMAT2", "Calreticulin", "GATM", "CD47", "CD56"]
    plotTSNEs(df, save_dir, params, savename, list_feat_all, list_feat_c)

    ### Correlation matrix and graph
    mapping_sp = {"mouse":"Mu", "hu":"Hu","mk":"Mo"}
    color_species = {val : params["color_map_species"][key] 
                        for key, val in mapping_sp.items()}
    df_mE_ = correlation_matrix_plot(df, save_dir, 
                            params["dico_color"], color_species,
                            keys_to_groupby=["Specie",
                                             "aec",
                                             'Sample_num'] +list_feat_all,
                            main_key = "Specie",
                            clustering_key = "aec",
                            savename="correlation",
                            mapping_key = mapping_sp)
    plot_correlation_network(df, save_dir, 
                            params["dico_color"], 
                            color_species,
                            keys_to_groupby=["Specie","aec",
                                            'Sample_num'] 
                                            +list_feat_all,
                            main_key = "Specie",
                            savename="correlation",
                            clustering_key = "aec",
                            mapping_key = {"mouse":"Mu", "hu":"Hu","mk":"Mo"})


    #####Plot volcano plots
    plotVolcanoes(df, save_dir, params, savename, list_feat_all)


    ###Plot KNN
    if region == "ctx" or region == "str":
        map_label={"hu":"Hu","mk":"Mo", "mouse":"Mu"}
        groupby = ["Specie","aec", 'Sample_num'] 
        node_mapping = {"Hu":1,"Mo":2, "Mu":3}
        list_ac = params["clusters_mo"]# ["HuMo%d"%k for k in range(1,11)] + ["A1"]
        df.Sample_num = df.Sample_num.map(str)
        df.Sample_num.unique()
        df_hu_mu = df[(df.Specie.isin(["hu", "mk"])) & (df.aec.isin(list_ac))]

        plotKNNGraph(df_hu_mu, groupby, list_feat_all, 
                        params["dico_color"],
                        save_dir, savename,
                        map_label=map_label, nodes_key="Specie",
                        node_mapping={"Hu":1,"Mo":2},
                        dicosp = {"Hu":"o","Mo":"s"})


if __name__ == "__main__":
    main()
