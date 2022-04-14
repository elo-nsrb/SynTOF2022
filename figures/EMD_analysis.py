import os
import pandas as pd
#import scanpy as sc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import anndata
import matplotlib.pyplot as plt
import matplotlib
import sys
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from scipy import stats
import sys
sys.path.append("../")
from list_feat_params import *
import os 
import itertools
import params_visu
from load_data import load_data



def compute_pairwise_emd_specie(df, 
                    list_clusters_primates,
                    list_feat_all,
                    dir_path,
                    list_species=["hu","mk"]):


    if not os.path.exists(dir_path + "df_wd_specie.csv"):
        df_hu_mk = df[(df.Specie.isin(list_species)) & (df.aec.isin(list_clusters_primates))]
        specie_pairs = list(itertools.combinations(list_species, 2))
        print(specie_pairs)
        df_ks = pd.DataFrame(columns=["type","marker", "cluster1", "Specie1", "cluster2", "Specie2", "wassertein"])
        #print(df_hu_mk)
        index=0
        cluster_method = "aec"
        list_cl = df_hu_mk[cluster_method].unique()
        list_of_pairs = [(list_cl[p1], list_cl[p2]) for p1 in range(len(list_cl)) for p2 in range(p1+1,len(list_cl))]
        tested = []
        for sp1, sp2 in specie_pairs:
            for cl,cl2 in list_of_pairs:

                    df_cl = df_hu_mk[(df_hu_mk["aec"] == cl)]

                    dfm = df_cl.melt(id_vars=['Specie'], value_vars=list_feat_all, var_name="marker")
                    #print(dfm)
                    gggg = dfm.groupby(["Specie", "marker", "value"]).size().reset_index()
                    gggg.rename({0:"count"}, axis=1, inplace=True)
                    gggg["log_count"] = gggg["count"].map(np.log)

                    df_cl_hu = gggg[gggg.Specie==sp1]
                    df_cl_mk = gggg[gggg.Specie==sp2]

                    df_cl2 = df_hu_mk[(df_hu_mk["aec"] == cl2)]

                    dfm2 = df_cl2.melt(id_vars=['Specie'], value_vars=list_feat_all, var_name="marker")
                    gggg2 = dfm2.groupby(["Specie", "marker", "value"]).size().reset_index()
                    gggg2.rename({0:"count"}, axis=1, inplace=True)
                    gggg2["log_count"] = gggg2["count"].map(np.log)

                    df_cl_hu2 = gggg2[gggg2.Specie==sp1]
                    df_cl_mk2 = gggg2[gggg2.Specie==sp2]

                    list_js = []
                    list_ks = []

                    for it in list_feat_all:
                        #print(it)
                        X = df_cl_hu[df_cl_hu.marker==it]["value"]
                        #X = (X -X.min())/(X.max() - X.min())
                        #print(X)

                        #print(X_dist)
                        Y = df_cl_mk[df_cl_mk.marker==it]["value"]
                        if len(X) != 0 and len(Y) != 0:
                            wd = wasserstein_distance(X, Y)
                            #print(pvalue)
                            df_ks.loc[index] = ["Within",it, cl, sp1, cl, sp2, wd]
                            index +=1
                            tested.append(((sp1,cl),(sp2,cl)))
                        #print(df_ks)
                        if cl != cl2:
                            Y2 = df_cl_mk2[df_cl_mk2.marker==it]["value"]
                            if len(X) != 0 and len(Y2) != 0:
                                wd = wasserstein_distance(X, Y2)
                                df_ks.loc[index] = ["Between-different",it, 
                                                    cl, sp1, cl2, sp2, wd]
                                index +=1
                                tested.append(((sp1,cl),(sp2,cl2)))


                            X2 = df_cl_hu2[df_cl_hu2.marker==it]["value"]
                            if len(X) != 0 and len(X2) != 0:
                                wd = wasserstein_distance(X, X2)
                                df_ks.loc[index] = ["Between-same",it, cl,
                                                    sp1, cl2, sp1, wd]
                                index +=1  
                                tested.append(((sp1,cl),(sp1,cl2)))
                            if len(Y) != 0 and len(Y2) != 0:
                                wd = wasserstein_distance(Y, Y2)
                                df_ks.loc[index] = ["Between-same",it, cl, sp2, cl2, sp2, wd]
                                index +=1   
                                tested.append(((sp2,cl),(sp2,cl2)))

                            if len(Y) != 0 and len(X2) != 0:
                                wd = wasserstein_distance(Y, X2)
                                df_ks.loc[index] = ["Between-different",it, cl, sp2, cl2, sp1,wd]
                                index +=1              
                                tested.append(((sp2,cl),(sp1,cl2)))
                        #Ygenerate = dist.generate(n=1000)
                        #js = distance.jensenshannon(Xgenerate, Ygenerate)
                        #print(js)
                        #list_js.append(js)

        print(df_ks)
        df_ks.to_csv(dir_path + "df_wd_specie.csv")
        print(tested)
    else:
        df_ks = pd.read_csv(dir_path + "df_wd_specie.csv")
    return df_ks


def pairwise_comp_per_subject(df, list_clusters_primates,
                                dir_path, list_feat_all,
                                list_species=["hu","mk"]):

    df_hu_mk = df[(df.Specie.isin(list_species)) & (df.aec.isin(list_clusters_primates))]
    df_wd_subject = pd.DataFrame(columns=["type","marker", "cluster1", "Subject1", "cluster2", "Subject2",  "wassertein"])
    #print(df_hu_mk)
    index=0
    list_sub = df_hu_mk.Sample_num.unique()
    cluster_method = "aec"
    list_cl = df_hu_mk[cluster_method].unique()
    list_of_pairs = [(list_cl[p1], list_cl[p2]) for p1 in range(len(list_cl)) for p2 in range(p1+1,len(list_cl))]
    for cl,cl2 in list_of_pairs:

            df_cl = df_hu_mk[(df_hu_mk["aec"] == cl)]

            dfm = df_cl.melt(id_vars=['Sample_num'], value_vars=list_feat_all, var_name="marker")
            #print(dfm)
            gggg = dfm.groupby(["Sample_num", "marker", "value"]).size().reset_index()
            gggg.rename({0:"count"}, axis=1, inplace=True)
            gggg["log_count"] = gggg["count"].map(np.log)
            for sub1 in list_sub:
                for sub2 in list_sub:
                    if sub1!=sub2:
                        df_cl_hu = gggg[gggg.Sample_num==sub1]
                        df_cl_mk = gggg[gggg.Sample_num==sub2]

                        df_cl2 = df_hu_mk[(df_hu_mk["aec"] == cl2)]

                        dfm2 = df_cl2.melt(id_vars=['Sample_num'], value_vars=list_feat_all, var_name="marker")
                        gggg2 = dfm2.groupby(["Sample_num", "marker", "value"]).size().reset_index()
                        gggg2.rename({0:"count"}, axis=1, inplace=True)
                        gggg2["log_count"] = gggg2["count"].map(np.log)

                        df_cl_hu2 = gggg2[gggg2.Sample_num==sub1]
                        df_cl_mk2 = gggg2[gggg2.Sample_num==sub2]

                        list_js = []
                        list_ks = []

                        for it in list_feat_all:
                            #print(it)
                            X = df_cl_hu[df_cl_hu.marker==it]["value"]
                            #X = (X -X.min())/(X.max() - X.min())
                            #print(X)

                            #print(X_dist)
                            Y = df_cl_mk[df_cl_mk.marker==it]["value"]
                            if len(X) != 0 and len(Y) != 0:
                                wd = wasserstein_distance(X, Y)
                                df_wd_subject.loc[index] = ["Within",it, cl, sub1, cl, sub2, wd]
                                index +=1
                            #print(df_ks)
                            if cl != cl2:
                                Y2 = df_cl_mk2[df_cl_mk2.marker==it]["value"]
                                if len(X) != 0 and len(Y2) != 0:
                                    wd = wasserstein_distance(X, Y2)
                                    df_wd_subject.loc[index] = ["Between-different",it, cl, sub1, cl2, sub2, wd]
                                    index +=1


                                X2 = df_cl_hu2[df_cl_hu2.marker==it]["value"]
                                if len(X) != 0 and len(X2) != 0:
                                    wd = wasserstein_distance(X, X2)
                                    df_wd_subject.loc[index] = ["Between-same",it, cl, sub1, cl2, sub1, wd]
                                    index +=1  
                                if len(Y) != 0 and len(Y2) != 0:
                                    wd = wasserstein_distance(Y, Y2)
                                    df_wd_subject.loc[index] = ["Between-same",it, cl, sub2, cl2, sub2, wd]
                                    index +=1   

                                if len(X) != 0 and len(X2) != 0:
                                    wd = wasserstein_distance(X, X2)
                                    df_wd_subject.loc[index] = ["Between-different",it, cl, sub2, cl2, sub1,wd]
                                    index +=1              
                        #Ygenerate = dist.generate(n=1000)
                        #js = distance.jensenshannon(Xgenerate, Ygenerate)
                        #print(js)
                        #list_js.append(js)


    df_wd_subject.to_csv(dir_path + "all_df_wd_subject.csv")
    return df_wd_subject


def plot_emd_comparison(df_ks, dir_path, key="subject"):
    from statannotations.Annotator import Annotator
    fig, ax = plt.subplots(figsize=(26, 10))
    matplotlib.rcParams.update({'font.size': 22})
    df_ks_s = df_ks[~df_ks["type"].isin(["Between-same"])]
    df_ks_s["type"].replace({"Within":"intra-cluster", "Between-different":"inter-cluster"},inplace=True)
    df_ks_m = df_ks_s.groupby(["type", "marker"])["wassertein"].mean().reset_index()
    #print(df_ks_m)
    sns.set(font_scale=2.5)
    sns.set_style("white")
    box_pair = [((mk,"intra-cluster"), (mk,"inter-cluster")) for mk in df_ks_m.marker.unique()]
    fig_args = {'x': "marker",
                'y': "wassertein",
                'hue':"type",
                'data': df_ks_s,

                'hue_order':['intra-cluster','inter-cluster'],
                'dodge': True}

    configuration = {'test':'Mann-Whitney',
                     'comparisons_correction':"BH",
                     'text_format':'star',}
    g = sns.stripplot(ax=ax,palette="Set2", **fig_args,)
    annotator = Annotator(ax=ax, pairs=box_pair,
                          **fig_args, plot='stripplot')
    annotator.configure(**configuration).apply_test().annotate()
    #fig.savefig(f'flu_dataset_log_scale_in_axes_strip.svg', format='svg')
    ymin, ymax = ax.get_xlim()
    #ax.hlines(y=0.05,xmin=ymin,  xmax =ymax, color= 'k', linestyle='--', linewidth=0.8)
    g.set_xticklabels(g.get_xticklabels(), rotation=80)
    ax.set_ylabel("Earth Mover's Distance (EMD)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    fig.tight_layout()
    plt.savefig(dir_path + "wassertein_stripplot_" + key + ".png")
    plt.close("all")
    
    fig, ax = plt.subplots(figsize=(10, 18))
    matplotlib.rcParams.update({'font.size': 26})
    df_ks_s = df_ks[~df_ks["type"].isin(["Between-same"])]
    df_ks_s["type"].replace({"Within":"intra-cluster", "Between-different":"inter-cluster"},inplace=True)
    df_ks_m = df_ks_s.groupby(["type", "marker"])["wassertein"].mean().reset_index()
    #print(df_ks_m)
    sns.set(font_scale=2.5)
    sns.set_style("white")
    box_pair = [("intra-cluster", "intra-cluster"),("inter-cluster","inter-cluster")]
    g = sns.barplot(y="wassertein", x="type", data=df_ks_s, ax=ax,palette="Set2",
            hue="type", order=['intra-cluster','inter-cluster'])
    annotator = Annotator(ax, box_pair, data=df_ks_s, x='type', y='wassertein',
            )#hue="type") #, order=['intra-cluster','inter-cluster'])
    annotator.configure(test='Mann-Whitney',  text_format="full", loc='inside', fontsize="18", comparisons_correction="BH")#, correction_format="replace")#,correction_format="replace")
    annotator.apply_and_annotate()
    ymin, ymax = ax.get_xlim()
    #ax.hlines(y=0.05,xmin=ymin,  xmax =ymax, color= 'k', linestyle='--', linewidth=0.8)
    #g.set_xticklabels(g.get_xticklabels(), rotation=90)
    fig.tight_layout()
    plt.savefig(dir_path + "wassertein_between_distributions_" + key + ".png")
    plt.close("all")
    

def plot_emd_subject(df_wd_subject, dir_path, key="subject"):
    from statannotations.Annotator import Annotator
    fig, ax = plt.subplots(figsize=(26, 10))
    matplotlib.rcParams.update({'font.size': 22})
    df_wd_subject_s = df_wd_subject[~df_wd_subject["type"].isin(["Between-same"])]
    df_wd_subject_s["type"].replace({"Within":"intra-cluster", "Between-different":"inter-cluster"},inplace=True)
    df_wd_subject_m = df_wd_subject_s.groupby(["type", "marker"])["wassertein"].mean().reset_index()
    #print(df_ks_m)
    sns.set(font_scale=2.5)
    sns.set_style("white")
    box_pair = [((mk,"intra-cluster"), (mk,"inter-cluster")) for mk in df_wd_subject_m.marker.unique()]
    fig_args = {'x': "marker",
                'y': "wassertein",
                'hue':"type",
                'data': df_wd_subject_s,

                'hue_order':['intra-cluster','inter-cluster'],
                'dodge': True}

    configuration = {'test':'Mann-Whitney',
                     'comparisons_correction':"BH",
                     'text_format':'star',}
    g = sns.stripplot(ax=ax,palette="Set2", **fig_args,)
    annotator = Annotator(ax=ax, pairs=box_pair,
                          **fig_args, plot='stripplot')
    annotator.configure(**configuration).apply_test().annotate()
    ymin, ymax = ax.get_xlim()
    #ax.hlines(y=0.05,xmin=ymin,  xmax =ymax, color= 'k', linestyle='--', linewidth=0.8)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    ax.set_ylabel("Earth Mover's Distance (EMD)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    fig.tight_layout()
    plt.savefig(dir_path + "wassertein_stripplot_sunject_emd_comparison" + ".png")
    plt.show()
    plt.close("all")



def main():
    region = "str"
    if region == "ctx":
        params = params_visu.params_ctx
        list_species = ["hu","mk", "mouse"]
    elif region == "str":
        params = params_visu.params_str
        list_species = ["hu","mk"]
    elif region == "hipp":
        params = params_visu.params_str
        list_species = ["hu","mouse"]
    list_clusters_primates = df.aec.unique().tolist()
        
    list_feat_all = list_feat_surface
    if "PrP" in list_feat_all:
        list_feat_all.remove("PrP")
    df, save_dir = load_data(params, region, list_feat_all) 
    df_ks = compute_pairwise_emd_specie(df, 
                                list_clusters_primates, 
                                list_feat_all,
                                save_dir,
                                list_species=list_species)
    plot_emd_comparison(df_ks, 
                        save_dir,
                        key="specie_all_clusters")

    df_wd_subject = pairwise_comp_per_subject(df, list_clusters_primates,
                                        save_dir,
                                        list_feat_all,
                                list_species=list_species)
    plot_emd_subject(df_wd_subject, save_dir, key="subject")

if __name__ == "__main__":
    main()
