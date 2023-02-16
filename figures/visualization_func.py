import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as transforms
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from scipy.stats import mannwhitneyu

#from statannot import add_stat_annotation
import umap
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from scipy.spatial import ConvexHull
import sys


from scipy.stats import pearsonr, ttest_ind, wilcoxon
#statsmodels.stats.multitest.fdrcorrection
import statsmodels.stats.multitest as ssm
from statsmodels.stats.multitest import fdrcorrection

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import networkx as nx
import matplotlib.patheffects as pe
from natsort import natsorted, index_natsorted, order_by_index

def get_center(df, list_feat, labels):
    clf = NearestCentroid()
    clf.fit(df[list_feat], df[labels])
    return clf.centroids_


def plot_fancy_scatter_clustering(sub_sample, dico_color,
                    save_dir, savename="TSNE",
                    dico_offset_x=None,
                    dico_offset_y=None,
                    annot=True,
                    cluster_key="aec", method_name="tsne"):
    fig, axes = plt.subplots(figsize=(12, 10)) 

    for i, label in enumerate(sub_sample[cluster_key].unique()):
        dt = sub_sample[sub_sample[cluster_key]==label]
        tt = len(dt)
        if (dico_offset_x is not None) & (dico_offset_y is not None):
            ctr = [dt[method_name + "_0"].mean()+dico_offset_x[label], dt[method_name + "_1"].mean()+dico_offset_y[label]]
        else:
            ctr = [dt[method_name + "_0"].mean(), dt[method_name + "_1"].mean()]


        plt.scatter(x=dt[method_name + "_0"], y=dt[method_name + "_1"],
                                #hue="aec",
                                s=12,
                                c=dico_color[label],
                                alpha=0.3,
                                marker='.',
                                #legend=True
                                )

        
        if annot:
            plt.annotate(label, 
                         ctr,#enters[i,:],
                         #sub_sample.loc[sub_sample['aec']==label,[method_name + "_0",method_name + "_1"]].mean(),
                         horizontalalignment='center',
                         verticalalignment='center',
                         #size=15, #weight='bold',
                         fontsize=22,
                         fontfamily="fantasy",
                         fontstyle="oblique",
                         #path_effects=[pe.withStroke(linewidth=0.2, foreground="black")],
                         color= dico_color[label])
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.xlabel(method_name + ' 1')
    plt.ylabel(method_name + ' 2')
    savepath = (save_dir  + savename  + method_name + cluster_key + ".svg")
    plt.savefig(savepath)
    plt.show()
    print("save plot " + savepath)
    plt.clf()
    plt.close('all')   
    
def plot_fancy_scatter(sub_sample, color_map,
                    save_dir, savename="TSNE",
                    dico_offset_x=None,
                    dico_offset_y=None,
                    mapping_label=None,
                    color_key="Species", method_name="tsne"):
    len_list_keys = len(sub_sample[color_key].unique())
    print(sub_sample[color_key].unique())
    fig, axes = plt.subplots(1, len_list_keys, 
                    figsize=(6*len_list_keys,6), sharex=True, sharey=True) 

    for i, label in enumerate(sub_sample[color_key].unique()):
        dt = sub_sample[sub_sample[color_key]==label]
        tt = len(dt)
        ctr = [dt[method_name + "_0"].mean(), dt[method_name + "_1"].mean()]
        color = color_map[label]
        if len_list_keys>1:
            ax = axes[i]
        else:
            ax = axes
        ax.scatter(x=sub_sample[method_name + "_0"], 
                        y=sub_sample[method_name + "_1"],
                                s=3,
                                c="grey",
                                alpha=0.05,
                                marker='.',
                                )

        ax.scatter(x=dt[method_name + "_0"], y=dt[method_name + "_1"],
                                s=8,
                                c=color_map[label],
                                alpha=0.1,
                                marker='.',
                                #legend=True
                                )
        if mapping_label is not None:
            ax.set_title(mapping_label[label],fontsize=22)
        else:
            ax.set_title(label,fontsize=22)
        ax.axis("off")

    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    savepath = (save_dir + method_name + '_' + color_key + savename + ".png")
    plt.savefig(savepath, transparent=True)
    plt.show()
    print("save plot " + savepath)
    plt.clf()
    plt.close('all')   
    
def plot_fancy_scatter_bacth_effect(sub_sample, mapping_color,
                    save_dir, savename="TSNE",
                    dico_offset_x=None,
                    dico_offset_y=None,
                    color_key="Sample_num",
                    stratified_key="Species",
                    mapping_label=None,
                    method_name="tsne"):
    len_list_keys = len(sub_sample[stratified_key].unique())
    fig, axes = plt.subplots(1, len_list_keys, 
                    figsize=(10*len_list_keys, 6), sharex=True, sharey=True) 
    s_sub_sample = sub_sample#.groupby(["Sample_num"]).sample(300, replace=True, random_state=1)
    s_sub_sample["Sample_num"] = np.asarray(
                                    s_sub_sample["Sample_num"]).astype('str')

    for i, label in enumerate(sub_sample[stratified_key].unique()):
        dt = s_sub_sample[s_sub_sample[stratified_key]==label]
        tt = len(dt)
        ctr = [dt[method_name + "_0"].mean(), dt[method_name + "_1"].mean()]
        if len_list_keys>1:
            ax = axes[i]
        else:
            ax = axes
        ax.scatter(x=s_sub_sample[method_name + "_0"], 
                        y=s_sub_sample[method_name + "_1"],
                                s=15,
                                c="grey",
                                alpha=0.05,
                                marker='.',
                                #legend=True
                                )
        dt_to_plot = dt.sort_values(color_key, ascending=False)
        codes, labels = pd.factorize(dt_to_plot[color_key], sort=True)
        codes_c = [mapping_color[ccc] for ccc in codes]
        x = dt_to_plot[method_name + "_0"]
        y = dt_to_plot[method_name + "_1"]
        plot_idx = np.random.permutation(x.shape[0])
        sc = ax.scatter(x=x.iloc[plot_idx], 
                             y=y.iloc[plot_idx],
                            c = codes_c,#colors.iloc[plot_idx,0],
                                s=10, 
                                alpha=0.3,
                                marker='.',
                                #legend=True
                                cmap=plt.get_cmap("Set2")
                                )

        h = lambda c: plt.Line2D([],[],color=c, ls="",marker="o")
        ax.legend(handles=[h(mapping_color[i]) for i in range(len(labels))],#sc.cmap(sc.norm(i))
               labels=list(labels), bbox_to_anchor=(1, 1))
        if mapping_label is not None:
            ax.set_title(mapping_label[label],fontsize=22)
        else:
            ax.set_title(label,fontsize=22)
        ax.axis("off")


    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    savepath = (save_dir + method_name + '_scatterplot_batch_effect_'  + color_key +
                    stratified_key + savename+ ".png")
    plt.savefig(savepath)
    plt.show()
    print("save plot " + savepath)
    plt.clf()
    plt.close('all')
    
#plot good tsne per marker level

def plot_scatter_marker_enrichment(sub_sample, list_feat_c,                 
                    save_dir, savename="TSNE",
                    nb_rows=2,
                    nb_cols=4,
                    figsize=(17, 9),       
                    method_name="tsne"):

    fig, axes = plt.subplots(nb_rows, nb_cols, 
                                figsize=figsize,
                                sharex=True, sharey=True) 
    axes=axes.flatten()
    for i, label in enumerate(list_feat_c):


        color =[str(item/255.) for item in sub_sample[label]]
        mx=np.max(np.log(sub_sample[label]+1))
        mn=np.min(np.log(sub_sample[label] +1))
        axes[i].scatter(x=sub_sample[method_name + "_0"], y=sub_sample[method_name + "_1"],
                                s=3,
                                c=(np.log(sub_sample[label] +1)-mn)/(mx-mn),
                                cmap="Purples",
                                alpha=1,
                                marker='.',
                                )
        axes[i].set_title(label,fontsize=30)
        axes[i].axis("off")

    norm = plt.Normalize(0, 1)
    sm =  ScalarMappable(norm=norm, cmap="Purples")
    sm.set_array([])
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()

    savepath = (save_dir + method_name + '_markers_level_cell' + savename + ".png")
    plt.savefig(savepath)
    plt.show()
    print("save plot " + savepath)
    plt.clf()
    plt.close('all')   
    
# correlation Matrix
from natsort import natsorted, index_natsorted, order_by_index,natsort_keygen



def correlation_matrix_plot(df, dir_path, dico_color_cluster,
                            color_species,
                            keys_to_groupby,
                            main_key = "Species",
                            clustering_key = "aec",
                            savename="correlation",
                            ):
    df_mE= df[keys_to_groupby].groupby([main_key,clustering_key ]).mean()
    df_mE = df_mE.reset_index().sort_values([main_key,clustering_key ], key=natsort_keygen()).set_index([main_key, clustering_key ])
    df_mE_ = df_mE.T.corr()

    sp_labels = df_mE_.columns.get_level_values(0)
    sp_pal = sns.cubehelix_palette(sp_labels.unique().size, light=.9, dark=.1, reverse=True, start=1, rot=-2)
    sp_lut = dict(zip(map(str, sp_labels.unique()), sp_pal))
    sp_colors = pd.Series(sp_labels, index=df_mE_.columns).map(color_species)


    # node aec colors
    node_labels = df_mE_.columns.get_level_values(clustering_key)
    node_pal = sns.cubehelix_palette(node_labels.unique().size,
                                    light=.9, dark=.1, reverse=True, 
                                    start=1, rot=-2)
    node_lut = dict(zip(map(str, node_labels.unique()), node_pal))

    node_colors = pd.Series(node_labels, index=df_mE_.columns).map(dico_color_cluster)

    #df of row and col maps
    aec_node_colors = pd.DataFrame(node_colors).join(pd.DataFrame(sp_colors))
    sns.set(font_scale=1.5)
    g= sns.clustermap(df_mE_,figsize=(20,16),
                yticklabels=True,
                      row_cluster=False, col_cluster=False,
                      xticklabels=True, 
                      #dendrogram_ratio=(0.1,.1),
                      cmap="Purples",
                        row_colors=aec_node_colors,
                      col_colors=aec_node_colors,linewidths=0)
    for label in sp_labels.unique():
            g.ax_col_dendrogram.bar(0, 0, color=color_species[str(label)], label=label, linewidth=0)

    l1 = g.ax_col_dendrogram.legend(title=main_key, loc="upper right", ncol=1, bbox_to_anchor=(0.1,-3))

    # node legend
    #for label in node_labels.unique():
    #    g.ax_row_dendrogram.bar(0, 0, color=dico_color[label], label=label, linewidth=0)

    #l2 = g.ax_row_dendrogram.legend(title='Cluster', loc='upper right', ncol=1, bbox_to_anchor=(5.5, 1))
    g.ax_heatmap.tick_params(left=False, bottom=False) 
    
    ll = [str(it).split('-')[1][:-2] for it in g.ax_heatmap.get_xticklabels()]
 
    g.ax_heatmap.set_xticklabels(ll, fontsize=18,horizontalalignment='left')
    g.ax_heatmap.set_yticklabels(ll, fontsize=18,horizontalalignment='left')

    g.cax.set_position([.1, .02, .01, .15])
    plt.savefig(dir_path + savename + "mean_expression_cluster_species_Less_marker.svg", bbox_inches="tight")
    plt.show()
    plt.close("all")
    print("save plot Correlation Matrix")
    return df_mE_

    


def plot_correlation_network(df, dir_path, dico_color_cluster,color_species,
                            keys_to_groupby,
                            main_key = "Species",
                            clustering_key = "aec",
                             savename="correlation"
                           ):
    df["Species"] = df["Species"].replace({"Ma":"NHP"})
    df_mE= df[keys_to_groupby].groupby([main_key,clustering_key ]).mean()
    df_mE = df_mE.reset_index().sort_values([main_key,clustering_key ], key=natsort_keygen()).set_index([main_key, clustering_key ])

    df_mE_ = df_mE.T.corr()
    df_mE_tmp =df_mE.T
    df_mE_tmp.columns = df_mE_tmp.columns.map('|'.join).str.strip('|')
    df_mE_tmp.index = df_mE_tmp.index.map('|'.join).str.strip('|')
    def calculate_pvalues(df, bonf_corr=0):
        df = df.dropna()._get_numeric_data()
        dfcols = pd.DataFrame(columns=df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')
        for r in df.columns:
            for c in df.columns:
                pvalues[r][c] = pearsonr(df[r], df[c])[1]
                bonf_corr+=1
        return pvalues, bonf_corr
    df_pvalue,bonf_corr = calculate_pvalues(df_mE_tmp)
    df_pvalue *= bonf_corr

    df_mE_ = np.abs(df_mE.T.corr())
    df_mE_.columns = df_mE_.columns.map('|'.join).str.strip('|')
    df_mE_.index = df_mE_.index.map('|'.join).str.strip('|')
    links = df_mE_.stack().reset_index()
    pvalue= df_pvalue.stack().reset_index()


    links.columns = ['from', 'to', 'value']
    links[["Species", "Cluster"]] = links["from"].str.split("|", expand=True)
    links["Species"].replace({"Hu":1,"NHP":2, "Mu":3}, inplace=True)
    mapp = {it:k for it, k in zip(links["Cluster"].unique(), range(len(links["Cluster"].unique())))}
    links["Cluster_map"] = links["Cluster"].replace(mapp)
    links["pvalue"] = pvalue[0]

    links_filtered=links.loc[ (np.abs(links['value']) >= df_mE_.mean().mean()) & (links['from'] != links['to']) & (links['pvalue'] <= 0.05)]
    G=nx.from_pandas_edgelist(links_filtered, 'from', 'to', 'value')
    links_filtered.set_index("from", inplace=True) 
    links.set_index("from", inplace=True) 
    nx.set_node_attributes(G, pd.Series(links.Species, index=links.index).to_dict(), name='Species')
    nx.set_node_attributes(G, pd.Series(links.Cluster, index=links.index).to_dict(), name='Cluster')
    nx.set_node_attributes(G, pd.Series(links.Cluster_map, index=links.index).to_dict(), name='Cluster_map')
    number_to_adjust_by = 5

    dico_spe = nx.get_node_attributes(G,"Species")
    node_list_sp = [ dico_spe[it] for it in G.nodes() ]

    nodePose = nx.layout.spring_layout(G,seed=2)
    node_list_hu = [int(it) for it in range(len(node_list_sp)) if node_list_sp[it] ==1]
    nodePose_list_hu = {k: nodePose[k] for k in nodePose.keys() if dico_spe[k]==1}
    nodePose_list_ma = {k: nodePose[k] for k in nodePose.keys() if dico_spe[k]==2}
    list_size_hu=[0 for _ in range(len(node_list_sp))]
    for k in node_list_hu:
        list_size_hu[k] = 100
    node_list_ma = [it for it in range(len(node_list_sp)) if node_list_sp[it] ==2]
    list_size_ma=[0 for _ in range(len(node_list_sp))]
    for k in node_list_ma:
        list_size_ma[k] = 100
    node_list_mouse = [it for it in range(len(node_list_sp)) if node_list_sp[it] ==3]
    dicosp = {"Hu":"o","NHP":"s", "Mu":"^"}
    node_colors = [dico_color_cluster[it.split('|')[1]] for it in G]
    node_shapes = [dicosp[it.split('|')[0]] for it in G]
    for i,node in enumerate(G.nodes()):
        G.nodes[node]['color'] = node_colors[i]
        G.nodes[node]['shape'] = node_shapes[i]


    mapping_ll = {it:it.split('|')[0] for it in G}
    nx.draw(G, nodePose,node_color="white",labels=mapping_ll,
            font_size=9, font_family="cursive",
            node_size=1, edge_color='#BABABA', linewidths=1)

    # Draw the nodes for each shape with the shape specified


    for shape in set(node_shapes):
        # the nodes with the desired shapes
        node_list = [node for node in G.nodes() if G.nodes[node]['shape'] == shape]

        nx.draw_networkx_nodes(G,nodePose,
                               nodelist = node_list,
                               node_color= [G.nodes[node]['color'] for node in node_list],
                               node_shape = shape)   

    #plt.tight_layout()

    plt.savefig(dir_path + savename + "graph.svg", format="svg")
    print("save plot Graph Correlation")
    plt.show()

def heatmap(df, keys,list_feat_all,
             dir_path, savename,
             cluster_method = "aec"):
    fig, axes = plt.subplots(figsize=(8, 8))  
    scaler = StandardScaler()

    df_ = df[list_feat_all + [cluster_method, 'Sample_num']]
    df_hu_m_k_s = df_.groupby([cluster_method, 'Sample_num']).mean()
    mm_hu = df_hu_m_k_s.reset_index()
    mmmm_hu = mm_hu.groupby([cluster_method]).mean().reset_index()
    mmmm_hu[list_feat_all] = scaler.fit_transform(mmmm_hu[list_feat_all])
    mmmm_hu = mmmm_hu.groupby([cluster_method]).mean().reset_index().sort_values(["aec"], key=natsort_keygen())
    nb_clusters = df_[cluster_method].nunique()

    sns.heatmap(mmmm_hu[list_feat_all].T,
                ax=axes,
                #cmap='YlGnBu',
                cmap="Purples",
                xticklabels=mmmm_hu.aec,
                yticklabels=list_feat_all,
                #row_cluster=False,
                #z_score=0,
                vmin=-1.5, vmax=3,
                cbar_kws={"orientation": "vertical", "label":"Mean Expression"},
                cbar_ax = fig.add_axes([0.98, .7, .03, .2]))

    for i in range(mmmm_hu[list_feat_all].T.shape[1] + 1):
                    axes.axvline(i, color='white', lw=1)
    savepath = (dir_path + savename + 'Heatmap_all__heatmap_' + ".svg")
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.show()

    plt.close("all")
    print("save heat map")
    fig, axes = plt.subplots(1, 3, figsize=(20,7))
    axes = axes.flatten()
    tmp = df.groupby(["Species","aec", "Sample_num"])["aec"].count().reset_index("Sample_num")
    tmp.columns = ["Sample_num", "count_per_cluster"]
    total = pd.DataFrame(df.groupby(["Species","aec"])["aec"].count())
    total.columns = ["Total"]
    total = tmp.join(total)
    palette = {"8358":"#B8ACC2","8363":"#7272E8", #5252A8",
        "8379":"#5252A8",#67388F",
        "8395":"#2D2D5C",
        "HF13_117" :"#8F6D38",
        "HF14_008":"#C2A938",##69614D
        "HF14_051":"#CCBD78",
        "HF14_053":"#F5EBC1",
        "HF14_057": "#2D2D5C",
        "HF14_076":"#B8ACC2",
        "WT1" : "#2D2D5C",
        "WT2":"#F5EBC1",
        "WT3" :"#C2A938",
        "WT4":"#B8ACC2",
        "WT5":"#67388F"}
    total["Proportion"] = total["count_per_cluster"].values/total["Total"].values
    total =total.reset_index()
    total["Species"] = total["Species"].replace({"Ma":"NHP", "Mi":"Mu"})
    for ii, it in enumerate(total.Species.unique()):
        ax= axes[ii]
        tmp = total[total.Species == it]
        colors = [palette[ll] for ll in tmp.Sample_num.unique()]

        tmp = tmp.pivot(index="aec", columns="Sample_num", values="Proportion").fillna(0)
        tmp = tmp.reindex(index=order_by_index(tmp.index, 
            index_natsorted(tmp.index)))
        tmp.plot(kind="bar", stacked=True, ax=ax, color=colors)
        ax.set_title(it)
        #ax.legend("")
        if False:
            for p in ax.patches:
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy() 
                ax.text(x+width/2, 
                y+height/2, 
                '{:.0f}%'.format(height*100), 
                horizontalalignment='center', 
                verticalalignment='center')
    savepath = (dir_path + savename + 'barplot_cluster_composition.svg')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")

    
### Define signature matrix
def create_signature_matrix(df, list_feat_all, cluster_method = 'aec'):
    df_ = df[list_feat_all + [cluster_method, 'Species', 'Brain_area', 'Sample_num' ]]
    df_m_k_s = df_.groupby([cluster_method, 'Species', 'Brain_area', 'Sample_num']).mean()
    sig_mm = df_m_k_s.reset_index()
    #sig_mm = sig_mm_.pivot(index='Sample_num', columns=['Brain_area', 'aec'], values=list_feat_all)
    #sig_mm.columns = sig_mm.columns.to_flat_index()
    return sig_mm

import matplotlib.transforms as transforms
import statsmodels.api as sm


def ttest_per_cluster_per_marker(sig_mm, list_feat_all):
    """ 
    DE using ttest across markers/clusters
    input signature matrix 
    outputs dataframes with P-value, corrected pvalue, log2_mean_ratio, log2_median_ratio
            after DE between MA vs HU, and MI vs HU
    """
    
    sig_mm["Species"] = sig_mm["Species"].replace({"Mi":"Mu", "Ma":"NHP"})
    sm_hu = sig_mm[sig_mm.Species=="Hu"]
    sm_ma = sig_mm[sig_mm.Species=="NHP"]
    sm_mouse = sig_mm[sig_mm.Species=="Mu"]
    df_p_value = pd.DataFrame(columns=["Species", "Brain_area", "aec",
                                       "marker", 'pvalue', "log2_mean_ratio","log2_median_ratio","log10_mean_ratio",
                                       "Effect_size_old","mean_mean_difference_ratio", "Effect_size","logFC" ])
    Spec = ["NHP", "Mu"]
    nb_test = 0
    for sp in Spec:
        for cl in sig_mm["aec"].unique():
            for ba in sig_mm["Brain_area"].unique():
                for marker in list_feat_all:
                    x = sm_hu[(sm_hu["Brain_area"] == ba) & (sm_hu["aec"] == cl)]
                    if sp == "Mu":
                        if ba != "NSTR":
                            y = sm_mouse[(sm_mouse["Brain_area"] == ba) & (sm_mouse["aec"] == cl)]
                            #effect = np.median(y[marker]) /np.median(x[marker])
                            if np.mean(x[marker]) < 1e-9:
                                if np.mean(y[marker])>1e-9: 
                                    print(np.mean(y[marker]))

                            effect = np.log2(np.mean(y[marker]) 
                                            /np.mean(x[marker]))
                            log10_fc = np.log10(np.mean(y[marker])
                                            /np.mean(x[marker]))

                            #effect = np.mean(y[marker]) /np.mean(x[marker])
                            effect2 = np.log2(np.median(y[marker])
                                    /np.median(x[marker]))
                            median_ratio  = ((np.median(y[marker])
                                            -np.median(x[marker]))
                                            /((np.median(y[marker])
                                                +np.median(x[marker]))/2)) 
                            mean_ratio  = (np.mean(y[marker])-np.mean(x[marker]))/((np.mean(y[marker])+np.mean(x[marker]))/2) 
                            log_fold_change = np.log2(np.abs(np.mean(y[marker])-np.mean(x[marker])) /np.maximum(np.abs(np.mean(y[marker])), np.abs(np.mean(x[marker]))))
                            std = np.sqrt((np.var(x[marker])+ np.var(y[marker]))/2)

                            effect_size  = (np.mean(y[marker])-np.mean(x[marker]))/(std +1)


                            re = stats.ttest_ind(x[marker], y[marker], alternative="two-sided")
                            df_p_value.loc[nb_test] =["Mu", ba,
                                                    cl, marker, re[1], 
                                                    effect, effect2, 
                                                    log10_fc,median_ratio, 
                                                    mean_ratio, effect_size,
                                                    log_fold_change]
                            nb_test +=1
                    elif sp == "NHP":
                        if ba != "HIPP":
                            y = sm_ma[(sm_ma["Brain_area"] == ba) & (sm_ma["aec"] == cl)]
                            re = stats.ttest_ind(x[marker], y[marker], alternative="two-sided")
                            effect = np.log2(np.mean(y[marker]) /np.mean(x[marker]))
                            log10_fc = np.log10(np.mean(y[marker]) /np.mean(x[marker]))

                            effect2 = np.log2(np.median(y[marker]) /np.median(x[marker]))
                            median_ratio  = ((np.median(y[marker])
                                            -np.median(x[marker]))
                                            /((np.median(y[marker])
                                                +np.median(x[marker]))/2))
                            mean_ratio  = ((np.mean(y[marker])
                                            -np.mean(x[marker]))
                                            /((np.mean(y[marker])
                                                +np.mean(x[marker]))/2))
                            log_fold_change = np.log2(np.abs(
                                        np.mean(y[marker])
                                        -np.mean(x[marker]))
                                        /np.maximum(np.abs(
                                            np.mean(y[marker])), 
                                            np.abs(np.mean(x[marker]))))
                            std = np.sqrt((np.var(x[marker])
                                        +np.var(y[marker]))/2)

                            effect_size  = ((np.mean(y[marker])
                                            -np.mean(x[marker]))
                                            /(std +1))
                            df_p_value.loc[nb_test] =["NHP", ba, cl,
                                            marker, re[1], effect,
                                            effect2, log10_fc,median_ratio,
                                            mean_ratio,effect_size, 
                                            log_fold_change]

                            nb_test +=1

    df_p_value.dropna(subset = ["pvalue"], inplace=True)
    df_p_value["bonferonni_corr"] = df_p_value.pvalue.values * nb_test
    q_value = sm.stats.fdrcorrection(df_p_value.pvalue.values, alpha=0.05,
                                                        method='indep', is_sorted=False)
    df_p_value["q_value"] = q_value[1]
    df_p_value["-log_q_value"] = -np.log(q_value[1])
    
    df_p_ma = df_p_value[df_p_value.Species =="NHP"]
    df_ma_sort = df_p_ma.sort_values(["q_value"], ascending=True)
    df_p_mouse = df_p_value[df_p_value.Species =="Mu"]
    df_mouse_sort = df_p_mouse.sort_values(["q_value"], ascending=True)
    return df_mouse_sort, df_ma_sort, df_p_value

def volcano_plot(df_,
                    dir_path, 
                    list_feat_all,
                     savename="HU_MK_test",
                    title="Human vs Monkey",
                    x="log2_mean_ratio",
                    clusters=["HuMa%d"%k for k in range(1,11)],
                    xlim = -np.log(0.05),
                    ylim = 0.5,
                     side_1="Human",
                     side_2="Monkey"
                    ):
    """
    Volcano plots in the manuscript
    inputs: DF from ttest_per_cluster_per_marker
            dir_path: directory where to save the figure
            title: title of the plot
            x: x axis
            clusters: clusters to consider
    """
    list_24 = ["#144d00", "#330e00","#ff4400","#4c3300", "#DAEA05", 
                "#ff80e5", "#B31109", "#E4C9FF",
                "#ffeabf", "#535ea6", #8A00FF
               "#005953", "#EBCB6F",  "#730000",
                "#7f6c20", "#b6f2de", "#00eeff", "#3d55f2", 
                 "#00ff44","#f23d6d","#ffaa00","#b386b0",
                 "#609fbf","#79bf60", "#cc00ff",  ]

    list_24.reverse()
    colormap_marker = {it:col for (it,col) in zip(list_feat_all, list_24[:len(list_feat_all)])}
    colormap_marker_it= {"DJ1":"#ffaa00","AS":"#b386b0","ApoE":"#f23d6d",
            "LRRK2":"#79bf60", "GAMT":"#609fbf", "VMAT2":"#00ffcc",
            "BIN1":"#3d55f2", "Calreticulin":"#cc00ff", "CD47":"#4c3300"}
    for key,val in colormap_marker_it.items():
        colormap_marker[key] = val
    sns.set(font_scale=1, style="white")
    #sns.set_style("white", {'axes.grid' : False})
    fig, ax = plt.subplots(figsize=(7, 7), sharey=False, sharex=True)

    for i, ba in enumerate(df_.Brain_area.unique()):
        jj = df_[(df_.Brain_area == ba)]

        jj.reset_index(inplace=True)
        jj['aec'] = jj['aec'].astype(str)
        jj_s = jj[((jj["-log_q_value"]>xlim) & (jj[x] >ylim))
                |((jj["-log_q_value"]>xlim) & (jj[x] <-ylim)) ]
        g =sns.scatterplot(ax=ax, data=jj_s, y="-log_q_value",
                x=x, hue="marker",palette=colormap_marker)

        jj_ns = jj[((jj["-log_q_value"]<xlim) & (jj[x] <ylim)) 
                | ((jj["-log_q_value"]<xlim) & (jj[x] >-ylim)) 
                |((jj["-log_q_value"]>xlim) 
                    & (jj[x] >-ylim) & (jj[x] <ylim))]
        g =sns.scatterplot(ax=ax, data=jj_ns, y="-log_q_value",
                x=x, color="grey", s=12)
        ax.set_xlim(-2, 2)

        if i==3:
                h,l = g.axes.get_legend_handles_labels()
                g.axes.legend_.remove()
                g.legend(h,l, bbox_to_anchor=(1.18, 1),
                       borderaxespad=0, ncol=1)
        else:
                h,l = g.axes.get_legend_handles_labels()
                g.axes.legend_.remove()
                g.legend(h,l, bbox_to_anchor=(1.18, 1),
                       borderaxespad=0, ncol=1)
        ax.axhline(y=xlim, color= 'k',
                linestyle='--', linewidth=0.8)
        ax.axvline(x=-ylim, color= 'k',
                linestyle='--', linewidth=0.8)
        ax.axvline(x=ylim, color= 'k', 
                linestyle='--', linewidth=0.8)
        for i in range(jj.shape[0]):

            ss = str(jj.loc[i, "aec"])
            #ax.text(x=jj.loc[i,"log_q_value"]+0.01,y=jj.loc[i, "marker"],s=ss, fontdict=dict(color='red',size=10), )
        #ax.title.set_text(ba)
        ax.title.set_fontsize(22)
        ax.text(-1.3, 12.5, "+ " + side_1, alpha=0.3)
        ax.tick_params(labelsize=22)
        ax.set_xlabel("Log2(Fold-change)")
        ax.set_ylabel("-log(Q-value)")
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
        ax.text(0.7,12.5, "+ " + side_2, alpha=0.3)
    plt.subplots_adjust(wspace=0.5)
    plt.suptitle(title, fontsize=22)
    plt.savefig(dir_path + savename + "volcano_"+
                    x+"_marker_std.svg", bbox_inches='tight')
    plt.show()
    plt.close("all")
    plt.clf()
    print("save plot volcano plot")

    
def volcanot_plot_reverse(df_, dir_path,
                            list_feat_all,
                            savename="HU_MU_",
                            x="log2_mean_ratio",
                            clusters=["HuMu1", "A1"],
                            title="Human vs Mouse",
                            xlim = -np.log(0.05),
                            ylim = 0.5,
                            side_1="Human",
                            side_2="Mouse"
                            ):
    """
    Same as volcano_plot but different colored dots
    """
    list_24 = ["#144d00", "#330e00","#ff4400","#4c3300",
                "#DAEA05", "#00ffcc", 
                "#cc00ff", "#ff80e5", "#B31109", "#E4C9FF", 
                "#ffeabf","#79bf60",  "#535ea6", #8A00FF
                 "#005953", "#609fbf",
               "#EBCB6F", "#ffaa00", "#730000",
                "#7f6c20", "#b6f2de", "#00eeff", "#3d55f2", 
               "#b386b0",  "#00ff44","#f23d6d" ]

    list_24.reverse()
    colormap_marker = {it:col for (it,col) in 
                    zip(list_feat_all, list_24[:len(list_feat_all)])}
    fig, ax = plt.subplots( figsize=(7, 7), sharey=False,sharex=True)
    #print(gg_mouse_sort)

    for i, ba in enumerate(df_.Brain_area.unique()):
        jj = df_[df_.Brain_area == ba]

        jj = jj[jj.aec.isin(clusters)]
        #jj["count"] = jj.groupby(["-log_q_value", x]).transform('count')["aec"].values

        #print(jj)
        jj.reset_index(inplace=True)
        jj['aec'] = jj['aec'].astype(str)
        jj["cluster"] = jj['aec']
        sns.set(font_scale=1)
        sns.set_style("white")

        jj_s = jj[((jj["-log_q_value"]>xlim) & (jj[x] >ylim)) |((jj["-log_q_value"]>xlim) & (jj[x] <-ylim)) & (jj["-log_q_value"]>xlim)]
        g =sns.scatterplot(ax=ax, data=jj_s, y="-log_q_value", x=x,  color="grey", s=12)
        jj_ns = jj[((jj["-log_q_value"]<xlim) & (jj[x] <ylim)) | ((jj["-log_q_value"]<xlim) & (jj[x] >-ylim))|((jj["-log_q_value"]>xlim) & (jj[x] >-ylim) & (jj[x] <ylim))]
        g =sns.scatterplot(ax=ax, data=jj_ns, y="-log_q_value",color="grey", s=12, x=x)
        jj_ns = jj[((jj["-log_q_value"]<xlim) & (jj[x] <ylim) & (jj[x] >-ylim)) ]
        g =sns.scatterplot(ax=ax, data=jj_ns, y="-log_q_value",hue="marker",palette=colormap_marker, x=x, style="cluster")
        ax.set_xlim(-2, 2)
        #ax.legend("")
        try:
            if i==0:
                    h,l = g.axes.get_legend_handles_labels()
                    g.axes.legend_.remove()

                    g.legend(h,l, bbox_to_anchor=(1.22, 1),
                           borderaxespad=0, ncol=1)
            else:
                    h,l = g.axes.get_legend_handles_labels()
                    g.axes.legend_.remove()
                    g.legend(h,l, bbox_to_anchor=(1.22, 1),
                           borderaxespad=0, ncol=1)
        except:
            pass
        for i in range(jj.shape[0]):

            ss = str(jj.loc[i, "aec"])
            #ax.text(x=jj.loc[i,"log_q_value"]+0.01,y=jj.loc[i, "marker"],s=ss, fontdict=dict(color='red',size=10), )
        ax.hlines(y=xlim,xmin=-3.0,  xmax =3, color= 'k', linestyle='--', linewidth=0.8)
        ax.vlines(x=-ylim,ymin=0.0,  ymax =25, color= 'k', linestyle='--', linewidth=0.8)
        ax.vlines(x=ylim,ymin=0.0,  ymax =25, color= 'k', linestyle='--', linewidth=0.8)
        ax.text(-1.3, 15, "+ " + side_1, alpha=0.3)
        ax.text(0.7,15, "+ " +side_2, alpha=0.3)
        ax.tick_params(labelsize=22)
        #ax.title.set_text(ba)
        ax.title.set_fontsize(22)
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
    plt.subplots_adjust(wspace=0.5)
    plt.suptitle(title, fontsize=22)
    plt.savefig(dir_path + savename + "reverse_volcano_"+x+"_aec_marker_std.svg", bbox_inches='tight')#, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close("all")
    print("save plot volcano plot")


def plotKNNGraph(df, groupby, list_feat, dico_color,
                 dir_path, savename,
                 map_label=None, nodes_key="Species",
                 node_mapping=None,
                 dicosp = {"Hu":"o","NHP":"s"}
                ):
    """
    groupby = [node_key, cluster_key, ...]
    """
    df["Species"] = df["Species"].replace({"Ma":"NHP"})
    df_mE= df[groupby+list_feat].groupby(groupby).mean()
    if map_label is not None:
        df_mE.rename(map_label, inplace=True)
    df_mE_tmp =df_mE.T
    df_mE_tmp.columns = df_mE_tmp.columns.map('|'.join).str.strip('|')

    def calculate_pvalues_wilcoxon(df, bonf_corr=0):
        df = df.dropna()._get_numeric_data()    
        dfcols = pd.DataFrame(columns=df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')
        for r in df.columns:
            for c in df.columns:
                if r!=c:
                    pvalues[r][c] = wilcoxon(df[r], df[c])[1]
                    bonf_corr+=1
                else:
                    pvalues[r][c] = 100
        return pvalues, bonf_corr
    df_pvalue,bonf_corr = calculate_pvalues_wilcoxon(df_mE_tmp)
    index = np.triu_indices(1)
    pp_value =np.asarray(df_pvalue)[np.triu_indices(np.asarray(df_pvalue).shape[0], k = 1)]

    _,pvaluecorrected = fdrcorrection(pp_value)
    df_pvalue *= bonf_corr
    size_X = np.asarray(df_pvalue).shape[0]
    X = np.zeros((size_X,size_X))
    X[np.triu_indices(X.shape[0], k = 1)] = pvaluecorrected

    df_pvalue.loc[:,:] = X + X.T + 100*np.eye(size_X)
    df_mE.index = df_mE.index.map('|'.join).str.strip('|').str.strip('|')

    df_mE_=pd.DataFrame(
        squareform(pdist(df_mE)),
        columns = df_mE.index.tolist(),
        index = df_mE.index.tolist()
    )
    df_mE_ = 1 -(df_mE_ - df_mE_.min().min())/(df_mE_.max()-df_mE_.min())
    links = df_mE_.stack().reset_index()
    pvalue= df_pvalue.stack().reset_index()
    links.columns = ['from', 'to', 'value']
    links["value"] = links["value"]
    columns = [nodes_key, "Cluster", "Sample_num"]
    links[columns] = links["from"].str.split("|", expand=True)
    links[[it+ "_to" for it in columns]] = links["to"].str.split("|", expand=True)
    if node_mapping is not None:
        links[nodes_key].rename(node_mapping, inplace=True)
        links[nodes_key+"_to"].rename(node_mapping, inplace=True)
    mapp = {it:k for it, k in zip(links["Cluster"].unique(), range(len(links["Cluster"].unique())))}
    links["Cluster_map"] = links["Cluster"].replace(mapp)
    links["pvalue"] = pvalue[0]
    print("Filter edges < the mean value")
    links_filtered=links.loc[(np.abs(links['value']) < df_mE_.mean().mean()) &  (links['from'] != links['to'])]

    G=nx.from_pandas_edgelist(links_filtered, 'from', 'to', 'value')
    links_filtered.set_index("from", inplace=True) 
    links.set_index("from", inplace=True)
    nx.set_node_attributes(G, pd.Series(links.Species, index=links.index).to_dict(), name=nodes_key)
    dico_spe = nx.get_node_attributes(G,nodes_key)
    print(nx.average_node_connectivity(G, flow_func=None))
    nx.set_node_attributes(G, pd.Series(links.Species, index=links.index).to_dict(), name=nodes_key)
    nx.set_node_attributes(G, pd.Series(links.Cluster, index=links.index).to_dict(), name='Cluster')
    nx.set_node_attributes(G, pd.Series(links.Cluster_map, index=links.index).to_dict(), name='Cluster_map')
    number_to_adjust_by = 5

    print(links_filtered)
    dico_spe = nx.get_node_attributes(G,nodes_key)
    node_list_sp = [ dico_spe[it] for it in G.nodes() ]

    edge_colors = []
    for e in G.edges:
        if df_pvalue.loc[e[0],e[1]]>0.001:
            edge_colors.append('#eaeaea')
        elif df_pvalue.loc[e[0],e[1]]<0.001:
            edge_colors.append("#bebebe")
        elif df_pvalue.loc[e[0],e[1]]<0.0001:
            edge_colors.append("#3b3b3b")

    edge_colors = []
    for e in G.edges:
        sp1= e[0].split('|')[0]
        sp2= e[1].split('|')[0]
        if sp1 == sp2:
            edge_colors.append((0.8,0.8,0.8,0.3))#(0.67, 0.72, 0.52, 0.4))#0,0,0,0))#((0.2,0.2,0.2,0.1))#(0,0,0,0))
        else:
            edge_colors.append((0.8,0.8,0.8,0.3))#(0.72, 0.48, 0.45, 0.4))#(0.2,0.2,0.2,0.1))#"#544B59")



    #print(G.edges)
    nodePose = nx.layout.spring_layout(G,scale=1000, seed=17)
    #print(G)
    node_list_hu = [int(it) for it in range(len(node_list_sp)) if node_list_sp[it] ==1]
    nodePose_list_hu = {k: nodePose[k] for k in nodePose.keys() if dico_spe[k]==1}
    nodePose_list_ma = {k: nodePose[k] for k in nodePose.keys() if dico_spe[k]==2}
    #print(node_list_hu)
    #print(nodePose_list_hu)
    list_size_hu=[0 for _ in range(len(node_list_sp))]
    for k in node_list_hu:
        list_size_hu[k] = 100
    node_list_ma = [it for it in range(len(node_list_sp)) if node_list_sp[it] ==2]
    list_size_ma=[0 for _ in range(len(node_list_sp))]
    for k in node_list_ma:
        list_size_ma[k] = 100
    node_colors = [dico_color[it.split('|')[1]] for it in G]
    node_shapes = [dicosp[it.split('|')[0]] for it in G]
    #print(node_shapes)
    for i,node in enumerate(G.nodes()):
        G.nodes[node]['color'] = node_colors[i]
        G.nodes[node]['shape'] = node_shapes[i]

    edges,weights = zip(*nx.get_edge_attributes(G,'value').items()) 
    mapping_ll = {it:it.split('|')[0] for it in G}
    nx.draw(G, nodePose,node_color="white",labels=mapping_ll,
            font_size=9, font_family="cursive",
            node_size=1,  linewidths=1,edge_color=edge_colors)#, edge_cmap=plt.cm.Greys)

    # Draw the nodes for each shape with the shape specified


    for shape in set(node_shapes):
        # the nodes with the desired shapes
        node_list = [node for node in G.nodes() if G.nodes[node]['shape'] == shape]

        nx.draw_networkx_nodes(G,nodePose,
                               nodelist = node_list,

                               alpha=0.6, linewidths=1,
                               node_color= [G.nodes[node]['color'] for node in node_list],
                               node_shape = shape)   

    #plt.tight_layout()

    plt.savefig(dir_path + savename + "graph_KNN_pvalue" + ".svg", format="svg")
    plt.show()

from statannotations.Annotator import Annotator

def get_freq_matrix(df, cluster_method):
    import scipy.stats as stats
    groups_by = ['Species', 'Brain_area', 'Sample_num']
    
    gf_kmeans = df.groupby(groups_by + [cluster_method]).agg({"Sample_num":"count"})
    gf_kmeans.rename(columns={'Sample_num':'events_clusters_subjects'}, inplace=True)
    gf_samples = df.groupby(groups_by).agg({'Sample_num':'count'})   
    gf_samples.rename(columns={'Sample_num':'events_samples'}, inplace=True)
    multi_sample = gf_kmeans.reset_index(cluster_method)
    multi_sample['events_samples'] = gf_samples['events_samples']
    multi_sample['freq'] = multi_sample['events_clusters_subjects'].div(multi_sample['events_samples'])
    mm = multi_sample.reset_index()
    return mm


def boxplot_freq(mm, mapping, save_dir, 
                savename, color_map, 
                order_t,
                list_no_hu,
                list_no_ma,
                list_no_mouse,
                multispecies=False):
    order = [mapping[it] for it in order_t]

    mm.loc[:,"aec"].replace(mapping, inplace=True)
    mm["Species"].replace({"Mi":"Mu"}, inplace=True)
    clusters = mapping.keys() 
    if multispecies:
        common_cl = [mapping[cl] for cl in clusters if (cl not in list_no_hu) & (cl not in list_no_mouse) ]
        common_cl_ma = [mapping[cl]  for cl in clusters if (cl not in list_no_hu) & (cl not in list_no_ma) ]
        box_pair = []
        if "Mu" in mm["Species"].unique().tolist():
            box_pair += [((cl, "Hu"), (cl,"Mu")) for cl in common_cl ]
        if "Ma" in mm["Species"].unique().tolist():
            box_pair += [((cl, "Hu"), (cl,"Ma")) for cl in common_cl_ma ]
        print(box_pair)
  
    fig, ax = plt.subplots(figsize=(15,8))
    sns.set(font_scale=2, style="white")
    #sns.set_style("white")
    sns.boxplot(ax=ax, x='aec', y='freq', hue='Species', data=mm, palette=color_map, order=order, linewidth=1 )
    if multispecies:
        annotator = Annotator(ax, box_pair, data=mm, x='aec', y='freq', order=order,hue='Species')
        annotator.configure(test='Mann-Whitney',  text_format="star", loc='inside', fontsize="18", comparisons_correction="BH")#, correction_format="replace")#,correction_format="replace")
        annotator.apply_and_annotate()

    plt.xticks(rotation=90, fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_ylabel("Mean events frequency per clusters", fontsize=30)
    ax.set_xlabel("")
    ax.legend("")
    ax.yaxis.grid(False) # Hide the horizontal gridlines
    ax.set_xticks([x-0.5 for x in range(1,len(clusters))],minor=True )
    ax.xaxis.grid(True,which="minor") # Show the vertical gridlinesgridlines
    plt.tight_layout()
    plt.savefig(save_dir + savename + "box_plot_with_bonferroni_correction.svg")
    plt.show()
