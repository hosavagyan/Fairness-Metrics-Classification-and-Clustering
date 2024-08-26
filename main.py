from Clustering_Plotting.Clustering import Cluster
from Fairness_Metric_Results.in_processing import in_process_pipline
from Clustering_Plotting.plots import plot_silhouette, plot_cluster_2dim, plot_cluster_groups
from Fairness_Metric_Results.constants import lst_of_data
from Load_Preprocess_Data.Bank_dataset import Bank
from Load_Preprocess_Data.Compas import COMPAS


def main(Dataset: type, sensitive_attr: str):
    # data loading and preparation
    data_class = Dataset(sensitive_attr=sensitive_attr)
    data_class_df = in_process_pipline(data_class)

    # re-clustering
    clust = Cluster(df=data_class_df, lst=[], old_groups=[], fairness_metrics=[])
    k_values, silhouette_scores, optimal_k, PC1, PC2, cluster_kmeans = clust.funcs_pipline()

    # save results to Excel file
    data_class_df.to_excel(
        f'Store_Results/{Dataset.__name__}_{sensitive_attr}_output.xlsx',
        index=False
    )

    # plots
    plot_cluster_groups(df=data_class_df)
    plot_silhouette(k_values, silhouette_scores)
    plot_cluster_2dim(groups=cluster_kmeans, data_scaled=clust.lst, PC1=PC1, PC2=PC2, plot_type='else')
    plot_cluster_groups(df=data_class_df, col='New Group')



if __name__ == '__main__':
    for data_class, sensitive_attr in lst_of_data:
        print('-'*64)
        print(10*'-' + f'Starting: {data_class.__name__}, sensitive attribute: {sensitive_attr}' + '-'*10)
        print('-'*64)
        main(Dataset=data_class, sensitive_attr=sensitive_attr)
        print('-'*64)
        print(10*'-' + f'Done wit: {data_class.__name__}, sensitive attribute: {sensitive_attr}' + '-'*10)
        print('-'*64)


    # bank = Bank(sensitive_attr='age')
    # bank_df = in_process_pipline(bank)
    # print(bank_df)
    # clust = Cluster(df=bank_df)
    # k_values, silhouette_scores, optimal_k, PC1, PC2, cluster_kmeans = clust.funcs_pipline()
    # print(bank_df)
    # plot_cluster_groups(df=bank_df)
    # plot_silhouette(k_values, silhouette_scores)
    # plot_cluster_2dim(groups=cluster_kmeans, data_scaled=clust.data, PC1=PC1, PC2=PC2, plot_type='PCA')
    # plot_cluster_groups(df=bank_df, col='New Group')

    # compas = COMPAS(sensitive_attr='sex')
    # compas_df = in_process_pipline(compas)
    # clust = Cluster(df=compas_df)
    # k_values, silhouette_scores, optimal_k, PC1, PC2, cluster_kmeans = clust.funcs_pipline()
    # plot_cluster_groups(df=compas_df)
    # plot_silhouette(k_values, silhouette_scores)
    # plot_cluster_2dim(groups=cluster_kmeans, data_scaled=clust.data, PC1=PC1, PC2=PC2, plot_type='PCA')
    # plot_cluster_groups(df=compas_df, col='New Group')

    # compas = COMPAS(sensitive_attr='age')
    # compas_df = in_process_pipline(compas)
    # clust = Cluster(df=compas_df)
    # k_values, silhouette_scores, optimal_k, PC1, PC2, cluster_kmeans = clust.funcs_pipline()
    # print(compas_df)
    # plot_cluster_groups(df=compas_df)
    # plot_silhouette(k_values, silhouette_scores)
    # plot_cluster_2dim(groups=cluster_kmeans, data_scaled=clust.data, PC1=PC1, PC2=PC2, plot_type='PCA')
    # plot_cluster_groups(df=compas_df, col='New Group')

    # adult = Adult(sensitive_attr='race')
    # adult_df = in_process_pipline(adult)
    # clust = Cluster(df=adult_df)
    # k_values, silhouette_scores, optimal_k, PC1, PC2, cluster_kmeans = clust.funcs_pipline()
    # plot_cluster_groups(df=adult_df)
    # plot_silhouette(k_values, silhouette_scores)
    # plot_cluster_2dim(groups=cluster_kmeans, data_scaled=clust.data, PC1=PC1, PC2=PC2, plot_type='else')
    # plot_cluster_groups(df=adult_df, col='New Group')