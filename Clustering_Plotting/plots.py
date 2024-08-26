import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_cluster_groups(df, col= 'Group'):

    for group in sorted(df[col].unique()):
        sns.set(style="whitegrid")

        # Create the plot
        plt.figure(figsize=(14, 8))
        filtered_df = df[df[col] == group]

        for metric in filtered_df['Metric'].unique():
            plt.plot(filtered_df['Model'].unique(),
                     filtered_df['Normalized Differences'][filtered_df['Metric'] == metric],
                     marker='o', label=metric)

        # Add title and labels
        plt.title(f'Normalized Differences for Group {group} across Different Models')
        plt.xlabel('Models')
        plt.ylabel('Normalized Differences')
        plt.legend(title='Fairness Metrics')
        plt.xticks(df['Model'].unique())
        plt.grid(True)
        plt.show()

def plot_silhouette(k_values, silhouette_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters k')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Method for Optimal Number of Clusters')
    plt.grid(True)
    plt.show()

def plot_cluster_2dim(groups, data_scaled, PC1,PC2,plot_type='PCA'):

    # Scatter plot of the clusters
    plt.figure(figsize=(8, 6))

    if plot_type == 'PCA':
        for cluster in np.unique(groups):
            plt.scatter(PC1[groups == cluster],
                        PC2[groups == cluster],
                        label=f'Cluster {cluster}')
    else:
        for cluster in np.unique(groups):
            plt.scatter(data_scaled[groups == cluster, 0],
                        data_scaled[groups == cluster, 1],
                        label=f'Cluster {cluster}')

    plt.title('K-Means Clustering of Fairness Metrics On Reduced Dimensions')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()


