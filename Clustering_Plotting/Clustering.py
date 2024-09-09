import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class Cluster:

    def __init__(self, df: pd.DataFrame, lst, old_groups, fairness_metrics):
        self.df = df
        self.old_groups = old_groups
        self.lst = lst
        self.fairness_metrics = fairness_metrics

    def _prepare_data(self):
        for i in self.df['Metric'].unique():

            self.fairness_metrics.append(i)

            vals = list(self.df['Normalized Differences'].loc[
                            self.df['Metric'] == i].values
                        )
            self.lst.append(vals)

            self.old_groups.append(
                self.df['Group'].loc[
                    self.df['Metric'] == i].values[0]
            )

        self.lst = np.array(self.lst)

    def _scaler(self):
        scaler = StandardScaler()
        self.lst = scaler.fit_transform(self.lst)

    def _silhouette_calculator(self):
        k_values = range(2, 11)
        silhouette_scores = []

        # KMeans Clustering
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters_kmeans = kmeans.fit_predict(self.lst)
            silhouette_avg = silhouette_score(self.lst, clusters_kmeans)
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters = {k}, the average silhouette score is: {silhouette_avg}")


        optimal_k = k_values[np.argmax(silhouette_scores)]

        return k_values, silhouette_scores, optimal_k

    def _PCA_decomposition(self):
        pca = PCA(n_components=2)  # We want the first two components for visualization
        principal_components = pca.fit_transform(self.lst)

        # principal_components now contains the transformed data along PC1 and PC2
        PC1 = principal_components[:, 0]
        PC2 = principal_components[:, 1]

        return PC1, PC2

    def final_clusters(self,optimal_k):
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters_kmeans = kmeans.fit_predict(self.lst)

        metrics_cluster = dict(
            zip(
                self.fairness_metrics, clusters_kmeans
            )
        )
        self.df['New Group'] = self.df['Metric'].apply(
            lambda x:
            metrics_cluster[x] if x in metrics_cluster.keys()
            else np.nan
        )
        return clusters_kmeans

    def funcs_pipline(self):
        self._prepare_data()
        self._scaler()
        k_values, silhouette_scores, optimal_k = self._silhouette_calculator()
        PC1, PC2 = self._PCA_decomposition()
        cluster_kmenas = self.final_clusters(optimal_k)
        return k_values, silhouette_scores, optimal_k, PC1, PC2, cluster_kmenas