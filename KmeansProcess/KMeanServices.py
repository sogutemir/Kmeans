import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import ConvexHull
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import os
import textwrap

class KMeansService:
    def __init__(self, n_clusters=3, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit_predict(self, df):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        df['cluster'] = kmeans.fit_predict(df.drop(columns='cluster', errors='ignore'))
        self.centers = kmeans.cluster_centers_
        return df

    def plot_clusters(self, df):
        cluster_colors = {0: 'red', 1: 'green', 2: 'blue'}
        
        plt.figure(figsize=(15, 10))
        
        sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue='cluster',
                        palette=cluster_colors, alpha=0.7, style='cluster', markers=['o', '^', 's'], s=50)
        
        plt.scatter(self.centers[:, 0], self.centers[:, 1], c='black', marker='X', s=200, label='Centroids')
        
        for i in range(self.n_clusters):
            points = df[df['cluster'] == i][df.columns[:2]].values
            if len(points) > 2:
                hull = ConvexHull(points)
                x_hull = np.append(points[hull.vertices,0], points[hull.vertices,0][0])
                y_hull = np.append(points[hull.vertices,1], points[hull.vertices,1][0])
                plt.fill(x_hull, y_hull, alpha=0.3, c=cluster_colors[i])

        plt.legend(title='Cluster', loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.title("Cluster plot with Convex Hull")

        if not os.path.exists('output'):
            os.makedirs('output')

        # Milisaniye çözünürlüğünde bir zaman damgası kullanarak benzersiz bir dosya adı oluşturun.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]  # Son üç karakteri (mikrosaniyeler) kesiyoruz.
        filename = f'output/cluster_plot_{timestamp}.png'

        plt.savefig(filename, bbox_inches='tight')

        # Şimdi grafiği göster
        plt.show()

