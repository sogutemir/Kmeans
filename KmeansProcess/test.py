import pandas as pd
from sklearn.cluster import KMeans
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np

# Veri setini yükleme
df = pd.read_csv('data\PFG.csv')  # Veri setinin yolunu belirtin
X = df.iloc[:, [0, 1]].values  # İlk iki sütunu alıyoruz

# Elbow Method ile optimal küme sayısını belirleme
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Elbow grafiğini çizme
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Optimal küme sayısını belirleyin (örneğin, grafiğe bakarak 3 olduğunu varsayalım)
optimal_clusters = 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Karar sınırlarını çizme
plt.figure(figsize=(10, 8))
plot_decision_regions(X, y_kmeans, clf=kmeans, legend=1)

# Küme merkezlerini çizme ve etiketleme
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.6, marker='o')
for i, c in enumerate(centers):
    plt.text(c[0], c[1], s=f'Center {i}', color='white', va='center', ha='center')

plt.title('K-Means Clustering with Decision Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
