from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


X, label = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=5, random_state=123)
X = StandardScaler().fit_transform(X)

###########################################################################
#DBSCAN
###########################################################################

def dbscannn():
    eps_value = 0.2
    min_samples_value = 5
    db = DBSCAN(eps=eps_value, min_samples=min_samples_value).fit(X)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    labels = db.labels_

    df['cluster'] = labels 

    unique_clusters = set(labels)
    cluster_counts = df['cluster'].value_counts()

    n_clusters = len(unique_clusters) - (1 if -1 in labels else 0)

    mask = labels != -1
    if len(set(labels[mask])) > 1:
        score = silhouette_score(X[mask], labels[mask])
        print("Silhouette Score:", score)
    else:
        score = 0
        print("Silhouette Score: 0")

    print (
        f"Clusters encontrados: {unique_clusters}\n"
        f"Número de clusters (sin ruido): {n_clusters}\n\n"
        f"{cluster_counts.to_string()}"

    )

####################PLOT########################

    fig, (ax_plot, ax_text) = plt.subplots(
        2, 1,
        figsize=(10,6),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    unique_clusters_no_noise = [c for c in unique_clusters if c != -1]

    for cluster_id in unique_clusters_no_noise:
        cluster_points = X[labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        
        #centroide estrella
        ax_plot.scatter(
            centroid[0],
            centroid[1],
            marker='*',
            s=350,
            edgecolor='black',
            linewidth=1.5
        )
        
        # Etiqueta de cluster
        ax_plot.text(
            centroid[0],
            centroid[1] + 0.15,
            f"Cluster {cluster_id}",
            fontsize=11,
            fontweight='bold',
            ha='center'
        )

    ax_plot.scatter(X[:,0], X[:,1], c=labels)
    ax_plot.set_xlabel("PC1")
    ax_plot.set_ylabel("PC2")
    ax_plot.set_title(f"DBSCAN (eps={eps_value}, min_samples={min_samples_value})")

    ax_plot.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax_plot.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax_plot.grid(True)

    '''
    info_text = (
        f"Clusters encontrados: {unique_clusters}\n"
        f"Número de clusters (sin ruido): {n_clusters}\n\n"
        f"{cluster_counts.to_string()}\n\n"
        f"Silhouette Score:, {score}"
    )
    '''

    info_text = (
        f"Silhouette Score:, {score}"
    )

    ax_text.axis("off")
    ax_text.text(0.01, 0.95, info_text, va='top', fontsize=10)

    plt.tight_layout()
    plt.show()




###########################################################################
#keimins
###########################################################################


def keimins():
    best_k = 0
    best_score = -1

    for k in range(2, 11):  
        kmeans = KMeans(n_clusters=k, random_state=123)
        labels = kmeans.fit_predict(X)
        
        score = silhouette_score(X, labels)
        
        if score > best_score:
            best_score = score
            best_k = k

    print(f"Mejor k: {best_k}")
    print(f"Mejor silhouette score: {best_score}")

    k = best_k

    kmeans = KMeans(n_clusters=k, random_state=123)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    fig, (ax_plot, ax_text) = plt.subplots(
        2, 1,
        figsize=(10,6),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Scatter de datos
    ax_plot.scatter(X[:, 0], X[:, 1], c=labels)

    # Centroides
    ax_plot.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300)

    ax_plot.set_xlabel("x1")
    ax_plot.set_ylabel("x2")
    ax_plot.set_title(f"K-means con k={k}")
    ax_plot.grid(True)

    info_text = (
        f"Mejor k: {best_k}\n"
        f"Mejor silhouette score: {best_score:.4f}"
    )

    ax_text.axis("off")
    ax_text.text(0.01, 0.95, info_text, va='top', fontsize=11)

    plt.tight_layout()
    plt.show()


###########################################################################
#aglomerado :P blehhh
###########################################################################

def aglomeradoblehh():
    k = 3

    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    print(f"k usado: {k}")
    print(f"Silhouette score: {score}")

    fig, (ax_plot, ax_text) = plt.subplots(
        2, 1,
        figsize=(10,6),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # scatter
    ax_plot.scatter(X[:, 0], X[:, 1], c=labels)

    ax_plot.set_xlabel("x1")
    ax_plot.set_ylabel("x2")
    ax_plot.set_title(f"Agglomerative Clustering con k={k}")
    ax_plot.grid(True)

    info_text = (
        f"Mejor k: {k}\n"
        f"Mejor silhouette score: {score:.4f}"
    )

    ax_text.axis("off")
    ax_text.text(0.01, 0.95, info_text, va='top', fontsize=11)

    plt.tight_layout()
    plt.show()





#################################################################
#################################################################
#################################################################


while True:
    print("1. DBSCAN\n2. K-means\n3. Agglomerative\n")
    option = int(input())

    match option:
        case 1:
            dbscannn()
        case 2:
            keimins()
        case 3:
            aglomeradoblehh()
        case _:
            print("Opción inválida")

    condition = input("Deseas hacer otro plot? (y/n): ")
    if condition.lower() != 'y':
        break
