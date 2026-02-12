from molfeat.trans.fp import FPVecTransformer
import pandas as pd
import numpy as np
import umap
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
from sklearn.decomposition import PCA
import colorcet

# --- Config ---
INPUT_FILE = "small_subset.csv"  
n_pca_components = 50          

# n_neighbors (Attraction): Low = precise local clusters, High = better global shape
neighbor_list = [5, 15, 50]
# min_dist (Repulsion): Low = tight clumps, High = spread out
dist_list = [0.0, 0.1, 0.5]

def get_fingerprints(df):
    print("Featurizing molecules (ECFP4)...")
    featurizer = FPVecTransformer(kind="ecfp", length=1024, radius=2)
    return np.array(featurizer(df['smiles'].tolist()))

def save_plot(embedding, title, filename):
    df_viz = pd.DataFrame(embedding, columns=['x', 'y'])

    # Render
    canvas = ds.Canvas(plot_width=600, plot_height=600)
    agg = canvas.points(df_viz, 'x', 'y')
    img = tf.shade(agg, cmap=colorcet.fire, how='log')

    # Save
    export_image(img, filename, background="black")
    print(f"   -> Saved plot: {filename}.png")

def main():
    # 1. Load Data
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 2. Pre-process (Features + PCA)
    # We do this ONCE so the loop is fast
    fps = get_fingerprints(df)

    print(f"Running PCA (1024 -> {n_pca_components} dims)...")
    pca = PCA(n_components=n_pca_components)
    pca_features = pca.fit_transform(fps)

    # 3. The Hyperparameter Loop
    print("\n--- Starting Hyperparameter Search ---")

    for neighbors in neighbor_list:
        for dist in dist_list:
            print(f"Running UMAP: neighbors={neighbors}, min_dist={dist}...")

            reducer = umap.UMAP(
                n_neighbors=neighbors,
                min_dist=dist,
                n_components=2,
                random_state=42,
                n_jobs=1
            )
            embedding = reducer.fit_transform(pca_features)

            # Save the image
            fname = f"umap_n{neighbors}_d{dist}"
            save_plot(embedding, f"n={neighbors} d={dist}", fname)

    print("Search complete!")

if __name__ == "__main__":
    main()
