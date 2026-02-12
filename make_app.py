import pandas as pd
import numpy as np
import umap
from molfeat.trans.fp import FPVecTransformer  
from sklearn.decomposition import IncrementalPCA

# 1. Load the data
print("Loading large dataset...")
try:
    df = pd.read_csv("large_subset.csv")
    if len(df) > 50000:
        print("Subsampling to 50k...")
        df = df.sample(50000, random_state=42)
except:
    print("Large file not found, loading Medium...")
    df = pd.read_csv("medium_subset.csv")

df = df[df['smiles'] != "_null_"].dropna(subset=['smiles']).reset_index(drop=True)

# 2. Find the score column
score_col = 'dock_score' if 'dock_score' in df.columns else 'score'

# 3. Generate Fingerprints
print("Generating fingerprints...")
featurizer = FPVecTransformer(kind="ecfp", length=1024, radius=2)
fps = np.array(featurizer(df['smiles'].tolist()))

# 4. Run PCA + UMAP
print("Running PCA...")
pca = IncrementalPCA(n_components=50).fit_transform(fps)

print("Running UMAP...")
mapper = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
coords = mapper.fit_transform(pca)

# 5. Save for App
df['UMAP_1'] = coords[:, 0]
df['UMAP_2'] = coords[:, 1]
df['label'] = df[score_col]
final_df = df[['smiles', 'label', 'UMAP_1', 'UMAP_2']]

print("Saving 'for_app.csv'...")
final_df.to_csv("for_app.csv", index=False)
print("DONE! Now download 'for_app.csv'")
