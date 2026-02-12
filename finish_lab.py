import pandas as pd
import numpy as np
import umap
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
from molfeat.trans.fp import FPVecTransformer
from sklearn.decomposition import IncrementalPCA
from autogluon.tabular import TabularPredictor
import colorcet
import sys
import shutil
import os

# --- CONFIG ---
TRAIN_FILE = "medium_subset.csv"
TEST_FILE = "large_subset.csv"
BATCH_SIZE = 20000

def get_fingerprints(smiles_list):
    trans = FPVecTransformer(kind="ecfp", length=1024, radius=2)
    try: return np.array(trans(smiles_list))
    except: return np.zeros((len(smiles_list), 1024))

def save_plot(df, col, name, cmap):
    print(f"   -> Plotting {name}...")
    cvs = ds.Canvas(plot_width=800, plot_height=800)
    agg = cvs.points(df, 'x', 'y', ds.mean(col))
    img = tf.shade(agg, cmap=cmap)
    export_image(img, name, background="black")

def find_score_column(df, filename):
    """Finds 'dock_score', 'score', or 'docking_score' automatically"""
    if 'dock_score' in df.columns: return 'dock_score'

    candidates = [c for c in df.columns if 'score' in c.lower() or 'dock' in c.lower()]
    if candidates:
        print(f"   (File {filename}: using column '{candidates[0]}')")
        return candidates[0]

    print(f"❌ ERROR: No score column found in {filename}!")
    sys.exit(1)

def main():

    # Load Training Data
    df_train = pd.read_csv(TRAIN_FILE)
    df_train = df_train[df_train['smiles'] != "_null_"].dropna(subset=['smiles'])

    train_score_col = find_score_column(df_train, TRAIN_FILE)

    # Generate features
    print("   Generating training features...")
    fps = get_fingerprints(df_train['smiles'].tolist())
    feat_df = pd.DataFrame(fps, columns=[f"fp_{i}" for i in range(1024)])
    feat_df['log_score'] = df_train[train_score_col]

    # Train quickly (2 min limit)
    print("   Training model (Limit: 120s)...")
    predictor = TabularPredictor(label='log_score', path='temp_ag_fix').fit(
        feat_df, time_limit=120, presets='medium_quality', verbosity=0
    )

    print("\n--- 2. Re-running UMAP for Large Dataset ---")
    ipca = IncrementalPCA(n_components=50)
    print("   Fitting PCA on Medium subset...")
    ipca.fit(fps)

    print("   Fitting UMAP on Medium subset...")
    mapper = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(ipca.transform(fps))

    print("\n--- 3. Processing Large Dataset (Predictions + UMAP) ---")
    df_test = pd.read_csv(TEST_FILE)
    df_test = df_test[df_test['smiles'] != "_null_"].dropna(subset=['smiles']).reset_index(drop=True)

    test_score_col = find_score_column(df_test, TEST_FILE)

    all_preds = []
    all_coords = []

    print(f"   Processing {len(df_test)} molecules in batches...")
    for i in range(0, len(df_test), BATCH_SIZE):
        if i % 100000 == 0: print(f"      Batch {i}...")
        batch = df_test.iloc[i:i+BATCH_SIZE]

        # Features
        batch_fps = get_fingerprints(batch['smiles'].tolist())

        # A. Predict Score
        batch_feat = pd.DataFrame(batch_fps, columns=[f"fp_{k}" for k in range(1024)])
        preds = predictor.predict(batch_feat)
        all_preds.extend(preds)

        # B. Get UMAP Coords
        pca_batch = ipca.transform(batch_fps)
        umap_batch = mapper.transform(pca_batch)
        all_coords.append(umap_batch)

    # Save Results
    coords = np.vstack(all_coords)
    df_test['x'] = coords[:, 0]
    df_test['y'] = coords[:, 1]
    df_test['pred_score'] = all_preds

    # Calculate Residuals
    df_test['residual'] = (df_test[test_score_col] - df_test['pred_score']).abs()

    print("\n--- 4. Saving Final Images ---")
    save_plot(df_test, test_score_col, "final_actual", colorcet.fire)
    save_plot(df_test, 'pred_score', "final_predicted", colorcet.fire)
    save_plot(df_test, 'residual', "final_residuals", colorcet.coolwarm)

    # Cleanup
    shutil.rmtree('temp_ag_fix', ignore_errors=True)
    print("You now have 'final_actual.png', 'final_predicted.png', and 'final_residuals.png'.")

if __name__ == "__main__":
    main()
