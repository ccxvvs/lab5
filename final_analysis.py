from molfeat.trans.fp import FPVecTransformer
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import IncrementalPCA
from autogluon.tabular import TabularPredictor
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
import colorcet
import gc
import sys

# --- CONFIGURATION ---
TRAIN_FILE = "medium_subset.csv" # Train map on Medium (100k)
TARGET_FILE = "large_subset.csv" # Visualize Large (1M)
MODEL_PATH = "ag_models_lsd"
BATCH_SIZE = 20000

BEST_NEIGHBORS = 15
BEST_DIST = 0.1

def get_fingerprints(smiles_list):
    featurizer = FPVecTransformer(kind="ecfp", length=1024, radius=2)
    try:
        return np.array(featurizer(smiles_list))
    except:
        return np.zeros((len(smiles_list), 1024))

def save_plot(df, column, title, filename, cmap):
    print(f"   -> Drawing {title} (using col: {column})...")
    canvas = ds.Canvas(plot_width=800, plot_height=800)
    agg = canvas.points(df, 'x', 'y', ds.mean(column))
    img = tf.shade(agg, cmap=cmap)
    export_image(img, filename, background="black")

def find_score_column(df):
    """Automatically finds the docking score column"""
    if 'dock_score' in df.columns: return 'dock_score'

    # Search for likely candidates
    candidates = [c for c in df.columns if 'score' in c.lower() or 'dock' in c.lower()]

    if candidates:
        print(f"⚠️ 'dock_score' not found. Auto-detected replacement: '{candidates[0]}'")
        return candidates[0]

    print(f"❌ ERROR: No score column found! Columns are: {list(df.columns)}")
    sys.exit(1) 

def main():
    print("--- STEP 1: Training the Map (Using Medium Subset) ---")

    # 1. Load Training Data
    df_train = pd.read_csv(TRAIN_FILE)
    df_train = df_train[df_train['smiles'] != "_null_"].dropna(subset=['smiles'])
    print(f"Loaded {len(df_train)} training molecules.")

    # 2. Fit PCA
    print("Fitting PCA on Medium subset...")
    ipca = IncrementalPCA(n_components=50)
    for i in range(0, len(df_train), BATCH_SIZE):
        batch = df_train.iloc[i:i+BATCH_SIZE]['smiles'].tolist()
        fps = get_fingerprints(batch)
        ipca.partial_fit(fps)

    # Transform Training Data for UMAP
    train_pca_list = []
    for i in range(0, len(df_train), BATCH_SIZE):
        batch = df_train.iloc[i:i+BATCH_SIZE]['smiles'].tolist()
        fps = get_fingerprints(batch)
        train_pca_list.append(ipca.transform(fps))
    train_pca = np.vstack(train_pca_list)

    # 3. Fit UMAP
    print(f"Fitting UMAP (n={BEST_NEIGHBORS}, d={BEST_DIST})...")
    mapper = umap.UMAP(
        n_neighbors=BEST_NEIGHBORS,
        min_dist=BEST_DIST,
        n_components=2,
        random_state=42,
        n_jobs=1
    ).fit(train_pca)

    del df_train, train_pca, train_pca_list
    gc.collect()
    print("Map trained!")

    # ---------------------------------------------------------
    print("\n--- STEP 2: Processing Large Dataset (1 Million Rows) ---")
    df_target = pd.read_csv(TARGET_FILE)

    # Clean Data
    df_target = df_target[df_target['smiles'] != "_null_"].dropna(subset=['smiles'])
    df_target = df_target.reset_index(drop=True)
    print(f"Loaded {len(df_target)} molecules.")

    score_col = find_score_column(df_target)

    # 4. Project Large Data
    all_coords = []
    for i in range(0, len(df_target), BATCH_SIZE):
        if i % 100000 == 0: print(f"   Projecting batch {i}/{len(df_target)}...")
        batch = df_target.iloc[i:i+BATCH_SIZE]['smiles'].tolist()
        fps = get_fingerprints(batch)
        pca_batch = ipca.transform(fps)
        umap_batch = mapper.transform(pca_batch)
        all_coords.append(umap_batch)

    coords = np.vstack(all_coords)
    df_target['x'] = coords[:, 0]
    df_target['y'] = coords[:, 1]

    # 5. Get Model Predictions
    print("\n--- STEP 3: Generating Predictions ---")
    try:
        predictor = TabularPredictor.load(MODEL_PATH)
        all_preds = []

        for i in range(0, len(df_target), BATCH_SIZE):
            if i % 100000 == 0: print(f"   Predicting batch {i}/{len(df_target)}...")
            batch_df = df_target.iloc[i : i + BATCH_SIZE].copy()
            fps = get_fingerprints(batch_df['smiles'].tolist())
            feat_cols = [f"fp_{k}" for k in range(1024)]
            feat_df = pd.DataFrame(fps, columns=feat_cols, index=batch_df.index)
            preds = predictor.predict(feat_df)
            all_preds.extend(preds)

        df_target['pred_score'] = all_preds

        # Calculate Residuals (Using the detected score column)
        min_score = df_target[score_col].min()
        df_target['actual_log'] = np.log(df_target[score_col] - min_score + 1)
        df_target['residual'] = (df_target['actual_log'] - df_target['pred_score']).abs()

    except Exception as e:
        print(f"⚠️ Model Error: {e}")
        df_target['pred_score'] = 0
        df_target['residual'] = 0

    # 6. Visualize
    print("\n--- STEP 4: Saving Plots ---")
    save_plot(df_target, score_col, "Actual Score (Large)", "viz_large_actual", colorcet.fire)
    save_plot(df_target, 'pred_score', "Predicted Score (Large)", "viz_large_predicted", colorcet.fire)
    save_plot(df_target, 'residual', "Residuals (Large)", "viz_large_residuals", colorcet.coolwarm)

    print("Success!")

if __name__ == "__main__":
    main()
