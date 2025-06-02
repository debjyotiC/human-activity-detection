import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
file_path = "Raw_Data/HAR_mag_8x1 1.xlsx"
df = pd.read_excel(file_path)

# 2. Split X/Y and balance
X = df.drop(columns=['label'])
Y = df['label']
sm = SMOTE(random_state=2)
X_res, Y_res = sm.fit_resample(X, Y)

# 3. Map numeric labels to names
label_map = {0: 'active', 1: 'fall', 2: 'idle', 3: 'jerk'}
Y_named = Y_res.map(label_map)

# 4. Prepare Diff/no-Diff versions
if 'Diff' not in X_res.columns:
    raise ValueError("'Diff' column missing")
X_with = X_res.copy()
X_without = X_res.drop(columns=['Diff'])

# 5. Scale
scaler = StandardScaler()
Xw_scaled = scaler.fit_transform(X_with)
Xo_scaled = scaler.fit_transform(X_without)

# 6. UMAP embedding
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
emb_with = reducer.fit_transform(Xw_scaled)
emb_without = reducer.fit_transform(Xo_scaled)

# 7. Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(x=emb_with[:,0], y=emb_with[:,1],
                hue=Y_named, palette='Set1', ax=axes[0], s=50)
axes[0].set_title("UMAP with 'Diff' Feature")
axes[0].legend(title='Label', loc='best')

sns.scatterplot(x=emb_without[:,0], y=emb_without[:,1],
                hue=Y_named, palette='Set1', ax=axes[1], s=50)
axes[1].set_title("UMAP without 'Diff' Feature")
axes[1].legend(title='Label', loc='best')

plt.tight_layout()
plt.show()
