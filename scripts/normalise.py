import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.load("X.npy")

N, S, F = X.shape

X_flat = X.reshape(-1, F)

# Identify real strokes BEFORE normalization
real_mask = np.any(X_flat != 0, axis=1)

print(f"Real strokes: {np.sum(real_mask)}")
print(f"Padding rows: {np.sum(~real_mask)}")

scaler = StandardScaler()

# Fit only on real strokes
scaler.fit(X_flat[real_mask])

# Copy so padding remains untouched
X_flat_norm = X_flat.copy()

# Normalize only real strokes
X_flat_norm[real_mask] = scaler.transform(X_flat[real_mask])

X_norm = X_flat_norm.reshape(N, S, F)

print("Normalized shape:", X_norm.shape)

np.save("X_norm.npy", X_norm)

# Save scaler statistics
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)

print("Saved normalized dataset.")