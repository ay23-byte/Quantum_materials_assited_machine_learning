from jarvis.db.figshare import data
import pandas as pd

print("Downloading JARVIS-DFT 3D dataset...")

dft_3d = data(dataset="dft_3d")

print(f"Number of materials: {len(dft_3d)}")

df = pd.DataFrame(dft_3d)

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())