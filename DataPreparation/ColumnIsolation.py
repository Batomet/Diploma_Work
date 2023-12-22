import os

import pandas as pd

bvh_dir = "BvhToCsvConverter"
csv_dir = "DataPreparation/data"

for filename in os.listdir(bvh_dir):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(bvh_dir, filename))

        df_only_hands = df.iloc[:, 16:36].join(df.iloc[:, 39:69])

        print(f"{filename} shape:", df_only_hands.shape)

        df_only_hands.to_csv(os.path.join(csv_dir, filename.replace(".csv", "_17-35-40-59.csv")), index=True)
        print(f"Converted {filename} to {filename.replace('.csv', '_17-35-40-59.csv')}")
