from pathlib import Path
import faiss
import numpy as np
import pandas as pd

class VectorIndex:
    def __init__(self, folder: str = "data/vec_index"):
        self.folder = Path(folder)
        self.index = faiss.read_index(str(self.folder / "index.faiss"))
        self.meta = pd.read_parquet(self.folder / "meta.parquet")

    def search(self, query_emb: np.ndarray, k: int = 5):
        D, I = self.index.search(query_emb.astype("float32"), k)
        rows = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            row = self.meta.iloc[idx].to_dict()
            row["score"] = float(dist)
            rows.append(row)
        return rows