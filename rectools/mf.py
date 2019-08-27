"""Matrix factorization."""
from pathlib import Path
import numpy as np

class MFRecommender:
    """Matrix factorization recommender used for evaluation."""

    def __init__(self, user_factors, item_factors, item_norms):
        """Initialize an MFRecommender instance."""
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.item_norms = item_norms

    def __repr__(self):
        temp = f"Factors: {self.user_factors.shape[1]}\n"
        temp += f"Users: {self.user_factors.shape[0]}\n"
        temp += f"Items: {self.item_factors.shape[0]}"
        return temp

    @classmethod
    def load(_, model_folder):
        model_folder = Path(model_folder)
        assert model_folder.is_dir()

        user_factors = np.load(model_folder / "user_factors.npy")
        item_factors = np.load(model_folder / "item_factors.npy")

        item_norms = np.linalg.norm(item_factors, axis=1)

        return MFRecommender(user_factors, item_factors, item_norms)

    def recommend(self, idx, k=10):
        if isinstance(idx, int):
            idx = np.array([idx])

        scores = np.dot(self.item_factors, self.user_factors[idx].T)
        #return scores
        top_idxs = np.argpartition(scores, -k, axis=0)[-k:]
        # return top_idxs

        pairs = sorted(zip(top_idxs, scores[top_idxs]), key=lambda x: -x[1])
        return [(self.index2item[idx], score) for idx, score in pairs]
