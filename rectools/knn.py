# -*- coding: utf-8 -*-
"""Nearest neighbors."""

from pathlib import Path

import implicit
import numpy as np
import pandas as pd
import scipy.sparse
from rankmetrics.metrics import recall_k_curve
from sklearn.preprocessing import MultiLabelBinarizer

from .preprocess import read_data, sample_users


def train(input_folder, output_folder, k=1000):
    """
    Train model.

    Read training data from specified input folder and write to specified output folder.
    """
    print("Training")
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    train_file = input_folder / "train.csv"
    dev_file = input_folder / "dev.csv"
    users_file = input_folder / "users.csv"
    items_file = input_folder / "items.csv"

    recall_file = output_folder / "recall.csv"
    model_file = output_folder / "knn.npz"

    output_folder.mkdir(exist_ok=True)
    print("Loading users and items")
    user_ids = list(pd.read_csv(users_file, dtype=object)["user_id"])
    item_ids = list(pd.read_csv(items_file, dtype=object)["item_id"])
    print("")
    print("Loading ratings")
    u_x_i_train = read_data(train_file, user_ids, item_ids)
    i_x_u_train = u_x_i_train.T.tocsr()
    print(f" ◦ {i_x_u_train.shape[1]:8,} users in training data")
    print(f" ◦ {i_x_u_train.shape[0]:8,} items in training data")
    print(f" ◦ {u_x_i_train.sum(axis=0).mean():11,.2f} users per item on average")
    print(f" ◦ {u_x_i_train.sum(axis=1).mean():11,.2f} items per user on average")

    print("")
    print(f"Training kNN model with k={k}")
    model = implicit.nearest_neighbours.CosineRecommender(K=k)
    model.fit(i_x_u_train)
    print(
        f" ◦ {model.similarity.shape[0]:,} x {model.similarity.shape[1]:,} sim matrix"
    )
    print("")
    print("Saving model")
    scipy.sparse.save_npz(model_file, model.similarity)
    print("")
    print("Evaluating")
    print(" ◦ Loading evaluation data")
    u_x_i_dev = read_data(dev_file, user_ids, item_ids)  # .toarray()

    sample = sample_users(u_x_i_dev, len(user_ids))

    print(f" ◦ Scoring for {sample.shape[0]:,} users")
    knn_model = KnnModel(item_ids, model.similarity)
    u_x_i_sample = u_x_i_train[sample].toarray()
    knn_preds_sample = knn_model.score(u_x_i_sample)

    # Remove consumed items from predictions
    knn_preds_sample[u_x_i_sample.astype(bool)] = 0

    max_k = 1000
    print(f" ◦ Computing recall@k for k=1 to k={max_k}")
    # Convert to multiple hot representation (required by rank metrics)
    u_x_i_dev_sample = u_x_i_dev[sample, :]
    u_x_i_dev_sample = u_x_i_dev_sample.astype(bool).astype(int)
    knn_recall_curve = recall_k_curve(u_x_i_dev_sample, knn_preds_sample, max_k=max_k)
    k = min(10, max_k)
    print(f" ◦ recall@{k}: {100*knn_recall_curve[k-1]:.2f}")

    df_knn_recall = pd.DataFrame(
        data={"k": list(range(1, max_k + 1)), "recall_k": knn_recall_curve * 100}
    )
    df_knn_recall.to_csv(recall_file, index=False, sep=";", decimal=",")
    print("")


class KnnModel:
    """A simple recommender model based on K nearest neighbors."""

    def __init__(self, item_ids, sim_matrix):
        """
        Initialize a KnnModel instance.

        :param item_ids: A list of the item IDs known to the recommender. Used to map
                         to the similarity matrix.
        :param sim_matrix: The similarity matrix as as csr_matrix.
        """
        self._item_ids = item_ids
        self._item_binarizer = MultiLabelBinarizer(classes=self._item_ids)
        self._item_binarizer.fit([])
        self._sim_matrix = sim_matrix

    def recommend(self, item_ids, k, exclude_consumed):
        """
        Recommend for consumption history.

        :param item_ids: List of consumed item IDs.
        :param k: Number of items to return.
        :param exclude_consumed: A value indicating whether consumed items should be
                                 removed from the results.
        """
        encoding = self._item_binarizer.transform([item_ids])
        scores = self.score(encoding)
        if exclude_consumed:
            for i, x in enumerate(encoding[0]):
                if not x:
                    continue
                scores[0][i] = 0

        n_drop = self._sim_matrix.shape[0] - k
        top_indices = np.argpartition(scores, n_drop, axis=1)[:, n_drop:]
        top_ids = [self._item_ids[int(idx)] for idx in top_indices[0]]
        top_scores = [round(s, 5) for s in scores[0, top_indices][0]]
        return sorted(zip(top_ids, top_scores), key=lambda x: x[1], reverse=True)

    def score(self, encodings):
        """Compute similarity scores for specified u_x_i matrix."""
        scores = scipy.sparse.csc_matrix.dot(encodings, self._sim_matrix)
        return scores

    @staticmethod
    def load(items_file, similarity_file):
        """Create a new KnnModel instance by loading items and similarity file."""
        item_ids = list(pd.read_csv(items_file, dtype=object)["item_id"])
        sim = scipy.sparse.load_npz(similarity_file)
        model = KnnModel(item_ids, sim)
        return model
