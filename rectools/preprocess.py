"""Functions for preprocessing and loading data."""
import logging
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def read_data(filename, users, items, sep=","):
    """
    Read a CSV file with implicit or explicit user ratings and return a ratings matrix.

    The CSV file must contain three columns, one for each of the following: users, items
    and ratings. In addition to the CSV file, two lists must be specified. These should
    contain user IDs and item IDs repectively, and the positions of users and items in
    these lists will define the indices of the ratings matrix.
    """
    df = pd.read_csv(
        filename,
        usecols=[0, 1, 2],
        names=["user", "item", "implicit"],
        header=0,
        dtype=object,
        sep=sep,
    )

    df["user"] = pd.Series(pd.Categorical(df["user"], categories=users, ordered=False))
    df["item"] = pd.Series(pd.Categorical(df["item"], categories=items, ordered=False))

    # Remove rows for which the user or item ID is unknown
    df = df[~pd.isnull(df["user"])]
    df = df[~pd.isnull(df["item"])]

    matrix = coo_matrix(
        (
            df["implicit"].astype(float),
            (df["user"].cat.codes.copy(), df["item"].cat.codes.copy()),
        ),
        shape=(len(users), len(items)),
    )
    return matrix.tocsr()


# TODO: Add seed


def sample_users(u_x_i, sample_size):
    """Sample a set of active users and return their indices."""
    active_mask = u_x_i.sum(axis=1).A1 > 0
    print(
        f"Users in test set: {u_x_i.shape[0]} (total), {np.sum(active_mask)} (active)"
    )
    active_indices = np.where(active_mask)[0]
    sample_indices = np.random.choice(
        active_indices, min(len(active_indices), sample_size), replace=False
    )
    return sample_indices
