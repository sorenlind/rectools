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


def train_dev_split(df, seed, n_dev=1000):
    """
    Create train/dev split.

    Sample n_dev random users. Then, for each of those users, sample one random
    observation. Those observations will be the dev set. Everything else will be the
    training set.
    """
    np.random.seed(seed)

    # Sample n_dev users
    grouped = df.groupby("user_id")
    group_ids = np.arange(grouped.ngroups)
    np.random.shuffle(group_ids)
    group_ids = group_ids[:n_dev]
    df_sample_u = df[grouped.ngroup().isin(group_ids)]

    # Sample exactly one document from every sampled user
    grouped = (
        df_sample_u.groupby("user_id").apply(lambda x: x.sample(1)).reset_index(level=1)
    )

    # Create dev mask
    dev_indices = list(grouped["level_1"])
    mask = np.zeros(df.shape[0]).astype(bool)
    mask[dev_indices] = 1

    # Apply mask
    df_train = df.iloc[~mask].copy()
    df_dev = df.iloc[mask].copy()

    return df_train, df_dev
