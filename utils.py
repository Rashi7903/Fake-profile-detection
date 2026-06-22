import pandas as pd

def add_derived_features(X):
    """
    Feature Engineering: Computes the Follower/Following ratio.
    Needs to be in a separate module to ensure stable pickling/unpickling across different scripts.
    """
    X = X.copy()
    X['follower_following_ratio'] = X['num_followers'] / (X['num_following'] + 1)
    return X
