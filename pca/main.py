import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

n_samples, n_features = X.shape


def main():

    # --- STANDARDIZING DATA ---

    """
    Effectively, we are moving the center of the data to the origin

    At the same time, we divide it by the std. dev. to ensure that
    a certain dimension does not dominate by virtue of a larger
    numerical range

    If we do not (normalize the data), axes later on would get picked
    first not because of their importance, but because of their scale.
    """

    # X has 150 samples w/ 4 features each

    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0, ddof=1)

    # Z = (X - μ) / σ
    X_std = (X - mean) / std_dev

    # --- CORRELATION MATRIX ---

    """
    This tells you how much each feature varies w.r.t each other feature

    The diagonal would comprise solely of 1s because each feature, w.r.t
    itself, would correlate perfectly.

    Each entry (i, j) in the matrix is the correlation between feature i
    and feature j. Thus, the shape is NUM_FEATURES x NUM_FEATURES, or
    4x4 in this case.
    """

    correlation_mat = (1 / (n_samples - 1)) * (X_std.T @ X_std)

    # --- PCA ---

    """We can skip the calculation of the covariance matrix, because when we standardize
    the original data prior to determining the correlation matrix, it's essentially
    the same as the correlation matrix
    """

    covariance_mat = correlation_mat

    # --- EIGENVALUES & EIGENVECTORS ---
    # https://youtu.be/PFDu9oVAE-g?feature=shared

    """
    Intuitively, eigenvectors are vectors who *remain* on their
    `span` even after a matrix transformation has been applied;
    they serve as the axes of the transformation.

    Each has a corresponding eigenvalue, that trivially tells you
    the magnitude of the stretch/squish applied to the eigenvector
    in question as a result of undergoing the matrix transformation.

    ---

    Thus, when applying the same concept to a covariance matrix
    (because it is just a matrix transformation at the end of
    the day) we see that the output eigenvector, which tells the
    directions of maximum variance in the data corresponds to a
    vector whose direction remains unchanged in some generic
    matrix transformation.

    In the same vein, the scaling factor i.e eigenvalue, in a generic
    matrix transformation, corresponds to the magnitude of variance
    along the aforementioned directions.

    Takeaway: cov mat eigenvec = direction of spread in data
    """

    eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)

    # Sorting
    eigenpairs = sorted(
        zip(eigenvalues, eigenvectors.T), key=lambda x: x[0], reverse=True
    )

    # Unzip back into separate arrays
    sorted_eigenvalues, sorted_eigenvectors_T = zip(*eigenpairs)
    sorted_eigenvectors = np.array(sorted_eigenvectors_T).T
    sorted_eigenvalues = np.array(sorted_eigenvalues)

    # Select top 2 principal components i.e eigenvectors
    k = 2
    top_k_eigenvectors = sorted_eigenvectors[:, :k]

    # Project the data onto the top `k` principal components
    # thereby effectively maintaining the corresponding % of data.
    X_pca = X_std @ top_k_eigenvectors

    # --- EXPLAINED VARIANCE RATIO ---

    explained_var_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cum_explained_var = np.cumsum(explained_var_ratio)

    print(f"{explained_var_ratio} -- Explained variance ratio per component")
    print(f"{cum_explained_var} -- Cumulative explained variance")
    print()
    print(f"Variance retained by top {k} components -- {cum_explained_var[k-1]:.2f}")


if __name__ == "__main__":
    main()
