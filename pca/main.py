# Necessary Imports
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data  # Feature matrix
y = iris.target  # Target labels (species)

feature_names = iris.feature_names
target_names = iris.target_names

n_samples, n_features = X.shape


def main():

    # --- STANDARDIZING DATA ---

    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0, ddof=1)

    # Z = (X - μ) / σ
    X_std = (X - mean) / std_dev

    # --- CORRELATION MATRIX ---

    correlation_mat = (1 / (n_samples - 1)) * (X_std.T @ X_std)

    # --- PCA ---

    # We can skip the calculation of the covariance matrix, because when we standardize
    # the original data prior to determining the correlation matrix, it's essentially
    # the same as the correlation matrix

    covariance_mat = correlation_mat

    # --- EIGENVALUES & EIGENVECTORS ---

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
