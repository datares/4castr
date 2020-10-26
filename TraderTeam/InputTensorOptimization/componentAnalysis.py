from sklearn.decomposition import PCA
from TraderTeam.InputTensorOptimization.redoRawData import testData
from TraderTeam.InputTensorOptimization.featureEngineering import z_score, makeFilter

# PCA is a dimensionality reduction process that does SVD M(M^t), where M = (covariance matrix) of the data.
# The first n principal components of the SVD are given by the eigenvectors
# corresponding to the n largest singular values

window = 200

rawData = z_score(testData.drop(columns='Date'), window)

cleanedData = makeFilter(testData)
cleanedData.drop(columns=["Date", "Open", "High", "Low", "Close", "Volume"], inplace=True)
adjustedData = z_score(cleanedData, window)

components = 2
pca1 = PCA(n_components=components)
pca1.fit(rawData)
pca2 = PCA(n_components=components)
pca2.fit(adjustedData)


if __name__ == "__main__":
    var1 = pca1.explained_variance_
    var2 = pca2.explained_variance_
    # how much of the variance in the data each principal component explains
    print(f"raw data explained variance: {var1 / sum(var1)}, ta data explained variance : {var2 / sum(var2)}")
    print(f"raw singular values, {pca1.singular_values_}, ta data singular values {pca2.explained_variance_}")

    results1 = pca1.score(rawData)
    results2 = pca2.score(adjustedData)
    # log likelihod
    print(f"raw score {results1}, ta score {results2}")
