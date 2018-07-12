/* 1.	 Standardize the d -dimensional dataset.
2.	 Construct the covariance matrix.
3.	 Decompose the covariance matrix into its eigenvectors and eigenvalues.
4.	 Select k eigenvectors that correspond to the k largest eigenvalues,
where k is the dimensionality of the new feature subspace ( k ≤ d ).
5.	 Construct a projection matrix W from the "top" k eigenvectors.
6.	 Transform the d -dimensional input dataset X using the projection
matrix W to obtain the new k -dimensional feature subspace. */

const svd = require('node-svd').svd;
const math = require('mathjs');

class PCA {
  constructor(nComponents = 0) {
    this.nComponents = nComponents;
  }

  fit(dataset) {
    // Standardize the d -dimensional dataset.
    this.dataset = dataset;
    this.mean = math.mean(this.dataset);
    this.std = math.std(this.dataset);
    this.scaledData = this.dataset.map(i =>
      i.map(j => (j - this.mean) / this.std)
    );
    this.scaledDataT = math.transpose(this.scaledData);

    //Get eigenVecs & eigenVals
    this.eigVecs = svd(this.scaledDataT).U;
    this.eigVals = svd(this.scaledDataT).S;
    this.total = math.sum(this.eigVals);

    this.revSortEigVal = this.eigVals.sort((a, b) => a < b);
    this.explainedVarianceRatio = this.revSortEigVal.map(
      x => (x * 100) / this.total
    );
    this.explainedVarianceRatio = this.explainedVarianceRatio.slice(
      0,
      this.nComponents
    );

    this.eigPairs = [];
    for (let i = 0; i < this.eigVals.length; i++) {
      this.eigPairs.push([
        math.abs(this.eigVals[i]),
        math.transpose(this.eigVecs)[i]
      ]);
    }
    this.wmatrix = math.transpose(this.eigVecs);
    console.log(this.wmatrix);
  }

  transform(data) {
    this.data = data;
    return math.dot(this.data, this.wmatrix);
  }
}
