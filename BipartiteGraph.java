public class BipartiteGraph {
	public int[] rowSum;
	public int[] colSum;

	public BipartiteGraph(int[] rows, int[] columns) {
		rows = rowSum;
		columns = colSum;

	}

	public int ElementsSum(int[] intlist) {
		int sum = 0;
		for (int i = 0; i < intlist.length; i++) {
			sum = sum + intlist[i];
		}
		return sum;
	}

	public int[][] sortRowSum(int[] rowSum) {
		int temp, temp2, n = rowSum.length;
		int[] indexStorer = new int[n];

		for (int k = 0; k < n; k++) {
			indexStorer[k] = k;
		}

		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				if (rowSum[i] < rowSum[j]) {
					temp = rowSum[i];
					rowSum[i] = rowSum[j];
					rowSum[j] = temp;
					temp2 = indexStorer[i];
					indexStorer[i] = indexStorer[j];
					indexStorer[j] = temp2;
				}
			}
		}
		int[][] both = { rowSum, indexStorer };
		return both;
	}

	public int[][] sortColSum(int[] colSum) {
		int temp, temp2, n = colSum.length;
		int[] indexStorer = new int[n];

		for (int k = 0; k < n; k++) {
			indexStorer[k] = k;
		}

		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				if (colSum[i] < colSum[j]) {
					temp = colSum[i];
					colSum[i] = colSum[j];
					colSum[j] = temp;
					temp2 = indexStorer[i];
					indexStorer[i] = indexStorer[j];
					indexStorer[j] = temp2;
				}
			}
		}
		int[][] both = { colSum, indexStorer };
		return both;
	}

	public int[] conjugateRowVector(int[] rowSum) {
		int n = rowSum.length;
		int[] Rstar = new int[n];
		for (int j = 0; j < n; j++) {
			for (int i = 0; i < n; i++) {
				if (j + 1 <= rowSum[i]) {
					Rstar[j]++;
				}
			}
		}
		return Rstar;
	}

	public int existMatrix(int[] rowSum, int[] colSum) {
		int rowelementssum = ElementsSum(rowSum);
		int colelementssum = ElementsSum(colSum);
		int[] RStar = conjugateRowVector(rowSum);
		int[] sortedRow = sortRowSum(rowSum)[0];
		int[] sortedCol = sortColSum(colSum)[0];

		if (sortedRow[0] <= rowSum.length && sortedCol[0] <= colSum.length) {
			if (rowelementssum == colelementssum) {
				int sumCol = 0, sumRstar = 0;
				if (sumCol <= sumRstar) {
					for (int i = 0; i < rowSum.length; i++) {
						sumCol = sumCol + colSum[i];
						sumRstar = sumRstar + RStar[i];
					}
					return 1;
				} else
					return 0;
			} else
				return 0;
		} else
			return 0;
	}

	public int[][] MatrixConstruct(int[] rowSum) {
		int n = rowSum.length;
		int[] sortedRow = new int[n];
		for (int k = 0; k < n; k++) {
			sortedRow[k] = sortRowSum(rowSum)[0][k];
		}
		int[][] matrixc = new int[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (sortedRow[i] != 0) {
					matrixc[i][j]++;
					sortedRow[i]--;
				}
			}
		}
		return matrixc;
	}

	public int[][] RyserMatrix(int[] rowSum, int[] colSum) {
		int exist = existMatrix(rowSum, colSum);
		if (exist == 1) {
			int[][] matConstruct = MatrixConstruct(rowSum);
			int dimension = rowSum.length;
			int[] newcol = new int[dimension];
			for (int i = 0; i < dimension; i++) {
				for (int j = 0; j < dimension; j++)
					newcol[i] += matConstruct[j][i];
			}
			int a = 0, b = 0;
			int count = 0;
			int nc = 0;
			for (int i = 0; i < dimension; i++) {
				if (newcol[i] > colSum[i]) {
					nc += newcol[i] - colSum[i];
				}
				if (colSum[i] > newcol[i])
					break;
			}
			while (count < nc) {
				a = 0;
				b = 0;
				for (int i = 0; i < dimension; i++)
					if (newcol[i] > colSum[i]) {
						a = i;
						break;
					}
				for (int i = 0; i < dimension; i++)
					if (newcol[i] < colSum[i]) {
						b = i;
						break;
					}
				for (int i = 0; i < dimension; i++) {
					if (matConstruct[i][a] == 1 && matConstruct[i][b] == 0) {
						matConstruct[i][a] = 0;
						matConstruct[i][b] = 1;
						newcol[a]--;
						newcol[b]++;
						break;
					}

				}
				count++;
			}
			return matConstruct;
		} else
			return null;
	}
	/*
	 * public void PrintMatrix(int[] rowSum1, int[] colSum1){ int[] rowSum =
	 * sortRowSum(rowSum1)[0]; int[] colSum = sortColSum(colSum1)[0]; int exist
	 * = existMatrix(rowSum, colSum); if(exist == 1){ int[][] displayMatrix =
	 * RyserMatrix(rowSum, colSum); int[] rowOrder = sortColSum(colSum)[1];
	 * int[] colOrder = sortRowSum(rowSum)[1];
	 * 
	 * for(int c=0; c<colSum.length; c++){ for(int r=0; r<rowSum.length; r++){
	 * System.out.print(displayMatrix[c][r]);
	 * //System.out.print(displayMatrix[rowOrder[c]][colSum[r]] + ", "); }
	 * System.out.println(); } } }
	 */
}
