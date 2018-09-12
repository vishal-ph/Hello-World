import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;

public class matrix {
	public static void main(String[] args) {
		PrintStream out = null;
		try {
			out = new PrintStream(new FileOutputStream("output.txt"));
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
			e.printStackTrace();
		}
		System.setOut(out);

		// int[] rowSum = {4,3,2,1};
		// int[] colSum = {3,3,2,2};

		BufferedReader br = null;

		try {
			String actionString;
			int dimension;
			br = new BufferedReader(new FileReader("input-3.txt"));
			actionString = br.readLine();
			dimension = Integer.parseInt(actionString);
			String[] row = br.readLine().split(",");
			String[] col = br.readLine().split(",");
			int[] rowSum = new int[dimension];
			int[] colSum = new int[dimension];
			for (int i = 0; i < row.length; i++) {
				rowSum[i] = Integer.parseInt(row[i]);
				colSum[i] = Integer.parseInt(col[i]);
			}

			int[] originalRowSum = new int[rowSum.length];
			for (int i = 0; i < rowSum.length; i++) {
				originalRowSum[i] = rowSum[i];
			}

			int[] originalColSum = new int[colSum.length];
			for (int i = 0; i < colSum.length; i++) {
				originalColSum[i] = colSum[i];
			}

			int[] newRowSum = new int[rowSum.length];
			for (int i = 0; i < rowSum.length; i++) {
				newRowSum[i] = rowSum[i];
			}

			int[] newColSum = new int[colSum.length];
			for (int i = 0; i < colSum.length; i++) {
				newColSum[i] = colSum[i];
			}

			int[] Rorder = new int[rowSum.length];
			int[] Corder = new int[colSum.length];

			BipartiteGraph Matrix = new BipartiteGraph(rowSum, colSum);

			int hai = Matrix.existMatrix(rowSum, colSum);
			System.out.println(hai);

			int[][] rowsorted = Matrix.sortRowSum(newRowSum);
			int[][] colsorted = Matrix.sortColSum(newColSum);

			for (int i = 0; i < rowSum.length; i++) {
				for (int j = 0; j < rowSum.length; j++) {
					if (originalRowSum[i] == rowsorted[0][j]) {
						Rorder[i] = j;
						rowsorted[0][j] = -10;
						break;
					}
				}
			}
			for (int i = 0; i < colSum.length; i++) {
				for (int j = 0; j < colSum.length; j++) {
					if (originalColSum[i] == colsorted[0][j]) {
						Corder[i] = j;
						colsorted[0][j] = -10;
						break;
					}
				}
			}
			int[][] matrixfinal = Matrix.RyserMatrix(rowSum, colSum);
			for (int i = 0; i < colSum.length; i++) {
				for (int j = 0; j < colSum.length - 1; j++) {
					System.out.print(matrixfinal[Rorder[i]][Corder[j]] + ",");
				}
				System.out.println(matrixfinal[Rorder[i]][Corder[colSum.length - 1]]);

			}

			/*
			 * File file = new File("output.txt");
			 * 
			 * if (!file.exists()) { file.createNewFile(); }
			 * 
			 * FileWriter fw = new FileWriter(file.getAbsoluteFile());
			 * BufferedWriter bw = new BufferedWriter(fw);
			 * bw.write(Integer.toString(hai)); int[] Ri = new int[dimension];
			 * int[] Ci = new int[dimension]; if (hai == 1) { for (int i = 0; i
			 * < dimension; i++) { bw.newLine(); String content = ""; int r = 0;
			 * int c = 0; for (int j = 0; j < dimension - 1; j++) { content =
			 * content + matrixfinal[i][j] + ","; r += matrixfinal[i][j]; c +=
			 * matrixfinal[j][i]; } r += matrixfinal[i][dimension - 1]; c +=
			 * matrixfinal[dimension - 1][i]; Ri[i] = r; Ci[i] = c; content =
			 * content + matrixfinal[i][dimension - 1]; bw.write(content); } }
			 * 
			 * bw.close();
			 */
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (br != null)
					br.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}
}
