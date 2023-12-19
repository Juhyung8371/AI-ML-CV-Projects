
package flappybird;

/**
 * Matrix.java - 2d array
 *
 * @author Juhyung Kim
 * @since 2018. 2. 21.
 */
public class Matrix {

	private int rows;
	private int cols;
	private double[][] data;

	/**
	 * Make a 2d empty matrix
	 *
	 * @param rows
	 * @param cols
	 */
	public Matrix(int rows, int cols) {
		this(rows, cols, false);
	}

	/**
	 * Make a 2d empty matrix
	 *
	 * @param rows
	 * @param cols
	 * @param fillRandom
	 *
	 * @see Matrix.fillRandom()
	 */
	public Matrix(int rows, int cols, boolean fillRandom) {
		this.rows = rows;
		this.cols = cols;
		this.data = new double[rows][cols];

		if (fillRandom)
			fillRandom();
	}

	/**
	 * Subtract the corresponding rows and columns
	 *
	 * @param toSubtract
	 */
	public void subtract(Matrix toSubtract) {

		// a should equal b in dimension
		if (!isSameSize(toSubtract)) {
			System.out.println(
					"Columns and Rows of A must match " + "Columns and Rows of B.");
			return;
		}

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				data[y][x] -= toSubtract.data[y][x];
			}
		}
	}

	/**
	 * Add the corresponding rows and columns
	 *
	 * @param toAdd
	 */
	public void add(Matrix toAdd) {

		// a should equal b in dimension
		if (!isSameSize(toAdd)) {
			System.out.println(
					"Columns and Rows of A must match " + "Columns and Rows of B.");
			return;
		}

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				data[y][x] -= toAdd.data[y][x];
			}
		}
	}

	/**
	 * Multiply the corresponding rows and columns
	 *
	 * @param toMultiply
	 */
	public void multiply(Matrix toMultiply) {

		// a should equal b in dimension
		if (!isSameSize(toMultiply)) {
			System.out.println(
					"Columns and Rows of A must match " + "Columns and Rows of B.");
			return;
		}

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				data[y][x] *= toMultiply.data[y][x];
			}
		}
	}

	/**
	 * Divide the corresponding rows and columns
	 *
	 * @param toDivide
	 */
	public void divide(Matrix toDivide) {

		// a should equal b in dimension
		if (!isSameSize(toDivide)) {
			System.out.println(
					"Columns and Rows of A must match " + "Columns and Rows of B.");
			return;
		}

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				data[y][x] /= toDivide.data[y][x];
			}
		}
	}

	/**
	 * Fill the data with random doubles 0 ~ 1.0 (exclusive)
	 */
	public void fillRandom() {

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				data[y][x] = Util.nextDouble();
			}
		}

	}

	/**
	 * Transpose the data so the rows and columns are swapped
	 */
	public void transpose() {

		double[][] newData = new double[rows][cols];

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				newData[y][x] = data[x][y];
			}
		}

		data = newData;

	}

	/**
	 * Stringify the data
	 */
	public String toString() {

		String output = "[";

		for (int y = 0; y < rows; y++) {

			output.concat("[");

			for (int x = 0; x < cols; x++) {

				output.concat(Double.toString(data[x][y]));

				if (x < cols - 1)
					output.concat(", ");

			}

			output.concat("]");

		}

		return output.concat("]");

	}

	/**
	 * Compare the number of rows and columns
	 *
	 * @param matrix
	 * @return
	 */
	public boolean isSameSize(Matrix matrix) {

		return cols == matrix.getColumns() && rows == matrix.getRows();
	}

	public int getRows() {

		return rows;
	}

	public int getColumns() {

		return cols;
	}

}
