
import java.util.Random;

public class DNA {

	private static final int LINE_LENGTH = 4;
	private static final int PIECE_SIZE = 5;

	private int size;
	private int[][] gene;
	private int xMax, yMax, realXMax, realYMax;
	private double mutationRate;

	private static final Random RAND = new Random();

	/**
	 * 
	 * @param size number of lines it contains
	 * @param isChild false, when instantiated manually
	 * @param xMax
	 * @param yMax
	 * @param mutationRate in decimal
	 */
	public DNA(int size, boolean isChild, int xMax, int yMax, double mutationRate) {

		this.gene = new int[size][PIECE_SIZE];
		this.size = size;
		this.xMax = xMax;
		this.yMax = yMax;
		this.realXMax = xMax - 2 * LINE_LENGTH;
		this.realYMax = yMax - 2 * LINE_LENGTH;
		this.mutationRate = mutationRate;

		if (!isChild) {
			for (int i = 0; i < size; i++) {
				gene[i] = getGenePiece();
			}
			mutate();
		}

	}

	/**
	 * Get a piece that makes gene
	 * x1, y1, x2, y2, color
	 * 
	 * @return
	 */
	private int[] getGenePiece() {

		//int r = (RAND.nextInt(2));

		//int g = RAND.nextInt(2);

		//int b = RAND.nextInt(2);

		//float a = RAND.nextFloat() * (0.5f - 0.4f) + 0.4f;
		//float r = RAND.nextFloat();
		//float g = RAND.nextFloat();
		//float b = RAND.nextFloat();

		//Color color = new Color(r, g, b, a);

		int[] cords = new int[PIECE_SIZE];

		// x1, y1, x2, y2
		cords[0] = RAND.nextInt(realXMax) + LINE_LENGTH;
		cords[1] = RAND.nextInt(realYMax) + LINE_LENGTH;

		cords[2] = cords[0];
		cords[3] = cords[1];

		cords[2] += (RAND.nextBoolean()) ? RAND.nextInt(LINE_LENGTH) : -RAND.nextInt(LINE_LENGTH);
		cords[3] += (RAND.nextBoolean()) ? RAND.nextInt(LINE_LENGTH) : -RAND.nextInt(LINE_LENGTH);

		//cords[4] = color.getRGB();
		//cords[4] = 0;

		return cords;

	}

	/**
	 * Mix two parents
	 * 
	 * @param p1
	 * @param p2
	 * @return
	 */
	public static DNA crossover(DNA p1, DNA p2) {

		DNA child = new DNA(p1.getSize(), true,
				p1.getxMax(), p1.getyMax(), p1.getMutationRate());

		int[][] gene = new int[child.getSize()][PIECE_SIZE];

		/*
		 * for (int i = 0; i < child.getSize(); i++) {
		 * 
		 * int a = RAND.nextInt(3);
		 * 
		 * if (a > 0) {
		 * gene[i] = p1.getGene()[i];
		 * } else {
		 * gene[i] = p2.getGene()[i];
		 * }
		 * 
		 * }
		 */

		int mid = RAND.nextInt(child.getSize());

		for (int i = 0; i < child.getSize(); i++) {

			if (i < mid)
				gene[i] = p1.getGene()[i];
			else
				gene[i] = p2.getGene()[i];
			
		}

		child.setGene(gene);

		child.mutate();

		return child;

	}

	/**
	 * Alter the gene by mutation rate
	 */
	private void mutate() {

		for (int i = 0; i < size; i++) {

			if (RAND.nextDouble() < mutationRate)
				gene[i] = getGenePiece();

		}

	}

	/**
	 * @return the size
	 */
	public int getSize() {

		return size;
	}

	/**
	 * @param size the size to set
	 */
	public void setSize(int size) {

		this.size = size;
	}

	/**
	 * @return the gene
	 */
	public int[][] getGene() {

		return gene;
	}

	/**
	 * @param gene the gene to set
	 */
	public void setGene(int[][] gene) {

		this.gene = gene;
	}

	/**
	 * @return the xMax
	 */
	public int getxMax() {

		return xMax;
	}

	/**
	 * @param xMax the xMax to set
	 */
	public void setxMax(int xMax) {

		this.xMax = xMax;
	}

	/**
	 * @return the yMax
	 */
	public int getyMax() {

		return yMax;
	}

	/**
	 * @param yMax the yMax to set
	 */
	public void setyMax(int yMax) {

		this.yMax = yMax;
	}

	/**
	 * @return the mutationRate
	 */
	public double getMutationRate() {

		return this.mutationRate;
	}

	/**
	 * @return the mutationRate
	 */
	public void setMutationRate(double mutationRate) {

		this.mutationRate = mutationRate;
	}

}
