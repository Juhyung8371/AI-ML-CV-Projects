import java.awt.image.BufferedImage;
import java.util.Random;

public class Population {

	private BufferedImage target;
	private int geneSize;
	private int maxPop;
	private Picture[] oldPop, newPop;
	private double mutationRate;
	private int generation;

	private boolean running;

	private static final Random RAND = new Random();

	public Population(BufferedImage target, int geneSize, int maxPop,
			int xMax, int yMax, double mutationRate) {

		this.generation = 0;
		this.running = false;

		this.target = target;
		this.geneSize = geneSize;
		this.maxPop = maxPop;
		this.oldPop = new Picture[maxPop];
		this.newPop = new Picture[maxPop];
		this.mutationRate = mutationRate;

		for (int i = 0; i < maxPop; i++) {
			oldPop[i] = new Picture(new DNA(geneSize, false, xMax, yMax, mutationRate));
			oldPop[i].calcFitness(target);
		}

	}

	/**
	 * Run a generation
	 */
	public void run() {

		// if (!running) return;

		// possible parents
		int[] fitnessIndex = { 0, 0, 0 };

		for (int i = 0; i < maxPop; i++) {
			if (oldPop[i].getFitness() > oldPop[fitnessIndex[0]].getFitness()) {

				fitnessIndex[2] = fitnessIndex[1];
				fitnessIndex[1] = fitnessIndex[0];
				fitnessIndex[0] = i;

			}
		}

		for (int i = 0; i < maxPop; i++) {

			int p1 = 0, p2 = 0;

			int random = RAND.nextInt(20);

			if (random > 10)
				p1 = fitnessIndex[0];
			else if (random > 2)
				p1 = fitnessIndex[1];
			else
				p1 = fitnessIndex[2];

			random = RAND.nextInt(20);

			if (random > 10)
				p2 = fitnessIndex[0];
			else if (random > 2)
				p2 = fitnessIndex[1];
			else
				p2 = fitnessIndex[2];

			newPop[i] = new Picture(DNA.crossover(
					oldPop[p1].getDna(),
					oldPop[p2].getDna()));

			newPop[i].calcFitness(target);
		}

		oldPop = newPop;
		newPop = new Picture[maxPop];

		generation++;

	}

	public boolean isRunning() {

		return this.running;
	}

	public void setRunning(boolean running) {

		this.running = running;
	}

	/**
	 * @return the generation
	 */
	public int getGeneration() {

		return generation;
	}

	/**
	 * @return the result
	 */
	public Picture[] getResult() {

		return oldPop;
	}

	/**
	 * @return the maxPop
	 */
	public int getMaxPop() {

		return maxPop;
	}

	/**
	 * get the highest
	 * 
	 * @return
	 */
	public double getHighestFitness() {

		int firstFitIndex = 0;

		for (int i = 0; i < maxPop; i++) {
			if (oldPop[i].getFitness() > oldPop[firstFitIndex].getFitness()) {
				firstFitIndex = i;
			}
		}
		return oldPop[firstFitIndex].getFitness();
	}

	/**
	 * @return the mutationRate
	 */
	public double getMutationRate() {

		return mutationRate;
	}

	/**
	 * @return the geneSize
	 */
	public int getGeneSize() {

		return geneSize;
	}

}
