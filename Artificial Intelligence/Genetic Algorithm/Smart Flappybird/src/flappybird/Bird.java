
package flappybird;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Polygon;
import java.awt.Rectangle;

public class Bird implements Comparable<Bird> {

	/**
	 * Size
	 */
	public static final int SIZE = 30;

	/**
	 * Number of inputs this receives
	 * 1. Horizontal distance to next column
	 * 2. vertical distance to next column
	 * 3. vertical distance to the center of the map
	 */
	private static final int INPUT_AMOUNT = 3;

	/**
	 * Number of hidden layers and its perceptrons in that layer
	 */
	private static final int[] HIDDEN_LAYER_AMOUNT = {6};

	/**
	 * Number of output perceptrons
	 */
	private static final int OUTPUT_AMOUNT = 1;

	/**
	 * Rate of learning per training. Lower the more precise it gets, but learning
	 * speed slow down.
	 */
	private static final float LEARNING_RATE = 0.1f;

	private static final float MUTATION_RATE = 0.01f;

	private static final int ACCELERATION = 2;

	private float fitness;

	private NeuralNetwork network;

	public Rectangle bounds;

	private int motionY = 0;

	private int distance = 0;

	public int score = 0;

	private Color color;

	public Bird() {

		// this.network = new NeuralNetwork();
		this.fitness = 0;

		this.bounds = new Rectangle(FlappyBird.WIDTH / 2 - SIZE / 2,
				FlappyBird.HEIGHT / 2 - SIZE / 2, SIZE, SIZE);

		network = new NeuralNetwork(INPUT_AMOUNT, HIDDEN_LAYER_AMOUNT, OUTPUT_AMOUNT,
				LEARNING_RATE, MUTATION_RATE, false);

		color = new Color(Util.nextInt(256), Util.nextInt(256), Util.nextInt(256));

	}

	public void draw(Graphics g) {
		drawBeak(g);
		drawBody(g);
		drawEye(g);
		drawPupil(g);
	}

	/**
	 * Draw the pupil of the bird.
	 * 
	 * @param g
	 */
	private void drawPupil(Graphics g) {

		int x = bounds.x + bounds.width * 2 / 3 + 1;
		int y = bounds.y + bounds.height * 1 / 4 + 2;
		int w = bounds.width * 1 / 5;
		int h = bounds.height * 1 / 5;

		g.setColor(Color.BLACK);
		g.fillOval(x, y, w, h);

	}

	/**
	 * Draw the eye of the bird
	 * 
	 * @param g
	 */
	private void drawEye(Graphics g) {

		int x = bounds.x + bounds.width * 2 / 3 - 3;
		int y = bounds.y + bounds.height * 1 / 4;
		int w = bounds.width * 1 / 3;
		int h = bounds.height * 1 / 3;

		g.setColor(Color.WHITE);
		g.fillOval(x, y, w, h);

	}

	/**
	 * Draw the body of the bird.
	 * 
	 * @param g
	 */
	private void drawBody(Graphics g) {

		g.setColor(this.color);
		g.fillOval(bounds.x, bounds.y, bounds.width, bounds.height);

	}

	/**
	 * Draw the beak of the bird.
	 * 
	 * @param g
	 */
	private void drawBeak(Graphics g) {

		// arrays of points that create the triangluar beak
		int[] xCoords = { (int) bounds.getMaxX(), (int) bounds.getMaxX(),
				(int) bounds.getMaxX() + 10 };

		int[] yCoords = { (int) bounds.getY() + 10, (int) bounds.getMaxY() - 10,
				(int) bounds.getCenterY() };

		Polygon beak = new Polygon(xCoords, yCoords, 3);

		// the beak
		g.setColor(Color.yellow);
		g.fillPolygon(beak);

	}

	public void resetPos() {

		this.bounds.x = FlappyBird.WIDTH / 2 - SIZE / 2;
		this.bounds.y = FlappyBird.HEIGHT / 2 - SIZE / 2;
		distance = 0;
		setScore(0);

	}

	public void fall() {

		if (motionY < 15)
			motionY += ACCELERATION;

		bounds.y += motionY;

	}

	public void act(float[] inputs) {

		// Even though the outputs are in array, the bird has only one output
		// perceptron
		// so I will not bother using for loop here
		float output = network.getOutputs(inputs)[0];

		distance++;

		// jump
		if (output > 0.5) 
			motionY = -15;

	}

	/**
	 * Mix the gene
	 *
	 * @param p1 parent
	 * @param p2 parent
	 * @return
	 */
	public static Bird crossover(Bird p1, Bird p2) {

		Bird child = new Bird();

		// + 1 is the output layer
		Perceptron[][] childLayers = new Perceptron[HIDDEN_LAYER_AMOUNT.length
				+ 1][];

		Perceptron[][] p1Layers = p1.getNetwork().getLayers();
		Perceptron[][] p2Layers = p2.getNetwork().getLayers();

		for (int y = 0; y < p1Layers.length; y++) {

			childLayers[y] = new Perceptron[p1Layers[y].length];

			for (int x = 0; x < p1Layers[y].length; x++) {

				Perceptron parentPerceptron1 = p1Layers[y][x];
				Perceptron parentPerceptron2 = p2Layers[y][x];

				int currentInputLength = parentPerceptron1.getInputLength();
				float mutationRate = parentPerceptron1.getMutationRate();
				float learningRate = parentPerceptron1.getLearningRate();
				float[] weights1 = parentPerceptron1.getWeights();
				float[] weights2 = parentPerceptron2.getWeights();

				Perceptron childPerceptron = new Perceptron(currentInputLength,
						learningRate, mutationRate);

				float[] childWeights = new float[weights1.length];

				int indexZ = (y == 0) ? INPUT_AMOUNT : p1Layers[y - 1].length;

				for (int z = 0; z < indexZ; z++) {
					
					childWeights[z] = (Util.nextInt(2) > 0) ? weights1[z]
							: weights2[z];
					
				}

				childPerceptron.setWeights(childWeights);
				childPerceptron.mutate();

				childLayers[y][x] = childPerceptron;

			}

		}

		NeuralNetwork childNetwork = new NeuralNetwork(INPUT_AMOUNT,
				HIDDEN_LAYER_AMOUNT, OUTPUT_AMOUNT, LEARNING_RATE, MUTATION_RATE,
				true);

		childNetwork.setLayers(childLayers);

		child.setNetwork(childNetwork);

		return child;

	}

	@Override
	public Bird clone() {

		Bird bird = new Bird();

		bird.fitness = fitness;
		bird.bounds = bounds;
		bird.network = network;
		bird.color = color;

		return bird;

	}

	/**
	 * Calculate the fitness
	 *
	 * @param bottomColumn
	 * @param gapHeight
	 */
	public void calcFitness(Rectangle bottomColumn, int gapHeight) {

		int dis = getDistanceTravelled();
		int gap = getVerticalGapDis(bottomColumn, gapHeight, true) * 2;

		fitness = (dis - gap) + score * 5;

	}

	/**
	 * @return the fitness
	 */
	public float getFitness() {

		return fitness;
	}

	/**
	 * @return the network
	 */
	public NeuralNetwork getNetwork() {

		return network;
	}

	/**
	 * @param network the network to set
	 */
	public void setNetwork(NeuralNetwork network) {

		this.network = network;
	}

	/**
	 * Horizontal distance travelled.
	 *
	 * @return
	 */
	public int getDistanceTravelled() {

		return distance;
	}

	/**
	 * Get the vertical distance to the gap of the column. Closer to 0 is good.
	 *
	 * @param bottomColumn
	 * @param gapHeight
	 * @return
	 */
	public int getVerticalGapDis(Rectangle bottomColumn, int gapHeight,
			boolean isAbs) {

		int birdY = bounds.y + bounds.height / 2;
		int gapY = bottomColumn.y - gapHeight / 2;

		int result = (birdY - gapY);

		if (isAbs && result < 0) {
			result = -result;
		}

		return result;

	}
	
	public int getDistanceFromCenterY(int centerY, boolean isAbs) {
		
		int birdY = bounds.y + bounds.height / 2;

		int result = (birdY - centerY);

		if (isAbs && result < 0) {
			result = -result;
		}

		return result;
		
	}

	/**
	 * @return the yMotion
	 */
	public int getMotionY() {

		return motionY;
	}

	/**
	 * @param yMotion the yMotion to set
	 */
	public void setMotionY(int yMotion) {

		this.motionY = yMotion;
	}

	/**
	 * @return the color
	 */
	public Color getColor() {

		return color;
	}

	/**
	 * @param color the color to set
	 */
	public void setColor(Color color) {

		this.color = color;
	}

	/**
	 * @return the score
	 */
	public int getScore() {
		return score;
	}

	/**
	 * @param score the score to set
	 */
	public void setScore(int score) {
		this.score = score;
	}
	
    @Override
    public int compareTo(Bird arg0) {

        if (this.getFitness() > arg0.getFitness())
            return 0;
        else
            return 1;
    }
    
}
