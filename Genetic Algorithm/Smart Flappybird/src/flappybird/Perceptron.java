
package flappybird;

/**
 * Perceptron.java - An artificial neuron
 *
 * @author Juhyung Kim
 * @since 2018. 2. 21.
 */
public class Perceptron {

	private float[] weights;
	private float bias;
	private float learningRate, mutationRate;
	private int inputLength;

	/**
	 *
	 * @param inputLength  number of inputs
	 * @param learningRate
	 */
	public Perceptron(int inputLength, float learningRate, float mutationRate) {

		this.inputLength = inputLength;
		this.learningRate = learningRate;
		this.mutationRate = mutationRate;

		this.weights = new float[inputLength];
		this.bias = Util.nextFloat(-1, 1);

		for (int i = 0; i < weights.length; i++) {

			this.weights[i] = Util.nextFloat(-1, 1);

		}

	}

	/**
	 * Mutate the perceptron's weights depend on the mutation rate
	 */
	public void mutate() {

		for (int i = 0; i < weights.length; i++) {

			if (Util.nextFloat() < mutationRate)
				this.weights[i] = Util.nextFloat();
		}

	}

	/**
	 * Adjust the weights of inputs
	 *
	 * @param inputs
	 * @param target
	 */
	public void train(float[] inputs, int target) {

		double guess = getOutput(inputs);

		double error = target - guess;

		// adjust the weights
		for (int i = 0; i < weights.length; i++) {
			weights[i] += inputs[i] * error * learningRate;
		}

		bias += bias * error * learningRate;

	}

	/**
	 * Calculate the sum of total inputs * weights + bias. For forward outputting.
	 *
	 * @param inputs
	 * @return
	 */
	public float getOutput(float[] inputs) {

		float sum = 0;

		for (int i = 0; i < weights.length; i++) {
			sum += inputs[i] * weights[i];
		}

		sum += bias;

		return activate(sum);
	}

	/**
	 * Used to manipulate the sum from feedForward to get result value.
	 *
	 * @param sum
	 * @return
	 */
	private float activate(float sum) {

		return (float) Util.sigmoid(sum);

	}

	/**
	 * @return the weights
	 */
	public float[] getWeights() {

		return weights;
	}

	/**
	 * @return the inputLength
	 */
	public int getInputLength() {

		return inputLength;
	}

	/**
	 * @return the bias
	 */
	public float getBias() {

		return bias;
	}

	/**
	 * @param bias the bias to set
	 */
	public void setBias(float bias) {

		this.bias = bias;
	}

	/**
	 * @param weights the weights to set
	 */
	public void setWeights(float[] weights) {

		this.weights = weights;
	}

	/**
	 * @return the learningRate
	 */
	public float getLearningRate() {

		return learningRate;
	}

	/**
	 * @param learningRate the learningRate to set
	 */
	public void setLearningRate(float learningRate) {

		this.learningRate = learningRate;
	}

	/**
	 * @return the mutationRate
	 */
	public float getMutationRate() {

		return mutationRate;
	}

	/**
	 * @param mutationRate the mutationRate to set
	 */
	public void setMutationRate(float mutationRate) {

		this.mutationRate = mutationRate;
	}

}
