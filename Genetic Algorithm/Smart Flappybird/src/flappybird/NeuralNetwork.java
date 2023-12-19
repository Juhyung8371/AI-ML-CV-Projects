
package flappybird;

/**
 * NeuralNetwork.java - A network of perceptrons
 *
 * @author Juhyung Kim
 * @since 2018. 2. 21.
 */
public class NeuralNetwork {

	private Perceptron[][] layers;
	private float mutationRate;
	private float learningRate; // used when there is given target
	private int outputCount;

	/**
	 * First layer is inputs (not perceptrons) Proceeding layers are perceptrons
	 * <p>
	 * Ex: 3 layered (input, hidden, output) network will have 2 perceptron layers
	 * (hidden, output)
	 *
	 * @param inputAmount  number of initial inputs
	 * @param hiddenCount  Length = number of hidden layers, Values = number of
	 *                     perceptrons in that layer
	 * @param outputCount  number of output perceptrons
	 * @param learningRate
	 * @param isChild
	 */
	public NeuralNetwork(int inputAmount, int[] hiddenCount, int outputCount,
			float learningRate, float mutationRate, boolean isChild) {

		this.setLearningRate(learningRate);
		this.mutationRate = mutationRate;
		this.outputCount = outputCount;

		if (!isChild) {

			int hiddenLayersCount = hiddenCount.length;

			// + 1 is the last layer, output layer
			this.layers = new Perceptron[hiddenLayersCount + 1][];

			// fill in the hidden layers with perceptrons
			for (int i = 0; i < hiddenLayersCount; i++) {

				int layerLength = hiddenCount[i];

				Perceptron[] hidden = new Perceptron[layerLength];

				// fill in this layer
				for (int x = 0; x < layerLength; x++) {

					// if it's not the first hidden layer,
					// the number of inputs = number of outputs of previous one
					int inputs = (i == 0) ? inputAmount : hiddenCount[i - 1];

					hidden[x] = new Perceptron(inputs, learningRate, mutationRate);

				}

				layers[i] = hidden;

			}

			// last layer is output
			Perceptron[] output = new Perceptron[outputCount];

			for (int i = 0; i < outputCount; i++) {
				output[i] = new Perceptron(hiddenCount[hiddenLayersCount - 1],
						learningRate, mutationRate);
			}

			layers[layers.length - 1] = output;

		}
	}

	/**
	 * Calculate the sum of total inputs * weights + bias. For forward outputting.
	 *
	 * @param inputs
	 * @return
	 */
	public float[] getOutputs(float[] inputsF) {

		float[] inputForNextLayer = new float[layers[0].length];
		float[] inputForNextLayerTemp = null;

		// Util.showOutputs(inputsF);
		// go through all layers and get the final outputs
		for (int i = 0; i < layers.length; i++) {

			if (i != 0) {
				inputForNextLayerTemp = inputForNextLayer;
				inputForNextLayer = new float[layers[i].length];
			}

			// get the outputs from each layer
			for (int x = 0; x < layers[i].length; x++) {
				// first hidden layer gets the inputs from the argument
				if (i == 0) {
					inputForNextLayer[x] = layers[i][x].getOutput(inputsF);
				} else {
					inputForNextLayer[x] = layers[i][x]
							.getOutput(inputForNextLayerTemp);
				}
			}

			// Util.showOutputs(inputForNextLayer);
		}

		// System.out.println("---------");
		// output from last layer
		return inputForNextLayer;

	}

	/**
	 * Print out the number of perceptrons in each layer, Also the number of inputs
	 * it receives.
	 */
	public void printSimpleDiagram() {

		for (int i = 0; i < layers.length; i++) {

			if (i < layers.length - 1)
				System.out.print("hidden layer " + (i + 1) + ":\t");
			else
				System.out.print("output layer:\t");

			for (int x = 0; x < layers[i].length; x++) {
				System.out.print(layers[i][x].getInputLength() + "   ");
			}

			System.out.println("");
		}

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
	 * @return the layers
	 */
	public Perceptron[][] getLayers() {

		return layers;
	}

	/**
	 * @param layers the layers to set
	 */
	public void setLayers(Perceptron[][] layers) {

		this.layers = layers;
	}

}
