
package flappybird;

import java.util.Random;

public class Util {

	private static final Random RANDOM = new Random();

	public static int nextInt(int max) {
		return RANDOM.nextInt(max);
	}

	public static double nextDouble() {

		return RANDOM.nextDouble();
	}

	public static float nextFloat() {

		return RANDOM.nextFloat();
	}

	public static float nextFloat(float min, float max) {

		return RANDOM.nextFloat() * (max - min) + min;
	}

	public static float sigmoid(float x) {

		return (float) (1 / (1 + Math.exp(-x)));

	}

	public static float sigmoid_der(float x) {

		double y = sigmoid(x);

		return (float) (y * (1 - y));

	}

	public static float normalize(float value, float max) {

		// clamp the value between its min/max limits
		if (value < -max)
			value = -max;
		else if (value > max)
			value = max;

		// normalize the clamped value
		return (value / max);
	}

	public static void showOutputs(float[][] outputs) {

		for (int i = 0; i < outputs.length; i++) {

			System.out.print("[");
			for (int j = 0; j < outputs[i].length; j++) {

				System.out.print(outputs[i][j] + ", ");
			}

			System.out.println("]");

		}

	}

	public static void showOutputs(float[] outputs) {

		System.out.print("[");
		for (int j = 0; j < outputs.length; j++) {

			System.out.print(outputs[j] + ", ");
		}

		System.out.println("]");

	}

}
