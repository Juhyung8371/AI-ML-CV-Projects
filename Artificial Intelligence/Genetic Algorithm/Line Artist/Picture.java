import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;

public class Picture {

	private DNA dna;
	private BufferedImage image;
	private double fitness;

	public Picture(DNA dna) {

		this.fitness = 0;
		this.dna = dna;
		this.image = new BufferedImage(dna.getxMax(), dna.getyMax(),
				BufferedImage.TYPE_INT_ARGB);
		makeImage();

	}

	private void makeImage() {

		Graphics gfx = image.createGraphics();
		
		gfx.setColor(Color.BLACK);

		int[][] gene = dna.getGene();

		for (int i = 0; i < gene.length; i++) {

			int[] current = gene[i];
			
		//	gfx.setColor(new Color(current[4], true));

			gfx.drawLine(current[0], current[1], current[2], current[3]);

		//gfx.fillRect(current[0], current[1], current[2], current[3]);
			
		}

		gfx.dispose();
		/*
		 * try {
		 * 
		 * ImageIO.write(image, "png", new File("a.png"));
		 * 
		 * } catch (IOException e) {
		 * e.printStackTrace();
		 * }
		 */
	}

	/**
	 * Get how close to the target is (no max, relative values)
	 * 
	 * @param taret
	 * @return
	 */
	public void calcFitness(BufferedImage target) {

		int width = target.getWidth();
		int height = target.getHeight();

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				
				if (image.getRGB(x, y) == target.getRGB(x, y)) {
					
					fitness++;
				}
				
				/*
				int rgb = image.getRGB(x, y);
				int targetRGB = target.getRGB(x, y);
				
				Color color = new Color(rgb);
				Color targetColor = new Color(targetRGB);
				
				int a = color.getAlpha();
				int a1 = targetColor.getAlpha();
						
        int r = color.getRed();
        int r1 = targetColor.getRed();
        
        int g = color.getGreen();
        int g1 = targetColor.getGreen();
        
        int b = color.getBlue();
        int b1 = targetColor.getBlue();
        
        double deltaA = a1 - a;
        double deltaR = r1 - r;
        double deltaG = g1 - g;
        double deltaB = b1 - b;
        
				if (deltaA < 20 && deltaR < 20 && 
						deltaG < 20 && deltaB < 20) {
					
					fitness++;
				}

				
				*/
				
			}
		}

		// to make selection more accurate, the pool curve is exponential
		fitness *= fitness;
		fitness /= (double) (width * height);

	}

	/**
	 * @return the fitness
	 */
	public double getFitness() {

		return fitness;
	}

	/**
	 * @return the dna
	 */
	public DNA getDna() {

		return dna;
	}

	/**
	 * @return the image
	 */
	public BufferedImage getImage() {

		return image;
	}

}
