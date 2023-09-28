import java.awt.Image;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class Running implements Runnable {

	private Population population;
	private JFrame frame;
	private double highestFit = 0;

	public Running(Population population, JFrame frame) {

		this.population = population;
		this.frame = frame;

	}

	@Override
	public void run() {

		while (true) {
			
			if(!population.isRunning()) continue;
			
			population.run();

			JLabel label = (JLabel) frame.getContentPane().getComponent(0);
			JLabel text = (JLabel) frame.getContentPane().getComponent(1);

			int highFitIndex = 0;
			Picture[] result = population.getResult();
			
			for (int i = 0; i < population.getMaxPop(); i++) {
				if (result[i].getFitness() > result[highFitIndex].getFitness()) {
					highFitIndex = i;
				}
			}
			
			if(highestFit < result[highFitIndex].getFitness())
				highestFit = result[highFitIndex].getFitness();
			
			Image bigimage = result[highFitIndex].getImage()
					.getScaledInstance(400, 400,  Image.SCALE_SMOOTH);
			
			label.setIcon(new ImageIcon(bigimage));
			
			text.setText("<html>Generation: " + population.getGeneration() + 
					"<br>Fitness: " + result[highFitIndex].getFitness() + 
					"<br>Highest: " + highestFit +
					"</html>");

		}

	}

}
