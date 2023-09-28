import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class Main {

	static BufferedImage target = null;
	static int xMax = 0, yMax = 0; // assigned later
	static int geneSize = 1000;
	static double mutationRate = 0.01;
	static int maxPop = 50;
	static String targetFileName = "a.png";

	public static void main(String[] args) {

		setTarget();
		
		Population pop = new Population(target, geneSize, maxPop,
				xMax, yMax, mutationRate);

		pop.setRunning(false); // just in case set it false again

		JFrame frame = createFrame(pop);

		Running run = new Running(pop, frame);

		Thread thr = new Thread(run);

		thr.start();

	}

	public static void setTarget() {

		// convert to black and white
		try {

			File targetFile = new File(targetFileName);
			BufferedImage orginalImage = ImageIO.read(targetFile);

			xMax = orginalImage.getWidth();
			yMax = orginalImage.getHeight();

			target = new BufferedImage(xMax, yMax, BufferedImage.TYPE_BYTE_BINARY);
			//target = new BufferedImage(xMax, yMax, BufferedImage.TYPE_INT_ARGB);

			Graphics2D graphics = target.createGraphics();
			graphics.drawImage(orginalImage, 0, 0, null);

			//ImageIO.write(target, "png", new File("aaaa.png"));

		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public static JFrame createFrame(final Population pop) {

		final JFrame frame = new JFrame("Genetics");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		frame.getContentPane().setLayout(new BorderLayout());

		JLabel picture = new JLabel();
		picture.setMinimumSize(new Dimension(300, 300));
		frame.getContentPane().add(picture, BorderLayout.CENTER);

		JLabel text = new JLabel("Generation: 0");
		frame.getContentPane().add(text, BorderLayout.EAST);

		final JButton button = new JButton("Continue running");
		button.setSize(new Dimension(50, 50));
		button.setLocation(500, 350);

		button.addActionListener(new ActionListener() {

			public void actionPerformed(ActionEvent e) {

				boolean isRunning = pop.isRunning();

				if (isRunning) {
					button.setText("Continue running");
				} else {
					button.setText("Stop running");
				}

				pop.setRunning(!isRunning);

			}
		});

		frame.getContentPane().add(button, BorderLayout.SOUTH);

		frame.setSize(800, 700);
		frame.setPreferredSize(new Dimension(800, 700));
		frame.pack();
		frame.setVisible(true);

		return frame;
	}

}
