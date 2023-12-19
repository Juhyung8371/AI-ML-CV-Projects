
package flappybird;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import javax.swing.JFrame;
import javax.swing.Timer;

public class FlappyBird implements ActionListener, MouseListener, KeyListener {

	public static FlappyBird flappyBird;

	public static final int WIDTH = 800, HEIGHT = 800;

	public Renderer renderer;

	public Bird[] birds, bestPop;

	public int population = 500;

	public int generation = 1;

	// if true, the bird in that index is dead
	public Boolean[] deadPopulation = new Boolean[population];
	
	// to score only once
	public Boolean[] isScored = new Boolean[population];

	public int deadAmount = 0;

	public ArrayList<Rectangle> columns;

	public boolean gameOver, started;

	public int groundHeight = 120;

	// gap between top and bottom columns
	public int space = 300;

	public int distanceBetweenColumns = 400;

	public float bestFit = -500;
	public float lastFit = 0;

	public int bestScore = 0;
	public int lastScore = 0;

	public FlappyBird() {

		JFrame jframe = new JFrame();
		Timer timer = new Timer(20, this);

		renderer = new Renderer();

		jframe.add(renderer);
		jframe.setTitle("Flappy Bird");
		jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jframe.setSize(WIDTH, HEIGHT);
		jframe.addMouseListener(this);
		jframe.addKeyListener(this);
		jframe.setResizable(false);
		jframe.setVisible(true);

		birds = new Bird[population];
		bestPop = new Bird[population];

		// initialize the birds
		for (int i = 0; i < birds.length; i++) {
			birds[i] = new Bird();
			bestPop[i] = birds[i];
			deadPopulation[i] = false;
			isScored[i] = false;
		}

		columns = new ArrayList<>();

		addColumn(true);
		addColumn(false);
		
		timer.start();
	}

	public void addColumn(boolean start) {

		int width = 100;
		int height = 100 + Util.nextInt(300);

		if (start) {

			// btm
			columns.add(new Rectangle(WIDTH + width + columns.size() * 300,
					HEIGHT - height - groundHeight, width, height));
			// top
			columns.add(new Rectangle(columns.get(columns.size() - 1).x, 0, width,
					HEIGHT - height - space));

		} else {

			Rectangle prevCol = columns.get(columns.size() - 1);

			columns.add(new Rectangle(prevCol.x + distanceBetweenColumns,
					HEIGHT - height - groundHeight, width, height));

			prevCol = columns.get(columns.size() - 1);

			columns.add(new Rectangle(prevCol.x, 0, width, HEIGHT - height - space));

		}

	}

	public void paintColumn(Graphics g, Rectangle column) {
		g.setColor(Color.green.darker());
		g.fillRect(column.x, column.y, column.width, column.height);
	}

	/**
	 * Restart from scratch
	 */
	public void restartGame() {
		resetGame(false);
	}

	public void updateColumns() {

		// check the columns
		for (int i = 0; i < columns.size(); i++) {

			// move the column
			Rectangle column = columns.get(i);
			column.x -= speed;
		}

		// add or remove them as needed
		Rectangle first_bottom_column = columns.get(0);
		Rectangle first_top_column = columns.get(1);

		if (first_bottom_column.x + first_bottom_column.width < WIDTH/2-first_bottom_column.width) {
			// reset isScored
			for (int i = 0; i < birds.length; i++)
				isScored[i] = false;
			
			columns.remove(first_bottom_column);
			columns.remove(first_top_column);
			addColumn(false);
		}

	}

	int speed = 10;

	@Override
	public void actionPerformed(ActionEvent e) {

		if (gameOver) {
			resetGame(true);
			generation++;
			return;
		}

		if (started) {

			updateColumns();

			// literate through the birds
			for (int i = 0; i < birds.length; i++) {

				// skip the dead one
				if (deadPopulation[i])
					continue;

				// the index of the bottom column that's in front of bird
				int colIndex = 0;

				Rectangle approaching_column_btm = columns.get(colIndex);
				Rectangle approaching_column_top = columns.get(colIndex + 1);

				// now calculate the inputs
				// The bird must anticipate the fall, so the inputs are
				// calculated before falling.
				int disToGap = approaching_column_btm.x - birds[i].bounds.x;
				int gapDeltaY = birds[i].getVerticalGapDis(approaching_column_btm,
						space, false);
				int centerDelatY = birds[i].getDistanceFromCenterY(HEIGHT/2,false);
				
				float input1 = Util.normalize((float) disToGap,
						(float) distanceBetweenColumns);
				float input2 = Util.normalize((float) gapDeltaY, (float) HEIGHT / 2);
				
				float input3 = Util.normalize((float) centerDelatY, (float) HEIGHT / 2);
				
				birds[i].act(new float[] { input1, input2, input3 });

				// let the bird fall
				birds[i].fall();

				// collision test
				int birdx = birds[i].bounds.x + birds[i].bounds.width / 2;

				// calculate score
				if (birdx > approaching_column_btm.x + approaching_column_btm.width
						&& isScored[i] == false) {
					birds[i].score++;
					isScored[i] = true;
				}

				// if hit column
				if (approaching_column_btm.intersects(birds[i].bounds)
						|| approaching_column_top.intersects(birds[i].bounds)) {
					deadPopulation[i] = true;
					birds[i].calcFitness(approaching_column_btm, space);
				}
				// if hit top or bottom
				else if (birds[i].bounds.y > HEIGHT - groundHeight
						|| birds[i].bounds.y < 0) {
					deadPopulation[i] = true;
					birds[i].calcFitness(columns.get(colIndex), space);
				}
				// if hit bottom
				else if (birds[i].bounds.y + birds[i].getMotionY() >= HEIGHT
						- groundHeight) {
					birds[i].bounds.y = HEIGHT - groundHeight
							- birds[i].bounds.height;
					deadPopulation[i] = true;
					birds[i].calcFitness(columns.get(colIndex), space);
				}

				if (deadPopulation[i]) {
					deadAmount++;
					// if all died
					if (deadAmount == population)
						gameOver = true;
				}
			}
		}
		renderer.repaint();
	}

	public void repaint(Graphics g) {

		g.setColor(Color.cyan);
		g.fillRect(0, 0, WIDTH, HEIGHT);

		g.setColor(Color.orange);
		g.fillRect(0, HEIGHT - 100, WIDTH, 100);

		g.setColor(Color.green);
		g.fillRect(0, HEIGHT - 120, WIDTH, 20);

		g.setFont(new Font("Arial", 1, 20));

		for (int i = 0; i < birds.length; i++) {

			if (deadPopulation[i])
				continue;

			birds[i].draw(g);

			g.setColor(Color.BLACK);
			g.drawString(String.valueOf(birds[i].score), birds[i].bounds.x - 5,
					birds[i].bounds.y - 5);
		}

		for (Rectangle column : columns) {
			paintColumn(g, column);
		}

		g.setColor(Color.BLACK);

		g.drawString("Gen: " + generation, WIDTH / 2 + 30, 120);
		g.drawString("Dead: " + deadAmount + "/" + population, WIDTH / 2 + 30, 160);

		g.drawString("Best Fitness: " + bestFit, WIDTH / 2 + 30, 200);
		g.drawString("Last Gen's Fitness: " + lastFit, WIDTH / 2 + 30, 240);

		g.drawString("Best Score: " + bestScore, WIDTH / 2 + 30, 280);
		g.drawString("Last Gen's Score: " + lastScore, WIDTH / 2 + 30, 320);

		if (!started) {
			g.drawString("Click to start!", 75, HEIGHT / 2 - 50);
		}

		if (gameOver) {
			g.drawString("Game Over!", WIDTH / 2 - 100, HEIGHT / 2 - 50);
		}

	}

	public static void main(String[] args) {

		flappyBird = new FlappyBird();

	}

	@Override
	public void mouseClicked(MouseEvent e) {
		restartGame();
		started = true;
	}

	@Override
	public void keyReleased(KeyEvent e) {

		if (e.getKeyCode() == KeyEvent.VK_SPACE) {
			restartGame();
			started = true;
		}

	}

	@Override
	public void mousePressed(MouseEvent e) {

	}

	@Override
	public void mouseReleased(MouseEvent e) {

	}

	@Override
	public void mouseEntered(MouseEvent e) {

	}

	@Override
	public void mouseExited(MouseEvent e) {

	}

	@Override
	public void keyTyped(KeyEvent e) {

	}

	@Override
	public void keyPressed(KeyEvent e) {

	}

	private void resetGame(boolean isNewGen) {

		if (isNewGen) {

			// ascending order
			Arrays.sort(birds, Comparator.comparing(Bird::getFitness));

			// top three gets to breed
			Bird[] rank = new Bird[3];

			// the top three
			rank[0] = birds[population - 1];
			rank[1] = birds[population - 2];
			rank[2] = birds[population - 3];

			// revive the dead ones
			for (int i = 0; i < birds.length; i++) {
				deadPopulation[i] = false;
				isScored[i] = false;
			}

			lastFit = rank[0].getFitness();

			if (lastFit > bestFit)
				bestFit = lastFit;

			lastScore = rank[0].score;

			if (lastScore > bestScore)
				bestScore = lastScore;

			// time to produce some offsprings

			// if the elite has an outstanding performance,
			// increase the chance of its reproduction
			if ((float) rank[0].getFitness() > (float) bestFit) {
				System.out.println("outstanding bird in gen: " + generation);

				for (int i = 0; i < birds.length; i++) {
					bestPop[i] = (Bird) birds[i].clone();
				}
			}


			Bird[] rank_underdog = new Bird[3];

			
			// if this generation is way worse than the best one
			if ((float) bestFit / (float) rank[0].getFitness() < 0.5) {

				System.out
						.println("Gen: " + generation + " is weak, clone the best");
				
				// but save the best ones just in case

				// the top three
				rank_underdog[0] = birds[population - 1];
				rank_underdog[1] = birds[population - 2];
				rank_underdog[2] = birds[population - 3];
				
				// now override with the saved ones
				birds = bestPop.clone();

				rank[0] = birds[population - 1];
				rank[1] = birds[population - 2];
				rank[2] = birds[population - 3];

			}

			int odd = 3;

			// reproduce
			for (int i = 0; i < birds.length; i++) {

				// introduce some variable
				// by replacing the bottom half with random birds
				if (i < birds.length / 2) {
					birds[i] = new Bird();
				}
				// otherwise do normal crossover
				else {
					int rand = Util.nextInt(odd);

					// parents
					Bird p1, p2;

					if (rand > 1)
						p1 = rank[0];
					else if (rand > 0)
						p1 = rank[1];
					else
						p1 = rank[2];

					rand = Util.nextInt(odd);

					if (rand > 1)
						p2 = rank[0];
					else if (rand > 0)
						p2 = rank[1];
					else
						p2 = rank[2];

					birds[i] = Bird.crossover(p1, p2);
				}
			}

			rank[0].resetPos();
			rank[1].resetPos();
			rank[2].resetPos();
			
			// let the elites survive
			birds[0] = rank[0];
			birds[1] = rank[1];
			birds[2] = rank[2];
			
			if (rank_underdog[0] != null) {

				rank_underdog[0].resetPos();
				rank_underdog[1].resetPos();
				rank_underdog[2].resetPos();

				// let the elites survive
				birds[3] = rank_underdog[0];
				birds[4] = rank_underdog[1];
				birds[5] = rank_underdog[2];
			}		
		}

		columns.clear();

		deadAmount = 0;

		addColumn(true);
		addColumn(false);

		if (!isNewGen) {
			gameOver = false;
			started = false;
		} else {
			gameOver = false;
			started = true;
		}
	}
}
