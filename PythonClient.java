import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;
import java.net.ServerSocket;  
import java.net.Socket;

public class PythonClient extends JFrame {

    private Ball ball;
    private Timer timer;

    public PythonClient() {
        initUI();
        startServerThread();
    }

    private void initUI() {
        ball = new Ball();
        add(ball);

        setSize(400, 400);
        setTitle("PythonClient");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        timer = new Timer(10, new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ball.repaint();
            }
        });
        timer.start();
    }
    private void startServerThread() {
        Thread serverThread = new Thread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                try (ServerSocket serverSocket = new ServerSocket(12345)) {
                    try (Socket clientSocket = serverSocket.accept()) {
                        handleClientConnection(clientSocket);
                    } catch (IOException ex) {
                        // Handle client connection exceptions
                        ex.printStackTrace();
                    }
                } catch (IOException ex) {
                    // Handle server socket exceptions (e.g., server not available)
                    System.out.println("Unable to connect to the server. Retrying in 5 seconds...");
                    try {
                        Thread.sleep(5000); // Retry after 5 seconds
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        });
        serverThread.start();
    }
    private void handleClientConnection(Socket clientSocket) {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()))) {
            String message;
            while ((message = in.readLine()) != null) {
                System.out.println("Received from Python: " + message);
                final String receivedMessage = message;
                SwingUtilities.invokeLater(() -> {
                    switch (receivedMessage) {
                        case "listening":
                            ball.setListeningAnimation();
                            break;
                        case "talking":
                            ball.setTalkingAnimation();
                            break;
                        case "thinking":
                            ball.setThinkingAnimation();
                            break;
                    }
                    ball.repaint();  // Trigger the repaint to update the UI
                });
            }
        } catch (IOException e) {
            e.printStackTrace();
            // Additional error handling can be done here
        }
    }
            
    public static void main(String[] args) {
        EventQueue.invokeLater(() -> {
            PythonClient client = new PythonClient();
            client.setVisible(true);
        });
    }
}

class Ball extends JPanel {
    private double size = 50;
    private double deltaSize = 1;
    private double maxSize = 60;
    private double minSize = 40;
    private boolean expanding = true;
    private Color color = Color.BLUE;

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        drawBall(g);
    }

    private void drawBall(Graphics g) {
        g.setColor(color);
        int x = getWidth() / 2 - (int)size / 2;
        int y = getHeight() / 2 - (int)size / 2;
        g.fillOval(x, y, (int)size, (int)size);

        // Update size for breathing effect
        if (expanding) {
            size += deltaSize;
            if (size >= maxSize) {
                expanding = false;
            }
        } else {
            size -= deltaSize;
            if (size <= minSize) {
                expanding = true;
            }
        }
    }

    public void setListeningAnimation() {
        color = Color.GREEN; // Listening color
        expanding = true;
        maxSize = 70;
        minSize = 50;
        deltaSize = 0.5;
    }

    public void setTalkingAnimation() {
        color = Color.RED; // Talking color
        expanding = true;
        maxSize = 60;
        minSize = 40;
        deltaSize = 1;
    }

    public void setThinkingAnimation() {
        color = Color.YELLOW; // Thinking color
        expanding = true;
        maxSize = 65;
        minSize = 45;
        deltaSize = 0.7;
    }
}
