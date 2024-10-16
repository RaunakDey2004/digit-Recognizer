import pygame
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('myapp/ml_model/mnist_model.h5')

# Initialize Pygame
pygame.init()

# Set up the drawing window
window_size = (280, 280)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Draw a Digit")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up the canvas
screen.fill(WHITE)


def preprocess_image(image):
    """
    Preprocess the captured image to match the format expected by the model.
    Args:
    image (pygame.Surface): Captured image from the Pygame surface.

    Returns:
    np.array: Preprocessed image.
    """
    image = pygame.surfarray.array3d(image)  # Convert to 3D array
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
    image = Image.fromarray(image)  # Convert to PIL Image
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.reshape(1, 28, 28, 1)  # Reshape to match model input shape

    return image


def predict_digit(image):
    """
    Predict the digit in the captured image using the trained model.
    Args:
    image (pygame.Surface): Captured image from the Pygame surface.

    Returns:
    int: Predicted digit.
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make a prediction
    prediction = model.predict(preprocessed_image)

    # Get the digit with the highest probability
    predicted_digit = np.argmax(prediction)

    return predicted_digit


running = True
drawing = False
last_pos = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            current_pos = event.pos
            pygame.draw.line(screen, BLACK, last_pos, current_pos, 10)
            last_pos = current_pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Capture the drawing as an image
                drawing_surface = screen.copy()
                predicted_digit = predict_digit(drawing_surface)
                print(f"The predicted digit is: {predicted_digit}")

    pygame.display.flip()

pygame.quit()
