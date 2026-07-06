import cv2
import numpy as np

def create_test_image(path="test_image.png"):
    """Create a simple test image so AIRE has something to observe."""
    # Create a blank 300x300 black canvas with 3 color channels (Blue, Green, Red)
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    # Draw a filled blue rectangle (AIRE's frist 'observed' shape)
    cv2.rectangle(image,(50, 50), (250, 250), (255, 0, 0), thickness=-1)

    cv2.imwrite(path, image)
    print(f"Test image created at: {path}")
    return path

def read_image_info(path):
    """AIRE's first 'Observe' capability: reads an image and reports basic info."""
    image = cv2.imread(path)

    if image is None:
        print("Could not read image.")
        return
    
    height, width, channels = image.shape
    print(f"Image loaded: {path}")
    print(f"Width: {width}px, Height: {height}px, Channels: {channels}")

if __name__ == "__main__":
    img_path = create_test_image()
    read_image_info(img_path)