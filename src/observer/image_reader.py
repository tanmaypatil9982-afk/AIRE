import cv2
import numpy as np


def create_test_image(path="test_image.png"):
    """Creates a test image with multiple shapes for AIRE to observe."""
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    # Blue filled rectangle
    cv2.rectangle(image, (30, 30), (150, 150), (255, 0, 0), thickness=-1)

    # Green filled circle
    cv2.circle(image, (220, 80), 50, (0, 255, 0), thickness=-1)

    # Red line
    cv2.line(image, (30, 250), (270, 250), (0, 0, 255), thickness=5)

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