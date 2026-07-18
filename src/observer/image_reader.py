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

def detect_edges(path, output_path="edges_output.png"):
    """Detects edges in an image — AIRE's first 'Understand' capability."""
    image = cv2.imread(path)

    if image is None:
        print("Could not read image.")
        return

    # Convert to grayscale first — edge detection works on brightness, not color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny edge detection: finds boundaries where brightness changes sharply
    edges = cv2.Canny(gray, threshold1=30, threshold2=90)

    cv2.imwrite(output_path, edges)
    print(f"Edge-detected image saved at: {output_path}")
    return output_path

if __name__ == "__main__":
    img_path = create_test_image()
    read_image_info(img_path)
    detect_edges(img_path)

def detect_shapes(path, output_path="contours_output.png"):
    """Finds individual shapes in the image and draws boxes around them."""
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=90)

    # Find contours: the outlines of separate connected shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} shape(s) in the image.")

    for i, contour in enumerate(contours):
        # Get a bounding box (x, y, width, height) around each shape
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        print(f"Shape {i+1}: position=({x},{y}), size=({w}x{h}), area={area:.0f}")

        # Draw a yellow rectangle around each detected shape
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), thickness=2)

    cv2.imwrite(output_path, image)
    print(f"Shapes marked and saved at: {output_path}")
    return output_path

if __name__ == "__main__":
    img_path = create_test_image()
    read_image_info(img_path)
    detect_edges(img_path)
    detect_shapes(img_path)