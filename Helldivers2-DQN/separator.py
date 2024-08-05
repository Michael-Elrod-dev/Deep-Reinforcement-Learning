import cv2
import numpy as np
from PIL import Image


def crop_area(image_path):

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    cropped_image = image[center_y - height // 4:center_y + height // 4, center_x - width // 4:center_x + width // 4]

    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (25, 25), 0)

    # Define the range for bright yellow color
    lower_yellow = np.array([20, 10, 160])
    upper_yellow = np.array([40, 80, 255])

    # Threshold the HSV image to get only bright yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    yellow_only = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours that have a significant area (to filter out noise)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # Assume the largest two contours are the target yellow objects
    if len(filtered_contours) >= 2:
        # Sort contours based on their area from largest to smallest
        filtered_contours.sort(key=cv2.contourArea, reverse=True)

        # Take the top two largest contours
        top_contours = filtered_contours[:2]

        # Calculate the bounding box for these two contours
        x_coords = []
        y_coords = []
        for cnt in top_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_coords.extend([x, x + w])
            y_coords.extend([y, y + h])

        # Create a bounding box around the extreme coordinates
        bounding_x1, bounding_x2 = min(x_coords), max(x_coords)
        bounding_y1, bounding_y2 = min(y_coords), max(y_coords)

    cropped_to_bounding_box = cropped_image[bounding_y1:bounding_y2, bounding_x1:bounding_x2]
    return cropped_to_bounding_box

def separate(image):

    

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store the separated images
    separated_images = []

    # Extract bounding box and image tuple
    bounding_boxes_with_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > 100 and abs(w - h) < max(w, h) * 0.5:
            cropped_image = image[y:y+h, x:x+w]
            bounding_boxes_with_images.append(((x, y, w, h), cropped_image))

    # Sort based on x coordinate of bounding box (from left to right)
    bounding_boxes_with_images.sort(key=lambda b: b[0][0])

    # Store the separated images after sorting
    separated_images = [img for _, img in bounding_boxes_with_images]

    return separated_images

def resize(location, outputpath):
    # Open an image file
    with Image.open(location) as img:
        # Resize the image
        img = img.resize((80, 80), Image.ANTIALIAS)
        # Save it back to disk
        img.save(outputpath, format='PNG')