import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from skimage import feature
import os

# Load and preprocess the image with noise removal
def load_and_preprocess(image_path):
    img = cv2.imread(image_path)  # Read the image from the specified path
    if img is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

    # Noise removal using median filter
    img = cv2.medianBlur(img, 5)  # Apply median blur to reduce noise
    
    return img

# Perform histogram equalization using CLAHE
def histogram_equalization(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)  # Split LAB into L, A, and B channels
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Create CLAHE object
    l_eq = clahe.apply(l)  # Apply CLAHE to the L channel
    lab_eq = cv2.merge([l_eq, a, b])  # Merge channels back
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)  # Convert back to RGB
    return img_eq

# Extract color features using color histograms from multiple color spaces
def color_features(img):
    features = []
    color_spaces = [
        ('RGB', cv2.COLOR_BGR2RGB),  # Corrected conversion
        ('HSV', cv2.COLOR_BGR2HSV),
        ('LAB', cv2.COLOR_BGR2LAB)
    ]
    
    for space_name, conversion in color_spaces:
        converted = cv2.cvtColor(img, conversion)  # Convert to specified color space
        for i in range(3):  # Loop through each channel
            hist = cv2.calcHist([converted], [i], None, [32], [0, 256])  # Calculate histogram
            hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten histogram
            features.extend(hist)  # Add to feature list
    
    return np.array(features)

# Extract shape features from a binary mask
def shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    if not contours:
        return np.zeros(5)  # Return zero features if no contours found
    
    contour = max(contours, key=cv2.contourArea)  # Get the largest contour
    
    # Calculate shape features
    area = cv2.contourArea(contour)  # Calculate area
    perimeter = cv2.arcLength(contour, True)  # Calculate perimeter
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0  # Calculate circularity
    
    # Fit ellipse to get eccentricity
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
    else:
        eccentricity = 0
    
    # Calculate convex hull and solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    return np.array([area, perimeter, circularity, eccentricity, solidity])

# Segment the nucleus and cytoplasm using adaptive thresholding and morphological operations
def segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    
    # Nucleus segmentation using Otsu's method
    _, nucleus_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up mask using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    nucleus_mask = cv2.morphologyEx(nucleus_mask, cv2.MORPH_OPEN, kernel)
    nucleus_mask = cv2.morphologyEx(nucleus_mask, cv2.MORPH_CLOSE, kernel)

    # Cytoplasm segmentation using adaptive thresholding
    cytoplasm_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    
    # Clean cytoplasm mask with morphological operations
    cytoplasm_mask = cv2.morphologyEx(cytoplasm_mask, cv2.MORPH_OPEN, kernel)

    return nucleus_mask.astype(np.uint8), cytoplasm_mask.astype(np.uint8)

# Extract texture features using Local Binary Pattern (LBP)
def extract_lbp_features(img, mask):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale if image is colored

    radius = 3  # Radius for LBP
    n_points = 8 * radius  # Number of points for LBP
    lbp = feature.local_binary_pattern(img, n_points, radius, method='uniform')  # Compute LBP

    masked_lbp = cv2.bitwise_and(lbp.astype(np.uint8), lbp.astype(np.uint8), mask=mask)  # Apply mask

    hist, _ = np.histogram(masked_lbp.ravel(), bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))  # Create histogram
    
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize histogram

    return hist

# Analyze a pap smear image and extract features
def analyze_pap_smear(image_path):
    img = load_and_preprocess(image_path)  # Load and preprocess the image
    img_eq = histogram_equalization(img)  # Apply histogram equalization
    
    nucleus_mask, cytoplasm_mask = segmentation(img_eq)  # Segment nucleus and cytoplasm
    
    features = []
    
    # Feature extraction
    features.extend(color_features(img_eq))  # Extract color features
    features.extend(shape_features(nucleus_mask))  # Extract shape features for nucleus
    features.extend(shape_features(cytoplasm_mask))  # Extract shape features for cytoplasm
    features.extend(extract_lbp_features(img_eq, nucleus_mask))  # Extract LBP features for nucleus
    features.extend(extract_lbp_features(img_eq, cytoplasm_mask))  # Extract LBP features for cytoplasm

    scaler = StandardScaler()  # Standardize features
    features_normalized = scaler.fit_transform(np.array(features).reshape(1, -1))  # Normalize features
    
    return features_normalized, img_eq, nucleus_mask, cytoplasm_mask

# Visualize the results of the analysis
def visualize_results(img, img_eq, nucleus_mask, cytoplasm_mask):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.imshow(img)  # Display original image
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(img_eq)  # Display histogram equalized image
    plt.title('After Histogram Equalization')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(nucleus_mask, cmap='gray')  # Display nucleus segmentation
    plt.title('Nucleus Segmentation')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(cytoplasm_mask, cmap='gray')  # Display cytoplasm segmentation
    plt.title('Cytoplasm Segmentation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()  # Show all plots

# Main block to process all images in the specified directory
if __name__ == "__main__":
    data_dir = r"C:\Users\admin\Desktop\FcvProject\Herlev Dataset\train\carcinoma_in_situ"
    for img_name in os.listdir(data_dir):  # Loop through each image in the directory
        img_path = os.path.join(data_dir, img_name)  # Construct full image path
        try:
            features, img_eq, nucleus_mask, cytoplasm_mask = analyze_pap_smear(img_path)  # Analyze the pap smear image
            visualize_results(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), img_eq, nucleus_mask, cytoplasm_mask)  # Visualize results
        except Exception as e:
            print(f"Error processing {img_name}: {e}")  # Print error if occurred
