import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from skimage import feature

 #Load and preprocess the image with noise removal
def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Noise removal using median filter
    img=cv2.medianBlur(img, 5)
    
    return img

#histogram equalization using CLAHE
def histogram_equalization(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return img_eq


#color features using color histograms from multiple color spaces
def color_features(img):
    features = []
    color_spaces = [
        ('RGB', cv2.COLOR_RGB2RGB),
        ('HSV', cv2.COLOR_RGB2HSV),
        ('LAB', cv2.COLOR_RGB2LAB)
    ]
    
    for space_name, conversion in color_spaces:
        converted = cv2.cvtColor(img, conversion)
        for i in range(3):
            hist = cv2.calcHist([converted], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
    
    return np.array(features)

#shape features from a binary mask
def shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(5)
    
    contour = max(contours, key=cv2.contourArea)
    
    # Calculate shape features
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
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


#Segment the nucleus and cytoplasm using adaptive thresholding and morphological operations
def segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
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


#texture features
def extract_lbp_features(img, mask):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    radius = 3
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(img, n_points, radius, method='uniform')

    masked_lbp = cv2.bitwise_and(lbp.astype(np.uint8), lbp.astype(np.uint8), mask=mask)

    hist, _ = np.histogram(masked_lbp.ravel(), bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist

'''features_normalized is the feature vector that is thr resultant of the pipeline defined'''

def analyze_pap_smear(image_path):
    img = load_and_preprocess(image_path)
    img_eq = histogram_equalization(img)
    
    nucleus_mask, cytoplasm_mask = segmentation(img_eq)
    
    features = []
    
    #feature extraction
    features.extend(color_features(img_eq))
    features.extend(shape_features(nucleus_mask))
    features.extend(shape_features(cytoplasm_mask))
    features.extend(extract_lbp_features(img_eq, nucleus_mask))
    features.extend(extract_lbp_features(img_eq, cytoplasm_mask))


    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(np.array(features).reshape(1, -1))
    
    return features_normalized, img_eq, nucleus_mask, cytoplasm_mask

def visualize_results(img, img_eq, nucleus_mask, cytoplasm_mask):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(img_eq)
    plt.title('After Histogram Equalization')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(nucleus_mask, cmap='gray')
    plt.title('Nucleus Segmentation')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(cytoplasm_mask, cmap='gray')
    plt.title('Cytoplasm Segmentation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

