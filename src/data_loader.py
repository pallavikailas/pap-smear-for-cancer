import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_glcm_features(image, distances=[5], angles=[0]):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    return np.hstack([contrast, dissimilarity, homogeneity])

def extract_features(image, distances=[5], angles=[0]):
    color_hist = extract_color_histogram(image)
    glcm_features = extract_glcm_features(image, distances, angles)
    return np.hstack([color_hist, glcm_features])

def get_data(data_dir='C:\\Users\\admin\\Desktop\\FcvProject\\Herlev Dataset\\train', distances=[5], angles=[0]):
    data, labels = [], []
    
    # Check if the provided directory exists
    if not os.path.exists(data_dir):
        print(f"Directory does not exist: {data_dir}")
        return np.array(data), np.array(labels)

    # Iterate over each class folder
    for class_label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_label)
        if os.path.isdir(class_dir):  # Ensure it's a directory
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Error loading image: {img_path}")
                        continue
                    features = extract_features(image, distances, angles)
                    data.append(features)
                    # Use the class label as the label (you can encode it as needed)
                    labels.append(class_label)  # Store the class label as a string or encode it

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return np.array(data), np.array(labels)

def main(data_dir, distances, angles):
    data, labels = get_data(data_dir, distances, angles)
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)

if __name__ == "__main__":
    data_dir = r'C:\Users\admin\Desktop\FcvProject\Herlev Dataset\train'  # Top-level directory
    distances = [5]
    angles = [0]
    main(data_dir, distances, angles)

