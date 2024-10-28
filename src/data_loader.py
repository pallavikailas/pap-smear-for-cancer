import os
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import StandardScaler

def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_glcm_features(image, distances=[5], angles=[0]):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast').flatten()
    dissimilarity = greycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = greycoprops(glcm, 'homogeneity').flatten()
    return np.hstack([contrast, dissimilarity, homogeneity])

def extract_features(image):
    color_hist = extract_color_histogram(image)
    glcm_features = extract_glcm_features(image)
    return np.hstack([color_hist, glcm_features])

def get_data(data_dir):
    data, labels = [], []
    for class_label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                features = extract_features(image)
                data.append(features)
                labels.append(int(class_label))
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return np.array(data), np.array(labels)
