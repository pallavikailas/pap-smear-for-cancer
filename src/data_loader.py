import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from feature_extraction import load_and_preprocess, histogram_equalization, color_features, shape_features, segmentation, extract_lbp_features

def extract_features(image):
    """
    Extracts features from the image using the defined functions from feature_extraction.
    """
    img_eq = histogram_equalization(image)
    nucleus_mask, cytoplasm_mask = segmentation(img_eq)

    features = []
    # Color features
    features.extend(color_features(img_eq))
    # Shape features for nucleus and cytoplasm
    features.extend(shape_features(nucleus_mask))
    features.extend(shape_features(cytoplasm_mask))
    # LBP texture features for nucleus and cytoplasm
    features.extend(extract_lbp_features(img_eq, nucleus_mask))
    features.extend(extract_lbp_features(img_eq, cytoplasm_mask))
    
    return np.array(features)


def get_data(data_dir):
    data, labels = [], []
    label_encoder = LabelEncoder()  # Create a LabelEncoder instance

    for class_label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                features = extract_features(image)
                data.append(features)
                labels.append(class_label)  # Keep the label as a string

    # Convert labels to numeric values
    labels = label_encoder.fit_transform(labels)
    
    # Standardize the features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    return np.array(data), np.array(labels)

