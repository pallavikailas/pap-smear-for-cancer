# pap-smear-for-cancer

This project is focused on detecting cancer through the analysis of Pap smear images. By applying basic machine learning algorithms, the model aims to classify cell samples and support early diagnosis in cervical cancer screening.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)


## Installation
To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/pallavikailas/pap-smear-for-cancer.git
cd pap-smear-for-cancer
pip install -r requirements.txt
```

## Project Structure
- `data/`: Contains the dataset files, including labelled images of Pap smears.
- `src/`: Source code for data preprocessing, feature extraction, and model training.
- `requirements.txt`: List of dependencies required for the project.

## Dataset
The project utilizes the Herlev Dataset from Kaggle, containing labelled images of Pap smear cells. The labels include normal and various stages of abnormality, aiding in training the model for cancerous vs. non-cancerous cell classification.

## Model
The model uses the ensembling of 3 machine-learning techniques to classify cells as healthy or potentially cancerous based on features extracted from Pap smear images.
