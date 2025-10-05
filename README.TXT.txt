README.TXT

# Spanish Regional Flag Classification Project

## Project Overview

This project implements a deep learning solution for classifying Spanish regional flags using FastAI and computer vision techniques. The model achieves 98.41% accuracy in identifying different autonomous community flags.

## Project Structure:

### Data Preparation

	Dataset: Images of Spanish regional flags organized in class-specific folders

	Preprocessing:

		Image resizing to 192x192 pixels using squish method

		Train-validation split (80%-20%) with reproducible random seed

		Data augmentation with lighting variations (no rotation)

### Model Architecture

	Base Model: ResNet18 pre-trained on ImageNet

	Transfer Learning: Fine-tuned for flag classification task

	Training: 5 epochs with discriminative learning rates

### Performance Results

		Final Validation Accuracy: 98.41%

		Error Rate: 1.59%

		Training Time: ~1 minute total
		
		Key Achievement: Near-perfect classification with minimal training time

## Technical Implementation

### Libraries Used

	FastAI (core and vision modules)

	Gradio for web interface

	Standard Python libraries (Path, datetime, warnings)

### Key Features

	Data Pipeline: Efficient DataBlock configuration with proper splits and transforms

	Model Training: Transfer learning with ResNet18 backbone

	Performance Analysis: Confusion matrix and top losses visualization

	Deployment Ready: Gradio web application for easy sharing


## File Structure

project/
├── ccaa_flag_model.pkl          # Trained model file
├── flags_train_notebook.ipynb      # Complete training pipeline
├── app.py               # Web application interface
├── requirements.txt            # Dependencies
└── "/home/obraisan/datasets"/
    └── banderas/          # Dataset directory
        ├── andalucia/
        ├── cataluna/
        ├── madrid/
        └── ... 

## Using the Web Application

> pip install -r requirements.txt
python gradio_app.py


Visit http://localhost:7860 to access the classifier interface.


## Making Predictions


> from fastai.vision.all import *
model = load_learner('flag_classifier.pkl')
prediction = model.predict('flag_image.jpg')


## Model Performance Insights


### Strengths

Excellent accuracy (98.41%) with limited training data

Fast convergence (2 epochs to reach >98% accuracy)

Robust to lighting variations

Generalizes well to unseen validation data


### Limitations

Manual rotation of training images limited augmentation options

Performance dependent on image quality and flag visibility

May struggle with unusual flag orientations or extreme lighting (some flags can be displayed with different symbols, model might not recognize some of them)


## Future Improvements

Expand dataset with more varied flag images

Add support for historical or modified flag versions

Create mobile application version


## Conclusion


This project demonstrates the effectiveness of transfer learning for specialized image classification tasks. The ResNet18 model, fine-tuned on regional flag data, achieves production-ready accuracy with minimal computational resources. The included Gradio interface makes the model accessible for practical use and demonstration purposes.


Project Completion Date: 5/10/2025
Final Model Accuracy: 98.41%
Status: Ready for deployment
