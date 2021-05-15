# CycleGAN-For-MRI

## Problem Statement
Misdiagnosis in the medical field is a very serious issue but it’s also uncomfortably common to occur. Imaging procedures in the medical field requires an expert radiologist’s opinion since interpreting them is not a simple binary process ( Normal or Abnormal). Even so, one radiologist may see something that another does not. This can lead to conflicting reports and make it difficult to effectively recommend treatment options to the patient.

One of the complicated tasks in medical imaging is to diagnose MRI(Magnetic Resonance Imaging). Sometimes to interpret the scan, the radiologist needs different variations of the imaging which can drastically enhance the accuracy of diagnosis by providing practitioners with a more comprehensive understanding.


But to have access to different imaging is difficult and expensive. With the help of deep learning, we can use style transfer to generate artificial MRI images of different contrast levels from existing MRI scans. This will help to provide a better diagnosis with the help of an additional image.

Let us build a Generative adversarial model(modified U-Net) which can generate artificial MRI images of different contrast levels from existing MRI scans.

## Data Set
The data can be downloaded from [here](https://github.com/BharathSD/CycleGAN-For-MRI/blob/main/MRI%2BT1_T2%2BDataset.RAR).\
Extract the rar file which has two folders <b>Tr1</b> and <b>Tr2</b>, which represents T1 weighted and T2 weighted MRI's respectively.\
Tr1 folder has <b>TrainT1</b> which has 43 T1 weighted MRI images.\
Tr2 folder has <b>TrainT2</b> which has 46 T2 weighted MRI images. 

## Package Dependencies
- numpy - 1.19.2
- tensorflow - 2.4.1 (if u have a GPU, use a GPU version)
- matplotlib - 3.3.2
- skimage - 0.17.2

## Pipeline
1. Importing Libraries
2. Data Loading and Visualization
3. Data Preprocessing
4. Model Building
5. Model Training

## Python Files
1. Utils.py : Contains utility class for data handling and some utility function
2. GifCreator.py : Utility code to create Gif using the images
3. CycleGAN.py : Contains cycle GAN utility code which handles training and visualization of the Cycle GAN results
4. main.py : Contains the network definitions for discriminator and generator.

## Running the project
python main.py

## side note
modify main.py file accordingly to change epochs, batch size, training data paths and netwrok definitions
