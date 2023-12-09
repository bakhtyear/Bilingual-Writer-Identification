# Bilingual-Writer-Identification
ETE4100-4200 Thesis Project Entitled: "Authorship Identification through Bilingual Handwritten Characters using Convolutional Neural Networks" 
by: Bakhtyear Fahim, 
Supervised by: Dr. Shah Ariful Hoque Chowdhury, Associate Professor,
Department of Electronics &amp; Telecommunication Engineering,
Rajshahi University of Engineering &amp; Technology
_________________________________________________________________________________________________________________________________________________________________________________________________________________________________
A CNN-based system to identify writers from Bengali and English handwritten characters. I have collected handwriting samples from my classmates. Then I have captured those documents using the phone’s camera. Then those photos were cropped, labeled, and augmented. Those images were divided into train data and test data. Then this data set was fed to an existing classifier model to see initial results. After Ablation study and HyperParameter tuning a CNN model with an accuracy of 98.36% was proposed. 
_________________________________________________________________________________________________________________________________________________________________________________________________________________________________
In the code section, you will find all the codes from resizing the images to training the model. In the 'Extras' folder use the 'gpu_async' code to make TensorFlow use Nvidia GPU and CUDA.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Python version: 3.9.13
Tensorflow: 2.7.0 
Nvidia CUDA: 11.2.0
Nvidia cuDNN: 8.1.1

List of Python Libraries Used:- 
os
numpy
cv2
PIL
shutil
matplotlib.pyplot
tensorflow
keras
tensorflow.keras
sklearn.model_selection
tensorflow.keras.optimizers
