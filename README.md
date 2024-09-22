# Skin-Cancer-Detector
Our Skin Cancer Detection System uses Convolutional Neural Networks (CNNs) to analyze images and identify skin lesions with 70.79% accuracy. Trained on the HAM10000 dataset, it aims to assist in early detection and will be deployed as a web application for real-time diagnosis, highlighting AI's impact in healthcare.

*Introduction*

I’ve always been fascinated by the potential of deep learning in healthcare. With this project, I set out to create a deep learning model to detect skin cancer from medical images, leveraging the power of Convolutional Neural Networks (CNNs). The project involves training a CNN on a popular skin cancer dataset and then using that trained model to make predictions on new images.

In this documentation, I’ll walk through each step of the process, from working with the dataset to model creation, evaluation, and making predictions. The final goal is to deploy the model for real-world applications, enabling it to assist in the early detection of skin cancer.

1. Dataset Overview

HAM10000 Dataset
For this project, I used the HAM10000 dataset, a well-known dataset in the field of dermatology. The dataset includes 10,015 images of pigmented skin lesions that fall into seven categories, including both benign and malignant conditions. These categories represent different types of skin diseases such as melanoma, basal cell carcinoma, and benign nevi.

The images in HAM10000 are labeled, making it ideal for training a supervised machine learning model like a CNN. The dataset also contains metadata like lesion type, age, and sex of the patient, but for this model, I focused solely on image data to detect skin cancer.

Data Preprocessing
To ensure uniformity across the dataset, I applied the following preprocessing steps:

Image Resizing: All images were resized to 128x128 pixels.
Normalization: The pixel values were scaled to fall between 0 and 1 to improve the model's performance during training.
#
from keras.preprocessing.image import ImageDataGenerator

# Image preprocessing
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    'path_to_train_data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')

validation_generator = datagen.flow_from_directory(
    'path_to_validation_data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')

2. Model Architecture
The backbone of this project is a Convolutional Neural Network (CNN). CNNs are especially effective for image classification tasks due to their ability to automatically extract important features from images.

CNN Layers
Here’s a summary of the layers in the model I built:

Convolutional Layers: Extracts features using filters.
MaxPooling Layers: Reduces the spatial dimensions of feature maps.
Flatten Layer: Converts 2D matrices into a 1D vector.
Dense (Fully Connected) Layers: Classifies the image based on the extracted features.
Output Layer: Uses a sigmoid activation function to predict if the lesion is cancerous or not.
Model Summary

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

3. Model Training and Evaluation
Once the model architecture was set, I trained the CNN using the HAM10000 dataset.

Training Process
Optimizer: Adam
Loss Function: Binary Crossentropy (since this is a binary classification task)
Metrics: Accuracy
The model was trained for 20 epochs with a batch size of 32.

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)
Test Data Evaluation
To check how well the model performs on unseen data, I evaluated it using a separate test set.

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
The model achieved an accuracy of 70.79% on the test set, which shows that it can differentiate between cancerous and non-cancerous images to a reasonable degree. This provides a solid foundation for further improvements.

4. Saving and Loading the Model
After training the model, I saved it so that I can easily reuse or improve it without retraining from scratch.

model.save('/content/drive/My Drive/SkinCancerDetector/my_model.h5')
To reload the model later, I can use the following code:

from keras.models import load_model

model = load_model('/content/drive/My Drive/SkinCancerDetector/my_model.h5')
5. Making Predictions on New Images
I created a user-friendly script to upload a new image and check whether it shows signs of skin cancer. Here’s how the process works:

Upload Image: The user uploads an image for diagnosis.
Preprocessing: The image is resized to 128x128 pixels and normalized.
Prediction: The CNN model predicts whether the image shows skin cancer and outputs the confidence level.
python
Copy code
from keras.preprocessing import image
import numpy as np

def predict_skin_cancer(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Load the saved model
    model = load_model('/content/drive/My Drive/SkinCancerDetector/my_model.h5')
    
    # Make prediction
    predictions = model.predict(img_array)
    confidence = predictions[0][0]
    
    # Interpretation
    if confidence >= 0.5:
        result = "Skin Cancer Detected"
    else:
        result = "No Skin Cancer"
    
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence * 100:.2f}%")
    
# Example usage
predict_skin_cancer('/path_to_new_image.jpg')
This script outputs the prediction result along with the confidence level to inform the user about the potential risk.

6. Future Scope: Deployment
While I’ve built and tested this model successfully, the next big step is to deploy it as a web application so that it can be used by anyone. My plan is to:

Deploy the model using Flask or FastAPI.
Create a simple web interface where users can upload their skin images.
Display the prediction results and confidence levels in a user-friendly format.
Integrate the deployment on a cloud platform like Heroku or AWS to ensure scalability and accessibility.
Deploying this model will help it transition from a research project to a real-world tool that can potentially assist in early skin cancer detection.

Conclusion
This project was an insightful journey into applying deep learning techniques to solve a real-world healthcare problem. With the HAM10000 dataset, I was able to train a Convolutional Neural Network that can detect skin cancer with a test accuracy of 70.79%. While there is room for improvement, the results so far are promising.

In the future, I plan to deploy this model, so it can be accessed by anyone, making it a useful tool in early skin cancer detection and awareness.

