# Skin Cancer Detection Using CNN with Streamlit Deployment

[Open in Google Colab](https://colab.research.google.com/drive/1Q55Xse-NMl-t6cMzkQC2tRUHrXgurMnN?usp=sharing)

A deep learning system that classifies skin lesions as benign or malignant using Convolutional Neural Networks (CNN). Achieves 90%+ accuracy on the HAM10000 dataset and features a Streamlit web interface with AI-powered dermatologist chat.

## Features
- 🖼️ Image upload interface for skin lesion analysis
- 🧠 CNN model with 90%+ validation accuracy
- 💬 Integrated Gemini AI for virtual dermatologist consultation
- 📊 Confidence level visualization with color-coded alerts
- 🔄 Session persistence and chat history
- ☁️ Cloud-ready deployment configuration

## Installation
pip install tensorflow keras streamlit pyngrok google-generativeai pillow

1. Clone repository:
git clone https://github.com/Yashwanth2408/Skin-Cancer-Detector.git
cd skin-cancer-detection

2. Add Gemini API key:
In app.py
genai.configure(api_key="YOUR_API_KEY")

## Usage
streamlit run app.py

For Colab deployment:
!ngrok authtoken YOUR_NGROK_TOKEN
!streamlit run app.py &>/dev/null&

## Project Structure
├── app.py # Streamlit interface
├── my_model.h5 # Trained CNN model
├── requirements.txt # Dependencies
└── README.md # Documentation

## Dataset Details
HAM10000 Dataset (10,015 images across 7 classes):

| Class ID | Description                 |
|----------|-----------------------------|
| nv       | Melanocytic nevi           |
| mel      | Melanoma                    |
| bkl      | Benign keratosis-like lesions|
| bcc      | Basal cell carcinoma        |
| akiec    | Actinic keratoses           |
| vasc     | Vascular lesions            |
| df       | Dermatofibroma              |

## Model Architecture

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```



## Evaluation Metrics
- Test Accuracy: 90.23%
- Precision: 89.7%
- Recall: 91.1%
- F1-Score: 90.4%

## Deployment
1. Create free accounts on:
   - [Ngrok](https://ngrok.com/) for tunneling
   - [Google AI Studio](https://aistudio.google.com/) for Gemini API

2. Replace in code:
   - `YOUR_API_KEY` with Gemini key
   - `YOUR_NGROK_TOKEN` with Ngrok auth token

3. Run deployment cell in Colab:
!ngrok authtoken YOUR_NGROK_TOKEN
!streamlit run app.py &>/dev/null&

## Future Enhancements
- Expand dataset with real patient cases
- Add multi-class classification support
- Implement DICOM medical imaging standard
- Develop mobile app version
- Integrate telemedicine features

## Download Trained Data Files

The trained data files required for this project are stored on Google Drive due to their size.

[Download trained data files from Google Drive](https://drive.google.com/drive/folders/13bC0VTjWUu4NOqkKbZuO6-gnXR7Qab93?usp=sharing)


## FAQ
**Q: How accurate is the model?**  
A: Current validation accuracy exceeds 90%, but always consult a dermatologist for medical diagnosis.

**Q: Can I use my own images?**  
A: Yes! The system accepts JPG/PNG images of skin lesions.

**Q: How to handle prediction errors?**  
A: Ensure images are well-lit, focused on lesion, and minimum 500x500 resolution.

## License
[MIT License](LICENSE)

## Acknowledgments
- HAM10000 dataset providers
- Streamlit development team
- Google Gemini API team
