🎶 Music Genre Classification using Machine Learning

This mini project demonstrates the use of deep learning techniques for classifying music tracks into various genres based on audio feature analysis. The project uses libraries like Librosa, TensorFlow, and Keras to preprocess audio data, extract features, train models, and evaluate performance.

------------------------------------------------------------

📁 Dataset

- Source: GTZAN Genre Classification Dataset – Kaggle
  https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

- Description: 
  The dataset contains 9990 music clips, each 30 seconds long, categorized into 10 music genres:
  blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

- Features Extracted:
  - Spectrogram
  - Chroma STFT
  - Zero Crossing Rate
  - Spectral Centroid
  - MFCCs (Mel Frequency Cepstral Coefficients)

------------------------------------------------------------

🧠 Project Workflow

1. Data Preprocessing
- Loaded audio files using librosa
- Extracted relevant features using librosa.feature
- Encoded categorical labels using LabelEncoder

2. Model Building
- Built a deep learning model using Keras with dense and dropout layers
- Used SparseCategoricalCrossentropy as the loss function
- Optimized using Adam optimizer

3. Model Training
- Trained the model on extracted features and encoded labels
- Used validation split and early stopping

4. Evaluation
- Tested on a hold-out test set
- Evaluated performance using accuracy and confusion matrix

------------------------------------------------------------

📊 Technologies Used

- Python
- Librosa
- TensorFlow & Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

------------------------------------------------------------

📈 Results & Observations

- Good performance in distinguishing genres like classical and metal.
- Some misclassification occurred between similar genres like rock and pop.
- Feature quality directly impacted accuracy.

------------------------------------------------------------

📌 Challenges Faced

- Avoiding overfitting in deep learning model
- Efficient processing of large audio data
- Differentiating overlapping genre characteristics

------------------------------------------------------------

✅ Conclusion

This project showcases how deep learning can be used to automate music genre classification. Although the results are promising, future improvements can include using CNNs on spectrograms and fine-tuning feature selection to improve accuracy.

------------------------------------------------------------

🔗 Dataset Link

Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

------------------------------------------------------------

📂 Folder Structure (Example)

MusicGenreClassification/
├── audio/                  - Original music clips
├── features/               - Extracted features in CSV or NumPy format
├── models/                 - Saved model files
├── notebooks/              - Jupyter notebooks
├── README.txt              - This file
└── requirements.txt        - List of Python libraries used

------------------------------------------------------------

🙋‍♀️ Author

Bhavana M  
BSc (Hons) Computer Science  
RV University, Bengaluru  
GitHub: https://github.com/bhavanagowda04

------------------------------------------------------------
