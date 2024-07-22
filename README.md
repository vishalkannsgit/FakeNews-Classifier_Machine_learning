# Fake News Classifier

## Project Title
Fake News Classifier

## Skills Takeaway From This Project
- Data cleaning
- Data augmentation
- Text pre-processing
- Feature extraction
- Sentiment Analysis
- Model Selection
- Model training and evaluation
- Hyper-parameter tuning
- Natural Language Processing (NLP)

## Domain
Media and Entertainment

## Problem Statement
Fake News classifier - create a neural network-based deep learning model to classify fake news.

## Business Use Cases
The proliferation of fake news on digital platforms poses significant risks to society by spreading misinformation and influencing public opinion. Developing an effective fake news classification system using deep learning can help mitigate these risks by automatically identifying and flagging misleading content. This project aims to build a robust and scalable deep learning model to classify news articles as either genuine or fake.

## Approach

### Tasks and Methodology
1. **Data Collection and Preprocessing**:
   - Download the data from the given link.
   - Clean and preprocess the data to handle missing values, outliers, and categorical variables.

2. **Exploratory Data Analysis**:
   - Perform EDA to understand data distributions and relationships between variables.

3. **Feature Engineering**:
   - Create new features that could enhance the predictive power of the model.

4. **Model Development**:
   - Create a baseline model.
   - Develop deep learning models: RNNs and Transformer-based models.

5. **Model Training and Evaluation**:
   - Evaluate the performance of various models developed.

6. **Model Selection**:
   - Compare various models using appropriate evaluation metrics.

7. **Hyper-parameter Tuning**.

### Results
The project aims to develop a high-performing fake news classifier capable of:
- Achieving a high F1-score to balance precision and recall.
- Handling diverse and large-scale datasets efficiently.
- Providing reliable predictions that can be integrated into news platforms to flag potentially fake news articles.

- ## Technical Tags
- Natural Language Processing (NLP)
- Text Preprocessing
- Tokenization
- Stemming
- Lemmatization
- Stop Word Removal
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Word Embeddings
- Word2Vec
- GloVe (Global Vectors for Word Representation)
- BERT (Bidirectional Encoder Representations from Transformers)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Units)

## Data Set
You can find the dataset here: [Fake News Dataset](https://drive.google.com/file/d/1Fd6qQOfUn-xgrcUAJjmdkIGzsV87_O3H/view?usp=drive_link)

### Data Set Explanation
(WELFake) is a dataset of 72,134 news articles with 35,028 real and 37,106 fake news. For this, authors merged four popular news datasets (i.e. Kaggle, McIntire, Reuters, BuzzFeed Political) to prevent over-fitting of classifiers and to provide more text data for better ML training. Dataset contains four columns:
- **Serial number**: Starting from 0.
- **Title**: About the text news heading.
- **Text**: About the news content.
- **Label**: 0 = fake and 1 = real.

There are 78,098 data entries in the CSV file out of which only 72,134 entries are accessed as per the data frame. The title column has the heading of the news article and text has the complete article. The label is the target column. You need to convert the title and text columns into mathematical embeddings to train a model.
