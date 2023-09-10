# text-classification-model-using-naive-bayes

Text classification using Naive Bayes is a popular and effective technique in natural language processing (NLP) and machine learning. Naive Bayes is a probabilistic algorithm that makes predictions based on the Bayes' theorem. It's called "naive" because it assumes that the features used for classification are independent of each other, which is often not the case in text data. Despite this simplifying assumption, Naive Bayes can perform surprisingly well for many text classification tasks.

Here's a step-by-step guide on how to perform text classification using Naive Bayes:

1. **Data Preparation**:
   - Collect and preprocess your text data. This typically involves cleaning the text by removing punctuation, converting text to lowercase, and tokenizing it into words or sub-word units (e.g., using techniques like tokenization or stemming).

2. **Feature Extraction**:
   - Represent each text document as a set of features. In text classification, the most common feature representation is the Bag of Words (BoW) or Term Frequency-Inverse Document Frequency (TF-IDF) vectors. These representations convert text documents into numerical vectors.

3. **Labeling**:
   - Assign labels or categories to each text document. For example, if you're doing sentiment analysis, the labels might be "positive," "negative," or "neutral."

4. **Split the Data**:
   - Divide your dataset into two parts: a training set and a testing set. The training set is used to train the Naive Bayes model, and the testing set is used to evaluate its performance.

5. **Training**:
   - Train a Naive Bayes classifier on the training data using the features you extracted and the corresponding labels. There are different variants of Naive Bayes classifiers, such as Multinomial Naive Bayes and Bernoulli Naive Bayes, which are commonly used for text classification. The choice of which one to use depends on the nature of your data and the specific problem.

6. **Testing and Evaluation**:
   - Use the trained Naive Bayes classifier to predict labels for the test data. Then, compare the predicted labels with the actual labels to evaluate the model's performance. Common evaluation metrics for text classification include accuracy, precision, recall, F1-score, and confusion matrix.

7. **Hyperparameter Tuning**:
   - Experiment with different hyperparameters, such as smoothing techniques (Laplace smoothing or add-one smoothing), to fine-tune your Naive Bayes model for better performance.

8. **Deployment**:
   - Once you're satisfied with the model's performance, you can deploy it to make predictions on new, unseen text data.


This code is a basic example of text classification using Multinomial Naive Bayes. Depending on your specific problem and dataset, you may need to perform more advanced preprocessing and tuning to achieve the best results.

# Example Case

**size:**
  x_train=14997 \n
  x_test=5000   \n
  y_train=14997 \n
  y_test=5000  \n

**frequency graph:**

  <img width="441" alt="image" src="https://github.com/RonSheoran123/spam-detection-model-using-naive-bayes/assets/106268100/e84e3f08-d886-421c-8b4f-4a92d281335f">

  Top 2000 words are taken into consideration

  Score on training data: 0.8935787157431486 \n
  Score on testing data: 0.8544

**Confusion Matrix:**

  <img width="441" alt="image" src="https://github.com/RonSheoran123/spam-detection-model-using-naive-bayes/assets/106268100/5c9409dd-7c6f-419d-ab0b-d2911af876c8">
\n
  <img width="409" alt="image" src="https://github.com/RonSheoran123/spam-detection-model-using-naive-bayes/assets/106268100/b9e5c6f3-b954-4cf3-973c-a7d7586551e0">


