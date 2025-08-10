# Phishing URL Detection using Machine Learning

This project implements various machine learning models to detect phishing URLs. Phishing is a type of online fraud where attackers attempt to obtain sensitive information by impersonating legitimate entities. This project aims to build a robust model to identify and flag such malicious URLs.

## Dataset

Data is collected from Kaggle.
Here is the dataset link: https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls?resource=download

## Methodology

The project follows a standard machine learning pipeline:

1.  **Data Loading and Exploration**: The dataset is loaded and basic statistics and label distribution are examined.
2.  **Data Preprocessing**: URLs are preprocessed, and labels are converted to a binary format (0 for good, 1 for bad).
3.  **Feature Engineering**: New features are created from the URLs, including:
    *   URL length
    *   Number of special characters
    *   Presence of common phishing keywords
    *   Domain length
4.  **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the URL text data into numerical features. An n-gram range of (1, 2) is used to capture both individual words and common two-word phrases.
5.  **Model Training and Evaluation**: Several classification models are trained and evaluated on the dataset:
    *   Logistic Regression
    *   Decision Tree
    *   Random Forest
    *   Support Vector Machine (SVM)
    *   Naive Bayes
    *   Gradient Boosting
    *   AdaBoost
    *   XGBoost
6.  **Hyperparameter Tuning**: GridSearchCV with cross-validation is used to find the optimal hyperparameters for the best-performing models (SVM and Logistic Regression).
7.  **Model Comparison**: The performance of all models, including the tuned ones, is compared based on accuracy and other relevant metrics.
8.  **Interactive URL Checker**: An interactive tool is created using `ipywidgets` to allow users to input a URL and get a prediction from the best-performing model.

## Results

The models were evaluated based on their accuracy on the test set. The initial evaluation with TF-IDF features showed promising results across several models. The inclusion of n-grams in the TF-IDF vectorization generally improved the performance of the models. Hyperparameter tuning on the SVM and Logistic Regression models further optimized their performance on the cross-validation sets.

The final comparison of tuned models on the test set showed competitive accuracies, with SVM and Logistic Regression achieving high performance in detecting both legitimate and phishing URLs.

Detailed accuracy results and confusion matrices for each model can be found in the notebook.

## How to Run the Notebook

1.  **Clone the repository**: Clone this GitHub repository to your local machine.
2.  **Open in Google Colab**: Upload the `phishing_url_detection.ipynb` notebook to Google Colab.
3.  **Mount Google Drive**: Ensure your Google Drive is mounted in the Colab notebook to access the dataset. You might need to adjust the `file_path` in the notebook to match the location of the dataset on your Google Drive.
4.  **Run all cells**: Execute all the code cells in the notebook sequentially.
5.  **Use the URL Checker**: Once all cells have run successfully, the interactive URL checker widget will appear at the end of the notebook. Enter a URL in the text box and click "Check URL" to get a prediction.

## Dependencies

The project requires the following libraries:

*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn
*   ipywidgets
*   xgboost

These dependencies can be installed using pip as shown in the notebook.

## Future Work

*   Explore more advanced feature engineering techniques, such as using features extracted from the HTML content of the URLs (if accessible and permissible).
*   Investigate deep learning models for URL classification.
*   Integrate the model into a web application or browser extension for real-time phishing detection.
*   Continuously update the dataset with new URL examples to keep the model up-to-date with evolving phishing techniques.
