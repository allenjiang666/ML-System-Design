# Machine Learning System Design

the ML system typically includes the three key parts you mentioned:

1. **Training**: This phase involves collecting and preprocessing data, choosing a model architecture, training the model on the data, and tuning hyperparameters. The goal is to produce a model that generalizes well to new, unseen data.
    
2. **Deployment**: After the model is trained and validated, it's deployed to a production environment where it can be used to make predictions. This phase involves considerations such as model serving infrastructure, monitoring, scalability, and versioning.
    
3. **Inference**: This phase is where the deployed model is used to make predictions on new data. Inference can happen in real-time (e.g., serving predictions through an API) or in batch mode (e.g., running predictions on a large dataset).

## High level breakdown

### 1\. **Training**

#### 1.1. **Data Collection and Preprocessing**

- **Data Collection**:
    - Identifying and integrating multiple data sources (e.g., databases, APIs, external datasets).
    - Handling data privacy and compliance issues.
- **Data Cleaning**:
    - Removing duplicates and handling missing values.
    - Outlier detection and handling.
- **Data Transformation**:
    - Normalization, standardization, and scaling.
    - Encoding categorical features (e.g., one-hot encoding).
- **Feature Engineering**:
    - Creating new features from existing ones (e.g., feature crosses).
    - Dimensionality reduction techniques (e.g., PCA).
- **Data Splitting**:
    - Dividing data into training, validation, and test sets.
    - Ensuring data stratification if needed.

#### 1.2. **Model Development**

- **Model Selection**:
    - Choosing the type of model (e.g., neural networks, decision trees, SVM).
    - Experimenting with different architectures and configurations.
- **Evaluation Metrics**:
    - Defining metrics like accuracy, precision, recall, F1 score, AUC.
    - Considering domain-specific metrics (e.g., BLEU for NLP).

- **Hyperparameter Tuning**:
    - Search methods (e.g., grid search, random search, Bayesian optimization).
    - Using automated tools (e.g., Hyperopt, Optuna).
- **Validation Strategy**:
    - K-fold cross-validation.
    - Early stopping and learning rate scheduling.

### 2\. **Deployment**

#### 2.1. **Model Packaging**

- **Serialization**:
    - Saving the model in a standardized format (e.g., `.h5`, `.pb`, `.pt`).
    - Ensuring compatibility with different serving platforms.
- **Containerization**:
    - Using Docker to package model with dependencies.
    - Creating reproducible environments.

#### 2.2. **Serving Infrastructure**

- **Real-time Serving**:
    - Using platforms like TensorFlow Serving, TorchServe, or cloud-native solutions (e.g., AWS Sagemaker).
    - Load balancing and horizontal scaling.
- **Batch Serving**:
    - Using tools like Apache Spark or Hadoop for large-scale inference.
    - Scheduling and orchestrating batch jobs.
- **Scalability & Latency Considerations**:
    - Implementing caching strategies.
    - Using GPUs or TPUs for accelerated inference.

#### 2.3. **Monitoring & Logging**

- **Performance Monitoring**:
    - Tracking metrics like response time, throughput.
    - Setting up alerts for performance degradation.
- **Model Drift Detection**:
    - Comparing distribution of incoming data with training data.
    - Setting up alerts for significant deviations.
- **Logging and Auditing**:
    - Capturing inputs and outputs of the model.
    - Keeping track of model versions and configurations.

### 3\. **Inference**

#### 3.1. **Input Data Processing**

- **Data Transformation**:
    - Applying the same preprocessing steps used during training.
    - Handling missing or unexpected input data.
- **Feature Vector Construction**:
    - Generating the final feature vector for prediction.
    - Ensuring consistency with training features.

#### 3.2. **Model Execution**

- **Batch Inference**:
    - Efficiently handling large datasets (e.g., using data loaders).
    - Parallel processing and optimization.
- **Real-time Inference**:
    - Low-latency inference (e.g., using TensorRT for optimized model execution).
    - Handling concurrent requests and scaling.
- **Error Handling**:
    - Managing model input errors or edge cases.
    - Implementing fallback mechanisms.

#### 3.3. **Post-Processing**

- **Result Transformation**:
    - Decoding or interpreting model outputs.
    - Mapping class probabilities to labels, thresholds, etc.
- **Actionable Insights**:
    - Integrating predictions into business workflows (e.g., recommendation engines).
    - Providing explanations (e.g., using SHAP or LIME).
- **Feedback Loop**:
    - Capturing user feedback for future training.
    - Logging data for continuous learning and improvement.


## Common problems

| Problem Type | System Output | Label | Popular Algorithm | Input Data | Preprocessing Steps | Model Evaluation Metrics | Example Use Case |
|--------------|---------------|-------|-------------------|------------|---------------------|--------------------------|-------------------|
| Binary Classification | Class label (0 or 1) | Binary (0 or 1) | Logistic Regression, SVM, Random Forest | Structured data, text, images | Feature scaling, encoding categorical variables, handling missing values | Accuracy, Precision, Recall, F1-score, ROC-AUC | Spam email detection |
| Multiclass Classification | Class label (one of several classes) | Categorical | Random Forest, SVM, Neural Networks | Structured data, text, images | Feature scaling, encoding categorical variables, handling missing values | Accuracy, Precision, Recall, F1-score (macro/micro averaged) | Image classification (e.g., classifying animals) |
| Regression | Continuous value | Continuous value | Linear Regression, Random Forest Regressor, Neural Networks | Structured data, time series | Feature scaling, handling outliers, feature engineering | Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared | House price prediction |
| Clustering | Cluster assignments | Unlabeled | K-means, DBSCAN, Hierarchical Clustering | Structured data, text embeddings | Feature scaling, dimensionality reduction | Silhouette score, Davies-Bouldin index, Calinski-Harabasz index | Customer segmentation |
| Time Series Forecasting | Future values | Historical values | ARIMA, Prophet, LSTM | Time-ordered numerical data | Detrending, seasonality adjustment, lag feature creation | Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE) | Stock price prediction |
| Natural Language Processing (Text Classification) | Class label | Categorical | BERT, RoBERTa, CNN | Text data | Tokenization, stop word removal, lemmatization | Accuracy, F1-score, Confusion matrix | Sentiment analysis of product reviews |
| Computer Vision (Object Detection) | Bounding boxes and class labels | Bounding box coordinates and class labels | YOLO, Faster R-CNN, SSD | Images | Resizing, normalization, data augmentation | Mean Average Precision (mAP), Intersection over Union (IoU) | Detecting objects in autonomous vehicles |
| Recommender Systems | Ranked list of items | User-item interaction data | Collaborative Filtering, Matrix Factorization, Neural Collaborative Filtering | User-item interaction data, item features | Encoding user and item IDs, normalizing ratings | Mean Average Precision at K (MAP@K), Normalized Discounted Cumulative Gain (NDCG) | Movie recommendation on streaming platforms |
