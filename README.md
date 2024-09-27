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

| **Problem** | **System Output** | **Label** | **Popular Algorithm** | **Input Data** | **Preprocessing Steps** | **Model Evaluation** |
| --- | --- | --- | --- | --- | --- | --- |
| **Recommendation System** | Ranked list of items or content suggestions | Item clicked / viewed / rated | Collaborative Filtering, Matrix Factorization, Deep Learning (DNNs) | User behavior data, item metadata, user profiles | Normalization, handling missing values, feature engineering (e.g., user-item interactions) | Precision, Recall, NDCG, Mean Average Precision (MAP) |
| **Search Ranking System** | Ordered list of search results | Relevance score (0, 1) | Gradient Boosted Trees, RankNet, LambdaMART | Query data, document features, user click data | Tokenization, stemming, normalization, feature scaling | NDCG, MRR (Mean Reciprocal Rank), DCG |
| **Spam/Fraud Detection System** | Binary classification: spam or not spam, fraud or not fraud | Spam/Not Spam, Fraud/Not Fraud | Logistic Regression, Random Forest, XGBoost, LSTM (for sequential data) | Transaction details, email/text content, metadata | Feature encoding (e.g., text encoding), outlier detection, data balancing (SMOTE) | Precision, Recall, F1-Score, ROC-AUC |
| **Ad-Click Prediction** | Probability of user clicking an ad | Clicked/Not Clicked (binary) | Logistic Regression, Factorization Machines, DeepFM, Wide & Deep | User behavior, ad features, user demographic data | Feature scaling, normalization, categorical encoding (one-hot, embeddings) | ROC-AUC, Log-loss, Precision, Recall |
| **Personalized News Feed** | Ranked list of news articles | Clicked, Liked, Shared | Neural Collaborative Filtering, Transformer-based models | User interaction data, article metadata, user profiles | Normalization, handling missing values, contextual feature extraction | Precision, Recall, User engagement metrics (CTR, time spent) |
| **Voice Assistant System** | Text from speech, intents, and actions | Transcript, Intent Label | RNNs, LSTMs, Transformer-based models (BERT, GPT) | Audio signals, user commands, context history | Audio feature extraction (MFCCs, spectrograms), noise reduction, segmentation | WER (Word Error Rate), Intent classification accuracy |
| **Image/Video Classification** | Predicted class label or multiple labels | Image/video category, tags | CNNs, ResNet, YOLO (for detection), Transformer-based models | Image frames, video metadata | Image resizing, normalization, augmentation (rotation, flip), color space conversion | Accuracy, Precision, Recall, mAP (mean Average Precision) |
| **Model Deployment & Monitoring** | Monitoring metrics like latency, error rates, drift detection | Error metrics, drift detection alert | No specific algorithm, focuses on deployment and monitoring tools | Model predictions, feature statistics, data streams | Feature drift detection, performance monitoring, alerting systems | Latency, accuracy, data drift, model stability |
| **Predictive Maintenance System** | Probability of failure within a certain timeframe | Failure / No Failure (binary) | Time Series Analysis, Anomaly Detection (Isolation Forest), LSTM | Sensor data, machine logs, usage patterns | Time-series decomposition, normalization, outlier detection, missing data handling | Precision, Recall, F1-Score, Downtime reduction |
| **Real-Time Content Moderation** | Binary classification or multi-class (e.g., spam, hate speech) | Content category/flag | CNNs for images, RNNs for text, Transformer-based models | Text, images, videos, metadata | Text preprocessing (tokenization, stemming), image resizing and normalization, video frame extraction | Precision, Recall, F1-Score, Real-time latency |
