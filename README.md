
# Text Augmentation Using Large Language Models (LLMs)

## Introduction
This project explores the potential of leveraging Large Language Models (LLMs) such as Meta's llama 2-70b to augment text data for handling class imbalance in text classification tasks. Using the AG News dataset, class imbalances were deliberately introduced to evaluate how LLM-based text augmentation compares against traditional methods like SMOTE in improving model performance.

## Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Dataset](#dataset)
4. [Features](#features)
5. [Methodology](#methodology)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results & Analysis](#results--analysis)
9. [Future Work](#future-work)
10. [Contributors](#contributors)
11. [License](#license)

## Objectives
- **Simulate Real-World Imbalances:** Introduce controlled imbalances in the AG News dataset and analyze their effects on classification performance.
- **Experiment with Balancing Techniques:** Compare traditional methods like SMOTE with LLM-generated synthetic data for minority class augmentation.
- **Evaluate Model Performance:** Assess performance using metrics like accuracy, precision, recall, and F1-score, emphasizing improvements for underrepresented classes.

## Dataset
The project uses the AG News dataset:
- **Source:** News articles categorized into four classes: World, Sports, Business, and Science/Technology.
- **Structure:**
  - Training Set: 120,000 samples (30,000 per class)
  - Test Set: 7,600 samples (1,900 per class)
- **Deliberate Modifications:** Science/Technology class instances were reduced using techniques like under-sampling and topic-specific sampling to simulate imbalances.

## Features
- **Class Imbalance Handling:** Investigates four imbalance creation techniques: severe under-sampling, topic-specific sampling, clustered minority sampling, and progressive rarity imbalances.
- **Augmentation Methods:**
  - Traditional: SMOTE
  - Advanced: LLM-generated synthetic text using llama 3b 70b.
- **Model Training:** Logistic Regression and Support Vector Machines (SVM) trained with TF-IDF vectorization.
- **Evaluation Metrics:** Accuracy, Precision, Recall, and F1-score.

## Methodology
### Data Preparation
- **Preprocessing:** Text converted to lowercase, punctuation and digits removed, duplicates and missing values filtered.
- **Feature Extraction:** TF-IDF vectorization to transform text into numerical representations.

### Augmentation
- **SMOTE:** Generates synthetic samples for minority classes using interpolation.
- **LLM (Meta's llama 2-70b):** Contextually prompted synthetic text generation tailored for each imbalance type.

### Model Training
- Logistic Regression and SVM were optimized using GridSearchCV for hyperparameter tuning and stratified cross-validation to ensure robust performance across classes.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/text-augmentation-llm.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the AG News dataset via the Hugging Face library.

## Usage
1. Preprocess the dataset:
   ```bash
   python preprocess.py --input data/ag_news.csv --output data/processed.csv
   ```
2. Create class imbalances:
   ```bash
   python create_imbalance.py --input data/processed.csv --method "clustered"
   ```
3. Augment data using llama 3b 70b:
   ```bash
   python augment_llm.py --input data/imbalanced.csv --output data/augmented.csv
   ```
4. Train models:
   ```bash
   python train_model.py --input data/augmented.csv --model "logistic_regression"
   ```

## Results & Analysis
- **Balanced Dataset:** Logistic Regression and SVM achieved ~92% F1-scores.
- **Imbalanced Dataset:** Performance on minority classes dropped significantly, with recalls often below 0.5.
- **Augmentation Results:**
  - SMOTE improved metrics but struggled in nuanced imbalance scenarios.
  - llama 3b 70b augmentation achieved superior recall (~88.4%) and F1-scores, especially for progressive rarity imbalances.
  
A detailed comparison of augmentation techniques showed that LLMs produced more semantically coherent and contextually diverse synthetic samples.

## Future Work
- **Enhance Augmentation:** Explore dynamic prompting for LLMs to generate more diverse and contextually relevant text.
- **Advanced Models:** Incorporate transformer-based classifiers like BERT or RoBERTa for better performance.
- **Real-World Validation:** Test methodologies on real-world datasets from domains like healthcare and finance.

## Contributors
- **Sejal Agarwal** (UMass Amherst) – [Email](mailto:sejalagarwal@umass.edu)
- **Siddharth Jain** (UMass Amherst) – [Email](mailto:siddharthjai@umass.edu)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
