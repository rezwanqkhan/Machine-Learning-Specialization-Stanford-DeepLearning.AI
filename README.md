# 🧠 Machine Learning Specialization Repository

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <br><br>
  <img src="https://img.shields.io/badge/Stanford-8C1515?style=for-the-badge" alt="Stanford">
  <img src="https://img.shields.io/badge/DeepLearning.AI-0056D2?style=for-the-badge" alt="DeepLearning.AI">
  <img src="https://img.shields.io/badge/Coursera-0056D2?style=for-the-badge&logo=coursera&logoColor=white" alt="Coursera">
</div>

<br>

<div align="center">
  <p><i>A comprehensive collection of assignments, labs, and projects from the Stanford & DeepLearning.AI Machine Learning Specialization</i></p>
</div>

## 📚 Repository Overview

This repository contains my complete work and solutions for the Machine Learning Specialization offered by Stanford University and DeepLearning.AI on Coursera. It includes all completed mandatory programming assignments and most optional labs from the three courses in the specialization. The repository serves as a comprehensive resource for anyone interested in learning machine learning concepts through practical implementation.

## 🗂️ Repository Structure

```
├── 1-Supervised Machine Learning Regression and Classification
│   ├── Week 1 - Introduction to ML, Linear Regression
│   ├── Week 2 - Multiple Linear Regression, Feature Scaling
│   └── Week 3 - Logistic Regression, Regularization
│
├── 2-Advanced Learning Algorithms
│   ├── Week 1 - Neural Networks, Forward Propagation
│   ├── Week 2 - Neural Network Training, Activation Functions
│   ├── Week 3 - Decision Trees, Tree Ensembles
│   └── Week 4 - ML Development Process, Bias/Variance
│
└── 3-Unsupervised Learning, Recommenders, Reinforcement Learning
    ├── Week 1 - Clustering, K-means, Anomaly Detection
    ├── Week 2 - Recommender Systems, Collaborative Filtering
    └── Week 3 - Reinforcement Learning, State-Action Value
```

## 🚀 How to Use This Repository

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/rezwanqkhan/Machine-Learning-Specialization-Stanford-DeepLearning.AI.git
   cd Machine-Learning-Specialization-Stanford-DeepLearning.AI
   ```

2. **Environment setup**

   **Option 1: Conda (Recommended for TensorFlow)**
   ```bash
   # Install Conda if you don't have it already
   # Download from: https://docs.conda.io/en/latest/miniconda.html
   
   # Create a conda environment with Python 3.9
   conda create -n ml_env python=3.9
   
   # Activate the environment
   conda activate ml_env
   
   # Install TensorFlow and other dependencies
   conda install tensorflow numpy pandas matplotlib scikit-learn jupyter
   ```

   **Option 2: Standard Python venv**
   ```bash
   # Create a virtual environment
   python -m venv ml_env
   
   # Activate the environment
   # On Windows:
   ml_env\Scripts\activate
   # On macOS/Linux:
   source ml_env/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **IDE Recommendations**
   - **PyCharm**: Excellent for working with Python projects, but note that for Course 2 (Neural Networks with TensorFlow), you should use the Conda environment option to manage Python version compatibility issues.
   - **VS Code**: Good alternative with excellent Jupyter notebook integration.
   - **JupyterLab**: Perfect for interactive exploration of the notebooks.

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

## 📘 What's Inside

### Course 1: Supervised Machine Learning: Regression and Classification
- **Week 1**: Introduction to Machine Learning, Linear Regression with One Variable
  - Cost function implementation
  - Gradient descent algorithm
- **Week 2**: Linear Regression with Multiple Variables, Feature Scaling, Gradient Descent
  - Feature normalization
  - Vectorized implementations
- **Week 3**: Logistic Regression, Regularization, Gradient Descent for Logistic Regression
  - Binary classification problems
  - Sigmoid function implementation
  - Regularized logistic regression

### Course 2: Advanced Learning Algorithms
- **Week 1**: Neural Networks, Forward Propagation, TensorFlow Implementation
  - Neural network architecture
  - TensorFlow basics
- **Week 2**: Neural Network Training, Activation Functions, Multiclass Classification
  - Backpropagation algorithms
  - ReLU, Sigmoid, and Softmax activations
- **Week 3**: Decision Trees, Tree Ensembles, Random Forests, Boosting
  - Decision tree implementation
  - Random forest applications
- **Week 4**: Advice for Applying Machine Learning, Bias/Variance, Machine Learning Development Process
  - Error analysis techniques
  - Training/testing data splits

### Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning
- **Week 1**: Clustering, K-means Algorithm, Anomaly Detection
  - K-means implementation
  - Anomaly detection systems
- **Week 2**: Recommender Systems, Collaborative Filtering, Content-based Filtering
  - Collaborative filtering algorithm
  - Content-based recommendations
- **Week 3**: Reinforcement Learning, State-Action Value Function, Continuous State Spaces
  - Q-learning implementation
  - Deep Q-learning neural networks

## 🛠️ Technologies Used

- **Python**: Primary programming language for all implementations
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization and result plotting
- **Scikit-learn**: Implementation of various ML algorithms and tools
- **TensorFlow**: Neural network implementation and deep learning frameworks
- **Jupyter Notebooks**: Interactive code development and visualization

## 🎯 Topics Covered

- Linear and Logistic Regression algorithms
- Neural Networks and Deep Learning architectures
- Decision Trees and Ensemble Methods
- Clustering and Anomaly Detection techniques
- Recommender Systems algorithms
- Reinforcement Learning approaches
- Feature Engineering and Selection methods
- Model Evaluation and Validation techniques
- Bias and Variance Analysis
- Hyperparameter Tuning strategies

## 💡 Learning & Practice

Feel free to use these materials to enhance your understanding of machine learning concepts. The notebooks are designed to be interactive and educational. I encourage you to:

- Experiment with the parameters to see how they affect model performance
- Modify the existing code to solve similar problems
- Use these implementations as a foundation for your own machine learning projects
- Practice implementing these algorithms from scratch to deepen your understanding

## ⚠️ Important Notes on TensorFlow and Python Versions

For Course 2 (Advanced Learning Algorithms), TensorFlow requires specific Python versions (Python 3.9 is recommended). Using Conda for environment management solves many compatibility issues:

```bash
# Create TensorFlow-compatible environment
conda create -n tf_env python=3.9
conda activate tf_env
conda install tensorflow
```

This approach helps avoid common dependency issues that occur with TensorFlow installations.

## 📝 Educational Note

This repository is created for educational purposes. The code and solutions are meant to serve as a learning resource for students and practitioners interested in machine learning. While the assignments are from the Coursera specialization, this repository is intended to help others learn through example implementations and thorough documentation.

## 👨‍💻 About the Author

Created with ❤️ by **Rezwanullah Khan QURAISHI**

Connect with me:
- [LinkedIn](https://www.linkedin.com/in/rezwanullah-quraishi-608314260/)
- [Google Play Store](https://play.google.com/store/apps/dev?id=8731190748414981062)

---

<div align="center">
  <h3>💫 Happy Coding! 💫</h3>
  <p><i>"The measure of intelligence is the ability to change."</i> — Albert Einstein</p>
  <p>May your models always converge and your gradients never vanish! 🚀</p>
</div>
