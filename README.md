# Deep Learning Course-University of Haifa

Welcome to my Deep Learning assignments repository. This collection includes implemented solutions and hands-on experiments from several homework assignments during a university course focused on deep learning fundamentals using **PyTorch**.

## Contents

| File                              | Description |
|-----------------------------------|-------------|
| `HW1_DeepLearningSol.ipynb`       | Binary and multi-class classification using feedforward neural networks (FNN). Part 1 predicts whether a person's income exceeds \$50K using tabular data. Part 2 classifies FashionMNIST images using two different FNN architectures with parameter constraints. |
| `HW2_DeepLearning_PT1_Sol.ipynb`  | Image classification on CIFAR-10 using Convolutional Neural Networks (CNNs). Compares 4 variants: baseline CNN, deeper CNN, larger kernels, and average pooling. Includes visualization of filters and feature maps. |
| `HW2_DeepLearning_PT2_Sol.ipynb`  | Transfer learning on a real-world weather classification dataset (Sunrise, Shine, Cloudy, Rain). Fine-tunes a pretrained CNN, applies a custom loss function, and visualizes model performance with confusion matrix and dimensionality reduction. |
| `HW3_DeepLearning_PT1_Sol.ipynb`  | Text generation using an LSTM network trained on song lyrics. Implements multiple sampling strategies (top-k, nucleus), generates songs based on user prompts, and analyzes generation quality and length. |
| `HW3_DeepLearning_PT2_Sol.ipynb`  | Variational Autoencoder (VAE) for MNIST digit images. Implements encoder, decoder, and custom VAE loss from scratch. Visualizes the latent space, interpolates between digits, and generates new samples based on learned representations. |


## Key Topics Covered

-  Linear & Logistic Regression
-  Gradient Descent Optimization
-  Word Embedding (Word2Vec)
-  Sentiment Analysis (IMDb)
-  Perceptron Algorithm & Theory
-  Support Vector Machines (SVM) with RBF & Polynomial Kernels
-  Hyperparameter tuning
-  Data normalization and impact on convergence

## Tech Stack

- **Language**: Python
- **Deep Learning**: PyTorch
- **Data Handling**: pandas, NumPy, torchvision
- **Visualization**: matplotlib, seaborn, WordCloud, t-SNE

## Sample Output

- Decision boundary visualizations for classification problems
- Training accuracy and loss plots
- Feature space transformation illustrations (e.g., RBF kernel)
- Word2Vec embeddings mapped for sentiment prediction
