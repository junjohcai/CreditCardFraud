# Credit Card Fraud Detection — Custom MLP in C#

A real-world **Binary Classification** project using a hand-built 3-layer Multi-Layer Perceptron (MLP) with backpropagation, applied to the Kaggle Credit Card Fraud Detection dataset.

## Dataset Description

**Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Detail | 
| ----- | ----- | 
| File | `creditcard.csv` (\~144 MB) | 
| Rows | 284,807 transactions | 
| Features | `Time`, `V1`–`V28` (PCA-anonymised), `Amount` — **30 total** | 
| Label | `Class`: `0` = Legitimate, `1` = Fraudulent | 
| Class imbalance | \~0.17% fraud (492 out of 284,807) | 

The dataset contains real credit card transactions made by European cardholders in September 2013. Features V1–V28 are the result of a PCA transformation to protect user confidentiality. `Time` and `Amount` are the only non-transformed features.

## Network Architecture

```
Input Layer   →   Hidden Layer   →   Output Layer
  30 neurons        16 neurons          1 neuron
  (features)     (sigmoid, bias)    (sigmoid → ≥0.5 = Fraud)

```

* **Activation:** Sigmoid on all layers

* **Loss:** Mean Squared Error (MSE)

* **Backpropagation:** Implemented from scratch in `BackNeural.cs`

## How to Run

### Prerequisites

* [.NET 10 SDK](https://dotnet.microsoft.com/download)

* `creditcard.csv` placed in the project root directory

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/junjohcai/CreditCardFraud.git
cd CreditCardFraud

```

**2. Download the dataset**

* Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

* Click **Download** and extract the zip

* Copy **`creditcard.csv`** into the project root folder so it looks like this:

```
CreditCardFraud/
├── creditcard.csv        ← place it here
├── Program.cs
├── BackNeural.cs
├── HiddenN.cs
├── InputN.cs
├── OuputN.cs
├── README.md
└── CreditCardFraud.csproj

```

> **Note:** `creditcard.csv` is not included in this repository because it exceeds GitHub's 100MB file size limit (\~144MB). You must download it manually from Kaggle.

**3. Run the project**

```bash
dotnet run

```

The program will:

1. Load and parse `creditcard.csv`

2. Normalise all features to \[0, 1\] using Min-Max scaling

3. Shuffle and split into 80% train / 20% test

4. Train the MLP for 500 epochs (printing loss every 100 epochs)

5. Save a `training_log.csv` with loss and accuracy per epoch

6. Print Accuracy, Precision, Recall, and F1 Score on the test set

## Hyperparameters

Defined as constants at the top of `Program.cs` for easy experimentation:

| Parameter | Value | Description | 
| ----- | ----- | ----- | 
| `HIDDEN_SIZE` | 16 | Number of hidden neurons | 
| `TRAIN_RATIO` | 0.80 | 80% train, 20% test | 
| `EPOCHS` | 500 | Training iterations over dataset | 
| `LOG_INTERVAL` | 100 | Print loss every N epochs | 
| `THRESHOLD` | 0.5 | Sigmoid cutoff for fraud classification | 
| `LRPOUT` | 0.2 | Learning rate for output layer weights | 
| `LRPIN` | 0.15 | Learning rate for input layer weights | 

### Hyperparameter Experimentation Results

| Run | Hidden Neurons | Epochs | F1 Score | Loss | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 16 | 500 | 92.30% | 0.013000 | baseline |
| 2 | 8 | 500 | 92.94% | 0.012920 | fewer neurons |
| 3 | 32 | 500 | 92.94% | 0.012950 | more neurons |
| 4 | 16 | 200 | 91.67% | 0.013277 | fewer epochs |

After doing separate configurations and runs, the models with 8 and 32 neurons with 500 epochs performed the best, both achieving a peak F1 score of 92.94%. The 8-neuron configuration resulted in a slightly lower loss of 0.012920 compared to the 32-neuron configuration which got 0.012950, which means that the lower number of neurons proved to provide an optimal balance without adding unnecessary complications to the setup. Moreover, the final run which reduced the amount of epochs performed the worst amongst them all, resulting in a higher loss and a lower F1 score.

## Evaluation Metrics

Because the dataset is heavily imbalanced (\~0.17% fraud), **Accuracy alone is misleading**. The following metrics are all reported:

| Metric | Formula | Why it matters | 
| ----- | ----- | ----- | 
| **Accuracy** | (TP+TN) / Total | Overall correctness | 
| **Precision** | TP / (TP+FP) | Of flagged fraud, how many were real | 
| **Recall** | TP / (TP+FN) | Of actual fraud, how many were caught | 
| **F1 Score** | 2 × P × R / (P+R) | Balanced metric for imbalanced classes | 

A full **Confusion Matrix** is also printed to the console.

## Output Files

| File | Description | 
| ----- | ----- | 
| `training_log.csv` | Epoch, AvgMSELoss, TrainAccuracy — use this to plot the loss curve | 

## Project Structure

| File | Role | 
| ----- | ----- | 
| `Program.cs` | Entry point: CSV loading, normalisation, train/test split, training loop, evaluation | 
| `BackNeural.cs` | `NeuralNet` class — forward pass, backprop, weight update, save/load | 
| `HiddenN.cs` | `HNeuron` — hidden layer neuron with weights, bias, error calculation | 
| `InputN.cs` | `INeuron` — input layer neuron with weight-to-hidden connections | 
| `OuputN.cs` | `ONeuron` — output layer neuron with bias and activation | 
| `creditcard.csv` | Dataset (not committed to git if large) | 
| `README.md` | This file | 