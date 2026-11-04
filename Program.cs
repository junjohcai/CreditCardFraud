// Credit Card Fraud Detection using a custom Multi-Layer Perceptron (MLP) in C#.
// This program loads a real-world dataset, trains the neural network we built,
// and evaluates how well it detects fraudulent transactions.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Backprop;

// Hyperparameters - change these to experiment with different configurations
const int    HIDDEN_SIZE   = 16;     // number of neurons in the hidden layer
const double TRAIN_RATIO   = 0.80;   // 80% of data used for training, 20% for testing
const int    EPOCHS        = 500;    // how many times we loop through the training data
const int    LOG_INTERVAL  = 100;    // print the loss every 100 epochs
const double THRESHOLD     = 0.5;   // if output >= 0.5 we classify it as fraud
const int    LEGIT_SAMPLE  = 5000;  // how many legitimate transactions to include in training
                                     // (we keep all fraud cases, but limit legit to avoid class imbalance)

// Build the path to the CSV file, which should be in the project root folder
string csvPath = Path.GetFullPath(
    Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "creditcard.csv"));

Console.WriteLine("===========================================");
Console.WriteLine("  Credit Card Fraud Detection - MLP (C#)");
Console.WriteLine("===========================================");
Console.WriteLine($"Dataset : {csvPath}");
Console.WriteLine($"Network : 30 -> {HIDDEN_SIZE} -> 1");
Console.WriteLine($"Epochs  : {EPOCHS}  |  Train/Test: {TRAIN_RATIO*100:0}%/{(1-TRAIN_RATIO)*100:0}%");
Console.WriteLine();

// Step 1 - Load the dataset from CSV
// Each row has 30 features (Time, V1-V28, Amount) and a Class label (0 = legit, 1 = fraud)
Console.WriteLine("[1] Loading dataset...");

var allInputs = new List<double[]>();
var allLabels = new List<double>();

using (var sr = new StreamReader(csvPath))
{
    string? header = sr.ReadLine(); // skip the header row
    string? line;
    while ((line = sr.ReadLine()) != null)
    {
        if (string.IsNullOrWhiteSpace(line)) continue;

        string[] parts = line.Split(',');

        // Read the first 30 columns as input features
        double[] features = new double[30];
        for (int i = 0; i < 30; i++)
            features[i] = double.Parse(parts[i], CultureInfo.InvariantCulture);

        // The last column is the class label (strip quotes if present)
        string classStr = parts[30].Trim('"');
        double label    = double.Parse(classStr, CultureInfo.InvariantCulture);

        allInputs.Add(features);
        allLabels.Add(label);
    }
}

int totalSamples = allInputs.Count;
Console.WriteLine($"    Loaded {totalSamples:N0} samples.");

// Step 1b - Stratified Sampling
// The dataset has ~284K rows which would take hours to train on.
// Instead, we keep ALL fraud cases and randomly sample a portion of legit cases.
// This keeps the dataset manageable while preserving the fraud signal.
Console.WriteLine($"[1b] Stratified sampling (all fraud + {LEGIT_SAMPLE:N0} legit)...");

var sampledInputs = new List<double[]>();
var sampledLabels = new List<double>();
int legitCount    = 0;

// Shuffle first so our legit sample is random, not just the first N rows
var preShuffleRng = new Random(42);
var preIdx = new List<int>(totalSamples);
for (int i = 0; i < totalSamples; i++) preIdx.Add(i);
for (int i = totalSamples - 1; i > 0; i--)
{
    int j = preShuffleRng.Next(i + 1);
    (preIdx[i], preIdx[j]) = (preIdx[j], preIdx[i]);
}

foreach (int idx in preIdx)
{
    if (allLabels[idx] == 1) // always include fraud rows
    {
        sampledInputs.Add(allInputs[idx]);
        sampledLabels.Add(1);
    }
    else if (legitCount < LEGIT_SAMPLE) // include legit rows up to the limit
    {
        sampledInputs.Add(allInputs[idx]);
        sampledLabels.Add(0);
        legitCount++;
    }
}

int fraudCount = sampledLabels.Count - legitCount;
Console.WriteLine($"    Sample: {sampledInputs.Count:N0} rows  (legit={legitCount:N0}, fraud={fraudCount:N0})");

allInputs    = sampledInputs;
allLabels    = sampledLabels;
totalSamples = allInputs.Count;

// Step 2 - Normalize features using Min-Max scaling to range [0, 1]
// This is important because the neural network performs better when all inputs
// are on the same scale instead of wildly different ranges.
Console.WriteLine("[2] Normalising features (Min-Max → [0,1])...");

double[] featureMin = new double[30];
double[] featureMax = new double[30];

for (int f = 0; f < 30; f++)
{
    featureMin[f] = double.MaxValue;
    featureMax[f] = double.MinValue;
}

foreach (double[] row in allInputs)
{
    for (int f = 0; f < 30; f++)
    {
        if (row[f] < featureMin[f]) featureMin[f] = row[f];
        if (row[f] > featureMax[f]) featureMax[f] = row[f];
    }
}

// Apply the normalization formula: (value - min) / (max - min)
foreach (double[] row in allInputs)
{
    for (int f = 0; f < 30; f++)
    {
        double range = featureMax[f] - featureMin[f];
        row[f] = (range == 0.0) ? 0.0 : (row[f] - featureMin[f]) / range;
    }
}

Console.WriteLine("    Normalisation complete.");

// Step 3 - Shuffle the data and split into training and test sets
// We use a fixed seed (42) so results are reproducible every run.
Console.WriteLine("[3] Shuffling and splitting dataset...");

var rng     = new Random(42);
var indices = new List<int>(totalSamples);
for (int i = 0; i < totalSamples; i++) indices.Add(i);
for (int i = totalSamples - 1; i > 0; i--)
{
    int j = rng.Next(i + 1);
    (indices[i], indices[j]) = (indices[j], indices[i]);
}

int trainCount = (int)(totalSamples * TRAIN_RATIO);
int testCount  = totalSamples - trainCount;

double[][] trainInputs = new double[trainCount][];
double[]   trainLabels = new double[trainCount];
double[][] testInputs  = new double[testCount][];
double[]   testLabels  = new double[testCount];

for (int i = 0; i < trainCount; i++)
{
    trainInputs[i] = allInputs[indices[i]];
    trainLabels[i] = allLabels[indices[i]];
}
for (int i = 0; i < testCount; i++)
{
    testInputs[i] = allInputs[indices[trainCount + i]];
    testLabels[i] = allLabels[indices[trainCount + i]];
}

int trainFraud = 0, trainLegit = 0;
foreach (double l in trainLabels) { if (l == 1) trainFraud++; else trainLegit++; }
Console.WriteLine($"    Train: {trainCount:N0}  (legit={trainLegit:N0}, fraud={trainFraud:N0})");
Console.WriteLine($"    Test : {testCount:N0}");

// Step 4 - Create the neural network
// Architecture: 30 input neurons -> 16 hidden neurons -> 1 output neuron
// The output is a value between 0 and 1; above 0.5 means we predict fraud.
Console.WriteLine("[4] Building network  30 → {0} → 1 ...", HIDDEN_SIZE);
var net = new NeuralNet(30, HIDDEN_SIZE, 1);

// Step 5 - Train the network
// For each epoch, we feed every training sample through the network,
// calculate how wrong it was (loss), and update the weights via backpropagation.
Console.WriteLine("[5] Training...");
Console.WriteLine();
Console.WriteLine($"{"Epoch",-10} {"Avg MSE Loss",-18} {"Train Acc",-14} {"Notes"}");
Console.WriteLine(new string('-', 60));

var lossLog = new List<(int epoch, double loss, double acc)>();

for (int epoch = 1; epoch <= EPOCHS; epoch++)
{
    double totalLoss = 0.0;
    int    correct   = 0;

    // Shuffle training samples each epoch so the network doesn't memorize the order
    for (int i = trainCount - 1; i > 0; i--)
    {
        int j = rng.Next(i + 1);
        (trainInputs[i], trainInputs[j]) = (trainInputs[j], trainInputs[i]);
        (trainLabels[i], trainLabels[j]) = (trainLabels[j], trainLabels[i]);
    }

    for (int s = 0; s < trainCount; s++)
    {
        // Load this sample's features into the input neurons
        for (int f = 0; f < 30; f++)
            net.setInputs(f, trainInputs[s][f]);

        // Tell the network what the correct answer should be
        net.setDesiredOutput(0, trainLabels[s]);

        // Run forward pass + backpropagation to update weights
        net.learn();

        // Calculate MSE loss for this sample: (actual - predicted)^2
        double output = net.getOuputData(0);
        double err    = trainLabels[s] - output;
        totalLoss    += err * err;

        // Count as correct if prediction matches the actual label
        double predicted = (output >= THRESHOLD) ? 1.0 : 0.0;
        if (predicted == trainLabels[s]) correct++;
    }

    double avgLoss  = totalLoss / trainCount;
    double trainAcc = (double)correct / trainCount * 100.0;

    lossLog.Add((epoch, avgLoss, trainAcc));

    if (epoch % LOG_INTERVAL == 0 || epoch == 1)
    {
        string note = epoch == 1 ? "<-- start" : (epoch == EPOCHS ? "<-- final" : "");
        Console.WriteLine($"{epoch,-10} {avgLoss,-18:F6} {trainAcc,-14:F2}% {note}");
    }
}

Console.WriteLine();

// Step 6 - Save the training log to a CSV file
// This lets us review or plot how the loss changed over time.
string projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
string logPath     = Path.Combine(projectRoot, "training_log.csv");
using (var sw = new StreamWriter(logPath))
{
    sw.WriteLine("Epoch,AvgMSELoss,TrainAccuracy");
    foreach (var (ep, loss, acc) in lossLog)
        sw.WriteLine($"{ep},{loss:F8},{acc:F4}");
}
Console.WriteLine($"[6] Training log saved → {logPath}");

// Step 7 - Evaluate on the test set
// The test set was never seen during training, so this gives us a fair picture
// of how well the model generalizes to new data.
Console.WriteLine("[7] Evaluating on test set...");

int TP = 0, TN = 0, FP = 0, FN = 0;

for (int s = 0; s < testCount; s++)
{
    for (int f = 0; f < 30; f++)
        net.setInputs(f, testInputs[s][f]);

    net.run();

    double output    = net.getOuputData(0);
    double predicted = (output >= THRESHOLD) ? 1.0 : 0.0;
    double actual    = testLabels[s];

    if      (predicted == 1 && actual == 1) TP++; // correctly caught fraud
    else if (predicted == 0 && actual == 0) TN++; // correctly identified as legit
    else if (predicted == 1 && actual == 0) FP++; // falsely flagged as fraud
    else if (predicted == 0 && actual == 1) FN++; // missed a fraud case
}

// Calculate evaluation metrics
// We use Precision, Recall, and F1 because accuracy alone is misleading
// for imbalanced datasets like this one (most transactions are legit).
double accuracy  = (double)(TP + TN) / testCount * 100.0;
double precision = (TP + FP) > 0 ? (double)TP / (TP + FP) * 100.0 : 0.0;
double recall    = (TP + FN) > 0 ? (double)TP / (TP + FN) * 100.0 : 0.0;
double f1        = (precision + recall) > 0
                     ? 2.0 * precision * recall / (precision + recall)
                     : 0.0;

Console.WriteLine();
Console.WriteLine("===========================================");
Console.WriteLine("       EVALUATION RESULTS (Test Set)");
Console.WriteLine("===========================================");
Console.WriteLine($"  Confusion Matrix:");
Console.WriteLine($"           Predicted 0   Predicted 1");
Console.WriteLine($"  Actual 0   {TN,9}     {FP,9}   (Legit)");
Console.WriteLine($"  Actual 1   {FN,9}     {TP,9}   (Fraud)");
Console.WriteLine();
Console.WriteLine($"  Accuracy  : {accuracy:F4}%");
Console.WriteLine($"  Precision : {precision:F4}%   (of flagged fraud, how many were real)");
Console.WriteLine($"  Recall    : {recall:F4}%   (of real fraud, how many were caught)");
Console.WriteLine($"  F1 Score  : {f1:F4}%");
Console.WriteLine("===========================================");
Console.WriteLine();

Console.WriteLine("===========================================");
Console.WriteLine("       HYPERPARAMETER CONFIGURATION");
Console.WriteLine("===========================================");
Console.WriteLine($"  Hidden neurons : {HIDDEN_SIZE}");
Console.WriteLine($"  Epochs         : {EPOCHS}");
Console.WriteLine($"  LR (output)    : 0.2   (LRPOUT in BackNeural.cs)");
Console.WriteLine($"  LR (input)     : 0.15  (LRPIN  in BackNeural.cs)");
Console.WriteLine($"  Train ratio    : {TRAIN_RATIO*100:0}%");
Console.WriteLine($"  Threshold      : {THRESHOLD}");
Console.WriteLine("===========================================");
Console.WriteLine();
Console.WriteLine("Done. Check training_log.csv for the loss curve.");
