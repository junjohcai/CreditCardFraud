// ============================================================
//  CreditCardFraud - Real-World MLP Demo
//  Dataset : creditcard.csv  (Kaggle Credit Card Fraud)
//  Network  : 30 inputs -> hiddenSize hidden -> 1 output
//  Task     : Binary Classification (0 = legit, 1 = fraud)
// ============================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Backprop;

// ── Hyperparameters ─────────────────────────────────────────
// Tweak these to experiment and document your findings.
const int    HIDDEN_SIZE   = 16;     // hidden neuron count
const double TRAIN_RATIO   = 0.80;   // 80% train / 20% test
const int    EPOCHS        = 500;    // total training epochs
const int    LOG_INTERVAL  = 100;    // print loss every N epochs
const double THRESHOLD     = 0.5;    // sigmoid output → binary
// Stratified sample: take this many legit + ALL fraud rows.
// Full dataset (284K rows) = ~2 min/epoch = impractical.
// 5000 legit + all 492 fraud ≈ 5492 rows → runs in seconds/epoch.
const int    LEGIT_SAMPLE  = 5000;   // max legit rows to keep
// ────────────────────────────────────────────────────────────

// Walk up from bin/Debug/net10.0/ to the project root
string csvPath = Path.GetFullPath(
    Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "creditcard.csv"));

Console.WriteLine("===========================================");
Console.WriteLine("  Credit Card Fraud Detection - MLP (C#)");
Console.WriteLine("===========================================");
Console.WriteLine($"Dataset : {csvPath}");
Console.WriteLine($"Network : 30 -> {HIDDEN_SIZE} -> 1");
Console.WriteLine($"Epochs  : {EPOCHS}  |  Train/Test: {TRAIN_RATIO*100:0}%/{(1-TRAIN_RATIO)*100:0}%");
Console.WriteLine();

// ── 1. Load & Parse CSV ─────────────────────────────────────
Console.WriteLine("[1] Loading dataset...");

var allInputs  = new List<double[]>();
var allLabels  = new List<double>();
using (var sr = new StreamReader(csvPath))
{
    string? header = sr.ReadLine(); // skip header
    string? line;
    while ((line = sr.ReadLine()) != null)
    {
        if (string.IsNullOrWhiteSpace(line)) continue;

        string[] parts = line.Split(',');
        // columns: Time, V1..V28, Amount, Class  (31 columns total)
        // We use V1..V28 + Amount = 29 features  (skip Time, index 0)
        // Actually the activity says 30 inputs — include Time too.
        double[] features = new double[30];
        for (int i = 0; i < 30; i++)
            features[i] = double.Parse(parts[i], CultureInfo.InvariantCulture);

        // Class is last column — strip surrounding quotes if present
        string classStr = parts[30].Trim('"');
        double label    = double.Parse(classStr, CultureInfo.InvariantCulture);

        allInputs.Add(features);
        allLabels.Add(label);
    }
}

int totalSamples = allInputs.Count;
Console.WriteLine($"    Loaded {totalSamples:N0} samples.");

// ── 1b. Stratified Sampling ──────────────────────────────────
// Keep ALL fraud rows + up to LEGIT_SAMPLE legit rows.
// This preserves the fraud signal while making training fast.
Console.WriteLine($"[1b] Stratified sampling (all fraud + {LEGIT_SAMPLE:N0} legit)...");

var sampledInputs = new List<double[]>();
var sampledLabels = new List<double>();
int legitCount = 0;

// Shuffle before sampling so legit sample is random
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
    if (allLabels[idx] == 1) // always keep fraud
    {
        sampledInputs.Add(allInputs[idx]);
        sampledLabels.Add(1);
    }
    else if (legitCount < LEGIT_SAMPLE)
    {
        sampledInputs.Add(allInputs[idx]);
        sampledLabels.Add(0);
        legitCount++;
    }
}

int fraudCount = sampledLabels.Count - legitCount;
Console.WriteLine($"    Sample: {sampledInputs.Count:N0} rows  (legit={legitCount:N0}, fraud={fraudCount:N0})");

// Replace full lists with sample
allInputs = sampledInputs;
allLabels = sampledLabels;
totalSamples = allInputs.Count;

// ── 2. Compute Min-Max per feature (on full set) ─────────────
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

// Apply normalisation in-place
foreach (double[] row in allInputs)
{
    for (int f = 0; f < 30; f++)
    {
        double range = featureMax[f] - featureMin[f];
        row[f] = (range == 0.0) ? 0.0 : (row[f] - featureMin[f]) / range;
    }
}

Console.WriteLine("    Normalisation complete.");

// ── 3. Shuffle & Train/Test Split ───────────────────────────
Console.WriteLine("[3] Shuffling and splitting dataset...");

// Shuffle with a fixed seed for reproducibility
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

// ── 4. Build Network ─────────────────────────────────────────
Console.WriteLine("[4] Building network  30 → {0} → 1 ...", HIDDEN_SIZE);
// NeuralNet(input, hidden, output)
var net = new NeuralNet(30, HIDDEN_SIZE, 1);

// ── 5. Training Loop ─────────────────────────────────────────
Console.WriteLine("[5] Training...");
Console.WriteLine();
Console.WriteLine($"{"Epoch",-10} {"Avg MSE Loss",-18} {"Train Acc",-14} {"Notes"}");
Console.WriteLine(new string('-', 60));

// Optional: track loss per epoch for CSV export
var lossLog = new List<(int epoch, double loss, double acc)>();

for (int epoch = 1; epoch <= EPOCHS; epoch++)
{
    double totalLoss = 0.0;
    int    correct   = 0;

    // Shuffle training order each epoch
    for (int i = trainCount - 1; i > 0; i--)
    {
        int j = rng.Next(i + 1);
        (trainInputs[i], trainInputs[j]) = (trainInputs[j], trainInputs[i]);
        (trainLabels[i], trainLabels[j]) = (trainLabels[j], trainLabels[i]);
    }

    for (int s = 0; s < trainCount; s++)
    {
        // Feed inputs
        for (int f = 0; f < 30; f++)
            net.setInputs(f, trainInputs[s][f]);

        // Set desired output
        net.setDesiredOutput(0, trainLabels[s]);

        // Forward + Backprop
        net.learn();

        // MSE loss for this sample
        double output = net.getOuputData(0);
        double err    = trainLabels[s] - output;
        totalLoss    += err * err;

        // Training accuracy
        double predicted = (output >= THRESHOLD) ? 1.0 : 0.0;
        if (predicted == trainLabels[s]) correct++;
    }

    double avgLoss = totalLoss / trainCount;
    double trainAcc = (double)correct / trainCount * 100.0;

    lossLog.Add((epoch, avgLoss, trainAcc));

    if (epoch % LOG_INTERVAL == 0 || epoch == 1)
    {
        string note = epoch == 1 ? "<-- start" : (epoch == EPOCHS ? "<-- final" : "");
        Console.WriteLine($"{epoch,-10} {avgLoss,-18:F6} {trainAcc,-14:F2}% {note}");
    }
}

Console.WriteLine();

// ── 6. Save Training Log ─────────────────────────────────────
// Save log to project root (next to Program.cs), not the build output folder
string projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
string logPath = Path.Combine(projectRoot, "training_log.csv");
using (var sw = new StreamWriter(logPath))
{
    sw.WriteLine("Epoch,AvgMSELoss,TrainAccuracy");
    foreach (var (ep, loss, acc) in lossLog)
        sw.WriteLine($"{ep},{loss:F8},{acc:F4}");
}
Console.WriteLine($"[6] Training log saved → {logPath}");

// ── 7. Evaluation on Test Set ────────────────────────────────
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

    if      (predicted == 1 && actual == 1) TP++;
    else if (predicted == 0 && actual == 0) TN++;
    else if (predicted == 1 && actual == 0) FP++;
    else if (predicted == 0 && actual == 1) FN++;
}

// Metrics
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

// ── 8. Hyperparameter Summary ────────────────────────────────
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
