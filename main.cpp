
/*********************************************************************
 *             -- Multilayer perceptron from scratch --              *
 *                                                                   *
 *                  © Kento Mackenzie Regalado. 2026                 *
 *                                                                   *
 *       Red Spartan | Batangas State University Malvar Campus       *
 *********************************************************************/

 /***********************************************************************************************
  * Disclaimer: This project implements a Multilayer Perceptron (MLP) from scratch without      *
  * using external ML libraries. The goal is to understand the underlying mathematics and       *
  * logic of neural networks. Therefore, the code is not optimized for speed, memory efficiency,* 
  * or large-scale training.                                                                    *
  ***********************************************************************************************/

#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <string>
#include <chrono>
#include <iomanip>
#include <optional>
#include <functional>
#include <thread>
#include <cstdint>

using Matrix = std::vector<std::vector<float>>;
using uchar = unsigned char;
using uint = unsigned int;

// initialize rng engine
static std::mt19937 engine(std::random_device{}());

/* ----------------------------------------------------------
 *                configure MNIST datasets                  *
   ---------------------------------------------------------*/

// Endian helpers (convert the big-endian order into the little-endian order)
static uint swap32(uint x)
{
    return (x >> 24) |
           ((x << 8) & 0x00FF0000) |
           ((x >> 8) & 0x0000FF00) |
           (x << 24);
}

// Reading the file
std::vector<uchar> read_bytes(const std::string &path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open " + path);
    std::streamsize sz = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uchar> buf(sz);
    if (!file.read(reinterpret_cast<char *>(buf.data()), sz))
        throw std::runtime_error("Read error " + path);
    return buf;
}

// MNIST container
struct MNIST
{
    uint N;                                 // number of samples
    uint rows, cols;                        // 28, 28
    std::vector<std::vector<uchar>> images; // outer: N , inner: 784
    std::vector<uchar> labels;              // size == N
};

// MNIST loader
MNIST load_datasets(const std::string &imgPath,
                    const std::string &lblPath)
{
    auto imgRaw = read_bytes(imgPath);
    auto lblRaw = read_bytes(lblPath);

    if (imgRaw.size() < 16 || lblRaw.size() < 8)
        throw std::runtime_error("file too small");

    const uint *pImg = reinterpret_cast<const uint *>(imgRaw.data());
    const uint *pLbl = reinterpret_cast<const uint *>(lblRaw.data());

    MNIST m;
    uint magicImg = swap32(pImg[0]);
    uint magicLbl = swap32(pLbl[0]);
    if (magicImg != 0x00000803 || magicLbl != 0x00000801)
        throw std::runtime_error("wrong magic number");

    m.N = swap32(pImg[1]);
    m.rows = swap32(pImg[2]);
    m.cols = swap32(pImg[3]);
    uint lblN = swap32(pLbl[1]);
    if (m.N != lblN)
        throw std::runtime_error("image/label count mismatch");

    const uchar *pix = imgRaw.data() + 16; // start of pixel data
    m.images.resize(m.N);
    for (uint i = 0; i < m.N; ++i)
    {
        m.images[i].resize(m.rows * m.cols);
        std::memcpy(m.images[i].data(),
                    pix + i * m.rows * m.cols,
                    m.rows * m.cols);
    }
    const uchar *lab = lblRaw.data() + 8;
    m.labels.assign(lab, lab + m.N);
    return m;
}

const char *label_to_char(int label)
{
    const char *char_name = "";
    switch (label)
    {
    case 0:
        char_name = "o";
        break;
    case 1:
        char_name = "ki";
        break;
    case 2:
        char_name = "su";
        break;
    case 3:
        char_name = "tsu";
        break;
    case 4:
        char_name = "na";
        break;
    case 5:
        char_name = "ha";
        break;
    case 6:
        char_name = "ma";
        break;
    case 7:
        char_name = "ya";
        break;
    case 8:
        char_name = "re";
        break;
    case 9:
        char_name = "wo";
        break;
    default:
        break;
    }
    return char_name;
}

// display image label onto the console
void display_digit_label(const MNIST &m, uint idx)
{
    if (idx >= m.N)
        return;

    const auto &img = m.images[idx];

    for (uint r = 0; r < m.rows; r += 2)
    {
        for (uint c = 0; c < m.cols; c++)
        {
            uchar b = img[r * m.cols + c];
            int pix = static_cast<int>(b);
            float val = static_cast<int>(pix) / 255.0f;
            uint32_t pixel = 232 + (uint32_t)(val * 23);

            printf("\x1b[48;5;%dm ", pixel);
        }
        printf("\x1b[0m\n");
    }

    int label = static_cast<int>(m.labels[idx]);
    std::cout << "Label = " << label
              << ", Character name = " << label_to_char(label) << "\n";
}

/* -----------------------------------------------------------------
 * Initializing parameters and miscellaneous functions for the model *
   ----------------------------------------------------------------- */
const size_t input_size = 784;
const size_t hidden1_size = 256;
const size_t hidden2_size = 128;
const size_t output_size = 10;
const size_t num_classes = 10;
const size_t batch_size = 32;
const size_t epochs = 50;
const size_t lr_step = 5;
const int early_stopping_patience = 5; // Stop after 5 epochs of no improvement

const float base_lr = 0.001f;
const float lr_decay = 0.15f;
const float weight_decay = 1e-4f;
const float dropout_rate = 0.2f;
const float beta_1 = 0.9f;
const float beta_2 = 0.999f;
const float epsilon = 1e-8f;

const size_t n_sample_validation_data = batch_size * 312; // 10,016
const size_t n_sample_train_data = 60000 - n_sample_validation_data;
const size_t n_sample_test_data = 10000;

Matrix W1(hidden1_size, std::vector<float>(input_size));
Matrix B1(hidden1_size, std::vector<float>(1, 0.0f));

Matrix W2(hidden2_size, std::vector<float>(hidden1_size));
Matrix B2(hidden2_size, std::vector<float>(1, 0.0f));

Matrix W3(output_size, std::vector<float>(hidden2_size));
Matrix B3(output_size, std::vector<float>(1, 0.0f));

Matrix transpose(Matrix &matrix)
{
    Matrix Tmatrix(matrix[0].size(), std::vector<float>(matrix.size()));

    for (size_t i = 0; i < matrix.size(); i++)
    {
        for (size_t j = 0; j < matrix[0].size(); j++)
        {
            Tmatrix[j][i] = matrix[i][j];
        }
    }
    return Tmatrix;
}

void dot_product(const Matrix &A, const Matrix &B, Matrix &C)
{
    if (A.empty() || B.empty())
        throw std::invalid_argument("Empty matrix");

    const size_t a_rows = A.size();
    const size_t a_cols = A[0].size();
    const size_t b_rows = B.size();
    const size_t b_cols = B[0].size();

    for (auto &row : C)
        std::fill(row.begin(), row.end(), 0.0f);

    if (a_cols != b_rows)
        throw std::invalid_argument("Incompatible dimensions for multiplication");

    for (size_t i = 0; i < a_rows; ++i)
        for (size_t k = 0; k < a_cols; ++k) // iterate over the "inner" dimension
        {
            for (size_t j = 0; j < b_cols; ++j)
                C[i][j] += A[i][k] * B[k][j]; // accumulate
        }
}

void add_bias(Matrix &A, const Matrix &bias)
{
    // bias = (hidden_layer_size, 1) = (128, 1)
    // hidden_layer1 = (128, 32)

    size_t n_neurons = A.size();
    size_t batch_size = A[0].size();

    for (size_t col = 0; col < batch_size; ++col)
    {
        for (size_t row = 0; row < n_neurons; ++row)
        {
            A[row][col] += bias[row][0];
        }
    }
}

void apply_relu(const Matrix &z, Matrix &a)
{
    const size_t neurons = z.size();
    const size_t batch = z[0].size();

    for (size_t i = 0; i < neurons; ++i)
    {
        for (size_t j = 0; j < batch; ++j)
        {
            a[i][j] = std::fmaxf(0.0f, z[i][j]);
        }
    }
}

void apply_relu_derivative(const Matrix &z, Matrix &dz)
{
    const size_t neurons = z.size();
    const size_t batch = z[0].size();

    for (size_t i = 0; i < neurons; ++i)
    {
        for (size_t j = 0; j < batch; ++j)
        {
            dz[i][j] = (z[i][j] <= 0.0f) ? 0.0f : 1.0f;
        }
    }
}

void apply_softmax(const Matrix &z, Matrix &a)
{

    const size_t neurons = z.size();
    const size_t batch = z[0].size();

    for (size_t b = 0; b < batch; ++b) // iterate over samples
    {
        // find max in this sample (for numerical stability)
        float m = z[0][b];
        for (size_t n = 1; n < neurons; ++n)
            m = std::max(m, z[n][b]);

        // compute exp(x - max) and running sum
        float sum = 0.0f;
        for (size_t n = 0; n < neurons; ++n)
        {
            a[n][b] = std::expf(z[n][b] - m);
            sum += a[n][b];
        }

        // normalize
        for (size_t n = 0; n < neurons; ++n)
            a[n][b] /= sum;
    }
}

float compute_loss(const Matrix &predicted, const Matrix &y_true)
{
    /*
     * yk - is the true probability of class k, typically represented as 1 for the correct class and 0 for all other classes.
     * pk - is the predicted probability of class k
     */

    size_t classes = predicted.size();
    size_t batch_size = predicted[0].size();

    float batch_loss = 0.0f;
    float eps = 1e-15f;

    for (size_t col = 0; col < batch_size; ++col)
    {
        float sample_loss = 0.0f;
        for (size_t row = 0; row < classes; ++row)
        {
            float yk = y_true[row][col];

            if (yk > 0.0f)
            {
                float pk = std::clamp(predicted[row][col], 1e-15f, 1.0f - 1e-15f);
                sample_loss -= yk * std::logf(pk);
            }
        }
        batch_loss += sample_loss;
    }
    return batch_loss / static_cast<float>(batch_size);
}

void apply_one_hot_encoding(Matrix &batch_y, int batch_i, int label)
{
    batch_y[label][batch_i] = 1.0f;
}

std::vector<int> get_argmax(Matrix &a3)
{
    std::vector<int> prediction(a3[0].size());
    // a3 = 10x32
    for (size_t i = 0; i < a3[0].size(); ++i)
    {
        int max_idx = 0;
        for (size_t j = 0; j < a3.size(); ++j)
        {
            if (a3[j][i] > a3[max_idx][i])
            {
                max_idx = j;
            }
        }
        prediction[i] = max_idx;
    }
    return prediction;
}

void deltaB(Matrix &dn, Matrix &db, size_t m)
{
    for (size_t i = 0; i < dn.size(); ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < dn[0].size(); ++j)
        {
            sum += dn[i][j];
        }
        db[i][0] = sum / static_cast<float>(m);
    }
}

void forward_pass(Matrix &batch_x,
                  Matrix &z1, Matrix &a1,
                  Matrix &z2, Matrix &a2,
                  Matrix &z3, Matrix &a3,
                  std::optional<std::reference_wrapper<Matrix>> noise1,
                  std::optional<std::reference_wrapper<Matrix>> noise2,
                  bool training)
{

    // hidden layer 1
    dot_product(W1, batch_x, z1);
    add_bias(z1, B1);
    apply_relu(z1, a1);

    // apply dropout to a1
    if (training)
    {
        for (size_t i = 0; i < hidden1_size; ++i)
        {
            for (size_t j = 0; j < batch_size; ++j)
            {
                float noise = noise1->get()[i][j];
                a1[i][j] *= noise;
            }
        }
    }

    // hidden layer 2
    dot_product(W2, a1, z2);
    add_bias(z2, B2);
    apply_relu(z2, a2);

    // apply dropout to a2
    if (training)
    {
        for (size_t i = 0; i < hidden2_size; ++i)
        {
            for (size_t j = 0; j < batch_size; ++j)
            {
                float noise = noise2->get()[i][j];
                a2[i][j] *= noise;
            }
        }
    }

    // output layer
    dot_product(W3, a2, z3);
    add_bias(z3, B3);
    apply_softmax(z3, a3);
}

void backward_pass(Matrix &batch_x, Matrix &batch_y_true,
                   Matrix &z1, Matrix &a1,
                   Matrix &z2, Matrix &a2,
                   Matrix &z3, Matrix &a3,
                   Matrix &dW1, Matrix &dB1,
                   Matrix &dW2, Matrix &dB2,
                   Matrix &dW3, Matrix &dB3,
                   Matrix &noise1, Matrix &noise2)
{
    //--------------------------------------------------
    Matrix delta1(hidden1_size, std::vector<float>(batch_size)); // 128x32
    Matrix delta2(hidden2_size, std::vector<float>(batch_size)); // 64x32
    Matrix delta3(output_size, std::vector<float>(batch_size));  // 10x32

    Matrix dz1(hidden1_size, std::vector<float>(batch_size)); // 128x32
    Matrix dz2(hidden2_size, std::vector<float>(batch_size)); // 64x32

    //--------------------------------------------------------
    /* Layer 3 (Output) gradient */
    //--------------------------------------------------------
    Matrix a2T = transpose(a2);

    for (size_t col = 0; col < a3[0].size(); ++col) // 10x32
    {
        for (size_t row = 0; row < a3.size(); ++row)
        {
            delta3[row][col] = a3[row][col] - batch_y_true[row][col];
        }
    }

    dot_product(delta3, a2T, dW3); // (output_size, hidden2_size)
    for (size_t i = 0; i < dW3.size(); ++i)
    {
        for (size_t j = 0; j < dW3[0].size(); ++j)
        {
            dW3[i][j] /= static_cast<float>(batch_size);
        }
    }

    deltaB(delta3, dB3, batch_size);

    // // ---------------------------------------------------
    // /** Layer 2 (Hidden 2) Gradients */
    // // ---------------------------------------------------
    // δ2 = (W3^T · δ3) ⊙ ReLU'(z2)
    Matrix a1T = transpose(a1);
    Matrix W3T = transpose(W3);
    apply_relu_derivative(z2, dz2);

    dot_product(W3T, delta3, delta2); // 64x32
    for (size_t i = 0; i < delta2.size(); ++i)
    {
        for (size_t j = 0; j < delta2[0].size(); ++j)
        {
            delta2[i][j] *= dz2[i][j] * noise2[i][j];
        }
    }

    dot_product(delta2, a1T, dW2);
    for (size_t i = 0; i < dW2.size(); ++i)
    {
        for (size_t j = 0; j < dW2[0].size(); ++j)
        {
            dW2[i][j] /= static_cast<float>(batch_size);
        }
    }

    deltaB(delta2, dB2, batch_size);

    // ---------------------------------------------------
    /** Layer 1 (Hidden 1) Gradients */
    // ---------------------------------------------------
    // δ1 = (W2^T · δ2) ⊙ ReLU'(z1)
    Matrix batch_xT = transpose(batch_x);
    Matrix W2T = transpose(W2);
    apply_relu_derivative(z1, dz1);

    dot_product(W2T, delta2, delta1);
    for (size_t i = 0; i < delta1.size(); ++i)
    {
        for (size_t j = 0; j < delta1[0].size(); ++j)
        {
            delta1[i][j] *= dz1[i][j] * noise1[i][j];
        }
    }

    dot_product(delta1, batch_xT, dW1);
    for (size_t i = 0; i < dW1.size(); ++i)
    {
        for (size_t j = 0; j < dW1[0].size(); ++j)
        {
            dW1[i][j] /= static_cast<float>(batch_size);
        }
    }

    deltaB(delta1, dB1, batch_size);
}

void update_parameters(Matrix &dW1, Matrix &dB1, Matrix &dW2, Matrix &dB2,
                       Matrix &dW3, Matrix &dB3,
                       Matrix &W1_mt, Matrix &B1_mt, Matrix &W1_vt, Matrix &B1_vt,
                       Matrix &W2_mt, Matrix &B2_mt, Matrix &W2_vt, Matrix &B2_vt,
                       Matrix &W3_mt, Matrix &B3_mt, Matrix &W3_vt, Matrix &B3_vt, size_t adam_steps, float lr)
{

    float t = static_cast<float>(adam_steps);
    float b1_t = (1.0f - std::powf(beta_1, t));
    float b2_t = (1.0f - std::powf(beta_2, t));

    // Update weight 1 and b1 with adam optmizer
    for (size_t i = 0; i < hidden1_size; ++i)
    {
        for (size_t j = 0; j < input_size; ++j)
        {
            W1_mt[i][j] = beta_1 * W1_mt[i][j] + (1.0f - beta_1) * dW1[i][j];
            W1_vt[i][j] = beta_2 * W1_vt[i][j] + (1.0f - beta_2) * dW1[i][j] * dW1[i][j];
            float w1_mt_hat = W1_mt[i][j] / b1_t;
            float w1_vt_hat = W1_vt[i][j] / b2_t;
            W1[i][j] -= lr * (w1_mt_hat / (std::sqrt(w1_vt_hat) + epsilon) + weight_decay * W1[i][j]);
        }
    }

    for (size_t i = 0; i < hidden1_size; ++i)
    {
        B1_mt[i][0] = beta_1 * B1_mt[i][0] + (1.0f - beta_1) * dB1[i][0];
        B1_vt[i][0] = beta_2 * B1_vt[i][0] + (1.0f - beta_2) * dB1[i][0] * dB1[i][0];
        float b1_mt_hat = B1_mt[i][0] / b1_t;
        float b1_vt_hat = B1_vt[i][0] / b2_t;
        B1[i][0] -= lr * b1_mt_hat / (std::sqrt(b1_vt_hat) + epsilon);
    }

    // Update weight 2 and b2
    for (size_t i = 0; i < hidden2_size; ++i)
    {
        for (size_t j = 0; j < hidden1_size; ++j)
        {
            W2_mt[i][j] = beta_1 * W2_mt[i][j] + (1.0f - beta_1) * dW2[i][j];
            W2_vt[i][j] = beta_2 * W2_vt[i][j] + (1.0f - beta_2) * dW2[i][j] * dW2[i][j];
            float w2_mt_hat = W2_mt[i][j] / b1_t;
            float w2_vt_hat = W2_vt[i][j] / b2_t;
            W2[i][j] -= lr * (w2_mt_hat / (std::sqrt(w2_vt_hat) + epsilon) + weight_decay * W2[i][j]);
        }
    }

    for (size_t i = 0; i < hidden2_size; ++i)
    {
        B2_mt[i][0] = beta_1 * B2_mt[i][0] + (1.0f - beta_1) * dB2[i][0];
        B2_vt[i][0] = beta_2 * B2_vt[i][0] + (1.0f - beta_2) * dB2[i][0] * dB2[i][0];
        float b2_mt_hat = B2_mt[i][0] / b1_t;
        float b2_vt_hat = B2_vt[i][0] / b2_t;
        B2[i][0] -= lr * b2_mt_hat / (std::sqrt(b2_vt_hat) + epsilon);
    }

    // Update weight 3 and b3
    for (size_t i = 0; i < output_size; ++i)
    {
        for (size_t j = 0; j < hidden2_size; ++j)
        {
            W3_mt[i][j] = beta_1 * W3_mt[i][j] + (1.0f - beta_1) * dW3[i][j];
            W3_vt[i][j] = beta_2 * W3_vt[i][j] + (1.0f - beta_2) * dW3[i][j] * dW3[i][j];
            float w3_mt_hat = W3_mt[i][j] / b1_t;
            float w3_vt_hat = W3_vt[i][j] / b2_t;
            W3[i][j] -= lr * (w3_mt_hat / (std::sqrt(w3_vt_hat) + epsilon) + weight_decay * W3[i][j]);
        }
    }

    for (size_t i = 0; i < output_size; ++i)
    {
        B3_mt[i][0] = beta_1 * B3_mt[i][0] + (1.0f - beta_1) * dB3[i][0];
        B3_vt[i][0] = beta_2 * B3_vt[i][0] + (1.0f - beta_2) * dB3[i][0] * dB3[i][0];
        float b3_mt_hat = B3_mt[i][0] / b1_t;
        float b3_vt_hat = B3_vt[i][0] / b2_t;
        B3[i][0] -= lr * b3_mt_hat / (std::sqrt(b3_vt_hat) + epsilon);
    }
}

float validate(const Matrix &validation_images,
               const std::vector<int> &validation_labels)
{
    size_t n_val_batches = n_sample_validation_data / batch_size;
    float total_val_loss = 0.0f;
    float total_val_correct = 0.0f;
    size_t total_samples = 0;

    Matrix val_z1(hidden1_size, std::vector<float>(batch_size));
    Matrix val_a1(hidden1_size, std::vector<float>(batch_size));
    Matrix val_z2(hidden2_size, std::vector<float>(batch_size));
    Matrix val_a2(hidden2_size, std::vector<float>(batch_size));
    Matrix val_z3(output_size, std::vector<float>(batch_size));
    Matrix val_a3(output_size, std::vector<float>(batch_size));

    Matrix val_batch_y(output_size, std::vector<float>(batch_size));
    Matrix val_batch_x(batch_size, std::vector<float>(input_size));

    for (size_t batch_idx = 0; batch_idx < n_val_batches; ++batch_idx)
    {
        // Reset matrices
        for (auto &row : val_batch_y)
            std::fill(row.begin(), row.end(), 0.0f);
        for (auto &row : val_z1)
            std::fill(row.begin(), row.end(), 0.0f);
        for (auto &row : val_a1)
            std::fill(row.begin(), row.end(), 0.0f);
        for (auto &row : val_z2)
            std::fill(row.begin(), row.end(), 0.0f);
        for (auto &row : val_a2)
            std::fill(row.begin(), row.end(), 0.0f);
        for (auto &row : val_z3)
            std::fill(row.begin(), row.end(), 0.0f);
        for (auto &row : val_a3)
            std::fill(row.begin(), row.end(), 0.0f);

        size_t start = batch_idx * batch_size;

        // Prepare batch
        for (size_t i = 0; i < batch_size; ++i)
        {
            size_t idx = start + i;
            val_batch_x[i] = validation_images[idx];
            apply_one_hot_encoding(val_batch_y, i, validation_labels[idx]);
        }

        Matrix Tval_batch_x = transpose(val_batch_x);

        // Forward pass (NO dropout)
        forward_pass(Tval_batch_x, val_z1, val_a1, val_z2, val_a2, val_z3, val_a3,
                     std::nullopt, std::nullopt, false);

        // Loss
        total_val_loss += compute_loss(val_a3, val_batch_y);

        // Accuracy
        std::vector<int> pred = get_argmax(val_a3);
        std::vector<int> truth = get_argmax(val_batch_y);
        for (size_t i = 0; i < batch_size; ++i)
            if (pred[i] == truth[i])
                total_val_correct++;

        total_samples += batch_size;
    }

    float avg_loss = total_val_loss / n_val_batches;
    // float val_acc = total_val_correct / total_samples * 100.0f;

    // std::cout << " | Val Acc: " << val_acc << "%";

    return avg_loss;
}

void initialize_weights()
{
    auto he = [](int fan_in)
    {
        float std = std::sqrt(2.0f / fan_in);
        return std::normal_distribution<float>(0.0f, std);
    };

    // generate random values for weight 1
    auto dist1 = he(input_size);
    for (size_t i = 0; i < hidden1_size; ++i)
    {
        for (size_t j = 0; j < input_size; ++j)
        {
            W1[i][j] = dist1(engine);
        }
    }

    // generate random values for weight 2
    auto dist2 = he(hidden1_size);
    for (size_t i = 0; i < hidden2_size; ++i)
    {
        for (size_t j = 0; j < hidden1_size; ++j)
        {
            W2[i][j] = dist2(engine);
        }
    }

    // generate random values for weight 3
    auto dist3 = he(hidden2_size);
    for (size_t i = 0; i < output_size; ++i)
    {
        for (size_t j = 0; j < hidden2_size; ++j)
        {
            W3[i][j] = dist3(engine);
        }
    }
}

void initialize_noise(Matrix &noise1, Matrix &noise2)
{
    float alpha = dropout_rate / (1.0f - dropout_rate);
    std::normal_distribution<float> noise_dist(1.0f, std::sqrt(alpha));

    for (size_t i = 0; i < hidden1_size; ++i)
    {
        for (size_t j = 0; j < batch_size; ++j)
        {
            noise1[i][j] = noise_dist(engine);
        }
    }

    for (size_t i = 0; i < hidden2_size; ++i)
    {
        for (size_t j = 0; j < batch_size; ++j)
        {
            noise2[i][j] = noise_dist(engine);
        }
    }
}

void initialize_data(MNIST &train_data, MNIST &test_data,
                     Matrix &train_images, std::vector<int> &train_labels,
                     Matrix &validation_images, std::vector<int> &validation_labels,
                     Matrix &test_images, std::vector<int> &test_labels,
                     size_t n_sample_train_data,
                     size_t n_sample_validation_data,
                     size_t n_sample_test_data)
{
    train_data = load_datasets("datasets/kmnist/train-images-idx3-ubyte",
                               "datasets/kmnist/train-labels-idx1-ubyte");

    test_data = load_datasets("datasets/kmnist/t10k-images-idx3-ubyte",
                              "datasets/kmnist/t10k-labels-idx1-ubyte");

    // training set data
    for (size_t i = 0; i < n_sample_train_data; ++i)
    {
        // assign training labels
        train_labels[i] = static_cast<int>(train_data.labels[i]);

        for (size_t j = 0; j < train_data.rows * train_data.cols; ++j)
        {
            train_images[i][j] = (float)static_cast<int>(train_data.images[i][j]) / 255;
        }
    }

    // validation set data
    for (size_t i = 0; i < n_sample_validation_data; ++i)
    {
        size_t off_set = n_sample_train_data + i;
        validation_labels[i] = static_cast<int>(train_data.labels[off_set]);
        for (size_t j = 0; j < train_data.rows * train_data.cols; ++j)
        {
            uint pix = static_cast<uint>(train_data.images[off_set][j]);
            validation_images[i][j] = static_cast<float>(pix) / 255.0f; // normalize
        }
    }

    // testing set data
    for (size_t i = 0; i < n_sample_test_data; ++i)
    {
        // assign training labels
        test_labels[i] = static_cast<int>(test_data.labels[i]);

        for (size_t j = 0; j < test_data.rows * test_data.cols; ++j)
        {
            test_images[i][j] = (float)static_cast<int>(test_data.images[i][j]) / 255;
        }
    }
}

int predict_single(std::vector<float> &target_img, Matrix &z1, Matrix &a1,
                   Matrix &z2, Matrix &a2, Matrix &z3, Matrix &a3, float &confidence_level)
{
    for (auto &row : z1)
        std::fill(row.begin(), row.end(), 0.0f);
    for (auto &row : z2)
        std::fill(row.begin(), row.end(), 0.0f);
    for (auto &row : z3)
        std::fill(row.begin(), row.end(), 0.0f);

    Matrix test_img(input_size, std::vector<float>(1));
    for (size_t i = 0; i < input_size; ++i)
    {
        test_img[i][0] = target_img[i];
    }

    forward_pass(test_img, z1, a1, z2, a2, z3, a3, std::nullopt, std::nullopt, false);
    std::vector<int> prediction = get_argmax(a3);

    float max = 0.0f;
    for (size_t i = 0; i < a3.size(); i++)
    {
        if (a3[i][0] > max)
        {
            max = a3[i][0];
        }
    }

    confidence_level = max * 100.0f;
    return prediction[0];
}

int main(void)
{

    // std::cout << L"Japan written in Japanese: 日本 (Nihon) \x2665 \x2591" << std::endl;
    std::cout << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10);
    // ------------------------------------------------------------

    // Weight and bias 1 parameters
    Matrix W1_mt(hidden1_size, std::vector<float>(input_size, 0.0f));
    Matrix B1_mt(hidden1_size, std::vector<float>(1, 0.0f));
    Matrix W1_vt(hidden1_size, std::vector<float>(input_size, 0.0f));
    Matrix B1_vt(hidden1_size, std::vector<float>(1, 0.0f));

    // Weight and bias 2 parameters
    Matrix W2_mt(hidden2_size, std::vector<float>(hidden1_size, 0.0f));
    Matrix B2_mt(hidden2_size, std::vector<float>(1, 0.0f));
    Matrix W2_vt(hidden2_size, std::vector<float>(hidden1_size, 0.0f));
    Matrix B2_vt(hidden2_size, std::vector<float>(1, 0.0f));

    // Weight and bias 3 parameters
    Matrix W3_mt(output_size, std::vector<float>(hidden2_size, 0.0f));
    Matrix B3_mt(output_size, std::vector<float>(1, 0.0f));
    Matrix W3_vt(output_size, std::vector<float>(hidden2_size, 0.0f));
    Matrix B3_vt(output_size, std::vector<float>(1, 0.0f));

    // generate random values for weight 1
    initialize_weights();

    // saving the best weight and bias parameters
    Matrix best_W1 = W1, best_B1 = B1;
    Matrix best_W2 = W2, best_B2 = B2;
    Matrix best_W3 = W3, best_B3 = B3;

    // ------------------------------------------------------------------------
    try
    {
        MNIST train_data, test_data;

        Matrix train_images(n_sample_train_data, std::vector<float>(input_size));
        Matrix validation_images(n_sample_validation_data, std::vector<float>(input_size));
        Matrix test_images(n_sample_test_data, std::vector<float>(input_size));

        std::vector<int> train_labels(n_sample_train_data);
        std::vector<int> validation_labels(n_sample_validation_data);
        std::vector<int> test_labels(n_sample_test_data);

        initialize_data(train_data, test_data,
                        train_images, train_labels,
                        validation_images, validation_labels,
                        test_images, test_labels,
                        n_sample_train_data,
                        n_sample_validation_data,
                        n_sample_test_data);

        // -------------------------------------------------------------------
        size_t n_train_batches = static_cast<size_t>(n_sample_train_data / batch_size);
        //---------------------------------------------------------------

        std::vector<int> random_train_data_indices(n_sample_train_data);
        std::iota(random_train_data_indices.begin(), random_train_data_indices.end(), 0);

        // define training data for SGD mini-batch
        Matrix batch_x(batch_size, std::vector<float>(input_size));  // 32x784
        Matrix batch_y(output_size, std::vector<float>(batch_size)); // 10x32

        // ---------------------------------------------------------
        Matrix z1(hidden1_size, std::vector<float>(batch_size));
        Matrix a1(hidden1_size, std::vector<float>(batch_size));

        Matrix z2(hidden2_size, std::vector<float>(batch_size));
        Matrix a2(hidden2_size, std::vector<float>(batch_size));

        Matrix z3(output_size, std::vector<float>(batch_size));
        Matrix a3(output_size, std::vector<float>(batch_size));

        // ---------------------------------------------------------
        Matrix dW1(hidden1_size, std::vector<float>(input_size));
        Matrix dB1(hidden1_size, std::vector<float>(1, 0.f));

        Matrix dW2(hidden2_size, std::vector<float>(hidden1_size));
        Matrix dB2(hidden2_size, std::vector<float>(1, 0.f));

        Matrix dW3(output_size, std::vector<float>(hidden2_size));
        Matrix dB3(output_size, std::vector<float>(1, 0.f));

        Matrix mask1(hidden1_size, std::vector<float>(batch_size));
        Matrix mask2(hidden2_size, std::vector<float>(batch_size));

        Matrix noise1(hidden1_size, std::vector<float>(batch_size));
        Matrix noise2(hidden2_size, std::vector<float>(batch_size));

        std::cout << "\n"
                  << " ***************************************************************\n"
                  << " *                                                             *\n"
                  << " *      Classify handwritten characters in ancient             *\n"
                  << " *      Japanese manuscripts (KMNIST) datasets                 *\n"
                  << " * ----------------------------------------------------------- *\n"
                  << " * Model Overview: Multi-layer perceptron with 2-hidden layers *\n"
                  << " *                                                             *\n"
                  << " *            -> Hidden layer 1: 128-neurons                   *\n"
                  << " *            -> Hidden layer 2: 64-neurons                    *\n"
                  << " *            -> uses SGD mini-batch (with size = 32)          *\n"
                  << " *            -> uses AdamW optimizer                          *\n"
                  << " *            -> uses Gaussian Dropout                         *\n"
                  << " *                                                             *\n"
                  << " ***************************************************************\n";

        std::cout << "\nTraining the model ......." << '\n';

        size_t adam_steps = 1;

        float best_val_loss = std::numeric_limits<float>::max();
        int patience_counter = 0;
        size_t best_epoch = 0;

        for (size_t epoch = 0; epoch < epochs; ++epoch)
        {
            std::shuffle(random_train_data_indices.begin(), random_train_data_indices.end(), engine);
            float total_correct = 0.0f;
            float total_loss = 0.0f;
            int total_samples = 0;

            for (size_t batch_idx = 0; batch_idx < n_train_batches; ++batch_idx)
            {
                // reset data
                for (auto &row : dW1)
                    std::fill(row.begin(), row.end(), 0.0f);
                for (auto &row : dB1)
                    std::fill(row.begin(), row.end(), 0.0f);
                for (auto &row : dW2)
                    std::fill(row.begin(), row.end(), 0.0f);
                for (auto &row : dB2)
                    std::fill(row.begin(), row.end(), 0.0f);
                for (auto &row : dW3)
                    std::fill(row.begin(), row.end(), 0.0f);
                for (auto &row : dB3)
                    std::fill(row.begin(), row.end(), 0.0f);

                for (auto &row : batch_y)
                    std::fill(row.begin(), row.end(), 0.0f);

                for (auto &row : z1)
                    std::fill(row.begin(), row.end(), 0.0f);
                for (auto &row : z2)
                    std::fill(row.begin(), row.end(), 0.0f);
                for (auto &row : z3)
                    std::fill(row.begin(), row.end(), 0.0f);

                size_t start = batch_idx * batch_size;
                size_t end = start + batch_size;
                std::vector<int> batch_indices(random_train_data_indices.begin() + start, random_train_data_indices.begin() + end); // size = 32

                for (size_t i = 0; i < batch_size; ++i)
                {
                    batch_x[i] = train_images[batch_indices[i]];
                    apply_one_hot_encoding(batch_y, i, train_labels[batch_indices[i]]);
                }

                // transpose batch_x
                Matrix Tbatch_x = transpose(batch_x);

                // initialize noise for dropout
                initialize_noise(noise1, noise2);

                // forward pass
                forward_pass(Tbatch_x, z1, a1, z2, a2, z3, a3, noise1, noise2, true);

                // get accuracies
                std::vector<int> prediction = get_argmax(a3);
                std::vector<int> true_labels = get_argmax(batch_y);
                float batch_correct = 0.0f;
                for (size_t max_arg = 0; max_arg < batch_size; ++max_arg)
                {
                    if (prediction[max_arg] == true_labels[max_arg])
                    {
                        batch_correct += 1.0f;
                    }
                }

                total_correct += batch_correct;
                total_samples += (float)batch_size;

                // Track loss
                float loss = compute_loss(a3, batch_y);
                total_loss += loss;

                // // backward pass
                backward_pass(Tbatch_x, batch_y, z1, a1, z2, a2, z3, a3, dW1, dB1, dW2, dB2, dW3, dB3, noise1, noise2);

                // learning rate decay
                float lr = base_lr * std::powf(lr_decay, epoch / lr_step);
                // // update parameters
                update_parameters(dW1, dB1, dW2, dB2, dW3, dB3,
                                  W1_mt, B1_mt, W1_vt, B1_vt,
                                  W2_mt, B2_mt, W2_vt, B2_vt,
                                  W3_mt, B3_mt, W3_vt, B3_vt, adam_steps, lr);

                ++adam_steps;
            }

            // validate
            float validation_loss = validate(validation_images, validation_labels);

            // NEW: Early stopping logic
            if (validation_loss < best_val_loss)
            {
                best_val_loss = validation_loss;
                patience_counter = 0;
                best_epoch = epoch;

                // Copy the best weights and biases
                best_W1 = W1;
                best_B1 = B1;
                best_W2 = W2;
                best_B2 = B2;
                best_W3 = W3;
                best_B3 = B3;
            }
            else
            {
                patience_counter++;
                if (patience_counter >= early_stopping_patience)
                {
                    std::cout << "\nEarly stopping triggered! Best epoch: " << best_epoch
                              << " with val loss: " << best_val_loss << '\n';
                    break; // Stop training
                }
            }
            std::cout << '\n';

            // Print progress
            float avg_loss = total_loss / n_train_batches;
            float accuracy = total_correct / total_samples * 100.0f;

            std::cout << " | Epoch : " << epoch
                      << " | (avg) Loss : " << avg_loss
                      << " | Accuracy : " << accuracy << "%"
                      << " | Val loss : " << validation_loss << '\n';
        }

        W1 = best_W1;
        B1 = best_B1;
        W2 = best_W2;
        B2 = best_B2;
        W3 = best_W3;
        B3 = best_B3;

        // testing data
        for (size_t i = 0; i < test_images.size(); ++i)
        {
            float confidence = 0.0f;
            int prediction = predict_single(test_images[i], z1, a1, z2, a2, z3, a3, confidence);

            display_digit_label(test_data, (uint)i);

            std::cout << "***************************************************************\n";
            std::cout << "*           True Label     : " << test_labels[i] << '\n';
            std::cout << "*           Prediction     : " << prediction << '\n';
            std::cout << "*           Confidence     : " << confidence << "%\n";
            std::cout << "***************************************************************\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        // // testing data
        // while (true)
        // {
        //     std::cout << "\nEnter test image index (0 - 9999), or -1 to quit: ";
        //     int image_index;
        //     std::cin >> image_index;

        //     if (image_index == -1)
        //         break;

        //     if (image_index < 0 || image_index > 9999)
        //     {
        //         std::cout << "Invalid image index." << '\n';
        //         continue;
        //     }

        //     float confidence = 0.0f;
        //     int prediction = predict_single(test_images[image_index], z1, a1, z2, a2, z3, a3, confidence);

        //     display_digit_label(test_data, (uint)image_index);

        //     std::cout << "***************************************************************\n";
        //     std::cout << "*           True Label     : " << test_labels[image_index] << '\n';
        //     std::cout << "*           Prediction     : " << prediction << '\n';
        //     std::cout << "*           Confidence     : " << confidence << "%\n";
        //     std::cout << "***************************************************************\n";
        // }
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
    }

    return 0;
}
