#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define nTraining 1000 // Number of testing samples
#define width 28
#define height 28
#define n1 width *height // Number of neurons in the input layer
#define n2 128           // Number of neurons in the hidden layer 1
#define n3 128           // Number of neurons in the hidden layer 2
#define n4 10            // Number of neurons in the output layer (number of classes): 0, 1, 2, ..., 9
#define epochs 100       // Number of epochs
#define learning_rate 1e-3
#define epsilon 1e-3

// Input layer - Hidden layer 1
double *w1[n1], *delta1[n1], *out1;

// Hidden layer 1 - Hidden layer 2
double *w2[n2], *delta2[n2], *out2, *in2, *theta2;

// Hidden layer 2 - Output layer
double *w3[n3], *delta3[n3], *out3, *in3, *theta3;

// Output layer
double *out4, *in4, *theta4;
double expected[n4];

// Image. In MNIST: 28x28 gray scale images.
int data[width][height];

// Testing image file name
char *testing_image_fn = "mnist/t10k-images.idx3-ubyte";

// Testing label file name
char *testing_label_fn = "mnist/t10k-labels.idx1-ubyte";

// Weights file name
char *model_fn = "model-neural-network.dat";

// Report file name
char *report_fn = "testing-report.dat";

FILE *f_testing_image_fn;
FILE *f_testing_label_fn;
FILE *f_model_fn;
// +--------------------+
// | About the software |
// +--------------------+
void about()
{
    printf("Training neural network for MNIST database\n");
    printf("Number of training samples: %d\n", nTraining);
    printf("Image size: %d x %d\n", width, height);
    printf("Number of neurons in the input layer: %d\n", n1);
    printf("Number of neurons in the hidden layer 1: %d\n", n2);
    printf("Number of neurons in the hidden layer 2: %d\n", n3);
    printf("Number of neurons in the output layer: %d\n", n4);
    printf("Number of epochs: %d\n", epochs);
    printf("Learning rate: %f\n", learning_rate);
    printf("Epsilon: %f\n", epsilon);
}

// +-----------------------------------+
// | Memory allocation for the network |
// +-----------------------------------+
void init_array()
{
    // Input layer - Hidden layer 1
    for (int i = 0; i < n1; ++i)
    {
        w1[i] = (double *)malloc(n2 * sizeof(double));
    }
    out1 = (double *)malloc(n1 * sizeof(double));

    // Hidden layer 1 - Hidden layer 2
    for (int i = 0; i < n2; ++i)
    {
        w2[i] = (double *)malloc(n3 * sizeof(double));
    }
    out2 = (double *)malloc(n2 * sizeof(double));
    in2 = (double *)malloc(n2 * sizeof(double));

    // Hidden layer 2 - Output layer
    for (int i = 0; i < n3; ++i)
    {
        w3[i] = (double *)malloc(n4 * sizeof(double));
    }
    out3 = (double *)malloc(n3 * sizeof(double));
    in3 = (double *)malloc(n3 * sizeof(double));

    // Output layer
    out4 = (double *)malloc(n4 * sizeof(double));
    in4 = (double *)malloc(n4 * sizeof(double));
}
// +----------------------------------------+
// | Load model of a trained Neural Network |
// +----------------------------------------+

void load_model()
{
    f_model_fn = fopen(model_fn, "r");

    // Input layer - Hidden layer 1
    for (int i = 0; i < n1; ++i)
    {
        for (int j = 0; j < n2; ++j)
        {
            fscanf(f_model_fn, "%lf", &w1[i][j]);
        }
    }

    // Hidden layer 1 - Hidden layer 2
    for (int i = 0; i < n2; ++i)
    {
        for (int j = 0; j < n3; ++j)
        {
            fscanf(f_model_fn, "%lf", &w2[i][j]);
        }
    }

    // Hidden layer 2 - Output layer
    for (int i = 0; i < n3; ++i)
    {
        for (int j = 0; j < n4; ++j)
        {
            fscanf(f_model_fn, "%lf", &w3[i][j]);
        }
    }

    fclose(f_model_fn);
}

// +---------------+
// | ReLU function |
// +---------------+
double reLU(double x)
{
    return x > 0 ? x : 0;
}

// +------------------+
// | Softmax function |
// +------------------+
double softmax(double x)
{
    double sum = 0.0;
    for (int i = 0; i < n4; ++i)
    {
        sum += exp(in4[i]);
    }

    return exp(x) / sum;
}

void perceptron()
{
    for (int i = 0; i < n2; ++i)
    {
        in2[i] = 0.0;
    }
    for (int i = 0; i < n3; ++i)
    {
        in3[i] = 0.0;
    }
    for (int i = 0; i < n4; ++i)
    {
        in4[i] = 0.0;
    }

    // Forward process
    // Input layer - Hidden layer 1
    for (int i = 0; i < n1; ++i)
    {
        for (int j = 0; j < n2; ++j)
        {
            in2[j] += out1[i] * w1[i][j];
        }
    }

    for (int i = 0; i < n2; ++i)
    {
        out2[i] = reLU(in2[i]);
    }

    // Hidden layer 1 - Hidden layer 2
    for (int i = 0; i < n2; ++i)
    {
        for (int j = 0; j < n3; ++j)
        {
            in3[j] += out2[i] * w2[i][j];
        }
    }

    for (int i = 0; i < n3; ++i)
    {
        out3[i] = reLU(in3[i]);
    }

    // Hidden layer 2 - Output layer
    for (int i = 0; i < n3; ++i)
    {
        for (int j = 0; j < n4; ++j)
        {
            in4[j] += out3[i] * w3[i][j];
        }
    }
    // for (int i = 0; i < n4; ++i)
    // {
    //     printf("%f ", in4[i]);
    // }
    // printf("\n");

    for (int i = 0; i < n4; ++i)
    {
        out4[i] = softmax(in4[i]);
        printf("%f ", out4[i]);
    }
    printf("\n");
}

// +---------------+
// | Norm L2 error |
// +---------------+

double square_error()
{
    double res = 0.0;
    for (int i = 0; i < n4; ++i)
    {
        res += (out4[i] - expected[i]) * (out4[i] - expected[i]);
    }
    res *= 0.5;
    return res;
}

int input()
{
    // Reading image
    char number;
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            fread(&number, sizeof(char), 1, f_testing_image_fn);
            if (number == 0)
            {
                data[i][j] = 0;
            }
            else
            {
                data[i][j] = 1;
            }
        }
    }

    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            int pos = i + (j)*width;
            out1[pos] = data[i][j];
        }
    }

    // Reading label
    fread(&number, sizeof(char), 1, f_testing_label_fn);
    for (int i = 0; i < n4; ++i)
    {
        expected[i] = 0.0;
    }
    expected[number] = 1.0;
    // print label
    // printf("Label: %d\n", number);

    return (int)(number);
}

int main(int argc, char **argv)
{
    about();

    f_testing_image_fn = fopen(testing_image_fn, "rb"); /// Binary image file
    f_testing_label_fn = fopen(testing_label_fn, "rb"); /// Binary image file

    if (f_testing_image_fn == NULL || f_testing_label_fn == NULL)
    {
        printf("Không thể mở tệp.\n");
        return 1; // Thoát chương trình nếu không mở được tệp
    }

    // Read the header of the image file
    char number;
    for (int i = 1; i <= 16; ++i)
    {
        fread(&number, sizeof(char), 1, f_testing_image_fn);
    }
    for (int i = 1; i <= 8; ++i)
    {
        fread(&number, sizeof(char), 1, f_testing_label_fn);
    }

    init_array();
    load_model();

    // print w1
    // for (int i = 0; i < n3; ++i)
    // {
    //     for (int j = 0; j < n4; ++j)
    //     {
    //         printf("%lf ", w3[i][j]);
    //     }
    //     printf("\n");
    // }
    int nCorrect = 0;
    for (int sample = 1; sample <= nTraining; ++sample)
    {
        // print sample
        printf("Sample: %d\n", sample);

        int label = input();
        printf("Label: %d\n", label);
        perceptron();

        //Prediction
        double maxElement = out4[0];
        int predict = 0;
        for (int i = 1; i < n4; ++i)
        {
            if (out4[i] > maxElement)
            {
                maxElement = out4[i];
                predict = i;
            }
        }

        //print predict
        printf("Predict: %d\n", predict);


        double error = square_error();
        printf("Error: %0.6lf\n", error);
        if (label == predict)
        {
            ++nCorrect;
            printf("Classification: YES. Label = %d. Predict = %d\n\n", label, predict);
        }
        else
        {
            printf("Classification: NO. Label = %d. Predict = %d\n", label, predict);
            printf("Image:\n");

            // print image
            // for (int j = 0; j < height; ++j)
            // {
            //     for (int i = 0; i < width; ++i)
            //     {
            //         printf("%d ", data[i][j]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");
        }
    }

    double accuracy = (double)nCorrect / nTraining * 100.0;

    // In thông điệp ra màn hình
    printf("Number of correct samples: %d / %d\n", nCorrect, nTraining);
    printf("Accuracy: %0.2lf\n", accuracy);

    return 0;
}