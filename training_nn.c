#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define nTraining 60000
#define width 28
#define height 28
#define n1 width *height // Number of neurons in the input layer
#define n2 128           // Number of neurons in the hidden layer 1
#define n3 128           // Number of neurons in the hidden layer 2
#define n4 10            // Number of neurons in the output layer (number of classes): 0, 1, 2, ..., 9
#define epochs 30       // Number of epochs
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

// Training image file name
char *training_image_fn = "mnist/train-images.idx3-ubyte";

// Training label file name
char *training_label_fn = "mnist/train-labels.idx1-ubyte";

// Weights file name
char *model_fn = "model-neural-network.dat";

// Report file name
char *report_fn = "training-report.dat";

FILE *f_training_image_fn;
FILE *f_training_label_fn;
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
        delta1[i] = (double *)malloc(n2 * sizeof(double));
    }
    out1 = (double *)malloc(n1 * sizeof(double));

    // Hidden layer 1 - Hidden layer 2
    for (int i = 0; i < n2; ++i)
    {
        w2[i] = (double *)malloc(n3 * sizeof(double));
        delta2[i] = (double *)malloc(n3 * sizeof(double));
    }
    out2 = (double *)malloc(n2 * sizeof(double));
    in2 = (double *)malloc(n2 * sizeof(double));
    theta2 = (double *)malloc(n2 * sizeof(double));

    // Hidden layer 2 - Output layer
    for (int i = 0; i < n3; ++i)
    {
        w3[i] = (double *)malloc(n4 * sizeof(double));
        delta3[i] = (double *)malloc(n4 * sizeof(double));
    }
    out3 = (double *)malloc(n3 * sizeof(double));
    in3 = (double *)malloc(n3 * sizeof(double));
    theta3 = (double *)malloc(n3 * sizeof(double));

    // Output layer
    out4 = (double *)malloc(n4 * sizeof(double));
    in4 = (double *)malloc(n4 * sizeof(double));
    theta4 = (double *)malloc(n4 * sizeof(double));

    // Initialization for weights from Input layer to Hidden layer 1
    for (int i = 0; i < n1; ++i)
    {
        for (int j = 0; j < n2; ++j)
        {
            

            w1[i][j] = (double)(rand() % 10 + 1) / (10 * n2);
        }
    }
    // Initialization for weights from Hidden layer 1 to Hidden layer 2
    for (int i = 0; i < n2; ++i)
    {
        for (int j = 0; j < n3; ++j)
        {
            
            w2[i][j] = (double)(rand() % 10 + 1) / (10 * n3);
        }
    }
    // Initialization for weights from Hidden layer 2 to Output layer
    for (int i = 0; i < n3; ++i)
    {
        for (int j = 0; j < n4; ++j)
        {
            w3[i][j] = (double)(rand() % 6) / 10.0;
        }
    }
}

// +--------------------------------------------------------------+
// | Reading input - gray scale image and the corresponding label |
// +--------------------------------------------------------------+
void input()
{
    // Reading image
    char number;
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            fread(&number, sizeof(char), 1, f_training_image_fn);

            //double x = number;
            
            //data[i][j] = (double)x / 255.0 ;

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
    // print image
    // for (int j = 0; j < height; ++j)
    // {
    //     for (int i = 0; i < width; ++i)
    //     {
    //         printf("%d ", data[i][j]);
    //     }
    //     printf("\n");
    // }


    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            int pos = i + (j)*width;
            out1[pos] = data[i][j];
        }
    }

    // //print out1 nhap
    // for (int i = 0; i < 28; ++i)
    // {
    //     for (int j = 0; j < 28; ++j)
    //     {
    //         printf("%.0f ", out1[i * 28 + j]);
    //     }
    //     printf("\n");
    // }

    // Reading label
    fread(&number, sizeof(char), 1, f_training_label_fn);
    for (int i = 0; i < n4; ++i)
    {
        expected[i] = 0.0;
    }
    expected[number] = 1.0;
    // print label
    printf("Label: %d\n", number);
    // for (int i = 0; i < n4; ++i)
    // {
    //     printf("%f ", expected[i]);
    // }
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

// +------------------------------+
// | Forward process - Perceptron |
// +------------------------------+
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

// +----------------------------+
// | Back Propagation Algorithm |
// +----------------------------+

void back_propagation()
{
    double sum;

    for (int i = 0; i < n4; ++i)
    {
        theta4[i] = out4[i] * (1 - out4[i]) * (expected[i] - out4[i]);
    }

    for (int i = 0; i < n3; ++i)
    {
        sum = 0.0;
        for (int j = 0; j < n4; ++j)
        {
            sum += w3[i][j] * theta4[j];
        }
        theta3[i] = out3[i] * (1 - out3[i]) * sum;
    }

    for (int i = 0; i < n2; ++i)
    {
        sum = 0.0;
        for (int j = 0; j < n3; ++j)
        {
            sum += w2[i][j] * theta3[j];
        }
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 0; i < n3; ++i)
    {
        for (int j = 0; j < n4; ++j)
        {
            delta3[i][j] = (learning_rate * theta4[j] * out3[i]);
            w3[i][j] += delta3[i][j];
        }
    }
    for (int i = 0; i < n2; ++i)
    {
        for (int j = 0; j < n3; ++j)
        {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]);
            w2[i][j] += delta2[i][j];
        }
    }
    for (int i = 0; i < n1; ++i)
    {
        for (int j = 0; j < n2; ++j)
        {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]);
            w1[i][j] += delta1[i][j];
        }
    }
}

// +-------------------------------------------------+
// | Learning process: Perceptron - Back propagation |
// +-------------------------------------------------+
int learning_process()
{
    for (int i = 0; i < n1; ++i)
    {
        for (int j = 0; j < n2; ++j)
        {
            delta1[i][j] = 0.0;
        }
    }

    for (int i = 0; i < n2; ++i)
    {
        for (int j = 0; j < n3; ++j)
        {
            delta2[i][j] = 0.0;
        }
    }

    for (int i = 0; i < n3; ++i)
    {
        for (int j = 0; j < n4; ++j)
        {
            delta3[i][j] = 0.0;
        }
    }

    for (int i = 1; i <= epochs; ++i)
    {
        perceptron();
        back_propagation();
        if (square_error() < epsilon)
        {
            return i;
        }
    }
    return epochs;

    // return epochs;
    // printf("Learning process\n");
}

// +------------------------+
// | Saving weights to file |
// +------------------------+
void write_matrix()
{
    f_model_fn = fopen(model_fn, "w");
    if (f_model_fn == NULL)
    {
        printf("Cannot write %s\n", f_model_fn);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n1; ++i)
    {
        for (int j = 0; j < n2; ++j)
        {
            fprintf(f_model_fn, "%lf ", w1[i][j]);
        }
        fprintf(f_model_fn, "\n");
    }

    for (int i = 0; i < n2; ++i)
    {
        for (int j = 0; j < n3; ++j)
        {
            fprintf(f_model_fn, "%lf ", w2[i][j]);
        }
        fprintf(f_model_fn, "\n");
    }

    for (int i = 0; i < n3; ++i)
    {
        for (int j = 0; j < n4; ++j)
        {
            fprintf(f_model_fn, "%lf ", w3[i][j]);
        }
        fprintf(f_model_fn, "\n");
    }

    fclose(f_model_fn);
}

int main(int argc, char **argv)
{
    about();

    f_training_image_fn = fopen(training_image_fn, "rb"); /// Binary image file
    f_training_label_fn = fopen(training_label_fn, "rb"); /// Binary image file

    if (f_training_image_fn == NULL || f_training_label_fn == NULL)
    {
        printf("Không thể mở tệp.\n");
        return 1; // Thoát chương trình nếu không mở được tệp
    }

    // Read the header of the image file
    char number;
    for (int i = 1; i <= 16; ++i)
    {
        fread(&number, sizeof(char), 1, f_training_image_fn);
    }
    for (int i = 1; i <= 8; ++i)
    {
        fread(&number, sizeof(char), 1, f_training_label_fn);
    }

    // Neural Network Initialization
    init_array();
    // nTraining
    for (int sample = 1; sample <= 1000; ++sample)
    {
        // print sample
        printf("Sample: %d\n", sample);

        // Getting (image, label)
        input();

        // Learning process : Perceptron - Back propagation
        int nIterations = learning_process();

        // Write report
        // print interations
        printf("Iterations: %d\nError: %0.6lf\n\n", nIterations, square_error());

        if (sample % 100 == 0)
        {
            printf("Saving the network to %s file.\n", model_fn);
            write_matrix(model_fn);
        }
    }

    write_matrix();

    // Đóng tệp sau khi sử dụng
    fclose(f_training_image_fn);
    fclose(f_training_label_fn);

    return 0;
}