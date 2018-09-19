#include <cstdio>
#define _USE_MATH_DEFINES
#include <cmath>
#include <array>
#include <vector>
#include <fftw3.h>

int nx = 4;
int ny = 3;

void dump_vector(double *vec)
{
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; i++)
        {
            printf("%f ", vec[i + nx * j]);
        }
        printf("\n");
    }
    printf("\n");
}

void fftw_dct(double *out, double *in)
{
    fftw_plan plan = fftw_plan_r2r_2d(nx, ny, in, out, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int q = 0; q < ny; ++q)
    {
        for (int p = 0; p < nx; ++p)
        {
            if (p == 0 && q == 0)
            {
                out[p + q * nx] *= 0.25 * std::sqrt(1.0 / (nx * ny));
            }
            else if (p == 0 || q == 0)
            {
                out[p + q * nx] *= 0.5 * std::sqrt(1.0 / (2.0 * nx * ny));
            }
            else
            {
                out[p + q * nx] *= 0.5 * std::sqrt(1.0 / (nx * ny));
            }
        }
    }
}

void fftw_idct(double *out, double *in)
{
    for (int q = 0; q < ny; ++q)
    {
        for (int p = 0; p < nx; ++p)
        {
            if (p == 0 && q == 0)
            {
                in[p + q * nx] *= 4.0 * std::sqrt(nx * ny);
            }
            else if (p == 0 || q == 0)
            {
                in[p + q * nx] *= 2.0 * std::sqrt(2.0 * nx * ny);
            }
            else
            {
                in[p + q * nx] *= 2.0 * std::sqrt(nx * ny);
            }
        }
    }

    fftw_plan plan = fftw_plan_r2r_2d(nx, ny, in, out, FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_execute(plan);
    // N << 1 means 2*N , N << 2 means 2^2 * N
    for (int i = 0, f = nx * ny << 2; i < nx * ny; ++i)
    {
        out[i] /= f;
    }
}

void my_dct(double *out, double *in)
{
    for (int q = 0; q < ny; ++q)
    {
        double lambda_q = (q == 0) ? std::sqrt(1.0 / ny) : std::sqrt(2.0 / ny);
        for (int p = 0; p < nx; ++p)
        {
            double lambda_p = (p == 0) ? std::sqrt(1.0 / nx) : std::sqrt(2.0 / nx);
            double sum = 0.0;
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    sum += in[i + j * nx] * std::cos(M_PI * p * (2 * i + 1) / (2 * nx)) * std::cos(M_PI * q * (2 * j + 1) / (2 * ny));
                }
            }
            sum *= lambda_p * lambda_q;
            out[p + q * nx] = sum;
        }
    }
}

void my_idct(double *out, double *in)
{
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            double sum = 0.0;
            for (int q = 0; q < ny; ++q)
            {
                double lambda_q = (q == 0) ? std::sqrt(1.0 / ny) : std::sqrt(2.0 / ny);
                for (int p = 0; p < nx; ++p)
                {
                    double lambda_p = (p == 0) ? std::sqrt(1.0 / nx) : std::sqrt(2.0 / nx);
                    sum += lambda_p * lambda_q * in[p + q * nx] * std::cos(M_PI * p * (2 * i + 1) / (2 * nx)) * std::cos(M_PI * q * (2 * j + 1) / (2 * ny));
                }
            }
            out[i + j * nx] = sum;
        }
    }
}

int main()
{
    double a[nx * ny] = {1.5, 0.6, 0.7, 0.8,
                         1.5, 1.6, 1.7, 1.8,
                         -0.5, -0.6, -0.7, -0.8};
    double b[nx * ny];
    printf("Original vector\n");
    dump_vector(a);

    printf("fftw DCT\n");
    fftw_dct(b, a);
    dump_vector(b);

    printf("fftw IDCT\n");
    fftw_idct(a, b);
    dump_vector(a);

    printf("my DCT\n");
    my_dct(b, a);
    dump_vector(b);

    printf("my IDCT\n");
    my_idct(a, b);
    dump_vector(a);
    return 0;
}