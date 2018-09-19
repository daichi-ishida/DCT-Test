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
            printf("%lf ", vec[i + nx * j]);
        }
        printf("\n");
    }
    printf("\n");
}

void fftw_dct(double *out, double *in)
{
    fftw_plan plan = fftw_plan_r2r_2d(ny, nx, in, out, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int q = 0; q < ny; ++q)
    {
        for (int p = 0; p < nx; ++p)
        {
            out[p + q * nx] *= 0.25 * std::sqrt(4.0 / (nx * ny));

            if (p == 0)
            {
                out[p + q * nx] *= std::sqrt(1.0 / 2.0);
            }
            if (q == 0)
            {
                out[p + q * nx] *= std::sqrt(1.0 / 2.0);
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
            in[p + q * nx] *= 4.0 * std::sqrt((nx * ny) / 4.0);

            if (p == 0)
            {
                in[p + q * nx] *= std::sqrt(2.0);
            }
            if (q == 0)
            {
                in[p + q * nx] *= std::sqrt(2.0);
            }
        }
    }

    fftw_plan plan = fftw_plan_r2r_2d(ny, nx, in, out, FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_execute(plan);
    // N << 1 means 2*N , N << 2 means 2^2 * N
    for (int i = 0, f = nx * ny << 2; i < nx * ny; ++i)
    {
        out[i] /= (double)f;
    }
}

void my_dct(double *out, double *in)
{
    for (int q = 0; q < ny; ++q)
    {
        double lambda_q = (q == 0) ? std::sqrt(1.0 / (double)ny) : std::sqrt(2.0 / (double)ny);
        for (int p = 0; p < nx; ++p)
        {
            double lambda_p = (p == 0) ? std::sqrt(1.0 / (double)nx) : std::sqrt(2.0 / (double)nx);
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
                double lambda_q = (q == 0) ? std::sqrt(1.0 / (double)ny) : std::sqrt(2.0 / (double)ny);
                for (int p = 0; p < nx; ++p)
                {
                    double lambda_p = (p == 0) ? std::sqrt(1.0 / (double)nx) : std::sqrt(2.0 / (double)nx);
                    sum += lambda_p * lambda_q * in[p + q * nx] * std::cos(M_PI * p * (2 * i + 1) / (2 * nx)) * std::cos(M_PI * q * (2 * j + 1) / (2 * ny));
                }
            }
            out[i + j * nx] = sum;
        }
    }
}

int main()
{
    double a[nx * ny] = {0.5, 0.6, 0.7, 0.8,
                         1.5, 20.6, 8.7, 1.8,
                         -0.5, -0.6, -2.7, -0.8};
    // double a[nx * ny] = {1.0, 0.0, 1.0, 0.0,
    //                      -1.0, 1.0, 0.0, -1.0,
    //                      1.0, 0.0, 1.0, 0.0,
    //                      1.0, 1.0, 0.0, 1.0};
    double b[nx * ny];
    printf("Original vector\n");
    dump_vector(a);

    printf("fftw DCT\n");
    fftw_dct(b, a);
    dump_vector(b);

    // printf("fftw IDCT\n");
    fftw_idct(a, b);
    // dump_vector(a);

    printf("my DCT\n");
    my_dct(b, a);
    dump_vector(b);

    // printf("my IDCT\n");
    my_idct(a, b);
    // dump_vector(a);
    return 0;
}