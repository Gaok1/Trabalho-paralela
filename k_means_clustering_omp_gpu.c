#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REPLICATION_FACTOR 1000
#define NUM_RUNS 30

// Tempo OMP GPU (REPLICATION_FACTOR=1000, NUM_RUNS=30): total ~5.104 s, medio ~0.170 s

typedef struct
{
    double x;
    double y;
    int group;
} observation;

static observation* load_dataset(const char* filename, size_t* out_size)
{
    FILE* f = fopen(filename, "r");
    if (!f)
    {
        return NULL;
    }

    char buffer[512];
    size_t capacity = 1024;
    size_t size = 0;
    observation* observations = (observation*)malloc(sizeof(observation) * capacity);
    if (!observations)
    {
        fclose(f);
        return NULL;
    }

    if (!fgets(buffer, sizeof(buffer), f))
    {
        fclose(f);
        free(observations);
        return NULL;
    }

    while (fgets(buffer, sizeof(buffer), f))
    {
        char* token = strtok(buffer, ",");
        if (!token)
        {
            continue;
        }
        token = strtok(NULL, ",");
        if (!token)
        {
            continue;
        }
        double x = strtod(token, NULL);
        token = strtok(NULL, ",");
        if (!token)
        {
            continue;
        }
        double y = strtod(token, NULL);

        if (size >= capacity)
        {
            capacity *= 2;
            observation* tmp = (observation*)realloc(observations, sizeof(observation) * capacity);
            if (!tmp)
            {
                free(observations);
                fclose(f);
                return NULL;
            }
            observations = tmp;
        }

        observations[size].x = x;
        observations[size].y = y;
        observations[size].group = 0;
        size++;
    }

    fclose(f);

    if (size == 0)
    {
        free(observations);
        return NULL;
    }

    size_t replicated_size = size * REPLICATION_FACTOR;
    observation* replicated = (observation*)malloc(sizeof(observation) * replicated_size);
    if (!replicated)
    {
        free(observations);
        return NULL;
    }

    for (size_t r = 0; r < REPLICATION_FACTOR; r++)
    {
        for (size_t i = 0; i < size; i++)
        {
            size_t idx = r * size + i;
            replicated[idx].x = observations[i].x;
            replicated[idx].y = observations[i].y;
            replicated[idx].group = 0;
        }
    }

    free(observations);
    *out_size = replicated_size;
    return replicated;
}


static void kMeans_omp_gpu(double* x,
                           double* y,
                           int* groups,
                           size_t n,
                           int k,
                           double* cent_x,
                           double* cent_y,
                           int* cent_count)
{
    if (k <= 1)
    {
        double sx = 0.0, sy = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            sx += x[i];
            sy += y[i];
            groups[i] = 0;
        }
        cent_x[0] = sx / (double)n;
        cent_y[0] = sy / (double)n;
        cent_count[0] = (int)n;
        return;
    }

    for (size_t i = 0; i < n; i++)
    {
        groups[i] = rand() % k;
    }

    size_t minAcceptedError = n / 10000;
    long long changed;
    do
    {
        for (int c = 0; c < k; c++)
        {
            cent_x[c] = 0.0;
            cent_y[c] = 0.0;
            cent_count[c] = 0;
        }

        for (size_t i = 0; i < n; i++)
        {
            int g = groups[i];
            cent_x[g] += x[i];
            cent_y[g] += y[i];
            cent_count[g] += 1;
        }

        for (int c = 0; c < k; c++)
        {
            if (cent_count[c] > 0)
            {
                cent_x[c] /= (double)cent_count[c];
                cent_y[c] /= (double)cent_count[c];
            }
        }

        changed = 0;

        // Offload da reatribuicao para GPU
        #pragma omp target teams distribute parallel for \    // offload: reatribui pontos na GPU
        map(to : x[0:n], y[0:n], cent_x[0:k], cent_y[0:k]) \
        map(tofrom : groups[0:n]) reduction(+ : changed)
        for (long long i = 0; i < (long long)n; i++)
        {
            double minD = DBL_MAX;
            int best = 0;
            for (int c = 0; c < k; c++)
            {
                double dx = cent_x[c] - x[i];
                double dy = cent_y[c] - y[i];
                double dist = dx * dx + dy * dy;
                if (dist < minD)
                {
                    minD = dist;
                    best = c;
                }
            }
            if (best != groups[i])
            {
                changed += 1;
                groups[i] = best;
            }
        }

    } while ((size_t)changed > minAcceptedError);
}

int main(void)
{
    const char* filename = "Instagram_visits_clustering.csv";
    int k = 5;

    size_t size = 0;
    observation* obs = load_dataset(filename, &size);
    if (!obs)
    {
        fprintf(stderr, "Erro ao carregar dataset.\n");
        return 1;
    }

    double* x = (double*)malloc(sizeof(double) * size);
    double* y = (double*)malloc(sizeof(double) * size);
    int* groups = (int*)malloc(sizeof(int) * size);
    if (!x || !y || !groups)
    {
        fprintf(stderr, "Erro de memoria.\n");
        free(obs);
        free(x);
        free(y);
        free(groups);
        return 1;
    }

    for (size_t i = 0; i < size; i++)
    {
        x[i] = obs[i].x;
        y[i] = obs[i].y;
        groups[i] = 0;
    }
    free(obs);

    double* cent_x = (double*)malloc(sizeof(double) * k);
    double* cent_y = (double*)malloc(sizeof(double) * k);
    int* cent_count = (int*)malloc(sizeof(int) * k);
    if (!cent_x || !cent_y || !cent_count)
    {
        fprintf(stderr, "Erro de memoria.\n");
        free(x);
        free(y);
        free(groups);
        free(cent_x);
        free(cent_y);
        free(cent_count);
        return 1;
    }

    srand((unsigned int)time(NULL));

    printf("K-Means OpenMP (GPU - target)\n");
    printf("Observacoes efetivas: %zu, clusters: %d\n", size, k);

    double start = omp_get_wtime();
    for (int run = 0; run < NUM_RUNS; run++)
    {
        for (size_t i = 0; i < size; i++)
        {
            groups[i] = 0;
        }
        kMeans_omp_gpu(x, y, groups, size, k, cent_x, cent_y, cent_count);
    }
    double end = omp_get_wtime();

    double elapsed = end - start;
    printf("Tempo total (OpenMP GPU, %d execucoes): %.6f s\n", NUM_RUNS, elapsed);
    printf("Tempo medio por execucao: %.6f s\n", elapsed / NUM_RUNS);

    for (int c = 0; c < k; c++)
    {
        printf("Cluster %d: centroid (%.4f, %.4f), pontos=%d\n", c, cent_x[c], cent_y[c], cent_count[c]);
    }

    free(x);
    free(y);
    free(groups);
    free(cent_x);
    free(cent_y);
    free(cent_count);
    return 0;
}
