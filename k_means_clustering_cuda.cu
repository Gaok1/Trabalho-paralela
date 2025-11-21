/**
 * @file k_means_clustering_cuda.cu
 * @brief Versão paralela do K-Means usando CUDA para GPU.
 *
 * Tempos de execução medidos (mesma configuração da versão sequencial,
* com REPLICATION_FACTOR = 1000 e NUM_RUNS = 30):
*  - CUDA (GPU): ~0.349 s
 *
 * As seções marcadas com [PARALELO-CUDA] indicam mudanças em relação
 * à versão sequencial k_means_clustering.c.
 */

#include <float.h>        /* DBL_MAX */
#include <math.h>         /* funções matemáticas básicas */
#include <cuda_runtime.h> /* CUDA runtime API */
#include <stdio.h>        /* printf, FILE */
#include <stdlib.h>       /* rand, malloc, free */
#include <string.h>       /* strtok */
#include <time.h>         /* time */

/* Mesmo fator de replicação/execuções da versão sequencial */
#define REPLICATION_FACTOR 1000
#define NUM_RUNS 30

typedef struct observation
{
    double x;
    double y;
    int group;
} observation;

/* leitura do dataset (igual à versão sequencial) */
static observation* load_dataset(const char* filename, size_t* out_size)
{
    FILE* f = fopen(filename, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir arquivo de dados: %s\n", filename);
        return NULL;
    }

    char buffer[512];
    size_t capacity = 1024;
    size_t size = 0;
    observation* observations =
        (observation*)malloc(sizeof(observation) * capacity);
    if (!observations)
    {
        fprintf(stderr, "Erro de memória ao alocar observações.\n");
        fclose(f);
        return NULL;
    }

    /* descarta cabeçalho */
    if (!fgets(buffer, sizeof(buffer), f))
    {
        fprintf(stderr, "Arquivo de dados vazio ou inválido.\n");
        fclose(f);
        free(observations);
        return NULL;
    }

    while (fgets(buffer, sizeof(buffer), f))
    {
        char* token = NULL;

        token = strtok(buffer, ",");
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
            observation* tmp =
                (observation*)realloc(observations,
                                      sizeof(observation) * capacity);
            if (!tmp)
            {
                fprintf(stderr,
                        "Erro de memória ao carregar observações.\n");
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
        fprintf(stderr, "Nenhuma observação válida encontrada no CSV.\n");
        free(observations);
        return NULL;
    }

    /* Replica a base em memória para aumentar o número de pontos */
    size_t replicated_size = size * REPLICATION_FACTOR;
    observation* replicated =
        (observation*)malloc(sizeof(observation) * replicated_size);
    if (!replicated)
    {
        fprintf(stderr,
                "Erro de memória ao replicar observações (fator %d).\n",
                REPLICATION_FACTOR);
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

/* [PARALELO-CUDA] Kernel para atribuição de pontos aos clusters */
__global__ void assign_clusters_kernel(const double* x,
                                       const double* y,
                                       int* groups,
                                       const double* cent_x,
                                       const double* cent_y,
                                       int k,
                                       int n,
                                       int* changed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }

    double minD = DBL_MAX;
    int best = -1;

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
        atomicAdd(changed, 1);
        groups[i] = best;
    }
}

/* [PARALELO-CUDA] Implementação de K-Means com passo de atribuição na GPU */
static void kMeans_cuda(double* h_x,
                        double* h_y,
                        int* h_groups,
                        size_t n,
                        int k,
                        double* h_cent_x,
                        double* h_cent_y,
                        int* h_cent_count)
{
    if (k <= 1)
    {
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            sum_x += h_x[i];
            sum_y += h_y[i];
            h_groups[i] = 0;
        }
        h_cent_x[0] = sum_x / (double)n;
        h_cent_y[0] = sum_y / (double)n;
        h_cent_count[0] = (int)n;
        return;
    }

    int N = (int)n;

    double *d_x = NULL, *d_y = NULL;
    int *d_groups = NULL, *d_changed = NULL;
    double *d_cent_x = NULL, *d_cent_y = NULL;

    cudaMalloc((void**)&d_x, sizeof(double) * N);
    cudaMalloc((void**)&d_y, sizeof(double) * N);
    cudaMalloc((void**)&d_groups, sizeof(int) * N);
    cudaMalloc((void**)&d_cent_x, sizeof(double) * k);
    cudaMalloc((void**)&d_cent_y, sizeof(double) * k);
    cudaMalloc((void**)&d_changed, sizeof(int));

    cudaMemcpy(d_x, h_x, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(double) * N, cudaMemcpyHostToDevice);

    /* inicialização aleatória dos grupos na CPU */
    for (size_t i = 0; i < n; i++)
    {
        h_groups[i] = rand() % k;
    }
    cudaMemcpy(d_groups, h_groups, sizeof(int) * N, cudaMemcpyHostToDevice);

    size_t minAcceptedError =
        n / 10000; /* critério de parada semelhante às outras versões */

    int h_changed = 0;

    do
    {
        /* passo 2: cálculo dos centróides na CPU */
        for (int c = 0; c < k; c++)
        {
            h_cent_x[c] = 0.0;
            h_cent_y[c] = 0.0;
            h_cent_count[c] = 0;
        }

        /* grupos estão atualizados na CPU após a primeira iteração */
        for (size_t i = 0; i < n; i++)
        {
            int g = h_groups[i];
            if (g >= 0 && g < k)
            {
                h_cent_x[g] += h_x[i];
                h_cent_y[g] += h_y[i];
                h_cent_count[g] += 1;
            }
        }

        for (int c = 0; c < k; c++)
        {
            if (h_cent_count[c] > 0)
            {
                h_cent_x[c] /= (double)h_cent_count[c];
                h_cent_y[c] /= (double)h_cent_count[c];
            }
        }

        cudaMemcpy(d_cent_x, h_cent_x, sizeof(double) * k,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_cent_y, h_cent_y, sizeof(double) * k,
                   cudaMemcpyHostToDevice);

        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int),
                   cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        assign_clusters_kernel<<<gridSize, blockSize>>>(
            d_x, d_y, d_groups, d_cent_x, d_cent_y, k, N, d_changed);

        cudaMemcpy(&h_changed, d_changed, sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_groups, d_groups, sizeof(int) * N,
                   cudaMemcpyDeviceToHost);

    } while ((size_t)h_changed > minAcceptedError);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_groups);
    cudaFree(d_cent_x);
    cudaFree(d_cent_y);
    cudaFree(d_changed);
}

int main()
{
    const char* filename = "Instagram_visits_clustering.csv";
    int k = 5;

    size_t size = 0;
    observation* obs = load_dataset(filename, &size);
    if (!obs)
    {
        return 1;
    }

    double* x = (double*)malloc(sizeof(double) * size);
    double* y = (double*)malloc(sizeof(double) * size);
    int* groups = (int*)malloc(sizeof(int) * size);
    if (!x || !y || !groups)
    {
        fprintf(stderr, "Erro de memória ao alocar vetores.\n");
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
        fprintf(stderr, "Erro de memória ao alocar centróides.\n");
        free(x);
        free(y);
        free(groups);
        free(cent_x);
        free(cent_y);
        free(cent_count);
        return 1;
    }

    srand((unsigned int)time(NULL));

    printf("K-Means CUDA (GPU)\n");
    printf("Observações efetivas: %zu, clusters: %d\n", size, k);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int run = 0; run < NUM_RUNS; run++)
    {
        for (size_t i = 0; i < size; i++)
        {
            groups[i] = 0;
        }
        kMeans_cuda(x, y, groups, size, k, cent_x, cent_y, cent_count);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    printf("Tempo total (CUDA, %d execuções): %.6f segundos\n", NUM_RUNS,
           elapsed_ms / 1000.0f);
    printf("Tempo médio por execução: %.6f segundos\n",
           (elapsed_ms / 1000.0f) / NUM_RUNS);

    for (int c = 0; c < k; c++)
    {
        printf("Cluster %d: centroid (%.4f, %.4f), pontos=%d\n", c,
               cent_x[c], cent_y[c], cent_count[c]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(x);
    free(y);
    free(groups);
    free(cent_x);
    free(cent_y);
    free(cent_count);

    return 0;
}
