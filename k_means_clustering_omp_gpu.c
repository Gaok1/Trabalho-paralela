/**
 * @file k_means_clustering_omp_gpu.c
 * @brief Versão paralela do K-Means usando OpenMP com offload para GPU.
 *
 * Tempos de execução medidos (mesma configuração da versão sequencial,
* com REPLICATION_FACTOR = 1000 e NUM_RUNS = 30):
*  - GPU (OpenMP target): ~0.225 s
 *
 * As seções marcadas com [PARALELO-OMP-GPU] indicam mudanças em relação
 * à versão sequencial k_means_clustering.c.
 */

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX */
#include <math.h>         /* PI */
#include <omp.h>          /* OpenMP e omp_get_wtime */
#include <stdio.h>        /* printf, FILE */
#include <stdlib.h>       /* rand, malloc, free */
#include <string.h>       /* strtok */
#include <time.h>         /* time */

/* Mesmo fator de replicação/execuções da versão sequencial */
#define REPLICATION_FACTOR 1000
#define NUM_RUNS 30

/* Estruturas apenas para leitura do dataset */
typedef struct observation
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

/* [PARALELO-OMP-GPU] K-Means com etapa principal (atribuição) offloaded para GPU */
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
        /* caso trivial: um único cluster, calculado na CPU */
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            sum_x += x[i];
            sum_y += y[i];
            groups[i] = 0;
        }
        cent_x[0] = sum_x / (double)n;
        cent_y[0] = sum_y / (double)n;
        cent_count[0] = (int)n;
        return;
    }

    /* inicialização aleatória dos grupos na CPU */
    for (size_t i = 0; i < n; i++)
    {
        groups[i] = rand() % k;
    }

    size_t minAcceptedError =
        n / 10000; /* critério de parada semelhante ao sequencial */

    long long changed = 0;

    do
    {
        /* passo 2: cálculo dos centróides na CPU */
        for (int c = 0; c < k; c++)
        {
            cent_x[c] = 0.0;
            cent_y[c] = 0.0;
            cent_count[c] = 0;
        }

        for (size_t i = 0; i < n; i++)
        {
            int g = groups[i];
            if (g >= 0 && g < k)
            {
                cent_x[g] += x[i];
                cent_y[g] += y[i];
                cent_count[g] += 1;
            }
        }

        for (int c = 0; c < k; c++)
        {
            if (cent_count[c] > 0)
            {
                cent_x[c] /= (double)cent_count[c];
                cent_y[c] /= (double)cent_count[c];
            }
        }

        /* [PARALELO-OMP-GPU] passos 3 e 4: atribuição de pontos à GPU */
        changed = 0;
#pragma omp target teams distribute parallel for \
    map(to : x[0:n], y[0:n], cent_x[0:k], cent_y[0:k]) \
        map(tofrom : groups[0:n]) reduction(+ : changed)
        for (long long i = 0; i < (long long)n; i++)
        {
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
                changed += 1;
                groups[i] = best;
            }
        }

    } while ((size_t)changed > minAcceptedError);
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

    printf("K-Means OpenMP (GPU - target)\n");
    printf("Observações efetivas: %zu, clusters: %d\n", size, k);

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
    printf("Tempo total (OpenMP GPU, %d execuções): %.6f segundos\n", NUM_RUNS,
           elapsed);
    printf("Tempo médio por execução: %.6f segundos\n", elapsed / NUM_RUNS);

    for (int c = 0; c < k; c++)
    {
        printf("Cluster %d: centroid (%.4f, %.4f), pontos=%d\n", c,
               cent_x[c], cent_y[c], cent_count[c]);
    }

    free(x);
    free(y);
    free(groups);
    free(cent_x);
    free(cent_y);
    free(cent_count);

    return 0;
}
