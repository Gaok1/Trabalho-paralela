/**
 * @file k_means_clustering_omp_cpu.c
 * @brief Versão paralela em OpenMP (CPU) do algoritmo K-Means.
 *
 * Tempos de execução medidos (mesma configuração da versão sequencial,
* com REPLICATION_FACTOR = 1000 e NUM_RUNS = 30):
*  - 1 thread : ~0.374 s
*  - 2 threads: ~0.318 s
*  - 4 threads: ~0.213 s
*  - 8 threads: ~0.179 s
*  - 16 threads: ~0.194 s
*  - 32 threads: ~0.177 s
 *
 * As seções marcadas com [PARALELO] indicam mudanças em relação
 * à versão sequencial k_means_clustering.c.
 */

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <omp.h>          /* OpenMP */
#include <stdio.h>        /* printf, FILE */
#include <stdlib.h>       /* rand, malloc, free */
#include <string.h>       /* memset, strtok */
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

typedef struct cluster
{
    double x;
    double y;
    size_t count;
} cluster;

int calculateNearst(observation* o, cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    int i = 0;
    for (; i < k; i++)
    {
        dist = (clusters[i].x - o->x) * (clusters[i].x - o->x) +
               (clusters[i].y - o->y) * (clusters[i].y - o->y);
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}

/* [PARALELO] Versão de K-Means com paralelização em OpenMP (CPU) */
cluster* kMeans_omp(observation observations[], size_t size, int k)
{
    cluster* clusters = NULL;
    if (k <= 1)
    {
        clusters = (cluster*)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        clusters->count = size;
        for (size_t i = 0; i < size; i++)
        {
            clusters->x += observations[i].x;
            clusters->y += observations[i].y;
            observations[i].group = 0;
        }
        clusters->x /= clusters->count;
        clusters->y /= clusters->count;
    }
    else if (k < (int)size)
    {
        clusters = (cluster*)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));

        /* etapa 1: agrupamento inicial aleatório (sequencial) */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }

        size_t changed = 0;
        size_t minAcceptedError =
            size /
            10000; /* faça até que 99,99% dos pontos não mudem de cluster */
        int t = 0;

        do
        {
            /* inicializa clusters */
            for (int i = 0; i < k; i++)
            {
                clusters[i].x = 0;
                clusters[i].y = 0;
                clusters[i].count = 0;
            }

            /* [PARALELO] etapa 2: acumula somas por cluster com buffers locais por thread
             * para reduzir o custo de atomics e melhorar a escalabilidade.
             */
#pragma omp parallel
            {
                double* local_x =
                    (double*)calloc((size_t)k, sizeof(double));
                double* local_y =
                    (double*)calloc((size_t)k, sizeof(double));
                size_t* local_count =
                    (size_t*)calloc((size_t)k, sizeof(size_t));

#pragma omp for private(t) nowait schedule(static)
                for (size_t j = 0; j < size; j++)
                {
                    t = observations[j].group;
                    local_x[t] += observations[j].x;
                    local_y[t] += observations[j].y;
                    local_count[t] += 1;
                }

#pragma omp critical
                {
                    for (int i = 0; i < k; i++)
                    {
                        clusters[i].x += local_x[i];
                        clusters[i].y += local_y[i];
                        clusters[i].count += local_count[i];
                    }
                }

                free(local_x);
                free(local_y);
                free(local_count);
            }

            /* cálculo dos centróides */
#pragma omp parallel for
            for (int i = 0; i < k; i++)
            {
                if (clusters[i].count > 0)
                {
                    clusters[i].x /= clusters[i].count;
                    clusters[i].y /= clusters[i].count;
                }
            }

            /* [PARALELO] etapas 3 e 4: reatribuição dos pontos */
            changed = 0;
#pragma omp parallel for private(t) reduction(+ : changed) schedule(static)
            for (size_t j = 0; j < size; j++)
            {
                t = calculateNearst(observations + j, clusters, k);
                if (t != observations[j].group)
                {
                    changed++;
                    observations[j].group = t;
                }
            }

        } while (changed > minAcceptedError);
    }
    else
    {
        /* se o número de clusters for maior que o de observações */
        clusters = (cluster*)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        for (int j = 0; j < (int)size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = j;
        }
    }
    return clusters;
}

/* reutiliza leitura do CSV semelhante à versão sequencial */
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

int main()
{
    const char* filename = "Instagram_visits_clustering.csv";
    int k = 5;

    size_t size = 0;
    observation* observations = load_dataset(filename, &size);
    if (!observations)
    {
        return 1;
    }

    int thread_configs[] = {1, 2, 4, 8, 16, 32};
    int num_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);

    printf("K-Means OpenMP (CPU)\n");
    printf("Observações efetivas: %zu, clusters: %d\n", size, k);

    for (int c = 0; c < num_configs; c++)
    {
        int threads = thread_configs[c];
        omp_set_num_threads(threads);

        /* re-inicializa grupos */
        srand((unsigned int)time(NULL));

        double start = omp_get_wtime();
        cluster* clusters = NULL;
        for (int run = 0; run < NUM_RUNS; run++)
        {
            for (size_t i = 0; i < size; i++)
            {
                observations[i].group = 0;
            }
            if (clusters)
            {
                free(clusters);
                clusters = NULL;
            }
            clusters = kMeans_omp(observations, size, k);
        }
        double end = omp_get_wtime();

        double elapsed = end - start;
        printf("Threads: %2d -> tempo total (%d execuções): %.6f segundos, "
               "médio: %.6f segundos\n",
               threads, NUM_RUNS, elapsed, elapsed / NUM_RUNS);

        free(clusters);
    }

    free(observations);

    return 0;
}
