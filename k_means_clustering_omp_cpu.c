#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REPLICATION_FACTOR 1000
#define NUM_RUNS 30

// Tempo OMP CPU (REPLICATION_FACTOR=1000, NUM_RUNS=30): totals 1t~11.48s 2t~6.82s 4t~4.906s 8t~4.643s 16t~4.523s 32t~4.266s ; medios 1t~0.383s 2t~0.227s 4t~0.164s 8t~0.155s 16t~0.151s 32t~0.142s
// Paralelizacao: somas com buffers locais por thread, depois redução; loops paralelos para centróides e reatribuição.

typedef struct
{
    double x;
    double y;
    int group;
} observation;

typedef struct
{
    double x;
    double y;
    size_t count;
} cluster;

static int calculateNearest(const observation* o, const cluster* clusters, int k)
{
    double minD = DBL_MAX;
    int index = 0;
    for (int i = 0; i < k; i++)
    {
        double dx = clusters[i].x - o->x;
        double dy = clusters[i].y - o->y;
        double dist = dx * dx + dy * dy;
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}

static cluster* kMeans_omp(observation* observations, size_t size, int k)
{
    cluster* clusters = NULL;
    if (k <= 1)
    {
        clusters = (cluster*)calloc(1, sizeof(cluster));
        clusters->count = size;
        for (size_t i = 0; i < size; i++)
        {
            clusters->x += observations[i].x;
            clusters->y += observations[i].y;
            observations[i].group = 0;
        }
        clusters->x /= clusters->count;
        clusters->y /= clusters->count;
        return clusters;
    }

    if (k >= (int)size)
    {
        clusters = (cluster*)calloc(k, sizeof(cluster));
        for (int j = 0; j < (int)size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = j;
        }
        return clusters;
    }

    clusters = (cluster*)calloc(k, sizeof(cluster));
    for (size_t j = 0; j < size; j++)
    {
        observations[j].group = rand() % k;
    }

    size_t minAcceptedError = size / 10000;
    size_t changed;
    do
    {
        for (int i = 0; i < k; i++)
        {
            clusters[i].x = 0.0;
            clusters[i].y = 0.0;
            clusters[i].count = 0;
        }

        #pragma omp parallel // acumula somas em buffers locais por thread
        {
            //aloca variaveis locais (cada thread tem uma)
            double* local_x = (double*)calloc((size_t)k, sizeof(double));
            double* local_y = (double*)calloc((size_t)k, sizeof(double));
            size_t* local_count = (size_t*)calloc((size_t)k, sizeof(size_t));

            #pragma omp for // divide as observacoes entre as threads
            for (size_t j = 0; j < size; j++)
            {
                int g = observations[j].group;
                local_x[g] += observations[j].x;
                local_y[g] += observations[j].y;
                local_count[g] += 1;
            }

            #pragma omp critical // reduz buffers locais no acumulador global
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

        #pragma omp parallel for // normaliza centróides em paralelo
        for (int i = 0; i < k; i++)
        {
            if (clusters[i].count > 0)
            {
                clusters[i].x /= clusters[i].count;
                clusters[i].y /= clusters[i].count;
            }
        }

        changed = 0;
        
        #pragma omp parallel for reduction(+ : changed) schedule(static) // reatribui pontos em paralelo
        for (size_t j = 0; j < size; j++)
        {
            int g = calculateNearest(&observations[j], clusters, k);
            if (g != observations[j].group)
            {
                changed++;
                observations[j].group = g;
            }
        }
    } while (changed > minAcceptedError);

    return clusters;
}

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

int main(void)
{
    const char* filename = "Instagram_visits_clustering.csv";
    int k = 5;

    size_t size = 0;
    observation* observations = load_dataset(filename, &size);
    if (!observations)
    {
        fprintf(stderr, "Erro ao carregar dataset.\n");
        return 1;
    }

    int thread_configs[] = {1, 2, 4, 8, 16, 32};
    int num_configs = (int)(sizeof(thread_configs) / sizeof(thread_configs[0]));

    printf("K-Means OpenMP (CPU)\n");
    printf("Observacoes efetivas: %zu, clusters: %d\n", size, k);

    for (int c = 0; c < num_configs; c++)
    {
        int threads = thread_configs[c];
        omp_set_num_threads(threads);

        srand((unsigned int)time(NULL));

        double start = omp_get_wtime();
        cluster* clusters = NULL;
        for (int run = 0; run < NUM_RUNS; run++)
        {
            for (size_t i = 0; i < size; i++)
            {
                observations[i].group = 0;
            }
            free(clusters);
            clusters = kMeans_omp(observations, size, k);
        }
        double end = omp_get_wtime();

        double elapsed = end - start;
        printf("Threads: %2d -> tempo total (%d execucoes): %.6f s, medio: %.6f s\n",
               threads, NUM_RUNS, elapsed, elapsed / NUM_RUNS);

        free(clusters);
    }

    free(observations);
    return 0;
}
