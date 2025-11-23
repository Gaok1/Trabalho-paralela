#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REPLICATION_FACTOR 1000
#define NUM_RUNS 30

// Tempos seq: total ~9.826 s, medio ~0.328 s (REPLICATION_FACTOR=1000, NUM_RUNS=30)

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

static cluster* kMeans(observation* observations, size_t size, int k)
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

        for (size_t j = 0; j < size; j++)
        {
            int g = observations[j].group;
            clusters[g].x += observations[j].x;
            clusters[g].y += observations[j].y;
            clusters[g].count += 1;
        }

        for (int i = 0; i < k; i++)
        {
            if (clusters[i].count > 0)
            {
                clusters[i].x /= clusters[i].count;
                clusters[i].y /= clusters[i].count;
            }
        }

        changed = 0;
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
    char buffer[512];
    size_t capacity = 4096;
    size_t size = 0;
    observation* observations = (observation*)malloc(sizeof(observation) * capacity);

    fgets(buffer, sizeof(buffer), f); /* descarta cabecalho */

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
            observations = tmp;
        }

        observations[size].x = x;
        observations[size].y = y;
        observations[size].group = 0;
        size++;
    }

    fclose(f);

    size_t replicated_size = size * REPLICATION_FACTOR;
    observation* replicated = (observation*)malloc(sizeof(observation) * replicated_size);
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

    srand((unsigned int)time(NULL));

    printf("K-Means sequencial (base replicada %d vezes, %d execucoes)\n",
           REPLICATION_FACTOR, NUM_RUNS);
    printf("Observacoes efetivas: %zu, clusters: %d\n", size, k);

    clock_t start = clock();
    cluster* clusters = NULL;
    for (int run = 0; run < NUM_RUNS; run++)
    {
        for (size_t i = 0; i < size; i++)
        {
            observations[i].group = 0;
        }
        free(clusters);
        clusters = kMeans(observations, size, k);
    }
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Tempo total (sequencial, %d execucoes): %.6f segundos\n", NUM_RUNS, elapsed);
    printf("Tempo medio por execucao: %.6f segundos\n", elapsed / NUM_RUNS);

    for (int i = 0; i < k; i++)
    {
        printf("Cluster %d: centroid (%.4f, %.4f), pontos=%zu\n", i,
               clusters[i].x, clusters[i].y, clusters[i].count);
    }

    free(observations);
    free(clusters);
    return 0;
}
