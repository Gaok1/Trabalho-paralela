/**
 * @file k_means_clustering.c
 * @brief Versão sequencial do algoritmo K-Means em C.
 *
 * Tempos de execução (medidos nesta máquina, com
 * REPLICATION_FACTOR = 1000 e NUM_RUNS = 30):
 *  - Sequencial (1 thread):    ~X.XXX s (preencher após testes finais)
 *  - OpenMP CPU (1,2,4,8,16,32 threads): ver arquivo k_means_clustering_omp_cpu.c
 *  - OpenMP GPU:               ver arquivo k_means_clustering_omp_gpu.c
 *  - CUDA:                     ver arquivo k_means_clustering_cuda.cu
 *
 * Esta versão é baseada no código aberto disponível em:
 * https://github.com/TheAlgorithms/C (k_means_clustering.c)
 * adaptada para ler a base real Instagram_visits_clustering.csv
 * e ampliada em memória para aumentar o tempo de execução.
 */

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <stdio.h>        /* printf, FILE */
#include <stdlib.h>       /* rand, malloc, free */
#include <string.h>       /* memset, strtok */
#include <time.h>         /* time, clock */

/* Fator de replicação da base em memória para aumentar o número de pontos */
#define REPLICATION_FACTOR 1000
/* Número de execuções completas do K-Means para a mesma base (para aumentar o tempo total).
 * Ajustado para garantir ~10 segundos na versão sequencial nesta máquina.
 */
#define NUM_RUNS 30

/* Tempos medidos nesta m�quina (REPLICATION_FACTOR = 1000, NUM_RUNS = 30)
 * - Sequencial (m�dia): ~0.451 s por execu��o (tempo total ~13.5 s)
 */

/*!
 * @addtogroup machine_learning Machine Learning Algorithms
 * @{
 * @addtogroup k_means K-Means Clustering Algorithm
 * @{
 */

/*! @struct observation
 *  a class to store points in 2d plane
 *  the name observation is used to denote
 *  a random point in plane
 */
typedef struct observation
{
    double x;  /**< abscissa of 2D data point */
    double y;  /**< ordinate of 2D data point */
    int group; /**< the group no in which this observation would go */
} observation;

/*! @struct cluster
 *  this class stores the coordinates
 *  of centroid of all the points
 *  in that cluster it also
 *  stores the count of observations
 *  belonging to this cluster
 */
typedef struct cluster
{
    double x;     /**< abscissa centroid of this cluster */
    double y;     /**< ordinate of centroid of this cluster */
    size_t count; /**< count of observations present in this cluster */
} cluster;

/*!
 * Returns the index of centroid nearest to
 * given observation
 *
 * @param o  observation
 * @param clusters  array of cluster having centroids coordinates
 * @param k  size of clusters array
 *
 * @returns the index of nearest centroid for given observation
 */
int calculateNearst(observation* o, cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    int i = 0;
    for (; i < k; i++)
    {
        /* Calculate Squared Distance*/
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

/*!
 * Calculate centoid and assign it to the cluster variable
 *
 * @param observations  an array of observations whose centroid is calculated
 * @param size  size of the observations array
 * @param centroid  a reference to cluster object to store information of
 * centroid
 */
void calculateCentroid(observation observations[], size_t size,
                       cluster* centroid)
{
    size_t i = 0;
    centroid->x = 0;
    centroid->y = 0;
    centroid->count = size;
    for (; i < size; i++)
    {
        centroid->x += observations[i].x;
        centroid->y += observations[i].y;
        observations[i].group = 0;
    }
    centroid->x /= centroid->count;
    centroid->y /= centroid->count;
}

/*!
 *    --K Means Algorithm--
 * 1. Assign each observation to one of k groups
 *    creating a random initial clustering
 * 2. Find the centroid of observations for each
 *    cluster to form new centroids
 * 3. Find the centroid which is nearest for each
 *    observation among the calculated centroids
 * 4. Assign the observation to its nearest centroid
 *    to create a new clustering.
 * 5. Repeat step 2,3,4 until there is no change
 *    the current clustering and is same as last
 *    clustering.
 *
 * @param observations  an array of observations to cluster
 * @param size  size of observations array
 * @param k  no of clusters to be made
 *
 * @returns pointer to cluster object
 */
cluster* kMeans(observation observations[], size_t size, int k)
{
    cluster* clusters = NULL;
    if (k <= 1)
    {
        /*
        If we have to cluster them only in one group
        then calculate centroid of observations and
        that will be a ingle cluster
        */
        clusters = (cluster*)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        calculateCentroid(observations, size, clusters);
    }
    else if (k < size)
    {
        clusters = malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        /* STEP 1 */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }
        size_t changed = 0;
        size_t minAcceptedError =
            size /
            10000;  // Do until 99.99 percent points are in correct cluster
        int t = 0;
        do
        {
            /* Initialize clusters */
            for (int i = 0; i < k; i++)
            {
                clusters[i].x = 0;
                clusters[i].y = 0;
                clusters[i].count = 0;
            }
            /* STEP 2*/
            for (size_t j = 0; j < size; j++)
            {
                t = observations[j].group;
                clusters[t].x += observations[j].x;
                clusters[t].y += observations[j].y;
                clusters[t].count++;
            }
            for (int i = 0; i < k; i++)
            {
                clusters[i].x /= clusters[i].count;
                clusters[i].y /= clusters[i].count;
            }
            /* STEP 3 and 4 */
            changed = 0;  // this variable stores change in clustering
            for (size_t j = 0; j < size; j++)
            {
                t = calculateNearst(observations + j, clusters, k);
                if (t != observations[j].group)
                {
                    changed++;
                    observations[j].group = t;
                }
            }
        } while (changed > minAcceptedError);  // Keep on grouping until we have
                                               // got almost best clustering
    }
    else
    {
        /* If no of clusters is more than observations
           each observation can be its own cluster
        */
        clusters = (cluster*)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        for (int j = 0; j < size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = j;
        }
    }
    return clusters;
}

/**
 * @}
 * @}
 */

/*!
 * A function to print observations and clusters
 * The code is taken from
 * http://rosettacode.org/wiki/K-means%2B%2B_clustering.
 * Even the K Means code is also inspired from it
 *
 * @note To print in a file use pipeline operator
 * ```sh
 * ./k_means_clustering > image.eps
 * ```
 *
 * @param observations  observations array
 * @param len  size of observation array
 * @param cent  clusters centroid's array
 * @param k  size of cent array
 */
void printEPS(observation pts[], size_t len, cluster cent[], int k)
{
    int W = 400, H = 400;
    double min_x = DBL_MAX, max_x = DBL_MIN, min_y = DBL_MAX, max_y = DBL_MIN;
    double scale = 0, cx = 0, cy = 0;
    double* colors = (double*)malloc(sizeof(double) * (k * 3));
    int i;
    size_t j;
    double kd = k * 1.0;
    for (i = 0; i < k; i++)
    {
        *(colors + 3 * i) = (3 * (i + 1) % k) / kd;
        *(colors + 3 * i + 1) = (7 * i % k) / kd;
        *(colors + 3 * i + 2) = (9 * i % k) / kd;
    }

    for (j = 0; j < len; j++)
    {
        if (max_x < pts[j].x)
        {
            max_x = pts[j].x;
        }
        if (min_x > pts[j].x)
        {
            min_x = pts[j].x;
        }
        if (max_y < pts[j].y)
        {
            max_y = pts[j].y;
        }
        if (min_y > pts[j].y)
        {
            min_y = pts[j].y;
        }
    }
    scale = W / (max_x - min_x);
    if (scale > (H / (max_y - min_y)))
    {
        scale = H / (max_y - min_y);
    };
    cx = (max_x + min_x) / 2;
    cy = (max_y + min_y) / 2;

    printf("%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: -5 -5 %d %d\n", W + 10,
           H + 10);
    printf(
        "/l {rlineto} def /m {rmoveto} def\n"
        "/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n"
        "/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath "
        "	gsave 1 setgray fill grestore gsave 3 setlinewidth"
        " 1 setgray stroke grestore 0 setgray stroke }def\n");
    for (int i = 0; i < k; i++)
    {
        printf("%g %g %g setrgbcolor\n", *(colors + 3 * i),
               *(colors + 3 * i + 1), *(colors + 3 * i + 2));
        for (j = 0; j < len; j++)
        {
            if (pts[j].group != i)
            {
                continue;
            }
            printf("%.3f %.3f c\n", (pts[j].x - cx) * scale + W / 2,
                   (pts[j].y - cy) * scale + H / 2);
        }
        printf("\n0 setgray %g %g s\n", (cent[i].x - cx) * scale + W / 2,
               (cent[i].y - cy) * scale + H / 2);
    }
    printf("\n%%%%EOF");

    // free accquired memory
    free(colors);
}

/*!
 * A function to test the kMeans function
 * Generates 100000 points in a circle of
 * radius 20.0 with center at (0,0)
 * and cluster them into 5 clusters
 *
 * <img alt="Output for 100000 points divided in 5 clusters" src=
 * "https://raw.githubusercontent.com/TheAlgorithms/C/docs/images/machine_learning/k_means_clustering/kMeansTest1.png"
 * width="400px" heiggt="400px">
 * @returns None
 */
static void test()
{
    size_t size = 100000L;
    observation* observations =
        (observation*)malloc(sizeof(observation) * size);
    double maxRadius = 20.00;
    double radius = 0;
    double ang = 0;
    size_t i = 0;
    for (; i < size; i++)
    {
        radius = maxRadius * ((double)rand() / RAND_MAX);
        ang = 2 * M_PI * ((double)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
    }
    int k = 5;  // No of clusters
    cluster* clusters = kMeans(observations, size, k);
    printEPS(observations, size, clusters, k);
    // Free the accquired memory
    free(observations);
    free(clusters);
}

/*!
 * A function to test the kMeans function
 * Generates 1000000 points in a circle of
 * radius 20.0 with center at (0,0)
 * and cluster them into 11 clusters
 *
 * <img alt="Output for 1000000 points divided in 11 clusters" src=
 * "https://raw.githubusercontent.com/TheAlgorithms/C/docs/images/machine_learning/k_means_clustering/kMeansTest2.png"
 * width="400px" heiggt="400px">
 * @returns None
 */
void test2()
{
    size_t size = 1000000L;
    observation* observations =
        (observation*)malloc(sizeof(observation) * size);
    double maxRadius = 20.00;
    double radius = 0;
    double ang = 0;
    size_t i = 0;
    for (; i < size; i++)
    {
        radius = maxRadius * ((double)rand() / RAND_MAX);
        ang = 2 * M_PI * ((double)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
    }
    int k = 11;  // No of clusters
    cluster* clusters = kMeans(observations, size, k);
    printEPS(observations, size, clusters, k);
    // Free the accquired memory
    free(observations);
    free(clusters);
}

/*!
 * This function calls the test
 * function ou executa o K-Means
 * sobre o dataset real.
 */
int main()
{
    /* 
     * Versão sequencial usando a base real Instagram_visits_clustering.csv.
     * Mantemos as funções de teste originais (test e test2) para referência,
     * mas o main abaixo foi adaptado para atender ao requisito do trabalho.
     */

    const char* filename = "Instagram_visits_clustering.csv";
    int k = 5; /* número padrão de clusters, pode ser ajustado via linha de comando futuramente */

    /* leitura do dataset real */
    FILE* f = fopen(filename, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir arquivo de dados: %s\n", filename);
        return 1;
    }

    /* conta linhas para estimar tamanho */
    char buffer[512];
    size_t capacity = 1024;
    size_t size = 0;
    observation* observations =
        (observation*)malloc(sizeof(observation) * capacity);

    /* descarta cabeçalho */
    if (!fgets(buffer, sizeof(buffer), f))
    {
        fprintf(stderr, "Arquivo de dados vazio ou inválido.\n");
        fclose(f);
        free(observations);
        return 1;
    }

    while (fgets(buffer, sizeof(buffer), f))
    {
        char* token = NULL;

        /* coluna 1: User ID (ignorada) */
        token = strtok(buffer, ",");
        if (!token)
        {
            continue;
        }

        /* coluna 2: Instagram visit score -> x */
        token = strtok(NULL, ",");
        if (!token)
        {
            continue;
        }
        double x = strtod(token, NULL);

        /* coluna 3: Spending_rank -> y */
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
                return 1;
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
        return 1;
    }

    /* Replica a base em memória para aumentar o número de pontos e,
     * consequentemente, o tempo de execução, conforme exigido no trabalho.
     */
    size_t replicated_size = size * REPLICATION_FACTOR;
    observation* replicated =
        (observation*)malloc(sizeof(observation) * replicated_size);
    if (!replicated)
    {
        fprintf(stderr,
                "Erro de memória ao replicar observações (fator %d).\n",
                REPLICATION_FACTOR);
        free(observations);
        return 1;
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
    observations = replicated;
    size = replicated_size;

    srand((unsigned int)time(NULL));

    printf("K-Means sequencial (base replicada %d vezes, %d execuções)\n",
           REPLICATION_FACTOR, NUM_RUNS);
    printf("Observações efetivas: %zu, clusters: %d\n", size, k);

    clock_t start = clock();
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
        clusters = kMeans(observations, size, k);
    }
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Tempo total (sequencial, %d execuções): %.6f segundos\n", NUM_RUNS,
           elapsed);
    printf("Tempo médio por execução: %.6f segundos\n", elapsed / NUM_RUNS);

    /* exemplo: impressão de centroids da última execução */
    for (int i = 0; i < k; i++)
    {
        printf("Cluster %d: centroid (%.4f, %.4f), pontos=%zu\n", i,
               clusters[i].x, clusters[i].y, clusters[i].count);
    }

    free(observations);
    free(clusters);

    return 0;
}
