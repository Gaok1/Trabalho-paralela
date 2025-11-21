Trabalho de Paralelização - K-Means Clustering
==============================================

Este projeto implementa o algoritmo de agrupamento K-Means utilizando:
- Versão sequencial em C;
- Versão paralela com OpenMP para CPU;
- Versão paralela com OpenMP com offload para GPU;
- Versão paralela em CUDA para GPU.

A base de dados utilizada é real, obtida do Kaggle:
Instagram_visits_clustering.csv
(User ID, Instagram visit score, Spending_rank(0 to 100))


Arquivos principais
-------------------
- `k_means_clustering.c`             : versão sequencial utilizando o CSV.
- `k_means_clustering_omp_cpu.c`    : versão paralela com OpenMP para CPU.
- `k_means_clustering_omp_gpu.c`    : versão paralela com OpenMP target (GPU).
- `k_means_clustering_cuda.cu`      : versão paralela com CUDA (GPU).
- `Instagram_visits_clustering.csv` : base de dados real.
- `instructions.txt`                : especificação do trabalho.


Como compilar
-------------
Supondo GCC com suporte a OpenMP e NVCC para CUDA:

1) Versão sequencial (CPU):
   gcc k_means_clustering.c -O2 -o kmeans_seq -lm

2) Versão OpenMP CPU:
   gcc k_means_clustering_omp_cpu.c -O2 -o kmeans_omp_cpu -fopenmp -lm

3) Versão OpenMP GPU (offload):
   - Necessário compilador com suporte a OpenMP target para GPU (por exemplo, GCC com offload configurado ou Clang/ICC).
   Exemplo genérico (pode variar conforme o ambiente):
   gcc k_means_clustering_omp_gpu.c -O2 -o kmeans_omp_gpu -fopenmp -lm

4) Versão CUDA:
   - Necessário NVCC instalado e GPU compatível com CUDA.
   nvcc k_means_clustering_cuda.cu -o kmeans_cuda


Como executar
-------------
Sempre a partir da pasta onde está o CSV:

1) Versão sequencial:
   ./kmeans_seq

   Saída principal:
   - Número de observações efetivas e clusters;
   - Tempo total e tempo médio por execução (NUM_RUNS);
   - Centróides finais de cada cluster.

2) Versão OpenMP CPU:
   ./kmeans_omp_cpu

   Saída principal:
   - Número de observações efetivas e clusters;
   - Tempos total e médio para 1, 2, 4, 8, 16 e 32 threads.

3) Versão OpenMP GPU:
   ./kmeans_omp_gpu

   Saída principal:
   - Número de observações efetivas e clusters;
   - Tempo total e médio no offload para GPU;
   - Centróides finais.

4) Versão CUDA:
   ./kmeans_cuda

   Saída principal:
   - Número de observações efetivas e clusters;
   - Tempo total e médio na GPU (medido com cudaEventElapsedTime);
   - Centróides finais.


Descrição da aplicação
----------------------
O algoritmo K-Means recebe um conjunto de pontos em 2D (x = Instagram visit score,
y = Spending_rank) e particiona esses pontos em k grupos (clusters) de forma que
os pontos de um mesmo cluster estejam mais próximos do seu centróide.

Fluxo geral:
1. Leitura do arquivo CSV `Instagram_visits_clustering.csv`;
2. Criação do vetor de observações (pontos 2D);
3. Inicialização aleatória dos grupos;
4. Laço principal:
   - Recalcula centróides com base nas atribuições atuais;
   - Reatribui cada ponto ao centróide mais próximo;
   - Repete até que o número de pontos que mudam de cluster seja
     menor que um erro mínimo (size/10000).

As versões paralelas (OpenMP CPU, OpenMP GPU e CUDA) seguem a mesma lógica geral,
mas aceleram principalmente a etapa de reatribuição dos pontos aos centróides.


Configuração dos testes de desempenho
-------------------------------------
Para satisfazer o requisito de o dataset real demandar pelo menos ~10 segundos
na versão sequencial, adotamos a seguinte configuração em todas as versões:

- REPLICATION_FACTOR = 1000  
  (cada linha do CSV é replicada 1000 vezes em memória, resultando em 2.600.000 observações);
- NUM_RUNS = 30  
  (o K-Means é executado 30 vezes para a mesma base, e medimos tempo total e médio).

Assim, a versão sequencial leva aproximadamente 13,5 segundos de tempo total nesta máquina.


Tempos de execução medidos (nesta máquina)
-----------------------------------------
Tempos médios por execução (30 execuções, mesma configuração em todas as versões):

- Versão sequencial (`kmeans_seq`), CPU:
  - Sequencial (1 thread): ~0,451 s  (tempo total ~13,5 s)

- Versão OpenMP CPU (`kmeans_omp_cpu`):
  - 1 thread : ~0,374 s  (speedup ≈ 1,2x)
  - 2 threads: ~0,318 s  (speedup ≈ 1,4x)
  - 4 threads: ~0,213 s  (speedup ≈ 2,1x)
  - 8 threads: ~0,179 s  (speedup ≈ 2,5x)
  - 16 threads: ~0,194 s (speedup ≈ 2,3x)
  - 32 threads: ~0,177 s (speedup ≈ 2,5x)

- Versão OpenMP GPU (`kmeans_omp_gpu`):
  - GPU (OpenMP target): ~0,225 s (speedup ≈ 2,0x vs sequencial)

- Versão CUDA (`kmeans_cuda`, executado via WSL + CUDA 12):
  - GPU (CUDA): ~0,349 s (speedup ≈ 1,3x vs sequencial)


Observações importantes
-----------------------
- A base de dados utilizada é real e foi mantida em `Instagram_visits_clustering.csv`;
- Para reproduzir os tempos, é necessário compilar com otimização (`-O2`) e
  garantir que REPLICATION_FACTOR e NUM_RUNS estejam configurados conforme descrito;
- As modificações de paralelização foram comentadas nos arquivos:
  - `k_means_clustering_omp_cpu.c`   : comentários [PARALELO] em laços com OpenMP;
  - `k_means_clustering_omp_gpu.c`   : comentários [PARALELO-OMP-GPU];
  - `k_means_clustering_cuda.cu`     : comentários [PARALELO-CUDA];
- A versão sequencial é baseada em código aberto do repositório:
  https://github.com/TheAlgorithms/C (arquivo original `k_means_clustering.c`),
  conforme exigido no enunciado.

