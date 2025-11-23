# Trabalho de Paralelização – K-Means Clustering

Este projeto implementa o algoritmo de agrupamento **K-Means** utilizando:
- Versão sequencial em C;
- Versão paralela com **OpenMP para CPU**;
- Versão paralela com **OpenMP com offload para GPU**;
- Versão paralela em **CUDA para GPU**.

A base de dados utilizada é real, obtida do Kaggle:
- Arquivo: `Instagram_visits_clustering.csv`  
- Colunas: `User ID`, `Instagram visit score`, `Spending_rank(0 to 100)`

---

## Arquivos principais

- `k_means_clustering.c`  
  Versão sequencial utilizando o CSV (C + `math.h`).

- `k_means_clustering_omp_cpu.c`  
  Versão paralela com **OpenMP para CPU**, com testes para 1, 2, 4, 8, 16 e 32 threads.

- `k_means_clustering_omp_gpu.c`  
  Versão paralela usando **OpenMP target** (offload para GPU).

- `k_means_clustering_cuda.cu`  
  Versão paralela em **CUDA**, compilada com `nvcc`.

- `Instagram_visits_clustering.csv`  
  Base de dados real utilizada em todas as versões.

- Pasta `executaveis/`  
  Binários gerados (seq., OpenMP CPU, OpenMP GPU, CUDA).

- Pasta `saidas/`  
  Logs de execução, com tempos medidos de cada versão.

---

## Como compilar

Assumindo GCC com suporte a OpenMP e NVCC instalado para CUDA:

```bash
# Versão sequencial (CPU)
gcc k_means_clustering.c -O2 -o kmeans_seq -lm

# Versão OpenMP CPU
gcc k_means_clustering_omp_cpu.c -O2 -o kmeans_omp_cpu -fopenmp -lm

# Versão OpenMP GPU (offload) – flags de target podem variar conforme ambiente
gcc k_means_clustering_omp_gpu.c -O2 -o kmeans_omp_gpu -fopenmp -lm

# Versão CUDA (em ambiente com nvcc)
nvcc k_means_clustering_cuda.cu -o kmeans_cuda
```
---

## Execução

Sempre executar a partir da pasta onde está o CSV:

Cada programa imprime:
- número de observações efetivas e de clusters;
- tempos de execução (total e médio, considerando múltiplas execuções);
- centróides finais de cada cluster (última execução).

Os logs já gerados estão em `saidas/`:
- `kmeans_seq.txt`
- `kmeans_omp_cpu.txt`
- `kmeans_omp_gpu.txt`
- `kmeans_cuda_wsl.txt`

---

## Configuração dos testes de desempenho

Para atender ao requisito do enunciado – **pelo menos ~10 s na versão sequencial** –
usamos a mesma configuração em todas as versões:

- `REPLICATION_FACTOR = 1000`  
  Cada linha do CSV é replicada 1000 vezes em memória ⇒ **2.600.000 observações**.

- `NUM_RUNS = 30`  
  O K-Means é executado 30 vezes sobre essa base; medimos o tempo total e
  calculamos o tempo médio por execução.

Assim, a versão sequencial leva ~13,5 s de tempo total nesta máquina.

---

## Resultados experimentais (médias por execução)

Configuração: `REPLICATION_FACTOR = 1000`, `NUM_RUNS = 30`, compilação com `-O2`.

### Tempos médios

| Versão                 | Configuração             | Tempo médio (s) | Speedup vs seq. |
|------------------------|--------------------------|-----------------|-----------------|
| Sequencial             | 1 thread                 | **0,451**       | 1,0×            |
| OpenMP CPU             | 1 thread                 | 0,374           | ~1,2×           |
| OpenMP CPU             | 2 threads                | 0,318           | ~1,4×           |
| OpenMP CPU             | 4 threads                | 0,213           | ~2,1×           |
| OpenMP CPU             | 8 threads                | 0,179           | ~2,5×           |
| OpenMP CPU             | 16 threads               | 0,194           | ~2,3×           |
| OpenMP CPU             | 32 threads               | 0,177           | ~2,5×           |
| OpenMP GPU (target)    | GPU                      | 0,225           | ~2,0×           |
| CUDA                   | GPU (via WSL + CUDA 12)  | 0,349           | ~1,3×           |

Os valores exatos podem ser conferidos nos arquivos:
- `saidas/kmeans_seq.txt`
- `saidas/kmeans_omp_cpu.txt`
- `saidas/kmeans_omp_gpu.txt`
- `saidas/kmeans_cuda_wsl.txt`

