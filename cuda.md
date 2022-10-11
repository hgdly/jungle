<!-- 
  cuda.md
  Cuda
  Hugo D.
  Created : 10 octobre 2022
  Updated : 11 octobre 2022
-->

# Cuda <!-- omit in toc -->

- [Motivation](#motivation)
  - [Architecture du CPU](#architecture-du-cpu)
  - [Architecture du GPU](#architecture-du-gpu)
- [Execution d'un programme CUDA](#execution-dun-programme-cuda)
  - [Étapes](#étapes)
- [Librairie](#librairie)
  - [Fonctions](#fonctions)
  - [Configuration du parallélisme](#configuration-du-parallélisme)
    - [Déclaration de fonction](#déclaration-de-fonction)
    - [Appel de fonction](#appel-de-fonction)
    - [Exemple](#exemple)
    - [Limites](#limites)
  - [Parallélisme à 3 dimensions](#parallélisme-à-3-dimensions)
- [Modèle d'exécution](#modèle-dexécution)
  - [Warps](#warps)
    - [Optimisation](#optimisation)
    - [Synchronisation](#synchronisation)
- [Mémoires du GPU](#mémoires-du-gpu)
  - [Registres](#registres)
  - [Mémoire globale](#mémoire-globale)
    - [Mémoire locale](#mémoire-locale)
  - [Mémoire constante](#mémoire-constante)
  - [Mémoire partagée](#mémoire-partagée)
  - [Qualificateurs de variables](#qualificateurs-de-variables)
  - [Exemple : DNA pattern matching](#exemple--dna-pattern-matching)
    - [Version 1](#version-1)
    - [Version 2 - Utilisation de la mémoire constante](#version-2---utilisation-de-la-mémoire-constante)
    - [Version 3 - Utilisation de la mémoire partagée](#version-3---utilisation-de-la-mémoire-partagée)
  - [Tuiles](#tuiles)
- [Algorithmes spécifiques au GPU](#algorithmes-spécifiques-au-gpu)
  - [Opération de Scan](#opération-de-scan)
    - [Exemple de scan](#exemple-de-scan)

## Motivation

Les GPUs (Graphic Processing Units) sont conçus pour du rendu en temps réel (jeux vidéos). Les exigences de traitements sont croissantes.
On utilise donc les ressources du GPU pour les calculs => GPGPU General-Purpose GPU computing.

### Architecture du CPU

Extraction du parallélisme dans un thread (exécution superscalaire, sans ordre)

- 1 Control
- 1 Cache
- N ALU (Arithmetic Logic Unit) => N coeurs
- DRAM

### Architecture du GPU

Exécution SIMT (single instruction-multiple threads)

- N Control
- N Cache
- N*15 (?) ALU
- DRAM

## Execution d'un programme CUDA

Le CPU initialise l'execution des kernels sur le GPU. Le GPU crée un grand nombre de threads qui exécutent le code du kernel.

### Étapes

Code du CPU
Code du GPU (kernel)

1. Allocation mémoire GPU (`cudaMalloc`)
2. Transfert des données CPU vers GPU, host vers device(`cudaMemcpy`)
3. Exécution du kernel
4. Transfert des données GPU vers CPU, device vers host (`cudaMemcpy`)

## Librairie

### Fonctions

- `cudaError_t cudaMalloc (void **devPtr, size_t size)`.
  Alloue la mémoire sur le device.
  - `devPtr`: Pointeur vers la mémoire allouée du device
  - `size`: Taille de la mémoire allouée en octet
- `cudaError_t cudaMemcpy (void *dst, const void *src, size_t count, cudaMemcpyKind kind)`. Copie les données entre host et device.
  - `dst`: Adresse mémoire de la destination
  - `src`: Taille de la mémoire allouée en bytes
  - `count`: Taille en octet à copier
  - `kind`: `cudaMemcpyHostToDevice` ou `cudaMemcpyDevicetoHost`
- `cudaError_t cudeFree (void *devPtr)`.
  Libère la mémoire sur le device.
  - `devPtr`: Pointeur vers la mémoire allouée du device à libérer

### Configuration du parallélisme

#### Déclaration de fonction

`__global__ void f()`: indique que la fonction est un kernel.

- Exécutée par le GPU
- Appellée par le CPU
  
`__device__ float g()`

- Exécutée par le GPU
- Appellée par le GPU

`__host__ int h()` (defaut)

- Exécutée par le CPU
- Appellée par le CPU

#### Appel de fonction

`f<<<x, y>>>(d_out, d_in)`

- `x`: nombre de blocs
- `y`: nombre de threads par bloc
- `x, y`: nombre de threads

`threadIdx.x`: identificateur du thread  
`blockIdx.x`: identificateur du bloc  
`blockDim.x`: dimension du bloc ou nombre de threads dans un bloc  
`gridDim.x`: nombre de blocs

Exemple de calcul d'index:

- N = 262144: Nombre de threads voulu
- y = 256
- x = N / y = 1024
  
bloc 0
|0|1|2|...|255|
|-|-|-|-|-|

bloc 1
|0|1|2|...|255|
|-|-|-|-|-|

...bloc 1023
|0|1|2|...|255|
|-|-|-|-|-|

Index du thread 2 du bloc 1: `blockIdx.x (1) * blockDim.x (256) + threadIdx.x (2) = 258`

#### Exemple

```c
void rgb2grey(uchar4 *hcimg, uchar *hgimg, int isize) 
{
  int bsize = 256;
  int gsize = ((isize + bsize -1) /bsize);
  int numc = isize*sizeof(uchar4);
  int numg = isize*sizeof(unsigned char);
  uchar4 *dcimg;
  unsigned char *dgimg;
  cudaMalloc((void **)&dcimg, numc);
  cudaMemcpy(dcimg, hcimg, numc, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&gdimg, numg);
  color2grey<<< gsize, bsize >>>(dcimg, dgimg, isize);
  cudaMemcpy(hgimg, dgimg, numg, cudaMemcpyDeviceToHost);
  cudaFree(dcimg); cudaFree(dgimg);
}

__global__
void color2grey(uchar4 *dcimg, uchar *dgimg, int isize)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  uchar4 p;
  if (index < isize)
  {
    p = dcimg[index];
    dgimg[index] = (299*p.x + 587*p.y + 114*p.z) / 1000;
  }
}
```

#### Limites

`gridDim.x` (`.y`, `.z`) doit être compris entre 1 et 65536.

Le nombre de threads par bloc est limité à 1024.

### Parallélisme à 3 dimensions

Exemple :

```c
#define BSIZE 16
void matmul(float *A, float *B, float *C, int N)
{
  int bytes = N*N*sizeof(float);
  int numb = N / BSIZE;
  if (N % BSIZE != 0) numb++;
  dim3 gsize(numb, numb, 1);
  dim3 bsize(BSIZE , BSIZE , 1);
  float *dA, *dB,*dC;
  cudaMalloc((void **)&dA, bytes);
  cudaMalloc((void **)&dB, bytes);
  cudaMalloc((void **)&dC,bytes);
  cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);
  matmulkernel<<<gsize,bsize>>>(dA,dB,dC,N);
  cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
__global__
void matmulkernel(float *dA, float *dB, float *dC, int N)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row<N) && (col<N)){
    float res = 0;
    for (int k=0; k<N ; k++)
    res += dA[row*N+k] * dB[k*N+col];
    dC[row*N + col] = res;
  }
}
```

## Modèle d'exécution

SM: Streaming Multiprocessor  
SP: Streaming Processor

Un bloc est affecté à un seul SM.  
Plusieurs blocs sur un même SM.  
Moins de 1536 threads par SM.  

### Warps

Un bloc est divisé en warps de 32 threads.  
32 threads avec des nombres consécutifs (0-31, 32-63, ...).  

Quand une instruction n'est pas prête à être exécutée ou a un long temps de latence, un autre warp est programmé.

#### Optimisation

On préfèrera une taille de bloc qui remplis des warps entiers (multiple de 32).  
Exemple :

- `kernel<<<N, 1>>>(...)`
- `kernel<<<N/32, 32>>>(...)`
- `kernel<<<N/128, 128>>>(...)`

On préfèrera également un grand nombre de threads par bloc pour garantir un grand nombre de warps sur le même SM. C'est ainsi que le GPU peut cacher de longues latences et afficher un débit d'instructions élevé.

#### Synchronisation

`__syncthreads()` agit comme une barrière. Créer un "barrage" et attend tous les threads du même bloc.

## Mémoires du GPU

### Registres

`idx` and `f`: variables privées des threads.

### Mémoire globale

CPU : lecture et écriture  
Threads du GPU : lecture et écriture

#### Mémoire locale

C'est une zone spécifique dans la mémoire globale (longue latence).  
Variable privées (arrays) mais ne peuvent pas être stockées dans les registres.

### Mémoire constante

CPU : lecture et écriture  
Threads du GPU : lecture  
La lecture seule réduit la latence

Tous les threads d'un warp accèdent à la même adresse simultanément => un seul accès. L'accès est plus rapide que pour la mémoire gloable.

Exemple

```c
#define SIZE 100

float h_val[SIZE] = { … };
__constant__ float d_val[SIZE];
cudaMemcpyToSymbol(d_val, h_val,SIZE*sizeof(float));
kernel<<<…,…>>>(…);

__global__ 
kernel(…)
{
  …
  for (int i=0; i<SIZE; i++)
    … = d_val[i] …;
}
```

### Mémoire partagée

Mémoire scratchpad sur puce => courte latence et taille limitée.  
Partagée par tous les threads d'un même bloc. L'allocation peut être statique ou dynamique.

Allocation statique:

```c
#define SIZE 1024

__global__ 
void kernel(…) 
{
  __shared__ float shared_data[SIZE];
  …
}
```

Allocation dynamique:

```c
#define SIZE 1024

int main()
{
  …
  kernel<<<…, …, SIZE*sizeof(float)>>>(…);
}

__global__ 
void kernel(…) 
{
  extern __shared__ float *shared_data;
  …
}
```

Le kernel doit copier les données depuis la mémoire globale sur la mémoire partagée.

```c
#define SIZE 1024

__global__ 
void kernel(float *data) 
{
  __shared__ float shared_data[SIZE];
  int idx = threadIdx.x;
  shared_data[idx] = data[idx];
  __syncthreads();
  … // data can be accessed in the fast shared memory
}
```

### Qualificateurs de variables

|Déclaration|Mémoire|Portée|Durée de vie|
|-:|-|-|-|
|`int var`|registre|thread|kernel|
|`int var[100]`|local|thread|kernel|
|`__device__ __shared__ int var`|shared|bloc|kernel|
|`__device__ int var`|global|grid|application|
|`__device__ __constant__ int var`|constant|grid|application|

Les pointeurs peuvent seulement pointer vers une zone allouées/déclarées dans la mémoire globale.

- Zone allouée par le host, le pointeur est passé en paramètre du kernel : `__global__ void kernel(float *prt){}`
- Pointeur calculé comme l'adresse d'une variable globale : `float *ptr = &globalVar;`

### Exemple : DNA pattern matching

#### Version 1

```c
#define SLEN 16384
#define PLEN 8
#define BSIZE 512

bool matches(unsigned char *pattern, unsigned char *seq)
{
  unsigned int bnum = (SLEN – PLEN + 1 + BSIZE-1)/BSIZE) ;
  unsigned int pbytes = PLEN*sizeof(char);
  unsigned int sbytes = SLEN*sizeof(char);
  bool h_match = FALSE; bool *d_match;
  unsigned char *d_seq, *d_pattern;
  cudaMalloc((void **)&d_seq, sbytes);
  cudaMemcpy(d_seq, seq, sbytes, CudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_pattern, pbytes);
  cudaMemcpy(d_pattern, pattern, pbytes, CudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_match, sizeof(bool));
  cudaMemcpy(d_match, &h_match, sizeof(bool), CudaMemcpyHostToDevice);
  search<<<bnum, BSIZE>>>(d_seq, d_pattern, d_match);
  cudaMemcpy(h_match, d_match, sizeof(bool), CudaMemcpyDeviceToHost);
  cudaFree(d_seq); cudaFree(d_pattern); cudaFree(d_match);
  return(h_match);
} 
```

```c
__global__
void search( unsigned char *d_seq, unsigned char *d_pattern,
bool *d_match)
{
  int gidx = blockIdx.x*blockDim.x+threadIdx.x;
  bool found = TRUE;
  if (idx < SLEN – PLEN + 1)
  {
    for (int i=0; i<PLEN; i++)
    if (d_seq[gidx+i] != d_pattern[i])
    found = FALSE;
    if (found)
    *d_match = TRUE;
  }
} 
```

#### Version 2 - Utilisation de la mémoire constante

```c
#define SLEN 16384
#define PLEN 8
#define BSIZE 512

bool matches(unsigned char *pattern, unsigned char *seq)
{
  unsigned int bnum = (SLEN – PLEN + 1 + BSIZE-1)/BSIZE) ;
  unsigned int pbytes = PLEN*sizeof(char);
  unsigned int sbytes = SLEN*sizeof(char);
  bool h_match = FALSE; bool *d_match;
  unsigned char *d_seq, *d_pattern;

  cudaMalloc((void **)&d_seq, sbytes);
  cudaMemcpy(d_seq, seq, sbytes, CudaMemcpyHostToDevice);
  __constant__ unsigned char c_pattern[PLEN];
  cudaMemcpyToSymbol(c_pattern, pattern, pbytes);
  cudaMalloc((void **)&d_match, sizeof(bool));
  cudaMemcpy(d_match, &h_match, sizeof(bool), CudaMemcpyHostToDevice);

  search<<<bnum, BSIZE>>>(d_seq, d_match);

  cudaMemcpy(h_match, d_match, sizeof(bool), CudaMemcpyDeviceToHost);

  cudaFree(d_seq); cudaFree(d_pattern); cudaFree(d_match);

  return(h_match);
} 
```

```c
__global__
void search( unsigned char *d_seq, unsigned char *d_pattern,
bool *d_match)
{
  int gidx = blockIdx.x*blockDim.x+threadIdx.x;
  bool found = TRUE;
  if (gidx < SLEN – PLEN + 1)
  {
    for (int i=0; i<PLEN; i++)
    if (d_seq[gidx+i] != c_pattern[i])
    found = FALSE;
    if (found)
    *d_match = TRUE;
  }
} 
```

#### Version 3 - Utilisation de la mémoire partagée

```c
__global__
void search(unsigned char *d_seq, bool *d_match)
{
  int gidx = blockIdx.x*blockDim.x+threadIdx.x;
  bool found = TRUE;
  __shared__ unsigned char sh_seq[BSIZE+PLEN–1];
  int lidx = threadIdx.x;
  sh_seq[lidx] = d_seq[gidx];
  if ( (lidx < PLEN– 1) && (gidx + BSIZE < SLEN) )
  sh_seq[lidx+BSIZE] = seq[gidx+BSIZE];
  __synchthreads();
  if (gidx < SLEN – PLEN + 1)
  {
    for (int i=0; i<PLEN; i++)
    if (sh_seq[lidx+i] != c_pattern[i])
    found = FALSE;
    if (found)
    *d_match = TRUE;
  }
}
```

### Tuiles

Il s'agit d'une stratégie pour réduire le trafic vers la mémoire globale.  
Pour limiter les accès à la mémoire globale, il faut découper les données et les placer dans des mémoires partagées.

Exemple:

```c
__global__ 
void MatmulKernel(float *A, float *B, float *C, int N)
{
  __shared__ float AS[BSIZE][BSIZE];
  __shared__ float BS[BSIZE][BSIZE];
  int row = blockIdx.y * BSIZE + threadIdx.y;
  int col= blockIdx.x * BSIZE + threadIdx.x;
  float res = 0;
  for (int ph=0 ; ph<(N/BSIZE); ph++)
  {
    AS[threadIdx.y][threadIdx.x] = A[row * N + ph * BSIZE + threadIdx.x];
    BS[threadIdx.y][threadIdx.x] = B[[(ph* BSIZE + threadIdx.y) * N + col];
    _syncthreads();
    for (int k=0; k<BSIZE; k++)
    {
      res += AS[threadIdx.y][k] * BS[k][threadIdx.x];
    }
    __syncthreads();
  }
  C[row*N + col] = res;
}
```

## Algorithmes spécifiques au GPU

### Opération de Scan

Définition:

- Une liste en entrée
- Un opérateur binaire associatif
- Un élément identité `[I op a = a]`

Inclusif:

```c
int acc = id_element;
for (i=0 ; i<n; i++) {
  acc = acc op in[i];
  out[i] = acc;
}
```

Exclusif:

```c
int acc = id_element;
for (i=0 ; i<n; i++) {
  out[i] = acc;
  acc = acc op in[i];
}
```

#### Exemple de scan

- Entrée : `[1 2 3 4 5 6 7 8]`
- Opérateur: +
- Élément identité: 0

- Sortie scan insclusif: `[1 3 6 10 15 21 28 36]`
- Sortie scan exclusif: `[0 1 3 6 10 15 21 28]`
