// the subroutine for GPU code can be found in several separated text file from the Brightspace. 
// You can add these subroutines to this main code.
////////////////////////////////////////////


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cuda.h"


//const int BLOCK_SIZE = 32;  // number of threads per block

// Input Array Variables
float* h_MatA = NULL;
float* d_MatA = NULL;

// Output Array
float* h_VecV = NULL;
float* d_VecV = NULL;
float* h_VecW = NULL;
float* d_VecW = NULL;
float* h_NormW = NULL;
float* d_NormW = NULL;



// Variables to change
int GlobalSize = 10000;         // this is the dimension of the matrix, GlobalSize*GlobalSize
int BlockSize = 32;            // number of threads in each block
const float EPS = 0.0000001;    // tolerence of the error
int max_iteration = 200;       // the maximum iteration steps

// Functions
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
float CPUReduce(float*, int);
void  Arguments(int, char**);
void checkCardVersion(void);

// Kernels
__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N);
__global__ void FindNormW(float* g_VecW, float* g_NormW, int N);
__global__ void NormalizeW(float* g_VecV,float* g_VecW, int N);
__global__ void ComputeLamda( float* g_VecV,float* g_VecW, float* g_Lamda,int N);


void CPU_AvProduct()
{
	int N = GlobalSize;
	int matIndex =0;
    for(int i=0;i<N;i++)
	{
		h_VecW[i] = 0;
		for(int j=0;j<N;j++)
		{
			matIndex = i*N + j;
			h_VecW[i] += h_MatA[matIndex] * h_VecV[j];
			
		}
	}
}

void CPU_NormalizeW()
{
	int N = GlobalSize;
	float normW=0;
	for(int i=0;i<N;i++)
		normW += h_VecW[i] * h_VecW[i];
	
	normW = sqrt(normW);
	for(int i=0;i<N;i++)
		h_VecV[i] = h_VecW[i]/normW;
}

float CPU_ComputeLamda()
{
	int N = GlobalSize;
	float lamda =0;
	for(int i=0;i<N;i++)
		lamda += h_VecV[i] * h_VecW[i];
	
	return lamda;
}

void RunCPUPowerMethod()
{
	printf("*************************************\n");
	float oldLambda =0;
	float lamda=0;
	
	//AvProduct
	CPU_AvProduct();
	
	//power loop
	for (int i=0;i<max_iteration;i++)
	{
		CPU_NormalizeW();
		CPU_AvProduct();
		lamda= CPU_ComputeLamda();
		printf("CPU lamda at %d: %f \n", i, lamda);
		// If residual is lass than epsilon break
		if(abs(oldLambda - lamda) < EPS)
			break;
		oldLambda = lamda;	
	
	}
	printf("*************************************\n");
	
}

/*****************************************************************************
This function finds the product of Matrix A and vector V
******************************************************************************
// parallelization method for the Matrix-vector multiplication as follows: 

// each thread handle a multiplication of each row of Matrix A and vector V;

// The share memory is limited for a block, instead of reading an entire row of matrix A or vector V from global memory to share memory, 
// a square submatrix of A is shared by a block, the size of square submatrix is BLOCK_SIZE*BLOCK_SIZE; Thus, a for-loop is used to
// handle a multiplication of each row of Matrix A and vector V step by step. In each step, two subvectors with size BLOCK_SIZE is multiplied.
***************************************************************************************************************************************************/
__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{ 
    unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    float Csub = 0;

    for (int k = 0; k < N; ++k) 
    {
        Csub += g_MatA[globalid*N+k] * g_VecV[k];
    }
    __syncthreads();
    g_VecW[globalid] = Csub;

}

__global__ void ComputeLamda(float* g_VecV, float* g_VecW, float* g_Lamda, int N)
{
    unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

   // atomic operations:
    atomicAdd(g_Lamda, g_VecV[globalid] * g_VecW[globalid]);    
}

/****************************************************
Normalizes vector W : W/norm(W)
****************************************************/
__global__ void FindNormW(float* g_VecW, float* g_NormW, int N)
{ 
   unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

   // atomic operations:
   atomicAdd(g_NormW, g_VecW[globalid]*g_VecW[globalid]);
}

__global__ void NormalizeW(float* g_VecW, float* g_NormW, float* g_VecV, int N)
{
    unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;
    g_VecV[globalid] = g_VecW[globalid]/g_NormW[0];
    __syncthreads();
}

// Host code
int main(int argc, char** argv)
{
    struct timespec t_start,t_end;
    double runtime;
    Arguments(argc, argv);
		
    int N = GlobalSize;
    printf("Matrix size %d X %d \n", N, N);
    size_t vec_size = N * sizeof(float);
    size_t mat_size = N * N * sizeof(float);
    size_t norm_size = sizeof(float);

  
    // Allocate normalized value in host memory
    h_NormW = (float*)malloc(norm_size);
    // Allocate input matrix in host memory
    h_MatA = (float*)malloc(mat_size);
    // Allocate initial vector V in host memory
    h_VecV = (float*)malloc(vec_size);
    // Allocate W vector for computations
    h_VecW = (float*)malloc(vec_size);
    
    


    // Initialize input matrix
    UploadArray(h_MatA, N);
    InitOne(h_VecV,N);

    printf("Power method in CPU starts\n");	   
    clock_gettime(CLOCK_REALTIME,&t_start);
    RunCPUPowerMethod();   // the lamda is already solved here
    clock_gettime(CLOCK_REALTIME,&t_end);
    runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9*(t_end.tv_nsec - t_start.tv_nsec);
    printf("CPU: run time = %f secs.\n",runtime);
    printf("Power method in CPU is finished\n");
    
    
    /////////////////////////////////////////////////
    // This is the starting points of GPU
    printf("Power method in GPU starts(global mem)\n");
    checkCardVersion();
    int i;
    // Initialize input matrix
    InitOne(h_VecV,N);
    
    clock_gettime(CLOCK_REALTIME,&t_start);  // Here I start to count

    // Set the kernel arguments
    int threadsPerBlock = BlockSize;   
   // int sharedMemSize = threadsPerBlock * threadsPerBlock * sizeof(float); // in per block, the memory is shared   
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate matrix and vectors in device memory
    cudaMalloc((void**)&d_MatA, mat_size); 
    cudaMalloc((void**)&d_VecV, vec_size); 
    cudaMalloc((void**)&d_VecW, vec_size); // This vector is only used by the device
    cudaMalloc((void**)&d_NormW, norm_size); 

   //Power method loops
    float oldLambda =0;
   
    //Copy from host memory to device memory
    cudaMemcpy(d_MatA, h_MatA, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice);
    // cutilCheckError(cutStopTimer(timer_mem));
	  
   //initial w-vector 
    Av_Product<<<blocksPerGrid, threadsPerBlock>>>(d_MatA, d_VecV, d_VecW, N);
    cudaDeviceSynchronize(); //Needed, kind of barrier to sychronize all threads
	
    // This part is the main code of the iteration process for the Power Method in GPU. 
    // Please finish this part based on the given code. Do not forget the command line 
    // cudaThreadSynchronize() after calling the function every time in CUDA to synchoronize the threads
    ////////////////////////////////////////////
    //   ///      //        //            //          //            //        //

   	for (i=0;i<max_iteration;i++)
	{     
        
        h_NormW[0] = 0;
        cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
        
        FindNormW<<<blocksPerGrid, threadsPerBlock>>>(d_VecW, d_NormW, N);
        cudaDeviceSynchronize(); 
       
        //need to transfer from device to host ??
        cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
        
        h_NormW[0]=sqrt(h_NormW[0]);
        
        //transfer back to device ??
        cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
       
        NormalizeW<<<blocksPerGrid, threadsPerBlock>>>(d_VecW, d_NormW, d_VecV, N);
        cudaDeviceSynchronize(); 
        
        Av_Product<<<blocksPerGrid, threadsPerBlock>>>(d_MatA, d_VecV, d_VecW, N);
        cudaDeviceSynchronize(); 
        

        h_NormW[0] = 0;
        cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
        
        ComputeLamda<<<blocksPerGrid, threadsPerBlock>>>(d_VecV, d_VecW, d_NormW, N);//???
        cudaDeviceSynchronize(); 

		//transfer d_NormW to h_NormW via cudaMemcpy
        cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);

        printf("GPU lamda at %d: %f \n", i, h_NormW[0]);
		// If residual is less than epsilon break
		if(abs(oldLambda - h_NormW[0]) < EPS)
			break;
		oldLambda = h_NormW[0];//
	} 
    
    clock_gettime(CLOCK_REALTIME,&t_end);
    runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9*(t_end.tv_nsec - t_start.tv_nsec);
    printf("GPU: run time = %f secs.\n",runtime);
    printf("GPU: run time per iteration = %f secs.\n",runtime/(i+1));
    // printf("Overall CPU Execution Time: %f (ms) \n", cutGetTimerValue(timer_CPU));
    // printf("Overall CPU Execution Time: %f (ms) \n", cutGetTimerValue(timer_CPU));

    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_MatA)
        cudaFree(d_MatA);
    if (d_VecV)
        cudaFree(d_VecV);
    if (d_VecW)
        cudaFree(d_VecW);
	  if (d_NormW)
		    cudaFree(d_NormW);
		
    // Free host memory
    if (h_MatA)
        free(h_MatA);
    if (h_VecV)
        free(h_VecV);
    if (h_VecW)
        free(h_VecW);
    if (h_NormW)
        free(h_NormW);
    
    exit(0);
}

// Allocates an array with zero value.
void InitOne(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = 0;
	data[0]=1;
}

void UploadArray(float* data, int n)
{
   int total = n*n;
   int value=1;
    for (int i = 0; i < total; i++)
    {
    	data[i] = (int) (rand() % (int)(101));//1;//value;
	    value ++; if(value>n) value =1;
      // data[i] = 1;
    }
}

// Obtain program arguments
void Arguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) 
    {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0)
        {
            GlobalSize = atoi(argv[i+1]);
		    i = i + 1;
        }
        if (strcmp(argv[i], "--max_iteration") == 0 || strcmp(argv[i], "-max_iteration") == 0)
        {
            max_iteration = atoi(argv[i+1]);
		    i = i + 1;
        }
    }
}


void checkCardVersion()
{
   cudaDeviceProp prop;
   
   cudaGetDeviceProperties(&prop, 0);
   
   printf("This GPU has major architecture %d, minor %d \n",prop.major,prop.minor);
   if(prop.major < 2)
   {
      fprintf(stderr,"Need compute capability 2 or higher.\n");
      exit(1);
   }
}
