// the subroutine for GPU code can be found in several separated text file from the Brightspace. 
// You can add these subroutines to this main code.
////////////////////////////////////////////
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cuda.h"

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
int GlobalSize = 2000;         // this is the dimension of the matrix, GlobalSize*GlobalSize
int BlockSize = 64;          // number of threads in each block
const float EPS = 0.000001;    // tolerence of the error
int max_iteration = 200;       // the maximum iteration steps

const int BLOCK_SIZE = 64;  // number of threads per block

// Functions
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
float CPUReduce(float*, int);
void  Arguments(int, char**);
void checkCardVersion(void);

// Kernels
// "global" indicates that the kernel is called by the host to execute in a GPU
// "device" indicates that the kernel is called by another kernel/device function
//Kernels never return a value, so return type must be set to void
//Any variables operated in the kernel need to be passed by reference
__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N);
__global__ void FindNormW(float* g_VecW, float * g_NormW, int N);
__global__ void NormalizeW(float* g_VecV,float* g_VecW, int N);
__global__ void ComputeLamda( float* g_VecV,float* g_VecW, float * g_Lamda,int N);

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
*****************************************************************************************************************************************************/
__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{    
    /*local variables, private to each thread 
       -every thread will have its own copy of the local variables*/
    //Each thread knows its own index, its block index and the grid
    // Block index
    int bx = blockIdx.x;
    // Thread index
    int tx = threadIdx.x;
    int aBegin = N * BLOCK_SIZE * bx;
    int aEnd   = aBegin + N - 1;
    int step  = BLOCK_SIZE;
    int bBegin = 0;//BLOCK_SIZE * bx;
    int bIndex=0;
    int aIndex =0;
    float Csub = 0;

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += step, b += step)
    {
        //__shared__ variables are visible to all threads in the thread block
        //and have the same lifetime as the thread block
        //__shared__float ...: statically allocated shared memory
        //This allocates BLOCK_SIZE * BLOCK_SIZE * 4 + BLOCK_SIZE * 4 bytes of shared mem per block
        __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];//32 * 32 = 1024
        __shared__ float bs[BLOCK_SIZE];        

        for (int aa = 0; aa < BLOCK_SIZE; aa+= 1)
        {
            aIndex = a+tx+aa*N;
            if( aIndex < N*N)
            {
                As[tx+aa*BLOCK_SIZE] = g_MatA[aIndex];//copy data from MatA in global memory to As in shared memory
                /* For BLOCK_SIZE = 32, N = 50, in block 0
                    Thread O                                 Thread 1
                1: As[0] = g_MatA[0]                    1: As[1] = g_MatA[1]
                2: As[0+32] = g_MatA[50]                2: As[1+32] = g_MatA[51]
                3: As[0+64] = g_MatA[100]
                
                NOTE: As is in shared memory: all threads in one block has access to it
                */
            }
            else
            {
                As[tx+aa*BLOCK_SIZE] = 0;
            }
        }

        bIndex = b+tx;
   	    if(bIndex<N)   
		      bs[tx] = g_VecV[bIndex];//copy data from VecV in global memory to bs in shared memory
	      else
		      bs[tx] = 0;

        __syncthreads();//ensure all the writes to shared memory has been completed

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            //write from shared to local memory
            //Csub is local variable: each thread has its own copy
            Csub += As[k+tx*BLOCK_SIZE] * bs[k];
        }
        __syncthreads();//ensure all the writes to local memory has been completed
    }
    //writing from local to global memory
    g_VecW[BLOCK_SIZE * bx + tx] = Csub;//variable in global memory can be seen by other thread blocks
}

__global__ void ComputeLamda( float* g_VecV, float* g_VecW, float * g_Lamda,int N)
{
    // dynamically allocated shared mem: extern__....
    // shared memory size per block(bytes) declared at kernel launch
    extern __shared__ float sdataVW[];
    /*local memory*/
    unsigned int tid = threadIdx.x;
    unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

    // For thread ids greater than data space
    if (globalid < N) 
    {
        sdataVW[tid] =  g_VecV[globalid] * g_VecW[globalid];
    }
    else
    {
        sdataVW[tid] = 0;  // Case of extra threads above N
    }

    // each thread loads one element from global to shared mem
    __syncthreads();

    // do reduction in shared mem(only works for even sizes)
    // on each iteration, divide the length of the block by half
    // each thread will add its element in the 1st half to the element in the 2nd half
    // and write to the 1st half
    for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
        if (tid < s)
        {
            sdataVW[tid] = sdataVW[tid] + sdataVW[tid+ s];
        }
        __syncthreads();
    }
    // atomic operations:
    if (tid == 0) 
        atomicAdd(g_Lamda,sdataVW[0]);
}

/****************************************************
Normalizes vector W : W/norm(W)
****************************************************/
__global__ void FindNormW(float* g_VecW, float * g_NormW, int N)
{ 
    // shared memory size declared at kernel launch
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

    // For thread ids greater than data space
    if (globalid < N) 
    {
        sdata[tid] =  g_VecW[globalid];
    }
    else 
    {
        sdata[tid] = 0;  // Case of extra threads above N
    }

    // each thread loads one element from global to shared mem
    __syncthreads();

    sdata[tid] = sdata[tid] * sdata[tid];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
        if (tid < s) 
        {
            sdata[tid] = sdata[tid] + sdata[tid+ s];
        }
        __syncthreads();
    }
    // atomic operations:
    if (tid == 0) 
        atomicAdd(g_NormW,sdata[0]);
}

__global__ void NormalizeW(float* g_VecW, float * g_NormW, float* g_VecV, int N)
{
    // shared memory size declared at kernel launch
    extern __shared__ float sNormData[];
    unsigned int tid = threadIdx.x;
    unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid==0) 
    {
        sNormData[0] =  g_NormW[0];
    }    
    __syncthreads();
    // For thread ids greater than data space
    if (globalid < N) 
    {
        g_VecV[globalid] = g_VecW[globalid]/sNormData[0];
    }
}
// Host code
int main(int argc, char** argv)
{
    struct timespec t_start,t_end;
    struct timespec t_start_memcpy_loc_6, t_end_memcpy_loc_6;
    struct timespec t_start_memcpy_loc_1, t_end_memcpy_loc_1;
    struct timespec t_start_memcpy_loc_2, t_end_memcpy_loc_2;
    struct timespec t_start_memcpy_loc_3, t_end_memcpy_loc_3;
    struct timespec t_start_memcpy_loc_4, t_end_memcpy_loc_4;
    struct timespec t_start_memcpy_loc_5, t_end_memcpy_loc_5;
    double runtime;
    double time_memcpy=0.0, runtime_nomemcpy;

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
    printf("Power method in GPU starts\n");
    checkCardVersion();

    // Initialize input matrix
    InitOne(h_VecV,N);
    
    clock_gettime(CLOCK_REALTIME,&t_start);  // Here I start to count

    // Set the kernel arguments
    //1D in this case, but can be set to 2/3D using the struct "dim3 variable_name(x,y,z)"
    //By default it is 1D : dim3 var_name(x,1,1) = dim3 var_name(x)
    int threadsPerBlock = BlockSize;   
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;//number of blocks in the grid
    
    int sharedMemSize = threadsPerBlock * threadsPerBlock * sizeof(float); // in per block, the memory is shared
    printf("Number of threads per block: %d\n", threadsPerBlock);
    printf("sharedMemSize = %.5d\n",sharedMemSize);

    // Allocate matrix and vectors in device memory
    cudaMalloc((void**)&d_MatA, mat_size); 
    cudaMalloc((void**)&d_VecV, vec_size); 
    cudaMalloc((void**)&d_VecW, vec_size); // This vector is only used by the device
    cudaMalloc((void**)&d_NormW, norm_size); 

    //Copy from host memory to device memory
    clock_gettime(CLOCK_REALTIME, &t_start_memcpy_loc_6); 
    cudaMemcpy(d_MatA, h_MatA, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_REALTIME, &t_end_memcpy_loc_6);
    // cutilCheckError(cutStopTimer(timer_mem));
	  
   //Power method loops
    float oldLambda =0;

    Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_MatA, d_VecV, d_VecW, N);
    cudaDeviceSynchronize(); //Needed, kind of barrier to sychronize all threads
	
    // This part is the main code of the iteration process for the Power Method in GPU. 
    // Please finish this part based on the given code. Do not forget the command line 
    // cudaThreadSynchronize() after calling the function every time in CUDA to synchoronize the threads
    ////////////////////////////////////////////
    //   ///      //        //            //          //            //        //

   	for (int i=0;i<max_iteration;i++)
	{       
        h_NormW[0] = 0;

        clock_gettime(CLOCK_REALTIME, &t_start_memcpy_loc_1);
        cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
        clock_gettime(CLOCK_REALTIME, &t_end_memcpy_loc_1);
        //runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9*(t_end.tv_nsec - t_start.tv_nsec);
        
        FindNormW<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_VecW, d_NormW, N);
        cudaDeviceSynchronize(); 
       
        //need to transfer from device to host ??
        clock_gettime(CLOCK_REALTIME, &t_start_memcpy_loc_2);
        cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
        clock_gettime(CLOCK_REALTIME, &t_end_memcpy_loc_2);

        h_NormW[0]=sqrt(h_NormW[0]);
        
        //transfer back to device ??
        clock_gettime(CLOCK_REALTIME, &t_start_memcpy_loc_3);
        cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
        clock_gettime(CLOCK_REALTIME, &t_end_memcpy_loc_3);
       
        NormalizeW<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_VecW, d_NormW, d_VecV, N);
        cudaDeviceSynchronize(); 
        
        Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_MatA, d_VecV, d_VecW, N);
        cudaDeviceSynchronize(); 
        
        h_NormW[0] = 0;
        clock_gettime(CLOCK_REALTIME, &t_start_memcpy_loc_4);
        cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
        clock_gettime(CLOCK_REALTIME, &t_end_memcpy_loc_4);

        ComputeLamda<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_VecV, d_VecW, d_NormW, N);
        cudaDeviceSynchronize(); 

		//transfer d_NormW to h_NormW via cudaMemcpy
        clock_gettime(CLOCK_REALTIME, &t_start_memcpy_loc_5);
        cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
        clock_gettime(CLOCK_REALTIME, &t_end_memcpy_loc_5);
        
        time_memcpy += (t_end_memcpy_loc_1.tv_sec - t_start_memcpy_loc_1.tv_sec) + 1e-9*(t_end_memcpy_loc_1.tv_nsec - t_start_memcpy_loc_1.tv_nsec)
                      +(t_end_memcpy_loc_2.tv_sec - t_start_memcpy_loc_2.tv_sec) + 1e-9*(t_end_memcpy_loc_2.tv_nsec - t_start_memcpy_loc_2.tv_nsec)
                      +(t_end_memcpy_loc_3.tv_sec - t_start_memcpy_loc_3.tv_sec) + 1e-9*(t_end_memcpy_loc_3.tv_nsec - t_start_memcpy_loc_3.tv_nsec)
                      +(t_end_memcpy_loc_4.tv_sec - t_start_memcpy_loc_4.tv_sec) + 1e-9*(t_end_memcpy_loc_4.tv_nsec - t_start_memcpy_loc_4.tv_nsec)
                      +(t_end_memcpy_loc_5.tv_sec - t_start_memcpy_loc_5.tv_sec) + 1e-9*(t_end_memcpy_loc_5.tv_nsec - t_start_memcpy_loc_5.tv_nsec);


        printf("GPU lamda at %d: %f \n", i, h_NormW[0]);
        //printf("GPU memcpy at %d: %f \n", i, time_memcpy);
		// If residual is less than epsilon break
		if(abs(oldLambda - h_NormW[0]) < EPS)
			break;
		oldLambda = h_NormW[0];//
	} 
    
    clock_gettime(CLOCK_REALTIME,&t_end);
    runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9*(t_end.tv_nsec - t_start.tv_nsec);
    time_memcpy += (t_end_memcpy_loc_6.tv_sec - t_start_memcpy_loc_6.tv_sec) + 1e-9*(t_end_memcpy_loc_6.tv_nsec - t_start_memcpy_loc_6.tv_nsec);
    runtime_nomemcpy = runtime - time_memcpy;
    printf("GPU: run time = %f secs.\n",runtime);
    printf("Memcpy: run time = %f secs.\n",time_memcpy);
    printf("GPU: run time without memcpy = %f secs.\n", runtime_nomemcpy);
    
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
   int value = 1;
    for (int i = 0; i < total; i++)
    {
    	data[i] = (int) (rand() % (int)(101));
        //print array
	    /*printf("%10.5f", data[i]);
        printf(((i % n) !=(n-1) ) ? "\t" : "\n");*/
        value ++; 
        if(value>n) 
            value =1;
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

//get compute capability here
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
