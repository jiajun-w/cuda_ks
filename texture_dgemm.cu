#include<stdio.h>
#include "cublas.h"
#include <cuda_runtime.h>
#include "cycleTimer.h"
#define warpSize 32

//texture<int2,1> texture_A;
texture<float,1> texture_B;

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__global__
void tileMul(float *_A, float *_B, float *_C,int m, int n, int k, int bm, int bn, int bk){
        //assume A is row-major and B is column-major and C has initial value all zero
	//tileA: 1 * 32*integer; tileB:32*integer * 32; tileC: 1*32 
	int id = threadIdx.x;
	float value;
	int i,j,l,loop;
	int tileNum,currentRow;
    int idMod32 = id%32;
    int idDivide32 = id/32;
	//int M = (m+bm-1)/bm;
	//int N = (n+bn-1)/bn;
	int K = (k+bk-1)/bk;
	int totTileNum = K;

//	use share memory
	__shared__ float rowA[32]; //FIXME:dynamic allocate tempC[bn*bm]
	__shared__ float tempC[1504]; //1536-32, bm*bn<1024, bm=31; bn=48
	
	//#pragma unroll, doesn't bring improvement
	for(tileNum=0 ; tileNum < totTileNum; tileNum++){
		/* tileC += tileA[i] * tileB[i]*/
		//rowC = rowA * blockB; (1,k) * (k, bn) = (1, bn) 
		//load rowA from _A to rowA[]
		for(currentRow=0; currentRow < bm; currentRow++){
			if(id<32) rowA[id] = _A[k*currentRow + tileNum*32 + id]; 
   			//int reduceSize = ((k - 32*loop)>0) ? 32 : (k%32);	
			int reduceSize = 32;
			//#pragma unroll
			for(loop=0; (loop*32)<bn; loop++){
			//for(loop=0; loop<bn; loop+=32){
        		value = rowA[idMod32] * tex1Dfetch(texture_B, idDivide32*k + tileNum*32 + idMod32);
        	    //Shuffle Warp Reduce 
        	    for (l=16; l>=1; l/=2)
        	        value += __shfl_down(value, l);
//      	      printf("Thread %d final value = %f\n", threadIdx.x, value);
        	    if(idMod32 == 0) {
					//tempC[currentRow][idDivide32+loop*32] +=value;		
					tempC[currentRow*bn + idDivide32+loop*32] +=value;		
					//printf("C_temp[%d][%d] final value = %f\n",currentRow,idDivide32+loop*32,value);
		    	}
				if(tileNum == totTileNum-1)
					_C[currentRow*bn + idDivide32+loop*32] = tempC[currentRow*bn + idDivide32+loop*32];
    		}
		}
		__syncthreads();
	}	
}

int main(int argc, char* argv[]){
	int m,n,k;
	int i,j,l;
//	m=4096; n=4096; k=4096;
	sscanf( argv[ 1 ], "%d", &m );
	sscanf( argv[ 2 ], "%d", &n );
	sscanf( argv[ 3 ], "%d", &k );
	float *A = (float*)malloc(sizeof(float)*m*k);
	float *B = (float*)malloc(sizeof(float)*k*n);
	float *C = (float*)malloc(sizeof(float)*m*n);
//for cublas
/*	for(j=0; j<k; j++){
		for(i=0; i<m; i++){
			A[j*m+i]=(10*i+j)*0.01; //store A in column major, size m*k
		}
	}
*/
//for tile_mul

	for(i=0; i<m; i++){
		for(j=0; j<k; j++){
			A[i*k+j]=(10*i+j)*0.01; //store A in row major, size m*k
		}
	}

	for(j=0; j<n; j++){
		for(i=0; i<k; i++){
			B[j*k+i]=(10*i+j)*0.01;//store B in column major, size k*n
                }
        }
	if(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) != cudaSuccess)
		printf("SharedMemBankSizeEightByte failed.\n");
	cudaSharedMemConfig pConfig;
	cudaDeviceGetSharedMemConfig(&pConfig);
	printf("cudaSharedMemBankSize=%d\n",pConfig);//cudaSharedMemBankSizeDefault = 0
						     //cudaSharedMemBankSizeFourByte = 1
						     //cudaSharedMemBankSizeEightByte = 2
	float* dev_A,*dev_B,*dev_C;
	cudaMalloc((void**)&dev_A,m*k*sizeof(float));	
	cudaMalloc((void**)&dev_B,k*n*sizeof(float));	
	cudaMalloc((void**)&dev_C,m*n*sizeof(float));	
	cudaMemcpy(dev_A,A,m*k*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B,B,k*n*sizeof(float),cudaMemcpyHostToDevice);
	cublasInit();
	float alpha = 1.0;
	float beta = 0.0;
	//cudaBindTexture(NULL,texture_A,dev_A,m*k*sizeof(float));
	cudaBindTexture(NULL,texture_B,dev_B,k*n*sizeof(float));
	int bm=m;
	int bn=n;
	int bk=32;
	double cpuStartTime = CycleTimer::currentSeconds();
	tileMul<<<1,1024>>>(dev_A, dev_B, dev_C, m, n, k,bm,bn,bk);
//	cublasDgemm( 'n', 'n', m, n, k, alpha, dev_A, m, dev_B, k, beta, dev_C, m);
	cudaThreadSynchronize();
	double cpuEndTime = CycleTimer::currentSeconds();
	double runtime = 1000.f * (cpuEndTime-cpuStartTime);
	double flop = (double)2*m*n*k;
        printf("Dgemm runtime: %.3f ms, GFLOPS=%.6f\n", runtime,flop/runtime/1000000 );
	cudaMemcpy(C,dev_C,m*n*sizeof(float),cudaMemcpyDeviceToHost);
//	cudaUnbindTexture(texture_A);
//	cudaUnbindTexture(texture_B);
	cublasShutdown();
	printf("cuda blas:\n");
	printf("m=%d,n=%d,k=%d ",m,n,k);
	for(i=0; i<m; i++){
                printf("\n");
                for(j=0; j<n; j++)
                        printf("%f      ",C[i*n+j]);
        }
		
	return 0;
}

