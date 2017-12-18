#include <iostream>
#include <sys/time.h>
#include <stdio.h>

int generateRandomNum(int max){
    return rand() % max;
}

void merge(int* sourceArr, int startIndex, int midIndex, int endIndex)
{
    int i = startIndex, j = midIndex + 1, k = 0;
    int temp[endIndex - startIndex + 1];
    
    while(i <= midIndex && j <= endIndex)
    {
        if(sourceArr[i] < sourceArr[j])
            temp[k++] = sourceArr[i++];
        else
            temp[k++] = sourceArr[j++];
    }
    
    //handle the longer array
    while(i <= midIndex){
        temp[k++] = sourceArr[i++];
    }
    
    while(j <= endIndex){
        temp[k++] = sourceArr[j++];
    }
    
    //copy back
    for(i = 0; i < endIndex - startIndex + 1; i++)
        sourceArr[i + startIndex] = temp[i];
    
}

void mergeSort(int* sourceArr, const int startIndex, const int endIndex)
{
    int midIndex = (startIndex + endIndex) / 2;
    if(startIndex < endIndex)
    {
        mergeSort(sourceArr, startIndex, midIndex);
        mergeSort(sourceArr, midIndex+1, endIndex);
        merge(sourceArr, startIndex, midIndex, endIndex);
    }
}

__device__ unsigned int findThreadId(dim3* threads, dim3* blocks){
     int x;
     return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}


__device__ void gpuToMerge(int* src, int* dest, int startIndex, int middleIndex, int endIndex){
	   int start = startIndex;
	   int middle = middleIndex;
	   for(int i = startIndex; i < endIndex; i++){
	   	   if(start < middleIndex && (middle >= endIndex || src[start] < src[middle])){
		   	    	   	   dest[i] = src[start++];	    
		   }else{
			dest[i] = src[middle++];
		   }
	   }
}


__global__ void cudaPerformSort(int* src, int* dest, int length, int range, int part, dim3* threads, dim3* block){
	 unsigned int id = findThreadId(threads, block);
	 int startIndex = length * range * part;
	 int middleIndex;
	 int endIndex;
	 for(int i = 0; i < part; i++){
	 	 if(startIndex >= length)
		 	       break;

	          middleIndex = min(startIndex + (range >> 1), length);
		  endIndex = min(startIndex + range, length);
		  gpuToMerge(src, dest, startIndex, middleIndex, endIndex);
		  startIndex += range;
	 }	   
}

/*
 * mergesort.cu
*/

void mergeSortCuda(int *data , const int length, dim3 dimBlock, dim3 dimGrid){
    int* gpuData;
    int* gpuTempData;
    dim3* threads;
    dim3* blocks;

    cudaMalloc(&gpuData, sizeof(int) * length);
    cudaMalloc(&gpuTempData, sizeof(int) * length);
    cudaMalloc(&threads, sizeof(dim3));
    cudaMalloc(&blocks, sizeof(dim3));

    
    cudaMemcpy(gpuData, data, sizeof(int) * length, cudaMemcpyHostToDevice);
    cudaMemcpy(threads, &dimBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(blocks, &dimGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    int* src = gpuData;
    int* temp = gpuTempData;

    for(int range = 2; range < (length << 1); range <<=1){
    	    long part = length / (dimBlock.x * dimGrid.x) * range + 1;
	    cudaPerformSort<<<dimGrid, dimBlock>>>(src, temp, length, range, part, threads, blocks);
	    src = src == gpuData ? gpuTempData : gpuData;
	    temp = temp == gpuData ? gpuTempData : gpuData;
    }

    cudaMemcpy(data, src, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(src);
    cudaFree(temp);
}

bool checkResult(int* a, int* b, int length){
     for(int i = 0; i < length ; i++){
     	     if(a[i] != b[i])
	          	     return false;
     }
     return true;
}


__device__
void mergeV(int* sourceArr, int* temp, int startIndex, int midIndex, int endIndex){
      int i = startIndex, j = midIndex + 1, k = 0;
   
    
    while(i <= midIndex && j <= endIndex)
    {
        if(sourceArr[i] < sourceArr[j])
            temp[k++] = sourceArr[i++];
        else
            temp[k++] = sourceArr[j++];
    }
    
    //handle the longer array
    while(i <= midIndex){
        temp[k++] = sourceArr[i++];
    }
    
    while(j <= endIndex){
        temp[k++] = sourceArr[j++];
    }
    
    //copy back
    for(i = 0; i < endIndex - startIndex + 1; i++)
        sourceArr[i + startIndex] = temp[i];
}

__global__ void mergesortV(int* data, int* dataTemp, int start,int end, int depth){
    int middle = (end + start) / 2;

    cudaStream_t s,s1;
    if(end < start){
            //execute on the left part
    	    cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking);
	    mergesortV<<< 1, 1, 0, s >>>(data, dataTemp, start, middle, depth+1);
	    cudaStreamDestroy(s);

    	    //execute on the right part
	    cudaStreamCreateWithFlags(&s1,cudaStreamNonBlocking);
	    mergesortV<<< 1, 1, 0, s1 >>>(data, dataTemp, middle + 1, end, depth+1);
	    cudaStreamDestroy(s1);
	    cudaDeviceSynchronize();

	    mergeV(data, dataTemp, start, middle, end);
	    
    }	 
}

extern "C"
void gpuMergeSort(int* a,int n){
    int* gpuData;
    int* gpuTempData;
    int left = 0;
    int right = n - 1;

    for(int i = 0; i < n; i++){
    	    printf("%d ", a[i]);
    }
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 100);

    cudaMalloc(&gpuData, n * sizeof(int));
    cudaMalloc(&gpuTempData, n * sizeof(int));

    
    //cudaMalloc((void**)&gpuData, n * sizeof(int));
    //cudaMalloc((void**)&gpuTempData, n * sizeof(int));

    cudaMemcpy(gpuData, a, n * sizeof(int), cudaMemcpyHostToDevice);

    mergesortV<<< 1, 1 >>>(gpuData, gpuTempData, left, right, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(a,gpuData, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpuTempData);
    cudaFree(gpuData);
}

int main(){

    const int length = 4;
    int* toSortArray = new int[length];
    int* cudaToSortArray = new int[length];
    
    //initiate the array
    for(int i = 0; i < length; i ++){
        toSortArray[i] = generateRandomNum(1000);
    }

    memcpy(cudaToSortArray, toSortArray, length * sizeof(int));

    dim3 dimBlock(32, 1, 1);
    dim3 dimGrid(8, 1, 1);

    struct timeval tpstart, tpend;
    long timeuse;

    for(int i = 0; i < 10; i ++){
    
    gettimeofday( &tpstart, NULL );
    //mergeSortCuda(cudaToSortArray, length, dimBlock, dimGrid);	
    gpuMergeSort(cudaToSortArray, length);
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("GPU looped version is finished in time %ld ms\n", timeuse);
    }

    gettimeofday( &tpstart, NULL );
    mergeSort(toSortArray, 0, length - 1);
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("CPU looped version is finished in time %ld ms\n", timeuse);

    std::cout << "Confirm sort result is : " << checkResult(cudaToSortArray, toSortArray, length) << std::endl;

    //for(int i=0; i < length; i++)
        //printf("%d ", toSortArray[i]);

	//printf("\n\n");
    for(int i=0; i < length; i++)
        printf("%d ", cudaToSortArray[i]);
        
    return 0;
}