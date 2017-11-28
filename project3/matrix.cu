#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <memory>
#include <cublas_v2.h>


class Matrix{
 
public:
    int row;
    int col;
    float* matrix;
   
    Matrix(int row, int col){
        this->row = row;
	this->col = col;
        matrix = generateRandomMatrix(row, col);
    }
    
    Matrix(int row, int col, float* elements){
    	this->row = row;
	this->col = col;
	matrix = new float[row * col];

        for(int i = 0 ; i < row; i++){
            for(int j = 0; j < col; j++){
    //            setValue(i, j, elements[i * col + j]);
	  	  setValue(i, j, *(elements + i * col + j));
            }
        }
    }


    ~Matrix(){
       delete matrix;
    }
    
    void printMatrix(){
        for(int i = 0 ; i < row; i++){
            for(int j = 0; j < col; j++){
                printf("%f \t ", getVal(i, j));
            }
            printf("\n");
        }
    }
    
    int getRow(){
        return row;
    }
    
    int getCol(){
        return col;
    }
    
    float getVal(int row, int col){
        return *(matrix + row * this->col + col);
    }
    
    void setValue(int row, int col, float val){
        *(matrix + row * this->col + col) = val;
    }
    
    float* getMatrix(){
        return matrix;
    }
    
    size_t getSize(){
        return row * col * sizeof(float);
    }
    
    static float matrixComparison(Matrix* a, Matrix* b){
    	  float re = 0.f;
   	  for(int i = 0; i < a->row; i++){
	  	  for(int j = 0; j < a->col; j++){
		   	  float tmp =  a->getVal(i,j) - b->getVal(i,j);
			  re = tmp * tmp;
		  }
	  }
	  return re;
    }
private:
    
    float* generateRandomMatrix(const int row, const int col){
        matrix = new float[row * col];
        
        for(int i = 0 ; i < row; i++){
            for(int j = 0; j < col; j++){
                setValue(i, j, generateRandomNum());
            }
        }
        return matrix;
    }
    
    float generateRandomNum(){
        return static_cast <float> (rand());
    }
};


Matrix* calculateMatrixMultiplication(Matrix a, Matrix b){
    assert(a.getCol() == b.getRow());
    //initiate new matrix
    Matrix* c = new Matrix(a.getRow(), b.getCol());
    
    //comppute multiplication.
    for(int i = 0 ; i < a.getRow(); i++){
        for(int j = 0; j < b.getCol(); j++){
            c->setValue(i, j, 0);
            for(int k = 0; k < a.getCol(); k++){
                double updateVal = c->getVal(i,j) + a.getVal(i,k) * b.getVal(k,j);
                c->setValue(i, j, updateVal);
            }
        }
    }
    
    return c;
}


__global__
void cudaCompute(float* a, float *b, float *c, const int m, const int n, const int k)
{
    float result = 0.f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row > m or col > k)return;
    
    for(int i = 0; i < k; i ++){
    	float aVal = *(a + row * k + i);
	float bVal = *(b + row * n + i);
        result += aVal * bVal;
    }
 
    *(c + row * n + col) = result;
    //c->setValue(row, col, result);
}

Matrix* CudacalculateMatrixMultiplication(Matrix a, Matrix b){
    assert(a.getCol() == b.getRow());
    Matrix* c = new Matrix(a.getRow(), b.getCol());
    
    //assign cuda memory
    //Matrix *d_a = new Matrix(a.row, a.col);
    //Matrix *d_b = new Matrix(b.row, b.col);
    //Matrix *d_c = new Matrix(c->row, c->col);
    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, a.getSize());
    cudaMalloc(&d_b, b.getSize());
    cudaMalloc(&d_c, c->getSize());
 
    cudaMemcpy(d_a, a.getMatrix(), a.getSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.getMatrix(), b.getSize(), cudaMemcpyHostToDevice);
    
    //initiate cuda computing size
    const int BLOCK_SIZE = 16;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((b.getCol() + dimBlock.x - 1) / dimBlock.x, (a.getRow() + dimBlock.y - 1) / dimBlock.y);
 
    //perform cuda calculation
    cudaCompute<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, a.row, b.col, a.col);
    
    //copy data back
    cudaMemcpy(c->getMatrix(), d_c, c->getRow() * c->getCol() * sizeof(float), cudaMemcpyDeviceToHost);
 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    return c;
}

/*
	cublas version multiplication
*/
Matrix* cudaBlasCalculateMatrixMultiplication(Matrix a, Matrix b){
  
    assert(a.getCol() == b.getRow());

    //allocate gpu space
    float* d_a, * d_b, *d_c;

    cudaMalloc(&d_a, a.getSize());
    cudaMalloc(&d_b, b.getSize());
    cudaMalloc(&d_c, a.row * b.col * sizeof(float));

    //copy two operation matrix to gpu
    cudaMemcpy(d_a, a.getMatrix(), a.getSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.getMatrix(), b.getSize(), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alf = 1.0f;
    const float beta = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, a.row, b.col, a.col, &alf, d_a, a.row, d_b, a.col, &beta, d_c, a.row);
    float* c = (float*)malloc(a.row * b.col * sizeof(float));

    //get the result back
    cudaMemcpy(c, d_c, a.row * b.col * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    Matrix* re = new Matrix(a.row, b.col, c);
  
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
    free(c);
    
    return re;
}

int main(){
    struct timeval tpstart, tpend;
    long timeuse;
    
    const int m(1000), n(1000);
    //Matrix a(m, n);
    //Matrix b(n, m);
    
    float tmp[] = {1, 1, 1, 1};
    Matrix a(2, 2, tmp);
    Matrix b(2, 2, tmp);

    // run cuda version
    gettimeofday( &tpstart, NULL );		
    //Matrix* cudaRe = CudacalculateMatrixMultiplication(a,b);
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("Cuda finish time %ld ms\n", timeuse);
    //cudaRe->printMatrix();

    //run cublas version
    gettimeofday( &tpstart, NULL );
    //Matrix* cudaBlasRe = cudaBlasCalculateMatrixMultiplication(a,b);
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("Cublas finish time %ld ms\n", timeuse);
    //cudaBlasRe->printMatrix();

    //float re = Matrix::matrixComparison(cudaRe, cudaBlasRe);
    //printf("Cudablas version error is %f\n", re);
    
    //run cpu version..
    gettimeofday( &tpstart, NULL );
    Matrix* cpuRe = calculateMatrixMultiplication(a, b);
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("Cpu finish time %ld ms\n", timeuse);
    cpuRe->printMatrix();

    //re = Matrix::matrixComparison(cpuRe, cudaBlasRe);
    //printf("cpu version error is %f\n", re);

    //delete cudaBlasRe;
    //delete cpuRe;

    return 0;
}
