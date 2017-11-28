#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <memory>
#include <vector>
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
    
    Matrix(Matrix* x){
         this->row = x->row;
	 this->col = x->col;
	 matrix = new float[x->row * x->col];
	 memcpy(matrix, x->matrix, x->getSize());
    }

    Matrix(int row, int col, float* elements, int cuBlasMode){
    	this->row = row;
	this->col = col;
	matrix = new float[row * col];
	//int count = 0;
        for(int i = 0 ; i < col; i++){
            for(int j = 0; j < row; j++){
	    	 // printf("result= %f\n", *(elements + count++));
	  	  setValue(j, i, *(elements + i * row + j));
            }
        }

    }

    ~Matrix(){
       delete matrix;
    }
    
    void transferMatrix(){
    	const size_t size = getSize();
    	float matrixData[size]; 
	int count = 0;
	for(int i = 0; i < col; i ++){
		for(int j = 0; j < row; j++){
			matrixData[count++] = getVal(j,i);
		}
	}

	//reset value
	count = 0; 
	for(int j = 0; j < row; j++){
		for(int i = 0; i < col; i++){
			setValue(j, i, matrixData[count++]);
		}
	}
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
        return static_cast <float> (rand() % 10);
    }
};


Matrix* calculateMatrixMultiplication(Matrix &a, Matrix &b){
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


/*
	@param
	@a: matrix a
	@b: matrix b
	@c: matrix c
	@m: matrix a row
	@n: matrix b col
	@k: matrix a col & matrix b row
*/
__global__
void cudaCompute(float* a, float *b, float *c, const int m, const int n, const int k)
{
    float result = 0.f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row > m or col > k)return;
    
    for(int i = 0; i < k; i ++){
    	float aVal = *(a + row * k + i);
	float bVal = *(b + i * n + col);
        result += aVal * bVal;
    }
 
    *(c + row * n + col) = result;
    //c->setValue(row, col, result);
}

Matrix* CudacalculateMatrixMultiplication(Matrix &a, Matrix &b){
    assert(a.getCol() == b.getRow());
    Matrix* c = new Matrix(a.getRow(), b.getCol());
    
    //assign cuda memory
    float* d_a, *d_b, *d_c;
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
    cudaMemcpy(c->getMatrix(), d_c, c->getSize(), cudaMemcpyDeviceToHost);
 
    //clean memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    return c;
}

/*
	cublas version multiplication
*/
Matrix* cudaBlasCalculateMatrixMultiplication(Matrix &a, Matrix &b){
  
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
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, a.row, b.col, a.col, &alf, d_a, a.row, d_b, a.col, &beta, d_c, a.row);
    //cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, b.col, a.row, a.col, &alf, d_a, a.col, d_b, a.row, &beta, d_c, a.col);

    float* c = (float*)malloc(a.row * b.col * sizeof(float));

    //get the result back
    cudaMemcpy(c, d_c, a.row * b.col * sizeof(float), cudaMemcpyDeviceToHost);
    //cublasGetMatrix(a.row, b.col, a.row * b.col * sizeof(float), d_c, a.row, c, a.row);
    cublasDestroy(handle);

    Matrix* re = new Matrix(a.row, b.col, c, 1);

    cudaFree(d_c);
    cudaFree (d_b);
    cudaFree(d_a);
    free(c);
    
    return re;
}

int main(){
    struct timeval tpstart, tpend;
    long timeuse;
    
    const int m(1000), k(500), n(1000);
    Matrix a(m, k);
    Matrix b(k, n);
    
    printf("print operation matries\n");
    //a.printMatrix();
    //b.printMatrix();
    float tmp[] = {0, 1, 2, 2, 0, 1};
    //Matrix a(2, 3, tmp);
    //Matrix b(3, 2, tmp);
    //a.printMatrix();
    //b.printMatrix();
    
    //run cublas version
    //prepare transferMatrix
    Matrix aT(&a), bT(&b);
    aT.transferMatrix();
    bT.transferMatrix();
    //aT.printMatrix();
    //bT.printMatrix();

    gettimeofday( &tpstart, NULL );
    Matrix* cudaBlasRe = cudaBlasCalculateMatrixMultiplication(aT, bT);
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("Cublas finish time %ld ms\n", timeuse);
    //cudaBlasRe->printMatrix();

    // run cuda version
    gettimeofday( &tpstart, NULL );		
    Matrix* cudaRe = CudacalculateMatrixMultiplication(a,b);
    cudaDeviceSynchronize();
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("Cuda finish time %ld ms\n", timeuse);
    //cudaRe->printMatrix();

    float re = Matrix::matrixComparison(cudaRe, cudaBlasRe);
    printf("Cuda version error is %f\n", re);
    
    //run cpu version..
    gettimeofday( &tpstart, NULL );
    Matrix* cpuRe = calculateMatrixMultiplication(a, b);
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("Cpu finish time %ld ms\n", timeuse);
    //cpuRe->printMatrix();

    re = Matrix::matrixComparison(cpuRe, cudaBlasRe);
    printf("cpu version error is %f\n", re);

    delete cudaBlasRe;
    delete cudaRe;
    delete cpuRe;

    return 0;
}
