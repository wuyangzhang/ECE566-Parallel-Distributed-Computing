#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

pthread_mutex_t lock[3];

void printAugmentedMatrix(const std::vector<std::vector<double>>*A, const std::vector<double>*b);

typedef struct GaussianEliminationParameters{
    std::vector<std::vector<double>>* A;
    std::vector<double>* b;
    std::vector<double>* y;
    int n;
    int numThread;
    
    GaussianEliminationParameters(std::vector<std::vector<double>>* A, std::vector<double>*b, std::vector<double>*y, const int& numThread, const int & n){
        this->A = A;
        this->b = b;
        this->y = y;
        this->numThread = numThread;
        this->n = n;
    }
    
    void print(){
        printAugmentedMatrix(A, b);
    }
    
}GaussianEliminationParameters;

GaussianEliminationParameters* gep;


typedef struct GaussianParallelParameters{
    int temp1;
    int temp2;
    int threadIndex;
    void setParameters(const int temp1, const int temp2, const int threadIndex){
        this->temp1 = temp1;
        this->temp2 = temp2;
        this->threadIndex = threadIndex;
    }
}GaussianParallelParameters;

void* divisionStepParallel(void* args){
    GaussianParallelParameters* gpp = (GaussianParallelParameters*) args;
    /* gpp->i = k */
    int k = gpp->temp1;
    for(int j = k + 1 + gpp->threadIndex; j < gep->n; j += gep->numThread){
        gep->A->at(k)[j] = gep->A->at(k)[j] / gep->A->at(k)[k];
    }
    return nullptr;
}

void* eliminationStepParallel(void *args){
    GaussianParallelParameters* gpp = (GaussianParallelParameters*) args;
    /* gpp->temp1 = k */
    int k = gpp->temp1;
    int i = gpp->temp2;
    for(int j = k + 1 + gpp->threadIndex; j < gep->n; j += gep->numThread){
        gep->A->at(i)[j] = gep->A->at(i)[j] - gep->A->at(i)[k] * gep->A->at(k)[j];
    }
    return nullptr;
}

pthread_barrier_t barrier;

void calculateGaussianElimination_Parallel_InLoop(){
    const int numThread = gep->numThread;
    GaussianParallelParameters** gpp = new GaussianParallelParameters*[numThread];
    for(int i = 0; i < numThread; i++){
        gpp[i] = new GaussianParallelParameters();
    }
    
    for(int k = 0; k < gep->n; k++){
        /* division step */
        pthread_t _thread[numThread];
        
        for(int thread = 0; thread < numThread; thread ++){
            gpp[thread]->setParameters(k, 0, thread);
            if(pthread_create(&_thread[thread], NULL, divisionStepParallel, gpp[thread]) != 0){
                printf("Error to create thread!\n");
            }
        }
        for(int thread = 0; thread < numThread; thread ++){
            pthread_join(_thread[thread], NULL);
        }
        
        gep->y->at(k) = gep->b->at(k) / gep->A->at(k)[k];
        gep->A->at(k)[k] = 1;
        
        for(int i = k+1; i < gep->n; i++){
            /* Elimination Step */
            for(int thread = 0; thread < numThread; thread ++){
                gpp[thread]->setParameters(k, i, thread);
                if(pthread_create(&_thread[thread], NULL, eliminationStepParallel, gpp[thread]) != 0){
                    printf("Error to create thread!\n");
                }
            }
            
            for(int thread = 0; thread < numThread; thread ++){
                pthread_join(_thread[thread], NULL);
            }
       
            gep->b->at(i) = gep->b->at(i) - gep->A->at(i)[k] * gep->y->at(k);
            gep->A->at(i)[k] = 0;
        }
        
    }
    
    delete [] gpp;
}

void* allStepParallel(void *args){
    /* division step */
    GaussianParallelParameters* gpp = (GaussianParallelParameters*) args;
    
     for(int k = gpp->temp1 + gpp->threadIndex; k < gep->n; k+=gep->numThread){
        for(int j = k+1; j < gep->n; j++){
            int re = gep->A->at(k)[j] / gep->A->at(k)[k];
            /* critical section */
//            pthread_mutex_lock(&lock[0]);
            gep->A->at(k)[j] = re;
//            pthread_mutex_unlock(&lock[0]);
        }
    
        gep->y->at(k) = gep->b->at(k) / gep->A->at(k)[k];
        gep->A->at(k)[k] = 1;
    
        for(int i = k+1; i < gep->n; i++){
            /* Elimination Step */
            for(int j = k+1; j < gep->n; j++){
                int re = gep->A->at(i)[j] - gep->A->at(i)[k] * gep->A->at(k)[j];
                /* critical section */
//                pthread_mutex_lock(&lock[1]);
                gep->A->at(i)[j] = re;
//                pthread_mutex_unlock(&lock[1]);
            }
            /* critical section */
            int re = gep->b->at(i) - gep->A->at(i)[k] * gep->y->at(k);
//            pthread_mutex_lock(&lock[2]);
            gep->b->at(i) = re;
            gep->A->at(i)[k] = 0;
//            pthread_mutex_unlock(&lock[2]);
        }
     }
    
    return nullptr;
}


void calculateGaussianElimination_Parallel_OutLoop(){
    int numThread = gep->numThread;
    pthread_t _thread[numThread];
    GaussianParallelParameters** gpp = new GaussianParallelParameters*[numThread];
    for(int i = 0; i < numThread; i++){
        gpp[i] = new GaussianParallelParameters();
    }
    
    for(int i = 0; i < 3; i++){
        if (pthread_mutex_init(&lock[i], NULL) != 0)
        {
            printf("\n mutex init failed\n");
        }
    }
    
    for(int thread = 0; thread < numThread; thread++){
        gpp[thread]->setParameters(0, 0, thread);
        if(pthread_create(&_thread[thread], NULL, allStepParallel, gpp[thread]) != 0){
            printf("Error to create thread!\n");
        }
    }
    
    for(int thread = 0; thread < numThread; thread ++){
        pthread_join(_thread[thread], NULL);
    }
    
    /* destory all locks */
    for(int i = 0 ; i < 3; i++){
        pthread_mutex_destroy(&lock[i]);
    }
    
}

void calculateGaussianElimination_Parallel(int version){
    if(version == 0){
        calculateGaussianElimination_Parallel_InLoop();
    }else{
        calculateGaussianElimination_Parallel_OutLoop();
    }
}

void calculateGaussianElimination_Serial(){
    
    for(int k = 0; k < gep->n; k++){
        /* division step */
        for(int j = k+1; j < gep->n; j++){
            gep->A->at(k)[j] = gep->A->at(k)[j] / gep->A->at(k)[k];
        }
        
        gep->y->at(k) = gep->b->at(k) / gep->A->at(k)[k];
        gep->A->at(k)[k] = 1;
        
        for(int i = k+1; i < gep->n; i++){
            /* Elimination Step */
            for(int j = k+1; j < gep->n; j++){
                gep->A->at(i)[j] = gep->A->at(i)[j] - gep->A->at(i)[k] * gep->A->at(k)[j];
            }
            gep->b->at(i) = gep->b->at(i) - gep->A->at(i)[k] * gep->y->at(k);
            gep->A->at(i)[k] = 0;
        }
        
    }
}

void printAugmentedMatrix(const std::vector<std::vector<double>>*A, const std::vector<double>*b){
    std::cout << "\n*************Print Augmented Matrix*****************\n";
    for(int i = 0;  i < A->size(); i ++){
        for(int j = 0; j < A->at(i).size();  j++){
            std::cout << A->at(i).at(j) << ", ";
        }
        std::cout << b->at(i) << std::endl;
    }
}

void generateRandomMatrix(const int matrixSize, std::vector<std::vector<double>>*A, std::vector<double>* b){
    srand((unsigned)time(0));
    for(int i = 0; i < matrixSize; i++){
        std::vector<double>tmp;
        for(int i = 0; i < matrixSize; i++){
            int random_integer = rand() % 100;
            tmp.push_back(random_integer);
        }
        A->push_back(tmp);
        int random_integer = rand() % 100;
        b->push_back(random_integer);
    }
}

int main(int argc, char** argv){
//    const int matrixSize = atoi(argv[1]);
//    const int numThread = atoi(argv[2]);
//    const int parallel_flag= atoi(argv[3]);
     const int matrixSize = 5;
     const int numThread = 2;
     const int parallel_flag = 1;
    
    bool parallel = false;
    if(parallel_flag == 1 ){
        parallel = true;
    }
    std::vector<std::vector<double>>*A = new std::vector<std::vector<double>>();
   
    std::vector<double>* b = new std::vector<double>();
    std::vector<double>* y = new std::vector<double>(matrixSize);
    generateRandomMatrix(matrixSize, A, b);
    
    gep = new GaussianEliminationParameters(A, b, y, numThread, matrixSize);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    if(parallel){
        calculateGaussianElimination_Parallel(1);
    }else{
        calculateGaussianElimination_Serial();
    }
    gettimeofday(&end, NULL);
    printf("total elapse time is : %ld ms\n", ((end.tv_sec * 1000 + end.tv_usec / 1000)- (start.tv_sec * 1000+ start.tv_usec / 1000)));
    
    printAugmentedMatrix(A, b);
    
    delete A;
    delete b;
    delete y;
    delete gep;
    return 0;
}
