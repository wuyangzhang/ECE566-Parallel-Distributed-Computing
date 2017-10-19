#include "HashTable.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

using namespace std;

void printAugmentedMatrix(const vector<vector<double>>*A, const vector<double>*b);

typedef struct GaussianEliminationParameters{
    vector<vector<double>>* A;
    vector<double>* b;
    vector<double>* y;
    int n;
    int numThread;

    GaussianEliminationParameters(vector<vector<double>>* A, vector<double>*b, vector<double>*y, const int& numThread, const int & n){
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

void calculateGaussianElimination_Parallel(){
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
//            for(int j = k+1; j < gep->n; j++){
//                gep->A->at(i)[j] = gep->A->at(i)[j] - gep->A->at(i)[k] * gep->A->at(k)[j];
//            }
            gep->b->at(i) = gep->b->at(i) - gep->A->at(i)[k] * gep->y->at(k);
            gep->A->at(i)[k] = 0;
        }

    }
    
    delete [] gpp;
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

void printAugmentedMatrix(const vector<vector<double>>*A, const vector<double>*b){
    cout << "\n*************Print Augmented Matrix*****************\n";
    for(int i = 0;  i < A->size(); i ++){
        for(int j = 0; j < A->at(i).size();  j++){
            cout << A->at(i).at(j) << ", ";
        }
        cout << b->at(i) << endl;
    }
}

void generateRandomMatrix(const int matrixSize, vector<vector<double>>*A, vector<double>* b){
    srand((unsigned)time(0));
    for(int i = 0; i < matrixSize; i++){
        vector<double>tmp;
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
        const int matrixSize = 100;
        const int numThread = 8;
    bool parallel = true;
    vector<vector<double>>*A = new vector<vector<double>>();
//    A->push_back({2,1,-1});
//    A->push_back({-3,-1,2});
//    A->push_back({-2,1,2});
//    vector<double>* b = new vector<double>({8,-11,-3});
    
//    A->push_back({1,3,1});
//    A->push_back({1,1,-1});
//    A->push_back({3,11,5});
//    vector<double>* b = new vector<double>({9,1,35});
    vector<double>* b = new vector<double>();
    vector<double>* y = new vector<double>(matrixSize);
    generateRandomMatrix(matrixSize, A, b);
    
    gep = new GaussianEliminationParameters(A, b, y, numThread, matrixSize);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    if(!parallel){
        calculateGaussianElimination_Parallel();
    }else{
        calculateGaussianElimination_Serial();
    }
    gettimeofday(&end, NULL);
    printf("total elapse time is : %ld\n", ((end.tv_sec * 1000000 + end.tv_usec)- (start.tv_sec * 1000000 + start.tv_usec)));
    
    //printAugmentedMatrix(A, b);

    delete A;
    delete b;
    delete y;
    delete gep;
    return 0;
}
