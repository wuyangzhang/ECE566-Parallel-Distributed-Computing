#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <memory>
#include <fstream>
#include <queue>

class Graph{
    
public:
    Graph(const int verticeNum){
        this->verticeNum = verticeNum;
        
        graphMatrix = new int*[verticeNum];
        for(int i = 0; i < verticeNum; i++){
            graphMatrix[i] = new int [verticeNum];
        }
    }
    
    Graph(std::string fileName){
        std::ifstream file;
        file.open(fileName);
        if(!file){
            printf("Unable to open file!\n");
            exit(1);
        }
        
        file >> verticeNum;
        //find vertice number
        //skip the second line
        int tmp;
        file >> tmp;
        //init matrix...
        graphMatrix = new int*[verticeNum];
        for(int i = 0; i < verticeNum; i++){
            graphMatrix[i] = new int [verticeNum];
        }
        
        for(int i = 0 ; i < verticeNum; i++){
            for(int j =0 ; j < verticeNum; j++){
                graphMatrix[i][j] = -1;
            }
        }
        
        int vA, vB;
        while (!file.eof()) {
            file >> vA >> vB;
            int linkWeight = generateRandomNum(100);
            graphMatrix[vA][vB] = linkWeight;
            graphMatrix[vB][vA] = linkWeight;
        }
        
        file.close();
    }

    
    void printGraphMatrix(){
        for(int i = 0 ; i < verticeNum; i++){
            for(int j =0 ; j < verticeNum; j++){
                printf("%d\t", graphMatrix[i][j]);
            }
            printf("\n");
        }
    }
    
    void BFS(){
        const int vNum = verticeNum;
        bool visited[vNum];
        for(int i = 0; i < vNum; i++){
            visited[i] = false;
        }
        std::queue<int> bfsQueue;
        
        for(int i = 0; i < verticeNum; i++){
            //if not visited, we do processing..
            if(!visited[i]){
                //mark the node as visited
                visited[i] = true;
                printf("visit node %d\n", i);
                bfsQueue.push(i);
                while(!bfsQueue.empty()){
                    i = bfsQueue.front();
                    bfsQueue.pop();
                    for(int j = 0; j < verticeNum; j++){
                        if(graphMatrix[i][j] != -1 && !visited[j]){
                            visited[j] = true;
                            printf("inner visit node %d\n", j);
                            bfsQueue.push(j);
                        }
                    }
                }
            }
        }
    }
    
    void _recursiveBFS(std::queue<int> bfsQueue, bool visited[]){
        while(!bfsQueue.empty()){
            int i = bfsQueue.front();
            bfsQueue.pop();
            for(int j = 0; j < verticeNum; j++){
                if(graphMatrix[i][j] != -1 && !visited[j]){
                    visited[j] = true;
                    printf("inner visit node %d\n", j);
                    bfsQueue.push(j);
                }
            }
            _recursiveBFS(bfsQueue, visited);
        }
    }
    
    void recursiveBFS(){
        printf("\nstart recursive version!\n");
        const int vNum = verticeNum;
        bool visited[vNum];
        for(int i = 0; i < vNum; i++){
            visited[i] = false;
        }
        std::queue<int> bfsQueue;
        for(int i = 0; i < verticeNum; i++){
            if(!visited[i]){
                printf("visit node %d\n", i);
                bfsQueue.push(i);
                visited[i] = true;
                _recursiveBFS(bfsQueue, visited);
            }
        }
    }
    
    int** graphMatrix;
    int verticeNum;

private:
    
    void generateRandomMatrix(int verticeNum){
        for(int i = 0 ; i < verticeNum; i++){
            for(int j =0 ; j < verticeNum; j++){
                graphMatrix[i][j] = generateRandomNum(100);
            }
        }
    }
    
    int generateRandomNum(int max){
        return rand() % 100;
    }
};


__global__
void cudaComputeBFS(bool* visited, int** graphMatrix, std::queue<int>* bfsQueue, int* i){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(graphMatrix[*i][x] != -1 && !visited[x]){
        visited[x] = true;
        printf("inner visit node %d\n", x);
        bfsQueue->push(x);
    }
    
}

void cudaBFS(Graph g){
    
    const int vNum = g.verticeNum;
    bool visited[vNum];
    for(int i = 0; i < vNum; i++){
        visited[i] = false;
    }
    std::queue<int> bfsQueue;
    
    for(int i = 0; i < vNum; i++){
        //if not visited, we do processing..
        if(!visited[i]){
            //mark the node as visited
            visited[i] = true;
            printf("visit node %d\n", i);
            bfsQueue.push(i);
            while(!bfsQueue.empty()){
                i = bfsQueue.front();
                bfsQueue.pop();
                //cuda optimization...
                
                const int BLOCK_SIZE = 64;
                dim3 dimBlock((vNum + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
                dim3 dimGrid(BLOCK_SIZE, 1, 1);

                bool* cudaVisited;
                int** cudaGraphMatrix;
                std::queue<int>* cudaBfsQueue;
		int* cudaI;

                cudaMalloc(&cudaVisited, sizeof(bool) * vNum);
                cudaMalloc(&cudaBfsQueue, sizeof(int) * vNum);
                cudaMalloc(&cudaGraphMatrix, sizeof(int *) * vNum);
		cudaMalloc(&cudaI, sizeof(int));

                for(int k = 0; k < vNum; k++){
                    cudaMalloc(&cudaGraphMatrix[i], sizeof(int) * vNum);
                }
                
                //copy data to device
                cudaMemcpy(cudaVisited, visited, sizeof(bool) * vNum, cudaMemcpyHostToDevice);
                cudaMemcpy(&cudaBfsQueue->front(), &bfsQueue.front(), sizeof(int) * vNum, cudaMemcpyHostToDevice);

                for(int k = 0; k < vNum; k++){
                    cudaMemcpy(cudaGraphMatrix, g.graphMatrix, sizeof(int) * vNum, cudaMemcpyHostToDevice);
                }

                cudaMemcpy(cudaI, &i, sizeof(int), cudaMemcpyHostToDevice);
                cudaComputeBFS<<<dimGrid, dimBlock>>>(cudaVisited, cudaGraphMatrix, cudaBfsQueue, cudaI);
                
                //update visited & queue...
                cudaMemcpy(visited, cudaVisited, sizeof(bool) * vNum, cudaMemcpyDeviceToHost);
                cudaMemcpy(&bfsQueue.front(), &cudaBfsQueue->front(), sizeof(int) * vNum, cudaMemcpyDeviceToHost);
     
		 cudaFree(cudaVisited);
		 cudaFree (cudaGraphMatrix);
		 cudaFree(cudaI);
		 cudaFree(cudaBfsQueue);
              
            }
        }
    }  
    
}


int main(){
    struct timeval tpstart, tpend;
    long timeuse;
    
    Graph graph("/Users/wuyang/Desktop/tinyG.txt");
    gettimeofday( &tpstart, NULL );
    graph.BFS();
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("CPU looped version is finished in time %ld ms\n", timeuse);
    
    gettimeofday( &tpstart, NULL );
    graph.recursiveBFS();
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("CPU recursive version is finished in time %ld ms\n", timeuse);
   
    
    return 0;
}

