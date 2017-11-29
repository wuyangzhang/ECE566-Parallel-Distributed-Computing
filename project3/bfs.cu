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
                //printf("visit node %d\n", i);
                bfsQueue.push(i);
                while(!bfsQueue.empty()){
                    i = bfsQueue.front();
                    bfsQueue.pop();
                    for(int j = 0; j < verticeNum; j++){
                        if(graphMatrix[i][j] != -1 && !visited[j]){
                            visited[j] = true;
                            //printf("inner visit node %d\n", j);
                            bfsQueue.push(j);
                        }
                    }
                }
            }
        }
    }
    
    void _recursiveBFS(std::queue<int>* bfsQueue, bool visited[]){
        while(!bfsQueue->empty()){
            int i = bfsQueue->front();
            bfsQueue->pop();
            for(int j = 0; j < verticeNum; j++){
                if(graphMatrix[i][j] != -1 && !visited[j]){
                    visited[j] = true;
                    //printf("inner visit node %d\n", j);
                    bfsQueue->push(j);
                }
            }
            _recursiveBFS(bfsQueue, visited);
        }
    }
    
    void recursiveBFS(){
        const int vNum = verticeNum;
        bool visited[vNum];
        for(int i = 0; i < vNum; i++){
            visited[i] = false;
        }
        std::queue<int> bfsQueue;
        for(int i = 0; i < verticeNum; i++){
            if(!visited[i]){
                //printf("visit node %d\n", i);
                bfsQueue.push(i);
                visited[i] = true;
                _recursiveBFS(&bfsQueue, visited);
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

struct Node{
    int data;
    Node* next;
    Node* prev;
    
    Node(int data){
        this->data = data;
    }
    Node(){}
    
}Node;


struct List{
    struct Node* head, *tail;
    int listSize;
    
    List(int vNum){
        head = (struct Node*)malloc(sizeof(Node) * vNum);
        tail = head + 1;
        head->next = tail;
        tail->prev = head;
        listSize = 0;
    }
    
    
    ~List(){
        delete []head;
    }
    
    void push(int data){
        struct Node* node = new struct Node(data);
        if(head->next == tail){
            head->next = node;
            tail->prev = node;
            node->next = tail;
            node->prev = head;
        }else{
            node->prev = tail->prev;
            node->next = tail;
            tail->prev->next = node;
            tail->prev = node;
        }
        
        listSize++;
    }
    
    int pop(){
        assert(!isEmpty());
        struct Node * tmp = head->next;
        tmp->prev->next = tmp->next;
        tmp->next->prev = tmp->prev;
        int n = tmp->data;
        delete tmp;
        listSize--;
        return n;
    }
    
    
     __device__
     void cudaPush(int data){
     struct Node* node = (struct Node*)malloc(sizeof(struct Node));
     node->data = data;
     if(head->next == tail){
     head->next = node;
     tail->prev = node;
     node->next = tail;
     node->prev = head;
     }else{
     node->prev = tail->prev;
     node->next = tail;
     tail->prev->next = node;
     tail->prev = node;
     }
     
     listSize++;
     }
     
     __device__
     int cudaPop(){
     assert(!isEmpty());
     struct Node * tmp = head->next;
     tmp->prev->next = tmp->next;
     tmp->next->prev = tmp->prev;
     int n = tmp->data;
     delete tmp;
     listSize--;
     return n;
     }
     
    __device__
    int cudaGetSize(){
        return listSize;
    }
     
     __device__
     bool cudaIsEmpty(){
     if(head->next == tail){
     return true;
     }else{
     return false;
     }
     }
    
    
    int getSize(){
        return listSize;
    }
    
   
    
    bool isEmpty(){
        if(head->next == tail){
            return true;
        }else{
            return false;
        }
    }
    
    void printList(){
        struct Node* node = head->next;
        while(node != tail){
            //printf("value %d\n", node->data);
            node = node->next;
        }
    }
};

__global__
void cudaComputeBFS(bool* visited, int** graphMatrix, List* bfsQueue, int* i){
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(graphMatrix[*i][x] != -1 && !visited[x]){
        visited[x] = true;
        //printf("inner visit node %d\n", x);
        bfsQueue->cudaPush(x);
    }

}

void cudaBFS(Graph g){
    
    const int vNum = g.verticeNum;
    bool visited[vNum];
    for(int i = 0; i < vNum; i++){
        visited[i] = false;
    }
    
    struct List bfsQueue(vNum);
    
    for(int i = 0; i < vNum; i++){
        //if not visited, we do processing..
        if(!visited[i]){
            //mark the node as visited
            visited[i] = true;
            printf("visit node %d\n", i);
            bfsQueue.push(i);

            while(!bfsQueue.isEmpty()){
                i = bfsQueue.pop();
                //cuda optimization...

                const int BLOCK_SIZE = 64;
                dim3 dimBlock((vNum + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
                dim3 dimGrid(BLOCK_SIZE, 1, 1);

                bool* cudaVisited;
                int** cudaGraphMatrix;
                List* cudaBfsQueue;
                int* cudaI;
                int* cudaSize;
                int* size;
                
                cudaMalloc(&cudaVisited, sizeof(bool) * vNum);
                cudaMalloc(&cudaBfsQueue->head, sizeof(Node) * vNum + 2);
                cudaMalloc(&cudaGraphMatrix, sizeof(int *) * vNum);
                cudaMalloc(&cudaI, sizeof(int*));
                cudaMalloc(&cudaSize, sizeof(int*));
                
                for(int k = 0; k < vNum; k++){
                    cudaMalloc(&cudaGraphMatrix[k], sizeof(int) * vNum);
                }

                //copy data to device
                cudaMemcpy(cudaVisited, visited, sizeof(bool) * vNum, cudaMemcpyHostToDevice);

                int count = 0;
                while(!bfsQueue.isEmpty()){
                    cudaMemcpy(&cudaBfsQueue->head + count++, &bfsQueue.head + count++, sizeof(Node), cudaMemcpyHostToDevice);
                }

                for(int k = 0; k < vNum; k++){
                    cudaMemcpy(cudaGraphMatrix, g.graphMatrix, sizeof(int) * vNum, cudaMemcpyHostToDevice);
                }

                cudaMemcpy(cudaI, &i, sizeof(int), cudaMemcpyHostToDevice);
                cudaComputeBFS<<<dimGrid, dimBlock>>>(cudaVisited, cudaGraphMatrix, cudaBfsQueue, cudaI);

                //update visited & queue...
                cudaMemcpy(visited, cudaVisited, sizeof(bool) * vNum, cudaMemcpyDeviceToHost);

                cudaMemcpy(&bfsQueue.head, &cudaBfsQueue->head, sizeof(Node) * vNum, cudaMemcpyDeviceToHost);

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
    
    Graph graph("./largeG.txt");
    gettimeofday( &tpstart, NULL );
    //graph.BFS();
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("CPU looped version is finished in time %ld ms\n", timeuse);
    
    gettimeofday( &tpstart, NULL );
    //graph.recursiveBFS();
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("CPU recursive version is finished in time %ld ms\n", timeuse);
    
    gettimeofday( &tpstart, NULL );
    //cudaBFS(graph);
    gettimeofday (&tpend, NULL);
    timeuse = 1000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec) / 1000;
    printf("GPU version is finished in time %ld ms\n", timeuse);
    
    return 0;
}
