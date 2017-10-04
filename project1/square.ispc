
static const float error = 0.00001;
static const uniform int taskNum = 128;

export void calRoot_ispc(uniform double times, uniform float values[], uniform float result[], uniform float firstGuess){

    foreach(i = 0 ... times){
        
        double x1 = -1.0;
        double x0 = -1.0;

        x0 = firstGuess;
        do{
            x1 = (x0 - (((x0 * x0) - values[i]) / (2 * x0)));
            x0 = x1;
        }while((x1 * x1) - values[i] > 1e-4);

        result[i] = x1;
    }
   
}

task void calRoot_ispc_task(uniform double times, uniform float values[], uniform float result[], uniform float firstGuess, uniform int range){
    uniform int start = taskIndex * range;
    uniform int end = min((int)times, start + range);

    foreach( i = start ... end){
        double x1 = -1.0;
        double x0 = -1.0;

        x0 = firstGuess;
        do{
            x1 = (x0 - (((x0 * x0) - values[i]) / (2 * x0)));
            x0 = x1;
        }while((x1 * x1) - values[i] > 1e-4);

        result[i] = x1;
    }
}

export void calRoot_ispc_task_exe(uniform double times, uniform float values[], uniform float result[], uniform float firstGuess){
    uniform int range = (int) times / taskNum;
    launch[times/range] calRoot_ispc_task(times, values, result, firstGuess, range);
}