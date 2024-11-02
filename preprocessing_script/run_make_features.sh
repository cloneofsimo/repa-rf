NUM_GPUS=7
START_INDEX=1
END_INDEX=7
NUM_CPUS=$(nproc)  # Get the number of available CPUs
CORES_PER_PROCESS=24
echo "Number of GPUs: $NUM_GPUS | Number of CPUs: $NUM_CPUS | Cores per process: $CORES_PER_PROCESS"

for ((i=START_INDEX; i<=END_INDEX; i++)); do

    GPU_INDEX=$((i % NUM_GPUS))
    CPU_START=$(( (i * CORES_PER_PROCESS) % NUM_CPUS ))
    CPU_END=$(( CPU_START + CORES_PER_PROCESS - 1 ))

    export CUDA_VISIBLE_DEVICES=$GPU_INDEX
    taskset -c $CPU_START-$CPU_END python make_features.py --device cuda --file_index $i &
    echo "Started process $i on GPU $GPU_INDEX with CPU $CPU_START-$CPU_END"
done

wait  # Wait for all background processes to finish
