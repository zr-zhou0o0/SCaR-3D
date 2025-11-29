# For CCS3D dataset
DATASETS=("bedroom" "bookcase" "desk" "livingroom")
DATASET_ROOT="dataset/CCS3D" 
OUTPUT_ROOT="outputs/CCS3D" 
CONFIG_ROOT="configs/CCS3D"
MODEL_ROOT="models/CCS3D"

for i in "${!DATASETS[@]}"; do
    echo "Dataset name: ${DATASETS[$i]}"

    DATASET_NAME="${DATASETS[$i]}"
    DATASET_DIR="${DATASET_ROOT}/${DATASET_NAME}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}"
    MASK_DIR="${OUTPUT_DIR}/masks"
    DIFMAP="${MASK_DIR}/difmap"
    RESULT_DIR="${OUTPUT_DIR}/results"
    CONFIG="${CONFIG_ROOT}/${DATASET_NAME}.yaml"
    MODEL_DIR="${MODEL_ROOT}/${DATASET_NAME}"

    if [ "$DATASET_NAME" = "bedroom" ]; then
        ITERATION=35000
    else
        ITERATION=40000
    fi

    python train.py -s ${DATASET_DIR} -m ${MODEL_DIR} -c pre --iteration 30000 --save_iterations 30000
    python train.py -s ${DATASET_DIR} -m ${MODEL_DIR} -c post --start_checkpoint 30000 --iteration ${ITERATION}

done


DATASETS=("Bench" "Desk" "Sill" "Swap" "Mustard")
DATASET_ROOT="dataset/3DGS-CD" 
OUTPUT_ROOT="outputs/3DGS-CD" 
CONFIG_ROOT="configs/3DGS-CD"
MODEL_ROOT="models/3DGS-CD"


for i in "${!DATASETS[@]}"; do
    echo "Dataset name: ${DATASETS[$i]}"

    DATASET_NAME="${DATASETS[$i]}"
    DATASET_DIR="${DATASET_ROOT}/${DATASET_NAME}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}"
    MASK_DIR="${OUTPUT_DIR}/masks"
    DIFMAP="${MASK_DIR}/difmap"
    RESULT_DIR="${OUTPUT_DIR}/results"
    CONFIG="${CONFIG_ROOT}/${DATASET_NAME}.yaml"
    MODEL_DIR="${MODEL_ROOT}/${DATASET_NAME}"

    if [ "$DATASET_NAME" = "Bench" ] ; then
        ITERATION=30000
    else
        ITERATION=40000
    fi

    python train.py -s ${DATASET_DIR} -m ${MODEL_DIR} -c pre --iteration 30000 --save_iterations 30000
    python train.py -s ${DATASET_DIR} -m ${MODEL_DIR} -c post --start_checkpoint 30000 --iteration ${ITERATION}


done


