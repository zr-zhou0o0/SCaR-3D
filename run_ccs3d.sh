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

    # Render Origin
    python render.py -m ${MODEL_DIR} -s ${DATASET_DIR} -c post --point_cloud_name point_cloud --render_name pre_origin --iteration 30000

    # Get Difference Map
    python change_detection.py -s ${DATASET_DIR} --mask_root ${MASK_DIR} -m ${MODEL_DIR} --config ${CONFIG} -c post --convert_SHs_python --point_cloud_name point_cloud --get_signed_difference --get_test_post --zero_fill
    
    # Get Trace
    python change_detection.py -s ${DATASET_DIR} --mask_root ${MASK_DIR} -m ${MODEL_DIR} --config ${CONFIG} -c post --convert_SHs_python --point_cloud_name point_cloud --iteration ${ITERATION} --get_trace --get_test_post --zero_fill

    # Get Fine Masks
    python change_detection.py -s ${DATASET_DIR} --mask_root ${MASK_DIR} -m ${MODEL_DIR} --config ${CONFIG} -c post --convert_SHs_python --point_cloud_name point_cloud --iteration ${ITERATION} --get_test_post --get_prompt --zero_fill

    # Evaluate
    python scar3d/evaluate.py --mask_folder ${MASK_DIR}/test-post_change_masks --gt_folder ${DATASET_DIR}/gt-post-mask --result_folder ${RESULT_DIR} --config ${CONFIG} --seq post


done











