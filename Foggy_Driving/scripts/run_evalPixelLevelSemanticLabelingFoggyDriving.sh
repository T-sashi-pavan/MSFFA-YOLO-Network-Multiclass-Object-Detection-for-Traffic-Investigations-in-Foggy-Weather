# Template Linux shell script that runs adjusted Cityscapes Python script for evaluation of semantic segmentation results on Foggy Driving.
# Please replace placeholders properly in order to be able to run this script.

export CITYSCAPES_DATASET="<Path to ***Foggy Driving*** root directory>"

export CITYSCAPES_RESULTS="<Path to directory with results of your method for all 101 images of Foggy Driving, saved in Cityscapes labelIds format>"

export EVALUATION_RESULTS_PATH="<Path to directory where evaluation log will be saved as a .json file>"
mkdir -p $EVALUATION_RESULTS_PATH

export EVALUATION_RESULTS_FILE="my_awesome_method.json"

EVALUATION_SCRIPT_PATH="<Path to directory of Cityscapes evaluation scripts>" # Should end with: cityscapesscripts/evaluation

# Please copy the provided Python script evalPixelLevelSemanticLabelingFoggyDriving.py to directory EVALUATION_SCRIPT_PATH in order to be able to run this script.
python ${EVALUATION_SCRIPT_PATH}/evalPixelLevelSemanticLabelingFoggyDriving.py
