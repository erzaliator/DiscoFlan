#please set the arguments below in the file submission_runner.sh to submit the system 
export DATASET_NAME=zho.rst.sctb
export DATASETFOLDER=/home/VD/kaveri/sharedtask2023/data/
export SAVE_MODEL_DIR=.
export CUDA_DEVICE_ID=3 #cuda device id, supports only single core









# #to the organizers: Please provide us files (predictions.csv  results.txt  task_preds.csv) for ***refine (FlanT5+Ref)*** models after the submission.

# #python script to train the refined  model
# echo "***Refined model for submission***"
# mkdir ./refined_runs/$DATASET_NAME -p
# nohup sh -c 'python runner.py --dataset_folder $DATASETFOLDER --dataset_name $DATASET_NAME --save_model_dir $SAVE_MODEL_DIR/refined_runs --refinement True --cuda $CUDA_DEVICE_ID; python rel_eval.py $DATASETFOLDER$DATASET_NAME"/"$DATASET_NAME"_test.rels" "./refined_runs/"$DATASET_NAME"/task_preds.csv"' > "refined_runs/runtime_"$DATASET_NAME".out" 2> "refined_runs/runtime_"$DATASET_NAME".err" &




# NOTE: refined runs yeilds error using disrpt eval (it is a generative model) so we report numbers using our script with scikitlearn's accuracy_score.
# We will use the unrefined model for the analysis and the refined model for the final submission.
# relevant files are stored in the folder: save_model_dir
# to the organizers: Please provide us runs for ***unrefined(FlanT5)*** models. We will use the unrefined model for the analysis and the refined model for the final submission.
# relavant files are stored in the folder: save_model_dir (predictions.csv  results.txt  task_preds.csv are needed for our analysis)

#python script to train the unrefined model
echo "***Unrefined model not for submission***"
mkdir ./unrefined_runs/$DATASET_NAME -p
nohup python runner.py --dataset_folder $DATASETFOLDER --dataset_name $DATASET_NAME --save_model_dir $SAVE_MODEL_DIR/unrefined_runs --refinement False --cuda $CUDA_DEVICE_ID > "unrefined_runs/runtime_"$DATASET_NAME".out" 2> "unrefined_runs/runtime_"$DATASET_NAME".err" &