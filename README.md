# DiscoFlan

## Overview: Refined and Unrefined models

This is Disrpt submission for our generative model. 
There are two models:
| DiscoFlan (Unrefined)                         | DiscoFlan+Ref (Refined)                           |
|-----------------------------------------------|---------------------------------------------------|
| X                                             | Our Submission                                    |
| Generates classes outside of submission rules | Generates submission file ("task_preds.csv")      |
| No analysis script in runner file             | Runs task rel_eval.py to obtain official results. |
| Pass --refined False to runner.py             | Pass --refined True to runner.py                  |





a) Refined models will be considered for submission hence the evaluation script is also provided. 

b) Unrefined models will not be considered for submission hence evaluation script is not provided. 


## Scripts and outputs


There are two way to run the model (synchronously or in the background):
1. ***submission_runner.sh*** trains the following models (refer to comments submission_runner.sh for other instructions)
2. ***submission_runner_nohup.sh*** does the same by spawning background processes.


Accuracy scores are automatically generated as output. Generated labels are stored in *predictions.csv* and the best model is saved in *best.pt*


## Instructions

create conda environment
```
cd DiscoFlan
mkdir refined_runs
mkdir unrefined_runs
conda create --name flant5 python=3.10
conda activate flant5
```

Install dependencies:
```
pip install -r requirements.txt
```

Provide the correct arguments by modifying submission_runner.sh and THEN run submission_runner.sh (Set Arguments: $DATASET_NAME, $DATASETFOLDER)
```
bash submission_runner.sh
```



***Misc***
Batchsize can be modified if necessary in runner.py. Cuda device id can be provided in submission_runner.sh.