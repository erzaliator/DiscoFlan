This is Disrpt submission for our generative model. 
There are two models:
| DiscoFlan (Unrefined)                         | DiscoFlan+Ref (Refined)                           |
|-----------------------------------------------|---------------------------------------------------|
| X                                             | Our Submission                                    |
| Generates classes outside of submission rules | Generates submission file ("task_preds.csv")      |
| No analysis script in runner file             | Runs task rel_eval.py to obtain official results. |
| Pass --refined False to runner.py             | Pass --refined True to runner.py                  |






Submission for refined and unrefined models. FlanT5+Ref is refined. FlanT5 is unrefined. 
a) Refined models will be considered for submission hence the evaluation script is also provided. 
b) Unrefined models will not be considered for submission hence evaluation script is not provided. 

We want to conduct analysis of models for camera ready submission hence we request organised to kindly mail us the following files:
All csv and tsv files created in .refined/ and .unrefined/ Thank you.


***submission_runner.py*** trains the following models (refer to comments submission_runner.py for other instructions)


*Instructions*

create conda environment
```conda create --name flant5 python=3.10
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


**Email for contact**
anuranjana25@gmail.com
kaveri@coli.uni-saarland.de


conda create --name flant5_v2 python=3.10
conda activate flant5_v2