# Graph_N_BEATS_Anomaly
## Neuro-inspired AI Lab.
## JuHyeon Kim



N-BEATS base Long Sequence TimeSeries Forecasting models

- N-BETATS
- N-HiTS
- IC-PN-BEATS (Ours)

## Exp Setting

- Use input 120 step and forecast 6(1 hour) or 18(3 hour) step 
- In the case of forecast 36 step, we set the input length 108 steps
- All the hyperparameters is in config's yaml file




# In Details


    src
    ├──  config
    │    └── config.yaml  - the default config file.
    │    └──  N_BEATS.yaml, N_HiTS.yaml, IC_PN_BEATS.yalm    - the specific config file 
    │                                                       for specific model
    │
    ├──  data  
    │    └── check_data.py  		   
    │    └── sea_fog_dataset.py   - file to make dataloader.
    │
    │
    ├──  models
    │   └── graph_learning_Attention    -This folder contains Graph Learning Module 
    |   └── layer                       - This folder contains GLU layer, None Graph Learning Module
    |   └── message_passing             - This folder contains Various Message Passing NN for IC-PN-BEATS
    │   └── Block.py     - this file contains the train loops.
    │   └── IC_PN_BEATS.py   - this file contains the inference process.
    │   └── N_model.py       - this file contains the inference process.
    │
    ├── runner              
    │   └── runner.py       - train loops, inference process
    │
    └── utils
    │    ├── dataset_utils.py
    │    ├── logger.py
    │    ├── result_visualization.py
    │    ├── scalers.py
    │    ├── score.py
    │    ├── train_helper.py
    │    └── utils.py
    │ 
    │ 
    ├── exp_octo.sh         - shell script for train
    ├──  run_dataset.py     - Get Dataset For train, valid, test
    ├──  run_gridsearch_exp.py      - Train multi model for Grid Search 
    └──  run_inference.py           - Only Inference the trained model
