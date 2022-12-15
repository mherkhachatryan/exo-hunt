# exo-hunt
Exo Planet classification software for course ASDS21 MLOps

# Instructions 
### CLI

To run CLI program follow instructions below:
1. `pip install -r requirements.txt`
2. In `hunter/configs.yaml` change your parameters if neccesary
3. In terminal run `python hunter/main.py`


### API
To run API follow instruction below. 

**Note**

Please note that current API only provides functionality to predict planetary system from test data. 

Hence before running API one should additionally give trained model path and data path from `config.yaml`. 
Note also that one should set `mode` to `test`. 

1. In terminal run `python api/run.py`. 
2. API endpoint is `predict_api`. Call it with this kind of body
    ```{"planetary_system": <int>}``` where `<int>` indicates index number in test set to predict if system 
has exoplanet or not. 