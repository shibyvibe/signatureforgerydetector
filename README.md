**Executive Summary**
Hard documents with wet signatures are still the norm when establishing contract agreements between entities. The signatures of authorizers of any future changes or transactional requests related to these agreements need to be vetted against the wet signatures in the original contract agreement. The vetting processing is currently manual in most organizations and typically manually comparing wet signatures against a signature on file and also to detect forgery. This solution uses AI based image processing capabilities to automate this manual activity.

**Approach**
This app compares a provided signature specimen against a set of 11 genuine signatures on file for a person to determine the probability that the provided specimen is a forgery. The AI model utilizes combination of DNN using a Triplet loss mining training approach and a logistic regression model for forgery decisioning. The model provides predict the probability with a sensitive of 86% and precision of 93%. More details can be found at https://docs.google.com/document/d/1Cquh-PaJi8LOWQ9mbTtfFg4QUjyaYuA-VmfiK3M9ccY/edit?usp=sharing  

**Instructions to access the service via Web UI**

The application is accesible via a Web UI developed in Flask. The application is deployed in Paperspace and the UI takes a person Id (for a person with signature specimens on file) and the new signature specimen to be checked.

Currently unable to forward the port in Paperspace to my local machine and hence the URL can only be access via curl. The curl scripts to test the web ui is in the curl_test directory.

**Trained Model weights and embedding for train and test** can be found in google drive (https://drive.google.com/drive/folders/1u-ZZtJ8doXgf1kLsZdt989t_aFY-Du_w?usp=sharing)

**Setup instructions**

1) Install the dependencies specified in the requirements.txt file
2) Download the code
3) Download the dataset from 
4) Run the training model (train_model_SignatureForgeryDetection_Resnet50.py)
    a) set the path correctly to the dataset location
    b) Set the paths correctly to where the trained model will be saved.
5) Set the path to the trained model correctly in predict_SignatureForgeryDetectorLib.py
6) Set the path to the datastore in SignerVerifier.py
7) Launch the app SignerVerifier.py
8) Access the app from the browser.

**Project Organizatoin**

    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- https://drive.google.com/file/d/1Sid1CBzCGTbHrGsE4vBQoJAPEeNB_4N5/view?usp=sharing
    │
    ├── notebooks          <- https://github.com/shibyvibe/SignatureVerification/blob/main/Signature%20Forgery%20Detection-Triplet%20Loss%20Mining-Resnet50.ipynb (Training)
    │                      <- https://github.com/shibyvibe/SignatureVerification/blob/main/Data%20Exploration.ipynb (Data Exploration)
    │                       
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_SignatureForgeryDetectorLib.py   <-- prediction model
    │   │   └── train_model_SignatureForgeryDetection_Resnet50.py <-- training model
    │   │
    │   └── test           <- Contains the pytest unit test scripts.
    │
    |_  SignerVerifier.py  <- Flask based UI to test signature specimens
    |_  curl_test          <- contain curl scripts to test the SignerVerifier UI
    
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
