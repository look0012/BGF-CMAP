# BGF-CMAP
Model operation process:
step 1:  Data preprocessing
    (1) 5fold_CV:  Three independent data sets were processed with the idea of 50-fold cross-validation
        1) CMI-9859: 80% training set, 10% test set, 10% verification set.
        2) CMI-9905: 80% training set, 10% test set, 10% verification set.
        3) CMI-20208:80% training set, 10% test set, 10% verification set.

step 2:  Feature extraction
    (1) attribute:  Attribute feature extraction
        Word2vector--->CMI-9859/CMI-9905/CMI-20208: Use word2vec's cbow to get 64-dimensional word embeddings
    (2) behavior:   Behavior feature extraction
        1) behavior-9859
            OpenNE-master: See the file README.md and requirements.txt for usage
            Behavior.py:   The behavior feature vector is obtained.
        2) behavior-9902
            See above for usage.
        3) behavior-20208
            See above for usage.

step 3:  Model training and prediction
    (1)CLF: Training and prediction of BGF-CMAP model
        1) 0/1/2/3/4:
            The prediction of the training of each model
        2) AdaB_draw/GBDT_draw/KNN_draw/LR_draw/RF_draw/SVM_draw:
            Draw the ROC curve and PR curve of the relevant model
        3) CaseStudy
            Experiments to verify whether the model is good or bad
    (2)GF_CLF:   Training and prediction of BGF-CMAP-GF model
        1) 0/1/2/3/4:
            The prediction of the training of each model
        2) GBDT_draw:
            Draw the ROC curve and PR curve of the BGF-CMAP-GF model
    (3)LAP_CLF:  Training and prediction of LAP model
         1) 0/1/2/3/4:
            The prediction of the training of each model
         2) GBDT_draw:
            Draw the ROC curve and PR curve of the LAP model


requirements:
astunparse             1.6.3
cached-property        1.5.2
cachetools             4.2.2
certifi                2021.5.30
chardet                4.0.0
cycler                 0.10.0
dataclasses            0.8
decorator              4.4.2
gast                   0.3.3
gensim                 3.6.0
google-auth            1.31.0
google-auth-oauthlib   0.4.4
google-pasta           0.2.0
grpcio                 1.38.0
h5py                   2.10.0
idna                   2.10
importlib-metadata     4.5.0
joblib                 1.0.1
Keras                  2.4.3
Keras-Preprocessing    1.1.2
kiwisolver             1.3.1
Markdown               3.3.4
matplotlib             3.3.4
networkx               2.5.1
numpy                  1.19.5
oauthlib               3.1.1
openne                 0.0.0
opt-einsum             3.3.0
pandas                 1.1.5
Pillow                 8.2.0
pip                    21.1.2
protobuf               3.17.3
pyasn1                 0.4.8
pyasn1-modules         0.2.8
pyparsing              2.4.7
python-dateutil        2.8.1
pytz                   2021.1
PyYAML                 5.4.1
requests               2.25.1
requests-oauthlib      1.3.0
rsa                    4.7.2
scikit-learn           0.24.2
scipy                  1.4.1
setuptools             52.0.0.post20210125
six                    1.16.0
sklearn                0.0
smart-open             5.1.0
tensorboard            2.2.2
tensorboard-plugin-wit 1.8.0
tensorflow             2.2.0
tensorflow-estimator   2.2.0
termcolor              1.1.0
threadpoolctl          2.1.0
typing-extensions      3.10.0.0
urllib3                1.26.5
Werkzeug               2.0.1
wheel                  0.36.2
wincertstore           0.2
wrapt                  1.12.1
zipp                   3.4.1
