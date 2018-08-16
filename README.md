Face to BMI: Keras Implementation https://arxiv.org/abs/1703.03156
# Methodology

The system is composed of two stages: (i) Feature extraction, and (ii) Regression model. For feature extraction, 
. Both of these models are deep convolutional models
with millions of parameters, and trained on millions of
images. The features from the fc6 layer are extracted for
each face image in our training set. For the BMI regression,
we use epsilon support vector regression models (Smola and
Vapnik 1997) due to its robust generalization behavior
