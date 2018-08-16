Face to BMI: Keras Implementation https://arxiv.org/abs/1703.03156
# Methodology

The system is composed of two stages: (i) Feature extraction, and (ii) Regression model. For feature extraction, we use the weights from VGG 16 (fc16 layer) are extracted for each face image in our training set. For BMI prediction, Support Vector Regression models is used using the feature extracted from previous layers.

Credits - E. Kocabey, M. Camurcu, F. Ofli, Y. Aytar, J. Marin, A. Torralba, and I. Weber. Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media. arXiv:1703.03156 [cs], Mar. 2017. arXiv: 1703.03156
