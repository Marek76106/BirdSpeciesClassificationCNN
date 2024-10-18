This repository contains a project developed as part of a Machine Learning course at the University of Osijek, focusing on bird species classification using convolutional neural networks. The goal is to train a model to classify 25 species of birds from the "Indian-Birds-Species-Image-Classification" dataset. The dataset consists of 37,000 images, with 30,000 used for training and 7,500 for validation. 


 Key Features:

- Architecture: The model uses a custom-built convolutional neural network (CNN) with multiple Conv2D, MaxPooling2D, SpatialDropout2D, and GlobalMaxPooling2D layers to prevent overfitting and enhance feature extraction.

- Dataset: The dataset includes images of 25 bird species found in India. Images are preprocessed into 100x100 resolution for input, and batches of 64 images are used for training and validation.

- Training Process:
  - Uses TensorFlow's `image_dataset_from_directory` for efficient dataset loading.
  - Dropout layers are integrated to mitigate overfitting during training.
  - Real-time metrics tracking using TensorBoard.
  - Early stopping mechanism based on validation accuracy to avoid unnecessary training after performance stabilizes.

- Optimizations:
  - The model is trained using the Adam optimizer with a learning rate of 0.001.
  - Additional training techniques like early stopping and image augmentation have been applied.
  

Performance:

Initial tests on the validation set achieved up to 85% accuracy with a resolution of 100x100 and optimized training parameters. Adjustments such as increased epochs, dropout layers, and higher input resolution significantly improved the model's ability to generalize across different bird species.


How to Run:

1. Organize the dataset into training and validation directories (`Birds_25/train`, `Birds_25/valid`), with subdirectories for each bird species.
2. Run the script to train the model. It uses TensorBoard for real-time metric tracking and EarlyStopping to prevent overfitting.
3. The model can be evaluated on both the training and validation sets after training.
