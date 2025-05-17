# 🥕🥔🍈 ResNet18 - Pretrained vs From Scratch: Vegetable Classification
This deep learning project uses ResNet18, a convolutional neural network, to classify Carrot, Papaya, and Potato images from the Vegetable Image Dataset on Kaggle. The project includes two main approaches:
- ✅ Transfer Learning with Pretrained ResNet18
- 🔨 Training ResNet18 from Scratch
It also includes hyperparameter tuning using Optuna and evaluates model performance using confusion matrix and classification reports.

## 📁 Dataset
### Vegetable Image Dataset
Source: https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset
This dataset includes 15 vegetable categories, but only 3 are used in this project:
- Carrot
- Papaya
- Potato
Each image is 224x224 pixels in JPG format.

## 🛠️ Setup
1. Install Dependencies
    !pip install optuna
2. Mount Google Drive (if using Colab)
    from google.colab import drive
    drive.mount('/content/drive')
3. Extract Dataset
    from zipfile import ZipFile
    with ZipFile('/content/drive/MyDrive/Dataset/Vegetable Images.zip') as zipObj:
        zipObj.extractall()
## 🧪 Experiments
###✅ Transfer Learning with Pretrained ResNet18
- Load a pretrained model using torchvision.models.resnet18(pretrained=True)
- Modify the final fully connected layer to classify 3 classes
- Data augmentation includes rotation, blur, color jitter, etc.
- Optimizers tried: SGD, Adam, Adadelta
- Loss Functions: CrossEntropyLoss, NLLLoss
- Hyperparameter tuning with Optuna

Best Accuracy: 100%
Best Params: Adadelta, CrossEntropyLoss, LR ≈ 0.0028

## 🔨 Training ResNet18 From Scratch
- Custom ResNet18 architecture built with PyTorch
- Same transformations and training logic as the pretrained version
- Tuned with Optuna for optimal performance

Best Accuracy: 100%
Best Params: Varies by trial

## 📊 Evaluation
Confusion Matrix and Classification Report
Both approaches achieved high accuracy on test data (600 images):
- ✅ Pretrained: 100% accuracy after tuning
- 🔨 From Scratch: 100% accuracy after tuning

Evaluation metrics include:
- Precision
- Recall
- F1-Score
- Confusion Matrix Heatmaps

## 📉 Visualization
- 📈 Accuracy and Loss curves over 10 epochs
- 📊 Optuna visualizations:
  - Parallel Coordinate Plot
  - Contour Plot
  - Slice Plot
  - Parameter Importance
  - Optimization History

## 🧪 Tech Stack
- Python
- PyTorch
- Optuna
- Matplotlib & Seaborn
- PIL & torchvision
- Google Colab (recommended for GPU)

📦 Folder Structure
project/
├── Vegetable Images/
│   ├── train/
│   ├── validation/
│   └── test/
├── model_tf_trial_*.pth
├── model_scr_trial_*.pth
├── resnet_tf.pt
├── resnet_sc.pt
├── notebook.ipynb
└── README.md

## 🚀 How to Run
1. Upload the dataset from Kaggle to your Google Drive
2. Mount Drive and extract the ZIP
3. Run the full pipeline from loading data, transforming, training, evaluating
4. Tune hyperparameters with Optuna if desired

## 🔍 Future Improvements
- Add more vegetable classes and retrain
- Use other architectures like EfficientNet or DenseNet
- Convert best model to ONNX or TorchScript for deployment
- Implement a basic UI for prediction

## 🤝 Credits
Developed as part of the Deep Learning coursework to compare pretrained models vs custom CNN architectures.
