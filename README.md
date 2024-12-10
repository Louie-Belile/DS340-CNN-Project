### README for DS340-CNN-Project

# DS340 CNN Project: Waste Classification Using Convolutional Neural Networks (WasteWise AI)

This project focuses on building and fine-tuning a Convolutional Neural Network (CNN) to classify waste into three categories: **Food Waste**, **Mixed Recycling**, and **Trash**. The ultimate goal is to improve waste sorting systems and promote sustainability by leveraging machine learning. The repository includes the base model, experiments with data augmentation, fine-tuning techniques, and a prototype design for an intelligent trash disposal station.

## Project Structure
```
DS340-CNN-Project/
│
├── waste-images/                           # Directory for training, validation, and test data
│   ├── train/                     # Training dataset
│   ├── val/                       # Validation dataset
│   └── test/                      # Testing dataset
│
│
├── notebooks/                     # Jupyter notebooks
│   └── GAN.ipynb                  # Exploratory Data Analysis
│   └── Experiments.ipynb    # Model building and fine-tuning
│   └── synthetic_data.ipynb
├── model_predictions.html         # Interactive HTML report of predictions
└── README.md                      # Project description and instructions
```

## How to Run the Model

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.7 or later
- TensorFlow 2.x
- Jupyter Notebook (optional, for running `.ipynb` files)
- Required Python libraries (install using `requirements.txt`)

### Steps to Run

1. **Clone the Repository**
   ```
   git clone https://github.com/Louie-Belile/DS340-CNN-Project.git
   cd DS340-CNN-Project
   ```

2. **Prepare the Data**
   Ensure the dataset is structured as follows:
   ```
   data/
   ├── train/
   │   ├── food_waste/
   │   ├── mixed_recycling/
   │   └── trash/
   ├── val/
   │   ├── food_waste/
   │   ├── mixed_recycling/
   │   └── trash/
   └── test/
       ├── food_waste/
       ├── mixed_recycling/
       └── trash/
   ```

3. **Run the Base Model**
   Use the `main.py` script to train and evaluate the base model:
   ```
   python main.py
   ```

4. **Explore Fine-Tuning Experiments**
   Open the `model_fine_tuning.ipynb` notebook in Jupyter Notebook to explore experiments on hyperparameter tuning, data augmentation, and architecture adjustments:
   ```
   jupyter notebook notebooks/model_fine_tuning.ipynb
   ```

5. **Evaluate the Model**
   Evaluate the trained models on the test set using the provided scripts or the interactive `model_predictions.html` file.

6. **View Results**
   Visualizations and metrics such as confusion matrices, training history, and predictions can be found in the `results/` folder.

### Key Features
- **Base Model**: A CNN trained with three convolutional layers and dropout regularization.
- **Data Augmentation**: Techniques like rotation, zoom, and brightness adjustment to improve generalization.
- **Fine-Tuning**: Experiments with dropout rates, learning rates, and model architecture adjustments.
- **Interactive Results**: View predictions and their confidence levels in the `model_predictions.html` file.
- **Prototype Design**: Intelligent trash disposal system prototype included.

### Notes
- The model achieves promising classification results, particularly in the underrepresented category of food waste, demonstrating the importance of addressing data imbalance.
- Fine-tuning experiments highlight potential avenues for improving performance in future iterations.
