

# Book Recommendation-System
> A comprehensive machine learning-based book recommendation system that provides personalized book suggestions using collaborative filtering techniques.

# 🚀 Features
    -  Collaborative Filtering: Uses K-Nearest Neighbors algorithm for book recommendations

    -  Streamlit Web Interface: User-friendly web application for easy interaction

    -  Modular Architecture: Well-structured codebase following ML best practices

    -  Training Pipeline: Automated data processing and model training pipeline

    -  Book Covers Integration: Displays book posters alongside recommendations

    -  Real-time Recommendations: Instant book suggestions based on user selection

# 🏗️ System Architecture
```
```bash
text
books_recommender/
├── components/          # Data processing stages
│     ├──data_ingestion.py
│     ├──data_validation.py
│     ├──data_transformation.py
│     └──model_trainer.py
├── config/             # Configuration management
│     ├── configuration.py
│     └── config.yaml
├── entity/             # Data entities and schemas
│     └── config_entity.py
├── pipeline/           # Training pipeline
│     └── training_pipeline.py
├── utils/              # Utility functions
│     └── util.py
├── exception/          # Exception handling
│     └── exception_handler.py
└── logger/             # Logging configuration
      └── log.py
```


## Workflow 
  ```          
- config.yaml
- entity
- config/configuration.py
- components
- pipeline
- main.py
- app.py
```

# 📋 Prerequisites

 - Python 3.10 or higher
 - pip (Python package manager) 

# 🛠️ Installation & Setup

## How to run?
### STEPS:

* 1 - Clone the repository

```bash
 git clone https://github.com/sdagne/Book_recommendation_system.git
 cd Book_recommendation_system
```
* 2 -  Virtual Environment
```
```bash
 >py -3.10 -m venv .venv

 >.venv\Scripts\activate
 >macOS/Linux:
```
```bash
 >python3.10 -m venv .venv
 >source .venv/bin/activate
```
 * 3 - Install Dependencies
  >pip install -r requirements.txt
* 4 - Install Package in Development Mode
 > pip install -e .

 ```

# 🎯 Usage
## Training the Model
 ### Option 1: Using the training pipeline

 > python main.py

### Option 2: Using the web interface

 -1. Start the Streamlit app:
  > streamlit run app.py
  -2. Click the "Train Recommendation System" button in the web interface

# Getting Recommendations
 - Start the web application:
  > streamlit run app.py
 - In the web interface:
    -   Select a book from the dropdown menu
    -   Click "Show Recommendation" to get similar book suggestions
    -   The system will display 5 recommended books with their covers
# 🔧 Configuration
- The system configuration is managed through config/config.yaml which includes:

    -   Data paths and serialized objects locations
    -   Model training parameters
    -   Preprocessing settings
    -   File paths for book data and images

# 📊 How It Works
###  1. Data Processing Pipeline
    - Data Ingestion: Loads book ratings and metadata

    - Data Validation: Ensures data quality and consistency

    - Data Transformation: Creates user-item interaction matrix

    - Model Training: Trains KNN model on book similarities

### 2. Recommendation Engine
    - Uses collaborative filtering with K-Nearest Neighbors

    - Finds books similar to the user's selection based on rating patterns

    - Returns top 5 recommendations with book covers

### 3. Web Interface
    - Built with Streamlit for easy interaction

    - Displays book recommendations in a clean grid layout


```
🗂️ Project Structure

```bash
.
├── books_recommender/          # Main package
│   ├── components/             # Pipeline components
│   ├── config/                 # Configuration management
│   ├── entity/                 # Data entities
│   ├── exception/              # Error handling
│   ├── logger/                 # Logging setup
│   ├── pipeline/               # Training pipeline
│   └── utils/                  # Utility functions
├── templates/                  # Template files
├── app.py                      # Streamlit application
├── main.py                     # Training script
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # Project documentation
```
# 🐳 Docker Support
 - The project includes Docker configuration for containerized deployment:

    - Dockerfile for building the application image

    - .dockerignore to exclude unnecessary files


   #  📈 Model Details
 - Algorithm: K-Nearest Neighbors (KNN)

 - Similarity Metric: Cosine similarity

 - Features: User-book rating patterns

 - Output: Top 5 most similar books

# 🎨 Interface Preview
- The web application features:

    - Clean, modern header with book emoji

    - Training initiation button

    - Book selection dropdown with search functionality

    - 5-column grid layout for recommendations

    - Book covers and titles display

# 👤 Author
Shewan Dagne
| Workplace Engineer | IT Systems | Automation & Scripting | RAG | Machine Learning

 - 📧 Email: shewan.dagne1@gmail.com
 - 🌐 GitHub: @sdagne
 - 💼 LinkedIn: Shewan Dagne

 - 📄 License
 - This project is licensed under the MIT License - see the LICENSE file for details.

📅 Last Updated
October 2025


```bash
## Now run
streamlit run app.py
```


