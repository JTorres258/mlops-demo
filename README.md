# 🐶 MLOps Demo – Dog Breed Classification (TensorFlow + FastAPI + Docker + MLFlow with Optuna)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![CI](https://github.com/JTorres258/mlops-demo/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/JTorres258/mlops-demo/actions/workflows/ci_pipeline.yml)


This project illustrates a full MLOps workflow for a deep-learning image‐classification system targeting dog breeds. Key components include:

- Training with TensorFlow 2.20 (GPU enabled)  
- Serving via FastAPI in a lightweight docker container  
- Experiment tracking using MLflow  
- Hyperparameter optimization with Optuna  
- Clean separation between training/dev environment and production inference image  

---

## 🚀 Features

- GPU-accelerated training (via Dev Container)  
- Experiment tracking (MLflow)  
- Hyperparameter tuning (Optuna)  
- Evaluation metrics including accuracy, Matthews correlation coefficient (MCC), ROC & PR curves, confusion matrix, per-class metrics  
- Production-ready API for inference  
- Dockerized architecture: one image for training/dev, one for serving  

---

## 📁 Project Structure

```bash
mlops-demo/
├── app/
│ ├── api/ # FastAPI inference application
│ └── train/ # Training, evaluation, dataset loading
│ ├── models/ # Saved trained models (.keras files)
│ └── configs/ # Training configs (YAML)
├── examples/ # Images with some dogs
├── learning-examples/ # Some basic examples of MLFlow, FastAPI, and Docker
├── tests/ # Simple tests scripts
├── .devcontainer/ # VS Code dev container setup for GPU training
├── Dockerfile.train # GPU training / dev image
├── Dockerfile.api # FastAPI runtime image
├── requirements_train.txt
├── requirements_api.txt
├── Makefile
├── LICENSE
└── README.md

```

## 🧠 Two-Environment Architecture

This project uses **two separate Docker images**, each designed for a different purpose:

| Purpose | Dockerfile | Runs In | Used For |
|--------|-------------|---------|----------|
| **Training / Development** | `Dockerfile.train` | VS Code Dev Container | Training, evaluation, MLflow, Optuna |
| **API / Serving** | `Dockerfile.api` | Standalone Docker Container | FastAPI inference server |

```bash
┌────────────────────────────────────────────────────────────┐
│                 Your Windows VS Code                       │
│                (edit files normally)                       │
│                                                            │
│   ┌────────────────────────────────────────────────────┐   │
│   │     Training Devcontainer (TF GPU)                 │   │
│   │  - Controlled by .devcontainer/devcontainer.json   │   │
│   │  - Uses Dockerfile.train                           │   │
│   │  - Where your training happens                     │   │
│   └────────────────────────────────────────────────────┘   │
│                                                            │
│   ┌────────────────────────────────────────────────────┐   │
│   │      API Container (FastAPI runtime)               │   │
│   │  - Controlled by Dockerfile.api                    │   │
│   │  - NOT a devcontainer                              │   │
│   │  - You run it with docker run / compose            │   │
│   └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘

```

---

## Dataset

The model is trained on the **Stanford Dogs Dataset**, which contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. There are 20,580 images, out of which 12,000 are used for training and 8580 for testing. Class labels and bounding box annotations are provided for all the 12,000 images.

**Source: https://www.tensorflow.org/datasets/catalog/stanford_dogs**

## Setup & Usage

### 🏋️‍♂️ Training & Development Environment (VS Code Dev Container)

#### 🎯 Goal
Develop, train, and evaluate models in a **reproducible GPU-enabled environment**.

#### 📦 Files

 `Dockerfile.train`

Defines a training environment for your machine-learning workflow. It's purpose is to build a container image that includes everything required to train your model reproducibly and consistently.

- Creates a GPU-enabled environment (TensorFlow or PyTorch + CUDA/cuDNN) so training can run efficiently.

- Defines the default working directory for training.

- Installs all training dependencies, such as:

    - MLFlow
    - Optuna
    - Data processing libraries


```bash
FROM tensorflow/tensorflow:2.20.0-gpu
WORKDIR /workspace

COPY requirements_train.txt .
RUN python3 -m pip install --upgrade pip && \
    pip install --ignore-installed -r requirements_train.txt

CMD ["bash"]
```

`requirements_train.txt`

Includes the Python packages required for training the model.

`.devcontainer/devcontainer.json`

Tells VS Code to open the project inside the GPU training container.

#### ▶ How to use (VS Code)

1. Open the project in VS Code.
2. Press: **F1 → “Dev Containers: Reopen in Container”**. This step requires Docker Desktop/Engine to be previously installed, and to have the Dev Container extension added to your VS Code.
3. VS Code builds and enters the GPU container.
4. Run training:

```bash
python -m app.train.main
```


5. Run evaluation:

```bash
python -m app.train.evaluate
```

---

### 🚀 API / Model Serving (FastAPI)

#### 🎯 Goal
Serve predictions through a lightweight, production-ready FastAPI container.

#### 📦 Files

`Dockerfile.api`

Defines the runtime (inference) environment for serving your trained machine-learning model through an API. Its purpose is to build a lightweight, production-ready container specifically for model deployment, not training.

- Creates a minimal environment focused on inference rather than training
(small base image, CPU-optimized libraries, no heavy training dependencies).

- Sets a working directory and environment variables for running the API.

- Installs only the packages needed to serve the model, such as:

    - FastAPI / Uvicorn
    - TensorFlow (CPU or Lite) or the model runtime you use
    - Pydantic, image processing utilities, etc.

- Copies the API application code (routes, model-loading logic, handlers).

- Copies the trained model artifacts required for inference.

- Defines the container entrypoint to start the API server:

```bash
FROM python:3.11-slim
WORKDIR /app

COPY requirements_api.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements_api.txt

COPY app ./app
COPY models ./models
COPY configs ./configs

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] # "--reload" and use "app.api.main:app"
```

`requirements_api.txt`

Includes the Python packages required for deploying the model.

#### ▶ Build the API image

````bash
docker build -t mlops-demo-api -f Dockerfile.api .
````
#### ▶ Run the API service

````bash
docker run --rm -p 8000:8000 mlops-demo-api
````
Then open:

- http://localhost:8000

#### 🔥 Hot reload (API development)

When working on the FastAPI application, you’ll often make changes to the code (endpoints, model loading, request/response logic, etc.).
If you run the API in a normal Docker container, you must rebuild the image every time you change a Python file:

```bash
docker build ...
docker run ...
```

This is slow and interrupts your development flow.

Hot reload solves this by:

- letting you modify your API code on your machine,
- instantly reflecting the changes inside the container,
- automatically restarting the server (via Uvicorn’s reload mode),
- and skipping the need to rebuild the image every time.

This makes development much faster and smoother.

To use hot reload, add the `--reload` flag when defining the container entrypoint to start the API server, and run the API container with your local code mounted into the container:

````bash
docker run --rm -p 8000:8000 -v %cd%:/app mlops-demo-api # Or "${PWD}:/app" for Windows
````

**IMPORTANT**

When you use ``--reload``, Uvicorn watches your local folder and runs the app from there instead of using the files inside the Docker image. This makes Python import modules from a different path.

That is why we need to add the `-v` flag (bind mount a volume) when running the docker. Paths inside the code are also affected, e.g., loading the model or the label text file.

---

## 📊 Evaluation $ Metrics

After running training/evaluation, you’ll find outputs such as:

- ConfusionMatrix.png  
- ROC.png  
- PrecisionRecall.png  
- classification_report.txt  
- per_class_metrics.json 

You can also launch the MLflow UI to explore experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

# 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

