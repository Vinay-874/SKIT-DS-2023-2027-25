# System Architecture

## Project

Solar Power Forecasting Using Machine Learning and Deep Learning

## Sprint

22 August 2026 – 15 September 2026

## Member

Member 4 – Full-Stack & Deployment Engineer

---

## 1. Architecture Objective

The proposed architecture will connect the user interface, backend
services and trained Machine Learning/Deep Learning models into an
end-to-end solar power forecasting system.

The architecture is planned so that the forecasting models can be
served through a backend API and accessed through an interactive
Streamlit dashboard.

---

## 2. High-Level Architecture

User
  ↓
Streamlit Dashboard
  ↓
FastAPI Backend
  ↓
Input Validation
  ↓
Pre-processing
  ↓
Trained ML/DL Model
  ↓
Solar Power Forecast
  ↓
FastAPI Response
  ↓
Streamlit Dashboard
  ↓
Visualization

---

## 3. Main Components

### 3.1 User

The user interacts with the forecasting system through the
dashboard and provides the information required for prediction.

### 3.2 Streamlit Dashboard

Streamlit will provide the interactive frontend of the system.

The dashboard will be responsible for:

- Accepting user inputs.
- Providing prediction controls.
- Displaying forecast results.
- Displaying visualizations.
- Presenting model-related results.

### 3.3 FastAPI Backend

FastAPI will act as the backend and model-serving interface.

It will be responsible for:

- Receiving prediction requests.
- Validating incoming data.
- Connecting the dashboard with the model.
- Sending input to the appropriate prediction pipeline.
- Returning the prediction response.

### 3.4 Pre-processing Layer

The preprocessing layer will prepare the input data in the format
required by the selected forecasting model.

The final preprocessing and scaling requirements will be confirmed
with the ML/model development team.

### 3.5 ML/DL Model Layer

The project will use Machine Learning and Deep Learning approaches
for solar power forecasting.

The project form specifies Random Forest, LSTM and GRU as part of
the planned technology/model stack.

### 3.6 Output Layer

The prediction returned by the backend will be displayed through
the Streamlit dashboard using suitable visualizations.

---

## 4. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Programming | Python | Main development language |
| Data Processing | Pandas, NumPy | Data handling and preprocessing |
| ML | Scikit-learn / Random Forest | Machine-learning forecasting |
| Deep Learning | TensorFlow/PyTorch / LSTM / GRU | Time-series forecasting |
| Backend | FastAPI | API and model-serving layer |
| Frontend | Streamlit | Interactive dashboard |

---

## 5. Proposed Data Flow

1. User opens the Streamlit dashboard.
2. User provides the required forecasting inputs.
3. Streamlit prepares the prediction request.
4. FastAPI receives the request.
5. FastAPI validates the input.
6. Required preprocessing is performed.
7. The selected trained model receives the processed input.
8. The model generates the solar power forecast.
9. FastAPI returns the prediction.
10. Streamlit displays the prediction and visualization.

---

## 6. Architecture Responsibilities

### Frontend

- Dashboard interface
- User input
- Prediction controls
- Result display
- Visualization

### Backend

- API endpoints
- Request handling
- Input validation
- Model-serving interface
- Response handling

### Model Layer

- Trained forecasting models
- Prediction pipeline
- Model-specific preprocessing requirements

---

## 7. Integration Plan

The architecture will be implemented in stages.

### Stage 1

Finalize requirements and dashboard design.

### Stage 2

Set up the project structure and development environment.

### Stage 3

Develop the FastAPI backend and API structure.

### Stage 4

Integrate the trained forecasting models.

### Stage 5

Connect Streamlit with the FastAPI backend.

### Stage 6

Perform validation, integration testing and final deployment.

---

## 8. Pending Information from ML Team

Before final model integration, the following information needs to
be confirmed:

- Final input feature list.
- Feature order.
- Target variable.
- Preprocessing/scaling method.
- Model file format.
- Model input shape.
- Prediction output format.
