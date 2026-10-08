# Model Serving Workflow

## Project

Solar Power Forecasting Using Machine Learning and Deep Learning

## Sprint

22 August 2026 – 15 September 2026

## Member

Member 4 – Full-Stack & Deployment Engineer

---

## 1. Objective

The purpose of the model-serving workflow is to define how the
trained Machine Learning and Deep Learning models will eventually
be connected with the application.

The workflow will provide a connection between the Streamlit
dashboard, FastAPI backend and trained forecasting models.

---

## 2. Proposed Workflow

User
  ↓
Streamlit Dashboard
  ↓
Prediction Request
  ↓
FastAPI Backend
  ↓
Input Validation
  ↓
Pre-processing
  ↓
Model Selection
  ↓
Trained ML/DL Model
  ↓
Prediction
  ↓
FastAPI Response
  ↓
Streamlit Dashboard
  ↓
Result Visualization

---

## 3. Workflow Steps

### Step 1 – User Input

The user will provide the information required by the forecasting
model through the Streamlit dashboard.

### Step 2 – Prediction Request

The Streamlit application will prepare the input data and send a
prediction request to the backend.

### Step 3 – Backend Request Handling

FastAPI will receive the prediction request and process the
incoming data.

### Step 4 – Input Validation

The backend will validate the structure and values of the received
input before sending it to the model pipeline.

### Step 5 – Pre-processing

The input data will be transformed into the format required by the
selected model.

The exact preprocessing and scaling method will be finalized with
the ML model development team.

### Step 6 – Model Selection

The system will use the selected trained forecasting model.

The project includes Machine Learning and Deep Learning approaches,
including Random Forest, LSTM and GRU.

### Step 7 – Model Inference

The processed input will be provided to the trained model and the
model will generate a solar power forecast.

### Step 8 – Backend Response

FastAPI will receive the model output and prepare the response.

### Step 9 – Dashboard Display

The prediction will be returned to the Streamlit dashboard and
displayed to the user.

### Step 10 – Visualization

The dashboard will present the forecasting result through suitable
visualizations.

---

## 4. Backend Responsibilities

The backend/model-serving layer will be responsible for:

- Receiving prediction requests.
- Validating input data.
- Connecting with the model pipeline.
- Handling model inference requests.
- Returning prediction results.
- Handling invalid or unsuccessful requests.

---

## 5. Frontend Responsibilities

The Streamlit dashboard will be responsible for:

- Providing the user interface.
- Collecting user inputs.
- Sending prediction requests.
- Displaying prediction results.
- Displaying forecasting visualizations.

---

## 6. Model Integration Requirements

Before the final implementation, the following information must be
confirmed with the ML development team:

- Final model input features.
- Feature order.
- Target variable.
- Preprocessing method.
- Scaling requirements.
- Model file format.
- Model input shape.
- Prediction output format.

---

## 7. Expected End-to-End Flow

The expected application flow is:

1. User opens the dashboard.
2. User enters the required information.
3. User requests a forecast.
4. Streamlit sends the request to FastAPI.
5. FastAPI validates the request.
6. The preprocessing pipeline prepares the input.
7. The trained model generates the forecast.
8. FastAPI returns the forecast.
9. Streamlit displays the result.
10. The user can view the forecast visualization.

---

## 8. Future Implementation

The workflow defined in this document will be used as the
reference for the later implementation of:

- Project structure.
- FastAPI backend.
- Model integration.
- Dashboard-backend connection.
- Backend and frontend validation.
- Final deployment and testing.