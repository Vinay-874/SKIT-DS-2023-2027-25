

## 1. Dashboard Objective

The objective of the dashboard is to provide an interactive
interface through which users can access the solar power forecasting
system and view forecasting results.

The dashboard is planned using Streamlit.

---

## 2. Proposed Dashboard Structure

The dashboard will contain the following major sections:

1. Project Header
2. Input Parameters
3. Model Selection
4. Prediction Control
5. Forecast Result
6. Forecast Visualization
7. Model Performance / Comparison
8. System Status

---

## 3. Dashboard Layout

### Section 1 – Project Header

The header will display:

- Project title
- Short description of the forecasting system
- Current system/model status

---

### Section 2 – Input Parameters

The dashboard will provide input controls for the parameters
required by the final forecasting model.

The final input feature list and feature order will be confirmed
with the ML/model development team.

---

### Section 3 – Model Selection

The dashboard may provide a model selection option for the trained
forecasting models.

Planned model approaches include:

- Random Forest
- LSTM
- GRU

The final selection mechanism will depend on the model integration
requirements.

---

### Section 4 – Prediction Control

A prediction/forecast button will allow the user to submit the
entered information to the forecasting system.

Proposed flow:

Dashboard Input
↓
Prediction Request
↓
FastAPI Backend
↓
Model
↓
Prediction Result

---

### Section 5 – Forecast Result

The dashboard will display the generated solar power forecast in a
clear and prominent format.

Possible information includes:

- Predicted value
- Selected model
- Prediction status

---

### Section 6 – Forecast Visualization

The dashboard will provide visual representations of forecasting
results.

Planned visualizations may include:

- Forecast trend
- Actual vs predicted values
- Model comparison

The final charts will depend on the data and model outputs available
during implementation.

---

### Section 7 – Model Performance

The dashboard may display relevant model performance information
to help compare the forecasting approaches.

Possible metrics will be finalized with the ML team.

---

### Section 8 – System Status

The dashboard should provide a simple indication of whether the
backend/model service is available.

Example:

System Status: Available

or

System Status: Service Unavailable

---

## 4. Proposed User Journey

1. User opens the dashboard.
2. User reviews the project information.
3. User provides the required forecasting inputs.
4. User selects the required model, where applicable.
5. User clicks the prediction button.
6. Dashboard sends the request to the backend.
7. Backend processes the request and obtains the model prediction.
8. Dashboard receives the response.
9. Forecast result is displayed.
10. Supporting visualizations are shown.

---

## 5. Frontend-Backend Interaction

The planned interaction is:

Streamlit Dashboard
↓
FastAPI API
↓
Validation
↓
Pre-processing
↓
ML/DL Model
↓
Prediction Response
↓
Streamlit Dashboard

---

## 6. Design Principles

The dashboard should be:

- Simple to understand.
- Easy to navigate.
- Clear in presenting the forecast.
- Suitable for model comparison.
- Responsive to user actions.
- Consistent with the forecasting workflow.

---

## 7. Future Implementation

The dashboard design in this document will be used as the
reference for the later Streamlit implementation.

The final input fields, model options, metrics and visualizations
will be finalized after coordination with the ML/model development
team.