# Smart Healthcare Clinic Management System

A web-based healthcare clinic management system with **machine learning-powered consultation demand forecasting** for healthcare resource planning.

Developed as a thesis project for **Accudetek Health Diagnostics**, the system combines clinic data management with a Random Forest forecasting model to help identify consultation demand trends across different patient groups and diagnoses.

> **Thesis Focus:** Machine Learning-Based Consultation Demand Forecasting for Healthcare Resource Planning

---

## Overview

The Smart Healthcare Clinic Management System is designed to support healthcare staff in managing consultation records while providing data-driven insights into future consultation demand.

The system uses historical consultation data to generate monthly demand forecasts based on:

* Diagnosis
* Age group
* Gender
* Monthly consultation patterns
* Historical consultation volume
* Seasonal and time-based trends

The forecasting component is intended to support **planning and resource allocation**, rather than replace clinical decision-making.

---

## Key Features

### Consultation Management

* Record and manage consultation information
* Track diagnosis, demographic information, service type, and consultation dates
* Search and review consultation records
* Organize historical consultation data for analytics

### Patient Management

* Patient account registration and authentication
* Patient profile management
* Consultation history
* Health information management
* Secure email verification

### Appointment Management

* Online appointment requests
* Appointment status tracking
* Staff confirmation and cancellation
* Appointment history
* Reminder notifications

### Analytics Dashboard

* Consultation volume overview
* Monthly consultation trends
* Diagnosis distribution
* Demographic breakdowns
* Forecast visualization
* Resource planning insights

### Machine Learning Forecasting

The system uses **Random Forest Regression** to forecast monthly consultation demand.

The model considers:

* Diagnosis
* Age group
* Gender
* Month
* Seasonal patterns
* Previous consultation volumes
* Lagged consultation values
* Rolling statistics
* Historical trends

The forecasting output is intended to help clinic administrators understand potential changes in consultation demand and prepare resources accordingly.

---

## Machine Learning Approach

### Algorithm

**Random Forest Regression**

The forecasting pipeline transforms consultation records into monthly time-series observations at the:

```text
Diagnosis × Age Group × Gender × Month
```

level.

Temporal features are then generated from the resulting monthly series.

### Features

The model can use:

* Monthly seasonality
* Time index
* 1-month lag
* 2-month lag
* 3-month lag
* 6-month lag
* 12-month lag
* Rolling averages
* Rolling standard deviation
* Recent trends
* Historical seasonal patterns

### Validation

The forecasting system uses time-aware validation rather than randomly mixing historical and future observations.

Evaluation includes:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

A naive previous-period forecasting baseline is also used for comparison.

---

## Dataset

The project uses consultation data structured according to the consultation-record format provided for the thesis project.

The dataset contains historical consultation records covering:

```text
January 2023 – August 2026
```

For model development, historical data is separated into training and evaluation periods to preserve the chronological nature of forecasting.

The project does **not** represent the data as a real-time clinical dataset or as a complete medical record of all Accudetek patients.

---

## System Architecture

The application follows a layered architecture:

```text
┌─────────────────────────────────────┐
│           Web Interface             │
│  Staff Portal / Patient Portal      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│          Flask Application           │
│ Authentication • Records • Reports  │
│ Appointments • Forecasting • Email  │
└──────────────────┬──────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
┌──────────────────┐ ┌──────────────────┐
│ Clinic Database  │ │ ML Forecasting   │
│                  │ │                  │
│ Users            │ │ Feature          │
│ Consultations    │ │ Engineering      │
│ Appointments     │ │ Random Forest    │
│ Staff            │ │ Prediction       │
│ Services         │ │ Evaluation       │
└──────────────────┘ └──────────────────┘
```

---

## Technology Stack

### Backend

* Python
* Flask
* Flask-SQLAlchemy
* SQLAlchemy
* python-dotenv

### Machine Learning

* scikit-learn
* pandas
* NumPy
* joblib

### Frontend

* HTML5
* CSS3
* JavaScript
* Bootstrap
* Tailwind CSS

### Database

* SQLite for local development
* MySQL-compatible database for deployment

### Development Tools

* Git
* GitHub
* Visual Studio Code
* pytest

---

## Project Structure

```text
THESIS_1/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── LICENSE
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── auth/
│   ├── dashboard/
│   ├── appointments/
│   ├── consultations/
│   ├── forecasting/
│   └── patient/
│
├── scripts/
│
├── tests/
│
├── instance/
│
├── screenshots/
│
├── model_training_evaluation_report.md
├── SETUP.md
└── README.md
```

---

## Installation

### Requirements

* Python 3.10 or later
* pip
* Git

### 1. Clone the repository

```bash
git clone https://github.com/VinceRG/THESIS_1.git
cd THESIS_1
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root.

Example:

```env
SECRET_KEY=your-secret-key

MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=your-email@example.com
MAIL_PASSWORD=your-app-password
```

Do not commit `.env` or other credentials to the repository.

---

## Running the Application

Start the Flask application with:

```bash
python app.py
```

The application will normally be available at:

```text
http://127.0.0.1:5000/
```

For development, use a separate test database and test email configuration where possible.

---

## Forecasting Workflow

The machine learning workflow follows these general steps:

```text
Consultation Records
        │
        ▼
Data Preparation
        │
        ▼
Monthly Aggregation
        │
        ▼
Diagnosis × Age × Gender Segmentation
        │
        ▼
Temporal Feature Engineering
        │
        ▼
Random Forest Regression
        │
        ▼
Model Evaluation
        │
        ▼
Monthly Consultation Forecast
        │
        ▼
Dashboard Visualization
```

The forecasting model is designed to provide an estimate of consultation demand based on historical patterns.

It should be interpreted as a **planning and analytical tool**, not as a medical diagnostic system.

---

## User Roles

### Staff

Staff users can:

* Manage consultation records
* Review patient information
* Manage appointments
* Access forecasting and analytics
* Review clinic service information

### Administrator

Administrators have additional management capabilities, including:

* User management
* System configuration
* Data management
* Forecasting administration
* Administrative reporting

### Patient

Patients can:

* Create an account
* Manage their profile
* View consultation history
* Request appointments
* Track appointment status
* Receive system notifications

---

## Security

The application includes standard web application security practices such as:

* Password hashing
* Session-based authentication
* Role-based access control
* Email verification
* Secure tokens
* CSRF protection
* ORM-based database access
* Environment-based secret configuration

Sensitive credentials and environment configuration should never be committed to the repository.

---

## Design

The system uses a custom **High-Contrast Utilitarian** visual theme designed for clarity and efficient use in an administrative healthcare environment.

### Core Design Tokens

| Role           | Color     |
| -------------- | --------- |
| Primary        | `#00477F` |
| Background     | `#EEF1F4` |
| Surface        | `#FFFFFF` |
| Text           | `#10151C` |
| Secondary Text | `#3A4451` |
| Success        | `#146C2E` |
| Warning        | `#8A4B00` |
| Error          | `#A11D1D` |
| Border         | `#1B2531` |
| Focus          | `#C95A00` |

The interface prioritizes readability, clear hierarchy, strong contrast, and practical information presentation over decorative visual effects.

---

## Open-Source Technologies

This project uses open-source technologies and libraries, including:

* Flask
* Bootstrap
* Tailwind CSS
* SQLAlchemy
* pandas
* NumPy
* scikit-learn
* joblib

Their respective licenses remain applicable to their original software.

The project's own source code is distributed under the license included in this repository.

---

## Documentation

Additional technical documentation is maintained separately from this public README.

For deeper development and handoff information, see:

* `SETUP.md`
* `model_training_evaluation_report.md`

These documents contain implementation-oriented information that is intentionally kept separate from the public project overview.

---

## Academic Context

This system was developed as part of a Bachelor of Science in Computer Science thesis at:

**Pamantasan ng Lungsod ng Pasig**

### Thesis

**Smart Healthcare Clinic Management: Predicting Consultation Case Trends at Accudetek Health Diagnostics Using Random Forest Approach**

The project explores the use of machine learning-based forecasting to support healthcare consultation demand analysis and resource planning.

---

## Limitations

The forecasting component has several important limitations:

* Predictions depend on the quality and availability of historical consultation data.
* Forecasts represent statistical estimates rather than guaranteed future demand.
* The system is intended for operational planning and analytics.
* It is not designed to provide medical diagnosis or treatment recommendations.
* Forecast performance may change as consultation patterns change over time.

---

## Credits

This project's interface is built on two open-source CSS libraries, both used under the MIT License:

* **Bootstrap v5.3.8** — Copyright (c) 2011–2026 The Bootstrap Authors
  [Bootstrap License](https://github.com/twbs/bootstrap/blob/main/LICENSE?utm_source=chatgpt.com)

* **Tailwind CSS v4.3.3** — Copyright (c) Tailwind Labs, Inc.
  [Tailwind CSS License](https://github.com/tailwindlabs/tailwindcss/blob/main/LICENSE?utm_source=chatgpt.com)

Neither library has been modified. Both are used as dependencies and restyled through their documented customization mechanisms, including Bootstrap Sass variables and Tailwind CSS's `@theme` layer.

The **High-Contrast Utilitarian** theme layer and custom styling in this repository are the work of **Vince Gonato** and are released under the MIT License. See the [`LICENSE`](LICENSE) file for the full terms.

### Attribution Locations

* **README.md** — Credits section
* **LICENSE** — MIT License for the project and its custom theme layer
* 
---

## License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for the complete license text.

Third-party libraries and frameworks remain subject to their respective licenses.

---

## Author

**Vince Gonato**

Bachelor of Science in Computer Science
Pamantasan ng Lungsod ng Pasig

2026
