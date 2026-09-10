# 🏥 Smart Healthcare Clinic Management System

A Flask-based **intelligent clinic management platform** with machine learning-powered consultation forecasting. This thesis project integrates patient portal management, staff dashboards, appointment scheduling, and predictive analytics using Random Forest models to optimize healthcare operations.

**📚 Thesis Focus:** Machine Learning-based Consultation Demand Forecasting for Healthcare Resource Planning

---

## 📋 Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Machine Learning Model](#machine-learning-model)
- [Key Modules](#key-modules)
- [User Roles & Access](#user-roles--access)
- [API Routes](#api-routes)
- [Database Schema](#database-schema)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### 👥 **User Management**
- Role-based access control (Staff, Admin, Patient)
- Secure staff login system
- Patient account registration with email verification
- Six-digit email verification for security
- Password reset functionality
- Session management

### 🏥 **Patient Portal**
- Profile management and history
- View consultation records
- Online appointment booking
- Appointment status tracking
- Booking confirmation and reminders
- Service catalog browsing
- Personal health information management

### 📊 **Staff Dashboard**
- Consultation management interface
- Appointment review and confirmation
- Patient management
- Consultation data entry
- Staff resource management
- Real-time analytics

### 🤖 **Machine Learning Forecasting**
- **Random Forest Regression Model** for consultation demand prediction
- Monthly consultation trend forecasting
- Demographic-based predictions (age group, gender, diagnosis)
- Hyperparameter tuning via RandomizedSearchCV
- Time-series validation methodology
- Model performance metrics and evaluation
- Automatic model retraining pipeline

### 📈 **Analytics & Reporting**
- Consultation trend analysis
- Service utilization reports
- Staff scheduling insights
- Forecast visualization
- Data export capabilities
- Performance dashboards

### 🗓️ **Appointment Management**
- Online appointment scheduling
- Automated booking confirmations
- Email reminders (1-day prior)
- Appointment history tracking
- Staff availability management
- Cancellation handling

### 🔒 **Security Features**
- Password hashing with werkzeug
- Email verification via secure tokens
- Session-based authentication
- CSRF protection
- Secure file uploads
- SQL injection prevention via SQLAlchemy ORM

---

## 🏗️ System Architecture

```
Smart Healthcare Clinic Management
│
├── Frontend Layer
│   ├── Landing Page
│   ├── Staff Portal
│   │   ├── Dashboard
│   │   ├── Consultations
│   │   ├── Appointments
│   │   ├── Forecasting
│   │   ├── Reports
│   │   └── Settings
│   │
│   └── Patient Portal
│       ├── Dashboard
│       ├── Appointments
│       ├── Profile
│       └── Health Records
│
├── Backend Layer (Flask)
│   ├── Authentication Module
│   ├── Appointment Management
│   ├── Consultation Tracking
│   ├── Forecasting Engine
│   ├── Email Service
│   ├── File Upload Handler
│   ├── Data Processing Pipeline
│   └── Analytics Module
│
├── Machine Learning Layer
│   ├── Random Forest Model
│   ├── Feature Engineering
│   ├── Model Training Pipeline
│   ├── Hyperparameter Tuning
│   └── Prediction Engine
│
└── Data Layer
    └── SQLite Database
        ├── Users
        ├── Consultations
        ├── Appointments
        ├── Staff
        └── Services
```

---

## 💻 Tech Stack

### Backend
- **Flask 3.0.3** - Web framework
- **Flask-SQLAlchemy 3.1.1** - ORM and database management
- **python-dotenv 1.0.1** - Environment variable management

### Machine Learning & Data Science
- **scikit-learn 1.5.2** - Random Forest, feature scaling, model evaluation
- **pandas 2.2.3** - Data manipulation and processing
- **numpy 2.1.0** - Numerical computations
- **matplotlib ≥3.9.0** - Data visualization
- **joblib 1.4.0** - Model serialization

### Data Handling
- **openpyxl 3.1.0** - Excel file support for data imports

### Database
- **SQLite** - Lightweight, file-based relational database

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling
- **JavaScript** - Interactivity

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment support

### Step 1: Clone the Repository
```bash
git clone https://github.com/VinceRG/THESIS_1.git
cd THESIS_1-main
```

### Step 2: Create Virtual Environment
```bash
# On Windows
python -m venv .venv
.\.venv\Scripts\activate

# On macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Environment Configuration
Create a `.env` file in the project root (optional for email features):
```env
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password
MAIL_USE_TLS=True
FLASK_ENV=development
FLASK_DEBUG=True
```

---

## ▶️ Running the Application

### Development Mode
```bash
python app.py
```

Access the application at: **http://127.0.0.1:5000/**

### Initial Access
- **Staff Portal:** Choose "Staff Portal" on the landing page
- **Patient Portal:** Choose "Patient Portal" or register a new account

### Appointment Reminders
To send 1-day advance appointment reminders (requires Windows Task Scheduler or cron job):

```bash
python -m flask --app app send-appointment-reminders
```

This can be scheduled as a daily automated task.

---

## 📁 Project Structure

```
THESIS_1-main/
├── app.py                              # Main Flask application (5,600+ lines)
├── requirements.txt                    # Python dependencies
├── .env                               # Environment variables (optional)
├── .gitignore                         # Git ignore rules
│
├── instance/
│   └── clinic.db                      # SQLite database (auto-created)
│
├── static/                            # Static assets
│   ├── css/                           # Stylesheets
│   │   ├── dashboard.css
│   │   ├── patient_portal.css
│   │   ├── forecasting.css
│   │   └── ...
│   ├── js/                            # JavaScript files
│   ├── Screenshot 2026-07-09.png      # Demo image
│   └── acudetek.jfif                  # Logo/branding
│
├── templates/                         # HTML templates
│   ├── landing.html                   # Entry point
│   ├── layouts/                       # Base templates
│   │   ├── staff_base.html
│   │   └── patient_base.html
│   ├── auth/                          # Authentication pages
│   │   ├── staff_login.html
│   │   ├── patient_register.html
│   │   └── patient_login.html
│   ├── dashboard/                     # Dashboard pages
│   ├── consultations/                 # Consultation pages
│   ├── appointments/                  # Appointment management
│   ├── forecasting/                   # Prediction interface
│   ├── reports/                       # Analytics & reports
│   ├── staff/                         # Staff management
│   ├── services/                      # Service catalog
│   ├── patient_portal/                # Patient-facing pages
│   └── ...
│
├── scripts/                           # Utility scripts
│   └── screenshot_routes.py           # Screenshot generation utility
│
├── tests/                             # Unit and integration tests
│   └── test_app.py                    # Test suite
│
├── scrape_accudetek_services.py       # Data scraping utility
├── model_training_evaluation_report.md # ML model documentation
├── README.md                          # Quick start guide
├── SETUP.md                           # Local setup instructions
└── screenshots/                       # Application screenshots
    ├── admin/                         # Admin interface screenshots
    └── patient/                       # Patient portal screenshots
```

---

## 🤖 Machine Learning Model

### Model Architecture
- **Algorithm:** Random Forest Regression
- **Purpose:** Forecast monthly consultation demand
- **Prediction Granularity:** Diagnosis × Age Group × Gender × Month

### Dataset Details
| Metric | Value |
|--------|-------|
| Source Records | 10,129 consultation records (Jan 2023 - Aug 2026) |
| Training Period | Jan 2023 - Dec 2025 |
| Training Records | 3,672 aggregated monthly segments |
| Diagnoses Modeled | 17 primary + "Other" category |
| Age Groups | 3 (pediatric, adult, geriatric) |
| Gender Categories | 2 (Male, Female) |

### Feature Engineering
The model incorporates sophisticated temporal features:

- **Temporal Features:**
  - Monthly seasonality (sine/cosine encoding)
  - Time index
  
- **Lag Features:**
  - 1, 2, 3, 6, and 12-month lags
  
- **Rolling Statistics:**
  - 3-month and 6-month rolling averages
  - Rolling standard deviation
  
- **Trend Indicators:**
  - Recent trend calculation
  - Historical seasonality patterns

### Model Validation
- **Approach:** Time-Series Cross-Validation (no data leakage)
- **Train/Test Split:** 30 months training, 6 months validation
- **Validation Period:** July 2025 - December 2025
- **Baseline:** Naive forecast (previous month = next month)

### Hyperparameter Tuning
- **Method:** RandomizedSearchCV
- **Tuned Parameters:**
  - n_estimators
  - max_depth
  - min_samples_split
  - min_samples_leaf
  
### Performance Metrics
The model predicts consultation volume with:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score (coefficient of determination)
- RMSE (Root Mean Squared Error)

### Model Training Routes
```python
# Fast training (for uploads, fixed parameters)
POST /upload → train_and_evaluate_model(fast=True)

# Full tuning (for administrators, hyperparameter search)
POST /retrain → train_and_evaluate_model(fast=False)
```

### Model Persistence
- Trained model saved via `joblib` for quick loading
- Can be retrained at any time via admin interface
- Supports incremental updates with new data

---

## 🔑 Key Modules

### Authentication (`auth/` routes)
```python
/auth/staff-login        # Staff login
/auth/patient-register   # Patient registration
/auth/patient-login      # Patient login
/auth/logout            # Session termination
/auth/verify-email/<token>  # Email verification
```

### Dashboard (`dashboard/` routes)
```python
/dashboard              # Main admin dashboard
/dashboard/stats       # Dashboard statistics
/dashboard/consultations   # Consultation overview
```

### Consultation Management (`consultations/` routes)
```python
/consultations         # List consultations
/consultations/new     # Create consultation
/consultations/<id>    # View consultation
/consultations/<id>/edit  # Edit consultation
```

### Appointment Management (`appointments/` routes)
```python
/appointments          # List appointments
/appointments/book     # New appointment
/appointments/<id>/confirm  # Staff confirmation
/appointments/<id>/cancel   # Cancellation
```

### Forecasting (`forecasting/` routes)
```python
/forecasting           # Forecast interface
/forecasting/predict   # Generate predictions
/forecasting/data      # Forecast data export
```

### Patient Portal (`patient/` routes)
```python
/patient/dashboard     # Patient dashboard
/patient/appointments  # Patient appointment history
/patient/health-records  # Medical records
/patient/profile       # Profile management
```

### Admin & Settings
```python
/settings              # System settings
/admin/users          # User management
/admin/services       # Service management
/audit-logs           # Action audit trail
```

---

## 👥 User Roles & Access

### Staff
- **Capabilities:**
  - View all patients and consultations
  - Manage appointments (confirm, cancel)
  - Enter and update consultation data
  - Access forecasting dashboard
  - Generate reports
  - Manage services and packages
  - View audit logs

### Patients
- **Capabilities:**
  - Register and login
  - View personal dashboard
  - Book appointments online
  - Check appointment status
  - Manage profile
  - View consultation history
  - Receive appointment reminders

### System Admin
- **Capabilities:**
  - All staff capabilities
  - User account management
  - System settings
  - Model retraining
  - Data import/export
  - Full audit trail access

---

## 🛣️ API Routes

### Public Routes
```
GET  /                     # Landing page
GET  /auth/staff-login     # Staff login form
GET  /auth/patient-login   # Patient login form
GET  /auth/patient-register # Registration form
POST /auth/staff-login     # Process staff login
POST /auth/patient-login   # Process patient login
POST /auth/patient-register # Process registration
```

### Authenticated Staff Routes
```
GET/POST /dashboard                    # Main dashboard
GET/POST /consultations                # Consultation list
GET/POST /consultations/new            # New consultation
GET/POST /appointments                 # Appointment management
GET/POST /forecasting                  # Forecast interface
POST     /forecasting/predict           # Generate predictions
GET/POST /reports                      # Reports
GET/POST /services                     # Service management
```

### Authenticated Patient Routes
```
GET /patient/dashboard                 # Patient dashboard
GET /patient/appointments              # My appointments
POST /patient/appointments/book        # Book appointment
GET /patient/health-records            # Health records
GET/POST /patient/profile              # Profile management
```

### Admin Routes
```
GET/POST /admin/users                  # User management
GET/POST /admin/services               # Service management
GET      /admin/settings               # System settings
GET      /audit-logs                   # Audit trail
POST     /upload                       # Data import (CSV/Excel)
POST     /retrain                      # Model retraining
```

---

## 🗄️ Database Schema

### Users Table
```sql
users (
    id INTEGER PRIMARY KEY,
    email VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR NOT NULL,
    first_name VARCHAR,
    last_name VARCHAR,
    role VARCHAR (staff/patient/admin),
    email_verified BOOLEAN,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
```

### Consultations Table
```sql
consultations (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    staff_id INTEGER,
    consultation_date DATETIME,
    diagnosis VARCHAR,
    description TEXT,
    service_type VARCHAR,
    age_group VARCHAR,
    gender VARCHAR,
    created_at TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES users(id),
    FOREIGN KEY (staff_id) REFERENCES users(id)
)
```

### Appointments Table
```sql
appointments (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    staff_id INTEGER,
    appointment_date DATETIME,
    service_type VARCHAR,
    status VARCHAR (pending/confirmed/cancelled),
    notes TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    reminder_sent BOOLEAN,
    FOREIGN KEY (patient_id) REFERENCES users(id)
)
```

### Services Table
```sql
services (
    id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    description TEXT,
    category VARCHAR,
    is_available BOOLEAN,
    created_at TIMESTAMP
)
```

---

## ⚙️ Configuration

### Email Configuration
The application supports appointment reminders via email. Configure in `.env`:

```env
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password  # Use Gmail App Password
MAIL_USE_TLS=True
```

### Database Configuration
SQLite is used by default. To change:

```python
# In app.py
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/clinic.db'
# Or for MySQL:
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://user:password@localhost/clinic'
```

### Flask Configuration
```python
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
```

---

## 🧪 Testing

Run the test suite:

```bash
# Using pytest (recommended)
pip install pytest
pytest tests/

# Or using unittest
python -m unittest tests.test_app
```

Test coverage includes:
- User authentication
- Appointment management
- Consultation tracking
- Email verification
- ML model predictions
- Data upload/import

---

## 🔧 Troubleshooting

### Common Issues

#### Database Missing
```
Error: instance/clinic.db not found
```
**Solution:** Run the app once - database will auto-create
```bash
python app.py
```

#### Port Already in Use
```
Error: Address already in use
```
**Solution:** Change port or kill existing process
```bash
python app.py  # App will use next available port
```

#### Email Verification Not Working
**Check:**
- `.env` file has correct credentials
- Gmail account has "Less secure app access" enabled OR use App Password
- SMTP settings are correct

#### Virtual Environment Issues
**Solution:**
```bash
# Remove and recreate
rmdir .venv  # or rm -rf .venv on Linux
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### Import Errors After Dependencies Install
```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +
# Reinstall
pip install -r requirements.txt --force-reinstall
```

---

## 🤝 Contributing

### Development Workflow
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and test locally
3. Commit with clear messages: `git commit -m "Add feature: description"`
4. Push to branch: `git push origin feature/your-feature`
5. Create Pull Request with detailed description

### Code Standards
- Follow PEP 8 for Python
- Use meaningful variable/function names
- Add docstrings for functions
- Comment complex logic
- Test before submitting

### Report Issues
Include:
- Python version
- Operating system
- Steps to reproduce
- Error messages/logs
- Screenshots if applicable

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💼 Author & Thesis Information

**Student:** Vince Gonato  
**Institution:** [Your University]  
**Program:** Bachelor of Science in Computer Science  
**Thesis Title:** Smart Healthcare Clinic Management System with Machine Learning-Based Consultation Forecasting  
**Year:** 2026

---

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/)
- [scikit-learn RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Pandas for Data Manipulation](https://pandas.pydata.org/)
- [Time Series Forecasting Best Practices](https://otexts.com/fpp2/)

### Related Documentation
- `SETUP.md` - Detailed local setup instructions
- `model_training_evaluation_report.md` - Complete ML model analysis
- `requirements.txt` - Exact dependency versions

---

## 🎓 Academic References

This system demonstrates:
- **Software Engineering:** Multi-tier architecture, design patterns, security
- **Data Science:** Feature engineering, time-series analysis, model validation
- **Machine Learning:** Random Forest, hyperparameter tuning, cross-validation
- **Database Design:** Relational modeling, normalization, integrity
- **Web Development:** User authentication, session management, responsive design

---

## 📞 Support

For questions or support:
- Check documentation in `/docs`
- Review existing issues in GitHub
- Contact maintainer: [your-email@example.com]

---

**Last Updated:** August 2026  
**Version:** 1.0.0  
**Status:** Production Ready ✅

---

Made with ❤️ for better healthcare management through intelligent forecasting
