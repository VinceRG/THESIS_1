# Smart Healthcare Clinic Management

This project is a Flask-based decision support system for predicting consultation case trends using a Random Forest model. The current scaffold includes:

- Authentication flow with login/logout
- Dashboard, consultation records, upload, forecasting, staff, reports, and settings pages
- SQLite-backed data models for users, consultations, and staff
- A simple Random Forest-based prediction workflow

## Run the app

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000/login
