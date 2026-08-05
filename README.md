# Smart Healthcare Clinic Management

This project is a Flask-based decision support system for predicting consultation case trends using a Random Forest model. The current scaffold includes:

- Staff login plus a polished landing page with separate staff and patient portal entry points
- Patient account registration with email-and-password login and six-digit email verification
- Secure patient portal for profile management, appointment history, and online appointment booking
- Dashboard, consultation records, upload, forecasting, staff, reports, and settings pages
- SQLite-backed data models for users, consultations, and staff
- A simple Random Forest-based prediction workflow

## Run the app

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000/ to choose the Staff or Patient Portal.

## Appointment reminders

The app sends a booking confirmation immediately and a confirmation email when staff confirm the appointment. To send the one-day reminder automatically, schedule this command to run once daily (for example, with Windows Task Scheduler):

```powershell
py -m flask --app app send-appointment-reminders
```
