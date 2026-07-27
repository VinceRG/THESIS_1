import csv
import json
import os
import re
import traceback
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from flask import Flask, Response, flash, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_, text
from sqlalchemy.exc import IntegrityError
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
SERVICE_CATALOG = [
    'General Physician Consultation',
    'Clinical Laboratory Services',
    'Clinical Microscopy',
    'Hematology',
    'Clinical Chemistry',
    'Immunology and Serology',
    'Drug Testing',
    'Electrocardiography (ECG)',
    'X-ray',
    'Ultrasound',
    'Vascular Studies',
    'Home Service',
    'Annual Physical Examination (APE)',
]

EQUIPMENT_INVENTORY = [
    'Automated Hematology Analyzer',
    'Automated Clinical Chemistry Analyzer',
    'Automated Immunoassay Analyzer',
    'Automated Electrolyte Analyzer',
    'Ultrasound Machine',
    'X-ray System/Machine',
    'Electrocardiograph (ECG) Machine',
]

FACILITY_STAFF_COMPLEMENT = [
    {'role': 'Pathologists', 'count': 2, 'notes': ''},
    {'role': 'Registered Medical Technologists', 'count': 6, 'notes': 'Includes 2 Drug Test Analysts and 1 HIV Counselor'},
    {'role': 'Registered Radiologic Technologists', 'count': 2, 'notes': ''},
    {'role': 'Laboratory Technicians', 'count': 3, 'notes': ''},
    {'role': 'Internal Medicine Physicians', 'count': 2, 'notes': ''},
    {'role': 'General Physicians', 'count': 4, 'notes': ''},
    {'role': 'Radiologists', 'count': 3, 'notes': ''},
]

FACILITY_STAFF_ROSTER = [
    {'name': 'Dr. Maria Santos', 'role': 'Pathologists'},
    {'name': 'Dr. Roberto Cruz', 'role': 'Pathologists'},
    {'name': 'Angela Reyes, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Mark Villanueva, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Patricia Mendoza, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Carlo Navarro, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Joanna Lim, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Rafael Torres, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Grace Aquino, RRT', 'role': 'Registered Radiologic Technologists'},
    {'name': 'Nico Garcia, RRT', 'role': 'Registered Radiologic Technologists'},
    {'name': 'Liza Ramos', 'role': 'Laboratory Technicians'},
    {'name': 'Benjie Castillo', 'role': 'Laboratory Technicians'},
    {'name': 'Mariel Dizon', 'role': 'Laboratory Technicians'},
    {'name': 'Dr. Elena Bautista', 'role': 'Internal Medicine Physicians'},
    {'name': 'Dr. Victor Manuel', 'role': 'Internal Medicine Physicians'},
    {'name': 'Dr. Ana Lopez', 'role': 'General Physicians'},
    {'name': 'Dr. Paolo Rivera', 'role': 'General Physicians'},
    {'name': 'Dr. Camille Tan', 'role': 'General Physicians'},
    {'name': 'Dr. Miguel Fernandez', 'role': 'General Physicians'},
    {'name': 'Dr. Andrea Sy', 'role': 'Radiologists'},
    {'name': 'Dr. Henry Ong', 'role': 'Radiologists'},
    {'name': 'Dr. Sofia Mercado', 'role': 'Radiologists'},
]

# Resource planning constants
ROOM_COUNT = 5
AVG_CONSULTATION_MINUTES = 20
WORKING_HOURS_PER_DAY = 8
DAYS_PER_MONTH = 22
STAFF_CAPACITY_PER_MONTH = 40
DEFAULT_BRANCH_CODE = 'MAIN'
DEFAULT_BRANCH_NAME = 'Accudetek Main Branch'
DEFAULT_BRANCH_ADDRESS = 'JL Building, 12 M.H. del Pilar St, San Nicolas, Pasig, 1600 Metro Manila'
SUPER_ADMIN_ROLE = 'superadmin'
MAIN_ADMIN_ROLES = {SUPER_ADMIN_ROLE, 'administrator', 'main_admin'}
ALL_BRANCHES_SCOPE = 'all'
USER_ROLE_OPTIONS = [SUPER_ADMIN_ROLE, 'administrator', 'branch_admin', 'staff']
SERVICE_MANAGER_ROLES = {SUPER_ADMIN_ROLE, 'administrator', 'main_admin', 'branch_admin'}
USER_ROLE_LABELS = {
    SUPER_ADMIN_ROLE: 'Super Admin',
    'administrator': 'Administrator',
    'main_admin': 'Main Administrator',
    'branch_admin': 'Branch Admin',
    'staff': 'Staff',
}
PATIENT_GENDER_OPTIONS = ['Female', 'Male']
PATIENT_AGE_GROUP_OPTIONS = ['Child', 'Adult', 'Senior']
CONSULTATION_TYPE_OPTIONS = [
    'Walk-in',
    'Follow-up',
    'Check-up',
    'Laboratory',
    'Imaging',
    'Annual Physical Examination',
]
APPOINTMENT_STATUS_OPTIONS = ['Pending', 'Confirmed', 'Completed', 'Cancelled']
ACTIVE_APPOINTMENT_SLOT_STATUSES = ['Pending', 'Confirmed']
COMMON_CONSULTATION_REASONS = [
    'Headache',
    'Fever',
    'Cough and colds',
    'Sore throat',
    'Dizziness',
    'Body weakness or fatigue',
    'Abdominal pain',
    'Urinary tract infection symptoms',
    'Gastrointestinal complaints',
    'Skin conditions and allergies',
    'Routine medical consultation and health clearance',
    'Follow-up consultation',
    'Hypertension monitoring and management',
    'Diabetes mellitus monitoring and management',
    'Asthma monitoring and management',
    'Preventive health check-up and wellness consultation',
]
SERVICE_RECOMMENDATION_MAP = {
    'General Physician Consultation': {
        'roles': ['General Physicians'],
        'equipment': [],
    },
    'Clinical Laboratory Services': {
        'roles': ['Registered Medical Technologists', 'Laboratory Technicians'],
        'equipment': ['Automated Hematology Analyzer', 'Automated Clinical Chemistry Analyzer'],
    },
    'Clinical Microscopy': {
        'roles': ['Registered Medical Technologists', 'Laboratory Technicians'],
        'equipment': ['Clinical microscopy laboratory resources'],
    },
    'Hematology': {
        'roles': ['Registered Medical Technologists'],
        'equipment': ['Automated Hematology Analyzer'],
    },
    'Clinical Chemistry': {
        'roles': ['Registered Medical Technologists'],
        'equipment': ['Automated Clinical Chemistry Analyzer', 'Automated Electrolyte Analyzer'],
    },
    'Immunology and Serology': {
        'roles': ['Registered Medical Technologists'],
        'equipment': ['Automated Immunoassay Analyzer'],
    },
    'Drug Testing': {
        'roles': ['Registered Medical Technologists', 'Drug Test Analysts'],
        'equipment': ['Drug testing laboratory resources'],
    },
    'Electrocardiography (ECG)': {
        'roles': ['General Physicians', 'Trained ECG Staff'],
        'equipment': ['Electrocardiograph (ECG) Machine'],
    },
    'X-ray': {
        'roles': ['Registered Radiologic Technologists', 'Radiologists'],
        'equipment': ['X-ray System/Machine'],
    },
    'Ultrasound': {
        'roles': ['Radiologists', 'Registered Radiologic Technologists'],
        'equipment': ['Ultrasound Machine'],
    },
    'Vascular Studies': {
        'roles': ['Radiologists', 'Registered Radiologic Technologists'],
        'equipment': ['Ultrasound Machine'],
    },
    'Home Service': {
        'roles': ['General Physicians', 'Registered Medical Technologists', 'Laboratory Technicians'],
        'equipment': ['Home service kit'],
    },
    'Annual Physical Examination (APE)': {
        'roles': ['General Physicians', 'Registered Medical Technologists', 'Registered Radiologic Technologists'],
        'equipment': ['Automated Hematology Analyzer', 'X-ray System/Machine', 'Electrocardiograph (ECG) Machine'],
    },
}
REASON_RECOMMENDATION_MAP = {
    'Headache': ['General Physicians'],
    'Fever': ['General Physicians'],
    'Cough and colds': ['General Physicians'],
    'Sore throat': ['General Physicians'],
    'Dizziness': ['General Physicians'],
    'Body weakness or fatigue': ['General Physicians'],
    'Abdominal pain': ['General Physicians', 'Internal Medicine Physicians'],
    'Urinary tract infection symptoms': ['General Physicians', 'Registered Medical Technologists'],
    'Gastrointestinal complaints': ['General Physicians', 'Internal Medicine Physicians'],
    'Skin conditions and allergies': ['General Physicians'],
    'Routine medical consultation and health clearance': ['General Physicians'],
    'Follow-up consultation': ['General Physicians'],
    'Hypertension monitoring and management': ['Internal Medicine Physicians', 'General Physicians'],
    'Diabetes mellitus monitoring and management': ['Internal Medicine Physicians', 'General Physicians'],
    'Asthma monitoring and management': ['Internal Medicine Physicians', 'General Physicians'],
    'Preventive health check-up and wellness consultation': ['General Physicians'],
}
REASON_KEYWORD_RECOMMENDATIONS = {
    'asthma': ['Internal Medicine Physicians', 'General Physicians'],
    'hypertension': ['Internal Medicine Physicians', 'General Physicians'],
    'diabetes': ['Internal Medicine Physicians', 'General Physicians'],
    'fever': ['General Physicians'],
    'cough': ['General Physicians'],
    'cold': ['General Physicians'],
    'sore throat': ['General Physicians'],
    'uti': ['General Physicians', 'Registered Medical Technologists'],
    'urinary': ['General Physicians', 'Registered Medical Technologists'],
    'abdominal': ['General Physicians', 'Internal Medicine Physicians'],
    'diarrhea': ['General Physicians', 'Internal Medicine Physicians'],
    'vomiting': ['General Physicians', 'Internal Medicine Physicians'],
    'skin': ['General Physicians'],
    'allergy': ['General Physicians'],
}
STAFF_FORECAST_ROLE_ALIASES = {
    'Drug Test Analysts': 'Registered Medical Technologists',
    'HIV Counselor': 'Registered Medical Technologists',
    'Trained ECG Staff': 'General Physicians',
}
DIAGNOSIS_STAFF_ROLE_RULES = [
    (['hypertension', 'diabetes', 'asthma', 'gastritis', 'gastro', 'abdominal', 'urinary', 'uti'], ['Internal Medicine Physicians']),
    (['x-ray', 'xray', 'ultrasound', 'vascular', 'radiology', 'imaging'], ['Radiologists', 'Registered Radiologic Technologists']),
    (['laboratory', 'hematology', 'chemistry', 'urinalysis', 'fecalysis', 'cbc', 'blood', 'drug test'], ['Registered Medical Technologists']),
    (['dermatitis', 'skin', 'allergy', 'rash', 'respiratory', 'cough', 'fever', 'headache', 'dizziness', 'fatigue'], ['General Physicians']),
]
STAFF_FORECAST_DAILY_DAYS = 14
MAX_RF_TARGET_LABELS = 40
OTHER_RF_TARGET_LABEL = 'Other Services'

# -------------------------------------------------------------
# Database setup
# -------------------------------------------------------------
db = SQLAlchemy()

class Branch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    code = db.Column(db.String(40), unique=True, nullable=False)
    address = db.Column(db.String(255), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    contact_number = db.Column(db.String(40), nullable=True)
    is_main = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer, db.ForeignKey('branch.id'), nullable=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(40), default='staff')
    branch = db.relationship('Branch', backref='users')

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer, db.ForeignKey('branch.id'), nullable=True)
    patient_number = db.Column(db.String(60), unique=True, nullable=False)
    full_name = db.Column(db.String(140), nullable=False)
    birthdate = db.Column(db.String(20), nullable=True)
    age = db.Column(db.Integer, nullable=True)
    age_group = db.Column(db.String(40), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    contact_number = db.Column(db.String(40), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    address = db.Column(db.String(255), nullable=True)
    emergency_contact_name = db.Column(db.String(140), nullable=True)
    emergency_contact_number = db.Column(db.String(40), nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    branch = db.relationship('Branch', backref='patients')

class ConsultationRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer, db.ForeignKey('branch.id'), nullable=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=True)
    consultation_date = db.Column(db.String(20), nullable=False)
    patient_age = db.Column(db.Integer, nullable=True)
    age_group = db.Column(db.String(40), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    diagnosis = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    physician = db.Column(db.String(100), nullable=False)
    consultation_type = db.Column(db.String(100), nullable=False)
    branch = db.relationship('Branch', backref='consultation_records')
    patient = db.relationship('Patient', backref='consultation_records')

class ClinicService(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(100), nullable=False)
    section = db.Column(db.String(140), nullable=True)
    service_name = db.Column(db.String(180), nullable=False)
    price_php = db.Column(db.String(40), nullable=True)
    required_roles = db.Column(db.Text, nullable=False, default='[]')
    required_equipment = db.Column(db.Text, nullable=False, default='[]')
    source_page = db.Column(db.String(80), nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    @staticmethod
    def _json_list(value):
        try:
            loaded = json.loads(value or '[]')
            return loaded if isinstance(loaded, list) else []
        except Exception:
            return []

    def role_items(self):
        return self._json_list(self.required_roles)

    def equipment_items(self):
        return self._json_list(self.required_equipment)

class BranchService(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer, db.ForeignKey('branch.id'), nullable=False)
    service_id = db.Column(db.Integer, db.ForeignKey('clinic_service.id'), nullable=False)
    custom_price_php = db.Column(db.String(40), nullable=True)
    is_available = db.Column(db.Boolean, default=True, nullable=False)
    branch_notes = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    branch = db.relationship('Branch', backref='branch_services')
    service = db.relationship('ClinicService', backref='branch_services')
    __table_args__ = (
        db.UniqueConstraint('branch_id', 'service_id', name='uq_branch_service_service'),
    )

class ClinicPackage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    package_name = db.Column(db.String(180), unique=True, nullable=False)
    price_php = db.Column(db.String(40), nullable=True)
    source_page = db.Column(db.String(80), nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

class PackageItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    package_id = db.Column(db.Integer, db.ForeignKey('clinic_package.id'), nullable=False)
    service_id = db.Column(db.Integer, db.ForeignKey('clinic_service.id'), nullable=True)
    item_name = db.Column(db.String(180), nullable=False)
    item_order = db.Column(db.Integer, default=0, nullable=False)
    required_roles = db.Column(db.Text, nullable=False, default='[]')
    required_equipment = db.Column(db.Text, nullable=False, default='[]')
    package = db.relationship('ClinicPackage', backref='items')
    service = db.relationship('ClinicService')

    @staticmethod
    def _json_list(value):
        try:
            loaded = json.loads(value or '[]')
            return loaded if isinstance(loaded, list) else []
        except Exception:
            return []

    def role_items(self):
        return self._json_list(self.required_roles)

    def equipment_items(self):
        return self._json_list(self.required_equipment)

class BranchPackage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer, db.ForeignKey('branch.id'), nullable=False)
    package_id = db.Column(db.Integer, db.ForeignKey('clinic_package.id'), nullable=False)
    custom_price_php = db.Column(db.String(40), nullable=True)
    is_available = db.Column(db.Boolean, default=True, nullable=False)
    branch_notes = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    branch = db.relationship('Branch', backref='branch_packages')
    package = db.relationship('ClinicPackage', backref='branch_packages')
    __table_args__ = (
        db.UniqueConstraint('branch_id', 'package_id', name='uq_branch_package_package'),
    )

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer, db.ForeignKey('branch.id'), nullable=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    appointment_date = db.Column(db.String(20), nullable=False)
    appointment_time = db.Column(db.String(20), nullable=True)
    selected_services = db.Column(db.Text, nullable=False, default='[]')
    selected_packages = db.Column(db.Text, nullable=False, default='[]')
    consultation_reasons = db.Column(db.Text, nullable=False, default='[]')
    other_reason = db.Column(db.String(255), nullable=True)
    recommended_roles = db.Column(db.Text, nullable=False, default='[]')
    recommended_equipment = db.Column(db.Text, nullable=False, default='[]')
    recommendation_notes = db.Column(db.Text, nullable=False, default='[]')
    status = db.Column(db.String(30), default='Pending', nullable=False)
    completed_at = db.Column(db.DateTime, nullable=True)
    converted_to_records = db.Column(db.Boolean, default=False, nullable=False)
    completion_notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    branch = db.relationship('Branch', backref='appointments')
    patient = db.relationship('Patient', backref='appointments')

    @staticmethod
    def _json_list(value):
        try:
            loaded = json.loads(value or '[]')
            return loaded if isinstance(loaded, list) else []
        except Exception:
            return []

    def service_items(self):
        return self._json_list(self.selected_services)

    def package_items(self):
        return self._json_list(self.selected_packages)

    def reason_items(self):
        return self._json_list(self.consultation_reasons)

    def role_items(self):
        return self._json_list(self.recommended_roles)

    def equipment_items(self):
        return self._json_list(self.recommended_equipment)

    def note_items(self):
        return self._json_list(self.recommendation_notes)

class AppointmentServiceResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey('appointment.id'), nullable=False)
    consultation_record_id = db.Column(db.Integer, db.ForeignKey('consultation_record.id'), nullable=True)
    service_name = db.Column(db.String(120), nullable=False)
    assigned_staff = db.Column(db.String(120), nullable=False)
    final_diagnosis = db.Column(db.String(120), nullable=False)
    service_notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    appointment = db.relationship('Appointment', backref='service_results')
    consultation_record = db.relationship('ConsultationRecord', backref='appointment_service_results')

class StaffMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer, db.ForeignKey('branch.id'), nullable=True)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(60), nullable=False)
    availability = db.Column(db.String(20), default='Available')
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    deleted_at = db.Column(db.DateTime, nullable=True)
    branch = db.relationship('Branch', backref='staff_members')

def build_facility_staff_roster():
    return [
        {
            'name': person['name'],
            'role': person['role'],
            'availability': 'Available',
        }
        for person in FACILITY_STAFF_ROSTER
    ]

def clean_service_catalog_value(value):
    text_value = str(value or '').strip()
    replacements = {
        'Hermatology': 'Hematology',
    }
    if 'B Vaccine' in text_value and any(ord(character) > 127 for character in text_value):
        return 'Hepa B Vaccine'
    return replacements.get(text_value, text_value)

def infer_service_requirements(category, section, service_name):
    category = clean_service_catalog_value(category)
    section = clean_service_catalog_value(section)
    service_name = clean_service_catalog_value(service_name)
    combined = f'{category} {section} {service_name}'.lower()
    roles = []
    equipment = []

    if category == 'Medical Consultation' or 'consultation' in combined or category == 'Vaccine':
        roles.extend(['General Physicians'])

    if category == 'Laboratory':
        roles.extend(['Registered Medical Technologists', 'Laboratory Technicians'])
        if 'hematology' in combined or 'cbc' in combined or 'platelet' in combined:
            equipment.append('Automated Hematology Analyzer')
        if any(term in combined for term in ['chemistry', 'glucose', 'creatinine', 'cholesterol', 'lipid', 'electrolyte', 'sodium', 'potassium']):
            equipment.append('Automated Clinical Chemistry Analyzer')
        if any(term in combined for term in ['electrolyte', 'sodium', 'potassium', 'chloride']):
            equipment.append('Automated Electrolyte Analyzer')
        if any(term in combined for term in ['immunology', 'serology', 'hormone', 'thyroid', 'hepatitis']):
            equipment.append('Automated Immunoassay Analyzer')
        if any(term in combined for term in ['urine', 'urinalysis', 'fecalysis', 'microscopy']):
            equipment.append('Clinical microscopy laboratory resources')
        if 'histopathology' in combined or 'biopsy' in combined:
            roles.append('Pathologists')

    if category == 'Drug Testing' or 'drug test' in combined:
        roles.extend(['Registered Medical Technologists', 'Drug Test Analysts'])
        equipment.append('Drug testing laboratory resources')

    if category == 'Imaging/Cardiology':
        if any(term in combined for term in ['x-ray', 'xray', 'chest', 'skull', 'spine', 'pelvis', 'knee', 'ankle', 'wrist']):
            roles.extend(['Registered Radiologic Technologists', 'Radiologists'])
            equipment.append('X-ray System/Machine')
        if any(term in combined for term in ['ultrasound', 'duplex', 'vascular']):
            roles.extend(['Radiologists', 'Registered Radiologic Technologists'])
            equipment.append('Ultrasound Machine')
        if any(term in combined for term in ['ecg', 'echo', 'echocardiogram']):
            roles.extend(['General Physicians', 'Trained ECG Staff'])
            equipment.append('Electrocardiograph (ECG) Machine')

    if category in {'Pre-Employment Checkup', 'Annual Checkup'}:
        roles.extend(['General Physicians', 'Registered Medical Technologists', 'Registered Radiologic Technologists'])
        equipment.extend(['Automated Hematology Analyzer', 'X-ray System/Machine'])

    if category == 'Home Service':
        roles.extend(['Registered Medical Technologists'])
        equipment.append('Home service kit')
        if 'x-ray' in combined or 'xray' in combined:
            roles.extend(['Registered Radiologic Technologists', 'Radiologists'])
            equipment.append('X-ray System/Machine')

    if not roles:
        roles = ['General Physicians']

    return {
        'roles': list(dict.fromkeys(roles)),
        'equipment': list(dict.fromkeys(equipment)),
    }

def service_booking_label(service):
    parts = [clean_service_catalog_value(service.service_name)]
    section = clean_service_catalog_value(service.section)
    category = clean_service_catalog_value(service.category)
    if section and section != category and section.lower() not in service.service_name.lower():
        parts.append(section)
    if category and category.lower() not in ' '.join(parts).lower():
        parts.append(category)
    return ' | '.join(parts)

def service_catalog_json(values):
    seen = set()
    cleaned_values = []
    for value in values:
        value = clean_service_catalog_value(value)
        if value and value not in seen:
            seen.add(value)
            cleaned_values.append(value)
    return json.dumps(cleaned_values)

def service_identity(category, section, service_name):
    return (
        clean_service_catalog_value(category).lower(),
        clean_service_catalog_value(section).lower(),
        clean_service_catalog_value(service_name).lower(),
    )

def normalize_match_text(value):
    return re.sub(r'[^a-z0-9]+', '', str(value or '').lower())

def infer_package_item_category(item_name):
    lower_name = str(item_name or '').lower()
    if any(term in lower_name for term in ['x-ray', 'xray', 'chest', 'ultrasound', 'duplex', 'ecg', 'echo']):
        return 'Imaging/Cardiology'
    if 'drug' in lower_name or 'met' == lower_name.strip() or 'thc' == lower_name.strip():
        return 'Drug Testing'
    if any(term in lower_name for term in ['physical exam', 'consultation', 'medical exam']):
        return 'Medical Consultation'
    return 'Laboratory'

def find_matching_clinic_service(item_name):
    needle = normalize_match_text(item_name)
    if not needle:
        return None
    services = ClinicService.query.all()
    for service in services:
        if normalize_match_text(service.service_name) == needle:
            return service
    for service in services:
        service_name = normalize_match_text(service.service_name)
        if needle in service_name or service_name in needle:
            return service
    return None

def package_item_requirements(item_name, linked_service=None):
    if linked_service:
        return linked_service.role_items(), linked_service.equipment_items()
    category = infer_package_item_category(item_name)
    requirements = infer_service_requirements(category, 'Package Item', item_name)
    return requirements['roles'], requirements['equipment']

def package_identity(package_name):
    return clean_service_catalog_value(package_name).lower()

def import_accudetek_service_catalog(app, seed_only=False):
    with app.app_context():
        if seed_only and ClinicService.query.first():
            return 0

        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'accudetek_services_scraped.csv')
        if not os.path.exists(csv_path):
            return 0

        existing = {
            service_identity(service.category, service.section, service.service_name)
            for service in ClinicService.query.all()
        }

        added_count = 0
        with open(csv_path, newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                category = clean_service_catalog_value(row.get('category', ''))
                section = clean_service_catalog_value(row.get('section', ''))
                service_name = clean_service_catalog_value(row.get('service_name', ''))
                price_php = clean_service_catalog_value(row.get('price_php', ''))
                if not category or not service_name:
                    continue

                identity = service_identity(category, section, service_name)
                if identity in existing:
                    continue

                requirements = infer_service_requirements(category, section, service_name)
                db.session.add(ClinicService(
                    category=category,
                    section=section,
                    service_name=service_name,
                    price_php=price_php,
                    required_roles=service_catalog_json(requirements['roles']),
                    required_equipment=service_catalog_json(requirements['equipment']),
                    source_page=clean_service_catalog_value(row.get('source_page', '')),
                    is_active=True,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ))
                existing.add(identity)
                added_count += 1

        db.session.commit()
        return added_count

def replace_package_items(package, item_names):
    PackageItem.query.filter_by(package_id=package.id).delete()
    for index, item_name in enumerate(item_names, start=1):
        item_name = clean_service_catalog_value(item_name)
        if not item_name:
            continue
        linked_service = find_matching_clinic_service(item_name)
        roles, equipment = package_item_requirements(item_name, linked_service)
        db.session.add(PackageItem(
            package_id=package.id,
            service_id=linked_service.id if linked_service else None,
            item_name=item_name,
            item_order=index,
            required_roles=service_catalog_json(roles),
            required_equipment=service_catalog_json(equipment),
        ))

def import_accudetek_package_catalog(app, seed_only=False):
    with app.app_context():
        if seed_only and ClinicPackage.query.first():
            return 0

        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'accudetek_packages_scraped.csv')
        if not os.path.exists(csv_path):
            return 0

        grouped = {}
        with open(csv_path, newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                package_name = clean_service_catalog_value(row.get('package_name', ''))
                item_name = clean_service_catalog_value(row.get('included_service', ''))
                if not package_name or not item_name:
                    continue
                grouped.setdefault(package_name, []).append(item_name)

        existing = {
            package_identity(package.package_name)
            for package in ClinicPackage.query.all()
        }
        added_count = 0
        for package_name, item_names in grouped.items():
            if package_identity(package_name) in existing:
                continue
            package = ClinicPackage(
                package_name=package_name,
                price_php='',
                source_page='packages',
                is_active=True,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            db.session.add(package)
            db.session.flush()
            replace_package_items(package, item_names)
            existing.add(package_identity(package_name))
            added_count += 1

        db.session.commit()
        return added_count

def ensure_branch_service_rows(branch_id):
    if not branch_id:
        return 0

    service_ids = [service.id for service in ClinicService.query.all()]
    existing_ids = {
        row.service_id
        for row in BranchService.query.filter_by(branch_id=branch_id).all()
    }
    added_count = 0
    for service_id in service_ids:
        if service_id in existing_ids:
            continue
        db.session.add(BranchService(
            branch_id=branch_id,
            service_id=service_id,
            custom_price_php='',
            is_available=True,
            branch_notes='',
            updated_at=datetime.now(),
        ))
        added_count += 1
    if added_count:
        db.session.commit()
    return added_count

def ensure_branch_package_rows(branch_id):
    if not branch_id:
        return 0

    package_ids = [package.id for package in ClinicPackage.query.all()]
    existing_ids = {
        row.package_id
        for row in BranchPackage.query.filter_by(branch_id=branch_id).all()
    }
    added_count = 0
    for package_id in package_ids:
        if package_id in existing_ids:
            continue
        db.session.add(BranchPackage(
            branch_id=branch_id,
            package_id=package_id,
            custom_price_php='',
            is_available=True,
            branch_notes='',
            updated_at=datetime.now(),
        ))
        added_count += 1
    if added_count:
        db.session.commit()
    return added_count

# -------------------------------------------------------------
# Enhanced Data processing functions
# -------------------------------------------------------------
def build_training_frame(df):
    """Clean and encode the raw DataFrame."""
    df = df.copy()
    for col in ['consultation_date', 'age_group', 'gender', 'diagnosis', 'department', 'physician', 'consultation_type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=['consultation_date', 'diagnosis'])
    df = df.drop_duplicates()
    if 'consultation_date' in df.columns:
        df['consultation_date'] = pd.to_datetime(df['consultation_date'], errors='coerce')
        df = df.dropna(subset=['consultation_date'])
        df['month'] = df['consultation_date'].dt.month
        df['year'] = df['consultation_date'].dt.year

    if df.empty:
        return df

    if 'diagnosis' in df.columns:
        df['diagnosis_label'] = df['diagnosis']

    # Encode categorical columns
    for col in ['age_group', 'gender', 'department', 'physician', 'consultation_type']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    if 'diagnosis' in df.columns:
        df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'].astype(str))

    return df

def build_forecasting_training_frame(df):
    """
    Build a rich time‑series frame with:
    - Multiple lags (1, 3, 6, 12 months)
    - Rolling averages (3-month)
    - Demographic proportions (using global averages as fallback)
    """
    raw = df.copy()
    for col in ['consultation_date', 'age_group', 'gender', 'diagnosis']:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip()
    raw = raw.dropna(subset=['consultation_date', 'diagnosis'])
    raw['consultation_date'] = pd.to_datetime(raw['consultation_date'], errors='coerce')
    raw = raw.dropna(subset=['consultation_date'])
    raw['month'] = raw['consultation_date'].dt.month
    raw['year'] = raw['consultation_date'].dt.year

    # Group by (year, month, diagnosis) to get total case count
    grouped = (raw.groupby(['year', 'month', 'diagnosis'], as_index=False)
               .size()
               .rename(columns={'size': 'case_count'}))

    # Compute demographic proportions per (year, month, diagnosis)
    demo = raw.groupby(['year', 'month', 'diagnosis'], group_keys=False).apply(
        lambda g: pd.Series({
            'pct_adult': (g['age_group'] == 'Adult').mean(),
            'pct_child': (g['age_group'] == 'Child').mean(),
            'pct_senior': (g['age_group'] == 'Senior').mean(),
            'pct_male': (g['gender'] == 'Male').mean(),
            'pct_female': (g['gender'] == 'Female').mean()
        })
    ).reset_index()

    merged = grouped.merge(demo, on=['year', 'month', 'diagnosis'], how='left')
    
    # Instead of hardcoding 0.5, use global average for each column if missing
    for col in ['pct_adult', 'pct_child', 'pct_senior', 'pct_male', 'pct_female']:
        global_mean = merged[col].mean()
        merged[col] = merged[col].fillna(global_mean if not np.isnan(global_mean) else 0.5)

    # Sort to compute lags correctly
    merged = merged.sort_values(['diagnosis', 'year', 'month']).reset_index(drop=True)

    # Add season
    merged['season'] = (merged['month'] - 1) // 3 + 1

    # ---- Multiple Lags ----
    merged['lag_1'] = merged.groupby('diagnosis')['case_count'].shift(1)
    merged['lag_3'] = merged.groupby('diagnosis')['case_count'].shift(3)
    merged['lag_6'] = merged.groupby('diagnosis')['case_count'].shift(6)
    merged['lag_12'] = merged.groupby('diagnosis')['case_count'].shift(12)
    
    # Rolling averages (using lag_1 and lag_3)
    merged['rolling_mean_3'] = merged.groupby('diagnosis')['case_count'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )

    # For rows where lag_1 is NaN (first month of a diagnosis), fill with overall median
    overall_median = merged['case_count'].median()
    for col in ['lag_1', 'lag_3', 'lag_6', 'lag_12']:
        merged[col] = merged[col].fillna(merged.groupby('diagnosis')[col].transform('median'))
        merged[col] = merged[col].fillna(overall_median)  # ultimate fallback
        
    merged['rolling_mean_3'] = merged['rolling_mean_3'].fillna(merged['lag_1'])

    # Drop rows where we still don't have a lag (shouldn't happen after fill)
    merged = merged.dropna(subset=['lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_mean_3'])

    if merged.empty:
        return merged

    # Encode diagnosis
    le = LabelEncoder()
    merged['diagnosis'] = le.fit_transform(merged['diagnosis'])
    merged.attrs['diagnosis_encoder'] = le

    return merged

# -------------------------------------------------------------
# Enhanced Model training with Cross-Validation & Hyperparameter Tuning
# -------------------------------------------------------------
def train_and_evaluate_model(df):
    training_df = build_forecasting_training_frame(df)
    if training_df.empty:
        raise ValueError('Insufficient data for model training')

    diagnosis_encoder = training_df.attrs.get('diagnosis_encoder')
    if diagnosis_encoder is None:
        diagnosis_encoder = LabelEncoder()
        raw_diag = df['diagnosis'].astype(str).str.strip()
        diagnosis_encoder.fit(raw_diag)
    label_mapping = {i: name for i, name in enumerate(diagnosis_encoder.classes_)}

    # Expanded feature set
    feature_columns = [
        'month', 'year', 'season', 'diagnosis',
        'lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_mean_3',
        'pct_adult', 'pct_child', 'pct_senior', 'pct_male', 'pct_female'
    ]
    missing = [col for col in feature_columns if col not in training_df.columns]
    if missing:
        raise ValueError(f'Missing columns in training frame: {missing}')

    X = training_df[feature_columns]
    y = training_df['case_count']

    # ---- Time Series Cross-Validation ----
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = {'r2': [], 'mae': [], 'rmse': []}

    # ---- Hyperparameter Tuning with RandomizedSearchCV ----
    param_dist = {
        'n_estimators': [100, 150, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    
    # We use a subset of the data for tuning to keep it fast
    tune_size = min(1000, len(X))
    X_tune = X.iloc[:tune_size]
    y_tune = y.iloc[:tune_size]
    
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=min(3, tscv.n_splits),
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    random_search.fit(X_tune, y_tune)
    best_model = random_search.best_estimator_

    # Evaluate on the full dataset using TimeSeriesSplit to get stability metrics
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if len(X_train) < 5 or len(X_test) < 1:
            continue
            
        model = RandomForestRegressor(**random_search.best_params_, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        cv_scores['r2'].append(r2_score(y_test, preds))
        cv_scores['mae'].append(mean_absolute_error(y_test, preds))
        cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, preds)))

    # Train the final best model on ALL data for production
    final_model = RandomForestRegressor(**random_search.best_params_, random_state=42)
    final_model.fit(X, y)
    
    # Get predictions on the full training set to compute in-sample metrics
    full_preds = final_model.predict(X)
    metrics = {
        'r2_score': round(r2_score(y, full_preds), 4),
        'mae': round(mean_absolute_error(y, full_preds), 4),
        'mse': round(mean_squared_error(y, full_preds), 4),
        'rmse': round(np.sqrt(mean_squared_error(y, full_preds)), 4),
        'cv_r2_mean': round(np.mean(cv_scores['r2']), 4) if cv_scores['r2'] else None,
        'cv_r2_std': round(np.std(cv_scores['r2']), 4) if cv_scores['r2'] else None,
        'best_params': random_search.best_params_
    }

    return final_model, metrics, feature_columns, label_mapping

# -------------------------------------------------------------
# Forecast generation with correct lag retrieval
# -------------------------------------------------------------
def generate_forecast_for_month(model, feature_columns, label_mapping, df, target_month, target_year):
    """
    Generate predictions for a specific month by correctly looking up
    lag_1, lag_3, lag_6, lag_12 from the historical data.
    """
    training_df = build_forecasting_training_frame(df)
    if training_df.empty:
        return []

    # For each diagnosis, get the row for the month just before the target
    # We need the state of the system at target_month - 1 to compute lags.
    prev_month = target_month - 1
    prev_year = target_year
    if prev_month == 0:
        prev_month = 12
        prev_year -= 1

    # We need lag_12: target_month - 12
    lag12_month = target_month - 12
    lag12_year = target_year
    if lag12_month <= 0:
        lag12_month += 12
        lag12_year -= 1

    # Get historical data for each diagnosis at the required lag periods
    # We'll build a lookup table for each diagnosis
    historical = training_df.copy()
    historical['year_month'] = historical['year'] * 100 + historical['month']

    prev_key = prev_year * 100 + prev_month
    lag3_key = prev_year * 100 + (prev_month - 2) if prev_month >= 3 else (prev_year - 1) * 100 + (prev_month + 9)
    lag6_key = prev_year * 100 + (prev_month - 5) if prev_month >= 6 else (prev_year - 1) * 100 + (prev_month + 6)
    lag12_key = lag12_year * 100 + lag12_month

    predictions = []
    diagnoses = training_df['diagnosis'].unique()

    for diag in diagnoses:
        diag_data = historical[historical['diagnosis'] == diag]
        
        # Get the most recent row for this diagnosis (to get demographics and basic info)
        latest_row = diag_data.sort_values(['year', 'month']).iloc[-1]
        
        # Get specific lag values
        lag_1_val = diag_data[diag_data['year_month'] == prev_key]['case_count'].values
        lag_3_val = diag_data[diag_data['year_month'] == lag3_key]['case_count'].values
        lag_6_val = diag_data[diag_data['year_month'] == lag6_key]['case_count'].values
        lag_12_val = diag_data[diag_data['year_month'] == lag12_key]['case_count'].values
        
        # Fallback: if specific lag not found, use the most recent available
        lag_1_val = lag_1_val[0] if len(lag_1_val) > 0 else latest_row['lag_1']
        lag_3_val = lag_3_val[0] if len(lag_3_val) > 0 else latest_row['lag_3']
        lag_6_val = lag_6_val[0] if len(lag_6_val) > 0 else latest_row['lag_6']
        lag_12_val = lag_12_val[0] if len(lag_12_val) > 0 else latest_row['lag_12']
        rolling_mean_3_val = (lag_1_val + lag_3_val) / 2  # approximation

        pred_row = {
            'month': target_month,
            'year': target_year,
            'season': (target_month - 1) // 3 + 1,
            'diagnosis': latest_row['diagnosis'],
            'lag_1': lag_1_val,
            'lag_3': lag_3_val,
            'lag_6': lag_6_val,
            'lag_12': lag_12_val,
            'rolling_mean_3': rolling_mean_3_val,
            'pct_adult': latest_row['pct_adult'],
            'pct_child': latest_row['pct_child'],
            'pct_senior': latest_row['pct_senior'],
            'pct_male': latest_row['pct_male'],
            'pct_female': latest_row['pct_female'],
        }
        predictions.append(pred_row)

    pred_df = pd.DataFrame(predictions)
    X_pred = pred_df[feature_columns]
    preds = model.predict(X_pred)

    results = []
    for i, diag_enc in enumerate(pred_df['diagnosis']):
        diag_name = label_mapping.get(diag_enc, f"Unknown_{diag_enc}")
        results.append((diag_name, max(0, round(preds[i]))))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def generate_forecast_for_specific_month(df, target_month, target_year):
    """Wrapper to generate forecast for a specific month."""
    if df.empty:
        return None, None, None
    try:
        model, metrics, feature_cols, label_mapping = train_and_evaluate_model(df)
        forecast = generate_forecast_for_month(model, feature_cols, label_mapping, df, target_month, target_year)
        total_pred = sum(count for _, count in forecast)
        return total_pred, forecast, metrics
    except ValueError as exc:
        if 'months of consultation data' not in str(exc):
            traceback.print_exc()
        return None, None, None
    except Exception:
        traceback.print_exc()
        return None, None, None

def group_common_forecast_targets(raw, max_targets=MAX_RF_TARGET_LABELS):
    """Keep common cases/services as model targets and bucket rare service labels."""
    original_count = int(raw['diagnosis'].nunique()) if 'diagnosis' in raw.columns else 0
    if original_count <= max_targets:
        return raw, {
            'original_diagnosis_count': original_count,
            'diagnosis_count_after_grouping': original_count,
            'grouped_rare_target_count': 0,
            'forecast_target_limit': max_targets,
        }

    common_targets = (
        raw['diagnosis']
        .value_counts()
        .head(max_targets)
        .index
    )
    grouped_raw = raw.copy()
    grouped_raw['diagnosis'] = grouped_raw['diagnosis'].where(
        grouped_raw['diagnosis'].isin(common_targets),
        OTHER_RF_TARGET_LABEL,
    )
    grouped_count = int(grouped_raw['diagnosis'].nunique())
    return grouped_raw, {
        'original_diagnosis_count': original_count,
        'diagnosis_count_after_grouping': grouped_count,
        'grouped_rare_target_count': max(0, original_count - max_targets),
        'forecast_target_limit': max_targets,
    }

def build_forecasting_training_frame(df):
    """Build one diagnosis-age-gender-month row per observed month, with honest lag features."""
    raw = df.copy()
    for col in ['consultation_date', 'diagnosis', 'age_group', 'gender']:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip()
    raw = raw.dropna(subset=['consultation_date', 'diagnosis'])
    raw['consultation_date'] = pd.to_datetime(raw['consultation_date'], errors='coerce')
    raw = raw.dropna(subset=['consultation_date'])
    raw = raw[raw['diagnosis'] != '']
    if 'age_group' not in raw.columns:
        raw['age_group'] = 'Unknown'
    if 'gender' not in raw.columns:
        raw['gender'] = 'Unknown'
    raw['age_group'] = raw['age_group'].replace(['', 'nan', 'None', 'NaT'], 'Unknown').fillna('Unknown')
    raw['gender'] = raw['gender'].replace(['', 'nan', 'None', 'NaT'], 'Unknown').fillna('Unknown')
    if raw.empty:
        return pd.DataFrame()

    raw, target_metadata = group_common_forecast_targets(raw)
    raw['period'] = raw['consultation_date'].dt.to_period('M')
    diagnoses = sorted(raw['diagnosis'].unique())
    age_groups = sorted(raw['age_group'].unique())
    genders = sorted(raw['gender'].unique())
    periods = pd.period_range(raw['period'].min(), raw['period'].max(), freq='M')
    full_index = pd.MultiIndex.from_product(
        [diagnoses, age_groups, genders, periods],
        names=['diagnosis', 'age_group', 'gender', 'period']
    )

    grouped = raw.groupby(['diagnosis', 'age_group', 'gender', 'period']).size().rename('case_count')
    merged = grouped.reindex(full_index, fill_value=0).reset_index()
    merged['period_start'] = merged['period'].dt.to_timestamp()
    merged['year'] = merged['period_start'].dt.year
    merged['month'] = merged['period_start'].dt.month
    merged['season'] = (merged['month'] - 1) // 3 + 1
    merged['time_index'] = (merged['year'] - merged['year'].min()) * 12 + merged['month']
    segment_columns = ['diagnosis', 'age_group', 'gender']
    merged = merged.sort_values(segment_columns + ['period']).reset_index(drop=True)

    segment_group = merged.groupby(segment_columns)['case_count']
    merged['lag_1'] = segment_group.shift(1)
    merged['lag_2'] = segment_group.shift(2)
    merged['lag_3'] = segment_group.shift(3)
    merged['lag_6'] = segment_group.shift(6)
    merged['lag_12'] = segment_group.shift(12)
    merged['rolling_mean_3'] = merged.groupby(segment_columns)['case_count'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )
    merged['rolling_mean_6'] = merged.groupby(segment_columns)['case_count'].transform(
        lambda x: x.rolling(6, min_periods=1).mean().shift(1)
    )
    merged['rolling_std_3'] = merged.groupby(segment_columns)['case_count'].transform(
        lambda x: x.rolling(3, min_periods=2).std().shift(1)
    )
    merged['trend_1'] = merged['lag_1'] - merged['lag_2']

    history_columns = [
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3', 'trend_1'
    ]
    merged[history_columns] = merged[history_columns].fillna(0)

    le = LabelEncoder()
    merged['diagnosis_code'] = le.fit_transform(merged['diagnosis'])
    diagnosis_dummies = pd.get_dummies(merged['diagnosis'], prefix='diagnosis', dtype=int)
    age_dummies = pd.get_dummies(merged['age_group'], prefix='age_group', dtype=int)
    gender_dummies = pd.get_dummies(merged['gender'], prefix='gender', dtype=int)
    merged = pd.concat([merged, diagnosis_dummies, age_dummies, gender_dummies], axis=1)
    merged.attrs['diagnosis_encoder'] = le
    merged.attrs.update(target_metadata)
    return merged

def _safe_r2(y_true, preds):
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0):
        return None
    return r2_score(y_true, preds)

def _regression_metrics(y_true, preds):
    mse = mean_squared_error(y_true, preds)
    score = _safe_r2(y_true, preds)
    return {
        'r2': round(score, 4) if score is not None else None,
        'mae': round(mean_absolute_error(y_true, preds), 4),
        'mse': round(mse, 4),
        'rmse': round(np.sqrt(mse), 4),
    }

def _model_verdict(metrics):
    validation_r2 = metrics.get('validation_r2')
    improvement = metrics.get('improvement_vs_baseline_pct')
    if validation_r2 is not None and validation_r2 >= 0.5 and improvement is not None and improvement > 0:
        return 'Acceptable'
    if validation_r2 is not None and validation_r2 > 0 and improvement is not None and improvement > 0:
        return 'Needs more data, but better than baseline'
    return 'Not acceptable yet'

def build_non_demographic_training_frame(df):
    """Build one diagnosis-month row per observed month without age or gender features."""
    raw = df.copy()
    for col in ['consultation_date', 'diagnosis']:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip()
    raw = raw.dropna(subset=['consultation_date', 'diagnosis'])
    raw['consultation_date'] = pd.to_datetime(raw['consultation_date'], errors='coerce')
    raw = raw.dropna(subset=['consultation_date'])
    raw = raw[raw['diagnosis'] != '']
    if raw.empty:
        return pd.DataFrame()

    raw, target_metadata = group_common_forecast_targets(raw)
    raw['period'] = raw['consultation_date'].dt.to_period('M')
    diagnoses = sorted(raw['diagnosis'].unique())
    periods = pd.period_range(raw['period'].min(), raw['period'].max(), freq='M')
    full_index = pd.MultiIndex.from_product(
        [diagnoses, periods],
        names=['diagnosis', 'period']
    )

    grouped = raw.groupby(['diagnosis', 'period']).size().rename('case_count')
    merged = grouped.reindex(full_index, fill_value=0).reset_index()
    merged['period_start'] = merged['period'].dt.to_timestamp()
    merged['year'] = merged['period_start'].dt.year
    merged['month'] = merged['period_start'].dt.month
    merged['season'] = (merged['month'] - 1) // 3 + 1
    merged['time_index'] = (merged['year'] - merged['year'].min()) * 12 + merged['month']
    merged = merged.sort_values(['diagnosis', 'period']).reset_index(drop=True)

    segment_group = merged.groupby(['diagnosis'])['case_count']
    merged['lag_1'] = segment_group.shift(1)
    merged['lag_2'] = segment_group.shift(2)
    merged['lag_3'] = segment_group.shift(3)
    merged['lag_6'] = segment_group.shift(6)
    merged['lag_12'] = segment_group.shift(12)
    merged['rolling_mean_3'] = merged.groupby(['diagnosis'])['case_count'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )
    merged['rolling_mean_6'] = merged.groupby(['diagnosis'])['case_count'].transform(
        lambda x: x.rolling(6, min_periods=1).mean().shift(1)
    )
    merged['rolling_std_3'] = merged.groupby(['diagnosis'])['case_count'].transform(
        lambda x: x.rolling(3, min_periods=2).std().shift(1)
    )
    merged['trend_1'] = merged['lag_1'] - merged['lag_2']

    history_columns = [
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3', 'trend_1'
    ]
    merged[history_columns] = merged[history_columns].fillna(0)
    diagnosis_dummies = pd.get_dummies(merged['diagnosis'], prefix='diagnosis', dtype=int)
    merged = pd.concat([merged, diagnosis_dummies], axis=1)
    merged.attrs.update(target_metadata)
    return merged

def evaluate_model_without_demographics(df):
    training_df = build_non_demographic_training_frame(df)
    if training_df.empty:
        return None

    diagnosis_feature_columns = sorted([
        col for col in training_df.columns
        if col.startswith('diagnosis_')
    ])
    feature_columns = [
        'month', 'season', 'time_index',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3', 'trend_1',
    ] + diagnosis_feature_columns

    training_df = training_df.sort_values(['period_start', 'diagnosis']).reset_index(drop=True)
    unique_periods = sorted(training_df['period'].unique())
    if len(unique_periods) < 8:
        return None

    holdout_months = min(6, max(2, len(unique_periods) // 4))
    validation_periods = unique_periods[-holdout_months:]
    train_df = training_df[~training_df['period'].isin(validation_periods)]
    validation_df = training_df[training_df['period'].isin(validation_periods)]

    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [4, 6, 8, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    segment_count = max(1, training_df[['diagnosis']].drop_duplicates().shape[0])
    cv_splits = min(3, max(2, len(train_df) // max(1, segment_count * 3)))
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=8,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    random_search.fit(train_df[feature_columns], train_df['case_count'])

    validation_preds = random_search.best_estimator_.predict(validation_df[feature_columns])
    validation_metrics = _regression_metrics(validation_df['case_count'], validation_preds)
    baseline_metrics = _regression_metrics(validation_df['case_count'], validation_df['lag_1'])
    improvement = None
    if baseline_metrics['mae']:
        improvement = round(((baseline_metrics['mae'] - validation_metrics['mae']) / baseline_metrics['mae']) * 100, 2)

    return {
        'training_grain': 'diagnosis by month',
        'training_rows': int(len(training_df)),
        'validation_r2': validation_metrics['r2'],
        'validation_mae': validation_metrics['mae'],
        'validation_rmse': validation_metrics['rmse'],
        'baseline_mae': baseline_metrics['mae'],
        'improvement_vs_baseline_pct': improvement,
        'training_months': len(unique_periods) - holdout_months,
        'validation_months': holdout_months,
        'validation_period_start': str(validation_periods[0]),
        'validation_period_end': str(validation_periods[-1]),
    }

def train_and_evaluate_model(df):
    training_df = build_forecasting_training_frame(df)
    if training_df.empty:
        raise ValueError('Insufficient data for model training')

    diagnosis_encoder = training_df.attrs.get('diagnosis_encoder')
    label_mapping = {i: name for i, name in enumerate(diagnosis_encoder.classes_)}
    diagnosis_feature_columns = sorted([
        col for col in training_df.columns
        if col.startswith('diagnosis_') and col != 'diagnosis_code'
    ])
    demographic_feature_columns = sorted([
        col for col in training_df.columns
        if col.startswith('age_group_') or col.startswith('gender_')
    ])
    feature_columns = [
        'month', 'season', 'time_index',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3', 'trend_1',
    ] + diagnosis_feature_columns + demographic_feature_columns

    training_df = training_df.sort_values(['period_start', 'diagnosis', 'age_group', 'gender']).reset_index(drop=True)
    unique_periods = sorted(training_df['period'].unique())
    if len(unique_periods) < 8:
        raise ValueError('At least 8 months of consultation data are needed for reliable validation')

    holdout_months = min(6, max(2, len(unique_periods) // 4))
    validation_periods = unique_periods[-holdout_months:]
    train_df = training_df[~training_df['period'].isin(validation_periods)]
    validation_df = training_df[training_df['period'].isin(validation_periods)]

    X_train = train_df[feature_columns]
    y_train = train_df['case_count']
    X_validation = validation_df[feature_columns]
    y_validation = validation_df['case_count']

    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [4, 6, 8, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    segment_count = max(1, training_df[['diagnosis', 'age_group', 'gender']].drop_duplicates().shape[0])
    cv_splits = min(3, max(2, len(train_df) // max(1, segment_count * 3)))
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=8,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    random_search.fit(X_train, y_train)

    validation_preds = random_search.best_estimator_.predict(X_validation)
    validation_metrics = _regression_metrics(y_validation, validation_preds)
    baseline_metrics = _regression_metrics(y_validation, validation_df['lag_1'])

    cv_scores = {'r2': [], 'mae': [], 'rmse': []}
    for train_idx, test_idx in TimeSeriesSplit(n_splits=cv_splits).split(training_df):
        fold_train = training_df.iloc[train_idx]
        fold_test = training_df.iloc[test_idx]
        model = RandomForestRegressor(**random_search.best_params_, random_state=42)
        model.fit(fold_train[feature_columns], fold_train['case_count'])
        preds = model.predict(fold_test[feature_columns])
        fold_metrics = _regression_metrics(fold_test['case_count'], preds)
        if fold_metrics['r2'] is not None:
            cv_scores['r2'].append(fold_metrics['r2'])
        cv_scores['mae'].append(fold_metrics['mae'])
        cv_scores['rmse'].append(fold_metrics['rmse'])

    final_model = RandomForestRegressor(**random_search.best_params_, random_state=42)
    final_model.fit(training_df[feature_columns], training_df['case_count'])
    training_metrics = _regression_metrics(
        training_df['case_count'],
        final_model.predict(training_df[feature_columns])
    )

    improvement = None
    if baseline_metrics['mae']:
        improvement = round(((baseline_metrics['mae'] - validation_metrics['mae']) / baseline_metrics['mae']) * 100, 2)

    metrics = {
        'total_training_rows': int(len(training_df)),
        'total_segments': int(training_df[['diagnosis', 'age_group', 'gender']].drop_duplicates().shape[0]),
        'diagnosis_count': int(training_df['diagnosis'].nunique()),
        'original_diagnosis_count': int(training_df.attrs.get('original_diagnosis_count', training_df['diagnosis'].nunique())),
        'grouped_rare_target_count': int(training_df.attrs.get('grouped_rare_target_count', 0)),
        'forecast_target_limit': int(training_df.attrs.get('forecast_target_limit', MAX_RF_TARGET_LABELS)),
        'age_group_count': int(training_df['age_group'].nunique()),
        'gender_count': int(training_df['gender'].nunique()),
        'data_period_start': str(unique_periods[0]),
        'data_period_end': str(unique_periods[-1]),
        'r2_score': validation_metrics['r2'],
        'mae': validation_metrics['mae'],
        'mse': validation_metrics['mse'],
        'rmse': validation_metrics['rmse'],
        'validation_r2': validation_metrics['r2'],
        'validation_mae': validation_metrics['mae'],
        'validation_mse': validation_metrics['mse'],
        'validation_rmse': validation_metrics['rmse'],
        'training_r2': training_metrics['r2'],
        'training_mae': training_metrics['mae'],
        'training_rmse': training_metrics['rmse'],
        'baseline_mae': baseline_metrics['mae'],
        'baseline_rmse': baseline_metrics['rmse'],
        'baseline_r2': baseline_metrics['r2'],
        'improvement_vs_baseline_pct': improvement,
        'cv_r2_mean': round(np.mean(cv_scores['r2']), 4) if cv_scores['r2'] else None,
        'cv_r2_std': round(np.std(cv_scores['r2']), 4) if cv_scores['r2'] else None,
        'cv_mae_mean': round(np.mean(cv_scores['mae']), 4) if cv_scores['mae'] else None,
        'cv_rmse_mean': round(np.mean(cv_scores['rmse']), 4) if cv_scores['rmse'] else None,
        'training_months': len(unique_periods) - holdout_months,
        'validation_months': holdout_months,
        'validation_period_start': str(validation_periods[0]),
        'validation_period_end': str(validation_periods[-1]),
        'best_params': random_search.best_params_,
    }
    try:
        metrics['model_b_without_demographics'] = evaluate_model_without_demographics(df)
    except Exception:
        traceback.print_exc()
        metrics['model_b_without_demographics'] = None
    metrics['model_verdict'] = _model_verdict(metrics)
    return final_model, metrics, feature_columns, label_mapping

def generate_forecast_for_month(model, feature_columns, label_mapping, df, target_month, target_year):
    training_df = build_forecasting_training_frame(df)
    if training_df.empty:
        generate_forecast_for_month.last_demographic_forecast = {
            'age_group': [],
            'gender': [],
            'segments': [],
        }
        return []

    target_period = pd.Period(year=target_year, month=target_month, freq='M')
    next_rows = []
    diagnosis_columns = [
        col for col in feature_columns
        if col.startswith('diagnosis_') and col != 'diagnosis_code'
    ]
    age_columns = [col for col in feature_columns if col.startswith('age_group_')]
    gender_columns = [col for col in feature_columns if col.startswith('gender_')]

    for (diagnosis, age_group, gender), segment_data in training_df.groupby(['diagnosis', 'age_group', 'gender']):
        history = segment_data.sort_values('period').copy()
        counts = history.set_index('period')['case_count']

        def lag_value(months_back):
            period = target_period - months_back
            return counts.get(period, 0)

        lag_1 = lag_value(1)
        lag_2 = lag_value(2)
        recent_3 = [lag_value(i) for i in range(1, 4)]
        recent_6 = [lag_value(i) for i in range(1, 7)]
        row = {
            'month': target_month,
            'season': (target_month - 1) // 3 + 1,
            'time_index': int((target_period.year - training_df['year'].min()) * 12 + target_period.month),
            'lag_1': lag_1,
            'lag_2': lag_2,
            'lag_3': lag_value(3),
            'lag_6': lag_value(6),
            'lag_12': lag_value(12),
            'rolling_mean_3': float(np.mean(recent_3)),
            'rolling_mean_6': float(np.mean(recent_6)),
            'rolling_std_3': float(np.std(recent_3, ddof=1)) if len(recent_3) > 1 else 0,
            'trend_1': lag_1 - lag_2,
            'diagnosis_code': int(history['diagnosis_code'].iloc[-1]),
            'diagnosis': diagnosis,
            'age_group': age_group,
            'gender': gender,
        }
        for col in diagnosis_columns:
            row[col] = 1 if col == f'diagnosis_{diagnosis}' else 0
        for col in age_columns:
            row[col] = 1 if col == f'age_group_{age_group}' else 0
        for col in gender_columns:
            row[col] = 1 if col == f'gender_{gender}' else 0
        next_rows.append(row)

    pred_df = pd.DataFrame(next_rows)
    preds = model.predict(pred_df[feature_columns])
    pred_df['predicted_cases'] = [max(0, round(value)) for value in preds]

    diagnosis_totals = (
        pred_df.groupby('diagnosis')['predicted_cases']
        .sum()
        .sort_values(ascending=False)
    )
    age_totals = (
        pred_df.groupby('age_group')['predicted_cases']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'predicted_cases': 'forecasted_cases'})
        .to_dict('records')
    )
    gender_totals = (
        pred_df.groupby('gender')['predicted_cases']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'predicted_cases': 'forecasted_cases'})
        .to_dict('records')
    )
    segment_rows = (
        pred_df[pred_df['predicted_cases'] > 0]
        .sort_values('predicted_cases', ascending=False)
        [['diagnosis', 'age_group', 'gender', 'predicted_cases']]
        .rename(columns={'predicted_cases': 'forecasted_cases'})
        .head(15)
        .to_dict('records')
    )

    results = [(diagnosis, int(count)) for diagnosis, count in diagnosis_totals.items()]
    generate_forecast_for_month.last_demographic_forecast = {
        'age_group': age_totals,
        'gender': gender_totals,
        'segments': segment_rows,
    }
    return results

def write_training_report(report_path, metrics):
    def month_label(value):
        try:
            return datetime.strptime(str(value), '%Y-%m').strftime('%B %Y')
        except Exception:
            return str(value)

    def fmt(value, suffix=''):
        if value is None:
            return 'N/A'
        return f'{value}{suffix}'

    model_b = metrics.get('model_b_without_demographics') or {}
    best_params = metrics.get('best_params') or {}

    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('Smart Healthcare Clinic Management - Model Training and Evaluation Report\n')
        handle.write('=======================================================================\n')
        handle.write('\n1. Model Objective\n')
        handle.write('\n')
        handle.write('The objective of this Random Forest regression model is to forecast the monthly consultation demand at Accudetek Health Diagnostics. ')
        handle.write('The model predicts the expected consultation volume by diagnosis, age group, and gender to provide decision-support information for clinic administrators in staff planning, resource preparedness, service readiness, and operational decision-making.\n')

        handle.write('\n2. Dataset and Feature Preparation\n')
        handle.write('\n')
        handle.write(f"Dataset Period: {month_label(metrics.get('data_period_start', 'N/A'))} - {month_label(metrics.get('data_period_end', 'N/A'))}\n\n")
        handle.write(f"Training Frame Rows: {metrics.get('total_training_rows', 'N/A')}\n\n")
        handle.write(f"Diagnosis-Age Group-Gender Segments: {metrics.get('total_segments', 'N/A')}\n\n")
        handle.write(f"Forecast Targets Covered: {metrics.get('diagnosis_count', 'N/A')}\n\n")
        if metrics.get('grouped_rare_target_count', 0):
            handle.write(f"Original Diagnosis/Service Labels: {metrics.get('original_diagnosis_count', 'N/A')}\n\n")
            handle.write(f"Rare Labels Grouped as Other Services: {metrics.get('grouped_rare_target_count', 'N/A')}\n\n")
            handle.write('Rare service labels were grouped only for Random Forest forecasting so the model remains focused on common consultation and service demand. The consultation records stored in the system remain unchanged.\n\n')
        handle.write(f"Age Groups Covered: {metrics.get('age_group_count', 'N/A')}\n\n")
        handle.write(f"Gender Categories Covered: {metrics.get('gender_count', 'N/A')}\n\n")
        handle.write('Training Granularity: Diagnosis x Age Group x Gender x Month\n\n')
        handle.write('Predictive Features\n\n')
        handle.write('The model was trained using the following features:\n\n')
        for feature in [
            'Diagnosis',
            'Age Group',
            'Gender',
            'Monthly Seasonality',
            'Time Index',
            'One-, Two-, Three-, Six-, and Twelve-Month Lag Values',
            'Rolling Averages',
            'Recent Trend Indicators',
        ]:
            handle.write(f'{feature}\n')
        handle.write('\n')
        handle.write('The consultation records were transformed into monthly consultation counts for each diagnosis-age group-gender segment. ')
        handle.write('Feature engineering was then performed by generating lag values, rolling averages, seasonal indicators, and trend features to enable the Random Forest model to learn recurring consultation patterns and temporal behavior.\n')

        handle.write('\n3. Model Validation Method\n')
        handle.write('\n')
        handle.write('To simulate real-world forecasting, the model used a time-based validation approach rather than a random train-test split.\n\n')
        handle.write('Earlier months were used for model training, while the most recent months were reserved for validation.\n\n')
        handle.write('This prevents information leakage and reflects the practical forecasting scenario in which future consultation demand must be predicted using only historical records.\n\n')
        handle.write(f"Training Months: {metrics.get('training_months')}\n\n")
        handle.write(f"Validation Months: {metrics.get('validation_months')}\n\n")
        handle.write(f"Validation Period: {month_label(metrics.get('validation_period_start'))} - {month_label(metrics.get('validation_period_end'))}\n\n")
        handle.write('Baseline Forecast\n\n')
        handle.write("The model was compared against a naive forecasting approach that assumes the next month's consultation count will be the same as the previous month's value.\n\n")
        handle.write('This baseline provides a simple benchmark for evaluating whether the Random Forest model learns meaningful consultation patterns beyond historical repetition.\n')

        handle.write('\n4. Demographic Feature Comparison\n')
        handle.write('\n')
        handle.write('Two Random Forest models were evaluated.\n\n')
        handle.write('Model A - Proposed Model\n\n')
        handle.write('Features Included:\n\n')
        for feature in ['Diagnosis', 'Age Group', 'Gender', 'Monthly Lag Values', 'Rolling Averages', 'Trend Features', 'Seasonal Indicators']:
            handle.write(f'{feature}\n')
        handle.write('\nValidation Results\n\n')
        handle.write(f"Validation R2: {fmt(metrics.get('validation_r2'))}\n")
        handle.write(f"Validation MAE: {fmt(metrics.get('validation_mae'))}\n")
        handle.write(f"Validation RMSE: {fmt(metrics.get('validation_rmse'))}\n")
        handle.write(f"Improvement over Baseline: {fmt(metrics.get('improvement_vs_baseline_pct'), '%')}\n")
        handle.write('\nModel B - Comparison Model\n\n')
        handle.write('Features Included\n\n')
        for feature in ['Diagnosis', 'Monthly Lag Values', 'Rolling Averages', 'Trend Features', 'Seasonal Indicators']:
            handle.write(f'{feature}\n')
        handle.write('\n(Age Group and Gender excluded.)\n\n')
        handle.write('Validation Results\n\n')
        handle.write(f"Validation R2: {fmt(model_b.get('validation_r2'))}\n")
        handle.write(f"Validation MAE: {fmt(model_b.get('validation_mae'))}\n")
        handle.write(f"Validation RMSE: {fmt(model_b.get('validation_rmse'))}\n")
        handle.write(f"Improvement over Baseline: {fmt(model_b.get('improvement_vs_baseline_pct'), '%')}\n")
        handle.write('\nInterpretation\n\n')
        if model_b:
            mae_delta = round(model_b.get('validation_mae', 0) - metrics.get('validation_mae', 0), 4)
            if mae_delta > 0:
                handle.write(f'Model A generated lower prediction error than Model B by {mae_delta} MAE points while producing detailed forecasts by diagnosis, age group, and gender.\n\n')
            elif mae_delta < 0:
                handle.write(f'Model B generated lower MAE by {abs(mae_delta)} points, but it performed a simpler diagnosis-level forecasting task and cannot produce age-group or gender-specific forecast outputs.\n\n')
            else:
                handle.write('Model A and Model B produced the same MAE on this validation split, but Model A provides demographic-specific forecast outputs required by the study objectives.\n\n')
            handle.write('Since the two models operate at different levels of granularity, their R2 values should not be interpreted as a direct measure of superiority.\n\n')
            handle.write('Model A was selected because it provides demographic-specific predictions that better support clinic planning, staffing, and resource allocation.\n')
        else:
            handle.write('Model B metrics were not available in this report. Run model retraining to generate the non-demographic comparison results.\n')

        handle.write('\n5. Model Performance\n')
        handle.write('\n')
        handle.write(f"Model Verdict: {metrics.get('model_verdict', 'Unknown')}\n")
        handle.write('\nMetric\tResult\n')
        handle.write(f"Validation R2\t{fmt(metrics.get('validation_r2'))}\n")
        handle.write(f"Validation MAE\t{fmt(metrics.get('validation_mae'))}\n")
        handle.write(f"Validation RMSE\t{fmt(metrics.get('validation_rmse'))}\n")
        handle.write(f"Baseline MAE\t{fmt(metrics.get('baseline_mae'))}\n")
        handle.write(f"Baseline RMSE\t{fmt(metrics.get('baseline_rmse'))}\n")
        handle.write(f"Improvement over Baseline\t{fmt(metrics.get('improvement_vs_baseline_pct'), '%')}\n")
        handle.write(f"Cross-Validation R2 Mean\t{fmt(metrics.get('cv_r2_mean'))} +/- {fmt(metrics.get('cv_r2_std'))}\n")
        handle.write(f"Cross-Validation MAE Mean\t{fmt(metrics.get('cv_mae_mean'))}\n")
        handle.write(f"Training R2 (Reference Only)\t{fmt(metrics.get('training_r2'))}\n")

        handle.write('\n6. Performance Metric Interpretation\n')
        handle.write('Validation R2\n\n')
        handle.write('Measures how well the model explains the variation in unseen validation data.\n\n')
        handle.write('Higher values indicate stronger predictive performance on future consultation records.\n\n')
        handle.write('Mean Absolute Error (MAE)\n\n')
        handle.write('Measures the average prediction error in consultation cases.\n\n')
        handle.write(f"An MAE of {fmt(metrics.get('validation_mae'))} means that, on average, the predicted consultation volume differs from the actual value by approximately {round(metrics.get('validation_mae', 0)) if metrics.get('validation_mae') is not None else 'N/A'} consultation cases per diagnosis-age group-gender segment.\n\n")
        handle.write('Root Mean Squared Error (RMSE)\n\n')
        handle.write('Measures the magnitude of prediction errors while assigning greater weight to larger mistakes.\n\n')
        handle.write('Lower RMSE values indicate better overall forecasting accuracy.\n\n')
        handle.write('Baseline Improvement\n\n')
        handle.write("This measures how much the Random Forest model reduced prediction error compared with simply using the previous month's consultation count as the forecast.\n\n")
        handle.write(f"A {fmt(metrics.get('improvement_vs_baseline_pct'), '%')} improvement demonstrates that the model learned meaningful consultation patterns rather than merely repeating historical observations.\n")

        handle.write('\n7. Final Interpretation\n')
        handle.write('\n')
        handle.write('Based on the validation results, the Random Forest regression model demonstrated acceptable performance for short-term consultation forecasting.\n\n')
        handle.write(f"The model achieved a validation R2 of {fmt(metrics.get('validation_r2'))}, indicating that it explained approximately {round(metrics.get('validation_r2', 0) * 100) if metrics.get('validation_r2') is not None else 'N/A'}% of the variation in unseen consultation demand.\n\n")
        handle.write(f"Its Mean Absolute Error of {fmt(metrics.get('validation_mae'))} indicates that the average forecasting error was approximately {round(metrics.get('validation_mae', 0)) if metrics.get('validation_mae') is not None else 'N/A'} consultation cases per diagnosis-age group-gender month segment.\n\n")
        handle.write(f"Compared with the naive last-month forecasting baseline, the proposed model reduced prediction error by {fmt(metrics.get('improvement_vs_baseline_pct'), '%')}, demonstrating that it successfully learned recurring consultation patterns rather than simply repeating previous observations.\n\n")
        handle.write('These findings support the feasibility of the proposed Smart Healthcare Clinic Management System as a decision-support tool for forecasting consultation trends, supporting staff planning, improving service preparedness, and assisting operational decision-making at Accudetek Health Diagnostics.\n')

        handle.write('\n8. Best Random Forest Parameters\n')
        handle.write(f"n_estimators      : {best_params.get('n_estimators', 'N/A')}\n")
        handle.write(f"max_depth         : {best_params.get('max_depth', 'N/A')}\n")
        handle.write(f"min_samples_split : {best_params.get('min_samples_split', 'N/A')}\n")
        handle.write(f"min_samples_leaf  : {best_params.get('min_samples_leaf', 'N/A')}\n")
        handle.write(f"max_features      : {best_params.get('max_features', 'N/A')}\n")

        handle.write('\n9. Study Limitation\n')
        handle.write('\n')
        handle.write("Note: The reported model performance was obtained using the simulated consultation dataset developed for this study. ")
        handle.write("Although the system demonstrates the technical feasibility of consultation forecasting, deployment in an actual clinical environment would require retraining and validation using Accudetek Health Diagnostics' historical consultation records to verify predictive performance under real-world conditions.\n")

def _pdf_escape(value):
    return str(value).replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

def _build_simple_pdf(title, lines):
    max_lines_per_page = 46
    pages = []
    current = [title, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', '']
    for line in lines:
        if len(current) >= max_lines_per_page:
            pages.append(current)
            current = [title + ' (continued)', '']
        current.append(str(line))
    pages.append(current)

    objects = []
    page_refs = []
    font_obj = 3
    next_obj = 4

    for page_lines in pages:
        content_lines = ['BT', '/F1 11 Tf', '50 790 Td', '14 TL']
        first = True
        for line in page_lines:
            text = _pdf_escape(line[:105])
            if first:
                content_lines.append(f'({text}) Tj')
                first = False
            else:
                content_lines.append(f'T* ({text}) Tj')
        content_lines.append('ET')
        content = '\n'.join(content_lines).encode('latin-1', errors='replace')

        content_obj = next_obj
        page_obj = next_obj + 1
        next_obj += 2
        objects.append((content_obj, b'<< /Length ' + str(len(content)).encode('ascii') + b' >>\nstream\n' + content + b'\nendstream'))
        objects.append((page_obj, f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_obj} 0 R >>'.encode('ascii')))
        page_refs.append(f'{page_obj} 0 R')

    objects.insert(0, (1, b'<< /Type /Catalog /Pages 2 0 R >>'))
    objects.insert(1, (2, f'<< /Type /Pages /Kids [{" ".join(page_refs)}] /Count {len(page_refs)} >>'.encode('ascii')))
    objects.insert(2, (font_obj, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'))
    objects = sorted(objects, key=lambda item: item[0])

    pdf = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for obj_num, body in objects:
        offsets.append(len(pdf))
        pdf.extend(f'{obj_num} 0 obj\n'.encode('ascii'))
        pdf.extend(body)
        pdf.extend(b'\nendobj\n')
    xref_pos = len(pdf)
    pdf.extend(f'xref\n0 {len(objects) + 1}\n'.encode('ascii'))
    pdf.extend(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        pdf.extend(f'{offset:010d} 00000 n \n'.encode('ascii'))
    pdf.extend(f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF'.encode('ascii'))
    return bytes(pdf)

def _jpeg_size(path):
    try:
        with open(path, 'rb') as handle:
            data = handle.read()
        index = 2
        while index < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            length = int.from_bytes(data[index + 2:index + 4], 'big')
            if marker in {0xC0, 0xC1, 0xC2}:
                height = int.from_bytes(data[index + 5:index + 7], 'big')
                width = int.from_bytes(data[index + 7:index + 9], 'big')
                return width, height, data
            index += 2 + length
    except Exception:
        return None
    return None

def _pdf_text(x, y, text, size=10, color='0.067 0.208 0.310', font='F1'):
    return f'{color} rg BT /{font} {size} Tf {x} {y} Td ({_pdf_escape(text)}) Tj ET'

def _build_report_pdf(payload, logo_path=None):
    width, height = 612, 842
    primary = '0.059 0.357 0.541'
    accent = '0.169 0.655 0.773'
    text = '0.067 0.208 0.310'
    muted = '0.392 0.455 0.545'
    light = '0.961 0.984 0.992'
    border = '0.831 0.875 0.918'
    colors = [(0.059, 0.357, 0.541), (0.169, 0.655, 0.773), (0.184, 0.620, 0.267), (0.851, 0.467, 0.024), (0.486, 0.227, 0.929), (0.863, 0.149, 0.149)]
    image_info = _jpeg_size(logo_path) if logo_path else None
    pages = []

    def clip(value, limit):
        return str(value)

    def new_page(title=None):
        commands = [
            f'{light} rg 0 0 {width} {height} re f',
            f'{primary} rg 0 760 {width} 82 re f',
            f'{accent} rg 0 754 {width} 6 re f',
        ]
        if image_info:
            commands.append('q 46 0 0 46 42 778 cm /Logo Do Q')
            title_x = 100
        else:
            commands.append('1 1 1 rg 42 778 46 46 re f')
            commands.append(_pdf_text(54, 795, 'A', 20, primary, 'F2'))
            title_x = 100
        commands.extend([
            _pdf_text(title_x, 805, 'Accudetek Health Diagnostics', 15, '1 1 1', 'F2'),
            _pdf_text(title_x, 786, title or payload['title'], 11, '1 1 1'),
            _pdf_text(420, 805, 'Generated', 9, '0.860 0.945 0.969'),
            _pdf_text(420, 789, payload['generated_at'], 11, '1 1 1', 'F2'),
        ])
        return commands

    def add_footer(commands):
        commands.append(_pdf_text(42, 42, 'Smart Healthcare Clinic Management', 8, muted))
        pages.append(commands)

    def wrap_words(value, max_chars):
        words = str(value).split()
        lines = []
        current = ''
        for word in words:
            candidate = word if not current else current + ' ' + word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or ['']

    def draw_metric_table(commands, x, y, sections):
        commands.extend([
            f'{primary} rg {x} {y - 4} 528 26 re f',
            _pdf_text(x + 14, y + 4, 'Metric', 10, '1 1 1', 'F2'),
            _pdf_text(x + 340, y + 4, 'Value', 10, '1 1 1', 'F2'),
        ])
        row_y = y - 24
        for section in sections[:7]:
            metric_lines = wrap_words(section.get('metric', ''), 48)
            value_lines = wrap_words(section.get('value', ''), 30)
            line_count = max(len(metric_lines), len(value_lines))
            row_height = 12 * line_count + 10
            commands.append(f'{border} RG {x} {row_y + 8} m {x + 528} {row_y + 8} l S')
            for line_index, line in enumerate(metric_lines):
                commands.append(_pdf_text(x + 14, row_y - (line_index * 12), line, 8, text, 'F2'))
            for line_index, line in enumerate(value_lines):
                commands.append(_pdf_text(x + 340, row_y - (line_index * 12), line, 8, primary, 'F2'))
            row_y -= row_height
        commands.append(f'{border} RG {x} {row_y + 8} m {x + 528} {row_y + 8} l S')
        commands.append(f'{border} RG {x} {row_y + 8} {528} {y + 18 - (row_y + 8)} re S')
        return row_y

    def draw_line_chart(commands, chart, title, note, x=42, y=705, w=528, h=330):
        commands.append(_pdf_text(x, y, clip(title, 74), 14, text, 'F2'))
        if note:
            commands.append(_pdf_text(x, y - 16, clip(note, 110), 8, muted))
        left, bottom, chart_w, chart_h = x + 36, y - h + 34, w - 58, h - 95
        commands.extend([f'1 1 1 rg {x} {bottom - 34} {w} {h - 35} re f', f'{border} RG {x} {bottom - 34} {w} {h - 35} re S', f'{border} RG {left} {bottom} {chart_w} {chart_h} re S'])
        labels = chart.get('labels', [])
        datasets = chart.get('datasets', [])
        max_value = max([max(ds.get('data') or [0]) for ds in datasets] + [1])
        y_max = max(10, int(np.ceil(max_value / 10.0) * 10))
        for step in range(6):
            yy = bottom + (chart_h * step / 5)
            value = int(y_max * step / 5)
            commands.append(f'0.906 0.925 0.953 RG {left} {yy:.2f} m {left + chart_w} {yy:.2f} l S')
            commands.append(_pdf_text(x + 4, yy - 3, value, 7, muted))
        label_divisor = max(1, len(labels) - 1)
        for idx, label in enumerate(labels):
            xx = left + (chart_w * idx / label_divisor)
            commands.append(_pdf_text(xx - 8, bottom - 18, clip(label, 6), 7, muted))
        legend_x, legend_y = left, bottom + chart_h + 16
        for ds_idx, dataset in enumerate(datasets):
            r, g, b = colors[ds_idx % len(colors)]
            label = dataset.get('year', dataset.get('label', 'Series'))
            commands.append(f'{r} {g} {b} rg {legend_x} {legend_y - 2} 12 4 re f')
            commands.append(_pdf_text(legend_x + 17, legend_y - 5, label, 8, text))
            legend_x += 68
            points = []
            values = dataset.get('data', [])
            point_divisor = max(1, len(values) - 1)
            for point_idx, value in enumerate(values):
                xx = left + (chart_w * point_idx / point_divisor)
                yy = bottom + (chart_h * (value / y_max))
                points.append((xx, yy))
            if points:
                path = [f'{r} {g} {b} RG 2 w {points[0][0]:.2f} {points[0][1]:.2f} m']
                path.extend(f'{xx:.2f} {yy:.2f} l' for xx, yy in points[1:])
                path.append('S')
                commands.append(' '.join(path))
                for xx, yy in points:
                    commands.append(f'{r} {g} {b} rg {xx - 2:.2f} {yy - 2:.2f} 4 4 re f')

    def draw_bar_chart(commands, chart, x=42, y=705, w=528, h=260):
        commands.append(_pdf_text(x, y, clip(chart.get('title', 'Chart'), 74), 14, text, 'F2'))
        labels = chart.get('labels', [])
        values = chart.get('data', [])
        max_value = max(values + [1])
        left, bottom, chart_w, chart_h = x + 34, y - h + 38, w - 62, h - 78
        commands.extend([f'1 1 1 rg {x} {bottom - 34} {w} {h - 25} re f', f'{border} RG {x} {bottom - 34} {w} {h - 25} re S'])
        bar_count = max(1, len(values))
        bar_w = min(56, chart_w / (bar_count * 1.7))
        gap = (chart_w - (bar_w * bar_count)) / max(1, bar_count)
        for idx, value in enumerate(values):
            r, g, b = colors[idx % len(colors)]
            bar_h = chart_h * (value / max_value)
            xx = left + gap / 2 + idx * (bar_w + gap)
            commands.append(f'{r} {g} {b} rg {xx:.2f} {bottom:.2f} {bar_w:.2f} {bar_h:.2f} re f')
            commands.append(_pdf_text(xx, bottom + bar_h + 6, value, 8, text, 'F2'))
            commands.append(_pdf_text(xx - 4, bottom - 18, clip(labels[idx], 12), 7, muted))
        commands.append(_pdf_text(x + 8, bottom + chart_h + 8, clip(chart.get('y_title', 'Consultations'), 45), 8, muted))

    def draw_table(commands, title, rows, x=42, y=705, w=528, max_rows=18):
        commands.append(_pdf_text(x, y, str(title), 14, text, 'F2'))
        if not rows:
            commands.append(_pdf_text(x, y - 24, 'No rows available.', 9, muted))
            return
        keys = list(rows[0].keys())[:4]
        col_w = w / len(keys)
        header_y = y - 28
        commands.append(f'{primary} rg {x} {header_y - 8} {w} 22 re f')
        for idx, key in enumerate(keys):
            header_lines = wrap_words(str(key).replace('_', ' ').title(), 18)
            commands.append(_pdf_text(x + idx * col_w + 6, header_y, header_lines[0], 8, '1 1 1', 'F2'))
        row_y = header_y - 22
        row_limit = min(max_rows, len(rows))
        for row in rows[:row_limit]:
            wrapped_cells = [wrap_words(row.get(key, ''), 20) for key in keys]
            line_count = max(len(lines) for lines in wrapped_cells)
            row_height = 12 * line_count + 10
            commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')
            for idx, key in enumerate(keys):
                for line_index, line in enumerate(wrapped_cells[idx]):
                    commands.append(_pdf_text(x + idx * col_w + 6, row_y - (line_index * 12), line, 7, text))
            row_y -= row_height
        commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')

    def draw_simple_metric_table(commands, title, sections, x=42, y=705, w=528):
        commands.append(_pdf_text(x, y, title, 14, text, 'F2'))
        header_y = y - 30
        table_top = header_y + 14
        metric_w = 190
        value_w = w - metric_w
        commands.append(f'{primary} rg {x} {header_y - 8} {w} 22 re f')
        commands.append(_pdf_text(x + 8, header_y, 'Metric', 9, '1 1 1', 'F2'))
        commands.append(_pdf_text(x + metric_w + 8, header_y, 'Value', 9, '1 1 1', 'F2'))
        row_y = header_y - 24
        for section in sections:
            metric_lines = wrap_words(section.get('metric', ''), 28)
            value_lines = wrap_words(section.get('value', ''), 48)
            line_count = max(len(metric_lines), len(value_lines))
            row_height = 12 * line_count + 8
            commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')
            for line_index, line in enumerate(metric_lines):
                commands.append(_pdf_text(x + 8, row_y - (line_index * 12), line, 8, text, 'F2'))
            for line_index, line in enumerate(value_lines):
                commands.append(_pdf_text(x + metric_w + 8, row_y - (line_index * 12), line, 8, primary))
            row_y -= row_height
        commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')
        commands.append(f'{border} RG {x} {row_y + 8} {w} {table_top - (row_y + 8)} re S')
        return row_y

    def draw_simple_rows_table(commands, title, rows, x=42, y=360, w=528):
        commands.append(_pdf_text(x, y, title, 14, text, 'F2'))
        if not rows:
            commands.append(_pdf_text(x, y - 24, 'No rows available.', 9, muted))
            return
        keys = list(rows[0].keys())[:2]
        col_w = w / max(1, len(keys))
        header_y = y - 30
        table_top = header_y + 14
        commands.append(f'{primary} rg {x} {header_y - 8} {w} 22 re f')
        for idx, key in enumerate(keys):
            commands.append(_pdf_text(x + idx * col_w + 8, header_y, str(key).replace('_', ' ').title(), 9, '1 1 1', 'F2'))
        row_y = header_y - 24
        for row in rows[:12]:
            wrapped = [wrap_words(row.get(key, ''), 32) for key in keys]
            line_count = max(len(lines) for lines in wrapped)
            row_height = 12 * line_count + 8
            commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')
            for idx, lines in enumerate(wrapped):
                for line_index, line in enumerate(lines):
                    commands.append(_pdf_text(x + idx * col_w + 8, row_y - (line_index * 12), line, 8, text))
            row_y -= row_height
        commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')
        commands.append(f'{border} RG {x} {row_y + 8} {w} {table_top - (row_y + 8)} re S')
        return row_y

    first = new_page()
    first.append(_pdf_text(42, 727, clip(payload['description'], 110), 11, text))
    first.append(_pdf_text(42, 711, clip(f"Branch: {payload.get('branch_name', DEFAULT_BRANCH_NAME)}", 90), 9, muted))
    if payload.get('key') == 'resource-recommendation':
        metric_end_y = draw_simple_metric_table(first, 'Resource Recommendation Summary', payload.get('sections', []), y=690)
        draw_simple_rows_table(first, payload.get('details_title', 'Details'), payload.get('rows', []), y=metric_end_y - 28)
        add_footer(first)
    else:
        metric_end_y = draw_metric_table(first, 42, 676, payload.get('sections', []))
    if payload.get('key') == 'resource-recommendation':
        extra_tables = payload.get('extra_tables', [])
        for index in range(0, len(extra_tables), 2):
            page = new_page('Staff Requirement Forecast')
            draw_table(page, extra_tables[index].get('title', 'Additional Details'), extra_tables[index].get('rows', []), y=705, max_rows=10)
            if index + 1 < len(extra_tables):
                draw_table(page, extra_tables[index + 1].get('title', 'Additional Details'), extra_tables[index + 1].get('rows', []), y=390, max_rows=10)
            add_footer(page)
    elif payload.get('key') == 'prediction':
        draw_table(first, payload.get('details_title', 'Diagnosis Forecast Details'), payload.get('rows', []), y=metric_end_y - 28, max_rows=8)
        add_footer(first)
        page = new_page('Prediction Demographic Forecast')
        demographic_tables = payload.get('extra_tables', [])
        if len(demographic_tables) > 0:
            draw_table(page, demographic_tables[0].get('title', 'Forecast by Age Group'), demographic_tables[0].get('rows', []), y=705, max_rows=5)
        if len(demographic_tables) > 1:
            draw_table(page, demographic_tables[1].get('title', 'Forecast by Gender'), demographic_tables[1].get('rows', []), y=520, max_rows=5)
        if len(demographic_tables) > 2:
            draw_table(page, demographic_tables[2].get('title', 'Top Diagnosis-Demographic Forecast Segments'), demographic_tables[2].get('rows', []), y=335, max_rows=8)
        add_footer(page)
    elif payload.get('key') in {'monthly-consultation', 'quarterly'} and payload.get('chart_rows', {}).get('datasets'):
        draw_line_chart(first, payload['chart_rows'], payload.get('details_title', 'Trend Graph'), payload.get('details_note', ''), y=500, h=360)
        add_footer(first)
    elif payload.get('rows'):
        add_footer(first)
        page = new_page(payload.get('details_title', payload['title']))
        draw_table(page, payload.get('details_title', 'Details'), payload['rows'])
        add_footer(page)
    else:
        add_footer(first)

    if payload.get('key') not in {'prediction', 'resource-recommendation'}:
        extra_charts = payload.get('extra_charts', [])
        for index in range(0, len(extra_charts), 2):
            page = new_page('Additional Report Graphs')
            draw_bar_chart(page, extra_charts[index], y=705, h=255)
            if index + 1 < len(extra_charts):
                draw_bar_chart(page, extra_charts[index + 1], y=405, h=255)
            add_footer(page)

        extra_tables = payload.get('extra_tables', [])
        for index in range(0, len(extra_tables), 2):
            page = new_page('Additional Report Tables')
            draw_table(page, extra_tables[index].get('title', 'Additional Details'), extra_tables[index].get('rows', []), y=705, max_rows=10)
            if index + 1 < len(extra_tables):
                draw_table(page, extra_tables[index + 1].get('title', 'Additional Details'), extra_tables[index + 1].get('rows', []), y=390, max_rows=10)
            add_footer(page)

    objects = [(1, b'<< /Type /Catalog /Pages 2 0 R >>'), (3, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'), (5, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>')]
    resources = '<< /Font << /F1 3 0 R /F2 5 0 R >>'
    next_obj = 6
    if image_info:
        img_w, img_h, img_data = image_info
        image_obj_num = next_obj
        next_obj += 1
        objects.append((image_obj_num, f'<< /Type /XObject /Subtype /Image /Width {img_w} /Height {img_h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(img_data)} >>\nstream\n'.encode('ascii') + img_data + b'\nendstream'))
        resources += f' /XObject << /Logo {image_obj_num} 0 R >>'
    resources += ' >>'
    page_refs = []
    for commands in pages:
        content = '\n'.join(commands).encode('latin-1', errors='replace')
        content_obj = next_obj
        page_obj = next_obj + 1
        next_obj += 2
        objects.append((content_obj, b'<< /Length ' + str(len(content)).encode('ascii') + b' >>\nstream\n' + content + b'\nendstream'))
        objects.append((page_obj, f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources {resources} /Contents {content_obj} 0 R >>'.encode('ascii')))
        page_refs.append(f'{page_obj} 0 R')
    objects.append((2, f'<< /Type /Pages /Kids [{" ".join(page_refs)}] /Count {len(page_refs)} >>'.encode('ascii')))
    objects = sorted(objects, key=lambda item: item[0])

    pdf = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for obj_num, body in objects:
        offsets.append(len(pdf))
        pdf.extend(f'{obj_num} 0 obj\n'.encode('ascii'))
        pdf.extend(body)
        pdf.extend(b'\nendobj\n')
    xref_pos = len(pdf)
    pdf.extend(f'xref\n0 {max(obj for obj, _ in objects) + 1}\n'.encode('ascii'))
    pdf.extend(b'0000000000 65535 f \n')
    offset_by_obj = {obj_num: offset for obj_num, offset in zip([obj for obj, _ in objects], offsets[1:])}
    for obj_num in range(1, max(obj for obj, _ in objects) + 1):
        pdf.extend(f'{offset_by_obj.get(obj_num, 0):010d} 00000 n \n'.encode('ascii'))
    pdf.extend(f'trailer\n<< /Size {max(obj for obj, _ in objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF'.encode('ascii'))
    return bytes(pdf)

# -------------------------------------------------------------
# Flask application
# -------------------------------------------------------------
def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///clinic.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    db.init_app(app)
    settings_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'app_settings.json')
    dashboard_cache_version = 7

    def branch_cache_file_path(branch_id):
        cache_key = branch_id if branch_id is not None else ALL_BRANCHES_SCOPE
        return os.path.join(app.config['UPLOAD_FOLDER'], f'dashboard_summary_branch_{cache_key}.json')

    def load_app_settings():
        defaults = {'staff_capacity_per_month': STAFF_CAPACITY_PER_MONTH}
        if not os.path.exists(settings_file_path):
            return defaults
        try:
            with open(settings_file_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            defaults.update({
                'staff_capacity_per_month': max(1, int(data.get('staff_capacity_per_month', STAFF_CAPACITY_PER_MONTH)))
            })
        except Exception:
            pass
        return defaults

    def save_app_settings(settings):
        with open(settings_file_path, 'w', encoding='utf-8') as handle:
            json.dump(settings, handle, indent=2)

    def load_cached_dashboard_summary(branch_id):
        cache_file_path = branch_cache_file_path(branch_id)
        if not os.path.exists(cache_file_path):
            return None
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as handle:
                summary = json.load(handle)
            if summary.get('cache_version') != dashboard_cache_version:
                return None
            return summary
        except Exception:
            return None

    def cache_dashboard_summary(summary, branch_id):
        cache_file_path = branch_cache_file_path(branch_id)
        try:
            summary['cache_version'] = dashboard_cache_version
            with open(cache_file_path, 'w', encoding='utf-8') as handle:
                json.dump(summary, handle, indent=2)
        except Exception:
            pass

    def remove_cached_dashboard_summary(branch_id):
        try:
            cache_file_path = branch_cache_file_path(branch_id)
            if os.path.exists(cache_file_path):
                os.remove(cache_file_path)
        except Exception:
            pass

    def invalidate_dashboard_caches(branch_id='selected'):
        if branch_id == 'selected':
            branch_id = selected_branch_scope()
        remove_cached_dashboard_summary(branch_id)
        remove_cached_dashboard_summary(None)

    def invalidate_all_dashboard_caches():
        remove_cached_dashboard_summary(None)
        for branch in Branch.query.all():
            remove_cached_dashboard_summary(branch.id)

    def ensure_default_branch():
        branch = Branch.query.filter_by(code=DEFAULT_BRANCH_CODE).first()
        if branch is None:
            branch = Branch(
                name=DEFAULT_BRANCH_NAME,
                code=DEFAULT_BRANCH_CODE,
                address=DEFAULT_BRANCH_ADDRESS,
                is_main=True,
                is_active=True,
            )
            db.session.add(branch)
            db.session.commit()
        return branch

    def user_home_branch_is_main():
        branch_id = session.get('branch_id')
        if not branch_id:
            return False
        try:
            branch = Branch.query.get(int(branch_id))
            return bool(branch and branch.is_main)
        except Exception:
            return False

    def can_view_all_branches():
        role = session.get('role')
        if role == SUPER_ADMIN_ROLE:
            return True
        return role in MAIN_ADMIN_ROLES and user_home_branch_is_main()

    def is_superadmin_user():
        return session.get('role') == SUPER_ADMIN_ROLE

    def can_manage_services():
        return session.get('role') in SERVICE_MANAGER_ROLES

    def require_service_manager():
        if can_manage_services():
            return None
        flash('Only administrators can manage clinic services.', 'error')
        return redirect(url_for('dashboard'))

    def selected_branch_scope():
        if can_view_all_branches():
            selected = session.get('selected_branch_id') or session.get('branch_id')
            if selected == ALL_BRANCHES_SCOPE:
                return None
        else:
            selected = session.get('branch_id')
        if selected:
            try:
                return int(selected)
            except (TypeError, ValueError):
                pass
        return ensure_default_branch().id

    def current_branch_id():
        selected = session.get('selected_branch_id')
        branch_id = session.get('branch_id')
        if can_view_all_branches() and selected and selected != ALL_BRANCHES_SCOPE:
            branch_id = selected
        if branch_id:
            try:
                return int(branch_id)
            except (TypeError, ValueError):
                pass
        return ensure_default_branch().id

    def current_branch():
        scope = selected_branch_scope()
        if scope is None:
            return None
        branch = Branch.query.get(scope)
        return branch or ensure_default_branch()

    def branch_scope_label():
        branch = current_branch()
        return branch.name if branch else 'All Branches'

    def scoped_query(query, model, branch_id='selected'):
        if branch_id == 'selected':
            branch_id = selected_branch_scope()
        if branch_id is None:
            return query
        return query.filter(model.branch_id == branch_id)

    def require_specific_branch(target_endpoint):
        if selected_branch_scope() is not None:
            return None
        flash('Select a specific branch before making changes.', 'error')
        return redirect(url_for(target_endpoint))

    def require_main_admin():
        if can_view_all_branches():
            return None
        flash('Only the main administrator can manage branches.', 'error')
        return redirect(url_for('dashboard'))

    def generate_patient_number(branch):
        prefix = f"{branch.code}-{datetime.now().strftime('%Y%m%d')}"
        next_number = Patient.query.filter(Patient.patient_number.like(f'{prefix}-%')).count() + 1
        while True:
            patient_number = f'{prefix}-{next_number:04d}'
            if not Patient.query.filter_by(patient_number=patient_number).first():
                return patient_number
            next_number += 1

    def get_scoped_patient_or_404(patient_id):
        return scoped_query(Patient.query, Patient).filter_by(id=patient_id).first_or_404()

    def categorize_age(age):
        if age is None:
            return ''
        if age <= 17:
            return 'Child'
        if age <= 59:
            return 'Adult'
        return 'Senior'

    def parse_iso_date(date_value):
        try:
            return datetime.strptime(str(date_value).strip(), '%Y-%m-%d').date()
        except (TypeError, ValueError):
            return None

    def calculate_age_from_birthdate(birthdate_value, reference_date=None):
        birthdate = parse_iso_date(birthdate_value)
        if birthdate is None:
            return None
        reference_date = reference_date or datetime.now().date()
        if reference_date < birthdate:
            return None
        age = reference_date.year - birthdate.year
        if (reference_date.month, reference_date.day) < (birthdate.month, birthdate.day):
            age -= 1
        if 0 <= age <= 130:
            return age
        return None

    def patient_form_values(patient=None):
        return {
            'patient_number': request.form.get('patient_number', patient.patient_number if patient else '').strip().upper(),
            'full_name': request.form.get('full_name', patient.full_name if patient else '').strip(),
            'birthdate': request.form.get('birthdate', patient.birthdate if patient else '').strip(),
            'gender': request.form.get('gender', patient.gender if patient else '').strip(),
            'contact_number': request.form.get('contact_number', patient.contact_number if patient else '').strip(),
            'email': request.form.get('email', patient.email if patient else '').strip(),
            'address': request.form.get('address', patient.address if patient else '').strip(),
            'emergency_contact_name': request.form.get(
                'emergency_contact_name',
                patient.emergency_contact_name if patient else ''
            ).strip(),
            'emergency_contact_number': request.form.get(
                'emergency_contact_number',
                patient.emergency_contact_number if patient else ''
            ).strip(),
        }

    def consultation_form_values(patient):
        return {
            'consultation_date': request.form.get('consultation_date', datetime.now().strftime('%Y-%m-%d')).strip(),
            'gender': request.form.get('gender', patient.gender).strip(),
            'diagnosis': request.form.get('diagnosis', '').strip(),
            'physician': request.form.get('physician', '').strip(),
            'consultation_type': request.form.get('consultation_type', 'Walk-in').strip(),
        }

    def unique_list(values):
        seen = set()
        results = []
        for value in values:
            value = str(value).strip()
            if value and value not in seen:
                seen.add(value)
                results.append(value)
        return results

    def json_list(values):
        return json.dumps(unique_list(values))

    def get_branch_service_options(branch_id=None, only_available=True):
        branch_id = branch_id or current_branch_id()
        ensure_branch_service_rows(branch_id)
        query = (
            BranchService.query
            .join(ClinicService)
            .filter(BranchService.branch_id == branch_id)
        )
        if only_available:
            query = query.filter(BranchService.is_available.is_(True), ClinicService.is_active.is_(True))

        rows = (
            query
            .order_by(ClinicService.category.asc(), ClinicService.section.asc(), ClinicService.service_name.asc())
            .all()
        )
        options = []
        for row in rows:
            service = row.service
            if not service:
                continue
            options.append({
                'setting': row,
                'service': service,
                'value': service_booking_label(service),
                'label': service.service_name,
                'category': service.category,
                'section': service.section,
                'price_php': row.custom_price_php or service.price_php or '',
                'roles': service.role_items(),
                'equipment': service.equipment_items(),
            })
        return options

    def service_option_groups(options):
        grouped = {}
        for option in options:
            grouped.setdefault(option['category'] or 'Other Services', []).append(option)
        return grouped

    def find_service_by_booking_label(label, branch_id=None):
        for option in get_branch_service_options(branch_id=branch_id, only_available=False):
            service = option['service']
            if option['value'] == label or service.service_name == label:
                return service
        return None

    def service_recommendation_map_from_options(options):
        return {
            option['value']: {
                'roles': option['roles'],
                'equipment': option['equipment'],
            }
            for option in options
        }

    def package_booking_label(package):
        return package.package_name

    def package_requirements(package):
        roles = []
        equipment = []
        for item in sorted(package.items, key=lambda entry: entry.item_order):
            roles.extend(item.role_items())
            equipment.extend(item.equipment_items())
        return {
            'roles': unique_list(roles),
            'equipment': unique_list(equipment),
        }

    def get_branch_package_options(branch_id=None, only_available=True):
        branch_id = branch_id or current_branch_id()
        ensure_branch_package_rows(branch_id)
        query = (
            BranchPackage.query
            .join(ClinicPackage)
            .filter(BranchPackage.branch_id == branch_id)
        )
        if only_available:
            query = query.filter(BranchPackage.is_available.is_(True), ClinicPackage.is_active.is_(True))
        rows = query.order_by(ClinicPackage.package_name.asc()).all()
        options = []
        for row in rows:
            package = row.package
            if not package:
                continue
            requirements = package_requirements(package)
            options.append({
                'setting': row,
                'package': package,
                'value': package_booking_label(package),
                'label': package.package_name,
                'price_php': row.custom_price_php or package.price_php or '',
                'item_count': len(package.items),
                'roles': requirements['roles'],
                'equipment': requirements['equipment'],
            })
        return options

    def find_package_by_booking_label(label, branch_id=None):
        for option in get_branch_package_options(branch_id=branch_id, only_available=False):
            package = option['package']
            if option['value'] == label or package.package_name == label:
                return package
        return None

    def package_recommendation_map_from_options(options):
        return {
            option['value']: {
                'roles': option['roles'],
                'equipment': option['equipment'],
            }
            for option in options
        }

    def normalize_appointment_time(value):
        text_value = str(value or '').strip()
        if not text_value:
            return ''
        match = re.match(r'^(\d{1,2}):(\d{2})', text_value)
        if not match:
            return ''
        hour = int(match.group(1))
        minute = int(match.group(2))
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return ''
        return f'{hour:02d}:{minute:02d}'

    def appointment_slot_conflict(branch_id, appointment_date, appointment_time, exclude_id=None):
        appointment_time = normalize_appointment_time(appointment_time)
        if not branch_id or not appointment_date or not appointment_time:
            return None
        query = Appointment.query.filter(
            Appointment.branch_id == branch_id,
            Appointment.appointment_date == appointment_date,
            Appointment.appointment_time == appointment_time,
            Appointment.status.in_(ACTIVE_APPOINTMENT_SLOT_STATUSES),
        )
        if exclude_id:
            query = query.filter(Appointment.id != exclude_id)
        return query.first()

    def appointment_form_values():
        return {
            'patient_id': request.form.get('patient_id', '').strip(),
            'appointment_date': request.form.get('appointment_date', '').strip(),
            'appointment_time': normalize_appointment_time(request.form.get('appointment_time', '')),
            'selected_services': unique_list(request.form.getlist('selected_services')),
            'selected_packages': unique_list(request.form.getlist('selected_packages')),
            'consultation_reasons': unique_list(request.form.getlist('consultation_reasons')),
            'other_reason': request.form.get('other_reason', '').strip(),
        }

    def build_appointment_recommendation(selected_services, consultation_reasons, other_reason='', selected_packages=None, branch_id=None):
        selected_packages = selected_packages or []
        roles = []
        equipment = []
        notes = []

        for service in selected_services:
            catalog_service = find_service_by_booking_label(service, branch_id=branch_id)
            if catalog_service:
                roles.extend(catalog_service.role_items())
                equipment.extend(catalog_service.equipment_items())
            else:
                mapping = SERVICE_RECOMMENDATION_MAP.get(service, {})
                roles.extend(mapping.get('roles', []))
                equipment.extend(mapping.get('equipment', []))

        for package_label in selected_packages:
            package = find_package_by_booking_label(package_label, branch_id=branch_id)
            if not package:
                continue
            requirements = package_requirements(package)
            roles.extend(requirements['roles'])
            equipment.extend(requirements['equipment'])
            notes.append(f'Package selected: {package.package_name} includes {len(package.items)} service item(s).')

        for reason in consultation_reasons:
            roles.extend(REASON_RECOMMENDATION_MAP.get(reason, []))

        lower_reason = other_reason.lower()
        for keyword, mapped_roles in REASON_KEYWORD_RECOMMENDATIONS.items():
            if keyword in lower_reason:
                roles.extend(mapped_roles)

        if (selected_services and len(selected_services) > 1) or selected_packages:
            notes.append('Stacked appointment: multiple services were selected, so the patient may need more than one staff role or service area.')

        roles = unique_list(roles)
        equipment = unique_list(equipment)
        notes = unique_list(notes)

        if not roles:
            roles = ['General Physicians']
            notes.append('No exact mapping was found, so the appointment should be reviewed by a General Physician first.')

        return roles, equipment, notes

    def appointment_service_department(service):
        catalog_service = find_service_by_booking_label(service)
        if catalog_service:
            category = (catalog_service.category or '').lower()
            section = (catalog_service.section or '').lower()
            name = (catalog_service.service_name or '').lower()
            combined = f'{category} {section} {name}'
            if category == 'laboratory':
                return 'Laboratory'
            if category == 'drug testing':
                return 'Drug Testing'
            if category == 'home service':
                return 'Home Service'
            if category == 'imaging/cardiology':
                if any(term in combined for term in ['ecg', 'echo', 'echocardiogram']):
                    return 'Cardiology'
                return 'Radiology'
            return 'Clinic'

        laboratory_services = {
            'Clinical Laboratory Services',
            'Clinical Microscopy',
            'Hematology',
            'Clinical Chemistry',
            'Immunology and Serology',
            'Drug Testing',
        }
        imaging_services = {'X-ray', 'Ultrasound', 'Vascular Studies'}
        if service in laboratory_services:
            return 'Laboratory'
        if service in imaging_services:
            return 'Radiology'
        if service == 'Electrocardiography (ECG)':
            return 'ECG'
        if service == 'Home Service':
            return 'Home Service'
        return 'Clinic'

    def appointment_consultation_type(service):
        catalog_service = find_service_by_booking_label(service)
        if catalog_service:
            category = (catalog_service.category or '').lower()
            section = (catalog_service.section or '').lower()
            name = (catalog_service.service_name or '').lower()
            combined = f'{category} {section} {name}'
            if category == 'laboratory':
                return 'Laboratory'
            if category == 'drug testing':
                return 'Drug Testing'
            if category == 'home service':
                return 'Home Service'
            if category == 'imaging/cardiology':
                if any(term in combined for term in ['ecg', 'echo', 'echocardiogram']):
                    return 'Cardiology'
                return 'Imaging'
            if category == 'annual checkup':
                return 'Annual Physical Examination'
            return 'Check-up'

        laboratory_services = {
            'Clinical Laboratory Services',
            'Clinical Microscopy',
            'Hematology',
            'Clinical Chemistry',
            'Immunology and Serology',
            'Drug Testing',
        }
        imaging_services = {'X-ray', 'Ultrasound', 'Vascular Studies'}
        if service == 'General Physician Consultation':
            return 'Check-up'
        if service in laboratory_services:
            return 'Laboratory'
        if service in imaging_services:
            return 'Imaging'
        if service == 'Annual Physical Examination (APE)':
            return 'Annual Physical Examination'
        return service

    def suggested_roles_for_service(service, appointment=None, branch_id=None):
        catalog_service = find_service_by_booking_label(service, branch_id=branch_id)
        if catalog_service:
            roles = catalog_service.role_items()
        else:
            mapping = SERVICE_RECOMMENDATION_MAP.get(service, {})
            roles = list(mapping.get('roles', []))
        if appointment and service == 'General Physician Consultation':
            roles.extend(appointment.role_items())
        if not roles:
            roles = ['General Physicians']
        return unique_list(roles)

    def default_appointment_diagnosis(appointment):
        diagnosis_defaults = {
            'Headache': 'Headache',
            'Fever': 'Fever',
            'Cough and colds': 'Upper Respiratory Infection',
            'Sore throat': 'Upper Respiratory Infection',
            'Dizziness': 'Dizziness',
            'Body weakness or fatigue': 'Fatigue',
            'Abdominal pain': 'Abdominal Pain',
            'Urinary tract infection symptoms': 'Urinary Tract Infection',
            'Gastrointestinal complaints': 'Gastrointestinal Complaint',
            'Skin conditions and allergies': 'Dermatitis',
            'Routine medical consultation and health clearance': 'Routine Medical Consultation',
            'Follow-up consultation': 'Follow-up Consultation',
            'Hypertension monitoring and management': 'Hypertension',
            'Diabetes mellitus monitoring and management': 'Diabetes',
            'Asthma monitoring and management': 'Asthma',
            'Preventive health check-up and wellness consultation': 'Preventive Health Check-up',
        }
        for reason in appointment.reason_items():
            if reason in diagnosis_defaults:
                return diagnosis_defaults[reason]
        if appointment.other_reason:
            return appointment.other_reason[:100]
        services = appointment.service_items()
        if services:
            return services[0]
        packages = appointment.package_items()
        return packages[0] if packages else 'Appointment Service'

    def package_item_department(item):
        if item.service:
            return appointment_service_department(service_booking_label(item.service))
        category = infer_package_item_category(item.item_name)
        if category == 'Laboratory':
            return 'Laboratory'
        if category == 'Drug Testing':
            return 'Drug Testing'
        if category == 'Imaging/Cardiology':
            lower_name = item.item_name.lower()
            if any(term in lower_name for term in ['ecg', 'echo', 'echocardiogram']):
                return 'Cardiology'
            return 'Radiology'
        return 'Clinic'

    def package_item_consultation_type(item):
        if item.service:
            return appointment_consultation_type(service_booking_label(item.service))
        category = infer_package_item_category(item.item_name)
        if category == 'Laboratory':
            return 'Laboratory'
        if category == 'Drug Testing':
            return 'Drug Testing'
        if category == 'Imaging/Cardiology':
            lower_name = item.item_name.lower()
            if any(term in lower_name for term in ['ecg', 'echo', 'echocardiogram']):
                return 'Cardiology'
            return 'Imaging'
        return 'Check-up'

    def appointment_completion_items(appointment):
        items = []
        seen = set()
        for service_label in appointment.service_items():
            if not service_label or service_label in seen:
                continue
            seen.add(service_label)
            items.append({
                'label': service_label,
                'roles': suggested_roles_for_service(service_label, appointment, branch_id=appointment.branch_id),
                'department': appointment_service_department(service_label),
                'consultation_type': appointment_consultation_type(service_label),
            })

        for package_label in appointment.package_items():
            package = find_package_by_booking_label(package_label, branch_id=appointment.branch_id)
            if not package:
                continue
            for item in sorted(package.items, key=lambda entry: entry.item_order):
                label = f'{package.package_name}: {item.item_name}'
                if label in seen:
                    continue
                seen.add(label)
                items.append({
                    'label': label,
                    'roles': item.role_items(),
                    'department': package_item_department(item),
                    'consultation_type': package_item_consultation_type(item),
                })
        return items

    def normalize_staff_forecast_role(role):
        role = clean_service_catalog_value(role)
        return STAFF_FORECAST_ROLE_ALIASES.get(role, role)

    def normalize_forecast_roles(roles):
        normalized = [
            normalize_staff_forecast_role(role)
            for role in roles
            if clean_service_catalog_value(role)
        ]
        return unique_list(normalized) or ['General Physicians']

    def roles_for_diagnosis_forecast(diagnosis):
        diagnosis_text = str(diagnosis or '').lower()
        roles = []
        for keywords, mapped_roles in DIAGNOSIS_STAFF_ROLE_RULES:
            if any(keyword in diagnosis_text for keyword in keywords):
                roles.extend(mapped_roles)
        return normalize_forecast_roles(roles)

    def roles_for_appointment_forecast(appointment):
        roles = []
        roles.extend(appointment.role_items())
        for service_label in appointment.service_items():
            roles.extend(suggested_roles_for_service(service_label, appointment, branch_id=appointment.branch_id))
        for package_label in appointment.package_items():
            package = find_package_by_booking_label(package_label, branch_id=appointment.branch_id)
            if package:
                roles.extend(package_requirements(package).get('roles', []))
        return normalize_forecast_roles(roles)

    def forecast_rows_to_role_counts(forecast_rows):
        role_counts = Counter()
        for row in forecast_rows or []:
            if isinstance(row, dict):
                diagnosis = row.get('diagnosis', '')
                count = row.get('predicted_next_month', row.get('forecasted_cases', 0))
            else:
                try:
                    diagnosis, count = row
                except (TypeError, ValueError):
                    continue
            try:
                count = max(0, int(round(float(count))))
            except (TypeError, ValueError):
                count = 0
            if count <= 0:
                continue
            for role in roles_for_diagnosis_forecast(diagnosis):
                role_counts[role] += count
        return role_counts

    def staff_forecast_status(demand, available_staff, capacity_per_staff):
        capacity = max(0, available_staff) * max(1, capacity_per_staff)
        required_staff = int(np.ceil(demand / max(1, capacity_per_staff))) if demand else 0
        gap = max(0, required_staff - available_staff)
        pressure = int(round((demand / max(1, capacity)) * 100)) if demand else 0
        if gap > 0:
            return required_staff, gap, pressure, 'Shortage', 'high'
        if pressure >= 75:
            return required_staff, gap, pressure, 'Monitor', 'moderate'
        return required_staff, gap, pressure, 'Sufficient', 'healthy'

    def build_staff_demand_forecast(staff_members, forecast_rows, predicted_month, predicted_year, predicted_month_label, branch=None):
        app_settings = load_app_settings()
        monthly_capacity_per_staff = max(1, int(app_settings['staff_capacity_per_month']))
        daily_capacity_per_staff = max(1, int(np.ceil(monthly_capacity_per_staff / max(1, DAYS_PER_MONTH))))
        active_staff_by_role = Counter(
            normalize_staff_forecast_role(member.role)
            for member in staff_members
            if member.is_active
        )
        available_staff_by_role = Counter(
            normalize_staff_forecast_role(member.role)
            for member in staff_members
            if member.is_active and member.availability == 'Available'
        )

        today = datetime.now().date()
        daily_end = today + timedelta(days=STAFF_FORECAST_DAILY_DAYS - 1)
        month_start = datetime(predicted_year, predicted_month, 1).date()
        if predicted_month == 12:
            month_end = datetime(predicted_year + 1, 1, 1).date() - timedelta(days=1)
        else:
            month_end = datetime(predicted_year, predicted_month + 1, 1).date() - timedelta(days=1)

        appointment_query = Appointment.query.filter(Appointment.status.in_(['Pending', 'Confirmed']))
        if branch is not None:
            appointment_query = appointment_query.filter(Appointment.branch_id == branch.id)
        appointments = appointment_query.all()

        daily_counts = Counter()
        monthly_scheduled_counts = Counter()
        scheduled_appointment_count = 0
        for appointment in appointments:
            appointment_date = parse_iso_date(appointment.appointment_date)
            if appointment_date is None:
                continue
            roles = roles_for_appointment_forecast(appointment)
            if month_start <= appointment_date <= month_end:
                scheduled_appointment_count += 1
                for role in roles:
                    monthly_scheduled_counts[role] += 1
            if today <= appointment_date <= daily_end:
                for role in roles:
                    daily_counts[(appointment_date, role)] += 1

        historical_role_counts = forecast_rows_to_role_counts(forecast_rows)
        role_names = sorted(set(historical_role_counts) | set(monthly_scheduled_counts) | set(active_staff_by_role))
        monthly_rows = []
        for role in role_names:
            historical_forecast = int(historical_role_counts.get(role, 0))
            scheduled_count = int(monthly_scheduled_counts.get(role, 0))
            planning_demand = max(historical_forecast, scheduled_count)
            available_staff = int(active_staff_by_role.get(role, 0))
            required_staff, gap, pressure, status, status_class = staff_forecast_status(
                planning_demand,
                available_staff,
                monthly_capacity_per_staff,
            )
            if gap:
                action = f'Add or schedule {gap} more {role}.'
            elif status == 'Monitor':
                action = f'Monitor {role} schedule because demand is close to capacity.'
            else:
                action = f'Current {role} coverage is enough for the forecast.'
            monthly_rows.append({
                'staff_role': role,
                'historical_forecast_cases': historical_forecast,
                'scheduled_appointments': scheduled_count,
                'planning_demand': planning_demand,
                'available_staff': available_staff,
                'required_staff': required_staff,
                'gap': gap,
                'pressure': pressure,
                'status': status,
                'status_class': status_class,
                'recommended_action': action,
            })
        monthly_rows.sort(key=lambda row: (row['gap'] == 0, -row['planning_demand'], row['staff_role']))

        daily_rows = []
        daily_role_totals = Counter()
        for offset in range(STAFF_FORECAST_DAILY_DAYS):
            forecast_date = today + timedelta(days=offset)
            roles_for_day = sorted({role for (date_value, role), count in daily_counts.items() if date_value == forecast_date})
            for role in roles_for_day:
                demand = int(daily_counts.get((forecast_date, role), 0))
                daily_role_totals[role] += demand
                available_staff = int(available_staff_by_role.get(role, active_staff_by_role.get(role, 0)))
                required_staff, gap, pressure, status, status_class = staff_forecast_status(
                    demand,
                    available_staff,
                    daily_capacity_per_staff,
                )
                daily_rows.append({
                    'date': forecast_date.strftime('%Y-%m-%d'),
                    'day': forecast_date.strftime('%b %d'),
                    'staff_role': role,
                    'scheduled_demand': demand,
                    'available_staff': available_staff,
                    'required_staff': required_staff,
                    'gap': gap,
                    'pressure': pressure,
                    'status': status,
                    'status_class': status_class,
                })

        daily_labels = [(today + timedelta(days=offset)).strftime('%b %d') for offset in range(STAFF_FORECAST_DAILY_DAYS)]
        daily_top_roles = [role for role, _ in daily_role_totals.most_common(5)]
        daily_chart = {
            'labels': daily_labels,
            'datasets': [
                {
                    'label': role,
                    'data': [
                        int(daily_counts.get((today + timedelta(days=offset), role), 0))
                        for offset in range(STAFF_FORECAST_DAILY_DAYS)
                    ],
                }
                for role in daily_top_roles
            ],
        }
        monthly_chart = {
            'labels': [row['staff_role'] for row in monthly_rows[:7]],
            'planning_demand': [row['planning_demand'] for row in monthly_rows[:7]],
            'capacity': [
                row['available_staff'] * monthly_capacity_per_staff
                for row in monthly_rows[:7]
            ],
        }

        shortage_rows = [row for row in monthly_rows if row['gap'] > 0]
        if shortage_rows:
            recommendation = 'Additional staff scheduling is needed for roles where the forecasted demand exceeds capacity.'
        elif any(row['status'] == 'Monitor' for row in monthly_rows):
            recommendation = 'Staffing is currently enough, but some roles are near capacity and should be monitored.'
        else:
            recommendation = 'Current role coverage is sufficient for the next-month staff demand forecast.'

        top_role = monthly_rows[0]['staff_role'] if monthly_rows else 'No role demand yet'
        top_role_demand = monthly_rows[0]['planning_demand'] if monthly_rows else 0

        return {
            'predicted_month_label': predicted_month_label,
            'monthly_capacity_per_staff': monthly_capacity_per_staff,
            'daily_capacity_per_staff': daily_capacity_per_staff,
            'scheduled_appointment_count': scheduled_appointment_count,
            'top_role': top_role,
            'top_role_demand': top_role_demand,
            'recommendation': recommendation,
            'monthly_rows': monthly_rows,
            'daily_rows': daily_rows,
            'daily_chart': daily_chart,
            'monthly_chart': monthly_chart,
        }

    def build_dashboard_context(records, staff_members, branch=None):
        total_consultations = len(records)

        monthly_counts = Counter()
        diagnosis_counts = Counter()
        gender_counts = Counter()
        age_group_counts = Counter()
        for record in records:
            try:
                d = pd.to_datetime(record.consultation_date, errors='coerce')
                if pd.notna(d):
                    monthly_counts[d.strftime('%Y-%m')] += 1
            except Exception:
                pass
            if record.diagnosis:
                diagnosis_counts[record.diagnosis] += 1
            if record.gender:
                gender_counts[record.gender] += 1
            if record.age_group:
                age_group_counts[record.age_group] += 1

        monthly_trend = [
            {'month': pd.to_datetime(month + '-01').strftime('%b %Y'), 'count': count}
            for month, count in sorted(monthly_counts.items())
        ]

        # Forecast the next calendar month from today's date.
        now = datetime.now()
        predicted_month = now.month + 1
        predicted_year = now.year
        if predicted_month > 12:
            predicted_month = 1
            predicted_year += 1
        reference_month = predicted_month
        reference_year = predicted_year - 1

        predicted_month_label = datetime(predicted_year, predicted_month, 1).strftime('%b %Y')
        predicted_month_full_label = datetime(predicted_year, predicted_month, 1).strftime('%B %Y')
        reference_month_label = datetime(reference_year, reference_month, 1).strftime('%b %Y')

        # Count actual consultations for the reference month (last year)
        reference_month_counts = Counter()
        for record in records:
            d = pd.to_datetime(record.consultation_date, errors='coerce')
            if pd.notna(d) and d.year == reference_year and d.month == reference_month:
                reference_month_counts[record.diagnosis] += 1

        # Generate forecast
        if records:
            df = pd.DataFrame([{
                'consultation_date': r.consultation_date,
                'age_group': r.age_group,
                'gender': r.gender,
                'diagnosis': r.diagnosis,
                'department': r.department,
                'physician': r.physician,
                'consultation_type': r.consultation_type,
            } for r in records])
            total_pred, forecast, rf_metrics = generate_forecast_for_specific_month(df, predicted_month, predicted_year)
            demographic_forecast = getattr(generate_forecast_for_month, 'last_demographic_forecast', {
                'age_group': [],
                'gender': [],
                'segments': [],
            })
        else:
            total_pred, forecast, rf_metrics = None, None, None
            demographic_forecast = {'age_group': [], 'gender': [], 'segments': []}

        if total_pred is not None and total_pred > 0:
            predicted_cases_next_month = total_pred
            predictions = []
            for diag, count in forecast[:5]:
                ref_count = reference_month_counts.get(diag, 0)
                predictions.append({
                    'diagnosis': diag,
                    'current_month': ref_count,
                    'predicted_next_month': count,
                    'predicted_month': predicted_month_label,
                    'trend': 'Increasing' if count > ref_count else 'Stable'
                })
            if not predictions:
                predictions = [{
                    'diagnosis': 'No data',
                    'current_month': 0,
                    'predicted_next_month': 0,
                    'predicted_month': predicted_month_label,
                    'trend': 'Stable'
                }]
            resource_recommendation = 'Based on enhanced Random Forest with CV tuning and seasonal lags.'
        else:
            # Naive fallback
            if len(monthly_trend) >= 2:
                prev_count = monthly_trend[-2]['count']
                curr_count = monthly_trend[-1]['count']
                growth = ((curr_count - prev_count) / prev_count) if prev_count else 0
                predicted_cases_next_month = max(0, int(round(curr_count * (1 + growth))))
            else:
                predicted_cases_next_month = total_consultations

            predictions = []
            for diag, cnt in diagnosis_counts.most_common(5):
                ref_count = reference_month_counts.get(diag, 0)
                if len(monthly_trend) >= 2:
                    prev = monthly_trend[-2]['count']
                    curr = monthly_trend[-1]['count']
                    gr = ((curr - prev) / prev) if prev else 0
                    pred_cnt = max(0, int(round(cnt * (1 + gr))))
                else:
                    pred_cnt = cnt
                predictions.append({
                    'diagnosis': diag,
                    'current_month': ref_count,
                    'predicted_next_month': pred_cnt,
                    'predicted_month': predicted_month_label,
                    'trend': 'Increasing' if pred_cnt > ref_count else 'Stable'
                })
            resource_recommendation = 'Naive fallback (RF unavailable).'
            rf_metrics = None
            demographic_forecast = {
                'age_group': [
                    {'age_group': group, 'forecasted_cases': count}
                    for group, count in age_group_counts.most_common()
                ],
                'gender': [
                    {'gender': gender, 'forecasted_cases': count}
                    for gender, count in gender_counts.most_common()
                ],
                'segments': [],
            }

        staff_demand_forecast = build_staff_demand_forecast(
            staff_members,
            forecast if forecast else predictions,
            predicted_month,
            predicted_year,
            predicted_month_label,
            branch=branch,
        )

        top_diagnosis = diagnosis_counts.most_common(1)[0][0] if diagnosis_counts else 'None'
        facility_staff_count = len(staff_members)
        app_settings = load_app_settings()
        staff_capacity_per_month = app_settings['staff_capacity_per_month']

        estimated_monthly_capacity = facility_staff_count * staff_capacity_per_month
        resource_readiness = min(100, max(0, round(estimated_monthly_capacity / max(1, predicted_cases_next_month) * 100)))
        pressure_ratio = predicted_cases_next_month / max(1, estimated_monthly_capacity)
        forecast_pressure_raw = int(pressure_ratio * 100)
        forecast_pressure = min(100, int(pressure_ratio * 100))
        recommended_staff_count = int(np.ceil(predicted_cases_next_month / max(1, staff_capacity_per_month)))
        staff_gap = max(0, recommended_staff_count - facility_staff_count)
        if pressure_ratio > 1.0:
            capacity_status = 'High'
            resource_forecast_recommendation = 'Predicted demand exceeds current staffing capacity; consider hiring additional staff or expanding service hours.'
        elif pressure_ratio > 0.75:
            capacity_status = 'Moderate'
            resource_forecast_recommendation = 'Service demand is approaching capacity; monitor staffing and diagnostic equipment availability closely.'
        else:
            capacity_status = 'Healthy'
            resource_forecast_recommendation = 'Current staffing and equipment capacity appears sufficient for forecasted demand.'

        actual_staff_by_role = Counter(member.role for member in staff_members)

        return {
            'branch_id': branch.id if branch else None,
            'branch_name': branch.name if branch else DEFAULT_BRANCH_NAME,
            'branch_code': branch.code if branch else DEFAULT_BRANCH_CODE,
            'total_consultations': total_consultations,
            'staff_count': len(staff_members),
            'facility_staff_count': facility_staff_count,
            'staff_capacity_per_month': staff_capacity_per_month,
            'top_diagnosis': top_diagnosis,
            'predicted_cases_next_month': predicted_cases_next_month,
            'resource_readiness': resource_readiness,
            'monthly_trend': monthly_trend,
            'consultation_distribution': diagnosis_counts.most_common(10),
            'gender_distribution': gender_counts.most_common(),
            'age_group_distribution': age_group_counts.most_common(),
            'top_cases': diagnosis_counts.most_common(5),
            'predictions': predictions,
            'demographic_forecast': demographic_forecast,
            'resource_recommendation': resource_recommendation,
            'reference_month': reference_month_label,
            'predicted_month_label': predicted_month_label,
            'predicted_month_full_label': predicted_month_full_label,
            'services': SERVICE_CATALOG,
            'equipment': EQUIPMENT_INVENTORY,
            'facility_staff_complement': FACILITY_STAFF_COMPLEMENT,
            'actual_staff_by_role': dict(actual_staff_by_role),
            'estimated_monthly_capacity': estimated_monthly_capacity,
            'forecast_pressure': forecast_pressure,
            'forecast_pressure_raw': forecast_pressure_raw,
            'recommended_staff_count': recommended_staff_count,
            'staff_gap': staff_gap,
            'capacity_status': capacity_status,
            'resource_forecast_recommendation': resource_forecast_recommendation,
            'staff_demand_forecast': staff_demand_forecast,
            'rf_metrics': rf_metrics,
        }

    def get_dashboard_summary(force_refresh=False, branch_id='selected'):
        if branch_id == 'selected':
            branch_id = selected_branch_scope()
        if force_refresh:
            remove_cached_dashboard_summary(branch_id)
            if branch_id is not None:
                remove_cached_dashboard_summary(None)
        if not force_refresh:
            cached = load_cached_dashboard_summary(branch_id)
            if cached is not None:
                return cached

        branch = None if branch_id is None else (Branch.query.get(branch_id) or ensure_default_branch())
        records = scoped_query(ConsultationRecord.query, ConsultationRecord, branch_id=branch_id).all()
        staff = scoped_query(StaffMember.query.filter_by(is_active=True), StaffMember, branch_id=branch_id).all()
        summary = build_dashboard_context(records, staff, branch=branch)
        if branch_id is None:
            summary['branch_name'] = 'All Branches'
            summary['branch_code'] = ALL_BRANCHES_SCOPE.upper()
        cache_dashboard_summary(summary, branch_id)
        return summary

    # -------------------------------------------------------------
    # Routes (unchanged except for dashboard/upload which use the new logic)
    # -------------------------------------------------------------
    @app.before_request
    def require_login():
        if request.endpoint in {'login', 'static'}:
            return None
        protected_endpoints = {
            'dashboard', 'records', 'upload', 'predict', 'staff', 'reports',
            'report_view', 'report_print', 'report_pdf',
            'permanent_delete_staff',
            'patients', 'create_patient', 'patient_detail', 'edit_patient',
            'archive_patient', 'restore_patient', 'create_patient_consultation',
            'appointments', 'create_appointment', 'update_appointment_status', 'complete_appointment',
            'services', 'create_service', 'edit_service', 'import_services', 'update_branch_service',
            'packages', 'create_package', 'edit_package', 'import_packages', 'update_branch_package',
            'settings', 'resources',
            'branches', 'create_branch', 'edit_branch', 'toggle_branch', 'select_branch',
            'create_user', 'assign_user_branch',
        }
        if request.endpoint in protected_endpoints and 'user_id' not in session:
            return redirect(url_for('login'))

    def format_appointment_time(value):
        text_value = str(value or '').strip()
        if not text_value:
            return 'Time not set'
        if text_value.upper().endswith((' AM', ' PM')):
            return text_value
        try:
            hour_text, minute_text = text_value.split(':')[:2]
            hour = int(hour_text)
            minute = int(minute_text)
            period = 'AM' if hour < 12 else 'PM'
            display_hour = hour % 12 or 12
            return f'{display_hour}:{minute:02d} {period}'
        except Exception:
            return text_value

    @app.context_processor
    def inject_branch_context():
        try:
            if can_view_all_branches():
                branch_options = Branch.query.filter_by(is_active=True).order_by(Branch.name.asc()).all()
            else:
                branch_options = []
            current_user = User.query.get(session['user_id']) if session.get('user_id') else None
            role = session.get('role', '')
            if role == SUPER_ADMIN_ROLE:
                access_label = 'All Branch Access'
            elif can_view_all_branches():
                access_label = 'Main Branch Access'
            elif session.get('user_id'):
                access_label = 'Assigned Branch Access'
            else:
                access_label = ''
            return {
                'active_branch': current_branch(),
                'active_branch_label': branch_scope_label(),
                'selected_branch_value': ALL_BRANCHES_SCOPE if selected_branch_scope() is None else str(selected_branch_scope()),
                'branch_options': branch_options,
                'can_view_all_branches': can_view_all_branches(),
                'is_superadmin': is_superadmin_user(),
                'can_manage_services': can_manage_services(),
                'current_user_name': current_user.username if current_user else session.get('username', ''),
                'current_user_role_label': USER_ROLE_LABELS.get(role, role.replace('_', ' ').title()),
                'current_access_label': access_label,
                'format_appointment_time': format_appointment_time,
            }
        except Exception:
            return {
                'active_branch': None,
                'active_branch_label': DEFAULT_BRANCH_NAME,
                'selected_branch_value': '',
                'branch_options': [],
                'can_view_all_branches': False,
                'is_superadmin': False,
                'can_manage_services': False,
                'current_user_name': session.get('username', ''),
                'current_user_role_label': USER_ROLE_LABELS.get(session.get('role', ''), ''),
                'current_access_label': '',
                'format_appointment_time': format_appointment_time,
            }

    def public_portal_branch():
        branch_id_value = request.form.get('branch_id') or request.args.get('branch_id')
        branch = None
        if branch_id_value:
            try:
                branch = Branch.query.filter_by(id=int(branch_id_value), is_active=True).first()
            except (TypeError, ValueError):
                branch = None
        if branch is None:
            branch = Branch.query.filter_by(is_main=True, is_active=True).first()
        return branch or Branch.query.filter_by(is_active=True).order_by(Branch.id.asc()).first() or ensure_default_branch()

    def public_patient_values():
        return {
            'full_name': request.form.get('full_name', '').strip(),
            'birthdate': request.form.get('birthdate', '').strip(),
            'gender': request.form.get('gender', '').strip(),
            'contact_number': request.form.get('contact_number', '').strip(),
            'email': request.form.get('email', '').strip().lower(),
            'address': request.form.get('address', '').strip(),
            'emergency_contact_name': request.form.get('emergency_contact_name', '').strip(),
            'emergency_contact_number': request.form.get('emergency_contact_number', '').strip(),
        }

    def find_or_create_public_patient(branch, values, reference_date):
        patient = None
        if values['email']:
            patient = Patient.query.filter_by(branch_id=branch.id, email=values['email'], is_active=True).first()
        if patient is None and values['contact_number']:
            patient = Patient.query.filter_by(
                branch_id=branch.id,
                full_name=values['full_name'],
                birthdate=values['birthdate'],
                contact_number=values['contact_number'],
                is_active=True,
            ).first()
        age = calculate_age_from_birthdate(values['birthdate'], reference_date)
        if patient is None:
            patient = Patient(
                branch_id=branch.id,
                patient_number=generate_patient_number(branch),
                full_name=values['full_name'],
                birthdate=values['birthdate'],
                age=age,
                age_group=categorize_age(age),
                gender=values['gender'],
                contact_number=values['contact_number'],
                email=values['email'],
                address=values['address'],
                emergency_contact_name=values['emergency_contact_name'],
                emergency_contact_number=values['emergency_contact_number'],
                is_active=True,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            db.session.add(patient)
            db.session.flush()
            return patient

        patient.full_name = values['full_name']
        patient.birthdate = values['birthdate']
        patient.age = age
        patient.age_group = categorize_age(age)
        patient.gender = values['gender']
        patient.contact_number = values['contact_number'] or patient.contact_number
        patient.email = values['email'] or patient.email
        patient.address = values['address'] or patient.address
        patient.emergency_contact_name = values['emergency_contact_name'] or patient.emergency_contact_name
        patient.emergency_contact_number = values['emergency_contact_number'] or patient.emergency_contact_number
        patient.updated_at = datetime.now()
        return patient

    @app.route('/patient-portal', methods=['GET', 'POST'])
    def patient_portal():
        branch = public_portal_branch()
        branch_options = Branch.query.filter_by(is_active=True).order_by(Branch.name.asc()).all()
        service_options = get_branch_service_options(branch.id, only_available=True)
        package_options = get_branch_package_options(branch.id, only_available=True)
        available_service_values = {option['value'] for option in service_options}
        available_package_values = {option['value'] for option in package_options}
        values = {
            'branch_id': str(branch.id),
            'appointment_date': datetime.now().strftime('%Y-%m-%d'),
            'appointment_time': '',
            'selected_services': [],
            'selected_packages': [],
            'consultation_reasons': [],
            'other_reason': '',
            **public_patient_values(),
        }

        if request.method == 'POST':
            values.update(public_patient_values())
            values['branch_id'] = str(branch.id)
            values['appointment_date'] = request.form.get('appointment_date', '').strip()
            values['appointment_time'] = normalize_appointment_time(request.form.get('appointment_time', ''))
            values['selected_services'] = unique_list(request.form.getlist('selected_services'))
            values['selected_packages'] = unique_list(request.form.getlist('selected_packages'))
            values['consultation_reasons'] = unique_list(request.form.getlist('consultation_reasons'))
            values['other_reason'] = request.form.get('other_reason', '').strip()

            appointment_date = parse_iso_date(values['appointment_date'])
            today = datetime.now().date()
            if not values['full_name']:
                flash('Full name is required.', 'error')
            elif not values['birthdate']:
                flash('Birthdate is required so the system can compute the age group.', 'error')
            elif calculate_age_from_birthdate(values['birthdate'], appointment_date or today) is None:
                flash('Please enter a valid birthdate.', 'error')
            elif values['gender'] not in PATIENT_GENDER_OPTIONS:
                flash('Please select a valid gender.', 'error')
            elif not values['contact_number'] and not values['email']:
                flash('Please provide either a contact number or email address.', 'error')
            elif appointment_date is None or appointment_date < today:
                flash('Please choose a valid appointment date.', 'error')
            elif not values['appointment_time']:
                flash('Please choose an appointment time.', 'error')
            elif appointment_slot_conflict(branch.id, values['appointment_date'], values['appointment_time']):
                flash('That appointment time is already taken for this branch. Please choose another time.', 'error')
            elif not values['selected_services'] and not values['selected_packages']:
                flash('Please select at least one service or package.', 'error')
            elif any(service not in available_service_values for service in values['selected_services']):
                flash('One or more selected services are not available for this branch.', 'error')
            elif any(package not in available_package_values for package in values['selected_packages']):
                flash('One or more selected packages are not available for this branch.', 'error')
            else:
                roles, equipment, notes = build_appointment_recommendation(
                    values['selected_services'],
                    values['consultation_reasons'],
                    values['other_reason'],
                    values['selected_packages'],
                    branch_id=branch.id,
                )
                patient = find_or_create_public_patient(branch, values, appointment_date)
                appointment = Appointment(
                    branch_id=branch.id,
                    patient_id=patient.id,
                    appointment_date=values['appointment_date'],
                    appointment_time=values['appointment_time'],
                    selected_services=json_list(values['selected_services']),
                    selected_packages=json_list(values['selected_packages']),
                    consultation_reasons=json_list(values['consultation_reasons']),
                    other_reason=values['other_reason'],
                    recommended_roles=json_list(roles),
                    recommended_equipment=json_list(equipment),
                    recommendation_notes=json_list(notes),
                    status='Pending',
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                db.session.add(appointment)
                try:
                    db.session.commit()
                except IntegrityError:
                    db.session.rollback()
                    flash('That appointment time was just taken. Please choose another time.', 'error')
                    return render_template(
                        'patient_portal/index.html',
                        branch=branch,
                        branch_options=branch_options,
                        service_options=service_options,
                        service_groups=service_option_groups(service_options),
                        package_options=package_options,
                        consultation_reason_options=COMMON_CONSULTATION_REASONS,
                        service_recommendations=service_recommendation_map_from_options(service_options),
                        package_recommendations=package_recommendation_map_from_options(package_options),
                        reason_recommendations=REASON_RECOMMENDATION_MAP,
                        keyword_recommendations=REASON_KEYWORD_RECOMMENDATIONS,
                        gender_options=PATIENT_GENDER_OPTIONS,
                        values=values,
                        today=datetime.now().strftime('%Y-%m-%d'),
                        current_date=datetime.now().strftime('%Y-%m-%d'),
                    )
                invalidate_dashboard_caches(branch.id)
                return render_template(
                    'patient_portal/success.html',
                    branch=branch,
                    patient=patient,
                    appointment=appointment,
                    current_date=datetime.now().strftime('%Y-%m-%d'),
                )

        return render_template(
            'patient_portal/index.html',
            branch=branch,
            branch_options=branch_options,
            service_options=service_options,
            service_groups=service_option_groups(service_options),
            package_options=package_options,
            consultation_reason_options=COMMON_CONSULTATION_REASONS,
            service_recommendations=service_recommendation_map_from_options(service_options),
            package_recommendations=package_recommendation_map_from_options(package_options),
            reason_recommendations=REASON_RECOMMENDATION_MAP,
            keyword_recommendations=REASON_KEYWORD_RECOMMENDATIONS,
            gender_options=PATIENT_GENDER_OPTIONS,
            values=values,
            today=datetime.now().strftime('%Y-%m-%d'),
            current_date=datetime.now().strftime('%Y-%m-%d'),
        )

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            user = User.query.filter_by(username=request.form['username']).first()
            if user and user.password == request.form['password']:
                if not user.branch_id:
                    user.branch_id = ensure_default_branch().id
                    db.session.commit()
                session['user_id'] = user.id
                session['username'] = user.username
                session['role'] = user.role
                session['branch_id'] = user.branch_id
                session['selected_branch_id'] = user.branch_id
                flash(f'Welcome back, {user.username}.', 'success')
                return redirect(url_for('dashboard'))
            flash('Invalid credentials', 'error')
        return render_template('auth/login.html')

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect(url_for('login'))

    @app.route('/branches/select', methods=['POST'])
    def select_branch():
        requested_scope = request.form.get('branch_scope', '').strip()
        if requested_scope == ALL_BRANCHES_SCOPE:
            if not can_view_all_branches():
                flash('You are not allowed to view all branches.', 'error')
            else:
                session['selected_branch_id'] = ALL_BRANCHES_SCOPE
                flash('Showing combined data from all branches.', 'success')
            return redirect(url_for('dashboard'))

        try:
            branch_id = int(requested_scope)
        except (TypeError, ValueError):
            flash('Invalid branch selection.', 'error')
            return redirect(url_for('dashboard'))

        branch = Branch.query.filter_by(id=branch_id, is_active=True).first()
        if branch is None:
            flash('Selected branch is not available.', 'error')
            return redirect(url_for('dashboard'))

        if not can_view_all_branches() and branch.id != session.get('branch_id'):
            flash('You are not allowed to access that branch.', 'error')
            return redirect(url_for('dashboard'))

        session['selected_branch_id'] = branch.id
        flash(f'Showing {branch.name}.', 'success')
        return redirect(url_for('dashboard'))

    @app.route('/branches')
    def branches():
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        branch_rows = []
        for branch in Branch.query.order_by(Branch.is_main.desc(), Branch.name.asc()).all():
            branch_rows.append({
                'branch': branch,
                'patients': Patient.query.filter_by(branch_id=branch.id).count(),
                'appointments': Appointment.query.filter_by(branch_id=branch.id).count(),
                'consultations': ConsultationRecord.query.filter_by(branch_id=branch.id).count(),
                'staff': StaffMember.query.filter_by(branch_id=branch.id, is_active=True).count(),
                'users': User.query.filter_by(branch_id=branch.id).count(),
            })

        return render_template(
            'branches/index.html',
            branch_rows=branch_rows,
            users=User.query.order_by(User.username.asc()).all(),
            active_branches=Branch.query.filter_by(is_active=True).order_by(Branch.name.asc()).all(),
            role_options=USER_ROLE_OPTIONS,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/branches/users/new', methods=['GET', 'POST'])
    def create_user():
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        active_branches = Branch.query.filter_by(is_active=True).order_by(Branch.name.asc()).all()
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            role = request.form.get('role', 'staff').strip()
            try:
                branch_id = int(request.form.get('branch_id'))
            except (TypeError, ValueError):
                flash('Please select a valid branch for the user.', 'error')
                return redirect(url_for('create_user'))

            branch = Branch.query.filter_by(id=branch_id, is_active=True).first()
            if not username or not password:
                flash('Username and password are required.', 'error')
                return redirect(url_for('create_user'))
            if role not in USER_ROLE_OPTIONS:
                flash('Selected role is not valid.', 'error')
                return redirect(url_for('create_user'))
            if branch is None:
                flash('Selected branch is not active.', 'error')
                return redirect(url_for('create_user'))
            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'error')
                return redirect(url_for('create_user'))

            db.session.add(User(
                username=username,
                password=password,
                role=role,
                branch_id=branch.id,
            ))
            db.session.commit()
            flash('User added successfully.', 'success')
            return redirect(url_for('branches'))

        return render_template(
            'branches/user_form.html',
            active_branches=active_branches,
            role_options=USER_ROLE_OPTIONS,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/branches/users/<int:user_id>/assign', methods=['POST'])
    def assign_user_branch(user_id):
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        user = User.query.get_or_404(user_id)
        try:
            branch_id = int(request.form.get('branch_id'))
        except (TypeError, ValueError):
            flash('Please select a valid branch for the user.', 'error')
            return redirect(url_for('branches'))

        branch = Branch.query.filter_by(id=branch_id, is_active=True).first()
        role = request.form.get('role', user.role).strip()
        if branch is None:
            flash('Selected branch is not active.', 'error')
            return redirect(url_for('branches'))
        if role not in USER_ROLE_OPTIONS:
            flash('Selected role is not valid.', 'error')
            return redirect(url_for('branches'))
        if user.id == session.get('user_id') and role not in MAIN_ADMIN_ROLES:
            flash('You cannot remove your own main administrator access.', 'error')
            return redirect(url_for('branches'))

        user.branch_id = branch.id
        user.role = role
        db.session.commit()
        if user.id == session.get('user_id'):
            session['branch_id'] = branch.id
            session['role'] = role
            if not can_view_all_branches() or session.get('selected_branch_id') != ALL_BRANCHES_SCOPE:
                session['selected_branch_id'] = branch.id
        flash('User branch access updated.', 'success')
        return redirect(url_for('branches'))

    @app.route('/branches/new', methods=['GET', 'POST'])
    def create_branch():
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        if request.method == 'POST':
            name = request.form.get('name', '').strip()
            code = request.form.get('code', '').strip().upper()
            address = request.form.get('address', '').strip()
            email = request.form.get('email', '').strip()
            contact_number = request.form.get('contact_number', '').strip()
            if not name or not code:
                flash('Branch name and code are required.', 'error')
                return redirect(url_for('create_branch'))
            if Branch.query.filter_by(code=code).first():
                flash('Branch code already exists.', 'error')
                return redirect(url_for('create_branch'))
            branch = Branch(
                name=name,
                code=code,
                address=address,
                email=email,
                contact_number=contact_number,
                is_main=False,
                is_active=True,
            )
            db.session.add(branch)
            db.session.commit()
            ensure_branch_service_rows(branch.id)
            ensure_branch_package_rows(branch.id)
            flash('Branch added successfully.', 'success')
            return redirect(url_for('branches'))

        return render_template(
            'branches/form.html',
            branch=None,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/branches/<int:branch_id>/edit', methods=['GET', 'POST'])
    def edit_branch(branch_id):
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        branch = Branch.query.get_or_404(branch_id)
        if request.method == 'POST':
            name = request.form.get('name', '').strip()
            code = branch.code if branch.is_main else request.form.get('code', '').strip().upper()
            address = request.form.get('address', '').strip()
            email = request.form.get('email', '').strip()
            contact_number = request.form.get('contact_number', '').strip()
            if not name or not code:
                flash('Branch name and code are required.', 'error')
                return redirect(url_for('edit_branch', branch_id=branch.id))
            duplicate = Branch.query.filter(Branch.code == code, Branch.id != branch.id).first()
            if duplicate:
                flash('Branch code already exists.', 'error')
                return redirect(url_for('edit_branch', branch_id=branch.id))
            branch.name = name
            branch.code = code
            branch.address = address
            branch.email = email
            branch.contact_number = contact_number
            db.session.commit()
            get_dashboard_summary(force_refresh=True, branch_id=branch.id)
            flash('Branch updated successfully.', 'success')
            return redirect(url_for('branches'))

        return render_template(
            'branches/form.html',
            branch=branch,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/branches/<int:branch_id>/toggle', methods=['POST'])
    def toggle_branch(branch_id):
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        branch = Branch.query.get_or_404(branch_id)
        if branch.is_main and branch.is_active:
            flash('The main branch cannot be deactivated.', 'error')
            return redirect(url_for('branches'))
        branch.is_active = not branch.is_active
        db.session.commit()
        if session.get('selected_branch_id') == branch.id and not branch.is_active:
            session['selected_branch_id'] = session.get('branch_id') or ensure_default_branch().id
        flash('Branch status updated.', 'success')
        return redirect(url_for('branches'))

    def service_role_options():
        return list(dict.fromkeys(
            [item['role'] for item in FACILITY_STAFF_COMPLEMENT] +
            ['Drug Test Analysts', 'HIV Counselor', 'Trained ECG Staff']
        ))

    def service_equipment_options():
        return list(dict.fromkeys(
            EQUIPMENT_INVENTORY +
            ['Clinical microscopy laboratory resources', 'Drug testing laboratory resources', 'Home service kit']
        ))

    def service_category_options():
        defaults = [
            'Medical Consultation',
            'Laboratory',
            'Drug Testing',
            'Imaging/Cardiology',
            'Vaccine',
            'Home Service',
            'Pre-Employment Checkup',
            'Annual Checkup',
        ]
        existing = [
            row[0]
            for row in ClinicService.query.with_entities(ClinicService.category).distinct().all()
            if row[0]
        ]
        return sorted(set(defaults + existing))

    def service_form_values(service=None):
        roles = request.form.getlist('required_roles')
        roles.extend(request.form.get('custom_roles', '').split(','))
        equipment = request.form.getlist('required_equipment')
        equipment.extend(request.form.get('custom_equipment', '').split(','))
        return {
            'category': clean_service_catalog_value(request.form.get('category', service.category if service else '')).strip(),
            'section': clean_service_catalog_value(request.form.get('section', service.section if service else '')).strip(),
            'service_name': clean_service_catalog_value(request.form.get('service_name', service.service_name if service else '')).strip(),
            'price_php': clean_service_catalog_value(request.form.get('price_php', service.price_php if service else '')).strip(),
            'source_page': clean_service_catalog_value(request.form.get('source_page', service.source_page if service else 'manual')).strip(),
            'required_roles': unique_list(roles),
            'required_equipment': unique_list(equipment),
            'is_active': request.form.get('is_active') == 'on',
        }

    @app.route('/services')
    def services():
        redirect_response = require_service_manager()
        if redirect_response:
            return redirect_response

        if not is_superadmin_user() and selected_branch_scope() is None:
            session['selected_branch_id'] = session.get('branch_id') or ensure_default_branch().id

        page = request.args.get('page', 1, type=int)
        search_query = request.args.get('q', '').strip()
        selected_category = request.args.get('category', '').strip()
        categories = service_category_options()
        branch_scope = selected_branch_scope()

        if is_superadmin_user() and branch_scope is None:
            query = ClinicService.query
            if search_query:
                like = f'%{search_query}%'
                query = query.filter(or_(
                    ClinicService.service_name.ilike(like),
                    ClinicService.section.ilike(like),
                    ClinicService.category.ilike(like),
                ))
            if selected_category:
                query = query.filter(ClinicService.category == selected_category)
            service_records = (
                query
                .order_by(ClinicService.category.asc(), ClinicService.section.asc(), ClinicService.service_name.asc())
                .paginate(page=page, per_page=25, error_out=False)
            )
            return render_template(
                'services/index.html',
                mode='global',
                services=service_records,
                categories=categories,
                selected_category=selected_category,
                search_query=search_query,
                branch=None,
                current_date=datetime.now().strftime('%Y-%m-%d'),
                current_time=datetime.now().strftime('%H:%M'),
            )

        branch = Branch.query.get(branch_scope) or ensure_default_branch()
        ensure_branch_service_rows(branch.id)
        query = (
            BranchService.query
            .join(ClinicService)
            .filter(BranchService.branch_id == branch.id)
        )
        if search_query:
            like = f'%{search_query}%'
            query = query.filter(or_(
                ClinicService.service_name.ilike(like),
                ClinicService.section.ilike(like),
                ClinicService.category.ilike(like),
            ))
        if selected_category:
            query = query.filter(ClinicService.category == selected_category)
        branch_services = (
            query
            .order_by(ClinicService.category.asc(), ClinicService.section.asc(), ClinicService.service_name.asc())
            .paginate(page=page, per_page=25, error_out=False)
        )
        return render_template(
            'services/index.html',
            mode='branch',
            services=branch_services,
            categories=categories,
            selected_category=selected_category,
            search_query=search_query,
            branch=branch,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/services/import', methods=['POST'])
    def import_services():
        redirect_response = require_service_manager()
        if redirect_response:
            return redirect_response
        if not is_superadmin_user():
            flash('Only the superadmin can sync the global service catalog.', 'error')
            return redirect(url_for('services'))

        added_count = import_accudetek_service_catalog(app, seed_only=False)
        for branch in Branch.query.all():
            ensure_branch_service_rows(branch.id)
        invalidate_all_dashboard_caches()
        flash(f'Website service catalog sync complete. Added {added_count} new service(s).', 'success')
        return redirect(url_for('services'))

    @app.route('/services/new', methods=['GET', 'POST'])
    def create_service():
        redirect_response = require_service_manager()
        if redirect_response:
            return redirect_response
        if not is_superadmin_user():
            flash('Only the superadmin can add global services.', 'error')
            return redirect(url_for('services'))

        if request.method == 'POST':
            values = service_form_values()
            if not values['category'] or not values['service_name']:
                flash('Category and service name are required.', 'error')
                return redirect(url_for('create_service'))
            duplicate = ClinicService.query.filter_by(
                category=values['category'],
                section=values['section'],
                service_name=values['service_name'],
            ).first()
            if duplicate:
                flash('That service already exists in the global catalog.', 'error')
                return redirect(url_for('create_service'))
            service = ClinicService(
                category=values['category'],
                section=values['section'],
                service_name=values['service_name'],
                price_php=values['price_php'],
                required_roles=json_list(values['required_roles']),
                required_equipment=json_list(values['required_equipment']),
                source_page=values['source_page'] or 'manual',
                is_active=values['is_active'],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            db.session.add(service)
            db.session.commit()
            for branch in Branch.query.all():
                ensure_branch_service_rows(branch.id)
            invalidate_all_dashboard_caches()
            flash('Global service added successfully.', 'success')
            return redirect(url_for('services'))

        return render_template(
            'services/form.html',
            service=None,
            categories=service_category_options(),
            role_options=service_role_options(),
            equipment_options=service_equipment_options(),
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/services/<int:service_id>/edit', methods=['GET', 'POST'])
    def edit_service(service_id):
        redirect_response = require_service_manager()
        if redirect_response:
            return redirect_response
        if not is_superadmin_user():
            flash('Only the superadmin can edit global services.', 'error')
            return redirect(url_for('services'))

        service = ClinicService.query.get_or_404(service_id)
        if request.method == 'POST':
            values = service_form_values(service)
            if not values['category'] or not values['service_name']:
                flash('Category and service name are required.', 'error')
                return redirect(url_for('edit_service', service_id=service.id))
            duplicate = ClinicService.query.filter(
                ClinicService.category == values['category'],
                ClinicService.section == values['section'],
                ClinicService.service_name == values['service_name'],
                ClinicService.id != service.id,
            ).first()
            if duplicate:
                flash('Another global service already uses those details.', 'error')
                return redirect(url_for('edit_service', service_id=service.id))

            service.category = values['category']
            service.section = values['section']
            service.service_name = values['service_name']
            service.price_php = values['price_php']
            service.required_roles = json_list(values['required_roles'])
            service.required_equipment = json_list(values['required_equipment'])
            service.source_page = values['source_page'] or service.source_page
            service.is_active = values['is_active']
            service.updated_at = datetime.now()
            db.session.commit()
            invalidate_all_dashboard_caches()
            flash('Global service updated successfully.', 'success')
            return redirect(url_for('services'))

        return render_template(
            'services/form.html',
            service=service,
            categories=service_category_options(),
            role_options=service_role_options(),
            equipment_options=service_equipment_options(),
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/services/<int:service_id>/branch', methods=['POST'])
    def update_branch_service(service_id):
        redirect_response = require_service_manager()
        if redirect_response:
            return redirect_response
        redirect_response = require_specific_branch('services')
        if redirect_response:
            return redirect_response

        service = ClinicService.query.get_or_404(service_id)
        branch_id = current_branch_id()
        setting = BranchService.query.filter_by(branch_id=branch_id, service_id=service.id).first()
        if setting is None:
            setting = BranchService(branch_id=branch_id, service_id=service.id)
            db.session.add(setting)

        setting.custom_price_php = request.form.get('custom_price_php', '').strip()
        setting.is_available = request.form.get('is_available') == 'on'
        setting.branch_notes = request.form.get('branch_notes', '').strip()
        setting.updated_at = datetime.now()
        db.session.commit()
        invalidate_dashboard_caches(branch_id)
        flash(f'Branch service setting updated for {service.service_name}.', 'success')
        return redirect(url_for(
            'services',
            page=request.form.get('page', 1),
            q=request.form.get('q', ''),
            category=request.form.get('category', ''),
        ))

    def package_form_values(package=None):
        item_text = request.form.get('included_services', '').strip()
        item_names = [
            clean_service_catalog_value(line.strip())
            for line in item_text.splitlines()
            if line.strip()
        ]
        return {
            'package_name': clean_service_catalog_value(request.form.get('package_name', package.package_name if package else '')).strip(),
            'price_php': clean_service_catalog_value(request.form.get('price_php', package.price_php if package else '')).strip(),
            'source_page': clean_service_catalog_value(request.form.get('source_page', package.source_page if package else 'manual')).strip(),
            'is_active': request.form.get('is_active') == 'on',
            'item_names': unique_list(item_names),
        }

    @app.route('/packages')
    def packages():
        redirect_response = require_service_manager()
        if redirect_response:
            return redirect_response

        if not is_superadmin_user() and selected_branch_scope() is None:
            session['selected_branch_id'] = session.get('branch_id') or ensure_default_branch().id

        page = request.args.get('page', 1, type=int)
        search_query = request.args.get('q', '').strip()
        branch_scope = selected_branch_scope()

        if is_superadmin_user() and branch_scope is None:
            query = ClinicPackage.query
            if search_query:
                query = query.filter(ClinicPackage.package_name.ilike(f'%{search_query}%'))
            package_records = (
                query
                .order_by(ClinicPackage.package_name.asc())
                .paginate(page=page, per_page=20, error_out=False)
            )
            return render_template(
                'packages/index.html',
                mode='global',
                packages=package_records,
                search_query=search_query,
                branch=None,
                package_requirements=package_requirements,
                current_date=datetime.now().strftime('%Y-%m-%d'),
                current_time=datetime.now().strftime('%H:%M'),
            )

        branch = Branch.query.get(branch_scope) or ensure_default_branch()
        ensure_branch_package_rows(branch.id)
        query = (
            BranchPackage.query
            .join(ClinicPackage)
            .filter(BranchPackage.branch_id == branch.id)
        )
        if search_query:
            query = query.filter(ClinicPackage.package_name.ilike(f'%{search_query}%'))
        branch_packages = (
            query
            .order_by(ClinicPackage.package_name.asc())
            .paginate(page=page, per_page=20, error_out=False)
        )
        return render_template(
            'packages/index.html',
            mode='branch',
            packages=branch_packages,
            search_query=search_query,
            branch=branch,
            package_requirements=package_requirements,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/packages/import', methods=['POST'])
    def import_packages():
        redirect_response = require_service_manager()
        if redirect_response:
            return redirect_response
        if not is_superadmin_user():
            flash('Only the superadmin can sync the global package catalog.', 'error')
            return redirect(url_for('packages'))

        added_count = import_accudetek_package_catalog(app, seed_only=False)
        for branch in Branch.query.all():
            ensure_branch_package_rows(branch.id)
        invalidate_all_dashboard_caches()
        flash(f'Website package catalog sync complete. Added {added_count} new package(s).', 'success')
        return redirect(url_for('packages'))

    @app.route('/packages/new', methods=['GET', 'POST'])
    def create_package():
        redirect_response = require_service_manager()
        if redirect_response:
            return redirect_response
        if not is_superadmin_user():
            flash('Only the superadmin can add global packages.', 'error')
            return redirect(url_for('packages'))

        if request.method == 'POST':
            values = package_form_values()
            if not values['package_name']:
                flash('Package name is required.', 'error')
                return redirect(url_for('create_package'))
            if not values['item_names']:
                flash('Enter at least one included service item.', 'error')
                return redirect(url_for('create_package'))
            if ClinicPackage.query.filter_by(package_name=values['package_name']).first():
                flash('That package already exists.', 'error')
                return redirect(url_for('create_package'))

            package = ClinicPackage(
                package_name=values['package_name'],
                price_php=values['price_php'],
                source_page=values['source_page'] or 'manual',
                is_active=values['is_active'],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            db.session.add(package)
            db.session.flush()
            replace_package_items(package, values['item_names'])
            db.session.commit()
            for branch in Branch.query.all():
                ensure_branch_package_rows(branch.id)
            invalidate_all_dashboard_caches()
            flash('Global package added successfully.', 'success')
            return redirect(url_for('packages'))

        return render_template(
            'packages/form.html',
            package=None,
            item_text='',
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/packages/<int:package_id>/edit', methods=['GET', 'POST'])
    def edit_package(package_id):
        redirect_response = require_service_manager()
        if redirect_response:
            return redirect_response
        if not is_superadmin_user():
            flash('Only the superadmin can edit global packages.', 'error')
            return redirect(url_for('packages'))

        package = ClinicPackage.query.get_or_404(package_id)
        if request.method == 'POST':
            values = package_form_values(package)
            if not values['package_name']:
                flash('Package name is required.', 'error')
                return redirect(url_for('edit_package', package_id=package.id))
            if not values['item_names']:
                flash('Enter at least one included service item.', 'error')
                return redirect(url_for('edit_package', package_id=package.id))
            duplicate = ClinicPackage.query.filter(
                ClinicPackage.package_name == values['package_name'],
                ClinicPackage.id != package.id,
            ).first()
            if duplicate:
                flash('Another package already uses that name.', 'error')
                return redirect(url_for('edit_package', package_id=package.id))

            package.package_name = values['package_name']
            package.price_php = values['price_php']
            package.source_page = values['source_page'] or package.source_page
            package.is_active = values['is_active']
            package.updated_at = datetime.now()
            replace_package_items(package, values['item_names'])
            db.session.commit()
            invalidate_all_dashboard_caches()
            flash('Global package updated successfully.', 'success')
            return redirect(url_for('packages'))

        item_text = '\n'.join(
            item.item_name
            for item in sorted(package.items, key=lambda entry: entry.item_order)
        )
        return render_template(
            'packages/form.html',
            package=package,
            item_text=item_text,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/packages/<int:package_id>/branch', methods=['POST'])
    def update_branch_package(package_id):
        redirect_response = require_service_manager()
        if redirect_response:
            return redirect_response
        redirect_response = require_specific_branch('packages')
        if redirect_response:
            return redirect_response

        package = ClinicPackage.query.get_or_404(package_id)
        branch_id = current_branch_id()
        setting = BranchPackage.query.filter_by(branch_id=branch_id, package_id=package.id).first()
        if setting is None:
            setting = BranchPackage(branch_id=branch_id, package_id=package.id)
            db.session.add(setting)

        setting.custom_price_php = request.form.get('custom_price_php', '').strip()
        setting.is_available = request.form.get('is_available') == 'on'
        setting.branch_notes = request.form.get('branch_notes', '').strip()
        setting.updated_at = datetime.now()
        db.session.commit()
        invalidate_dashboard_caches(branch_id)
        flash(f'Branch package setting updated for {package.package_name}.', 'success')
        return redirect(url_for(
            'packages',
            page=request.form.get('page', 1),
            q=request.form.get('q', ''),
        ))

    @app.route('/dashboard')
    def dashboard():
        summary = get_dashboard_summary()
        return render_template(
            'dashboard/index.html',
            summary=summary,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/patients')
    def patients():
        page = request.args.get('page', 1, type=int)
        search_query = request.args.get('q', '').strip()
        selected_gender = request.args.get('gender', '').strip()
        selected_age_group = request.args.get('age_group', '').strip()
        selected_status = request.args.get('status', 'active').strip() or 'active'

        query = scoped_query(Patient.query, Patient)
        base_query = query

        if selected_status == 'active':
            query = query.filter(Patient.is_active.is_(True))
        elif selected_status == 'inactive':
            query = query.filter(Patient.is_active.is_(False))
        elif selected_status != 'all':
            selected_status = 'active'
            query = query.filter(Patient.is_active.is_(True))

        if selected_gender:
            query = query.filter(Patient.gender == selected_gender)
        if selected_age_group:
            query = query.filter(Patient.age_group == selected_age_group)
        if search_query:
            like_query = f'%{search_query}%'
            query = query.filter(or_(
                Patient.patient_number.ilike(like_query),
                Patient.full_name.ilike(like_query),
                Patient.contact_number.ilike(like_query),
                Patient.email.ilike(like_query),
            ))

        patient_records = (
            query
            .order_by(Patient.is_active.desc(), Patient.updated_at.desc(), Patient.full_name.asc())
            .paginate(page=page, per_page=10, error_out=False)
        )

        active_count = base_query.filter(Patient.is_active.is_(True)).count()
        inactive_count = base_query.filter(Patient.is_active.is_(False)).count()

        return render_template(
            'patients/index.html',
            patients=patient_records,
            search_query=search_query,
            selected_gender=selected_gender,
            selected_age_group=selected_age_group,
            selected_status=selected_status,
            gender_options=PATIENT_GENDER_OPTIONS,
            age_group_options=PATIENT_AGE_GROUP_OPTIONS,
            active_count=active_count,
            inactive_count=inactive_count,
            all_branches_view=selected_branch_scope() is None,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/patients/new', methods=['GET', 'POST'])
    def create_patient():
        redirect_response = require_specific_branch('patients')
        if redirect_response:
            return redirect_response

        branch = Branch.query.get(current_branch_id()) or ensure_default_branch()
        patient = None
        if request.method == 'POST':
            values = patient_form_values()
            if not values['patient_number']:
                values['patient_number'] = generate_patient_number(branch)

            if not values['full_name']:
                flash('Patient full name is required.', 'error')
                return redirect(url_for('create_patient'))
            if not values['birthdate']:
                flash('Patient birthdate is required so the system can calculate age.', 'error')
                return redirect(url_for('create_patient'))
            age = calculate_age_from_birthdate(values['birthdate'])
            if age is None:
                flash('Please enter a valid birthdate. Age must be from 0 to 130.', 'error')
                return redirect(url_for('create_patient'))
            if values['gender'] not in PATIENT_GENDER_OPTIONS:
                flash('Please select a valid gender.', 'error')
                return redirect(url_for('create_patient'))
            if Patient.query.filter_by(patient_number=values['patient_number']).first():
                flash('Patient number already exists.', 'error')
                return redirect(url_for('create_patient'))

            patient = Patient(
                branch_id=branch.id,
                patient_number=values['patient_number'],
                full_name=values['full_name'],
                birthdate=values['birthdate'],
                age=age,
                age_group=categorize_age(age),
                gender=values['gender'],
                contact_number=values['contact_number'],
                email=values['email'],
                address=values['address'],
                emergency_contact_name=values['emergency_contact_name'],
                emergency_contact_number=values['emergency_contact_number'],
                is_active=True,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            db.session.add(patient)
            db.session.commit()
            flash('Patient record added successfully.', 'success')
            return redirect(url_for('patient_detail', patient_id=patient.id))

        return render_template(
            'patients/form.html',
            patient=patient,
            gender_options=PATIENT_GENDER_OPTIONS,
            age_group_options=PATIENT_AGE_GROUP_OPTIONS,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/patients/<int:patient_id>')
    def patient_detail(patient_id):
        patient = get_scoped_patient_or_404(patient_id)
        linked_consultations = (
            ConsultationRecord.query
            .filter_by(patient_id=patient.id)
            .order_by(ConsultationRecord.consultation_date.desc())
            .limit(10)
            .all()
        )
        consultation_count = ConsultationRecord.query.filter_by(patient_id=patient.id).count()
        patient_appointments = (
            Appointment.query
            .filter_by(patient_id=patient.id)
            .order_by(Appointment.appointment_date.desc(), Appointment.appointment_time.desc())
            .limit(5)
            .all()
        )
        return render_template(
            'patients/detail.html',
            patient=patient,
            linked_consultations=linked_consultations,
            consultation_count=consultation_count,
            patient_appointments=patient_appointments,
            all_branches_view=selected_branch_scope() is None,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/patients/<int:patient_id>/consultations/new', methods=['GET', 'POST'])
    def create_patient_consultation(patient_id):
        redirect_response = require_specific_branch('patients')
        if redirect_response:
            return redirect_response

        patient = Patient.query.filter_by(id=patient_id, branch_id=current_branch_id()).first_or_404()
        if not patient.is_active:
            flash('Reactivate the patient record before adding a consultation.', 'error')
            return redirect(url_for('patient_detail', patient_id=patient.id))

        staff_members = (
            StaffMember.query
            .filter_by(branch_id=current_branch_id(), is_active=True)
            .order_by(StaffMember.role.asc(), StaffMember.name.asc())
            .all()
        )

        if request.method == 'POST':
            values = consultation_form_values(patient)
            consultation_date = parse_iso_date(values['consultation_date'])

            if not values['consultation_date']:
                flash('Consultation date is required.', 'error')
                return redirect(url_for('create_patient_consultation', patient_id=patient.id))
            if consultation_date is None:
                flash('Please enter a valid consultation date.', 'error')
                return redirect(url_for('create_patient_consultation', patient_id=patient.id))
            patient_age = calculate_age_from_birthdate(patient.birthdate, consultation_date)
            if patient_age is None:
                flash('Patient birthdate is missing or invalid, so consultation age cannot be calculated.', 'error')
                return redirect(url_for('create_patient_consultation', patient_id=patient.id))
            if values['gender'] not in PATIENT_GENDER_OPTIONS:
                flash('Please select a valid gender.', 'error')
                return redirect(url_for('create_patient_consultation', patient_id=patient.id))
            if not values['diagnosis']:
                flash('Diagnosis is required.', 'error')
                return redirect(url_for('create_patient_consultation', patient_id=patient.id))
            if not values['physician']:
                flash('Physician or attending staff is required.', 'error')
                return redirect(url_for('create_patient_consultation', patient_id=patient.id))
            if values['consultation_type'] not in CONSULTATION_TYPE_OPTIONS:
                flash('Please select a valid consultation type.', 'error')
                return redirect(url_for('create_patient_consultation', patient_id=patient.id))

            record = ConsultationRecord(
                branch_id=current_branch_id(),
                patient_id=patient.id,
                consultation_date=values['consultation_date'],
                patient_age=patient_age,
                age_group=categorize_age(patient_age),
                gender=values['gender'],
                diagnosis=values['diagnosis'],
                department='Clinic',
                physician=values['physician'],
                consultation_type=values['consultation_type'],
            )
            db.session.add(record)
            db.session.commit()
            get_dashboard_summary(force_refresh=True)
            flash('Consultation linked to patient record successfully.', 'success')
            return redirect(url_for('patient_detail', patient_id=patient.id))

        return render_template(
            'patients/consultation_form.html',
            patient=patient,
            staff_members=staff_members,
            gender_options=PATIENT_GENDER_OPTIONS,
            consultation_type_options=CONSULTATION_TYPE_OPTIONS,
            today=datetime.now().strftime('%Y-%m-%d'),
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/patients/<int:patient_id>/edit', methods=['GET', 'POST'])
    def edit_patient(patient_id):
        redirect_response = require_specific_branch('patients')
        if redirect_response:
            return redirect_response

        patient = Patient.query.filter_by(id=patient_id, branch_id=current_branch_id()).first_or_404()
        if request.method == 'POST':
            values = patient_form_values(patient)
            if not values['patient_number']:
                flash('Patient number is required.', 'error')
                return redirect(url_for('edit_patient', patient_id=patient.id))
            if not values['full_name']:
                flash('Patient full name is required.', 'error')
                return redirect(url_for('edit_patient', patient_id=patient.id))
            if not values['birthdate']:
                flash('Patient birthdate is required so the system can calculate age.', 'error')
                return redirect(url_for('edit_patient', patient_id=patient.id))
            age = calculate_age_from_birthdate(values['birthdate'])
            if age is None:
                flash('Please enter a valid birthdate. Age must be from 0 to 130.', 'error')
                return redirect(url_for('edit_patient', patient_id=patient.id))
            if values['gender'] not in PATIENT_GENDER_OPTIONS:
                flash('Please select a valid gender.', 'error')
                return redirect(url_for('edit_patient', patient_id=patient.id))

            duplicate = Patient.query.filter(
                Patient.patient_number == values['patient_number'],
                Patient.id != patient.id,
            ).first()
            if duplicate:
                flash('Patient number already exists.', 'error')
                return redirect(url_for('edit_patient', patient_id=patient.id))

            patient.patient_number = values['patient_number']
            patient.full_name = values['full_name']
            patient.birthdate = values['birthdate']
            patient.age = age
            patient.age_group = categorize_age(age)
            patient.gender = values['gender']
            patient.contact_number = values['contact_number']
            patient.email = values['email']
            patient.address = values['address']
            patient.emergency_contact_name = values['emergency_contact_name']
            patient.emergency_contact_number = values['emergency_contact_number']
            patient.updated_at = datetime.now()
            db.session.commit()
            flash('Patient record updated successfully.', 'success')
            return redirect(url_for('patient_detail', patient_id=patient.id))

        return render_template(
            'patients/form.html',
            patient=patient,
            gender_options=PATIENT_GENDER_OPTIONS,
            age_group_options=PATIENT_AGE_GROUP_OPTIONS,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/patients/<int:patient_id>/archive', methods=['POST'])
    def archive_patient(patient_id):
        redirect_response = require_specific_branch('patients')
        if redirect_response:
            return redirect_response

        patient = Patient.query.filter_by(id=patient_id, branch_id=current_branch_id()).first_or_404()
        patient.is_active = False
        patient.updated_at = datetime.now()
        db.session.commit()
        flash('Patient record archived.', 'success')
        return redirect(url_for('patients'))

    @app.route('/patients/<int:patient_id>/restore', methods=['POST'])
    def restore_patient(patient_id):
        redirect_response = require_specific_branch('patients')
        if redirect_response:
            return redirect_response

        patient = Patient.query.filter_by(id=patient_id, branch_id=current_branch_id()).first_or_404()
        patient.is_active = True
        patient.updated_at = datetime.now()
        db.session.commit()
        flash('Patient record reactivated.', 'success')
        return redirect(url_for('patient_detail', patient_id=patient.id))

    @app.route('/appointments')
    def appointments():
        page = request.args.get('page', 1, type=int)
        selected_status = request.args.get('status', '').strip()
        selected_date = request.args.get('date', '').strip()

        query = scoped_query(Appointment.query, Appointment)
        if selected_status:
            query = query.filter(Appointment.status == selected_status)
        if selected_date:
            query = query.filter(Appointment.appointment_date == selected_date)

        appointment_records = (
            query
            .order_by(Appointment.appointment_date.desc(), Appointment.appointment_time.desc(), Appointment.created_at.desc())
            .paginate(page=page, per_page=10, error_out=False)
        )

        return render_template(
            'appointments/index.html',
            appointments=appointment_records,
            status_options=APPOINTMENT_STATUS_OPTIONS,
            selected_status=selected_status,
            selected_date=selected_date,
            all_branches_view=selected_branch_scope() is None,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/appointments/new', methods=['GET', 'POST'])
    @app.route('/patients/<int:patient_id>/appointments/new', methods=['GET', 'POST'])
    def create_appointment(patient_id=None):
        redirect_response = require_specific_branch('appointments')
        if redirect_response:
            return redirect_response

        selected_patient = None
        if patient_id is not None:
            selected_patient = Patient.query.filter_by(
                id=patient_id,
                branch_id=current_branch_id(),
                is_active=True,
            ).first_or_404()

        active_patients = (
            Patient.query
            .filter_by(branch_id=current_branch_id(), is_active=True)
            .order_by(Patient.full_name.asc())
            .all()
        )
        service_options = get_branch_service_options(current_branch_id(), only_available=True)
        available_service_values = {option['value'] for option in service_options}
        package_options = get_branch_package_options(current_branch_id(), only_available=True)
        available_package_values = {option['value'] for option in package_options}

        if request.method == 'POST':
            values = appointment_form_values()
            patient = selected_patient
            if patient is None:
                try:
                    form_patient_id = int(values['patient_id'])
                except (TypeError, ValueError):
                    flash('Please select a valid patient.', 'error')
                    return redirect(url_for('create_appointment'))
                patient = Patient.query.filter_by(
                    id=form_patient_id,
                    branch_id=current_branch_id(),
                    is_active=True,
                ).first()

            appointment_date = parse_iso_date(values['appointment_date'])
            if patient is None:
                flash('Please select an active patient for the appointment.', 'error')
                return redirect(url_for('create_appointment'))
            if appointment_date is None:
                flash('Please enter a valid appointment date.', 'error')
                return redirect(url_for('create_appointment', patient_id=patient.id) if selected_patient else url_for('create_appointment'))
            if not values['appointment_time']:
                flash('Please choose an appointment time.', 'error')
                return redirect(url_for('create_appointment', patient_id=patient.id) if selected_patient else url_for('create_appointment'))
            if appointment_slot_conflict(current_branch_id(), values['appointment_date'], values['appointment_time']):
                flash('That appointment time is already taken for this branch. Please choose another time.', 'error')
                return redirect(url_for('create_appointment', patient_id=patient.id) if selected_patient else url_for('create_appointment'))
            if not values['selected_services'] and not values['selected_packages']:
                flash('Select at least one clinic service or package.', 'error')
                return redirect(url_for('create_appointment', patient_id=patient.id) if selected_patient else url_for('create_appointment'))
            unavailable_services = [
                service for service in values['selected_services']
                if service not in available_service_values
            ]
            if unavailable_services:
                flash('One or more selected services are not available for this branch.', 'error')
                return redirect(url_for('create_appointment', patient_id=patient.id) if selected_patient else url_for('create_appointment'))
            unavailable_packages = [
                package for package in values['selected_packages']
                if package not in available_package_values
            ]
            if unavailable_packages:
                flash('One or more selected packages are not available for this branch.', 'error')
                return redirect(url_for('create_appointment', patient_id=patient.id) if selected_patient else url_for('create_appointment'))
            includes_consultation = any('consultation' in service.lower() for service in values['selected_services'])
            if includes_consultation and not values['consultation_reasons'] and not values['other_reason']:
                flash('Select or enter a consultation reason for physician consultation.', 'error')
                return redirect(url_for('create_appointment', patient_id=patient.id) if selected_patient else url_for('create_appointment'))

            roles, equipment, notes = build_appointment_recommendation(
                values['selected_services'],
                values['consultation_reasons'],
                values['other_reason'],
                values['selected_packages'],
            )

            appointment = Appointment(
                branch_id=current_branch_id(),
                patient_id=patient.id,
                appointment_date=values['appointment_date'],
                appointment_time=values['appointment_time'],
                selected_services=json_list(values['selected_services']),
                selected_packages=json_list(values['selected_packages']),
                consultation_reasons=json_list(values['consultation_reasons']),
                other_reason=values['other_reason'],
                recommended_roles=json_list(roles),
                recommended_equipment=json_list(equipment),
                recommendation_notes=json_list(notes),
                status='Pending',
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            db.session.add(appointment)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                flash('That appointment time was just taken. Please choose another time.', 'error')
                return redirect(url_for('create_appointment', patient_id=patient.id) if selected_patient else url_for('create_appointment'))
            invalidate_dashboard_caches(current_branch_id())
            flash('Appointment booked with staff recommendation.', 'success')
            return redirect(url_for('appointments'))

        return render_template(
            'appointments/form.html',
            selected_patient=selected_patient,
            active_patients=active_patients,
            service_options=service_options,
            service_groups=service_option_groups(service_options),
            package_options=package_options,
            consultation_reason_options=COMMON_CONSULTATION_REASONS,
            service_recommendations=service_recommendation_map_from_options(service_options),
            package_recommendations=package_recommendation_map_from_options(package_options),
            reason_recommendations=REASON_RECOMMENDATION_MAP,
            keyword_recommendations=REASON_KEYWORD_RECOMMENDATIONS,
            today=datetime.now().strftime('%Y-%m-%d'),
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/appointments/<int:appointment_id>/status', methods=['POST'])
    def update_appointment_status(appointment_id):
        redirect_response = require_specific_branch('appointments')
        if redirect_response:
            return redirect_response

        appointment = Appointment.query.filter_by(id=appointment_id, branch_id=current_branch_id()).first_or_404()
        status = request.form.get('status', appointment.status).strip()
        if status not in APPOINTMENT_STATUS_OPTIONS:
            flash('Selected appointment status is not valid.', 'error')
            return redirect(url_for('appointments'))

        if appointment.converted_to_records and status != 'Completed':
            flash('Completed appointments already converted to consultation records cannot be moved back.', 'error')
            return redirect(url_for('appointments'))

        if status == 'Completed' and not appointment.converted_to_records:
            return redirect(url_for('complete_appointment', appointment_id=appointment.id))

        if status in ACTIVE_APPOINTMENT_SLOT_STATUSES and appointment_slot_conflict(
            appointment.branch_id,
            appointment.appointment_date,
            appointment.appointment_time,
            exclude_id=appointment.id,
        ):
            flash('That appointment time is already taken for this branch. Choose another slot before reactivating this appointment.', 'error')
            return redirect(url_for('appointments'))

        appointment.status = status
        appointment.updated_at = datetime.now()
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            flash('That appointment time was just taken. Please choose another time.', 'error')
            return redirect(url_for('appointments'))
        invalidate_dashboard_caches(current_branch_id())
        flash('Appointment status updated.', 'success')
        return redirect(url_for('appointments'))

    @app.route('/appointments/<int:appointment_id>/complete', methods=['GET', 'POST'])
    def complete_appointment(appointment_id):
        redirect_response = require_specific_branch('appointments')
        if redirect_response:
            return redirect_response

        appointment = Appointment.query.filter_by(id=appointment_id, branch_id=current_branch_id()).first_or_404()
        patient = appointment.patient
        if patient is None:
            flash('This appointment has no linked patient record.', 'error')
            return redirect(url_for('appointments'))
        if appointment.converted_to_records:
            flash('This appointment was already converted to consultation records.', 'warning')
            return redirect(url_for('appointments'))

        completion_plan = appointment_completion_items(appointment)
        if not completion_plan:
            flash('This appointment has no selected services or package items to complete.', 'error')
            return redirect(url_for('appointments'))

        staff_members = (
            StaffMember.query
            .filter_by(branch_id=current_branch_id(), is_active=True)
            .order_by(StaffMember.role.asc(), StaffMember.name.asc())
            .all()
        )
        service_roles = {
            item['label']: item['roles']
            for item in completion_plan
        }
        service_departments = {
            item['label']: item['department']
            for item in completion_plan
        }
        service_types = {
            item['label']: item['consultation_type']
            for item in completion_plan
        }
        default_diagnosis = default_appointment_diagnosis(appointment)

        if request.method == 'POST':
            completed_date_value = request.form.get('completed_date', appointment.appointment_date).strip()
            completed_date = parse_iso_date(completed_date_value)
            if completed_date is None:
                flash('Please enter a valid completed date.', 'error')
                return redirect(url_for('complete_appointment', appointment_id=appointment.id))

            patient_age = calculate_age_from_birthdate(patient.birthdate, completed_date)
            if patient_age is None:
                flash('Patient birthdate is missing or invalid, so consultation age cannot be calculated.', 'error')
                return redirect(url_for('complete_appointment', appointment_id=appointment.id))
            if patient.gender not in PATIENT_GENDER_OPTIONS:
                flash('Patient gender is missing or invalid.', 'error')
                return redirect(url_for('complete_appointment', appointment_id=appointment.id))

            completion_items = []
            for index, item in enumerate(completion_plan):
                service = item['label']
                assigned_staff = request.form.get(f'assigned_staff_{index}', '').strip()
                final_diagnosis = request.form.get(f'final_diagnosis_{index}', default_diagnosis).strip()
                service_notes = request.form.get(f'service_notes_{index}', '').strip()

                if not assigned_staff:
                    flash(f'Assigned staff is required for {service}.', 'error')
                    return redirect(url_for('complete_appointment', appointment_id=appointment.id))
                if not final_diagnosis:
                    flash(f'Final diagnosis or case is required for {service}.', 'error')
                    return redirect(url_for('complete_appointment', appointment_id=appointment.id))

                completion_items.append({
                    'service': service,
                    'assigned_staff': assigned_staff,
                    'final_diagnosis': final_diagnosis,
                    'service_notes': service_notes,
                    'department': item['department'],
                    'consultation_type': item['consultation_type'],
                })

            for item in completion_items:
                record = ConsultationRecord(
                    branch_id=current_branch_id(),
                    patient_id=patient.id,
                    consultation_date=completed_date_value,
                    patient_age=patient_age,
                    age_group=categorize_age(patient_age),
                    gender=patient.gender,
                    diagnosis=item['final_diagnosis'],
                    department=item['department'],
                    physician=item['assigned_staff'],
                    consultation_type=item['consultation_type'],
                )
                db.session.add(record)
                db.session.flush()
                db.session.add(AppointmentServiceResult(
                    appointment_id=appointment.id,
                    consultation_record_id=record.id,
                    service_name=item['service'],
                    assigned_staff=item['assigned_staff'],
                    final_diagnosis=item['final_diagnosis'],
                    service_notes=item['service_notes'],
                ))

            appointment.status = 'Completed'
            appointment.completed_at = datetime.now()
            appointment.converted_to_records = True
            appointment.completion_notes = request.form.get('completion_notes', '').strip()
            appointment.updated_at = datetime.now()
            db.session.commit()
            get_dashboard_summary(force_refresh=True)
            flash(f'Appointment completed and {len(completion_items)} consultation record(s) were created.', 'success')
            return redirect(url_for('appointments'))

        return render_template(
            'appointments/complete.html',
            appointment=appointment,
            patient=patient,
            services=[item['label'] for item in completion_plan],
            service_roles=service_roles,
            service_departments=service_departments,
            service_types=service_types,
            staff_members=staff_members,
            default_diagnosis=default_diagnosis,
            completed_date=appointment.appointment_date,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/resources')
    def resources():
        summary = get_dashboard_summary()
        return render_template(
            'resources/index.html',
            summary=summary,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/records')
    def records():
        page = request.args.get('page', 1, type=int)
        records = (
            scoped_query(ConsultationRecord.query, ConsultationRecord)
            .order_by(ConsultationRecord.consultation_date.desc())
            .paginate(page=page, per_page=10, error_out=False)
        )
        return render_template(
            'consultations/index.html',
            records=records,
            all_branches_view=selected_branch_scope() is None,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/records/clear', methods=['POST'])
    def clear_records():
        redirect_response = require_specific_branch('records')
        if redirect_response:
            return redirect_response
        deleted_count = ConsultationRecord.query.filter_by(branch_id=current_branch_id()).delete()
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash(f'Cleared {deleted_count} consultation records for the current branch.', 'success')
        return redirect(url_for('records'))

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if selected_branch_scope() is None:
            flash('Select a specific branch before uploading consultation data.', 'error')
            return redirect(url_for('records'))

        if request.method == 'POST':
            file = request.files.get('file')
            if not file:
                flash('No file selected', 'error')
                return redirect(url_for('upload'))

            filename = file.filename
            if not filename.lower().endswith(('.xlsx', '.csv')):
                flash('Invalid file type. Please upload an Excel or CSV file.', 'error')
                return redirect(url_for('upload'))

            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            try:
                df = pd.read_excel(path) if filename.lower().endswith('.xlsx') else pd.read_csv(path)
            except Exception:
                flash('Unable to read the file. Please upload a valid Excel or CSV file.', 'error')
                return redirect(url_for('upload'))

            required_columns = {'consultation_date', 'age_group', 'gender', 'diagnosis', 'department', 'physician', 'consultation_type'}
            missing_columns = required_columns - set(df.columns.str.lower())
            if missing_columns:
                flash(f'Missing required columns: {", ".join(sorted(missing_columns))}', 'error')
                return redirect(url_for('upload'))

            df.columns = df.columns.str.lower()
            upload_columns = [
                'consultation_date', 'age_group', 'gender', 'diagnosis',
                'department', 'physician', 'consultation_type'
            ]

            def normalize_record_value(column, value):
                if column == 'consultation_date':
                    parsed = pd.to_datetime(value, errors='coerce')
                    if pd.notna(parsed):
                        return parsed.strftime('%Y-%m-%d')
                return str(value).strip()

            def record_key(values):
                return tuple(normalize_record_value(column, values.get(column, '')) for column in upload_columns)

            existing_keys = {
                record_key({
                    'consultation_date': record.consultation_date,
                    'age_group': record.age_group,
                    'gender': record.gender,
                    'diagnosis': record.diagnosis,
                    'department': record.department,
                    'physician': record.physician,
                    'consultation_type': record.consultation_type,
                })
                for record in ConsultationRecord.query.filter_by(branch_id=current_branch_id()).all()
            }

            added_count = 0
            skipped_count = 0
            for _, row in df.iterrows():
                values = {column: normalize_record_value(column, row.get(column, '')) for column in upload_columns}
                key = tuple(values[column] for column in upload_columns)
                if key in existing_keys:
                    skipped_count += 1
                    continue
                existing_keys.add(key)
                record = ConsultationRecord(
                    branch_id=current_branch_id(),
                    consultation_date=values['consultation_date'],
                    age_group=values['age_group'],
                    gender=values['gender'],
                    diagnosis=values['diagnosis'],
                    department=values['department'],
                    physician=values['physician'],
                    consultation_type=values['consultation_type'],
                )
                db.session.add(record)
                added_count += 1
            db.session.commit()

            records = ConsultationRecord.query.filter_by(branch_id=current_branch_id()).all()
            if records:
                df = pd.DataFrame([{
                    'consultation_date': r.consultation_date,
                    'age_group': r.age_group,
                    'gender': r.gender,
                    'diagnosis': r.diagnosis,
                    'department': r.department,
                    'physician': r.physician,
                    'consultation_type': r.consultation_type,
                } for r in records])
                try:
                    _, metrics, _, _ = train_and_evaluate_model(df)
                    report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'training_report.txt')
                    with open(report_path, 'w', encoding='utf-8') as handle:
                        handle.write('Smart Healthcare Clinic Management - Enhanced Training Report\n')
                        handle.write('==========================================================\n')
                        handle.write(f"R2 Score: {metrics['r2_score']}\n")
                        handle.write(f"MAE: {metrics['mae']}\n")
                        handle.write(f"MSE: {metrics['mse']}\n")
                        handle.write(f"RMSE: {metrics['rmse']}\n")
                        if metrics.get('cv_r2_mean'):
                            handle.write(f"CV R2 Mean: {metrics['cv_r2_mean']} (±{metrics['cv_r2_std']})\n")
                        handle.write(f"Best Params: {metrics['best_params']}\n")
                    write_training_report(report_path, metrics)
                    session['last_report'] = report_path
                except Exception as e:
                    traceback.print_exc()
                    flash(f'Model retraining failed: {str(e)}', 'error')
            else:
                flash('No records to train model.', 'warning')

            get_dashboard_summary(force_refresh=True)
            flash(f'Data uploaded and model retrained successfully. Added {added_count} new records; skipped {skipped_count} duplicates.', 'success')
            return redirect(url_for('predict'))

        return render_template('consultations/upload.html')

    @app.route('/predict')
    def predict():
        summary = get_dashboard_summary()
        if not summary.get('total_consultations'):
            flash('No consultation records found. Please upload data first.', 'warning')
            return render_template('forecasting/index.html',
                                   metrics={'r2_score': 0, 'mae': 0, 'mse': 0, 'rmse': 0},
                                   top_cases=[],
                                   forecast=[],
                                   summary={
                                       'predicted_month_full_label': 'Next Month',
                                       'predictions': [],
                                       'demographic_forecast': {'age_group': [], 'gender': [], 'segments': []}
                                   })

        metrics = summary.get('rf_metrics') or {'r2_score': 0, 'mae': 0, 'mse': 0, 'rmse': 0}
        top_cases = [f"{diag} ({count})" for diag, count in summary.get('top_cases', [])]
        forecast = [
            (item.get('diagnosis'), item.get('predicted_next_month'))
            for item in summary.get('predictions', [])
        ]

        return render_template('forecasting/index.html',
                               metrics=metrics,
                               top_cases=top_cases,
                               forecast=forecast,
                               summary=summary)

    @app.route('/retrain', methods=['POST'])
    def retrain():
        redirect_response = require_specific_branch('predict')
        if redirect_response:
            return redirect_response
        records = ConsultationRecord.query.filter_by(branch_id=current_branch_id()).all()
        if not records:
            flash('No consultation records available to retrain the model', 'error')
            return redirect(url_for('predict'))

        df = pd.DataFrame([{
            'consultation_date': r.consultation_date,
            'age_group': r.age_group,
            'gender': r.gender,
            'diagnosis': r.diagnosis,
            'department': r.department,
            'physician': r.physician,
            'consultation_type': r.consultation_type,
        } for r in records])

        try:
            _, metrics, _, _ = train_and_evaluate_model(df)
            report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'training_report.txt')
            with open(report_path, 'w', encoding='utf-8') as handle:
                handle.write('Smart Healthcare Clinic Management - Enhanced Training Report\n')
                handle.write('==========================================================\n')
                handle.write(f"R2 Score: {metrics['r2_score']}\n")
                handle.write(f"MAE: {metrics['mae']}\n")
                handle.write(f"MSE: {metrics['mse']}\n")
                handle.write(f"RMSE: {metrics['rmse']}\n")
                if metrics.get('cv_r2_mean'):
                    handle.write(f"CV R2 Mean: {metrics['cv_r2_mean']} (±{metrics['cv_r2_std']})\n")
                handle.write(f"Best Params: {metrics['best_params']}\n")
            write_training_report(report_path, metrics)
            session['last_report'] = report_path
            get_dashboard_summary(force_refresh=True)
            flash('Model retrained successfully using the existing consultation data', 'success')
        except Exception as e:
            traceback.print_exc()
            flash(f'Retraining failed: {str(e)}', 'error')

        return redirect(url_for('predict'))

    @app.route('/staff')
    def staff():
        cutoff = datetime.now() - timedelta(days=30)
        expired_staff = scoped_query(StaffMember.query, StaffMember).filter(
            StaffMember.is_active.is_(False),
            StaffMember.deleted_at.isnot(None),
            StaffMember.deleted_at < cutoff,
        ).all()
        for member in expired_staff:
            db.session.delete(member)
        if expired_staff:
            db.session.commit()
            get_dashboard_summary(force_refresh=True)

        page = request.args.get('page', 1, type=int)
        active_staff = (
            scoped_query(StaffMember.query.filter_by(is_active=True), StaffMember)
            .order_by(StaffMember.role.asc(), StaffMember.name.asc())
            .paginate(page=page, per_page=10, error_out=False)
        )
        inactive_staff = (
            scoped_query(StaffMember.query.filter_by(is_active=False), StaffMember)
            .order_by(StaffMember.deleted_at.desc(), StaffMember.name.asc())
            .all()
        )
        return render_template(
            'staff/index.html',
            active_staff=active_staff,
            inactive_staff=inactive_staff,
            all_branches_view=selected_branch_scope() is None,
        )

    staff_role_options = [item['role'] for item in FACILITY_STAFF_COMPLEMENT]
    staff_availability_options = ['Available', 'Busy', 'On Leave', 'Unavailable']

    @app.route('/staff/load-facility-complement', methods=['POST'])
    def load_facility_complement():
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        StaffMember.query.filter_by(is_active=True, branch_id=current_branch_id()).delete()
        for person in build_facility_staff_roster():
            db.session.add(StaffMember(
                branch_id=current_branch_id(),
                name=person['name'],
                role=person['role'],
                availability=person['availability'],
                is_active=True,
                deleted_at=None,
            ))
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash('Loaded the official 22-person clinic staff complement.', 'success')
        return redirect(url_for('staff'))

    @app.route('/staff/new', methods=['GET', 'POST'])
    def create_staff():
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        if request.method == 'POST':
            new_staff = StaffMember(
                branch_id=current_branch_id(),
                name=request.form.get('name', '').strip(),
                role=request.form.get('role', '').strip(),
                availability=request.form.get('availability', 'Available').strip(),
                is_active=True,
                deleted_at=None,
            )
            db.session.add(new_staff)
            db.session.commit()
            get_dashboard_summary(force_refresh=True)
            flash('Staff member added successfully.', 'success')
            return redirect(url_for('staff'))
        return render_template(
            'staff/form.html',
            staff_member=None,
            role_options=staff_role_options,
            availability_options=staff_availability_options,
        )

    @app.route('/staff/<int:staff_id>/edit', methods=['GET', 'POST'])
    def edit_staff(staff_id):
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        staff_member = StaffMember.query.filter_by(id=staff_id, branch_id=current_branch_id()).first_or_404()
        if request.method == 'POST':
            staff_member.name = request.form.get('name', staff_member.name).strip()
            staff_member.role = request.form.get('role', staff_member.role).strip()
            staff_member.availability = request.form.get('availability', staff_member.availability).strip()
            db.session.commit()
            get_dashboard_summary(force_refresh=True)
            flash('Staff member updated successfully.', 'success')
            return redirect(url_for('staff'))
        return render_template(
            'staff/form.html',
            staff_member=staff_member,
            role_options=staff_role_options,
            availability_options=staff_availability_options,
        )

    @app.route('/staff/<int:staff_id>/delete', methods=['POST'])
    def delete_staff(staff_id):
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        staff_member = StaffMember.query.filter_by(id=staff_id, branch_id=current_branch_id()).first_or_404()
        staff_member.is_active = False
        staff_member.deleted_at = datetime.now()
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash('Staff member removed from active roster (soft deleted).', 'success')
        return redirect(url_for('staff'))

    @app.route('/staff/<int:staff_id>/restore', methods=['POST'])
    def restore_staff(staff_id):
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        staff_member = StaffMember.query.filter_by(id=staff_id, branch_id=current_branch_id()).first_or_404()
        staff_member.is_active = True
        staff_member.deleted_at = None
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash('Staff member restored to active roster.', 'success')
        return redirect(url_for('staff'))

    @app.route('/staff/<int:staff_id>/permanent-delete', methods=['POST'])
    def permanent_delete_staff(staff_id):
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        staff_member = StaffMember.query.filter_by(id=staff_id, branch_id=current_branch_id()).first_or_404()
        db.session.delete(staff_member)
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash('Staff member permanently deleted.', 'success')
        return redirect(url_for('staff'))

    report_definitions = {
        'monthly-consultation': {
            'title': 'Monthly Consultation Report',
            'description': 'Monthly consultation totals and most common diagnoses.'
        },
        'quarterly': {
            'title': 'Quarterly Report',
            'description': 'Quarterly operating summary using consultation volume and distribution.'
        },
        'prediction': {
            'title': 'Prediction Report',
            'description': 'Forecasted next-month diagnosis demand and model validation metrics.'
        },
        'resource-recommendation': {
            'title': 'Resource Recommendation Report',
            'description': 'Staffing, doctor role, medical staff, and service readiness recommendations.'
        },
        'appointment-service-demand': {
            'title': 'Appointment and Service Demand Report',
            'description': 'Patient appointment requests, selected services/packages, time-slot usage, and recommended staff roles.'
        },
    }

    def records_to_dataframe():
        records = scoped_query(ConsultationRecord.query, ConsultationRecord).all()
        return pd.DataFrame([{
            'consultation_date': r.consultation_date,
            'age_group': r.age_group,
            'gender': r.gender,
            'diagnosis': r.diagnosis,
            'department': r.department,
            'physician': r.physician,
            'consultation_type': r.consultation_type,
        } for r in records])

    def build_report_payload(report_key):
        if report_key not in report_definitions:
            return None

        summary = get_dashboard_summary()
        df = records_to_dataframe()
        title = report_definitions[report_key]['title']
        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
        sections = []
        rows = []
        chart_rows = []
        extra_charts = []
        extra_tables = []
        details_title = 'Details'
        details_note = 'Supporting rows used for this report.'

        if not df.empty:
            df['consultation_date'] = pd.to_datetime(df['consultation_date'], errors='coerce')
            df = df.dropna(subset=['consultation_date'])
            for col in ['age_group', 'gender']:
                if col not in df.columns:
                    df[col] = 'Unknown'
                df[col] = df[col].fillna('Unknown').replace('', 'Unknown')

        if report_key == 'monthly-consultation':
            monthly_rows = []
            average_monthly_consultations = 0
            average_monthly_top_diagnosis_consultations = 0
            chart_rows = {
                'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'x_title': 'Month of Year',
                'datasets': []
            }
            forecast_month_label = summary.get('predicted_month_label', 'Next Month')
            try:
                forecast_month_label = datetime.strptime(forecast_month_label, '%b %Y').strftime('%B %Y')
            except Exception:
                pass
            if not df.empty:
                monthly = (
                    df.assign(month=df['consultation_date'].dt.strftime('%Y-%m'))
                    .groupby('month')
                    .size()
                    .reset_index(name='consultations')
                    .sort_values('month', ascending=False)
                )
                monthly_rows = monthly.head(12).to_dict('records')
                monthly_for_chart = df.assign(
                    year=df['consultation_date'].dt.year,
                    month_number=df['consultation_date'].dt.month
                ).groupby(['year', 'month_number']).size().reset_index(name='consultations')
                chart_rows['datasets'] = [
                    {
                        'year': int(year),
                        'data': [
                            int(year_data.loc[year_data['month_number'] == month_number, 'consultations'].sum())
                            for month_number in range(1, 13)
                        ]
                    }
                    for year, year_data in monthly_for_chart.groupby('year')
                ]
                average_monthly_consultations = round(monthly['consultations'].mean(), 2)
                top_diagnosis = summary.get('top_diagnosis', 'None')
                if top_diagnosis != 'None':
                    top_diagnosis_monthly = (
                        df[df['diagnosis'] == top_diagnosis]
                        .assign(month=df['consultation_date'].dt.strftime('%Y-%m'))
                        .groupby('month')
                        .size()
                        .reset_index(name='consultations')
                    )
                    if not top_diagnosis_monthly.empty:
                        average_monthly_top_diagnosis_consultations = round(top_diagnosis_monthly['consultations'].mean(), 2)
                age_monthly = (
                    df.assign(month=df['consultation_date'].dt.strftime('%Y-%m'))
                    .groupby(['age_group', 'month'])
                    .size()
                    .reset_index(name='consultations')
                )
                if not age_monthly.empty:
                    age_summary = (
                        age_monthly.groupby('age_group')['consultations']
                        .mean()
                        .round(2)
                        .reset_index(name='consultations')
                        .sort_values('consultations', ascending=False)
                    )
                    extra_charts.append({
                        'title': 'Average Monthly Consultations by Age Group',
                        'type': 'bar',
                        'labels': age_summary['age_group'].tolist(),
                        'data': age_summary['consultations'].tolist(),
                        'y_title': 'Average Consultations'
                    })
                gender_monthly = (
                    df.assign(month=df['consultation_date'].dt.strftime('%Y-%m'))
                    .groupby(['gender', 'month'])
                    .size()
                    .reset_index(name='consultations')
                )
                if not gender_monthly.empty:
                    gender_summary = (
                        gender_monthly.groupby('gender')['consultations']
                        .mean()
                        .round(2)
                        .reset_index(name='consultations')
                        .sort_values('consultations', ascending=False)
                    )
                    extra_charts.append({
                        'title': 'Average Monthly Consultations by Sex',
                        'type': 'bar',
                        'labels': gender_summary['gender'].tolist(),
                        'data': gender_summary['consultations'].tolist(),
                        'y_title': 'Average Consultations'
                    })
            sections = [
                {
                    'metric': 'Average Monthly Consultations',
                    'value': average_monthly_consultations,
                    'explanation': 'Average number of consultations per month based on the uploaded records.'
                },
                {
                    'metric': 'Top Diagnosis',
                    'value': summary.get('top_diagnosis', 'None'),
                    'explanation': 'The diagnosis with the highest total count across all uploaded records.'
                },
                {
                    'metric': 'Average Monthly Consultations of Top Diagnosis',
                    'value': average_monthly_top_diagnosis_consultations,
                    'explanation': 'Average monthly consultation count for the top diagnosis across uploaded records.'
                },
                {
                    'metric': f'Forecasted Total Cases for the Next Month ({forecast_month_label})',
                    'value': summary.get('predicted_cases_next_month', 0),
                    'explanation': 'The forecasted total consultation demand for the next calendar month. This helps compare recent actual volume with expected upcoming demand.'
                },
            ]
            rows = []
            details_title = 'Monthly Consultation Trend Graph'
            details_note = 'This graph shows monthly consultation totals by year so the clinic can compare seasonal demand patterns and growth more easily.'

        elif report_key == 'quarterly':
            quarterly_rows = []
            average_quarterly_consultations = 0
            chart_rows = {
                'labels': ['Q1', 'Q2', 'Q3', 'Q4'],
                'x_title': 'Quarter of Year',
                'datasets': []
            }
            if not df.empty:
                quarterly = (
                    df.assign(quarter=df['consultation_date'].dt.to_period('Q').astype(str))
                    .groupby('quarter')
                    .size()
                    .reset_index(name='consultations')
                    .sort_values('quarter', ascending=False)
                )
                quarterly_rows = quarterly.head(8).to_dict('records')
                average_quarterly_consultations = round(quarterly['consultations'].mean(), 2)
                quarterly_for_chart = df.assign(
                    year=df['consultation_date'].dt.year,
                    quarter_number=df['consultation_date'].dt.quarter
                ).groupby(['year', 'quarter_number']).size().reset_index(name='consultations')
                chart_rows['datasets'] = [
                    {
                        'year': int(year),
                        'data': [
                            int(year_data.loc[year_data['quarter_number'] == quarter_number, 'consultations'].sum())
                            for quarter_number in range(1, 5)
                        ]
                    }
                    for year, year_data in quarterly_for_chart.groupby('year')
                ]
                age_quarterly = (
                    df.assign(quarter=df['consultation_date'].dt.to_period('Q').astype(str))
                    .groupby(['age_group', 'quarter'])
                    .size()
                    .reset_index(name='consultations')
                )
                if not age_quarterly.empty:
                    age_summary = (
                        age_quarterly.groupby('age_group')['consultations']
                        .mean()
                        .round(2)
                        .reset_index(name='consultations')
                        .sort_values('consultations', ascending=False)
                    )
                    extra_charts.append({
                        'title': 'Average Quarterly Consultations by Age Group',
                        'type': 'bar',
                        'labels': age_summary['age_group'].tolist(),
                        'data': age_summary['consultations'].tolist(),
                        'y_title': 'Average Consultations'
                    })
                gender_quarterly = (
                    df.assign(quarter=df['consultation_date'].dt.to_period('Q').astype(str))
                    .groupby(['gender', 'quarter'])
                    .size()
                    .reset_index(name='consultations')
                )
                if not gender_quarterly.empty:
                    gender_summary = (
                        gender_quarterly.groupby('gender')['consultations']
                        .mean()
                        .round(2)
                        .reset_index(name='consultations')
                        .sort_values('consultations', ascending=False)
                    )
                    extra_charts.append({
                        'title': 'Average Quarterly Consultations by Sex',
                        'type': 'bar',
                        'labels': gender_summary['gender'].tolist(),
                        'data': gender_summary['consultations'].tolist(),
                        'y_title': 'Average Consultations'
                    })
            sections = [
                {
                    'metric': 'Average Consultations per Quarter',
                    'value': average_quarterly_consultations,
                    'explanation': 'Average number of consultations per quarter based on uploaded records.'
                },
                {
                    'metric': 'Active Staff',
                    'value': summary.get('staff_count', 0),
                    'explanation': 'Number of active staff members currently counted for capacity planning.'
                },
                {
                    'metric': 'Capacity Status',
                    'value': summary.get('capacity_status', 'Unknown'),
                    'explanation': 'Overall demand-versus-staffing status based on predicted cases and estimated monthly capacity.'
                },
            ]
            rows = []
            details_title = 'Quarterly Consultation Trend Graph'
            details_note = 'This graph has four points per year, one for each quarter. Each colored line represents a year so quarter-to-quarter changes are easier to compare.'

        elif report_key == 'prediction':
            forecast = summary.get('predictions', [])
            metrics = summary.get('rf_metrics') or {}
            demographic_forecast = summary.get('demographic_forecast') or {}
            sections = [
                {
                    'metric': 'Predicted Month',
                    'value': summary.get('predicted_month_label', 'Unknown'),
                    'explanation': 'The future month being forecasted by the model.'
                },
                {
                    'metric': 'Predicted Cases Next Month',
                    'value': summary.get('predicted_cases_next_month', 0),
                    'explanation': 'The predicted total number of consultations expected for the forecast month.'
                },
                {
                    'metric': 'Model Verdict',
                    'value': metrics.get('model_verdict', 'Available after prediction/retrain'),
                    'explanation': 'A short quality label based on validation R2 and whether the model beats the last-month baseline.'
                },
                {
                    'metric': 'Validation R2',
                    'value': metrics.get('validation_r2', metrics.get('r2_score', 'N/A')),
                    'explanation': 'How well the model explained unseen validation months. Higher is better; values above 0 mean it beats predicting the average.'
                },
                {
                    'metric': 'Validation MAE',
                    'value': metrics.get('validation_mae', metrics.get('mae', 'N/A')),
                    'explanation': 'Average prediction error measured in consultation cases per diagnosis-month.'
                },
            ]
            rows = forecast
            details_title = 'Diagnosis Forecast Details'
            details_note = 'This table lists the forecasted next-month case count by diagnosis so staff and resources can be planned by expected demand area.'
            extra_tables = [
                {
                    'title': 'Forecast by Age Group',
                    'rows': demographic_forecast.get('age_group', [])
                },
                {
                    'title': 'Forecast by Gender',
                    'rows': demographic_forecast.get('gender', [])
                },
                {
                    'title': 'Top Diagnosis-Demographic Forecast Segments',
                    'rows': demographic_forecast.get('segments', [])
                },
            ]

        elif report_key == 'resource-recommendation':
            staff_forecast = summary.get('staff_demand_forecast') or {}
            monthly_staff_rows = [
                {
                    'staff_role': row.get('staff_role'),
                    'planning_demand': row.get('planning_demand'),
                    'available_staff': row.get('available_staff'),
                    'status': row.get('status'),
                }
                for row in staff_forecast.get('monthly_rows', [])
                if row.get('planning_demand', 0) > 0 or row.get('available_staff', 0) > 0
            ]
            daily_staff_rows = [
                {
                    'date': row.get('date'),
                    'staff_role': row.get('staff_role'),
                    'scheduled_demand': row.get('scheduled_demand'),
                    'status': row.get('status'),
                }
                for row in staff_forecast.get('daily_rows', [])[:18]
            ]
            sections = [
                {
                    'metric': 'Forecast Month',
                    'value': summary.get('predicted_month_full_label', summary.get('predicted_month_label', 'Next Month')),
                    'explanation': 'The month used for the staff and resource planning forecast.'
                },
                {
                    'metric': 'Top Staff Role Needed',
                    'value': f"{staff_forecast.get('top_role', 'No role demand yet')} ({staff_forecast.get('top_role_demand', 0)} forecasted cases or service needs)",
                    'explanation': 'The staff role with the highest next-month planning demand.'
                },
                {
                    'metric': 'Scheduled Appointments in Forecast Month',
                    'value': staff_forecast.get('scheduled_appointment_count', 0),
                    'explanation': 'Already booked appointments for the forecast month that are considered in staff planning.'
                },
                {
                    'metric': 'Estimated Monthly Capacity',
                    'value': summary.get('estimated_monthly_capacity', 0),
                    'explanation': 'Estimated number of consultations current active staff can support in one month.'
                },
                {
                    'metric': 'Forecast Pressure',
                    'value': f"{summary.get('forecast_pressure', 0)}%",
                    'explanation': 'Predicted demand divided by estimated monthly staff capacity.'
                },
                {
                    'metric': 'Capacity Status',
                    'value': summary.get('capacity_status', 'Unknown'),
                    'explanation': 'Simple status label summarizing whether forecasted demand is healthy, moderate, or high.'
                },
                {
                    'metric': 'Resource Recommendation',
                    'value': summary.get('resource_forecast_recommendation', 'N/A'),
                    'explanation': 'Recommended staffing/resource action based on forecast pressure.'
                },
            ]
            rows = [
                {'role': role, 'actual_count': count}
                for role, count in summary.get('actual_staff_by_role', {}).items()
            ]
            details_title = 'Active Staff By Role'
            details_note = 'This table shows current active staff counts by role, which are used when estimating monthly capacity.'
            extra_tables = [
                {
                    'title': 'Monthly Staff Requirement Forecast',
                    'rows': monthly_staff_rows,
                },
                {
                    'title': 'Daily Staff Requirement Forecast from Appointments',
                    'rows': daily_staff_rows,
                },
            ]

        elif report_key == 'appointment-service-demand':
            appointments = scoped_query(Appointment.query, Appointment).all()
            today_date = datetime.now().date()
            status_counts = Counter((appointment.status or 'Unknown') for appointment in appointments)
            active_appointments = [
                appointment for appointment in appointments
                if appointment.status in ACTIVE_APPOINTMENT_SLOT_STATUSES
            ]
            upcoming_active = []
            for appointment in active_appointments:
                appointment_date = parse_iso_date(appointment.appointment_date)
                if appointment_date and appointment_date >= today_date:
                    upcoming_active.append(appointment)
            upcoming_active.sort(key=lambda item: (item.appointment_date or '', item.appointment_time or '', item.created_at or datetime.min))

            service_counter = Counter()
            package_counter = Counter()
            combined_request_counter = Counter()
            role_counter = Counter()
            converted_count = 0
            for appointment in appointments:
                if appointment.status == 'Cancelled':
                    continue
                if appointment.converted_to_records:
                    converted_count += 1
                for service in appointment.service_items():
                    service_counter[service] += 1
                    combined_request_counter[service] += 1
                for package in appointment.package_items():
                    label = f'Package: {package}'
                    package_counter[package] += 1
                    combined_request_counter[label] += 1
                for role in appointment.role_items():
                    role_counter[role] += 1

            active_slot_count = len({
                (appointment.branch_id, appointment.appointment_date, appointment.appointment_time)
                for appointment in active_appointments
                if appointment.appointment_date and appointment.appointment_time
            })
            top_request, top_request_count = ('None', 0)
            if combined_request_counter:
                top_request, top_request_count = combined_request_counter.most_common(1)[0]
            top_role, top_role_count = ('None', 0)
            if role_counter:
                top_role, top_role_count = role_counter.most_common(1)[0]

            rows = [
                {
                    'date': appointment.appointment_date,
                    'time': format_appointment_time(appointment.appointment_time),
                    'patient': appointment.patient.full_name if appointment.patient else 'Unknown patient',
                    'status': appointment.status,
                }
                for appointment in upcoming_active[:20]
            ]
            top_request_rows = [
                {'request': request_label, 'count': int(count)}
                for request_label, count in combined_request_counter.most_common(15)
            ]
            role_rows = [
                {'staff_role': role, 'appointment_count': int(count)}
                for role, count in role_counter.most_common(15)
            ]
            package_rows = [
                {'package': package, 'appointment_count': int(count)}
                for package, count in package_counter.most_common(10)
            ]
            sections = [
                {
                    'metric': 'Total Appointment Requests',
                    'value': len(appointments),
                    'explanation': 'Total appointment requests currently stored for the selected branch scope.'
                },
                {
                    'metric': 'Pending Appointments',
                    'value': status_counts.get('Pending', 0),
                    'explanation': 'Appointments waiting for clinic confirmation.'
                },
                {
                    'metric': 'Confirmed Appointments',
                    'value': status_counts.get('Confirmed', 0),
                    'explanation': 'Appointments accepted by clinic staff and counted as active scheduled demand.'
                },
                {
                    'metric': 'Upcoming Active Time Slots',
                    'value': active_slot_count,
                    'explanation': 'Unique active branch/date/time slots currently occupied by pending or confirmed appointments.'
                },
                {
                    'metric': 'Top Requested Service or Package',
                    'value': f'{top_request} ({top_request_count})',
                    'explanation': 'Most frequently selected service or package from appointment requests.'
                },
                {
                    'metric': 'Top Recommended Staff Role',
                    'value': f'{top_role} ({top_role_count})',
                    'explanation': 'Most frequently recommended staff role based on selected services, packages, and reasons.'
                },
                {
                    'metric': 'Converted to Consultation Records',
                    'value': converted_count,
                    'explanation': 'Completed appointments that already generated consultation records.'
                },
            ]
            extra_charts = [
                {
                    'title': 'Appointments by Status',
                    'type': 'bar',
                    'labels': list(status_counts.keys()),
                    'data': [int(value) for value in status_counts.values()],
                    'y_title': 'Appointments',
                    'tooltip_suffix': 'appointments',
                },
                {
                    'title': 'Top Requested Services and Packages',
                    'type': 'bar',
                    'labels': [label for label, _count in combined_request_counter.most_common(8)],
                    'data': [int(count) for _label, count in combined_request_counter.most_common(8)],
                    'y_title': 'Requests',
                    'tooltip_suffix': 'requests',
                },
                {
                    'title': 'Recommended Staff Roles from Appointments',
                    'type': 'bar',
                    'labels': [label for label, _count in role_counter.most_common(8)],
                    'data': [int(count) for _label, count in role_counter.most_common(8)],
                    'y_title': 'Appointments',
                    'tooltip_suffix': 'appointments',
                },
            ]
            extra_charts = [chart for chart in extra_charts if chart.get('labels')]
            extra_tables = [
                {
                    'title': 'Top Requested Services and Packages',
                    'rows': top_request_rows,
                },
                {
                    'title': 'Recommended Staff Roles from Appointments',
                    'rows': role_rows,
                },
                {
                    'title': 'Top Requested Packages',
                    'rows': package_rows,
                },
            ]
            details_title = 'Upcoming Active Appointment Slots'
            details_note = 'This table shows pending and confirmed appointment slots. The system prevents two active appointments from using the same branch, date, and time.'

        return {
            'key': report_key,
            'title': title,
            'description': report_definitions[report_key]['description'],
            'generated_at': generated_at,
            'branch_name': summary.get('branch_name', DEFAULT_BRANCH_NAME),
            'branch_code': summary.get('branch_code', DEFAULT_BRANCH_CODE),
            'sections': sections,
            'rows': rows,
            'chart_rows': chart_rows,
            'extra_charts': extra_charts,
            'extra_tables': [table for table in extra_tables if table.get('rows')],
            'details_title': details_title,
            'details_note': details_note,
        }

    def report_lines(payload):
        lines = [payload['description'], f"Branch: {payload.get('branch_name', DEFAULT_BRANCH_NAME)}", '']
        for section in payload['sections']:
            lines.append(f"{section['metric']}: {section['value']}")
            lines.append(f"  Why shown: {section['explanation']}")
        if payload['rows']:
            lines.append('')
            lines.append(payload.get('details_title', 'Details') + ':')
            lines.append(payload.get('details_note', ''))
            for row in payload['rows']:
                if isinstance(row, dict):
                    lines.append(' | '.join(f'{key}: {value}' for key, value in row.items()))
                else:
                    lines.append(str(row))
        return lines

    @app.route('/reports')
    def reports():
        return render_template(
            'reports/index.html',
            reports=report_definitions,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/reports/<report_key>')
    def report_view(report_key):
        payload = build_report_payload(report_key)
        if payload is None:
            flash('Report not found.', 'error')
            return redirect(url_for('reports'))
        return render_template(
            'reports/view.html',
            report=payload,
            print_mode=False,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/reports/<report_key>/print')
    def report_print(report_key):
        payload = build_report_payload(report_key)
        if payload is None:
            flash('Report not found.', 'error')
            return redirect(url_for('reports'))
        return render_template(
            'reports/view.html',
            report=payload,
            print_mode=True,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/reports/<report_key>/pdf')
    def report_pdf(report_key):
        payload = build_report_payload(report_key)
        if payload is None:
            flash('Report not found.', 'error')
            return redirect(url_for('reports'))
        logo_path = os.path.join(app.static_folder, 'acudetek.jfif')
        pdf_bytes = _build_report_pdf(payload, logo_path=logo_path)
        filename = f"{report_key}-{datetime.now().strftime('%Y%m%d')}.pdf"
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )

    @app.route('/settings', methods=['GET', 'POST'])
    def settings():
        app_settings = load_app_settings()
        if request.method == 'POST':
            try:
                capacity = max(1, int(request.form.get('staff_capacity_per_month', STAFF_CAPACITY_PER_MONTH)))
                app_settings['staff_capacity_per_month'] = capacity
                save_app_settings(app_settings)
                get_dashboard_summary(force_refresh=True)
                flash('Staff capacity setting updated.', 'success')
            except Exception:
                flash('Please enter a valid staff capacity number.', 'error')
            return redirect(url_for('settings'))

        return render_template(
            'settings/index.html',
            settings=app_settings,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    return app

# -------------------------------------------------------------
# Database initialisation and schema migration
# -------------------------------------------------------------
def migrate_staff_member_schema(app):
    with app.app_context():
        with db.engine.begin() as conn:
            result = conn.execute(text("PRAGMA table_info(staff_member)"))
            columns = {row[1] for row in result.fetchall()}
            if 'is_active' not in columns:
                conn.execute(text("ALTER TABLE staff_member ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1"))
            if 'deleted_at' not in columns:
                conn.execute(text("ALTER TABLE staff_member ADD COLUMN deleted_at DATETIME"))
            if 'department' in columns:
                conn.execute(text("DROP TABLE IF EXISTS staff_member_new"))
                conn.execute(text("""
                    CREATE TABLE staff_member_new (
                        id INTEGER NOT NULL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        role VARCHAR(60) NOT NULL,
                        availability VARCHAR(20),
                        is_active BOOLEAN NOT NULL DEFAULT 1,
                        deleted_at DATETIME
                    )
                """))
                conn.execute(text("""
                    INSERT INTO staff_member_new (id, name, role, availability, is_active, deleted_at)
                    SELECT id, name, role, availability, is_active, deleted_at
                    FROM staff_member
                """))
                conn.execute(text("DROP TABLE staff_member"))
                conn.execute(text("ALTER TABLE staff_member_new RENAME TO staff_member"))

def migrate_branch_schema(app):
    with app.app_context():
        with db.engine.begin() as conn:
            result = conn.execute(text("PRAGMA table_info(branch)"))
            branch_columns = {row[1] for row in result.fetchall()}
            if 'email' not in branch_columns:
                conn.execute(text("ALTER TABLE branch ADD COLUMN email VARCHAR(120)"))

        branch = Branch.query.filter_by(code=DEFAULT_BRANCH_CODE).first()
        if branch is None:
            branch = Branch(
                name=DEFAULT_BRANCH_NAME,
                code=DEFAULT_BRANCH_CODE,
                address=DEFAULT_BRANCH_ADDRESS,
                is_main=True,
                is_active=True,
            )
            db.session.add(branch)
            db.session.commit()

        with db.engine.begin() as conn:
            for table_name in ('user', 'consultation_record', 'staff_member', 'patient', 'appointment'):
                result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = {row[1] for row in result.fetchall()}
                if 'branch_id' not in columns:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN branch_id INTEGER"))
                conn.execute(
                    text(f"UPDATE {table_name} SET branch_id = :branch_id WHERE branch_id IS NULL"),
                    {'branch_id': branch.id}
                )
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_branch_id ON {table_name}(branch_id)"))

def migrate_patient_schema(app):
    with app.app_context():
        with db.engine.begin() as conn:
            result = conn.execute(text("PRAGMA table_info(patient)"))
            patient_columns = {row[1] for row in result.fetchall()}
            if 'age' not in patient_columns:
                conn.execute(text("ALTER TABLE patient ADD COLUMN age INTEGER"))

            result = conn.execute(text("PRAGMA table_info(consultation_record)"))
            consultation_columns = {row[1] for row in result.fetchall()}
            if 'patient_id' not in consultation_columns:
                conn.execute(text("ALTER TABLE consultation_record ADD COLUMN patient_id INTEGER"))
            if 'patient_age' not in consultation_columns:
                conn.execute(text("ALTER TABLE consultation_record ADD COLUMN patient_age INTEGER"))

            result = conn.execute(text("PRAGMA table_info(appointment)"))
            appointment_columns = {row[1] for row in result.fetchall()}
            if 'completed_at' not in appointment_columns:
                conn.execute(text("ALTER TABLE appointment ADD COLUMN completed_at DATETIME"))
            if 'converted_to_records' not in appointment_columns:
                conn.execute(text("ALTER TABLE appointment ADD COLUMN converted_to_records BOOLEAN NOT NULL DEFAULT 0"))
            if 'completion_notes' not in appointment_columns:
                conn.execute(text("ALTER TABLE appointment ADD COLUMN completion_notes TEXT"))
            if 'selected_packages' not in appointment_columns:
                conn.execute(text("ALTER TABLE appointment ADD COLUMN selected_packages TEXT NOT NULL DEFAULT '[]'"))

            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_consultation_record_patient_id ON consultation_record(patient_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_patient_number ON patient(patient_number)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_patient_name ON patient(full_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_appointment_patient_id ON appointment(patient_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_appointment_date ON appointment(appointment_date)"))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_active_appointment_slot "
                "ON appointment(branch_id, appointment_date, appointment_time) "
                "WHERE status IN ('Pending', 'Confirmed') "
                "AND appointment_time IS NOT NULL "
                "AND appointment_time != ''"
            ))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_appointment_service_result_appointment_id ON appointment_service_result(appointment_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_appointment_service_result_record_id ON appointment_service_result(consultation_record_id)"))

def migrate_service_catalog_schema(app):
    with app.app_context():
        with db.engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_clinic_service_category ON clinic_service(category)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_clinic_service_name ON clinic_service(service_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_branch_service_branch_id ON branch_service(branch_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_branch_service_service_id ON branch_service(service_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_clinic_package_name ON clinic_package(package_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_package_item_package_id ON package_item(package_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_package_item_service_id ON package_item(service_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_branch_package_branch_id ON branch_package(branch_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_branch_package_package_id ON branch_package(package_id)"))

def init_db(app=None):
    app = app or flask_app
    with app.app_context():
        db.create_all()
        migrate_staff_member_schema(app)
        migrate_branch_schema(app)
        migrate_patient_schema(app)
        migrate_service_catalog_schema(app)
        default_branch = Branch.query.filter_by(code=DEFAULT_BRANCH_CODE).first()
        import_accudetek_service_catalog(app, seed_only=True)
        import_accudetek_package_catalog(app, seed_only=True)
        for branch in Branch.query.all():
            ensure_branch_service_rows(branch.id)
            ensure_branch_package_rows(branch.id)
        if not User.query.filter_by(username='superadmin').first():
            db.session.add(User(
                username='superadmin',
                password='superadmin123',
                role=SUPER_ADMIN_ROLE,
                branch_id=default_branch.id,
            ))
        if not User.query.filter_by(username='admin').first():
            db.session.add(User(username='admin', password='admin123', role='administrator', branch_id=default_branch.id))
        if not User.query.filter_by(username='staff').first():
            db.session.add(User(username='staff', password='staff123', role='staff', branch_id=default_branch.id))
        if not StaffMember.query.first():
            for person in build_facility_staff_roster():
                db.session.add(StaffMember(
                    branch_id=default_branch.id,
                    name=person['name'],
                    role=person['role'],
                    availability=person['availability'],
                    is_active=True,
                    deleted_at=None,
                ))
        db.session.commit()

flask_app = create_app()

if __name__ == '__main__':
    init_db(flask_app)
    flask_app.run(debug=True)

app = flask_app
