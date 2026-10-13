from flask import Flask
from playwright.sync_api import sync_playwright

temp_app = Flask('temp')
temp_app.secret_key = 'dev-secret'
serializer = temp_app.session_interface.get_signing_serializer(temp_app)

session_data = {
    'patient_account_id': 1,
    'patient_portal_patient_id': 12,
    '_fresh': True
}
cookie_val = serializer.dumps(session_data)
print("Signed cookie:", cookie_val)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    context.add_cookies([{
        'name': 'session',
        'value': cookie_val,
        'domain': '127.0.0.1',
        'path': '/',
        'httpOnly': True
    }])
    page = context.new_page()
    page.goto('http://127.0.0.1:5000/patient-profile', wait_until='domcontentloaded')
    print("Direct Profile URL:", page.url)
    print("Page Title:", page.title())
    print("Profile headings:", page.locator('h1, h2, h3').all_inner_texts()[:5])
    browser.close()
