from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    
    # 1. Login
    page.goto('http://127.0.0.1:5000/login', wait_until='domcontentloaded')
    page.fill('input[name="username"]', 'superadmin')
    page.fill('input[name="password"]', '12345678')
    page.click('button[type="submit"]')
    page.wait_for_timeout(2000)
    print('After login URL:', page.url)
    
    # 2. Patients page
    page.goto('http://127.0.0.1:5000/patients', wait_until='domcontentloaded')
    page.wait_for_timeout(1000)
    print('Patients page URL:', page.url)
    patient_links = page.locator('table a[href*="/patients/"]')
    print('Patient links count:', patient_links.count())
    if patient_links.count() > 0:
        first_patient = patient_links.first
        print('First patient href:', first_patient.get_attribute('href'))
        print('First patient text:', first_patient.inner_text())
        first_patient.click()
        page.wait_for_timeout(1500)
        print('Individual patient profile URL:', page.url)
        print('Profile headings:', page.locator('h1, h2, h3').all_inner_texts()[:5])
    
    browser.close()
