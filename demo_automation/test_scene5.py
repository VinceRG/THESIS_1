from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.goto('http://127.0.0.1:5000/login', wait_until='domcontentloaded')
    page.fill('input[name="username"]', 'superadmin')
    page.fill('input[name="password"]', '12345678')
    page.click('button[type="submit"]')
    page.wait_for_timeout(1000)
    page.goto('http://127.0.0.1:5000/predict', wait_until='domcontentloaded')
    
    retrain = page.locator('button:has-text("Retrain"), a:has-text("Retrain"), form[action*="retrain"]')
    print('Retrain count:', retrain.count())
    for i in range(retrain.count()):
        print('Retrain element:', retrain.nth(i).inner_text().strip())

    fi = page.locator('h2, h3, h4, th, .card-header')
    for h in fi.all():
        t = h.inner_text().strip()
        if any(w in t.lower() for w in ['feature', 'importance', 'metric', 'indicator', 'algorithm', 'validation', 'demographic']):
            print('Match:', t)

    browser.close()
