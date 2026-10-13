from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.goto('http://127.0.0.1:5000/login', wait_until='domcontentloaded')
    page.fill('input[name="username"]', 'superadmin')
    page.fill('input[name="password"]', '12345678')
    page.click('button[type="submit"]')
    page.wait_for_timeout(1000)
    page.goto('http://127.0.0.1:5000/dashboard', wait_until='domcontentloaded')
    print('Dashboard URL:', page.url)
    print('Cards & Titles:', [t.inner_text().strip() for t in page.locator('.stat-card, .card-title, h2, h3, canvas').all()[:15]])
    browser.close()
