from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.goto('http://127.0.0.1:5000/login', wait_until='domcontentloaded')
    page.fill('input[name="username"]', 'superadmin')
    page.fill('input[name="password"]', '12345678')
    page.click('button[type="submit"]')
    page.wait_for_timeout(1000)
    page.goto('http://127.0.0.1:5000/resources', wait_until='domcontentloaded')
    print('Resources URL:', page.url)
    print('Headings:', page.locator('h1, h2, h3, h4').all_inner_texts()[:15])
    print('Tabs / Buttons:', [b.inner_text().strip() for b in page.locator('button, a.btn, .tab, nav a').all()[:20] if b.inner_text().strip()])
    browser.close()
