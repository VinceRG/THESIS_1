from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.goto('http://127.0.0.1:5000/login', wait_until='domcontentloaded')
    page.fill('input[name="username"]', 'superadmin')
    page.fill('input[name="password"]', '12345678')
    page.click('button[type="submit"]')
    page.wait_for_timeout(1000)
    page.goto('http://127.0.0.1:5000/appointments', wait_until='domcontentloaded')
    print('Appointments URL:', page.url)
    print('Headings:', page.locator('h1, h2, h3').all_inner_texts()[:5])
    rows = page.locator('table tr')
    print('Table rows count:', rows.count())
    for i in range(min(4, rows.count())):
        print(f'Row {i}:', rows.nth(i).inner_text().replace('\n', ' | '))
    browser.close()
