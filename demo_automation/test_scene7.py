from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.goto('http://127.0.0.1:5000/login', wait_until='domcontentloaded')
    page.fill('input[name="username"]', 'superadmin')
    page.fill('input[name="password"]', '12345678')
    page.click('button[type="submit"]')
    page.wait_for_timeout(1000)
    page.goto('http://127.0.0.1:5000/reports', wait_until='domcontentloaded')
    report_links = page.locator('a[href*="/reports/"]')
    print('Report links count:', report_links.count())
    for i in range(report_links.count()):
        href = report_links.nth(i).get_attribute('href')
        txt = report_links.nth(i).inner_text().strip()
        print(f'{txt[:30]} -> {href}')
        
    # Open Monthly Consultation Report
    page.goto('http://127.0.0.1:5000/reports/monthly_consultation', wait_until='domcontentloaded')
    print('Report detail URL:', page.url)
    export_buttons = page.locator('a[href*="pdf"], a[href*="print"], button:has-text("PDF"), button:has-text("Print")')
    print('Export buttons count:', export_buttons.count())
    for i in range(export_buttons.count()):
        print('Export btn:', export_buttons.nth(i).inner_text().strip(), export_buttons.nth(i).get_attribute('href'))

    browser.close()
