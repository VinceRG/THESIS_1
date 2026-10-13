from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.goto('http://127.0.0.1:5000/patient-login', wait_until='domcontentloaded')
    page.fill('input[name="email"]', 'vincegonato48@gmail.com')
    page.fill('input[name="password"]', '12345678')
    page.click('button[type="submit"]')
    page.wait_for_timeout(3000)
    print("Page URL after submit:", page.url)
    flashes = page.locator('.flash-message').all_inner_texts()
    print("Flashes:", flashes)
    browser.close()
