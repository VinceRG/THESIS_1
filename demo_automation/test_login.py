from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('http://127.0.0.1:5000/patient-login')
    page.fill('input[name="email"]', 'vincegonato48@gmail.com')
    page.fill('input[name="password"]', '12345678')
    page.click('button[type="submit"]')
    page.wait_for_timeout(3000)
    print("Patient login URL:", page.url)
    content = page.content()
    if 'flash-error' in content or 'flash-danger' in content or 'error' in content:
        print("Page text snippet:", page.inner_text('body')[:300])
    browser.close()
