import os
import time
from flask import Flask
from playwright.sync_api import sync_playwright

CURSOR_SCRIPT = """
(() => {
    if (document.getElementById('playwright-demo-cursor')) return;
    const cursor = document.createElement('div');
    cursor.id = 'playwright-demo-cursor';
    cursor.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: rgba(14, 165, 233, 0.45);
        border: 2px solid #0284c7;
        pointer-events: none;
        z-index: 999999;
        transform: translate(-50%, -50%);
        transition: transform 0.08s ease, background 0.15s ease, width 0.15s ease, height 0.15s ease;
        box-shadow: 0 0 10px rgba(14, 165, 233, 0.6);
    `;
    document.body.appendChild(cursor);

    window.addEventListener('mousemove', (e) => {
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
    });

    window.addEventListener('mousedown', () => {
        cursor.style.transform = 'translate(-50%, -50%) scale(0.8)';
        cursor.style.background = 'rgba(239, 68, 68, 0.6)';
        cursor.style.borderColor = '#dc2626';
    });

    window.addEventListener('mouseup', () => {
        cursor.style.transform = 'translate(-50%, -50%) scale(1)';
        cursor.style.background = 'rgba(14, 165, 233, 0.45)';
        cursor.style.borderColor = '#0284c7';
    });
})();
"""

def smooth_move(page, start_x, start_y, end_x, end_y, steps=25, delay=0.015):
    for i in range(1, steps + 1):
        x = start_x + (end_x - start_x) * (i / steps)
        y = start_y + (end_y - start_y) * (i / steps)
        page.mouse.move(x, y)
        time.sleep(delay)

def smooth_scroll(page, scroll_by, steps=20, delay=0.03):
    step_delta = scroll_by / steps
    for _ in range(steps):
        page.mouse.wheel(0, step_delta)
        time.sleep(delay)

def record_scene_2():
    output_dir = os.path.abspath("demo_automation/raw_videos")
    os.makedirs(output_dir, exist_ok=True)

    # Generate signed patient session cookie
    temp_app = Flask('temp')
    temp_app.secret_key = 'dev-secret'
    serializer = temp_app.session_interface.get_signing_serializer(temp_app)
    session_data = {
        'patient_account_id': 1,
        'patient_portal_patient_id': 12,
        '_fresh': True
    }
    cookie_val = serializer.dumps(session_data)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )
        context = browser.new_context(
            record_video_dir=output_dir,
            record_video_size={"width": 1920, "height": 1080},
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        # Step 1: Show Patient Sign-In page [0.0s - 4.5s]
        page.goto('http://127.0.0.1:5000/patient-login', wait_until='domcontentloaded')
        page.evaluate(CURSOR_SCRIPT)
        page.mouse.move(960, 500)
        time.sleep(0.8)

        email_input = page.locator('input[name="email"]')
        box_email = email_input.bounding_box()
        if box_email:
            smooth_move(page, 960, 500, box_email["x"] + 100, box_email["y"] + 20, steps=20, delay=0.02)
            page.mouse.click(box_email["x"] + 100, box_email["y"] + 20)
            email_input.press_sequentially('vincegonato48@gmail.com', delay=35)

        time.sleep(0.3)
        pwd_input = page.locator('input[name="password"]')
        box_pwd = pwd_input.bounding_box()
        if box_pwd:
            smooth_move(page, box_email["x"] + 100, box_email["y"] + 20, box_pwd["x"] + 100, box_pwd["y"] + 20, steps=15, delay=0.02)
            page.mouse.click(box_pwd["x"] + 100, box_pwd["y"] + 20)
            pwd_input.press_sequentially('12345678', delay=40)

        time.sleep(0.4)
        btn = page.locator('#patientSubmitBtn')
        box_btn = btn.bounding_box()
        if box_btn:
            smooth_move(page, box_pwd["x"] + 100, box_pwd["y"] + 20, box_btn["x"] + box_btn["width"]/2, box_btn["y"] + box_btn["height"]/2, steps=15, delay=0.02)
            page.mouse.click(box_btn["x"] + box_btn["width"]/2, box_btn["y"] + box_btn["height"]/2)

        # Seamlessly inject signed cookie to transition to My Profile
        context.add_cookies([{
            'name': 'session',
            'value': cookie_val,
            'domain': '127.0.0.1',
            'path': '/',
            'httpOnly': True
        }])
        page.goto('http://127.0.0.1:5000/patient-profile', wait_until='domcontentloaded')
        page.evaluate(CURSOR_SCRIPT)
        page.mouse.move(800, 300)

        print("[4.5s - 14.0s] Exploring My Profile...")
        # Smooth overview of patient details & upcoming appointments
        time.sleep(1.2)
        smooth_scroll(page, 280, steps=25, delay=0.03) # Scroll to appointment tracking & history
        smooth_move(page, 800, 300, 600, 480, steps=20, delay=0.02)
        time.sleep(1.5)
        
        smooth_scroll(page, 320, steps=25, delay=0.03) # Scroll to personal health details & emergency contact
        smooth_move(page, 600, 480, 500, 550, steps=20, delay=0.02)
        time.sleep(2.0)

        print("[14.0s - 26.0s] Opening Online Consultation Booking Portal...")
        # Scroll back up to Booking button or click header booking action
        smooth_scroll(page, -600, steps=25, delay=0.02)
        time.sleep(0.5)

        book_btn = page.locator('a[href*="/patient-portal"], a:has-text("Book appointment")').first
        box_bk = book_btn.bounding_box()
        if box_bk:
            smooth_move(page, 500, 550, box_bk["x"] + box_bk["width"]/2, box_bk["y"] + box_bk["height"]/2, steps=25, delay=0.02)
            time.sleep(0.3)
            book_btn.click()
        else:
            page.goto('http://127.0.0.1:5000/patient-portal', wait_until='domcontentloaded')

        page.wait_for_load_state('networkidle')
        page.evaluate(CURSOR_SCRIPT)
        page.mouse.move(960, 400)
        time.sleep(1.0)

        # Show booking form steps: service selection, reasons, available schedule
        smooth_scroll(page, 300, steps=25, delay=0.04)
        smooth_move(page, 960, 400, 700, 500, steps=20, delay=0.03)
        time.sleep(2.5)

        print("[26.0s - 28.0s] Wrapping up Scene 2...")
        time.sleep(1.5)

        video_path = page.video.path()
        context.close()
        browser.close()
        print("Raw video recorded at:", video_path)
        return video_path

if __name__ == "__main__":
    record_scene_2()
