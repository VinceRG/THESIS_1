import os
import time
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

def smooth_move(page, start_x, start_y, end_x, end_y, steps=30, delay=0.02):
    for i in range(1, steps + 1):
        x = start_x + (end_x - start_x) * (i / steps)
        y = start_y + (end_y - start_y) * (i / steps)
        page.mouse.move(x, y)
        time.sleep(delay)

def smooth_scroll(page, scroll_by, steps=25, delay=0.035):
    step_delta = scroll_by / steps
    for _ in range(steps):
        page.mouse.wheel(0, step_delta)
        time.sleep(delay)

def record_scene_3():
    output_dir = os.path.abspath("demo_automation/raw_videos")
    os.makedirs(output_dir, exist_ok=True)

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

        # Step 0: Login silently first
        page.goto('http://127.0.0.1:5000/login', wait_until='domcontentloaded')
        page.fill('input[name="username"]', 'superadmin')
        page.fill('input[name="password"]', '12345678')
        page.click('button[type="submit"]')
        page.wait_for_load_state('networkidle')
        time.sleep(1.0)

        # Inject cursor
        page.evaluate(CURSOR_SCRIPT)
        page.mouse.move(960, 540)

        print("[0.0s - 5.5s] Staff portal overview, navigating to Appointments queue...")
        time.sleep(1.5)
        smooth_move(page, 960, 540, 200, 320, steps=35, delay=0.03)
        time.sleep(1.0)

        apt_nav = page.locator('nav a[href*="/appointments"], a[href="/appointments"]').first
        box_nav = apt_nav.bounding_box()
        if box_nav:
            smooth_move(page, 200, 320, box_nav["x"] + box_nav["width"]/2, box_nav["y"] + box_nav["height"]/2, steps=25, delay=0.02)
            time.sleep(0.5)
            apt_nav.click()
        else:
            page.goto('http://127.0.0.1:5000/appointments', wait_until='domcontentloaded')

        page.wait_for_load_state('networkidle')
        page.evaluate(CURSOR_SCRIPT)
        page.mouse.move(500, 350)
        time.sleep(1.5)

        print("[5.5s - 14.0s] Inspecting incoming booking requests...")
        smooth_scroll(page, 200, steps=25, delay=0.04)
        time.sleep(2.5)

        print("[14.0s - 21.0s] Verifying mapped clinical specialist roles...")
        row = page.locator('table tbody tr').first
        box_row = row.bounding_box()
        if box_row:
            smooth_move(page, 500, 350, box_row["x"] + 850, box_row["y"] + 35, steps=35, delay=0.03)
            time.sleep(3.5)

        print("[21.0s - 26.0s] Confirming appointment...")
        status_select = row.locator('select')
        if status_select.count() > 0:
            box_sel = status_select.bounding_box()
            if box_sel:
                smooth_move(page, box_row["x"] + 850, box_row["y"] + 35, box_sel["x"] + box_sel["width"]/2, box_sel["y"] + box_sel["height"]/2, steps=25, delay=0.02)
                time.sleep(0.5)
                status_select.select_option(label='Confirmed')
                time.sleep(1.0)
        else:
            conf_btn = row.locator('input[value="confirmed"], button:has-text("Confirmed"), a:has-text("Confirmed")').first
            if conf_btn.count() > 0:
                box_c = conf_btn.bounding_box()
                if box_c:
                    smooth_move(page, box_row["x"] + 850, box_row["y"] + 35, box_c["x"] + box_c["width"]/2, box_c["y"] + box_c["height"]/2, steps=25, delay=0.02)
                    conf_btn.click()
                    time.sleep(1.0)

        save_btn = row.locator('button:has-text("Save")').first
        if save_btn.count() > 0:
            box_s = save_btn.bounding_box()
            if box_s:
                smooth_move(page, box_s["x"] - 50, box_s["y"], box_s["x"] + box_s["width"]/2, box_s["y"] + box_s["height"]/2, steps=20, delay=0.02)
                time.sleep(0.4)
                save_btn.click()
                time.sleep(1.5)

        print("[26.0s - 28.5s] Wrapping up Scene 3...")
        smooth_scroll(page, 150, steps=20, delay=0.04)
        time.sleep(2.0)

        video_path = page.video.path()
        context.close()
        browser.close()
        print("Raw video recorded at:", video_path)
        return video_path

if __name__ == "__main__":
    record_scene_3()
