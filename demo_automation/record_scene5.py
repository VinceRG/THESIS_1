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

def record_scene_5():
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

        print("[0.0s - 8.0s] Navigating to Predictions module...")
        time.sleep(2.0)
        smooth_move(page, 960, 540, 200, 390, steps=35, delay=0.03)
        time.sleep(1.0)
        
        pred_nav = page.locator('nav a[href*="/predict"], a[href="/predict"]').first
        box_nav = pred_nav.bounding_box()
        if box_nav:
            smooth_move(page, 200, 390, box_nav["x"] + box_nav["width"]/2, box_nav["y"] + box_nav["height"]/2, steps=25, delay=0.02)
            time.sleep(0.4)
            pred_nav.click()
        else:
            page.goto('http://127.0.0.1:5000/predict', wait_until='domcontentloaded')

        page.wait_for_load_state('networkidle')
        page.evaluate(CURSOR_SCRIPT)
        page.mouse.move(500, 300)
        time.sleep(2.0)

        print("[8.0s - 16.0s] Inspecting Random Forest Algorithm & Performance Metrics...")
        smooth_move(page, 500, 300, 430, 220, steps=35, delay=0.03) # Algorithm card
        time.sleep(2.5)
        smooth_move(page, 430, 220, 750, 220, steps=30, delay=0.03) # Holdout R2
        time.sleep(2.0)
        smooth_move(page, 750, 220, 1050, 220, steps=30, delay=0.03) # MAE / RMSE
        time.sleep(2.5)

        print("[16.0s - 29.0s] Highlighting Demographics Comparison, Sex Distribution & Multi-dimensional Features...")
        smooth_scroll(page, 480, steps=30, delay=0.04)
        time.sleep(1.5)
        smooth_move(page, 1050, 220, 600, 450, steps=35, delay=0.03) # Model comparison
        time.sleep(4.0)
        smooth_scroll(page, 400, steps=25, delay=0.04) # Segments table
        time.sleep(3.5)

        print("[29.0s - 39.5s] Showing Retrain Model button in Model Controls...")
        smooth_scroll(page, -880, steps=35, delay=0.03)
        time.sleep(1.5)

        retrain_btn = page.locator('button:has-text("Retrain Model"), a:has-text("Retrain Model")').first
        box_btn = retrain_btn.bounding_box()
        if box_btn:
            smooth_move(page, 600, 450, box_btn["x"] + box_btn["width"]/2, box_btn["y"] + box_btn["height"]/2, steps=35, delay=0.03)
            time.sleep(5.0)
        else:
            smooth_move(page, 600, 450, 1100, 220, steps=30, delay=0.03)
            time.sleep(5.0)

        print("[40.0s] Scene 5 visual choreography complete.")

        video_path = page.video.path()
        context.close()
        browser.close()
        print("Raw video recorded at:", video_path)
        return video_path

if __name__ == "__main__":
    record_scene_5()
