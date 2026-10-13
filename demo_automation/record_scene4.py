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

def record_scene_4():
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

        # Inject cursor into Dashboard
        page.evaluate(CURSOR_SCRIPT)
        page.mouse.move(960, 540)

        print("[0.0s - 5.0s] Overview of consultation volume counters...")
        smooth_move(page, 960, 540, 420, 260, steps=30, delay=0.025)
        time.sleep(2.0)
        smooth_move(page, 420, 260, 680, 260, steps=25, delay=0.025)
        time.sleep(1.5)

        print("[5.0s - 11.0s] Historical trend chart...")
        smooth_scroll(page, 320, steps=25, delay=0.03)
        time.sleep(0.5)
        smooth_move(page, 680, 260, 550, 480, steps=30, delay=0.025)
        time.sleep(2.5)

        print("[11.0s - 17.0s] Top Diagnosis Cases breakdown (Pie Chart)...")
        smooth_move(page, 550, 480, 1250, 480, steps=35, delay=0.025)
        time.sleep(3.0)

        print("[17.0s - 22.5s] Age & Gender distributions...")
        smooth_scroll(page, 450, steps=25, delay=0.035)
        time.sleep(0.5)
        smooth_move(page, 1250, 480, 580, 520, steps=30, delay=0.025)
        time.sleep(1.5)
        smooth_move(page, 580, 520, 1200, 520, steps=30, delay=0.025)
        time.sleep(1.5)

        print("[22.5s - 24.0s] Wrapping up Scene 4...")
        time.sleep(1.5)

        video_path = page.video.path()
        context.close()
        browser.close()
        print("Raw video recorded at:", video_path)
        return video_path

if __name__ == "__main__":
    record_scene_4()
