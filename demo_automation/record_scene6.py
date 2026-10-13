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

def smooth_move(page, start_x, start_y, end_x, end_y, steps=25, delay=0.02):
    for i in range(1, steps + 1):
        x = start_x + (end_x - start_x) * (i / steps)
        y = start_y + (end_y - start_y) * (i / steps)
        page.mouse.move(x, y)
        time.sleep(delay)

def smooth_scroll(page, scroll_by, steps=25, delay=0.025):
    step_delta = scroll_by / steps
    for _ in range(steps):
        page.mouse.wheel(0, step_delta)
        time.sleep(delay)

def record_scene_6():
    output_dir = os.path.abspath("demo_automation/raw_videos")
    os.makedirs(output_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )
        
        # 1. Login silently in a temporary unrecorded context
        login_context = browser.new_context()
        login_page = login_context.new_page()
        login_page.goto('http://127.0.0.1:5000/login', wait_until='domcontentloaded')
        login_page.fill('input[name="username"]', 'superadmin')
        login_page.fill('input[name="password"]', '12345678')
        login_page.click('button[type="submit"]')
        login_page.wait_for_load_state('networkidle')
        state = login_context.storage_state()
        login_context.close()

        # 2. Recording context with saved auth state
        context = browser.new_context(
            storage_state=state,
            record_video_dir=output_dir,
            record_video_size={"width": 1920, "height": 1080},
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        # Start directly on Resource Planning
        page.goto('http://127.0.0.1:5000/resources', wait_until='networkidle')
        page.evaluate(CURSOR_SCRIPT)
        page.mouse.move(960, 540)
        time.sleep(0.5)

        # [0.0s - 6.0s] Introduction & Practical Staffing Decision
        print("[0.0s - 6.0s] Overview & Page Header...")
        smooth_move(page, 960, 540, 520, 180, steps=30, delay=0.025)
        time.sleep(2.5)

        # [6.0s - 11.0s] RF Model Consultation Demand
        print("[6.0s - 11.0s] Highlighting Forecast Month & Expected Patients...")
        smooth_move(page, 520, 180, 430, 240, steps=25, delay=0.025) # Forecast Month
        time.sleep(1.2)
        smooth_move(page, 430, 240, 780, 240, steps=25, delay=0.025) # Forecasted Patients
        time.sleep(1.5)

        # [11.0s - 17.5s] System Recommends Clinical Staff & Specialist Roles
        print("[11.0s - 17.5s] Top Staff Role Needed & Capacity Status...")
        smooth_move(page, 780, 240, 1140, 240, steps=25, delay=0.025) # Top Staff Role
        time.sleep(1.2)
        smooth_move(page, 1140, 240, 1500, 240, steps=25, delay=0.025) # Capacity Status
        time.sleep(1.0)
        smooth_scroll(page, 280, steps=25, delay=0.025)
        time.sleep(0.5)
        smooth_move(page, 1500, 240, 800, 480, steps=25, delay=0.025) # Staff Capacity Plan
        time.sleep(1.5)

        # [17.5s - 23.5s] Daily Forecast (Supervisor shift scheduling)
        print("[17.5s - 23.5s] Daily Forecast Chart & Supervisor Shift Scheduling...")
        smooth_scroll(page, 520, steps=25, delay=0.025)
        time.sleep(0.6)
        smooth_move(page, 800, 480, 680, 520, steps=25, delay=0.025) # Daily Staff Chart
        time.sleep(2.0)
        smooth_move(page, 680, 520, 1380, 480, steps=25, delay=0.025) # Admin Staffing Alerts
        time.sleep(1.8)

        # [23.5s - 30.5s] Monthly Forecast (Strategic physician & specialist hiring)
        print("[23.5s - 30.5s] Monthly Forecast Chart & Strategic Hiring...")
        smooth_scroll(page, 620, steps=25, delay=0.025)
        time.sleep(0.6)
        smooth_move(page, 1380, 480, 660, 520, steps=25, delay=0.025) # Monthly Staff Forecast Chart
        time.sleep(2.5)
        smooth_move(page, 660, 520, 1380, 520, steps=25, delay=0.025) # Monthly Mapped Requirement Table
        time.sleep(2.0)

        # [30.5s - 38.0s] Bridging the Gap (Conclusion)
        print("[30.5s - 38.0s] Conclusion & System Summary...")
        smooth_scroll(page, -420, steps=25, delay=0.025)
        time.sleep(0.8)
        smooth_move(page, 1380, 520, 960, 500, steps=25, delay=0.025)
        time.sleep(3.2)

        video_path = page.video.path()
        context.close()
        browser.close()
        print("Raw video recorded at:", video_path)
        return video_path

if __name__ == "__main__":
    record_scene_6()
