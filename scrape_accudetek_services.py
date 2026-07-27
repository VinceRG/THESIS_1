import csv
import json
import re
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path


PAGES = {
    "home": "https://accudetekhealthdiagnostics.com/",
    "services": "https://accudetekhealthdiagnostics.com/services/",
    "laboratory": "https://accudetekhealthdiagnostics.com/laboratory/",
    "drug_testing": "https://accudetekhealthdiagnostics.com/drug-testing/",
    "imaging": "https://accudetekhealthdiagnostics.com/imaging/",
    "home_service": "https://accudetekhealthdiagnostics.com/home-service/",
    "packages": "https://accudetekhealthdiagnostics.com/packages/",
}

RAW_HTML_DIR = Path("uploads") / "accudetek_raw_pages"

SKIP_TEXTS = {
    "Skip to content",
    "Search this website",
    "Submit search",
    "Home",
    "About Us",
    "Services",
    "Home Service",
    "Packages",
    "Clinic Location",
    "Careers",
    "Contact Us",
    "Facebook",
    "Instagram",
    "Mail",
    "Close Menu",
    "Book Appointment",
    "Schedule Appointment",
    "Learn more",
    "See More",
    "Visit Us",
    "Our Services",
    "Accudetek Health Diagnostics",
}

FOOTER_STARTS = (
    "Accudetek Health Diagnostics is a Clinical Laboratory",
    "Address:",
    "Monday - Saturday",
    "Sunday",
    "CP No:",
)


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.current_tag = None
        self.capture = []

    def handle_starttag(self, tag, attrs):
        if tag in {"h1", "h2", "h3", "li", "p"}:
            self.current_tag = tag
            self.capture = []
        elif tag == "br" and self.current_tag:
            self.capture.append(" ")

    def handle_data(self, data):
        if self.current_tag:
            self.capture.append(data)

    def handle_endtag(self, tag):
        if tag == self.current_tag:
            text = re.sub(r"\s+", " ", "".join(self.capture)).strip()
            if text:
                self.items.append({"tag": tag, "text": text})
            self.current_tag = None
            self.capture = []


def fetch_text(page_key, url):
    raw_path = RAW_HTML_DIR / f"{page_key}.html"
    if raw_path.exists():
        html = raw_path.read_text(encoding="utf-8", errors="replace")
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8", errors="replace")

    parser = TextParser()
    parser.feed(html)

    cleaned = []
    for item in parser.items:
        text = item["text"].strip()
        if not text or text in SKIP_TEXTS:
            continue
        if any(text.startswith(prefix) for prefix in FOOTER_STARTS):
            break
        if text.startswith("#Pasyente"):
            continue
        cleaned.append(item)
    return cleaned


def category_for_heading(text):
    lower = text.lower()
    laboratory_sections = {
        "laboratory",
        "hematology",
        "hermatology",
        "blood chemistry",
        "electrolytes",
        "clinical microscopy",
        "enzymes",
        "bacteriology",
        "hormones",
        "serology",
        "thyroid function test",
        "random urine test",
        "24 hours urine test",
        "hepatitis",
        "histopathologyu",
        "fluid analysis",
        "others",
        "laboratory price list",
    }
    imaging_sections = {
        "imaging",
        "x-ray",
        "xray",
        "ultrasound",
        "2d echocardiogram",
        "ecg",
        "duplex scan",
        "radiology price list",
        "imaging price list",
        "cardiology price list",
        "vascular studies",
        "2d echo pedia",
        "2d echo adult",
    }
    if "vaccine" in lower:
        return "Vaccine"
    if lower in laboratory_sections:
        return "Laboratory"
    if "drug" in lower:
        return "Drug Testing"
    if lower in imaging_sections:
        return "Imaging/Cardiology"
    if "consultation" in lower:
        return "Medical Consultation"
    if "pre-employment" in lower:
        return "Pre-Employment Checkup"
    if "annual" in lower:
        return "Annual Checkup"
    if "home service" in lower or "comfort of your home" in lower:
        return "Home Service"
    if "package" in lower:
        return "Package"
    return text


def parse_services(items, source_key):
    rows = []
    heading = ""
    price_buffer = []
    service_buffer = []

    def flush_pairs():
        nonlocal service_buffer, price_buffer
        if not service_buffer:
            price_buffer = []
            return

        for index, name in enumerate(service_buffer):
            price = price_buffer[index] if index < len(price_buffer) else ""
            rows.append({
                "source_page": source_key,
                "category": category_for_heading(heading),
                "section": heading,
                "service_name": name,
                "price_php": price,
            })
        service_buffer = []
        price_buffer = []

    for item in items:
        tag = item["tag"]
        text = item["text"]
        if tag in {"h1", "h2", "h3"}:
            if text.lower() == "price" or text.startswith("*100 PHP"):
                price_buffer = []
                continue
            if service_buffer:
                flush_pairs()
            heading = text
            continue

        if tag != "li":
            continue

        normalized = re.sub(r"^\d+\.\s*", "", text).strip()
        price_match = re.match(r"PHP\s*([0-9,.]+)", normalized, flags=re.I)
        if price_match:
            price_buffer.append(price_match.group(1).replace(",", ""))
        else:
            service_buffer.append(normalized)

    flush_pairs()
    return rows


def parse_packages(items):
    rows = []
    package_name = ""
    for item in items:
        text = item["text"]
        if item["tag"] in {"h1", "h2", "h3"}:
            continue
        if item["tag"] == "p" and "package" in text.lower() and len(text) < 90:
            package_name = text
            continue
        if item["tag"] == "li" and package_name:
            rows.append({
                "source_page": "packages",
                "package_name": package_name,
                "included_service": re.sub(r"^\d+\.\s*", "", text).strip(),
            })
    return rows


def dedupe_services(rows):
    seen = set()
    unique_rows = []
    for row in rows:
        key = (row["category"], row["section"], row["service_name"], row["price_php"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def main():
    scraped_pages = {key: fetch_text(key, url) for key, url in PAGES.items()}

    service_rows = []
    for key in ["services", "laboratory", "drug_testing", "imaging", "home_service"]:
        service_rows.extend(parse_services(scraped_pages[key], key))
    service_rows = dedupe_services(service_rows)

    package_rows = parse_packages(scraped_pages["packages"])
    summary = {
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
        "source_urls": PAGES,
        "service_count": len(service_rows),
        "package_item_count": len(package_rows),
        "service_categories": sorted({row["category"] for row in service_rows}),
        "notes": [
            "Public Accudetek pages were used as the source.",
            "Prices are stored only when the website presents a PHP price beside the item.",
            "Review this list with the clinic before using it as the live appointment catalog.",
        ],
    }

    out_dir = Path("uploads")
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "accudetek_services_scraped.json"
    csv_path = out_dir / "accudetek_services_scraped.csv"
    package_csv_path = out_dir / "accudetek_packages_scraped.csv"
    summary_path = out_dir / "accudetek_services_summary.txt"

    json_path.write_text(
        json.dumps({"summary": summary, "services": service_rows, "packages": package_rows}, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_page", "category", "section", "service_name", "price_php"])
        writer.writeheader()
        writer.writerows(service_rows)

    with package_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_page", "package_name", "included_service"])
        writer.writeheader()
        writer.writerows(package_rows)

    summary_lines = [
        "Accudetek Website Service Scrape",
        "=================================",
        f"Scraped at: {summary['scraped_at']}",
        f"Services found: {summary['service_count']}",
        f"Package items found: {summary['package_item_count']}",
        "",
        "Categories:",
    ]
    summary_lines.extend(f"- {category}" for category in summary["service_categories"])
    summary_lines.extend(["", "Source URLs:"])
    summary_lines.extend(f"- {name}: {url}" for name, url in PAGES.items())
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(json_path)
    print(csv_path)
    print(package_csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
