import os
import json
import base64
import requests
import shutil
from fpdf import FPDF
from PIL import Image

# Define base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
JSON_DIR = os.path.join(BASE_DIR, "json")
PDF_DIR = os.path.join(BASE_DIR, "pdf")

# Setup folder structure
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# Copy a sample image from evaluation/test_data/ if images/ is empty
images_in_dir = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not images_in_dir:
    test_data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "evaluation", "test_data"))
    if os.path.exists(test_data_dir):
        sample_images = [f for f in os.listdir(test_data_dir) if f.lower().endswith('.png')]
        if sample_images:
            sample_source = os.path.join(test_data_dir, sample_images[0])
            sample_dest = os.path.join(IMAGES_DIR, sample_images[0])
            shutil.copy(sample_source, sample_dest)
            print(f"[*] Copied sample image '{sample_images[0]}' from evaluation/test_data/ as a starter.")
            images_in_dir = [sample_images[0]]
        else:
            print("[!] No sample images found in evaluation/test_data/.")
    else:
        print("[!] Evaluation test data directory not found.")

# Load dynamic configurations from interpretation-api
llm_models_path = os.path.abspath(os.path.join(BASE_DIR, "..", "interpretation-api", "config", "llm_models.json"))
llm_list = []
llm_names = {}
if os.path.exists(llm_models_path):
    try:
        with open(llm_models_path, "r", encoding="utf-8") as f:
            llm_list = json.load(f)
            for m in llm_list:
                llm_names[m["id"]] = m["name"]
    except Exception as e:
        print(f"[!] Error loading llm_models.json: {e}")
else:
    print(f"[!] llm_models.json not found at {llm_models_path}")

styles_path = os.path.abspath(os.path.join(BASE_DIR, "..", "interpretation-api", "config", "response_styles.json"))
style_list = []
style_names = {}
if os.path.exists(styles_path):
    try:
        with open(styles_path, "r", encoding="utf-8") as f:
            style_list = json.load(f)
            for s in style_list:
                style_names[s["id"]] = s["name"]
    except Exception as e:
        print(f"[!] Error loading response_styles.json: {e}")
else:
    print(f"[!] response_styles.json not found at {styles_path}")

API_URL = "http://127.0.0.1:8080/interpret"
INFERENCE_MODEL = "vit-b16-augreg-in21k"

def sanitize_text(text):
    if not text:
        return ""
    replacements = {
        "—": "-",    # em-dash
        "–": "-",    # en-dash
        "“": '"',    # left double smart quote
        "”": '"',    # right double smart quote
        "‘": "'",    # left single smart quote
        "’": "'",    # right single smart quote
        "…": "...",  # ellipsis
        "•": "*",    # bullet point
        "\u2013": "-", # en-dash
        "\u2014": "-", # em-dash
        "\u201c": '"', # smart double open
        "\u201d": '"', # smart double close
        "\u2018": "'", # smart single open
        "\u2019": "'", # smart single close
        "\u2026": "...", # ellipsis
        "\u2022": "*", # bullet
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    return text.encode('latin-1', 'replace').decode('latin-1')

def print_markdown_paragraph(pdf, text, font_size=9.5, line_height=4.5):
    """
    Parses and renders markdown text to FPDF using the python-markdown library and fpdf2's write_html.
    """
    import markdown
    html = markdown.markdown(text)
    html = html.replace('<p>', '<p style="line-height: 1.5;">')
    html = html.replace('<li>', '<li style="line-height: 1.5;">')
    pdf.set_font("helvetica", size=font_size)
    pdf.write_html(html)
    pdf.ln(line_height)

def generate_pdf_page(pdf, image_path, image_name, llm_id, llm_name, style_id, style_name, preds, interpretation):
    """
    Appends a formatted validation page to the master PDF document, keeping original aspect ratio of the image.
    """
    image_name = sanitize_text(image_name)
    llm_name = sanitize_text(llm_name)
    style_name = sanitize_text(style_name)
    interpretation = sanitize_text(interpretation)

    text_len = len(interpretation)
    if text_len >= 4000:
        font_size = 7.0
        line_height = 2.8
        img_display_w = 55
    elif text_len >= 2000:
        font_size = 8.0
        line_height = 3.2
        img_display_w = 60
    else:
        font_size = 9.0
        line_height = 3.8
        img_display_w = 65

    pdf.add_page()
    
    # 1. Header
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 8, "FORM VALIDASI INTERPRETASI KEPRIBADIAN", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("helvetica", "I", 9)
    pdf.cell(0, 5, "Asesmen Kualitas Hasil Interpretasi Berdasarkan Big Five Personality", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(3)
    
    current_y = pdf.get_y()
    pdf.line(15, current_y, 195, current_y)
    pdf.ln(4)
    
    # 2. Large Centered Image | Metadata Box Below
    y_start = pdf.get_y()
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
        aspect_ratio = img_h / img_w
        img_display_h = img_display_w * aspect_ratio
    except Exception as e:
        print(f"[!] Error loading image size for {image_name}: {e}")
        img_display_h = img_display_w
    
    x_center = (210 - img_display_w) / 2
    pdf.image(image_path, x=x_center, y=y_start, w=img_display_w)
    
    y_below_img = y_start + img_display_h + 3
    pdf.set_y(y_below_img)
    
    scores = []
    for trait, value in preds.items():
        short_trait = trait[0]
        scores.append(f"{short_trait}: {value:.3f}")
    scores_str = " | ".join(scores)

    pdf.set_fill_color(245, 245, 245)
    box_y = pdf.get_y()
    pdf.rect(15, box_y, 180, 25, "F")
    
    pdf.set_xy(18, box_y + 2)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.cell(32, 5, "Nama File Gambar:")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(58, 5, image_name)
    
    pdf.set_font("helvetica", "B", 8.5)
    pdf.cell(32, 5, "Model Inferensi:")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(0, 5, INFERENCE_MODEL, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_x(18)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.cell(32, 5, "Model LLM Evaluasi:")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(58, 5, llm_name)
    
    pdf.set_font("helvetica", "B", 8.5)
    pdf.cell(32, 5, "Gaya Respon:")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(0, 5, style_name, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_x(18)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.cell(32, 5, "Prediksi Big Five:")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(0, 5, scores_str, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_x(18)
    pdf.set_font("helvetica", "I", 7.5)
    pdf.cell(0, 4.5, "*O: Openness | C: Conscientiousness | E: Extraversion | A: Agreeableness | N: Neuroticism", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_y(box_y + 25 + 4)
    
    pdf.set_font("helvetica", "B", 10.5)
    pdf.cell(0, 7, "Hasil Interpretasi Kepribadian (Bahasa Indonesia):", new_x="LMARGIN", new_y="NEXT")
    
    line_y = pdf.get_y()
    pdf.line(15, line_y, 195, line_y)
    pdf.ln(3)
    
    print_markdown_paragraph(pdf, interpretation, font_size=font_size, line_height=line_height)
    pdf.ln(4)
    
    pdf.add_page()
        
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 6, "FORMULIR PENILAIAN VALIDASI PSIKOLOG", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "I", 8.5)
    pdf.cell(0, 5.5, "Mohon beri tanda silang/centang [ X ] pada angka 1 sampai 5 untuk kriteria berikut:", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    x_left = pdf.get_x()
    pdf.set_font("helvetica", "B", 9)
    pdf.cell(90, 8, " Kriteria Asesmen", border=1, align="L")
    pdf.cell(18, 8, "1 (S. Buruk)", border=1, align="C")
    pdf.cell(18, 8, "2 (Buruk)", border=1, align="C")
    pdf.cell(18, 8, "3 (Cukup)", border=1, align="C")
    pdf.cell(18, 8, "4 (Baik)", border=1, align="C")
    pdf.cell(18, 8, "5 (S. Baik)", border=1, align="C")
    pdf.ln(8)
    
    metrics_path = os.path.abspath(os.path.join(BASE_DIR, "..", "evaluation", "metrics_config.json"))
    criterias = []
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                metrics_list = config.get("metrics", [])
                for idx, m in enumerate(metrics_list, 1):
                    desc = m["criteria"].split(". ")[0] + "."
                    criterias.append((f"{idx}. {m['name']}", desc))
        except Exception as e:
            print(f"[!] Error loading metrics_config.json: {e}")
            
    if not criterias:
        criterias = [
            ("1. Koherensi Gambar", "Evaluasi apakah deskripsi fitur fisik atau kesan visual sesuai dengan foto."),
            ("2. Akurasi Penilaian Psikologis", "Evaluasi apakah analisis secara akurat mencerminkan skor numerik Big Five."),
            ("3. Persona Psikolog", "Evaluasi apakah respon menggunakan nada hangat, empatik, dan profesional.")
        ]
        
    overall_idx = len(criterias) + 1
    criterias.append((f"{overall_idx}. Penilaian Keseluruhan", "Evaluasi menyeluruh terhadap kelayakan and kualitas respon interpretasi kepribadian."))
    
    for title, desc in criterias:
        row_start_x = pdf.get_x()
        row_start_y = pdf.get_y()
        
        pdf.set_font("helvetica", "B", 8.5)
        pdf.cell(90, 4.5, f" {sanitize_text(title)}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(row_start_x)
        pdf.set_font("helvetica", "I", 7.5)
        pdf.multi_cell(90, 3.5, f" {sanitize_text(desc)}")
        
        row_end_y = pdf.get_y()
        row_height = max(10, row_end_y - row_start_y + 1.5)
        
        pdf.rect(row_start_x, row_start_y, 90, row_height)
        
        for val in range(1, 6):
            pdf.set_xy(row_start_x + 90 + (val - 1) * 18, row_start_y)
            pdf.set_font("helvetica", "", 9.5)
            pdf.cell(18, row_height, "[     ]", border=1, align="C")
            
        pdf.set_xy(row_start_x, row_start_y + row_height)
        
    pdf.ln(3)
    
    pdf.set_font("helvetica", "B", 9.5)
    pdf.cell(0, 5, "Catatan / Saran Perbaikan dari Psikolog:", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.cell(0, 16, "", border=1, new_x="LMARGIN", new_y="NEXT")

def run_pipeline():
    print("="*60)
    print("      Personality Interpretation Validation Pipeline")
    print("="*60)
    
    try:
        r = requests.get("http://127.0.0.1:8080/system/status", timeout=3)
        print("[*] Checked interpretation-api: Active and listening.")
    except Exception:
        print("[WARNING] interpretation-api does not seem to respond at http://127.0.0.1:8080.")
        print("[WARNING] Please make sure your Go interpretation-api is running locally!")
        print("-" * 60)
        
    images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"[!] No images found to process. Please place images in {IMAGES_DIR} directory.")
        return
        
    print(f"[*] Found {len(images)} image(s) to evaluate.")
    print(f"[*] Found {len(llm_list)} LLM model(s) and {len(style_list)} Response Style(s).")
    total_combinations = len(images) * len(llm_list) * len(style_list)
    print(f"[*] Total combinations to run: {total_combinations}")
    print("-" * 60)
    
    processed = 0
    skipped = 0
    failures = 0
    
    master_pdf = FPDF(format='A4', unit='mm')
    master_pdf.set_margins(15, 15, 15)
    has_pages = False
    
    for img_file in images:
        image_path = os.path.join(IMAGES_DIR, img_file)
        image_name, img_ext = os.path.splitext(img_file)
        
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        except Exception as e:
            print(f"[!] Failed to read image {img_file}: {e}")
            failures += 1
            continue
            
        for llm in llm_list:
            llm_id = llm["id"]
            llm_name = llm["name"]
            llm_clean = llm_id.replace("/", "_")
            
            for style in style_list:
                style_id = style["id"]
                style_name = style["name"]
                
                filename_base = f"{image_name}_{llm_clean}_{style_id}"
                json_path = os.path.join(JSON_DIR, f"{filename_base}.json")
                
                if os.path.exists(json_path):
                    skipped += 1
                    try:
                        with open(json_path, "r", encoding="utf-8") as json_f:
                            saved_data = json.load(json_f)
                            
                        preds = saved_data["results"]["predictions"]
                        interpretation = saved_data["results"]["interpretation"]
                        
                        print(f"[-] Loaded from cache: {img_file} | {llm_name} | {style_name}")
                        
                        generate_pdf_page(
                            pdf=master_pdf,
                            image_path=image_path,
                            image_name=img_file,
                            llm_id=llm_id,
                            llm_name=llm_name,
                            style_id=style_id,
                            style_name=style_name,
                            preds=preds,
                            interpretation=interpretation
                        )
                        has_pages = True
                        continue
                    except Exception as e:
                        print(f"    [!] Error reading JSON cache for {filename_base}, reprocessing: {e}")
                
                print(f"[+] Processing API: {img_file} | {llm_name} | {style_name}")
                
                try:
                    with open(image_path, "rb") as img_f:
                        files = {"image": (img_file, img_f, f"image/{img_ext.replace('.', '')}")}
                        data = {
                            "inference_model": INFERENCE_MODEL,
                            "llm_model": llm_id,
                            "style_id": style_id
                        }
                        
                        resp = requests.post(API_URL, files=files, data=data, timeout=300)
                        
                    if resp.status_code != 200:
                        print(f"    [!] API returned error status {resp.status_code}: {resp.text}")
                        failures += 1
                        continue
                        
                    resp_json = resp.json()
                    preds = resp_json.get("predictions", {})
                    interpretation = resp_json.get("interpretation", "")
                    
                    json_data = {
                        "parameters": {
                            "inference_model": INFERENCE_MODEL,
                            "llm_model": llm_id,
                            "response_style": style_id
                        },
                        "results": {
                            "predictions": preds,
                            "interpretation": interpretation
                        },
                        "image_base64": img_base64
                    }
                    
                    with open(json_path, "w", encoding="utf-8") as json_f:
                        json.dump(json_data, json_f, indent=4, ensure_ascii=False)
                        
                    generate_pdf_page(
                        pdf=master_pdf,
                        image_path=image_path,
                        image_name=img_file,
                        llm_id=llm_id,
                        llm_name=llm_name,
                        style_id=style_id,
                        style_name=style_name,
                        preds=preds,
                        interpretation=interpretation
                    )
                    has_pages = True
                    processed += 1
                    print(f"    [Success] JSON saved and compiled to report.")
                    
                except requests.exceptions.ConnectionError:
                    print("    [!] Error: Cannot connect to interpretation-api. Is it running?")
                    failures += 1
                except Exception as e:
                    print(f"    [!] Error during pipeline processing: {e}")
                    failures += 1

    if has_pages:
        if len(images) == 1:
            img_name, _ = os.path.splitext(images[0])
            report_name = f"validation_report_{img_name}.pdf"
        else:
            report_name = "validation_report_combined.pdf"
            
        master_pdf_path = os.path.join(PDF_DIR, report_name)
        try:
            master_pdf.output(master_pdf_path)
            print("-" * 60)
            print(f"[*] Compiled Master PDF saved to: {master_pdf_path}")
        except Exception as e:
            print(f"[!] Error saving master PDF: {e}")
            
    print("="*60)
    print("                      Pipeline Summary")
    print("="*60)
    print(f"Total newly processed: {processed}")
    print(f"Total loaded from cache: {skipped}")
    print(f"Total failures:          {failures}")
    print("="*60)
    print(f"[*] JSON output directory: {JSON_DIR}")
    print(f"[*] PDF report location:   {PDF_DIR}")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
