import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from paddleocr import PaddleOCR
from google.cloud import translate_v3 as translate

# Silence C++ backend logging
os.environ["GLOG_minloglevel"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- JSON Config Loader ---
def load_configs(path="terms.json"):
    if not os.path.exists(path):
        print(f"⚠️ Config not found at {path}, using hardcoded defaults.")
        return ["color details", "quantity", "cost", "unit of measure"]
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    headers = [str(h).lower() for h in data.get("dnt_headers", [])]
    return headers

DNT_HEADERS = load_configs()

# --- Setup Translation Client ---
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
client = translate.TranslationServiceClient()
location = "global"
parent = f"projects/{project_id}/locations/{location}"

def translate_text(text, target_lang='zh'):
    if not text.strip(): return text
    
    try:
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": "en",
                "target_language_code": target_lang,
            }
        )
        return response.translations[0].translated_text
    except Exception as e:
        print(f"   ⚠️ Translation failed for '{text[:10]}...': {e}")
        return text 

def extract_structural_grid(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    return cv2.add(h_lines, v_lines), v_lines

def slice_merged_blocks(blocks, v_lines):
    sliced_blocks = []
    
    for raw_text, x_min, y_min, x_max, y_max, cx, cy in blocks:
        if (x_max - x_min) < 50:
            sliced_blocks.append((raw_text, x_min, y_min, x_max, y_max, cx, cy))
            continue
            
        line_crossings = []
        for x in range(x_min + 5, x_max - 5):
            if v_lines[int(cy), x] > 128:
                line_crossings.append(x)
                
        if line_crossings:
            split_x = int(np.mean(line_crossings))
            ratio = (split_x - x_min) / float(x_max - x_min)
            split_index = int(len(raw_text) * ratio)
            
            space_idx = raw_text.rfind(' ', 0, split_index + 3)
            if space_idx == -1: 
                space_idx = raw_text.find(' ', split_index - 3)
                
            if space_idx != -1 and 0 < space_idx < len(raw_text) - 1:
                text_left = raw_text[:space_idx].strip()
                text_right = raw_text[space_idx:].strip()
            else: 
                text_left = raw_text[:split_index].strip()
                text_right = raw_text[split_index:].strip()

            l_xmax = split_x - 2
            l_cx = x_min + ((l_xmax - x_min) / 2)
            if text_left:
                sliced_blocks.append((text_left, x_min, y_min, l_xmax, y_max, l_cx, cy))
                
            r_xmin = split_x + 2
            r_cx = r_xmin + ((x_max - r_xmin) / 2)
            if text_right:
                sliced_blocks.append((text_right, r_xmin, y_min, x_max, y_max, r_cx, cy))
                
            print(f"   ✂️ Sliced merged block at X:{split_x} -> ['{text_left}', '{text_right}']")
        else:
            sliced_blocks.append((raw_text, x_min, y_min, x_max, y_max, cx, cy))
            
    return sliced_blocks

def run_translation_engine(target_lang='zh'):
    print(f"\n📦 INITIALIZING BATCH TRANSLATION ENGINE (Target: {target_lang})")

    # Dynamic Path Setup
    if os.path.exists("/input"):
        input_dir = "/input"
    elif os.path.exists("input"):
        input_dir = "input"
    else:
        input_dir = "../input"

    if os.path.exists("/output"):
        output_dir = "/output"
    elif os.path.exists("output"):
        output_dir = "output"
    else:
        output_dir = "../output"
        
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"❌ ERROR: Input directory '{input_dir}' not found.")
        return

    # Scan for images
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"❌ ERROR: No images found in '{input_dir}'.")
        print("   Make sure your Docker volume mount (-v) is pointing to the right folder.")
        return

    print(f"📁 Found {len(image_files)} image(s) to process.")

    # ==========================================
    # LOAD AI MODELS ONCE (Memory Optimization)
    # ==========================================
    print("⏳ Loading PaddleOCR AI Models into memory...")
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    # ==========================================
    # BATCH PROCESSING LOOP
    # ==========================================
    for img_index, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        print(f"\n==========================================")
        print(f"🔄 PROCESSING [{img_index}/{len(image_files)}]: {filename}")
        print(f"==========================================")

        cv_img = cv2.imread(img_path)
        if cv_img is None:
            print(f"❌ ERROR: OpenCV could not read {filename}. Skipping.")
            continue
            
        img_h, img_w, _ = cv_img.shape
        pil_img = Image.open(img_path).convert("RGB")
        
        # 1. CANDIDATE BOX GENERATION
        b, g, r = cv2.split(cv_img)
        edges = cv2.bitwise_or(cv2.Canny(b, 50, 150), cv2.bitwise_or(cv2.Canny(g, 50, 150), cv2.Canny(r, 50, 150)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        candidate_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 5000 or (w >= img_w - 20 and h >= img_h - 20): continue
            if not (0.3 <= w / float(h) <= 3.0): continue
            candidate_boxes.append({
                "coords": [x, y, x + w, y + h], 
                "score": cv2.countNonZero(edges[y:y+h, x:x+w]) / float(area),
                "area": area
            })

        candidate_boxes.sort(key=lambda item: item["score"], reverse=True)
        forbidden_zone = [candidate_boxes[0]["coords"][0]-10, candidate_boxes[0]["coords"][1]-10, candidate_boxes[0]["coords"][2]+10, candidate_boxes[0]["coords"][3]+10] if candidate_boxes else [600, 50, 1150, 450]
        largest_boxes = sorted(candidate_boxes, key=lambda item: item["area"], reverse=True)
        master_grid_mask, v_lines = extract_structural_grid(cv_img)

        # 2. OCR EXTRACTION & SLICING
        results = ocr_model.ocr(img_path, cls=True)
        if not results or not results[0]: 
            print(f"⚠️ No text found in {filename}. Skipping.")
            continue

        raw_blocks = []
        for line in results[0]:
            coords, text_data = line[0], line[1]
            raw_text = text_data[0].strip()
            if not raw_text: continue
                
            x_min, y_min = int(min(pt[0] for pt in coords)), int(min(pt[1] for pt in coords))
            x_max, y_max = int(max(pt[0] for pt in coords)), int(max(pt[1] for pt in coords))
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            
            if (forbidden_zone[0] <= cx <= forbidden_zone[2] and forbidden_zone[1] <= cy <= forbidden_zone[3]):
                continue 
                
            raw_blocks.append((raw_text, x_min, y_min, x_max, y_max, cx, cy))

        raw_blocks = slice_merged_blocks(raw_blocks, v_lines)

        # 3. DYNAMIC TABLE & COLUMN MAPPING
        table_bbox = None
        dnt_column_zones = []
        dnt_y_ceiling = None
        
        for raw_text, x_min, y_min, x_max, y_max, cx, cy in raw_blocks:
            raw_text_lower = raw_text.lower()
            if any(h in raw_text_lower for h in DNT_HEADERS):
                if dnt_y_ceiling is None or y_min < dnt_y_ceiling:
                    dnt_y_ceiling = y_min - 5
                    
                if not table_bbox:
                    for box in largest_boxes:
                        bx1, by1, bx2, by2 = box["coords"]
                        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                            table_bbox = box["coords"]
                            break
                
                if table_bbox:
                    left_bound, right_bound = x_min, x_max
                    for x in range(int(cx), int(table_bbox[0]), -1):
                        if v_lines[int(cy), x] > 128:
                            left_bound = x
                            break
                    for x in range(int(cx), int(table_bbox[2])):
                        if v_lines[int(cy), x] > 128:
                            right_bound = x
                            break
                            
                    dnt_column_zones.append((left_bound, right_bound))

        # 3.5 THE FILTER PASS
        blocks_to_translate = []
        for raw_text, x_min, y_min, x_max, y_max, cx, cy in raw_blocks:
            skip_block = False
            if table_bbox and dnt_y_ceiling is not None:
                if table_bbox[0] <= cx <= table_bbox[2] and table_bbox[1] <= cy <= table_bbox[3]:
                    if cy > dnt_y_ceiling:
                        for (z_min, z_max) in dnt_column_zones:
                            if z_min <= cx <= z_max:
                                skip_block = True
                                break
            if not skip_block:
                blocks_to_translate.append((raw_text, x_min, y_min, x_max, y_max))

        # 4. SURGICAL WHITEOUT & GRID HEAL
        clean_draw = ImageDraw.Draw(pil_img)
        for _, x_min, y_min, x_max, y_max in blocks_to_translate:
            clean_draw.rectangle([x_min - 2, y_min - 2, x_max + 2, y_max + 2], fill="white")

        chewed_cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        healed_cv_img = np.where(master_grid_mask[:, :, None] == 255, cv_img, chewed_cv_img)
        pil_img = Image.fromarray(cv2.cvtColor(healed_cv_img, cv2.COLOR_BGR2RGB))

        # 5. MATHEMATICAL RENDERING & TRANSLATION
        print(f"✍️ Rendering {len(blocks_to_translate)} targeted text blocks...")
        draw = ImageDraw.Draw(pil_img)
        
        if blocks_to_translate:
            box_heights = [y_max - y_min for _, _, y_min, _, y_max in blocks_to_translate]
            median_h = max(10, int(np.median(box_heights)))
            master_font_size = int(median_h * 0.75) 
            
            for raw_text, x_min, y_min, x_max, y_max in blocks_to_translate:
                box_w = max(1, x_max - x_min)
                box_h = max(1, y_max - y_min)
                cx, cy = x_min + (box_w / 2), y_min + (box_h / 2)
                
                translated_text = translate_text(raw_text, target_lang=target_lang)
                
                font_size = master_font_size
                try: 
                    font_path = os.path.join(os.path.dirname(__file__), "fonts", "simhei.ttf")
                    font = ImageFont.truetype(font_path, font_size)
                except IOError: 
                    font = ImageFont.load_default()
                    
                bbox = draw.textbbox((0, 0), translated_text, font=font)
                text_w = bbox[2] - bbox[0]
                
                if text_w > box_w and text_w > 0:
                    scale = box_w / float(text_w)
                    font_size = max(8, int(font_size * scale)) 
                    try: 
                        font = ImageFont.truetype(font_path, font_size)
                    except IOError: 
                        font = ImageFont.load_default()
                    bbox = draw.textbbox((0, 0), translated_text, font=font)
                
                text_h = bbox[3] - bbox[1]
                y_paste = (cy - (text_h / 2)) - bbox[1] 
                x_paste = (x_min + 2) - bbox[0]
                
                draw.text((x_paste, y_paste), translated_text, fill="black", font=font)

        # Dynamic Naming to prevent overwriting
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"translated_{base_name}.png")
        pil_img.save(output_path)
        print(f"🎉 Engine complete! Image saved to: {output_path}")

    print("\n✅ BATCH PROCESSING COMPLETE!\n")

if __name__ == "__main__":
   run_translation_engine()