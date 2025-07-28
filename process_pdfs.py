import os
import json
import io
from pathlib import Path
import fitz
import pytesseract
from PIL import Image
import numpy as np
from collections import Counter
import re
import cv2



def is_scanned_pdf(pdf_path):
    """Check if PDF is scanned (has little or no text content)"""
    doc = fitz.open(pdf_path)
    text_found = False
    for page in doc:
        if page.get_text().strip():  
            text_found = True
            break
    doc.close()
    return not text_found

def extract_text_with_fitz(pdf_path):
    """Extract all text with attributes using PyMuPDF (for text-based PDFs)"""
    doc = fitz.open(pdf_path)
    all_text_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        blocks = page.get_text("dict")
        
        for block in blocks["blocks"]:
            if "lines" in block:
                block_text = ""
                block_attributes = []
                
                for line in block["lines"]:
                    line_text = ""
                    line_attributes = []
                    
                    for span in line["spans"]:
                        text = span['text'].strip()
                        if text:
                            line_text += text + " "
                            line_attributes.append({
                                'font': span['font'],
                                'size': round(span['size'], 1),
                                'flags': span['flags'],
                                'color': span.get('color', 0)
                            })
                    
                    if line_text.strip() and line_attributes:
                        most_common_font = Counter([attr['font'] for attr in line_attributes]).most_common(1)[0][0]
                        most_common_size = Counter([attr['size'] for attr in line_attributes]).most_common(1)[0][0]
                        most_common_flags = Counter([attr['flags'] for attr in line_attributes]).most_common(1)[0][0]
                        most_common_color = Counter([attr['color'] for attr in line_attributes]).most_common(1)[0][0]
                        
                        all_text_data.append({
                            'text': line_text.strip(),
                            'page': page_num + 1,
                            'font': most_common_font,
                            'size': most_common_size,
                            'flags': most_common_flags,
                            'color': most_common_color,
                            'y_position': line['bbox'][1],
                            'page_height': page_height,
                            'bbox': line['bbox'],
                            'block_bbox': block['bbox'],
                            'is_bold': bool(most_common_flags & 2**4),
                            'is_italic': bool(most_common_flags & 2**1),
                            'is_underlined': bool(most_common_flags & 2**2),
                        })
    
    doc.close()
    return all_text_data

def extract_text_with_ocr(pdf_path):
    """Extract text with bounding boxes using OCR (for scanned PDFs)"""
    doc = fitz.open(pdf_path)
    all_text_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        img_np = np.array(img)
        
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        ocr_data = pytesseract.image_to_data(
            gray, 
            output_type=pytesseract.Output.DICT,
            config='--psm 6'
        )
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if text and int(ocr_data['conf'][i]) > 60: 
                x, y, w, h = (
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['width'][i],
                    ocr_data['height'][i]
                )
                
                font_size = h * 72 / 96  
                is_bold = ocr_data['font_weight'][i] > 500 if 'font_weight' in ocr_data else False
                is_uppercase = text == text.upper()
                ends_with_colon = text.endswith(':')
                starts_with_number = bool(re.match(r'^[0-9IVX]+\.', text))
                
                all_text_data.append({
                    'text': text,
                    'page': page_num + 1,
                    'font': 'ocr_font', 
                    'size': font_size,
                    'flags': (4 if is_bold else 0),  
                    'color': 0,  
                    'y_position': y,
                    'page_height': page_height,
                    'bbox': [x, y, x + w, y + h],
                    'block_bbox': [x, y, x + w, y + h],
                    'is_bold': is_bold,
                    'is_italic': False,  
                    'is_underlined': False, 
                    'is_uppercase': is_uppercase,
                    'ends_with_colon': ends_with_colon,
                    'starts_with_number': starts_with_number
                })
    
    doc.close()
    return all_text_data

def find_majority_text_attributes(all_text_data):
    """Find the most common text attributes (body text)"""
    if not all_text_data:
        return {
            'font': 'ocr_font',
            'size': 12,  
            'flags': 0,
            'color': 0,
            'count': 0
        }
    
    attribute_combinations = []
    
    for item in all_text_data:
        combo = (item['font'], item['size'], item['flags'], item['color'])
        attribute_combinations.append(combo)
    
    most_common_combo = Counter(attribute_combinations).most_common(1)[0]
    majority_attributes = most_common_combo[0]
    
    return {
        'font': majority_attributes[0],
        'size': majority_attributes[1],
        'flags': majority_attributes[2],
        'color': majority_attributes[3],
        'count': most_common_combo[1]
    }

def is_likely_heading(item, majority_attrs, prev_item=None):
    """Enhanced heading detection with multiple criteria including color"""
    if not item['text'].strip():
        return False
    
    text = item['text'].strip()
    text_lower = text.lower()
        
    if len(text) < 3:
        return False
            
    if (re.match(r'^(page|fig(ure)?|table)\s+\d+$', text_lower) or
       re.match(r'^\d+[/-]\d+$', text) or  
       re.match(r'^\d+\.$', text) or       
       re.match(r'^\d+\.\s', text) or      
       re.match(r'^\d{4}$', text) or
       re.match(r'^\d+(\.\d+)?(\s+\d+(\.\d+)?)*$', text) or
       re.match(r'^\d{4}$', text) or
       re.match(r'^[\d\.\,\(\)\s]+$', text) or
       re.match(r'^\d{4}\s+\d{4}$', text) or  
       re.match(r'^[A-Za-z]+\s+\d{4}$', text) or 
       re.match(r'^[A-Za-z]+\s*:\s*[-–—]+$', text) or
       text_lower in ['continued', 'cont\'d', '...', 'date', 'footer', 'header']):
        return False
        
    is_larger = item['size'] > majority_attrs['size']
    is_bold = item.get('is_bold', False) or (item['flags'] & 2**4)
    different_color = item['color'] != majority_attrs['color']
    
    is_uppercase = item.get('is_uppercase', text == text.upper())
    ends_with_colon = item.get('ends_with_colon', text.endswith(':'))
    starts_with_number = item.get('starts_with_number', bool(re.match(r'^(Appendix|Chapter|Section|Part|Art(icle)?|[\dIVX]+\.)', text, re.I)))
    
    is_centered = abs((item['bbox'][0] + item['bbox'][2])/2 - item['block_bbox'][0] - (item['block_bbox'][2] - item['block_bbox'][0])/2) < 10
    is_left_aligned = item['bbox'][0] < item['block_bbox'][0] + 20
            
    is_short = len(text.split()) <= 8  
    has_heading_keywords = bool(re.search(r'\b(Abstract|Introduction|Background|Method|Results|Conclusion|References|Acknowledgements?)\b', text, re.I))
        
    if prev_item:
        vertical_gap = item['bbox'][1] - prev_item['bbox'][3] 
        large_gap = vertical_gap > item['size'] * 1.5  
    else:
        large_gap = True  
            
    score = 0
    if is_larger: score += 2
    if is_bold: score += 2
    if different_color: score += 1
    if is_centered: score += 1
    if is_left_aligned and not is_centered: score += 0.5  
    if is_short: score += 1
    if is_uppercase: score += 1
    if ends_with_colon: score += 1
    if starts_with_number: score += 2
    if has_heading_keywords: score += 2
    if large_gap: score += 2
            
    if (is_uppercase and is_bold and is_larger) or (has_heading_keywords and is_bold):
        score += 3
        
    return score >= 5

def extract_headings(all_text_data, majority_attrs):
    """Extract headings using multiple criteria with improved grouping"""
    headings = []
    prev_item = None
    
    for i, item in enumerate(all_text_data):
        if is_likely_heading(item, majority_attrs, prev_item):        
            next_item = all_text_data[i+1] if i+1 < len(all_text_data) else None
            if next_item and is_likely_heading(next_item, majority_attrs, item):
                combined_text = item['text'] + ' ' + next_item['text']
                combined_item = item.copy()
                combined_item['text'] = combined_text
                combined_item['bbox'] = [
                    min(item['bbox'][0], next_item['bbox'][0]),
                    min(item['bbox'][1], next_item['bbox'][1]),
                    max(item['bbox'][2], next_item['bbox'][2]),
                    max(item['bbox'][3], next_item['bbox'][3])
                ]
                headings.append(combined_item)
                prev_item = combined_item
                continue  
            
            headings.append(item)
        prev_item = item
        
    if not headings:
        for item in all_text_data:
            if (item['size'] >= majority_attrs['size'] and 
                (item.get('is_bold', False) or item['color'] != majority_attrs['color']) and 
                len(item['text'].split()) <= 10):
                headings.append(item)
    
    if not headings:
        return []
    
    headings_sorted = sorted(headings, key=lambda x: (-x['size'], -x.get('is_bold', False), x['bbox'][1]))
                
    unique_combos = {}
    for h in headings_sorted:
        key = (h.get('font', 'ocr_font'), h['size'], h.get('is_bold', False), 
              h.get('is_italic', False), h.get('is_underlined', False), h['color'])
        if key not in unique_combos:
            unique_combos[key] = []
        unique_combos[key].append(h)
    
    sorted_groups = sorted(unique_combos.items(), 
                         key=lambda x: (-x[1][0]['size'], 
                                      -x[1][0].get('is_bold', False), 
                                      x[1][0]['color'] != majority_attrs['color']))
    
    for level, (_, group_items) in enumerate(sorted_groups, 1):
        for item in group_items:
            item['level'] = f'H{min(level, 3)}' 
    
    headings_sorted = sorted(headings_sorted, key=lambda x: (x['page'], x['bbox'][1]))
    
    filtered_headings = []
    for heading in headings_sorted:
        text = heading['text'].strip()      
        if (re.fullmatch(r'[\.\-_ ]+', text) or  
            len(text) < 3 or
            re.match(r'^\.+$', text)):  
            continue
        
        if (re.match(r'^\d+$', text) and 
            heading['y_position'] > heading['page_height'] - 50):  
            continue
            
        filtered_headings.append(heading)
    
    return filtered_headings

def process_pdf(pdf_path):
    """Process a single PDF file and return the outline data"""
    scanned = is_scanned_pdf(pdf_path)
    
    if scanned:
        all_text_data = extract_text_with_ocr(pdf_path)
    else:
        all_text_data = extract_text_with_fitz(pdf_path)
    
    majority_attrs = find_majority_text_attributes(all_text_data)
    headings = extract_headings(all_text_data, majority_attrs)
    
    if len(headings) > 1:
        title = headings[0]['text']
        outline_headings = headings[1:]
    elif headings:
        title = headings[0]['text']
        outline_headings = []
    else:
        title = "Document"
        outline_headings = []
    
    outline_data = {
        "title": title,
        "outline": []
    }
    
    for heading in outline_headings:
        outline_data["outline"].append({
            "level": heading.get('level', 'H1'),
            "text": heading['text'],
            "page": heading['page']
        })
    
    return outline_data

def process_pdfs():
    """Process all PDFs in the input directory"""
    print("Starting processing PDFs")
    
    input_dir = Path(r"D:\harsh\Code_Playground\Adobe Internship\Challenge_1aFinal\Challenge_1a\input")
    output_dir = Path(r"D:\harsh\Code_Playground\Adobe Internship\Challenge_1aFinal\Challenge_1a\ouput")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing {pdf_file.name}...")
            outline_data = process_pdf(pdf_file)
            
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w") as f:
                json.dump(outline_data, f, indent=2)
            
            print(f"Processed {pdf_file.name} -> {output_file.name}")
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")
    
    print("Completed processing PDFs")

if __name__ == "__main__":
    process_pdfs()