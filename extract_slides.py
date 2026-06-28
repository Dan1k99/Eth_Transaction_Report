import zipfile
import re
import xml.etree.ElementTree as ET

pptx_path = r'c:\Users\dani9\Downloads\The_Illuminated_Ledger.pptx'
output_path = r'c:\Users\dani9\.gemini\antigravity\scratch\blockchain\slides_content.txt'

with zipfile.ZipFile(pptx_path, 'r') as zf:
    slides = [name for name in zf.namelist() if name.startswith('ppt/slides/slide') and name.endswith('.xml')]
    slides.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    out_lines = []
    for slide in slides:
        out_lines.append(f"=== {slide} ===")
        xml_content = zf.read(slide)
        root = ET.fromstring(xml_content)
        
        texts = []
        for elem in root.iter():
            if elem.tag.endswith('}t'):
                if elem.text:
                    texts.append(elem.text)
        slide_text = " ".join(texts)
        out_lines.append(slide_text)
        out_lines.append("")
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(out_lines))

print(f"Extracted {len(slides)} slides to {output_path}")
