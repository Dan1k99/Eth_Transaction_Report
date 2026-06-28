import zipfile
import xml.etree.ElementTree as ET

pptx_path = r'c:\Users\dani9\Downloads\The_Illuminated_Ledger.pptx'
with zipfile.ZipFile(pptx_path, 'r') as zf:
    for i in range(1, 17):
        slide_name = f'ppt/slides/slide{i}.xml'
        if slide_name in zf.namelist():
            xml_content = zf.read(slide_name)
            root = ET.fromstring(xml_content)
            # Find all text-like elements
            texts = []
            for elem in root.iter():
                # Let's check text content of any element
                if elem.text and elem.text.strip():
                    texts.append(f"{elem.tag.split('}')[-1]}: {elem.text.strip()}")
            if texts:
                print(f"--- Slide {i} ---")
                print("\n".join(texts[:10])) # show first 10 text elements
                print(f"Total text elements: {len(texts)}")
