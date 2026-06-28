import pypdf

pdf_path = r'c:\Users\dani9\.gemini\antigravity\scratch\blockchain\Etherium_transaction_paper.pdf'
output_path = r'c:\Users\dani9\.gemini\antigravity\scratch\blockchain\pdf_content.txt'

reader = pypdf.PdfReader(pdf_path)
print(f"Number of pages: {len(reader.pages)}")

out_lines = []
for i, page in enumerate(reader.pages):
    out_lines.append(f"=== PAGE {i+1} ===")
    out_lines.append(page.extract_text() or "")
    out_lines.append("")

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(out_lines))

print(f"Extracted PDF to {output_path}")
