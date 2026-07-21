from docx import Document
from pathlib import Path

doc = Document(
    Path(r"c:\Users\user\Desktop\code project masters\extras\THESIS_OBANOR_TEHILLA_49_.docx")
)
out = Path(r"c:\Users\user\Desktop\code project masters\scripts\thesis_paragraph_map.txt")
lines = []
for i, para in enumerate(doc.paragraphs):
    t = para.text.strip()
    if t:
        lines.append(f"{i}\t{t}")
out.write_text("\n".join(lines), encoding="utf-8")
print("written", out, "paragraphs", len(lines))
