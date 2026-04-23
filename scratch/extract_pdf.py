import os
import sys

try:
    import pypdf
except ImportError:
    os.system(f"{sys.executable} -m pip install pypdf")
    import pypdf

folder = r"C:\Users\PHN-MasterClass\Downloads\agent_data_pdf"
out_file = r"c:\Projects AI_ML\LLM\whatsapp-agent\pdf_text.txt"

with open(out_file, "w", encoding="utf-8") as out:
    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder, filename)
            out.write(f"\n{'='*40}\n")
            out.write(f"FILE: {filename}\n")
            out.write(f"{'='*40}\n\n")
            
            try:
                with open(filepath, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            out.write(text + "\n")
            except Exception as e:
                out.write(f"ERROR READING PDF: {e}\n")

print("Done extracting text.")
