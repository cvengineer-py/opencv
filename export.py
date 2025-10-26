from fpdf import FPDF
import os

folder = r"D:\clg\Lab - CV"
output_folder = r"D:\clg\Lab - CV\pdfs"
os.makedirs(output_folder, exist_ok=True)

FONT_PATH = r"C:\Users\91960\OneDrive\Desktop\dejavu-fonts-ttf-2.37\ttf\DejaVuSansMono.ttf"

for filename in os.listdir(folder):
    if filename.endswith(".py"):
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        pdf = FPDF(orientation='L', unit='mm', format='A4')  # Landscape
        pdf.add_page()
        pdf.add_font("DejaVu", "", FONT_PATH)
        pdf.set_font("DejaVu", "", 8)  # smaller font

        page_width = pdf.w - 2 * pdf.l_margin
        for line in code.splitlines():
            # Break line manually if too long
            while len(line) * 2.5 > page_width:  # approximate width per char
                pdf.multi_cell(0, 5, line[:int(page_width/2.5)])
                line = line[int(page_width/2.5):]
            pdf.multi_cell(0, 5, line)

        pdf.output(os.path.join(output_folder, filename.replace(".py", ".pdf")))

print("All .py files converted to PDF!")
