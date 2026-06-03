from fpdf import FPDF
import markdown

pdf = FPDF()
pdf.add_page()
pdf.set_font('helvetica', size=10)

html = markdown.markdown('Hello\n\n* Item 1\n* Item 2')

# Replace tags to inject line-height
html_styled = html.replace('<p>', '<p style="line-height: 1.5;">')
html_styled = html_styled.replace('<li>', '<li style="line-height: 1.5;">')

print("Before:", pdf.get_y())
pdf.write_html(html)
print("After default:", pdf.get_y())

pdf.add_page()
pdf.write_html(html_styled)
print("After styled:", pdf.get_y())
