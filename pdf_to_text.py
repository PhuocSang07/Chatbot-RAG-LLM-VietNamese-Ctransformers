from pypdf import PdfReader 

# creating a pdf reader object 
reader = PdfReader('./documents/thptdaosontay_437958_1dc-su-12-tu-nhien_5920212.pdf') 

# printing number of pages in pdf file 
print(len(reader.pages)) 

# getting a specific page from the pdf file 
page = reader.pages

# extracting text from page 
text = []
for p in page:
    text.append(p.extract_text()) 

# opening the file with utf-8 encoding
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write("".join(t for t in text))
