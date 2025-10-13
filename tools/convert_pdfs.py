import os
from pdf2image import convert_from_path

# Folders
PDF_FOLDER = 'pdfs'
OUTPUT_FOLDER = 'all_images'

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# List all PDF files
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
print(f"Found {len(pdf_files)} PDF(s).")

if not pdf_files:
    print("No PDFs found. Exiting.")
    exit()

# Convert each PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    print(f"Converting: {pdf_file}")
    try:
        images = convert_from_path(pdf_path)
        for i, image in enumerate(images):
            image_name = f"{os.path.splitext(pdf_file)[0]}_page_{i+1}.png"
            image_path = os.path.join(OUTPUT_FOLDER, image_name)
            image.save(image_path, 'PNG')
        print(f"Saved {len(images)} image(s) from {pdf_file}")
    except Exception as e:
        print(f"Error converting {pdf_file}: {e}")
