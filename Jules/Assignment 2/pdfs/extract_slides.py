import fitz  # PyMuPDF
import os

def extract_slides_from_pdf(pdf_path, output_folder="extracted_slides", slides_to_extract=None):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    total_slides = len(pdf_document)

    # Determine which pages to process
    if slides_to_extract is None:
        # If no specific list is provided, extract everything
        pages_to_process = range(total_slides)
        print(f"Successfully opened {pdf_path}. Extracting all {total_slides} slides...")
    else:
        # Convert user's 1-based slide numbers to Python's 0-based page index
        pages_to_process = []
        for slide_num in slides_to_extract:
            if 1 <= slide_num <= total_slides:
                pages_to_process.append(slide_num - 1)
            else:
                print(f"Warning: Slide {slide_num} does not exist in this {total_slides}-page PDF. Skipping.")
        
        print(f"Successfully opened {pdf_path}. Extracting {len(pages_to_process)} specific slide(s)...")

    # Iterate through the selected pages
    for page_number in pages_to_process:
        page = pdf_document.load_page(page_number)

        # Set the resolution (zoom factor). 
        zoom_x = 2.0
        zoom_y = 2.0
        matrix = fitz.Matrix(zoom_x, zoom_y)

        # Render the page to an image (pixmap)
        pix = page.get_pixmap(matrix=matrix)

        # Format the filename with leading zeros based on the actual slide number
        slide_label = page_number + 1
        filename = f"slide_{slide_label:03d}.png"
        output_path = os.path.join(output_folder, filename)

        # Save the image
        pix.save(output_path)
        print(f"Saved: {filename}")

    pdf_document.close()
    print("Extraction complete!")

# --- How to use ---
if __name__ == "__main__":
    pdf_file_path = r"Assignment 2\pdfs\3.2 - The lifting line model.pdf" 
    
    # Put the exact slide numbers you want in this list (e.g., slides 1, 4, 5, and 12)
    my_slides = [3, 9, 10, 11, 12, 16, 17, 19, 21, 23, 25]
    
    # Run the function with the specific list
    extract_slides_from_pdf(
        pdf_path=pdf_file_path, 
        output_folder="pdfs/slides", 
        slides_to_extract=my_slides
    )
    
    # NOTE: If you ever want to go back to extracting ALL slides, 
    # just remove the `slides_to_extract` part like this:
    # extract_slides_from_pdf(pdf_file_path, output_folder="all_slides")