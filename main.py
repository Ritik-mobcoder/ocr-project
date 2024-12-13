import os 
from Extraction import PDFImageExtractor , PDFTextExtractor , ImageTextExtractor , ImageProcessor

if __name__ == "__main__":
    pdf_path = "input/sample.pdf"  
    img_path = "input/page_01.jpg"
    output = "output/extracted_images" 

    if not os.path.exists(output):
        os.makedirs(output)
    #PDF TO Image
    extractor = PDFImageExtractor(pdf_path, output)
    images_found = extractor.extract_images()

    print("Images found on pages:", images_found)
    # PDF TO Text
    text_output_path = "output/extracted_text.txt"  
    extractor = PDFTextExtractor(pdf_path, text_output_path)
    extractor.extract_text_to_file()
    
    #Image To Text
    text_output_path = 'output/image_text.txt'
    extractor = ImageTextExtractor(img_path, text_output_path)
    ocr_result = extractor.extract_text_to_img()
    
    #Imaage to Table
    image_processor = ImageProcessor("input/nutrition_table.jpg")
    image_processor.process()
    print("Image To Table Created")


