import os
import zipfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
from Extraction import (
    PDFImageExtractor,
    PDFTextExtractor,
    ImageTextExtractor,
    ImageProcessor,
)


app = FastAPI()


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content_type = file.content_type

    if content_type not in ["application/pdf", "image/jpeg", "image/png", "image.jpg"]:
        return JSONResponse(
            content={
                "message": "Unsupported file type. Only PDF or image files (JPEG, PNG) are allowed."
            },
            status_code=400,
        )

    input_dir = "input"
    output_dir = "output"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        file_path = os.path.join(input_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        if content_type == "application/pdf":
            # PDF Processing
            pdf_output_dir = os.path.join(output_dir, "pdf")
            os.makedirs(pdf_output_dir, exist_ok=True)

            pdf_extractor = PDFImageExtractor(file_path, pdf_output_dir)
            images_found = pdf_extractor.extract_images()

            text_output_path = os.path.join(pdf_output_dir, "extracted_text.txt")
            pdf_text_extractor = PDFTextExtractor(file_path, text_output_path)
            pdf_text_extractor.extract_text_to_file()

        else:
            image_output_dir = os.path.join(output_dir, "image")
            os.makedirs(image_output_dir, exist_ok=True)

            text_output_path = os.path.join(image_output_dir, "image_text.txt")
            img_text_extractor = ImageTextExtractor(file_path, text_output_path)
            ocr_result = img_text_extractor.extract_text_to_img()

            # Image to Table
            nutrition_table_path = os.path.join(input_dir, file.filename)
            image_processor = ImageProcessor(nutrition_table_path)
            image_processor.process()

        # Create a zip file of the outputs
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zipf:
            for foldername, subfolders, filenames in os.walk(output_dir):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
        memory_file.seek(0)

        zip_filename = "extracted_output.zip"
        return StreamingResponse(
            memory_file,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
        )

    except Exception as e:
        return JSONResponse(
            content={"message": f"Error during processing: {str(e)}"}, status_code=500
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        for folder in [output_dir, input_dir]:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    for f in files:
                        os.remove(os.path.join(root, f))
                    for d in dirs:
                        os.rmdir(os.path.join(root, d))
