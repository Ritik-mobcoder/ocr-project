import fitz
import os
from pypdf import PdfReader
import cv2
import numpy as np
import pytesseract
import subprocess


class PDFImageExtractor:
    def __init__(self, pdf_path, images_output_folder):
        self.pdf_path = pdf_path
        self.images_output_folder = images_output_folder
        self.image_exists = []

    def extract_images(self):
        pdf_file = fitz.open(self.pdf_path)

        for page_index in range(len(pdf_file)):
            page = pdf_file.load_page(page_index)
            image_list = page.get_images(full=True)

            if image_list:
                self.image_exists.append("Yes")
                print(
                    f"[+] Found a total of {len(image_list)} images on page {page_index}"
                )
            else:
                self.image_exists.append("No")
                print("[!] No images found on page", page_index)

            for image_index, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_name = f"image{page_index + 1}_{image_index}.{image_ext}"
                image_path = os.path.join(self.images_output_folder, image_name)

                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
                    print(f"[+] Image saved as {image_name}")

        pdf_file.close()
        return self.image_exists


class PDFTextExtractor:
    def __init__(self, pdf_path, text_output_path):
        self.pdf_path = pdf_path
        self.text_output_path = text_output_path

    def extract_text_to_file(self):

        try:
            reader = PdfReader(self.pdf_path)
            with open(self.text_output_path, "w", encoding="utf-8") as text_file:
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    text_file.write(f"Page {page_num + 1}\n{'=' * 40}\n{text}\n\n")
            print(f"Text successfully extracted to {self.text_output_path}")
        except Exception as e:
            print(f"An error occurred: {e}")


class ImageTextExtractor:
    def __init__(self, img_path, text_output_path):
        self.img_path = img_path
        self.text_output_path = text_output_path

    def noise_removal(self, image):
        kernal = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernal, iterations=1)
        kernal = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernal, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernal)
        image = cv2.medianBlur(image, 3)
        return image

    def extract_text_to_img(self):
        img = cv2.imread(self.img_path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get binary image
        thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)

        # Remove noise from the binary image
        no_noise = self.noise_removal(im_bw)

        # Extract text using pytesseract
        orc_result = pytesseract.image_to_string(no_noise)

        with open(self.text_output_path, "w") as text_file:
            text_file.write(orc_result)

        return orc_result


class ImageProcessor:
    def __init__(self, image_file):
        self.image_file = image_file
        self.img = cv2.imread(image_file)
        self.grayscale_image = None
        self.thresholded_image = None
        self.inverted_image = None
        self.dilated_image = None
        self.rectangular_contours = []
        self.contour_with_max_area = None
        self.perspective_corrected_image = None
        self.perspective_corrected_image_with_padding = None
        self.bounding_boxes = []
        self.table = []

    def preprocess_image(self):
        self.grayscale_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.thresholded_image = cv2.threshold(
            self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def find_contours(self):
        self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=5)
        contours, _ = cv2.findContours(
            self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                self.rectangular_contours.append(approx)
        self.find_largest_contour()

    def find_largest_contour(self):
        max_area = 0
        for contour in self.rectangular_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                self.contour_with_max_area = contour

    def order_points(self, pts):
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def calculate_distance(p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    def apply_perspective_transform(self):
        if self.contour_with_max_area is not None:
            contour_with_max_area_ordered = self.order_points(
                self.contour_with_max_area
            )
            existing_image_width = self.img.shape[1]
            existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)
            distance_top_left_top_right = self.calculate_distance(
                contour_with_max_area_ordered[0], contour_with_max_area_ordered[1]
            )
            distance_top_left_bottom_left = self.calculate_distance(
                contour_with_max_area_ordered[0], contour_with_max_area_ordered[3]
            )
            aspect_ratio = distance_top_left_bottom_left / distance_top_left_top_right
            new_image_width = existing_image_width_reduced_by_10_percent
            new_image_height = int(new_image_width * aspect_ratio)

            pts1 = np.float32(contour_with_max_area_ordered)
            pts2 = np.float32(
                [
                    [0, 0],
                    [new_image_width, 0],
                    [new_image_width, new_image_height],
                    [0, new_image_height],
                ]
            )
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.perspective_corrected_image = cv2.warpPerspective(
                self.img, matrix, (new_image_width, new_image_height)
            )
            self.add_padding()

    def add_padding(self):
        image_height = self.img.shape[0]
        padding = int(image_height * 0.1)
        self.perspective_corrected_image_with_padding = cv2.copyMakeBorder(
            self.perspective_corrected_image,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

    # step B
    def extract_table_structure(self):
        img1 = self.perspective_corrected_image_with_padding
        grayscale_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[
            1
        ]
        inverted_image = cv2.bitwise_not(thresholded_image)

        hor = np.array([[1, 1, 1, 1, 1, 1]])
        ver = np.array([[1], [1], [1], [1], [1], [1], [1]])

        vertical_lines_eroded_image = cv2.erode(inverted_image, hor, iterations=10)
        vertical_lines_eroded_image = cv2.dilate(
            vertical_lines_eroded_image, hor, iterations=10
        )
        horizontal_lines_eroded_image = cv2.erode(inverted_image, ver, iterations=10)
        horizontal_lines_eroded_image = cv2.dilate(
            horizontal_lines_eroded_image, ver, iterations=10
        )
        cv2.imwrite(
            "table_img/horizontal_lines_eroded_image.jpg", horizontal_lines_eroded_image
        )

        combined_image = cv2.add(
            vertical_lines_eroded_image, horizontal_lines_eroded_image
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        combined_image_dilated = cv2.dilate(combined_image, kernel, iterations=5)
        cv2.imwrite("table_img/combined_image_dilated.jpg", combined_image_dilated)

        image_without_lines = cv2.subtract(inverted_image, combined_image_dilated)
        cv2.imwrite("table_img/image_without_lines.jpg", image_without_lines)
        image_without_lines_noise_removed = cv2.erode(
            image_without_lines, kernel, iterations=1
        )
        image_without_lines_noise_removed = cv2.dilate(
            image_without_lines_noise_removed, kernel, iterations=1
        )
        cv2.imwrite(
            "table_img/image_without_lines_noise_removed.jpg",
            image_without_lines_noise_removed,
        )

        # section C
        kernel_to_remove_gaps_between_words = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        )
        dilated_image = cv2.dilate(
            image_without_lines_noise_removed,
            kernel_to_remove_gaps_between_words,
            iterations=5,
        )
        simple_kernel = np.ones((5, 5), np.uint8)
        dilated_image = cv2.dilate(dilated_image, simple_kernel, iterations=2)
        cv2.imwrite("table_img/dilated_image.jpg", dilated_image)
        self.bounding_boxes = self.find_bounding_boxes(dilated_image)

    def find_bounding_boxes(self, dilated_image):
        contours, _ = cv2.findContours(
            dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
        return sorted(bounding_boxes, key=lambda x: x[1])

    def extract_text(self):
        self.table = []
        img1 = self.perspective_corrected_image_with_padding
        os.makedirs("output", exist_ok=True)
        current_row = []
        image_number = 0
        heights = [h for _, _, _, h in self.bounding_boxes]
        mean_height = np.mean(heights)
        half_of_mean_height = mean_height / 2

        rows = []
        current_row = [self.bounding_boxes[0]]
        for bounding_box in self.bounding_boxes[1:]:
            if abs(bounding_box[1] - current_row[-1][1]) <= half_of_mean_height:
                current_row.append(bounding_box)
            else:
                rows.append(current_row)
                current_row = [bounding_box]
        rows.append(current_row)

        for row in rows:
            row.sort(key=lambda x: x[0])
            current_row = []
            for bounding_box in row:
                x, y, w, h = bounding_box
                y = max(y - 5, 0)
                cropped_image = img1[y : y + h, x : x + w]
                image_slice_path = f"table_img/{image_number}.jpg"
                cv2.imwrite(image_slice_path, cropped_image)
                text = self.get_text_from_tesseract(image_slice_path)
                current_row.append(text)
                image_number += 1
            self.table.append(current_row)

    @staticmethod
    def get_text_from_tesseract(image_path):
        output = subprocess.getoutput(
            f"tesseract {image_path} - -l eng --oem 3 --psm 7 --dpi 72 "
            '-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789().calmg* "'
        )
        return output.strip()

    def generate_csv_file(self, output_path="output/image_table.csv"):
        with open(output_path, "w") as f:
            for row in self.table:
                f.write(",".join(row) + "\n")

    def process(self):
        self.preprocess_image()
        self.find_contours()
        self.apply_perspective_transform()
        self.extract_table_structure()
        self.extract_text()
        self.generate_csv_file()


# image_processor = ImageProcessor("input/nutrition_table.jpg")
# image_processor.process()
# result_image = image_processor.perspective_corrected_image_with_padding
# cv2.imwrite("input/result_image.jpg", result_image)
# print("Shdh")
# image_processor = ImageProcessor("input/nutrition_table.jpg")
# image_processor.process()
