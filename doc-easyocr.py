import time
import base64
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from tqdm import tqdm
from io import BytesIO
from PIL import Image, ImageOps

from doctr.models._utils import estimate_orientation
from pdf2image import convert_from_path
#from docx2pdf import convert as docx_to_pdf


def invert_image(img):
    # For RGBA images, handle transparency
    if img.mode == 'RGBA':
        # Split into RGB and alpha channels
        rgb_img = img.convert('RGB')
        # Invert the RGB image
        inverted_rgb = ImageOps.invert(rgb_img)
        # Combine with original alpha channel
        inverted_img = inverted_rgb.convert('RGBA')
    else:
        # Directly invert for RGB images
        inverted_img = ImageOps.invert(img)

    return inverted_img

def convert_to_imgs(file):
    """
    Конвертирует файлы DOCX и PDF в изображения, сохраняет метаданные.
    """
    images = []

    path = pathlib.Path(file)
    
    if not path.exists():
        print(f"Файл не найден: {file}")

    # Convert PDF -> Images
    try:
        new_imgs = convert_from_path(file, dpi=300, thread_count=4)#, output_folder="./outpics")
        images.extend(new_imgs)
    except Exception as e:
        print(f"Ошибка обработки PDF {file}: {e}")

    return images

def prompt_easyocr_with_retry(reader, img_nparr, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        try:
            # Your operation here
            start = time.time()
           
            ocr = reader.readtext(img_nparr, detail = 0, paragraph = True)
            end = time.time()
            processing_time_s = end - start

            return ocr, processing_time_s

        except Exception as e:
            attempts += 1
            print(f"EeasyOCR model prompt {attempts} failed: {str(e)}")
            # Optional delay before retry
            time.sleep(1)

            if attempts == max_attempts:
                print(f"[REJECT] EeasyOCR model prompt {attempts} failed: {str(e)}")
                return None, None

def drop_to_rejects(pil_image, outfile, page):
    rej_fname = f"./rejects/{outfile}-{page}.png"
    pil_image.save(rej_fname, format="PNG")
    print(f"[REJECT] Saved file in: {rej_fname}")

def process_images(reader, outfile, images):
    responses = []
    total_doc_time = 0
    rejected_img_count = 0

    for idx, img in enumerate(images):
        buffered = BytesIO()
        stock_pil_img = img
        img_nparr = np.array(img)
        angle = estimate_orientation(img_nparr)

        if angle > 0: # document specific normalization
            angle = -angle

        img = img.rotate(angle, expand=True)
        print(f"angle: {angle}")
        #img = invert_image(img)
        img.save(buffered, format="PNG")
        img_nparr = np.array(img)
        #plt.imshow(img_array)
        #plt.axis('off')  # Hide axes
        #plt.show()

        ocr, processing_time_s = prompt_easyocr_with_retry(reader, img_nparr)
        if ocr == None:
            drop_to_rejects(stock_pil_img, outfile, page=idx)
            rejected_img_count += 1

        else:
            total_doc_time += processing_time_s
            responses.append({'page': idx, 'ocr': ocr, 'time': processing_time_s, 'image': stock_pil_img})
            print(f"{outfile}, page: {idx}, proc. time: {processing_time_s} s")

            with open(f"{outfile}-{idx}.txt", 'w+', encoding="utf-8") as f:
                f.write(' '.join(ocr))

    mean_time = total_doc_time / (len(images) - rejected_img_count)
    print(f"Mean page processing time for current document: {mean_time}")

def main():
    reader = easyocr.Reader(['ru']) # this needs to run only once to load the model into memory
    path = pathlib.Path("input")
    # List only files
    files = [item for item in path.iterdir() if item.is_file()]
    for infile in tqdm(files):
        images = convert_to_imgs(infile)
        outfile = "output/" + pathlib.Path(infile).stem
        process_images(reader, outfile, images)

main()