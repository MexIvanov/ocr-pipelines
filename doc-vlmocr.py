import time
import base64
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from ollama import Client
from tqdm import tqdm
from io import BytesIO
from PIL import Image, ImageOps
from doctr.models._utils import estimate_orientation
from pdf2image import convert_from_path


MODEL = "benhaotang/Nanonets-OCR-s:F16"
client = Client(host="localhost:11434")

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
    """Convert PDF to images
    """

    images = []
    path = Path(file)
    
    if not path.exists():
        print(f"File not found: {file}")

    try:
        new_imgs = convert_from_path(file, dpi=300, thread_count=4) #, output_folder="./outpics")
        images.extend(new_imgs)
    except Exception as e:
        print(f"Error converting PDF {file}: {e}")

    return images

def prompt_ollama_with_retry(img_b64, client, options, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        try:
            start = time.time()

            response = client.chat(model=MODEL, messages=[
                {
                    'role': "system",
                    'content': "You are a helpful assistant."
                },
                {
                    'role': "user",
                    'images': [img_b64],
                    'content': "Extract the text from the above document as if you were reading it naturally. Use only cyrillic russian language, except for words clearly written in english. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes. Do not repeat sentences multiple times.",
                },
            ],
            options=options) #, 'temperature': 0.0, 'repeat_penalty': 1.05 })
            ocr = response['message']['content']
            end = time.time()
            processing_time_s = end - start

            return ocr, processing_time_s

        except Exception as e:
            attempts += 1
            print(f"Ollama prompt {attempts} failed: {str(e)}")
            # Optional delay before retry
            time.sleep(1)

            if attempts == max_attempts:
                print(f"[REJECT] Ollama prompt {attempts} failed: {str(e)}")
                return None, None

def drop_to_rejects(pil_image, outfile, page):
    rej_fname = f"./rejects/{outfile}-{page}.png"
    pil_image.save(rej_fname, format="PNG")
    print(f"[REJECT] Saved file in: {rej_fname}")

def write_file(fname, text):
    with open(fname, 'w+', encoding="utf-8") as f:
        f.write(text)

def get_non_processed_pdfs(input_folder="./input", output_folder="./output"):
    """
    Process filenames from input and output folders.
    
    Args:
        input_folder (str): Name of folder containing PDF files
        output_folder (str): Name of folder containing paginated text files
    
    Returns:
        set: Set of unique filenames from input folder - output 
    """
    # Get list of PDF filenames from input folder
    input_path = Path(input_folder)
    pdf_files = set(item.stem for item in input_path.iterdir() 
                if item.is_file() and item.suffix.lower() == '.pdf')
    
    # Get base filenames from output folder
    output_path = Path(output_folder)
    output_files = set(item.stem.split('-')[0] for item in output_path.iterdir()
                   if item.is_file() and item.suffix.lower() == '.txt'
                   and '-' in item.stem)
    
    # Return difference between sets
    diff = pdf_files - output_files
    np_pdf_files = {f'{name}.pdf' for name in diff}
    return np_pdf_files

def process_images(client, outfile, images):
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
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        #plt.imshow(img_array)
        #plt.axis('off')  # Hide axes
        #plt.show()

        ocr, processing_time_s = prompt_ollama_with_retry(img_b64, client, {'num_predict': 4096})
        if ocr == None:
            drop_to_rejects(stock_pil_img, outfile, page=idx)
            rejected_img_count += 1

        else:
            total_doc_time += processing_time_s
            responses.append({'page': idx, 'ocr': ocr, 'time': processing_time_s, 'image': stock_pil_img})
            print(f"{outfile}, page: {idx}, proc. time: {processing_time_s} s")
            write_file(f"./output/{outfile}-{idx}.txt", ocr)

    mean_time = total_doc_time / (len(images) - rejected_img_count)
    print(f"Mean page processing time for current document: {mean_time}")

    for page in responses:
        buffered = BytesIO()
        page['image'].save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        retries = 0
        while page['time'] > mean_time + 15.0 and retries < 3: #5.0:
            ocr, ptime = process_single_image(client, outfile, img_b64, page["page"])
            page['time'] = ptime
            page['ocr'] = ocr

            retries += 1
            print(retries)
            
       
        if page['time'] > mean_time + 15.0 or page['ocr'] == None:
            drop_to_rejects(page['image'], outfile, page=page['page'])
            print(f"[REJECT] Giving up ocr after {retries} retries")
        
        elif retries > 0:
            write_file(f"output/{outfile}-{page['page']}.txt", page['ocr'])

def process_single_image(client, outfile, img_b64, page):
    ocr, processing_time_s = prompt_ollama_with_retry(img_b64, client, {"num_predict": 4096, "temperature": 0.7, "repeat_penalty": 0.7 })
    print(f"{outfile}, page: {page}, proc. time: {processing_time_s} s")
    
    return ocr, processing_time_s

def main():
    files = get_non_processed_pdfs()
    for infile in tqdm(files):
        images = convert_to_imgs(infile)
        outfile = Path(infile).stem
        process_images(client, outfile, images)

main()