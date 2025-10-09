"""
This script provides utility functions for processing PDF documents in the context of a Java TA Knowledge-Base Chatbot. 
It includes functionalities to detect visual elements such as tables, images, and drawings within PDF pages, 
extract structured text from tables, and compress images for efficient processing. 
The `analyze_visual_content_with_gpt4v` function utilizes the GPT-4V model to analyze visual content and generate 
detailed descriptions relevant to Java programming. Additionally, the script optimizes the extraction of text and 
visual content from PDF pages, and intelligently chunks the content into manageable sizes for further processing.
"""

import streamlit as st
from openai import OpenAI
import os
import fitz          # PyMuPDF for PDF parsing
import base64        # Image encoding
from PIL import Image
import io

def detect_visual_elements(page):

    """
    Detects the presence of visual elements in a given page.
    This function analyzes a page to identify the presence of tables, images, 
    and vector drawings. It also calculates the text density to determine if 
    the page is likely to contain diagrams based on the amount of text relative 
    to the visual content.
    Parameters:
        page (object): The page object to analyze, which should provide methods 
                       to find tables, get images, get drawings, and retrieve 
                       text and its dimensions.
    Returns:
        tuple: A tuple containing:
            - has_visual_content (bool): True if the page contains visual elements, 
              False otherwise.
            - num_tables (int): The number of tables found on the page.
            - num_images (int): The number of images found on the page.
            - num_drawings (int): The number of vector drawings found on the page.
    """
    
    # Look for tables
    table_finder = page.find_tables()
    tables = list(table_finder)  # Convert TableFinder to a list
    
    # Look for images
    images = page.get_images()
    
    # Look for vector drawings
    drawings = page.get_drawings()
    # Heuristic: sparse text plus visuals likely indicates a diagram
    text = page.get_text().strip()
    text_density = len(text) / (page.rect.width * page.rect.height) if page.rect.width * page.rect.height > 0 else 0
    
    has_visual_content = len(tables) > 0 or len(images) > 0 or len(drawings) > 5 or text_density < 0.001
    
    return has_visual_content, len(tables), len(images), len(drawings)

def extract_table_text(page):

    """
    Extracts text from tables found on a given page and formats them as Markdown.
    Args:
        page (object): An object representing the page from which to extract tables.
    Returns:
        str: A string containing the Markdown representation of all extracted tables.
        Each table is prefixed with 'TABLE:' and separated by newlines.
    Notes:
        - The function handles exceptions silently, continuing to process other tables
          if an error occurs during extraction.
        - The first row of each table is treated as the header, and the subsequent rows
          are treated as data.
    """
    
    table_finder = page.find_tables()
    tables = list(table_finder)  # Convert TableFinder to a list
    table_texts = []
    
    for table in tables:
        try:
            table_data = table.extract()
            if table_data:
                # Render table as Markdown
                markdown_table = "| " + " | ".join(str(cell) if cell else "" for cell in table_data[0]) + " |\n"
                markdown_table += "|" + "---|" * len(table_data[0]) + "\n"
                
                for row in table_data[1:]:
                    markdown_table += "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |\n"
                
                table_texts.append(f"TABLE:\n{markdown_table}\n")
        except:
            continue
    
    return "\n".join(table_texts)

def compress_image(pix, max_size=(800, 600), quality=85):

    """
    Compresses an image to a specified maximum size and quality.
    Args:
        pix (PIL.Image.Image): The input image in pixel format.
        max_size (tuple): A tuple specifying the maximum width and height of the image (default is (800, 600)).
        quality (int): The quality of the output JPEG image (default is 85).
    Returns:
        str: A base64 encoded string of the compressed JPEG image.
    Notes:
        - The function resizes the image if it exceeds the specified maximum dimensions.
        - It converts images with transparency (RGBA, LA, P) to RGB before saving to JPEG.
        - The output image is optimized for size while maintaining the specified quality.
    """
    
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    
    # Resize if the image is large
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed, then save as JPEG for better compression
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=quality, optimize=True)
    return base64.b64encode(output.getvalue()).decode("utf-8")

def analyze_visual_content_with_gpt4v(base64_image, client):

    """
    Analyze visual content using GPT-4V and convert it to a text description.

    Args:
        base64_image (str): A base64 encoded string representing the image to be analyzed.
        client (object): An instance of the client used to interact with the GPT-4V API.

    Returns:
        str: A detailed description of the visual content, including UML diagrams, code snippets, 
             flowcharts, tables, mathematical formulas, and other relevant educational content 
             related to Java programming. In case of an error, returns an error message.

    Raises:
        Exception: Catches and returns any exceptions that occur during the API call.

    Notes:
        - Ensure that the base64_image is properly formatted and valid.
        - The client must be authenticated and configured to access the GPT-4V API.
        - The response may vary based on the content of the image and the model's capabilities.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this image from a Java programming course document. Describe any:
                                1. UML diagrams (class diagrams, sequence diagrams, etc.) - describe the classes,
                                relationships, methods, and fields
                                2. Code snippets or examples - transcribe the code exactly
                                3. Flowcharts or algorithmic diagrams - describe the logic flow
                                4. Tables with data - convert to markdown table format
                                5. Matematical formulas or expressions
                                6. Any other educational content relevant to Java programming

                                Be precise and detailed in your description so it can be used as context for answering
                                student questions."""
                                                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error analyzing visual content: {str(e)}]"

def extract_pages_optimized(pdf_path):

    """
    Extracts text and analyzes visual content from a PDF file efficiently.
    Args:
        pdf_path (str): The file path to the PDF document from which to extract content.
    Returns:
        list: A list of dictionaries, each containing:
            - page (int): The page number (1-indexed).
            - content (str): Combined text, structured tables, and visual elements description.
            - has_visuals (bool): Indicates if the page contains significant visual elements.
            - stats (str): A summary of the number of tables, images, and drawings on the page.
    Notes:
        - The function utilizes the PyMuPDF library (imported as fitz) to open and process the PDF.
        - OpenAI's API is used for analyzing visual content, requiring an API key stored in environment variables.
        - The function checks for significant visual elements and processes them only if certain conditions are met.
        - The output is structured to provide a comprehensive overview of each page's content and visual elements.
    """

    doc = fitz.open(pdf_path)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    page_data = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        
        # Extract tables as structured text
        table_text = extract_table_text(page)
        
        # Check for significant visual content
        has_visual, num_tables, num_images, num_drawings = detect_visual_elements(page)
        
        visual_description = ""
        if has_visual and (num_images > 0 or num_drawings > 10 or (len(text) < 100 and num_drawings > 0)):
            # Only process visuals for pages likely to contain diagrams
            pix = page.get_pixmap(dpi=150)  # Higher DPI improves OCR
            compressed_img = compress_image(pix)
            visual_description = analyze_visual_content_with_gpt4v(compressed_img, client)
            
            st.info(f"📊 Analyzed visual content on page {i+1}: {num_tables} tables, {num_images} images, {num_drawings} drawings")
        
        # Combine content
        combined_content = []
        if text:
            combined_content.append(f"TEXT CONTENT:\n{text}")
        if table_text:
            combined_content.append(f"STRUCTURED TABLES:\n{table_text}")
        if visual_description:
            combined_content.append(f"VISUAL ELEMENTS DESCRIPTION:\n{visual_description}")
        
        page_content = "\n\n".join(combined_content)
        page_data.append({
            "page": i + 1, 
            "content": page_content,
            "has_visuals": has_visual,
            "stats": f"Tables: {num_tables}, Images: {num_images}, Drawings: {num_drawings}"
        })
    
    return page_data

def smart_chunk_content(page_data, max_chunk_size=4000):
    
    """
    Splits the content of pages into smaller chunks based on a maximum chunk size.
    Args:
        page_data (list): A list of dictionaries, where each dictionary contains:
            - "content" (str): The text content of the page.
            - "page" (int): The page number.
        max_chunk_size (int): The maximum size of each chunk in characters. Default is 4000.
    Returns:
        list: A list of dictionaries, each containing:
            - "text" (str): The chunked text content.
            - "metadata" (dict): A dictionary with:
                - "page" (int): The page number from which the chunk was created.
                - "chunk" (int): The chunk number for the respective page.
    Notes:
        - The function ensures that chunks do not exceed the specified maximum size.
        - It attempts to keep visual elements together by splitting content at double newlines.
        - If a section exceeds the maximum size, it creates a new chunk for that section.
    """
    
    chunks = []
    
    for page in page_data:
        content = page["content"]
        page_num = page["page"]
        
        if len(content) <= max_chunk_size:
            chunks.append({
                "text": content,
                "metadata": {"page": page_num, "chunk": 1}
            })
        else:
            # Split by sections; keep visual elements together
            sections = content.split("\n\n")
            current_chunk = ""
            chunk_num = 1
            
            for section in sections:
                if len(current_chunk) + len(section) + 2 <= max_chunk_size:
                    current_chunk += section + "\n\n"
                else:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "metadata": {"page": page_num, "chunk": chunk_num}
                        })
                        chunk_num += 1
                    current_chunk = section + "\n\n"
            
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {"page": page_num, "chunk": chunk_num}
                })
    
    return chunks