# %pip install PyPDF2
# %pip install PyCryptodome 
import os
from PyPDF2 import PdfReader
import shutil
import logging
LOG = logging.getLogger(__name__)
# Set the logging level to DEBUG
LOG.setLevel(logging.DEBUG)

def count_pdf_pages(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        LOG.debug(f"Error reading {pdf_path}: {e}")
        return 0

def organize_pdfs(folder_path, max_pages_per_folder=1000):
    # Creating a new directory to store sub-folders
    new_folders_path = 'New_Folder'

    os.makedirs(new_folders_path, exist_ok=True)

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    current_folder_count = 0
    current_page_count = 0
    total_pages = 0
    

    for pdf_file in pdf_files:
        current_folder_path = os.path.join(new_folders_path, f'{current_folder_count}')
        os.makedirs(current_folder_path, exist_ok=True)
        
        pdf_path = os.path.join(folder_path, pdf_file)
        num_pages = count_pdf_pages(pdf_path)
        if current_page_count + num_pages > max_pages_per_folder:
            current_folder_count += 1
            #current_folder_path = os.path.join(new_folders_path, f'{current_folder_count}')
            #os.makedirs(current_folder_path, exist_ok=True)
            total_pages += current_page_count
            current_page_count = 0
        
        shutil.copy(pdf_path, os.path.join(current_folder_path, pdf_file))
        current_page_count += num_pages
    
    LOG.debug(f"Organized {len(pdf_files)} PDFs into {current_folder_count} folders inside 'New_Folders'.")
    LOG.debug("total_pages:",total_pages)

folder_path = 'Database_NEW'
organize_pdfs(folder_path)