#!pip install python-dotenv
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
from trulens_eval import Feedback, TruLlama, OpenAI
from trulens_eval.feedback import Groundedness
import nest_asyncio
from IPython.display import Markdown, display
import json

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")

def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")

def append_to_json(query, response, filename):
    # Load existing data from the file or create an empty list if the file doesn't exist
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = []

    # Append new entry to the data list
    entry = {"query": query, "response": response}
    data.append(entry)

    # Write the updated data to the JSON file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)


def append_to_json_with_context(query, response, context_nodes, filename):
    # Load existing data from the file or create an empty list if the file doesn't exist
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = []

    # Append new entry to the data list
    entry = {
        "query": query,
        "response": response,
        "num_nodes": len(context_nodes),
        "node_content": {}
    }

    # Add content for each context node
    for i, context_node_content in enumerate(context_nodes, start=1):
        context_node_key = f"content_{i}"
        entry["node_content"][context_node_key] = context_node_content.text

    data.append(entry)

    # Write the updated data to the JSON file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)


# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    #num_tokens = len(encoding.encode(string, disallowed_special = set()))

    return num_tokens

def semantic_chunker(threshold_amount):
    """Creates an instance of a SemanticChunker. Input threshold value for breakpoint splitting"""
    text_splitter = SemanticChunker(
        OpenAIEmbeddings(model = "text-embedding-3-small"),
        breakpoint_threshold_type="percentile",
        #breakpoint_threshold_type="standard_deviation"
        #breakpoint_threshold_type="interquartile"
        breakpoint_threshold_amount = threshold_amount
    )
    return text_splitter

def print_all_chunks(list_of_docs,with_page_content=False, with_num_tokens=True):

    for i in range(len(list_of_docs)):
        num_of_tokens = num_tokens_from_string(list_of_docs[i].page_content, "cl100k_base")
        if with_num_tokens == True:
            print(f"num. of tokens in chunk {i} is: {num_of_tokens}")
        if with_page_content == True:
            #print(f"-----------------------------CHUNK {i} ----------------------------------")
            #print(list_of_docs[i].page_content)
            print(f"---")
            print(list_of_docs[i].page_content)

def find_idxs_below_min_chunk_size(list_of_docs, min_chunk_size=20, show_indices=True):

    idxs_below_min_chunk_size = []

    for i in range(len(list_of_docs)):
        num_of_tokens = num_tokens_from_string(list_of_docs[i].page_content, "cl100k_base")

        # if chunk size is below min_chunk_size, append index to list
        if num_of_tokens < min_chunk_size:
            idxs_below_min_chunk_size.append(i)

    num_of_idxs_in_list = len(idxs_below_min_chunk_size)

    if show_indices==True:
        print(f"{num_of_idxs_in_list} indices below min threshold: {idxs_below_min_chunk_size}")

    return idxs_below_min_chunk_size

def find_idxs_above_max_chunk_size(list_of_docs, max_chunk_size=512):

    idxs_above_max_chunk_size = []

    for i in range(len(list_of_docs)):
        num_of_tokens = num_tokens_from_string(list_of_docs[i].page_content, "cl100k_base")

        # if chunk size is below min_chunk_size, append index to list
        if num_of_tokens > max_chunk_size:
            idxs_above_max_chunk_size.append(i)

    num_of_idxs_in_list = len(idxs_above_max_chunk_size)

    print(f"{num_of_idxs_in_list} indices above max threshold: {idxs_above_max_chunk_size}")

    return idxs_above_max_chunk_size

def remove_below_min_chunks_list(list_of_docs, list_blw_min_idx, min_chunk_size=20):

    # inverse list order to not get indexing problems
    sorted_max_indices = sorted(list_blw_min_idx, reverse=True)
    #print(sorted_max_indices)

    for m in sorted_max_indices:

        num_of_tokens = num_tokens_from_string(list_of_docs[m].page_content, "cl100k_base")

        if num_of_tokens < min_chunk_size and m != 0:
            update_data = {
                "page_content": list_of_docs[m-1].page_content + " \n" + list_of_docs[m].page_content,
            }
            new_doc_node = list_of_docs[m-1].copy(update = update_data)

            # delete the two old nodes and subsitute it with the new node
            list_of_docs.pop(m)
            list_of_docs.pop(m-1)

            #insert new merged node into list
            list_of_docs.insert(m-1,new_doc_node)

def list_of_str_2_list_of_docs(list_of_str, doc_obj):

    list_of_docs = []

    for n in range(len(list_of_str)):

        data = {
            "page_content": list_of_str[n]
        }
        new_doc_obj = doc_obj.copy(update = data)
        list_of_docs.append(new_doc_obj)

    #print(list_of_docs)
    return list_of_docs

def remove_above_max_chunks_list(list_of_docs, list_abv_max_idx, breakpoint_thresh_value=95):

    # inverse list order to not get indexing problems
    sorted_max_indices = sorted(list_abv_max_idx, reverse=True)
    #print(sorted_max_indices)

    for m in tqdm(sorted_max_indices):

            # semantically split chunk
            subchunks = semantic_chunker(breakpoint_thresh_value).split_text(list_of_docs[m].page_content)

            # right now subchunks is a list of str (i.e. each element only contains text).
            # convert this list of strings into a list of docs
            node = list_of_docs[m].copy() # create copy of parent node (needed for list_of_str_2_list_of_docs fct.)
            docs = list_of_str_2_list_of_docs(subchunks, node)

            # Check, that the created subchunks are of token size > min_chunk_size (=20) --> if not, merge them with previous subchunk
            subchunk_idxs_below_min_thresh = find_idxs_below_min_chunk_size(docs, min_chunk_size=100, show_indices=False)
            remove_below_min_chunks_list(docs, subchunk_idxs_below_min_thresh, min_chunk_size=100)

            num_of_subchunks=len(docs)

            #create copy of the node
            copy_node_2 = list_of_docs[m].copy()
            #print(copy_node)

            # delete node from list
            list_of_docs.pop(m)

            for n in reversed(range(num_of_subchunks)):

                update_data = {
                    "page_content": docs[n].page_content
                }
                copy_node_2_updated = copy_node_2.copy(update = update_data)

                #insert new merged node into list
                list_of_docs.insert(m, copy_node_2_updated)

def write_results_in_txt(list_of_docs,leitlinien_doc,list_of_above_max_indices, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"-----------------------------RESULTS ----------------------------------\n")
        file.write(f"Leitlinien doc: {leitlinien_doc} \n")
        file.write(f"final num. of chunks: {len(list_of_docs)}\n")
        file.write(f"chunks over max. token limit: {list_of_above_max_indices}\n")

        for i in range(len(list_of_docs)):
            num_of_tokens = num_tokens_from_string(list_of_docs[i].page_content, "cl100k_base")
            file.write(f"num. of tokens in chunk {i} is: {num_of_tokens} \n")
            file.write(f"-----------------------------CHUNK {i} ----------------------------------\n")
            file.write(list_of_docs[i].page_content + '\n')

def add_last_page_delimiter(markdown_text):
    '''
    Add a --- delimiter for the last page
    '''
    # Ensure the markdown text ends with a newline before adding the delimiter
    if not markdown_text.endswith('\n'):
        markdown_text += '\n'

    # Add the --- delimiter at the end
    updated_markdown_text = markdown_text + '---\n'

    return updated_markdown_text

def count_and_update_delimiters(markdown_text):
    # Compile regex pattern to match "---" delimiters on separate lines
    delimiter_pattern = re.compile(r'^\s*---\s*$', re.MULTILINE)

    # Initialize count
    delimiter_count = 0

    # Function to replace each "---" delimiter with "--- {count} delimiter(s) found"
    def replace(match):
        nonlocal delimiter_count
        delimiter_count += 1
        return f'\n(Page {delimiter_count})\n---\n'

    # Replace each "---" delimiter
    updated_markdown_text = re.sub(delimiter_pattern, replace, markdown_text)

    return updated_markdown_text

def modify_headers_with_page_numbers(markdown_text):
    # Compile regex pattern to match headers (###, ##, #) and "page_nr: x"
    header_pattern = re.compile(r'^(#+)\s+(.*)$', re.MULTILINE)
    page_number_pattern = re.compile(r'(Page \s*(\d+))', re.IGNORECASE)

    # Function to replace each header with header + page number
    def replace(match):
        header_level = match.group(1)
        header_text = match.group(2)
        page_nr_match = page_number_pattern.search(markdown_text, match.end())
        if page_nr_match:
            page_nr = page_nr_match.group(1)
            return f'{header_level} {header_text} ({page_nr})'
        else:
            return match.group(0)

    # Replace each header
    updated_markdown_text = re.sub(header_pattern, replace, markdown_text)

    return updated_markdown_text

def update_metadata_with_page_numbers(md_header_splits):
    '''
    This function:
    1. reads the page numbers, previously inserted into the headers, and writes them into a seperate metadata entry: Page Number
    2. deletes all the page numbers inserted previously into the header metadata entries
    '''
    # Compile regex pattern to extract page numbers from headers
    page_number_pattern = re.compile(r'\(Page\s+(\d+)\)')

    for header_split in md_header_splits:
        metadata = header_split.metadata
        page_content = header_split.page_content
        page_numbers = []

        if metadata:
            last_header_value = list(metadata.values())[-1]
            page_number_match = page_number_pattern.search(last_header_value)
            if page_number_match:
                page_number = int(page_number_match.group(1))
                page_numbers.append(page_number)
                metadata['Page Number'] = page_numbers
            for key, value in metadata.items():
                if re.search(r'\bHeader\b', key):
                  # Remove the page number and surrounding brackets from the header
                  value_without_page_number = re.sub(page_number_pattern, '', value).strip()

                  # Update the metadata dictionary
                  metadata[key] = value_without_page_number
                if key == 'Page Number':
                    page_nums = set(int(match.group(1)) for match in page_number_pattern.finditer(page_content))

                    # If page_numbers is None, set it to [1]
                    if page_numbers is None:
                        page_numbers = [1]
                    for num in page_nums:
                        if num not in page_numbers:
                            page_numbers.append(num)
                            metadata['Page Number'] = page_numbers

def update_page_content(md_header_splits):
    '''
    This function deletes all the page numbers (i.e. (Page x)) inserted previously into the page content
    '''
    # Compile regex pattern to match (Page x)
    page_number_pattern = re.compile(r'\(Page\s+\d+\)')

    for header_split in md_header_splits:
        page_content = header_split.page_content
        # Remove (Page x) from the page content
        updated_page_content = re.sub(page_number_pattern, '', page_content)
        # Update the page_content in md_header_splits
        header_split.page_content = updated_page_content

def update_metadata_with_source(md_header_splits,guidline_metadata):
    '''
    This function includes the document source into the final chunks metadata
    '''
    for header_split in md_header_splits:
        guideline_name = guidline_metadata['Guideline_Name']

        if guideline_name.endswith('.pdf'):
          guideline_name = guideline_name[:-4]

        header_split.metadata['source'] = guideline_name

def update_metadata_with_validity(md_header_splits, guidline_metadata):
    '''
    This function includes the document source into the final chunks metadata
    '''
    for header_split in md_header_splits:
        # split.metadata['Page Number'][0] +=1
        if "abgelaufen" not in guidline_metadata['Guideline_Name']:
            header_split.metadata["Gültigkeit"] = "Gültig"
        else:
            header_split.metadata["Gültigkeit"] = "Abgelaufen"

def update_metadata_with_Fachgesellschaft(md_header_splits, guidline_metadata):
    '''
    This function includes the Fachgesellschaft into the final chunks metadata
    '''
    for header_split in md_header_splits:
        header_split.metadata['Fachgesellschaft'] = guidline_metadata['Fachgesellschaft']

def update_metadata_with_href(md_header_splits, guidline_metadata):
    '''
    This function includes the download_href into the final chunks metadata
    '''
    for header_split in md_header_splits:
        header_split.metadata['href'] = guidline_metadata['download_href']

def normalize_unicode(input_string):
    return unicodedata.normalize('NFC', input_string)