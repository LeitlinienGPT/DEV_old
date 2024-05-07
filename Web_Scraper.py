import asyncio
from playwright.async_api import async_playwright
import aiohttp
# import aiofiles
import asyncio
import os
import re
import json

def save_metadata_to_json(metadata, folder, filename='guideline_metadata.json'):
    """Save metadata dictionary to a JSON file in the specified folder."""
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)
    
    # Full path to the metadata file
    metadata_file_path = os.path.join(folder, filename)
    
    # Write the dictionary to a JSON file
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    print(f"Saved metadata to {metadata_file_path}")

# The function to download the file
async def download_file(url, folder, name):
    timeout = aiohttp.ClientTimeout(total=600)  # Sets a longer timeout

    if not os.path.isdir(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, name)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                with open(os.path.join(file_path), 'wb') as f:
                    async for chunk in resp.content.iter_chunked(1024):  # Reads in chunks of 1024 bytes
                        f.write(chunk)
                print(f"Downloaded {name}")

            else:
                print(f"Failed to download {url}. Status code: {resp.status}")

async def run():
    download_folder = 'Database'
    Guideline_Metadata = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # Use headless=False to see the browser UI
        context = await browser.new_context()
        context.set_default_timeout(120000)
        page = await context.new_page()

        # Navigate to the page
        await page.goto('https://register.awmf.org/de/leitlinien/aktuelle-leitlinien')

        # Wait for the list under "NACH FACHGESELLSCHAFT" to load
        await page.wait_for_selector('ion-col.guideline-listing-title.md.hydrated')

        # Wait for the "NACH FACH" tab to be available and click on it
        await page.click('ion-segment-button[value="discipline"]')

        # Wait for the list under "NACH FACH" to load
        await page.wait_for_selector('ion-col.guideline-listing-title.md.hydrated')

        # Collect all link URLs
        link_elements = await page.query_selector_all('ion-col.guideline-listing-title.md.hydrated a')
        main_links = []
        Fachgesellschaften= []
        for link_element in link_elements:
            href = await link_element.get_attribute('href')
            main_links.append(href)

            # Retrieve and clean up Fachgesellschaften 
            Fachgesellschaft = await link_element.text_content()
            Fachgesellschaft = Fachgesellschaft.replace("; DGf", "").strip()
            Fachgesellschaften.append(Fachgesellschaft)

        print("Fachgesellschaften: ",len(Fachgesellschaften))

        # Loop through the collected links and click on each one
        for i in range(len(main_links)):
            # Navigate to the link
            print(f'Fachgesellschaft: {Fachgesellschaften[i]}')
            #Fachgesellschaften[i] = Fachgesellschaften[5]
            # await page.goto('https://register.awmf.org/de/leitlinien/aktuelle-leitlinien/fachgesellschaft/065')
            await page.goto(f'https://register.awmf.org{main_links[i]}')
            await page.wait_for_load_state('networkidle')

            # Locate the h3 header
            header = await page.query_selector('h3:text("Beteiligungen an Leitlinien anderer Fachgesellschaften")')

            # Get the y-coordinate of the header
            header_box = await header.bounding_box() if header else None
            header_y = header_box['y'] if header_box else float('inf')

            # Collect sub_links only above the h3 header
            sub_link_elements = await page.query_selector_all('ion-col.guideline-listing-title.md.hydrated a')
            sub_links = []
            for sub_link in sub_link_elements:
                sub_link_box = await sub_link.bounding_box()
                if sub_link_box is None:
                    break
                if sub_link_box['y'] < header_y:
                    href = await sub_link.get_attribute('href')
                    sub_links.append(href)
            print("sub_links:", len(sub_links))
            
            for sub_link in sub_links:
                await page.wait_for_load_state('networkidle')

                # Click on sub-link and wait for navigation
                await page.goto(sub_link)

                try:
                    # Attempt to find the 'Download' button with a 60 second timeout
                    first_download_button = page.locator('text=Download').first # 
                    download_href = await first_download_button.get_attribute('href')
                except Exception as e:  # Broad exception to catch any issue that might occur
                    print(f"Error with 'Download' button: {str(e)}")
                    # If the 'Download' button is not found, look for the 'weiterlesen' button
                    print("Download button not found, looking for 'weiterlesen' button.")
                    weiterlesen_button = page.locator('text=weiterlesen').first
                    weiterlesen_href = await weiterlesen_button.get_attribute('href')
                    print("weiterlesen_href: ",weiterlesen_href)
                    await page.goto(f'{weiterlesen_href}')
                    await page.wait_for_load_state('networkidle')
                    first_download_button = page.locator('text=Download').first
                    download_href = await first_download_button.get_attribute('href')

                # Retrieve the 'href' attribute of the first download button
                download_href = await first_download_button.get_attribute('href')
                
                Guideline_Name = download_href.rsplit('/', 1)[-1]
                Guideline_Name_tmp = re.sub(r'[<>:"/\\|?*]', '', Guideline_Name)

                final_link = f"https://register.awmf.org{download_href}"

                if Guideline_Name_tmp not in Guideline_Metadata:
                    
                    Guideline_Metadata[Guideline_Name_tmp] = {
                                                        'Fachgesellschaft': [Fachgesellschaften[i]],
                                                        'download_href': f"https://register.awmf.org{download_href}",
                                                        'Guideline_Name': Guideline_Name
                                                    }

                    # Call the download function                
                    await download_file(final_link, download_folder, Guideline_Name_tmp)
                else:
                    if Fachgesellschaften[i] not in Guideline_Metadata[Guideline_Name_tmp]['Fachgesellschaft']:
                        Guideline_Metadata[Guideline_Name_tmp]['Fachgesellschaft'].append(Fachgesellschaften[i])
                        print(f"Skipping already processed link: {Guideline_Name_tmp}")
                        print(f"Fachgesellschaft now: {Guideline_Metadata[Guideline_Name_tmp]['Fachgesellschaft']}")
                await page.go_back()
            await page.go_back()
        save_metadata_to_json(Guideline_Metadata,download_folder)

        # Close the browser
        await browser.close()

# The entry point for the script execution
if __name__ == '__main__':
    asyncio.run(run())