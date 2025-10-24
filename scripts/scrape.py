
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import json
import csv
import time
from datetime import datetime
import re
import os
import sys

class TrumpTranscriptScraper:
    def __init__(self, driver_path=None):
        self.base_url = "https://www.rev.com"
        self.driver = None
        self.driver_path = driver_path
        self.speeches = []
        
        self.known_urls = [
            "https://www.rev.com/transcripts/rutte-and-trump-at-white-house",
            "https://www.rev.com/transcripts/trump-speaks-at-senate-luncheon",
            "https://www.rev.com/transcripts/zelensky-meeting-at-white-house",
            "https://www.rev.com/transcripts/white-house-ivf-announcement",
            "https://www.rev.com/transcripts/argentina-president-visits-white-house",
            "https://www.rev.com/transcripts/trump-press-gaggle-aboard-airforce-one-10-14-25",
            "https://www.rev.com/transcripts/trump-speaks-at-ceasefire-ceremony",
            "https://www.rev.com/transcripts/trump-announces-hostages-released",
            "https://www.rev.com/transcripts/finnish-president-visits-white-house",
            "https://www.rev.com/transcripts/white-house-cabinet-meeting-10-09-25",
            "https://www.rev.com/transcripts/antifa-roundtable",
            "https://www.rev.com/transcripts/trump-and-carney-at-the-white-house",
            "https://www.rev.com/transcripts/trump-executive-order-10-06-25",
            "https://www.rev.com/transcripts/trump-makes-drug-announcement",
            "https://www.rev.com/transcripts/hegseth-and-trump-address-to-military",
            "https://www.rev.com/transcripts/trump-and-netanyahu-press-conference",
            "https://www.rev.com/transcripts/tiktok-executive-order",
            "https://www.rev.com/transcripts/trump-speaks-at-un",
            "https://www.rev.com/transcripts/h-1b-executive-order",
            "https://www.rev.com/transcripts/polish-president-visits-the-white-house",
            "https://www.rev.com/transcripts/trump-cabinet-meeting-8-26-25",
            "https://www.rev.com/transcripts/executive-orders-on-8-25-25",
            "https://www.rev.com/transcripts/european-leaders-meet-at-white-house",
            "https://www.rev.com/transcripts/trump-meets-with-zelenskyy",
            "https://www.rev.com/transcripts/trump-hosts-world-leaders-at-the-white-house",
            "https://www.rev.com/transcripts/trump-and-starmer-hold-press-conference",
            "https://www.rev.com/transcripts/trump-announces-eu-deal",
            "https://www.rev.com/transcripts/starmer-arrives-in-scotland",
            "https://www.rev.com/transcripts/trump-ai-action-plan",
            "https://www.rev.com/transcripts/philippine-president-visits-white-house",
            "https://www.rev.com/transcripts/trump-speaks-to-republican-senators",
            "https://www.rev.com/transcripts/trump-welcomes-prince-of-bahrain",
            "https://www.rev.com/transcripts/trump-meets-with-rutte",
            "https://www.rev.com/transcripts/african-leaders-at-white-house",
            "https://www.rev.com/transcripts/white-house-dinner-with-netanyahu",
            "https://www.rev.com/transcripts/trump-speaks-at-military-family-picnic",
            "https://www.rev.com/transcripts/trump-at-2025-nato-summit",
            "https://www.rev.com/transcripts/trump-speaks-on-iran-bombing",
            "https://www.rev.com/transcripts/trump-and-starmer-speak-to-press-at-g7",
            "https://www.rev.com/transcripts/trump-and-carney-speak-to-press",
            "https://www.rev.com/transcripts/trump-bill-signing-6-12-25",
            "https://www.rev.com/transcripts/german-chancellor-visits-white-house",
            "https://www.rev.com/transcripts/trump-and-musk-press-conference",
        ]
    
    def setup_driver(self):        
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        try:
            if self.driver_path:
                service = Service(self.driver_path)
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                if os.path.exists('chromedriver.exe'):
                    service = Service('./chromedriver.exe')
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
                else:
                    self.driver = webdriver.Chrome(options=chrome_options)
            
            print("WebDriver ready")
        except Exception as e:
            print(f"Error: {e}")
            print("\nerror")
            sys.exit(1)
    
    def find_more_trump_transcripts(self):
        """Try to find more Trump transcripts by:"""
        print("\nSearching...")
        additional_urls = []
        
        try:
            print("  Checking main transcript library...")
            self.driver.get("https://www.rev.com/blog/transcripts")
            time.sleep(5)
            
            for i in range(5):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True).lower()
                
                if '/blog/transcripts/' in href and ('trump' in text or 'trump' in href.lower()):
                    full_url = f"https://www.rev.com{href}" if href.startswith('/') else href
                    if full_url not in self.known_urls and full_url not in additional_urls:
                        additional_urls.append(full_url)
                        print(f"    Found: {text[:60]}")
        except Exception as e:
            print(f"  Error searching: {e}")
        
        print(f"  Found {len(additional_urls)} additional transcripts")
        return additional_urls
    
    def scrape_transcript(self, url):
        print(f"\nScraping: {url}")
        
        try:
            time.sleep(2)  
            self.driver.get(url)
            time.sleep(5)  
            
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            data = {
                'url': url,
                'title': '',
                'date': '',
                'transcript': '',
                'scraped_at': datetime.now().isoformat()
            }
            
            title_tag = soup.find('h1')
            if title_tag:
                data['title'] = title_tag.get_text(strip=True)
                print(f"  Title: {data['title']}")
            
            date_tag = soup.find('time')
            if date_tag:
                data['date'] = date_tag.get('datetime', date_tag.get_text(strip=True))
            
            if not hasattr(self, '_saved_debug_html'):
                with open('single_transcript_debug.html', 'w', encoding='utf-8') as f:
                    f.write(self.driver.page_source)
                print("  (Saved debug HTML)")
                self._saved_debug_html = True
            
            transcript_text = ""
            
            main_content = soup.find('div', id='main-content')
            if main_content:
                paragraphs = main_content.find_all('p')
                if paragraphs:
                    transcript_text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    print(f"  Found via main-content div: {len(paragraphs)} paragraphs")
            
            if not transcript_text:
                toc_div = soup.find('div', attrs={'fs-toc-element': 'contents'})
                if toc_div:
                    paragraphs = toc_div.find_all('p')
                    if paragraphs:
                        transcript_text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                        print(f"  Found via toc div: {len(paragraphs)} paragraphs")
            
            if not transcript_text:
                body_container = soup.find('div', class_=re.compile('article.*body', re.I))
                if body_container:
                    paragraphs = body_container.find_all('p')
                    if len(paragraphs) > 5:
                        transcript_text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                        print(f"  Found via body container: {len(paragraphs)} paragraphs")
            
            if not transcript_text:
                all_paragraphs = soup.find_all('p')
                if len(all_paragraphs) > 10:
                    content_paragraphs = []
                    for p in all_paragraphs:
                        text = p.get_text(strip=True)
                        if len(text) > 30 and 'copyright' not in text.lower() and 'fair use' not in text.lower():
                            content_paragraphs.append(text)
                    
                    if content_paragraphs:
                        transcript_text = '\n\n'.join(content_paragraphs)
                        print(f"  Found via filtered paragraphs: {len(content_paragraphs)} paragraphs")
            
            data['transcript'] = transcript_text
            print(f"  Date: {data['date']}")
            print(f"  Transcript length: {len(transcript_text)} characters")
            
            if len(transcript_text) > 500:
                return data
            else:
                print(" Warning")
                if len(transcript_text) > 100:
                    return data
                return None
                
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def scrape_all(self):
        print("=" * 70)
        print("all")
        print("=" * 70)
        
        try:
            self.setup_driver()
            
            additional_urls = self.find_more_trump_transcripts()
            all_urls = self.known_urls + additional_urls
            
            print(f"\n{'=' * 70}")
            print(f"SCRAPING {len(all_urls)} TRANSCRIPTS")
            print("=" * 70)
            
            for i, url in enumerate(all_urls, 1):
                print(f"\n[{i}/{len(all_urls)}]", end=" ")
                
                data = self.scrape_transcript(url)
                if data and data['transcript']:
                    self.speeches.append(data)
                else:
                    print("  Skipped")
            
            self.save_results()
            
        finally:
            if self.driver:
                print("\ndone")
                self.driver.quit()
    
    def save_results(self):
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        
        if not self.speeches:
            print("No speeches scraped!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_file = f"trump_speeches_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.speeches, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(self.speeches)} speeches to {json_file}")
        
        csv_file = f"trump_speeches_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'date', 'url', 'transcript', 'scraped_at'])
            writer.writeheader()
            for speech in self.speeches:
                writer.writerow({
                    'title': speech.get('title', ''),
                    'date': speech.get('date', ''),
                    'url': speech.get('url', ''),
                    'transcript': speech.get('transcript', ''),
                    'scraped_at': speech.get('scraped_at', '')
                })
        print(f" Saved {len(self.speeches)} speeches to {csv_file}")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total speeches: {len(self.speeches)}")
        
        total_words = sum(len(speech['transcript'].split()) for speech in self.speeches)
        print(f"Total words: {total_words:,}")
        
        if self.speeches:
            print("\nScraped transcripts:")
            for speech in self.speeches:
                print(f"  - {speech['title']}")

def main():
    driver_path = sys.argv[1] if len(sys.argv) > 1 else None
    scraper = TrumpTranscriptScraper(driver_path)
    scraper.scrape_all()

if __name__ == "__main__":
    main()

