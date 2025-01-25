import requests
from bs4 import BeautifulSoup
import json
import time

class BaseScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
    
    def fetch_html(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {e}")
            return None

    def parse(self, html):
        raise NotImplementedError("Subclasses must implement parse method")

    def save_to_file(self, data, filename):
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

class AmazonScraper(BaseScraper):
    def parse(self, html):
        soup = BeautifulSoup(html, 'lxml')
        products = []
        
        try:
            items = soup.find_all('div', {'data-component-type': 's-search-result'})
            for item in items:
                product = {
                    'title': self._get_title(item),
                    'price': self._get_price(item),
                    'rating': self._get_rating(item),
                    'url': self._get_url(item)
                }
                products.append(product)
            return products
        except Exception as e:
            print(f"Error parsing Amazon page: {e}")
            return []

    def _get_title(self, item):
        title_tag = item.find('span', class_='a-size-medium')
        return title_tag.text.strip() if title_tag else None

    def _get_price(self, item):
        price_span = item.find('span', class_='a-offscreen')
        return price_span.text if price_span else None

    def _get_rating(self, item):
        rating_tag = item.find('span', class_='a-icon-alt')
        return rating_tag.text.split()[0] if rating_tag else None

    def _get_url(self, item):
        link_tag = item.find('a', class_='a-link-normal')
        return f"https://www.amazon.com{link_tag['href']}" if link_tag else None

class TargetScraper(BaseScraper):
    def parse(self, html):
        soup = BeautifulSoup(html, 'lxml')
        products = []
        
        try:
            items = soup.find_all('div', {'data-test': 'productCard'})
            for item in items:
                product = {
                    'title': self._get_title(item),
                    'price': self._get_price(item),
                    'rating': self._get_rating(item),
                    'url': self._get_url(item)
                }
                products.append(product)
            return products
        except Exception as e:
            print(f"Error parsing Target page: {e}")
            return []

    def _get_title(self, item):
        title_tag = item.find('a', {'data-test': 'product-title'})
        return title_tag.text.strip() if title_tag else None

    def _get_price(self, item):
        price_div = item.find('div', {'data-test': 'product-price'})
        return price_div.text.strip() if price_div else None

    def _get_rating(self, item):
        rating_div = item.find('div', {'data-test': 'ratings'})
        return rating_div.text.strip() if rating_div else None

    def _get_url(self, item):
        link_tag = item.find('a', {'data-test': 'product-title'})
        return f"https://www.target.com{link_tag['href']}" if link_tag else None

if __name__ == "__main__":
    # Example usage
    amazon_scraper = AmazonScraper()
    target_scraper = TargetScraper()

    # Amazon example
    amazon_html = amazon_scraper.fetch_html('https://www.amazon.com/s?k=laptop')
    if amazon_html:
        amazon_products = amazon_scraper.parse(amazon_html)
        amazon_scraper.save_to_file(amazon_products, 'amazon_products.json')

    # Target example
    target_html = target_scraper.fetch_html('https://www.target.com/s?searchTerm=laptop')
    if target_html:
        target_products = target_scraper.parse(target_html)
        target_scraper.save_to_file(target_products, 'target_products.json')

    time.sleep(1)  # Respectful delay between requests