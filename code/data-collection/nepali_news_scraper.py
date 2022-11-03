from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm.autonotebook import tqdm

class NepaliNewsScraper:
    def __init__(self, category):
        """Initializer with a category name.

        Parameters
        ----------
        category : str
            News category to be scraped. This parameter will also serve as the
            value for 'category' column in the exported dataframe.
        """
        self.category = category
        
    def reset_arrays(self):
        """Resets all defined arrays for clearance.
        """
        self.headings = []
        self.urls = []
        self.contents = []
        
    def __extract_news(self, url):
        """Gets soup object from the url specified.

        Parameters
        ----------
        url : str
            URL to be scraped.

        Returns
        -------
        BeautifulSoup
            Response content in soup format.
        """
        response = requests.get(url)
        content = response.content
        soup = BeautifulSoup(content, 'html.parser')
        
        return soup
    
    def export_dataframe_to_csv(self, export_file_name):
        """Exports dataframe associated with the class to CSV.

        Parameters
        ----------
        export_file_name : str
            Destination file path and file name combined.
        """
        filename = f'./{self.category.lower()}/{self.source.lower()}_{self.category.lower()}.csv'
        if export_file_name != None:
            filename = f'./{self.category.lower()}/{export_file_name}'
        self.df.to_csv(filename, encoding='utf-8', index=False)
    
    def __create_dataframe(self, export_to_csv, export_file_name=None):
        """Create dataframe from class variables.

        Parameters
        ----------
        export_to_csv : bool
            Whether to export or not.
        export_file_name : str, optional
            Destination file path and file name combined, by default None

        Returns
        -------
        DataFrame
            Created dataframe.
        """

        # Make sure that all data lengths are same
        assert(len(self.urls) == len(self.headings) == len(self.contents))
        
        df = pd.DataFrame(
            data={
                'source': self.source.capitalize(),
                'category': self.category,
                'heading': self.headings,
                'content': self.contents
                }
            )
        self.df = df
        print('Dataframe created.')
        
        if export_to_csv:
            self.export_dataframe_to_csv(export_file_name)
        
        return df
    
    #####################################################################
    #                           EKANTIPUR                               #
    #####################################################################  
    
    # JS HELPER 
    # window.scrollTo(0, document.body.scrollHeight)
    
    def fetch_articles_from_ekantipur_file(self, filepath, export_to_csv):
        """Fetch articles from Ekantipur static HTML files.

        Parameters
        ----------
        filepath : str
            Source HTML file.
        export_to_csv : bool
            Whether to export or not.
        """
        
        self.source = 'Ekantipur'
        self.reset_arrays()
    
        # Read HTML from file
        with open(filepath, encoding='utf-8') as html_file:
            content = html_file.read()
        
        # Parse HTML and get article tag
        soup = BeautifulSoup(content, 'html.parser')
        article_tags = soup.find_all('article', class_='normal')
        print(f'{len(article_tags)} {self.category} articles found.')

        # Extract URLs, Headings and Contents
        urls = [tag.find('a').get('href') for tag in article_tags]
        self.urls = urls
        print(f'Extracted {len(urls)} urls.')
        
        headings = [tag.find('a').text for tag in article_tags]
        self.headings = headings
        print(f'Extracted {len(headings)} headings.')
        
        self.contents = []
        
        print('Scraping contents from urls...')
        for url in tqdm(urls):
            soup = self.__extract_news(url)
            text = soup.find('div', class_='description current-news-block').text
            self.contents.append(text)
            
        print(f'Extracted {len(self.contents)} contents.')
        
        self.__create_dataframe(export_to_csv)
        
        
    #####################################################################
    #                           NAGARIK NEWS                            #
    #####################################################################
    
    # JS HELPER 
    # window.scrollTo(0, document.body.scrollHeight)
    # document.getElementById('loadmore').click()
    
    def fetch_articles_from_nagarik_file(self, filepath, export_to_csv):
        """Fetch articles from Nagariknews static HTML files.

        Parameters
        ----------
        filepath : str
            Source HTML file.
        export_to_csv : bool
            Whether to export or not.
        """
    
        self.source = 'Nagarik'
        self.reset_arrays()

        # Read HTML from file
        with open(filepath, encoding='utf-8') as html_file:
            content = html_file.read()
        
        # Parse HTML and get article tag
        soup = BeautifulSoup(content, 'html.parser')
        article_tags = soup.find_all('article', class_='list-group-item')
        print(f'{len(article_tags)} Nagarik {self.category} articles found.')

        # Extract URLs, Headings and Contents
        urls = [tag.find('a').get('href') for tag in article_tags]
        self.urls = urls
        print(f'Extracted {len(urls)} urls.')
        
        headings = [tag.find('a').get('title') for tag in article_tags]
        self.headings = headings
        print(f'Extracted {len(headings)} headings.')
        
        self.contents = []
        
        print('Scraping contents from urls...')
        for url in tqdm(urls):

            text = ''
            soup = self.__extract_news(url)
            content_tag = soup.find('div', class_='content text-justify')
            if content_tag != None:
                text = ' '.join([p_content.text for p_content in content_tag.find_all('p', class_=None)])
            self.contents.append(text)
            
        print(f'Extracted {len(self.contents)} contents.')
        
        self.__create_dataframe(export_to_csv)
        
    #####################################################################
    #                           ONLINEKHABAR                            #
    #####################################################################
    
    def fetch_articles_from_onlinekhabar(self, ok_category_url, page_from, page_to, export_to_csv):
        """Fetch articles from Onlinekhabar category-specific pages

        Parameters
        ----------
        ok_category_url : str
            Category URL of Onlinekhabar page from where news is to be extracted.
        page_from : int
            Start index of page
        page_to : int
            End index of page
        export_to_csv : bool
            Whether to export or not.
        """
    
        self.source = 'Onlinekhabar'
        self.reset_arrays()
        
        # Error Count
        error_count = 0
            
        # Get main category urls
        ok_category_urls = [ok_category_url + '/page/' + str(i+1) for i in range(page_from, page_to+1)]
        
        # Get posts from each paginated pages
        for page_url in tqdm(ok_category_urls):
            try:
                content = self.__extract_news(page_url)

                if content != None:
                    grid_12_content = content.find('div', class_='ok-grid-12')

                if grid_12_content != None:
                    news_posts = grid_12_content.find_all('div', class_='ok-news-post')

                # Extract URLs, Headings and Contents
                if len(news_posts) > 0:
                    urls = [news_post.find('a').get('href') for news_post in news_posts]
                    self.urls += urls

                    headings = [news_post.find('h2').text for news_post in news_posts]
                    self.headings += headings
                    
            except Exception as e:
                error_count += 1
                print(str(e))
        
        print(f'Extracted {len(self.urls)} urls.')
        print(f'Extracted {len(self.headings)} headings.')

        print(f'{error_count} errors !')

        # Getting contents
        print('Scraping contents from urls...')
        for url in tqdm(self.urls):
            text = ''
            soup = self.__extract_news(url)
            content_tag = soup.find('div', class_='ok18-single-post-content-wrap')
            if content_tag != None:
                text = ' '.join([p_content.text for p_content in content_tag.find_all('p', class_=None)])
            self.contents.append(text)
            
        print(f'Extracted {len(self.contents)} contents.')

        self.__create_dataframe(export_to_csv)
        
    #####################################################################
    #                           GORKHAPATRA                             #
    #####################################################################
    
    def fetch_articles_from_gorkhapatra_file(self, filepath, export_to_csv):
        """Fetch articles from Gorkhapatra static HTML files.

        Parameters
        ----------
        filepath : str
            Source HTML file.
        export_to_csv : bool
            Whether to export or not.
        """

        self.source = 'Gorkhapatra'
        self.reset_arrays()

        # Read HTML from file
        with open(filepath, encoding='utf-8') as html_file:
            content = html_file.read()
        
        # Parse HTML and get business tag
        soup = BeautifulSoup(content, 'html.parser')
        business_tags = soup.find_all('div', class_='business')
        print(f'{len(business_tags)} {self.category} articles found.')

        # Extract URLs, Headings and Contents
        urls = [tag.find('a').get('href') for tag in business_tags]
        self.urls = urls
        print(f'Extracted {len(urls)} urls.')
        
        headings = [tag.find('p', class_='trand middle-font').text.strip() for tag in business_tags]
        self.headings = headings
        print(f'Extracted {len(headings)} headings.')        
        
        self.contents = []
        
        print('Scraping contents from urls...')
        for url in tqdm(urls):

            text = ''
            soup = self.__extract_news(url)
            content_tag = soup.find('div', id='content')
            text = ' '.join([p_content.text.strip() for p_content in content_tag.find_all('p', class_=None)])
            self.contents.append(text)
            
        print(f'Extracted {len(self.contents)} contents.')
        
        
        self.__create_dataframe(export_to_csv)
        
    
    #####################################################################
    #                           NEPALKHABAR                             #
    #####################################################################
    
    def fetch_articles_from_nepalkhabar(self, nk_category_url, page_from, page_to, export_to_csv):
        """Fetch articles from Nepalkhabar category-specific pages

        Parameters
        ----------
        nk_category_url : str
            Category URL of Onlinekhabar page from where news is to be extracted.
        page_from : int
            Start index of page
        page_to : int
            End index of page
        export_to_csv : bool
            Whether to export or not.
        """
    
        self.source = 'Nepalkhabar'
        self.reset_arrays()
        nk_root_category_url = 'https://nepalkhabar.com/category'
    
        # Get main category urls
        nk_category_urls = [nk_category_url + '?page=' + str(i) for i in range(page_from, page_to+1)]

        # Get posts from each paginated pages
        for page_url in tqdm(nk_category_urls):
            content = self.__extract_news(page_url)
            
            list_items = content.find_all('li', class_='nk-item-list-first-page')
    
            # Extract URLs, Headings and Contents
            urls = [nk_root_category_url + list_item.find('a').get('href') for list_item in list_items]
            self.urls += urls
                
            headings = [list_item.find('img').get('alt') for list_item in list_items]
            self.headings += headings
            
        print(f'Extracted {len(self.urls)} urls.')
        print(f'Extracted {len(self.headings)} headings.')
        
        # Getting contents
        print('Scraping contents from urls...')
        for url in tqdm(self.urls):
            text = ''
            soup = self.__extract_news(url)
            body_content = soup.find('div', id='body-content')
            text = ' '.join(p_tag.text for p_tag in body_content.find_all('p', class_=None))
            self.contents.append(text)
            
        print(f'Extracted {len(self.contents)} contents.')
        
        self.__create_dataframe(export_to_csv)
        
    #####################################################################
    #                           NEPALIPATRA                             #
    #####################################################################
    
    def fetch_articles_from_nepalipatra(self, np_category_url, page_from, page_to, export_to_csv):
        """Fetch articles from Nepalipatra category-specific pages

        Parameters
        ----------
        np_category_url : str
            Category URL of Nepalipatra page from where news is to be extracted.
        page_from : int
            Start index of page
        page_to : int
            End index of page
        export_to_csv : bool
            Whether to export or not.
        """
    
        self.source = 'Nepalipatra'
        self.reset_arrays()
        np_root_url = 'https://www.nepalipatra.com'
    
        # Get main category urls
        np_category_urls = [np_category_url + '?page=' + str(i) for i in range(page_from, page_to+1)]

        # Get posts from each paginated pages
        for page_url in tqdm(np_category_urls):
            soup = self.__extract_news(page_url)
            h2_tags = soup.find('div', class_='col-md-8').find_all('h2')
    
            # Extract URLs, Headings and Contents
            urls = [np_root_url + h2_tag.find('a').get('href') for h2_tag in h2_tags]
            self.urls += urls
                
            headings = [np_root_url + h2_tag.find('a').text.strip() for h2_tag in h2_tags]
            self.headings += headings
            
        print(f'Extracted {len(self.urls)} urls.')
        print(f'Extracted {len(self.headings)} headings.')
        
        # Getting contents
        print('Scraping contents from urls...')
        for url in tqdm(self.urls):
            text = ''
            soup = self.__extract_news(url)
            news_detail = soup.find('div', class_='news-detail')
            if news_detail != None:
                text = ' '.join(p_tag.text for p_tag in news_detail.find_all('p', class_=None))
            else:
                text = ''
            self.contents.append(text)
            
        print(f'Extracted {len(self.contents)} contents.')
        
        self.__create_dataframe(export_to_csv)
        
    #####################################################################
    #                            RATOPATI                               #
    #####################################################################
    
    def fetch_articles_from_ratopati(self, rp_category_url, page_from, page_to, export_to_csv):
        """Fetch articles from Ratopati category-specific pages

        Parameters
        ----------
        rp_category_url : str
            Category URL of Onlinekhabar page from where news is to be extracted.
        page_from : int
            Start index of page
        page_to : int
            End index of page
        export_to_csv : bool
            Whether to export or not.
        """
    
        self.source = 'Ratopati'
        self.reset_arrays()
        rp_root_url = 'https://ratopati.com/category'
    
        # Get main category urls
        rp_category_urls = [rp_category_url + '?page=' + str(i) for i in range(page_from, page_to+1)]

        # Get posts from each paginated pages
        for page_url in tqdm(rp_category_urls):
            soup = self.__extract_news(page_url)
            
            item_contents = soup.find('div', class_='ot-articles-material-blog-list').find_all('div', class_='item-content')
    
            # Extract URLs, Headings and Contents
            urls = [rp_root_url + item_content.find('a').get('href') for item_content in item_contents]
            self.urls += urls
                
            headings = [item_content.find('a').text.strip() for item_content in item_contents]
            self.headings += headings
            
        print(f'Extracted {len(self.urls)} urls.')
        print(f'Extracted {len(self.headings)} headings.')
        
        # Getting contents
        print('Scraping contents from urls...')
        for url in tqdm(self.urls):
            text = ''
            soup = self.__extract_news(url)
            text = ''
            rp_table = soup.find('div', class_='ratopati-table-border-layout')
            text = ' '.join(p_tag.text for p_tag in rp_table.find_all('p', class_=None))
            self.contents.append(text)
            
        print(f'Extracted {len(self.contents)} contents.')
        
        self.__create_dataframe(export_to_csv)
        
    #####################################################################
    #                           CRIMENEWS                               #
    #####################################################################
    
    def fetch_articles_from_crimenews(self, cn_category_url, page_from, page_to, export_to_csv, export_file_name):
        """Fetch articles from Crimenews category-specific pages

        Parameters
        ----------
        cn_category_url : str
            Category URL of Onlinekhabar page from where news is to be extracted.
        page_from : int
            Start index of page
        page_to : int
            End index of page
        export_to_csv : bool
            Whether to export or not.
        """
    
        self.source = 'Crimenews'
        self.reset_arrays()
        cn_root_url = 'https://crimenewsnepal.com'
    
        # Get main category urls
        cn_category_urls = [cn_category_url + '/page/' + str(i) for i in range(page_from, page_to+1)]

        # Get posts from each paginated pages
        for page_url in tqdm(cn_category_urls):
            soup = self.__extract_news(page_url)
            
            section_category_news = soup.find('section', class_='category-news')
            h2_tags = section_category_news.find_all('h2', class_='entry-title')
    
            # Extract URLs, Headings and Contents
            urls = [h2_tag.find('a').get('href') for h2_tag in h2_tags]
            self.urls += urls
                
            headings = [h2_tag.find('a').text.strip() for h2_tag in h2_tags]
            self.headings += headings
            
        print(f'Extracted {len(self.urls)} urls.')
        print(f'Extracted {len(self.headings)} headings.')
        
        # Getting contents
        print('Scraping contents from urls...')
        for url in tqdm(self.urls):
            text = ''
            soup = self.__extract_news(url)
            text = ''
            news_detail = soup.find('div', class_='newsDetailContentWrapper')
            text = ' '.join(p_tag.text for p_tag in news_detail.find_all('p', class_=None))
            self.contents.append(text)
            
        print(f'Extracted {len(self.contents)} contents.')
        
        self.__create_dataframe(export_to_csv, export_file_name)