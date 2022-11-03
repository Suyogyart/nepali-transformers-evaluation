# Data Collection

## Nepali News Scraping

### Scraper usage
1. Initialize the scraper with desired **category** 
2. Specify [category-specific URL](#examples-of-category-specific-urls)
3. Specify the scraping page range for paginated websites
4. Call the respective methods to fetch articles, this will set the **source** of news.
5. Get the Pandas DataFrame generated.

```python
# Example 1
ns = NepaliNewsScraper(category='Sports')
ns.fetch_articles_from_onlinekhabar(
    ok_category_url='https://www.onlinekhabar.com/content/sports-news', 
    page_from=1, 
    page_to=2, 
    export_to_csv=False
)
```
### Process
1. Get the main URLs for news articles from given page ranges of the category-specific URLs.
2. Scrape the *headings* and *content* from the main URL of the news article which will serve as **heading** and **content** columns in the DataFrame.

### Examples of Category-specific URLs

These URLs are basically the ones which have pagination and depending on the number of pages available, we are required to set `page_from` and `page_to` parameters with valid values.

**For Onlinekhabar**
- *Sports* - https://www.onlinekhabar.com/content/sports-news
- *Opinion* - https://www.onlinekhabar.com/content/opinion/page/2
- *Tourism* - https://www.onlinekhabar.com/content/tourism/page/35

**For Nepalkhabar**
- *Sports* - https://nepalkhabar.com/category/sports
- *Literature* - https://nepalkhabar.com/category/literature?page=2
- *Society* - https://nepalkhabar.com/category/society?page=10

**For Ekantipur**
- *World* - https://ekantipur.com/world
- *Lifestyle* - https://ekantipur.com/lifestyle
- *Health* - https://ekantipur.com/health

`?page=<page_number>` or `/page/<page_number>` will be appended to the category specific URLs based on the websites to get the list of articles from specific URLs.

For some news websites such as **Ekantipur, Gorkhapatra and Nagariknews**, they do not have pagination, they have JavaScript enabled webpages to load more news. Hence, we scroll down until desired number of news articles are available, then save the page. So, we scrape news from the static HTML files.

## Exploratory Data Analysis of Whole Dataset
