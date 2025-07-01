import axios from 'axios';
import * as cheerio from 'cheerio';
import { logger } from '@/config/logger';
import { config } from '@/config/environment';
import { Article, NewsSource } from '@/types';

export class DataAggregationService {
  private static instance: DataAggregationService;
  
  public static getInstance(): DataAggregationService {
    if (!DataAggregationService.instance) {
      DataAggregationService.instance = new DataAggregationService();
    }
    return DataAggregationService.instance;
  }

  /**
   * Parse RSS feed and extract articles
   */
  async parseRssFeed(rssUrl: string): Promise<Partial<Article>[]> {
    try {
      logger.info(`Parsing RSS feed: ${rssUrl}`);
      
      const response = await axios.get(rssUrl, {
        headers: {
          'User-Agent': 'Veritas-Lens/1.0.0'
        },
        timeout: 30000
      });

      const $ = cheerio.load(response.data, { xmlMode: true });
      const articles: Partial<Article>[] = [];

      $('item').each((index, element) => {
        const $item = $(element);
        
        const article: Partial<Article> = {
          title: $item.find('title').text().trim(),
          url: $item.find('link').text().trim(),
          description: $item.find('description').text().trim(),
          author: $item.find('author').text().trim() || $item.find('dc\\:creator').text().trim(),
          publishedAt: new Date($item.find('pubDate').text() || $item.find('dc\\:date').text()),
          source: this.extractDomainFromUrl(rssUrl),
          isLabeled: false,
          createdAt: new Date(),
          updatedAt: new Date()
        };

        // Extract content from description if no separate content field
        if (article.description && !article.content) {
          article.content = this.cleanHtmlContent(article.description);
        }

        if (article.title && article.url) {
          articles.push(article);
        }
      });

      logger.info(`Parsed ${articles.length} articles from RSS feed: ${rssUrl}`);
      return articles;
    } catch (error) {
      logger.error(`Error parsing RSS feed ${rssUrl}:`, error);
      throw new Error(`Failed to parse RSS feed: ${error}`);
    }
  }

  /**
   * Scrape article content from URL
   */
  async scrapeArticleContent(url: string, selectors?: any): Promise<Partial<Article>> {
    try {
      logger.info(`Scraping article content: ${url}`);
      
      const response = await axios.get(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; Veritas-Lens/1.0.0; +https://veritaslens.com)'
        },
        timeout: 30000
      });

      const $ = cheerio.load(response.data);

      // Default selectors - can be overridden per source
      const defaultSelectors = {
        title: 'h1, .headline, .title, [class*="title"], [class*="headline"]',
        content: '.content, .article-content, .post-content, [class*="content"], p',
        author: '.author, .byline, [class*="author"], [class*="byline"]',
        publishedAt: '.date, .published, [datetime], [class*="date"], [class*="time"]'
      };

      const finalSelectors = { ...defaultSelectors, ...selectors };

      const article: Partial<Article> = {
        url,
        title: $(finalSelectors.title).first().text().trim(),
        content: this.extractTextContent($, finalSelectors.content),
        author: $(finalSelectors.author).first().text().trim(),
        source: this.extractDomainFromUrl(url),
        isLabeled: false,
        createdAt: new Date(),
        updatedAt: new Date()
      };

      // Try to extract published date
      const dateElement = $(finalSelectors.publishedAt).first();
      const dateText = dateElement.attr('datetime') || dateElement.text();
      if (dateText) {
        const publishedDate = new Date(dateText);
        if (!isNaN(publishedDate.getTime())) {
          article.publishedAt = publishedDate;
        }
      }

      logger.info(`Successfully scraped article: ${article.title}`);
      return article;
    } catch (error) {
      logger.error(`Error scraping article ${url}:`, error);
      throw new Error(`Failed to scrape article: ${error}`);
    }
  }

  /**
   * Fetch articles from News API
   */
  async fetchFromNewsApi(query: string = 'politics', sources?: string[]): Promise<Partial<Article>[]> {
    if (!config.apis.newsApi.key) {
      throw new Error('News API key not configured');
    }

    try {
      logger.info(`Fetching articles from News API with query: ${query}`);
      
      const params: any = {
        q: query,
        language: 'en',
        sortBy: 'publishedAt',
        apiKey: config.apis.newsApi.key
      };

      if (sources && sources.length > 0) {
        params.sources = sources.join(',');
      }

      const response = await axios.get(`${config.apis.newsApi.baseUrl}/everything`, {
        params,
        timeout: 30000
      });

      const articles: Partial<Article>[] = response.data.articles.map((apiArticle: any) => ({
        title: apiArticle.title,
        description: apiArticle.description,
        content: apiArticle.content,
        url: apiArticle.url,
        source: apiArticle.source.name,
        author: apiArticle.author,
        publishedAt: new Date(apiArticle.publishedAt),
        isLabeled: false,
        createdAt: new Date(),
        updatedAt: new Date()
      }));

      logger.info(`Fetched ${articles.length} articles from News API`);
      return articles;
    } catch (error) {
      logger.error('Error fetching from News API:', error);
      throw new Error(`Failed to fetch from News API: ${error}`);
    }
  }

  /**
   * Aggregate data from all configured sources
   */
  async aggregateFromAllSources(): Promise<Partial<Article>[]> {
    const allArticles: Partial<Article>[] = [];

    try {
      // Fetch from RSS feeds
      for (const rssUrl of config.dataAggregation.rssFeeds) {
        try {
          const rssArticles = await this.parseRssFeed(rssUrl);
          allArticles.push(...rssArticles.slice(0, config.dataAggregation.maxArticlesPerSource));
        } catch (error) {
          logger.error(`Failed to fetch from RSS ${rssUrl}:`, error);
        }
      }

      // Fetch from News API if configured
      if (config.apis.newsApi.key) {
        try {
          const newsApiArticles = await this.fetchFromNewsApi();
          allArticles.push(...newsApiArticles.slice(0, config.dataAggregation.maxArticlesPerSource));
        } catch (error) {
          logger.error('Failed to fetch from News API:', error);
        }
      }

      // Remove duplicates based on URL
      const uniqueArticles = this.removeDuplicateArticles(allArticles);
      
      logger.info(`Aggregated ${uniqueArticles.length} unique articles from all sources`);
      return uniqueArticles;
    } catch (error) {
      logger.error('Error during data aggregation:', error);
      throw error;
    }
  }

  /**
   * Extract domain from URL for source identification
   */
  private extractDomainFromUrl(url: string): string {
    try {
      const urlObj = new URL(url);
      return urlObj.hostname.replace('www.', '');
    } catch (error) {
      return 'unknown';
    }
  }

  /**
   * Clean HTML content and extract text
   */
  private cleanHtmlContent(html: string): string {
    const $ = cheerio.load(html);
    return $.text().trim().replace(/\s+/g, ' ');
  }

  /**
   * Extract text content from multiple elements
   */
  private extractTextContent($: cheerio.CheerioAPI, selector: string): string {
    const elements = $(selector);
    const textParts: string[] = [];

    elements.each((index, element) => {
      const text = $(element).text().trim();
      if (text && text.length > 50) { // Only include substantial text blocks
        textParts.push(text);
      }
    });

    return textParts.join(' ').replace(/\s+/g, ' ').trim();
  }

  /**
   * Remove duplicate articles based on URL and title similarity
   */
  private removeDuplicateArticles(articles: Partial<Article>[]): Partial<Article>[] {
    const seen = new Set<string>();
    const unique: Partial<Article>[] = [];

    for (const article of articles) {
      if (!article.url || !article.title) continue;
      
      const key = `${article.url}-${article.title.toLowerCase().slice(0, 50)}`;
      
      if (!seen.has(key)) {
        seen.add(key);
        unique.push(article);
      }
    }

    return unique;
  }

  /**
   * Validate article data before processing
   */
  validateArticle(article: Partial<Article>): boolean {
    return !!(
      article.title && 
      article.title.length > 10 &&
      article.content && 
      article.content.length > 100 &&
      article.url &&
      article.source
    );
  }
}
