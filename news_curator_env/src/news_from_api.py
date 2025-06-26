import requests

def collect_from_newsapi(api_key, query="", sources="", language="en"):
    url = "https://newsapi.org/v2/everything"
    
    params = {
        'apiKey': api_key,
        'q': query,
        'sources': sources,
        'language': language,
        'sortBy': 'publishedAt',
        'page': 1,
        'pageSize': 100
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        articles = []
        
        for article in data['articles']:
            articles.append({
                'title': article['title'],
                'content': article.get('content', ''),
                'description': article.get('description', ''),
                'url': article['url'],
                'source': article['source']['name'],
                'published': article['publishedAt'],
                'image_url': article.get('urlToImage')
            })
        
        return articles
    else:
        print(f"Error: {response.status_code}")
        return []