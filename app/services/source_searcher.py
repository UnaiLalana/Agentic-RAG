import logging
from duckduckgo_search import DDGS

class SourceSearcher:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SourceSearcher, cls).__new__(cls)
            cls._instance.ddgs = DDGS()
        return cls._instance

    def find_source(self, text: str) -> str:
        """
        Searches the web for snippets matching the text to find a likely source URL.
        Returns the URL string or an empty string if nothing is found.
        """
        if not text.strip():
            return ""
            
        try:
            # Extract a meaningful query portion (e.g., first 300 characters)
            query = text[:300].replace('\n', ' ').strip()
            # DuckDuckGo query might fail if it's too long or weird characters
            results = list(self.ddgs.text(query, max_results=1))
            
            if results and len(results) > 0:
                return results[0].get('href', "")
        except Exception as e:
            logging.error(f"[SourceSearcher] Search failed for query '{text[:30]}...': {e}")
            
        return ""
