import logging
import re
import time
from urllib.parse import urlparse

from ddgs import DDGS


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
        Returns the best URL string or an empty string if nothing is found.
        """
        if not text or not text.strip():
            return ""

        try:
            # Clean text: remove newlines, extra spaces, and double quotes that would break the query
            clean_text = re.sub(r'\s+', ' ', text).replace('"', '').strip()
            if not clean_text:
                return ""

            # Split into sentences to avoid searching a mix of original and copied text
            sentences = re.split(r'(?<=[.!?])\s+', clean_text)
            if not sentences:
                sentences = [clean_text]

            # Filter reasonably long sentences and sort by length descending
            valid_sentences = [s.strip() for s in sentences if len(s.split()) >= 8]
            if not valid_sentences:
                valid_sentences = [clean_text]

            valid_sentences.sort(key=lambda s: len(s.split()), reverse=True)

            # Check up to top 3 longest sentences
            best_candidate = None

            for idx, candidate_sentence in enumerate(valid_sentences[:3]):
                result = self._search_best_for_sentence(candidate_sentence)

                if result:
                    if best_candidate is None or result["score"] > best_candidate["score"]:
                        best_candidate = result

                # Be nice to DDG between sentences
                if idx < len(valid_sentences[:3]) - 1:
                    time.sleep(1)

            if best_candidate:
                logging.info(
                    f"[SourceSearcher] Best source found: {best_candidate['url']} "
                    f"(score={best_candidate['score']:.2f}, hits={best_candidate['hits']})"
                )
                return best_candidate["url"]

        except Exception as e:
            logging.error(f"[SourceSearcher] Search failed: {e}")

        return ""

    # -------------------------
    # Internal helpers
    # -------------------------

    def _search_best_for_sentence(self, sentence: str, per_query_results: int = 5):
        """
        Search a single sentence using multiple query variants and return the best candidate dict:
        {
            "url": str,
            "title": str,
            "body": str,
            "score": float,
            "hits": int,
            "best_rank": int,
            "queries": [...]
        }
        """
        if not sentence or len(sentence.split()) < 5:
            return None

        queries = self._build_queries(sentence)
        if not queries:
            return None

        candidates = {}
        normalized_sentence = self._normalize_text(sentence)

        for q_idx, query in enumerate(queries):
            try:
                results = list(self.ddgs.text(query, max_results=per_query_results))
            except Exception as e:
                logging.warning(f"[SourceSearcher] DDGS query failed for '{query}': {e}")
                results = []

            for rank, res in enumerate(results, start=1):
                url = res.get("href", "")
                if not url:
                    continue

                norm_url = self._normalize_url(url)
                if not norm_url:
                    continue

                title = res.get("title", "") or ""
                body = res.get("body", "") or ""
                # Hard validation: skip DDG fallback garbage that doesn't really match the sentence
                if not self._is_valid_candidate(sentence, title, body):
                    continue

                if norm_url not in candidates:
                    candidates[norm_url] = {
                        "url": url,
                        "title": title,
                        "body": body,
                        "score": 0.0,
                        "hits": 0,
                        "best_rank": rank,
                        "queries": [],
                    }

                # Scoring:
                # - appearing in a query matters a lot
                # - better rank matters
                # - trusted domains get a small bonus
                # - textual snippet overlap gets bonus
                rank_score = max(0, (per_query_results + 1 - rank))  # rank1=5, rank2=4...
                query_presence_score = 4.0
                trust_score = self._domain_bonus(url)
                snippet_score = self._snippet_match_score(sentence, title, body)

                total_add = query_presence_score + rank_score + trust_score + snippet_score

                candidates[norm_url]["score"] += total_add
                candidates[norm_url]["hits"] += 1
                candidates[norm_url]["best_rank"] = min(candidates[norm_url]["best_rank"], rank)
                candidates[norm_url]["queries"].append(query)

            # Small delay between queries to avoid rate limiting
            if q_idx < len(queries) - 1:
                time.sleep(1)

        if not candidates:
            return None

        # Optional hard filter: remove obviously weak candidates if no snippet overlap at all
        filtered_candidates = []
        for c in candidates.values():
            combined = f"{c['title']} {c['body']}"
            overlap = self._quick_overlap_score(normalized_sentence, self._normalize_text(combined))
            if overlap > 0:
                filtered_candidates.append(c)

        if filtered_candidates:
            ranked = sorted(
                filtered_candidates,
                key=lambda x: (x["score"], x["hits"], -x["best_rank"]),
                reverse=True
            )
        else:
            ranked = sorted(
                candidates.values(),
                key=lambda x: (x["score"], x["hits"], -x["best_rank"]),
                reverse=True
            )

        return ranked[0]

    def _build_queries(self, sentence: str):
        """
        Build multiple query variants for a sentence:
        - full exact sentence
        - first 12 words exact
        - middle 12 words exact
        - last 12 words exact
        - first 15 words without quotes
        """
        words = sentence.split()
        queries = []

        # 1) full exact sentence
        queries.append(f'"{sentence}"')

        # 2) first subphrase
        if len(words) >= 12:
            queries.append(f'"{" ".join(words[:12])}"')

        # 3) middle subphrase
        if len(words) >= 18:
            mid_start = max(0, len(words) // 2 - 6)
            queries.append(f'"{" ".join(words[mid_start:mid_start + 12])}"')

        # 4) last subphrase
        if len(words) >= 12:
            queries.append(f'"{" ".join(words[-12:])}"')

        # 5) keyword-style fallback without quotes
        queries.append(" ".join(words[:15]))

        # Remove duplicates while preserving order
        seen = set()
        final_queries = []
        for q in queries:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                final_queries.append(q)

        return final_queries

    def _normalize_text(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r'[^\w\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _normalize_url(self, url: str) -> str:
        if not url:
            return ""

        try:
            parsed = urlparse(url)
            scheme = parsed.scheme.lower() if parsed.scheme else "https"
            netloc = parsed.netloc.lower().replace("www.", "")
            path = parsed.path.rstrip("/")
            return f"{scheme}://{netloc}{path}"
        except Exception:
            return url.strip().lower()

    def _get_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            return ""

    def _domain_bonus(self, url: str) -> float:
        """
        Small bonus for trusted/high-authority domains.
        Keep it small so it doesn't overpower real matching.
        """
        domain = self._get_domain(url)

        trusted = [
            "ibm.com",
            "microsoft.com",
            "learn.microsoft.com",
            "aws.amazon.com",
            "cloud.google.com",
            "developer.mozilla.org",
            "wikipedia.org",
            "openai.com",
        ]

        for d in trusted:
            if domain.endswith(d):
                return 1.5

        return 0.0

    def _quick_overlap_score(self, normalized_sentence: str, normalized_candidate_text: str) -> int:
        """
        Returns how many 4-grams from the sentence appear in title/body.
        """
        sentence_words = normalized_sentence.split()
        if len(sentence_words) < 4:
            return 0

        n = min(4, len(sentence_words))
        ngrams = [
            " ".join(sentence_words[i:i+n])
            for i in range(len(sentence_words) - n + 1)
        ]

        hits = 0
        for ng in ngrams:
            if ng in normalized_candidate_text:
                hits += 1

        return hits

    def _snippet_match_score(self, sentence: str, title: str, body: str) -> float:
        """
        Score based on snippet overlap between target sentence and DDG title/body.
        This helps reject DDG fallback garbage when exact query fails.
        """
        normalized_sentence = self._normalize_text(sentence)
        normalized_candidate = self._normalize_text(f"{title} {body}")

        overlap_hits = self._quick_overlap_score(normalized_sentence, normalized_candidate)

        # Cap to avoid huge overweighting
        return min(overlap_hits * 1.5, 6.0)