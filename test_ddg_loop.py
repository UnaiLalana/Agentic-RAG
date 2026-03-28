from ddgs import DDGS
import time
from urllib.parse import urlparse


sentences = [
    "Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language.",
    "NLP enables computers and digital devices to recognize, understand and generate text and speech by combining computational linguistics, the rule-based modeling of human language together with statistical modeling, machine learning and deep learning.",
    "NLP research has helped enable the era of generative AI, from the communication skills of large language models (LLMs) to the ability of image generation models to understand requests. NLP is already part of everyday life for many, powering search engines, prompting chatbots for customer service with spoken commands, voice-operated GPS systems and question-answering digital assistants on smartphones such as Amazon’s Alexa, Apple’s Siri and Microsoft’s Cortana."
]


def normalize_url(url: str) -> str:
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


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def build_queries(sentence: str):
    words = sentence.split()
    queries = []

    # 1) frase exacta completa
    queries.append(f'"{sentence}"')

    # 2) primera subfrase
    if len(words) >= 12:
        queries.append(f'"{" ".join(words[:12])}"')

    # 3) subfrase central
    if len(words) >= 18:
        mid_start = max(0, len(words) // 2 - 6)
        queries.append(f'"{" ".join(words[mid_start:mid_start + 12])}"')

    # 4) última subfrase
    if len(words) >= 12:
        queries.append(f'"{" ".join(words[-12:])}"')

    # 5) keywords sin comillas
    queries.append(" ".join(words[:15]))

    # quitar duplicados preservando orden
    seen = set()
    final_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            final_queries.append(q)

    return final_queries


def domain_bonus(url: str) -> float:
    """
    Bonus opcional simple para dominios más 'fuertes' / probables.
    Puedes ajustar esto luego.
    """
    domain = get_domain(url)

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


def search_top1_for_sentence(ddgs, sentence: str, per_query_results: int = 5, sleep_s: float = 1.5):
    queries = build_queries(sentence)
    candidates = {}

    print("=" * 120)
    print(f"Sentence:\n{sentence}\n")
    print("Queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    print()

    for q_index, query in enumerate(queries):
        print(f"Searching query {q_index + 1}/{len(queries)}: {query}")

        try:
            results = list(ddgs.text(query, max_results=per_query_results))
            print(f"  Raw results: {len(results)}")

            for rank, r in enumerate(results, start=1):
                url = r.get("href")
                if not url:
                    continue

                norm_url = normalize_url(url)

                if norm_url not in candidates:
                    candidates[norm_url] = {
                        "url": url,
                        "title": r.get("title"),
                        "body": r.get("body"),
                        "score": 0.0,
                        "hits": 0,
                        "best_rank": rank,
                        "queries": [],
                    }

                # scoring:
                # - aparecer en una query suma bastante
                # - mejor ranking suma más
                # - dominio confiable suma un poco
                rank_score = max(0, (per_query_results + 1 - rank))  # rank1=5, rank2=4...
                query_presence_score = 4.0
                trust_score = domain_bonus(url)

                total_add = query_presence_score + rank_score + trust_score

                candidates[norm_url]["score"] += total_add
                candidates[norm_url]["hits"] += 1
                candidates[norm_url]["best_rank"] = min(candidates[norm_url]["best_rank"], rank)
                candidates[norm_url]["queries"].append(query)

        except Exception as e:
            print(f"  ERROR: {e}")

        time.sleep(sleep_s)

    if not candidates:
        print("\nNo candidates found.\n")
        return None

    ranked = sorted(
        candidates.values(),
        key=lambda x: (x["score"], x["hits"], -x["best_rank"]),
        reverse=True
    )

    top1 = ranked[0]

    print("\n" + "-" * 120)
    print("TOP 1 RESULT:")
    print(f"URL:   {top1['url']}")
    print(f"Title: {top1['title']}")
    print(f"Body:  {top1['body']}")
    print(f"Score: {top1['score']:.2f}")
    print(f"Hits:  {top1['hits']}")
    print(f"Best rank seen: {top1['best_rank']}")
    print("Matched by queries:")
    for q in list(dict.fromkeys(top1["queries"])):
        print(f"  - {q}")
    print()

    return top1


def main():
    with DDGS() as ddgs:
        for idx, sentence in enumerate(sentences, 1):
            print(f"\n\n########## SENTENCE {idx}/{len(sentences)} ##########\n")
            search_top1_for_sentence(ddgs, sentence, per_query_results=5, sleep_s=2.0)


if __name__ == "__main__":
    main()