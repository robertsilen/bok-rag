"""Claude API client for chat responses and query rewriting."""

import anthropic
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, SYSTEM_PROMPT

REWRITE_MODEL = "claude-haiku-4-5-20251001"

REWRITE_SYSTEM = (
    "Du hjälper till att söka i en bok (PDF). "
    "Givet en konversationshistorik och en ny fråga, skriv om frågan så att den "
    "innehåller alla relevanta namn, begrepp och nyckelord som behövs för att "
    "hitta rätt textstycken i boken — även sådant som bara nämndes i historiken. "
    "Frågan ska fungera fristående utan historiken. "
    "Svara ENBART med den omskrivna frågan, inget annat."
)


def rewrite_query(query, history):
    """Rewrite query using conversation history for better search.
    Returns the rewritten query, or the original if no history or on error.
    """
    if not history or not ANTHROPIC_API_KEY:
        return query

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    conv = ""
    for m in history[-6:]:  # last 3 exchanges max
        role = "Användare" if m["role"] == "user" else "Assistent"
        conv += f"{role}: {m['content'][:300]}\n"

    try:
        response = client.messages.create(
            model=REWRITE_MODEL,
            max_tokens=200,
            system=REWRITE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Historik:\n{conv}\nNy fråga: {query}",
            }],
        )
        rewritten = response.content[0].text.strip()
        return rewritten if rewritten else query
    except Exception:
        return query


def build_context(chunks):
    """Build context string from retrieved chunks."""
    parts = []
    for i, c in enumerate(chunks, 1):
        text = ""
        if c.get("context_before"):
            text += f"[...] {c['context_before']}\n\n"
        text += c["chunk_text"]
        if c.get("context_after"):
            text += f"\n\n{c['context_after']} [...]"
        parts.append(f"### Källa {i} (sida {c['page_number']})\n{text}")
    return "\n\n---\n\n".join(parts)



def chat(query, chunks, history=None):
    """Send query + context to Claude. Returns (response_text, page_numbers).
    Raises ValueError for API key issues, anthropic.RateLimitError for rate limits.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("Ogiltig eller saknad ANTHROPIC_API_KEY. Kontrollera din .env-fil.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    context = build_context(chunks)

    messages = []
    if history:
        messages.extend(history)

    user_content = f"## Kontext från boken\n\n{context}\n\n## Fråga\n\n{query}"
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
    except anthropic.AuthenticationError:
        raise ValueError("Ogiltig eller saknad ANTHROPIC_API_KEY. Kontrollera din .env-fil.")
    except anthropic.RateLimitError:
        raise ValueError("Rate limit nådd. Vänta en stund och försök igen.")

    answer = response.content[0].text
    page_numbers = sorted(set(c["page_number"] for c in chunks))
    return answer, page_numbers
