import json
import re
from collections import Counter

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from Document import docs

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

FIELDS = {
    "date":   r"(\d{4}-\d{2}-\d{2})",
    "amount": r"(\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
    "id":     r"((?:INV|REQ|ORD)-\d+)",
    "ticket": r"ticket #(\d+)",
    "phone":  r"(\+91[ -]?\d{5}[ -]?\d{5})",
}


def ask(prompt):
    try:
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        print("LLM error:", e)
        return ""


def same(a, b):
    # "+91-98765-11111" and "+91 9876511111" are the same value
    da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
    if len(da) >= 6 and da == db:
        return True
    return a.strip().lower() == b.strip().lower()


# 1. deterministic
def regex_find(text, pattern):
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else None


# 2. heuristic
def llm_find(text, field):
    reply = ask(
        f"Extract only the {field} from this text. "
        f"Reply with just the value on one line, or NONE if it is not there.\n\n{text}"
    )
    reply = reply.splitlines()[0].strip().strip('"') if reply else ""
    return None if reply.upper() in ("", "NONE") else reply


# 3. semantic
def llm_find_all(text):
    raw = ask(
        "Extract the concrete facts (names, dates, amounts, statuses, errors, actions) "
        'from the text below. Reply ONLY with a JSON list like '
        '[{"field": "...", "value": "..."}], or [] if none.\n\n' + text
    )
    try:
        items = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        return [(str(i["field"]).strip(), str(i["value"]).strip()) for i in items]
    except Exception:
        return []


def route(text):
    results = []

    for field, pattern in FIELDS.items():
        value = regex_find(text, pattern)
        if value:
            results.append((field, value, "deterministic"))
        elif re.search(rf"\b{field}\b", text, re.IGNORECASE):
            value = llm_find(text, field)
            if value:
                results.append((field, value, "heuristic"))

    msg = re.search(r"msg='([^']*)'", text)
    if msg or len(results) < 2:
        for field, value in llm_find_all(msg.group(1) if msg else text):
            # skip empty values and anything we already found
            if value and not any(same(value, v) for _, v, _ in results):
                results.append((field, value, "semantic"))

    return results


counts = Counter()
for doc_id, text in docs.items():
    for field, value, strategy in route(text):
        print(f"{doc_id} | {field} | {value} | {strategy}")
        counts[strategy] += 1

print(
    f"\n{counts['deterministic']} deterministic, {counts['heuristic']} heuristic, "
    f"{counts['semantic']} semantic out of {sum(counts.values())} total fields."
)
