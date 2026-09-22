import json
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from Document import docs  # documents come from Document.py

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# field types we are watching for
FIELDS = ["date", "amount", "id", "ticket", "phone", "priority", "assigned_to"]

# regex patterns (priority has no clean regex on purpose)
PATTERNS = {
    "date": r"\b\d{4}-\d{2}-\d{2}\b",
    "amount": r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
    "id": r"\b(?:INV|REQ|ORD)-\d+\b",
    "ticket": r"\bticket\s*(?:number|no\.?|id)?\s*[:#]?\s*#?(\d+)",
    "phone": r"\+91[\s-]?\d{5}[\s-]?\d{5}"
}

# labels that tell us the doc should contain that field
HINTS = {
    "date": r"\bdate\s*:",
    "amount": r"\b(?:amount|total)\s*:",
    "id": r"\b(?:id|invoice|ticket)\s*[:=]",
    "phone": r"\bphone\s*:",
    "priority": r"\bpriority\s*:",
    "ticket": r"\bticket\b",
    "assigned_to": r"\bassigned\s+to\b",
}


# 1. deterministic - pure regex, free
def deterministic_extract(text, field_type):
    pat = PATTERNS.get(field_type)
    if not pat:
        return None
    m = re.search(pat, text, re.IGNORECASE)
    if not m:
        return None
    # if the pattern has a group, return just that part
    return m.group(1) if m.groups() else m.group()


# 2. heuristic - one short llm call for a known field
def heuristic_extract(text, field_type):
    prompt = (
        f"Extract only the {field_type} from this text. "
        f"Reply with just the value, nothing else. If it is not there, reply NONE.\n\n"
        f"Text: {text}"
    )
    try:
        reply = llm.invoke(prompt).content.strip().strip('"')
    except Exception as e:
        print("heuristic failed:", e)
        return None
    if not reply or reply.upper() == "NONE":
        return None
    return reply


# 3. semantic - open ended, we don't know what fields exist
def semantic_extract(text):
    prompt = (
        "Extract the concrete facts from the text below: names, dates or times, "
        "amounts, statuses, topics, errors, actions taken. "
        "Skip filler words like 'someone' or 'mentioned'. "
        'Reply ONLY with a JSON list like [{"field": "...", "value": "..."}]. '
        "No explanation, no markdown. Reply [] if there are no facts.\n\n"
        f"Text: {text}"
    )
    try:
        raw = llm.invoke(prompt).content
    except Exception as e:
        print("semantic failed:", e)
        return []

    # pull out the json list even if the model adds extra text
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [(str(i.get("field")), str(i.get("value"))) for i in items if isinstance(i, dict)]


def normalize(value):
    # "+91-98765-11111" and "+91 9876511111" count as the same value
    v = value.lower()
    digits = re.sub(r"\D", "", v)
    return digits if len(digits) >= 6 else v.strip()


# 4. router
def route(text):
    results = []

    for field in FIELDS:
        value = deterministic_extract(text, field)
        if value:
            results.append((field, value, "deterministic"))
            continue

        # only ask the llm if the doc actually looks like it has this field
        if re.search(HINTS[field], text, re.IGNORECASE):
            value = heuristic_extract(text, field)
            if value:
                results.append((field, value, "heuristic"))

    # free text log message -> semantic on just the message
    msg = re.search(r"msg='([^']*)'", text)
    if msg:
        for field, value in semantic_extract(msg.group(1)):
            results.append((field, value, "semantic"))

    elif len(results) < 2:
        already = {normalize(v) for _, v, _ in results}
        for field, value in semantic_extract(text):
            if normalize(value) not in already:
                results.append((field, value, "semantic"))

    return results


counts = {"deterministic": 0, "heuristic": 0, "semantic": 0}

for doc_id, text in docs.items():
    for field, value, strategy_used in route(text):
        print(f"{doc_id} | {field} | {value} | {strategy_used}")
        counts[strategy_used] += 1

total = sum(counts.values())
print(
    f"\n{counts['deterministic']} deterministic, {counts['heuristic']} heuristic, "
    f"{counts['semantic']} semantic out of {total} total fields."
)