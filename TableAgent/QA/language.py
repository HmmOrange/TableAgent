from __future__ import annotations

import re


_QUESTION_WRAPPER = re.compile(
    r"^\s*[❓?]*\s*(?:câu\s+hỏi|question|질문)\s*:\s*",
    flags=re.IGNORECASE,
)
_PARENTHETICAL_GLOSS = re.compile(r"\([^()]*\)")
_PARENTHETICAL_CONTENT = re.compile(r"\(([^()]*)\)")
_VIETNAMESE_MARKERS = re.compile(
    r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    flags=re.IGNORECASE,
)


def required_answer_language(question: str) -> str:
    """Resolve an explicit target language from the grammatical question text."""
    text = _QUESTION_WRAPPER.sub("", str(question or "").strip())
    text = _PARENTHETICAL_GLOSS.sub("", text)

    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean"
    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese"
    if _VIETNAMESE_MARKERS.search(text):
        return "Vietnamese"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"
    if re.search(r"[A-Za-z]", text):
        return "English"
    return "the primary grammatical language of the question"


def explicit_question_identifiers(question: str) -> tuple[str, ...]:
    """Return labels explicitly enumerated in parenthetical question text."""
    identifiers = []
    for content in _PARENTHETICAL_CONTENT.findall(str(question or "")):
        parts = [part.strip(" \t\r\n.:؛،") for part in re.split(r"[,;]", content)]
        if len(parts) < 2:
            continue
        for part in parts:
            if part and part not in identifiers:
                identifiers.append(part)
    return tuple(identifiers)


def answer_uses_required_language(answer: str, language: str, *, question: str = "") -> bool:
    """Reject prose dominated by a different script while allowing numeric answers."""
    text = str(answer or "")
    for identifier in explicit_question_identifiers(question):
        text = re.sub(re.escape(identifier), "", text, flags=re.IGNORECASE)
    hangul = sum("\uac00" <= character <= "\ud7af" for character in text)
    japanese = sum("\u3040" <= character <= "\u30ff" for character in text)
    chinese = sum("\u4e00" <= character <= "\u9fff" for character in text)
    latin = sum("a" <= character.casefold() <= "z" for character in text)
    alphabetic = hangul + japanese + chinese + latin
    if alphabetic < 5:
        return True

    if language == "Korean":
        return hangul / alphabetic >= 0.30
    if language == "Japanese":
        return (japanese + chinese) / alphabetic >= 0.30
    if language == "Chinese":
        return chinese / alphabetic >= 0.30
    if language == "Vietnamese":
        return bool(_VIETNAMESE_MARKERS.search(text)) and latin / alphabetic >= 0.50
    if language == "English":
        return latin / alphabetic >= 0.50
    return True


def ordinal_header_for_language(language: str) -> str:
    return {
        "Vietnamese": "STT",
        "English": "No.",
        "Korean": "번호",
        "Japanese": "番号",
        "Chinese": "序号",
    }.get(language, "No.")


def item_header_for_language(language: str) -> str:
    return {
        "Vietnamese": "Nội dung",
        "English": "Item",
        "Korean": "항목",
        "Japanese": "項目",
        "Chinese": "项目",
    }.get(language, "Item")
