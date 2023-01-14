from typing import List


def includes_all_texts(lines: List[str], texts: List[str]) -> bool:
    remaining_text_set = set(texts)
    for L in lines:
        for text in list(remaining_text_set):
            if L.find(text) >= 0:
                remaining_text_set.discard(text)
                if not remaining_text_set:
                    return True
    return False


def includes_any_of_texts(lines: List[str], texts: List[str]) -> bool:
    for L in lines:
        for text in texts:
            if L.find(text) >= 0:
                return True
    return False


# def split_posi_nega_words(raw_words: Iterable[str]) -> Tuple[List[str], List[str]]:
#     posi_raw_words = []
#     nega_raw_words = []
#     for rw in raw_words:
#         if rw.startswith("-"):
#             w = rw[1:]
#             if w:
#                 nega_raw_words.append(rw[1:])
#         elif rw.startswith("+"):
#             w = rw[1:]
#             if w:
#                 posi_raw_words.append(rw[1:])
#         else:
#             if rw:
#                 posi_raw_words.append(rw)
#     return posi_raw_words, nega_raw_words
