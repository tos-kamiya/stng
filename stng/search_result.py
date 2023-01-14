from typing import Callable, List, Tuple

import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from .iter_funcs import ranges_overwrapping

ANSI_ESCAPE_CLEAR_CUR_LINE = "\x1b[1K\n\x1b[1A"

Vec = np.array
Pos = Tuple[int, int]
SLPPD = Tuple[float, int, Pos, List[str], str]  # similarity, length, pos, paragraph, document file


def prune_overlapped_paragraphs(slppds: List[SLPPD]) -> List[SLPPD]:
    if not slppds:
        return slppds
    dropped_index_set = set()
    for i, (slppd1, slppd2) in enumerate(zip(slppds, slppds[1:])):
        ip1, sr1 = slppd1[0], slppd1[2]
        ip2, sr2 = slppd2[0], slppd2[2]
        if ranges_overwrapping(sr1, sr2):
            if ip1 < ip2:
                dropped_index_set.add(i)
            else:
                dropped_index_set.add(i + 1)
    return [ipsrls for i, ipsrls in enumerate(slppds) if i not in dropped_index_set]


PAT_ZsPlus = re.compile('[\u00020\u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000]+')


def excerpt_text(query_vec: Vec, lines: List[str], model: SentenceTransformer, length_to_excerpt: int) -> str:
    if not lines:
        return ""

    lines = [re.sub(PAT_ZsPlus, ' ', L) for L in lines]

    if len(lines) == 1:
        return lines[0][:length_to_excerpt]

    len_lines = len(lines)
    max_sim_data = None
    for p in range(len_lines):
        para_textlen = len(lines[p])
        if para_textlen == 0:
            continue  # for p
        q = p + 1
        while q < len_lines and para_textlen < length_to_excerpt:
            para_textlen += len(lines[q])
            q += 1
        vec = model.encode('\n'.join(lines[p:q]))
        sim = np.inner(query_vec, vec)
        if max_sim_data is None or sim > max_sim_data[0]:
            max_sim_data = sim, (p, q)
        if q == len_lines:
            break  # for p
    assert max_sim_data is not None

    b, e = max_sim_data[1]
    excerpt = "|".join(lines[b:e])
    excerpt = excerpt[:length_to_excerpt]
    return excerpt


def trim_search_results(search_results: List[SLPPD], top_n: int):
    search_results.sort(reverse=True)
    del search_results[top_n:]


def print_intermediate_search_result(search_results: List[SLPPD], done_files: int, elapsed_time: float):
    if search_results:
        sim, para_len, pos, _para, df = search_results[0]
        print("%s[%d docs done in %.0fs, %.2f docs/s] cur top-1: %.4f %d %s:%d-%d" % (ANSI_ESCAPE_CLEAR_CUR_LINE, done_files, elapsed_time, done_files / elapsed_time, sim, para_len, df, pos[0] + 1, pos[1]), end="", file=sys.stderr, flush=True)
