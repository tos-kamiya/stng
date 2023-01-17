from typing import List, Tuple
from dataclasses import dataclass

import sys

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from .iter_funcs import ranges_overwrapping

ANSI_ESCAPE_CLEAR_CUR_LINE = "\x1b[1K\n\x1b[1A"

Vec = npt.NDArray
Pos = Tuple[int, int]
SLPLD = Tuple[float, int, Pos, List[str], str]  # similarity, length, pos, lines, document file


@dataclass(frozen=False)
class DoneDocFileInfo:
    w_para: int = 0
    wo_para: int = 0

    def __iadd__(self, di):
        self.w_para += di.w_para
        self.wo_para += di.wo_para
        return self


def prune_overlapped_paragraphs(slplds: List[SLPLD]) -> List[SLPLD]:
    if not slplds:
        return slplds
    dropped_index_set = set()
    for i, (slppd1, slppd2) in enumerate(zip(slplds, slplds[1:])):
        ip1, sr1 = slppd1[0], slppd1[2]
        ip2, sr2 = slppd2[0], slppd2[2]
        if ranges_overwrapping(sr1, sr2):
            if ip1 < ip2:
                dropped_index_set.add(i)
            else:
                dropped_index_set.add(i + 1)
    return [ipsrls for i, ipsrls in enumerate(slplds) if i not in dropped_index_set]


def excerpt_text(query_vec: Vec, lines: List[str], model: SentenceTransformer, length_to_excerpt: int) -> str:
    if not lines:
        return ""

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
        vec = model.encode("\n".join(lines[p:q]))
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


def trim_search_results(search_results: List[SLPLD], top_k: int):
    search_results.sort(reverse=True)
    del search_results[top_k:]


def print_intermediate_search_result(search_results: List[SLPLD], done_info: DoneDocFileInfo, elapsed_time: float):
    if search_results:
        done_files = done_info.w_para + done_info.wo_para
        sim, para_len, pos, _para, df = search_results[0]
        print(
            "%s[Info] %d docs done in %.0fs, %.2f docs/s. cur top-1: %.4f %d %s:%d-%d"
            % (
                ANSI_ESCAPE_CLEAR_CUR_LINE,
                done_files,
                elapsed_time,
                done_files / elapsed_time,
                sim,
                para_len,
                df,
                pos[0] + 1,
                pos[1],
            ),
            end="",
            file=sys.stderr,
            flush=True,
        )
