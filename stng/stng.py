from typing import Iterable, Iterator, List, Optional, Tuple

from glob import iglob
import io
import importlib
from itertools import groupby
import os
import platform
import sys
from time import time

from docopt import docopt
from init_attrs_with_kwargs import InitAttrsWKwArgs
import numpy as np
from win_wildcard import expand_windows_wildcard, get_windows_shell
from sentence_transformers import SentenceTransformer

from .iter_funcs import sliding_window_iter
from .dum_scanner import dum_scan, dum_scan_it, to_lines
from .search_result import (
    ANSI_ESCAPE_CLEAR_CUR_LINE,
    SLPLD,
    excerpt_text,
    trim_search_results,
    print_intermediate_search_result,
    prune_overlapped_paragraphs,
)
from .search_result import DoneDocFileInfo, Vec


_script_dir = os.path.dirname(os.path.realpath(__file__))

# DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_MODEL = "sentence-transformers/stsb-xlm-r-multilingual"
VERSION = importlib.metadata.version("stng")
DEFAULT_TOP_K = 20
DEFAULT_WINDOW_SIZE = 20
DEFAULT_EXCERPT_CHARS = 80
DEFAULT_PREFER_LONGER_THAN = 80


class CLArgs(InitAttrsWKwArgs):
    query: Optional[str]
    query_file: Optional[str]
    file: List[str]
    verbose: bool
    model: str
    top_k: int
    paragraph_search: bool
    window: int
    excerpt_length: int
    quote: bool
    header: bool
    help: bool
    version: bool
    unix_wildcard: bool


__doc__: str = """Sentence-transformer-based Natural-language grep.

Usage:
  {stng} [options] [-t CHARS|-q] <query> <file>...
  {stng} [options] [-t CHARS|-q] -f QUERYFILE <file>...
  {stng} --help
  {stng} --version

Options:
  -v, --verbose                 Verbose.
  -m MODEL, --model=MODEL       Model name [default: {dm}].
  -k NUM, --top-k=NUM           Show top NUM files [default: {dtk}].
  -p, --paragraph-search        Search paragraphs in documents.
  -w NUM, --window=NUM          Line window size [default: {dws}].
  -f QUERYFILE, --query-file=QUERYFILE  Read query text from the file.
  -t CHARS, --excerpt-length=CHARS      Length of the text to be excerpted [default: {dec}].
  -q, --quote                   Show text instead of excerpt.
  -H, --header                  Print the header line.
  -u, --unix-wildcard           Use Unix-style pattern expansion on Windows.
""".format(
    dm=DEFAULT_MODEL,
    dtk=DEFAULT_TOP_K,
    dws=DEFAULT_WINDOW_SIZE,
    dplt=DEFAULT_PREFER_LONGER_THAN,
    dec=DEFAULT_EXCERPT_CHARS,
    stng="stng",
)


def do_extract_query_lines(query: Optional[str], query_file: Optional[str]) -> List[str]:
    if query == "-" or query_file == "-":
        query = sys.stdin.read()
    elif query_file is not None:
        _fn, err, text = dum_scan(query_file)
        if err is not None:
            sys.exit("Error in reading query file: %s" % err)
        query = text
    elif query is not None:
        pass
    else:
        assert False, "both query and query_file are None"
    assert query is not None
    lines = to_lines(query)
    return lines


def expand_file_iter(target_files: Iterable[str], windows_style: bool = False) -> Iterator[str]:
    if windows_style and get_windows_shell() is not None:
        for f in target_files:
            if f == "-":
                for L in sys.stdin:
                    L = L.rstrip()
                    yield L
            else:
                for gf in expand_windows_wildcard(f):
                    if os.path.isfile(gf):
                        yield gf
    else:
        for f in target_files:
            if f == "-":
                for L in sys.stdin:
                    L = L.rstrip()
                    yield L
            elif "*" in f:
                for gf in iglob(f, recursive=True):
                    if os.path.isfile(gf):
                        yield gf
            else:
                yield f


DF_POS_LINES = Tuple[str, Tuple[int, int], List[str]]


def calc_paras_similarity(
    search_results: List[SLPLD],
    query_vec: Vec,
    df_pos_lines_it: Iterable[DF_POS_LINES],
    model: SentenceTransformer,
    paragraph_search: bool,
    top_k: int,
) -> None:
    # for each paragraph in the file, calculate the similarity to the query
    para_texts = ["\n".join(lines[pos[0] : pos[1]]) for _df, pos, lines in df_pos_lines_it]
    para_vecs = model.encode(para_texts)

    i = -1
    for df, g in groupby(df_pos_lines_it, key=lambda dpp: dpp[0]):
        slplds: List[SLPLD] = []
        for _df, pos, lines in g:
            i += 1
            para_vec = para_vecs[i]
            sim = np.inner(query_vec, para_vec)
            para_len = sum(len(L) for L in lines[pos[0] : pos[1]])
            slplds.append((sim, para_len, pos, lines, df))

        # pick up paragraphs for the file
        if paragraph_search:
            slplds = prune_overlapped_paragraphs(slplds)  # remove paragraphs that overlap
            slplds.sort(reverse=True)
            del slplds[top_k:]
        else:
            slplds = [max(slplds)]  # extract only the most similar paragraphs in the file

        # update search results
        if len(search_results) < top_k or slplds[0] > search_results[-1]:
            search_results.extend(slplds)
            if len(search_results) >= top_k:
                trim_search_results(search_results, top_k)


def chunked_para_iter(doc_file_it: Iterable[str], window: int, chunk_size: int) -> Iterator[Tuple[List[DF_POS_LINES], DoneDocFileInfo]]:
    df_pos_lines: List[DF_POS_LINES] = []
    df_info: DoneDocFileInfo = DoneDocFileInfo()
    for df, err, text in dum_scan_it(doc_file_it):
        if err is not None:
            print(ANSI_ESCAPE_CLEAR_CUR_LINE + "[Warning] reading file %s: %s" % (df, err), file=sys.stderr, flush=True)
            df_info.wo_para += 1
            continue

        # extract paragraphs
        assert text is not None
        len_df_pos_lines = len(df_pos_lines)
        lines = to_lines(text)
        for pos in sliding_window_iter(len(lines), window):
            df_pos_lines.append((df, pos, lines))

        if len(df_pos_lines) > len_df_pos_lines:
            df_info.w_para += 1
        else:
            df_info.wo_para += 1

        if len(df_pos_lines) >= chunk_size:
            yield df_pos_lines, df_info
            df_pos_lines, df_info = [], DoneDocFileInfo()
    else:
        if df_pos_lines:
            yield df_pos_lines, df_info
            df_pos_lines, df_info = [], DoneDocFileInfo()


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding=sys.stderr.encoding, errors="replace")

    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--expand-wildcard":
            file_pats = argv[i + 1 :]
            for fp in file_pats:
                print("%s:" % fp)
                for f in expand_file_iter([fp], windows_style=True):
                    print("    %s" % f)
            return

    # A charm to make ANSI escape sequences work on Windows
    if platform.system() == "Windows":
        import colorama

        colorama.init()

    # command-line analysis
    raw_args = docopt(__doc__, argv=argv, version="dvg %s" % VERSION)
    a = CLArgs(_cast_str_values=True, **raw_args)

    model = SentenceTransformer(a.model)

    lines = do_extract_query_lines(a.query, a.query_file)
    query_vec = model.encode(["\n".join(lines)])

    chunk_size = 500

    # search for document files that are similar to the query
    if a.verbose:
        print("", end="", file=sys.stderr, flush=True)
    search_results: List[SLPLD] = []
    df_info: DoneDocFileInfo = DoneDocFileInfo()
    t0 = time()
    try:
        for df_pos_lines, di in chunked_para_iter(expand_file_iter(a.file), a.window, chunk_size):
            calc_paras_similarity(search_results, query_vec, df_pos_lines, model, a.paragraph_search, a.top_k)
            df_info += di
            if a.verbose:
                print_intermediate_search_result(search_results, df_info, time() - t0)
    except FileNotFoundError as e:
        if a.verbose:
            print(ANSI_ESCAPE_CLEAR_CUR_LINE, file=sys.stderr, flush=True)
        sys.exit(str(e))
    except KeyboardInterrupt:
        if a.verbose:
            print(
                ANSI_ESCAPE_CLEAR_CUR_LINE
                + "[Warning] Interrupted. Shows the search results up to now.\n",
                file=sys.stderr,
                flush=True,
            )
    if a.verbose:
        print(
            ANSI_ESCAPE_CLEAR_CUR_LINE
            + "[Info] document files with paragraphs: %d\n" % df_info.w_para
            + "[Info] skipped document files (from which no paragraph extracted): %d\n" % df_info.wo_para,
            file=sys.stderr,
            flush=True,
        )

    # output search results
    trim_search_results(search_results, a.top_k)
    if a.header:
        print("\t".join(["sim", "chars", "location", "text"]))
    for sim, para_len, (b, e), lines, df in search_results:
        para = lines[b:e]
        if a.quote:
            print("%.4f\t%d\t%s:%d-%d" % (sim, para_len, df, b + 1, e))
            for L in para:
                print("> %s" % L)
            print()
        else:
            excerpt = excerpt_text(query_vec, para, model, a.excerpt_length)
            print("%.4f\t%d\t%s:%d-%d\t%s" % (sim, para_len, df, b + 1, e, excerpt))


if __name__ == "__main__":
    main()
