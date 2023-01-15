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
from .scanners import Scanner, ScanError, ScanErrorNotFile
from .search_result import ANSI_ESCAPE_CLEAR_CUR_LINE, SLPLD, excerpt_text, trim_search_results, print_intermediate_search_result, prune_overlapped_paragraphs
from .search_result import Vec


_script_dir = os.path.dirname(os.path.realpath(__file__))

# DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_MODEL = 'sentence-transformers/stsb-xlm-r-multilingual'
VERSION = importlib.metadata.version("stng")
DEFAULT_TOP_N = 20
DEFAULT_WINDOW_SIZE = 20
DEFAULT_EXCERPT_CHARS = 80
DEFAULT_PREFER_LONGER_THAN = 80


class CLArgs(InitAttrsWKwArgs):
    query: Optional[str]
    query_file: Optional[str]
    file: List[str]
    verbose: bool
    model: str
    top_n: int
    paragraph_search: bool
    window: int
    excerpt_length: int
    header: bool
    help: bool
    version: bool
    unix_wildcard: bool
    vv: bool


__doc__: str = """Sentence-transformer-based Natural-language grep.

Usage:
  stng [options]  <query> <file>...
  stng [options] -f QUERYFILE <file>...
  stng --help
  stng --version

Options:
  --verbose, -v                 Verbose.
  --model=MODEL, -m MODEL       Model name [default: {dm}].
  --top-n=NUM, -n NUM           Show top NUM files [default: {dtn}].
  --paragraph-search, -p        Search paragraphs in documents.
  --window=NUM, -w NUM          Line window size [default: {dws}].
  --query-file=QUERYFILE, -f QUERYFILE  Read query text from the file.
  --excerpt-length=CHARS, -t CHARS      Length of the text to be excerpted [default: {dec}].
  --header, -H                  Print the header line.
  --unix-wildcard, -u           Use Unix-style pattern expansion on Windows.
  --vv                          Show name of each input file (for debug).
""".format(
    dm=DEFAULT_MODEL, dtn=DEFAULT_TOP_N, dws=DEFAULT_WINDOW_SIZE, dplt=DEFAULT_PREFER_LONGER_THAN, dec=DEFAULT_EXCERPT_CHARS
)


def do_extract_query_lines(query: Optional[str], query_file: Optional[str]) -> List[str]:
    if query == '-' or query_file == '-':
        lines = sys.stdin.read().splitlines()
    elif query_file is not None:
        scanner = Scanner()
        try:
            lines = scanner.scan(query_file)
        except ScanError as e:
            sys.exit("Error in reading query file: %s" % e)
        finally:
            del scanner
    elif query is not None:
        scanner = Scanner()
        lines = scanner.to_lines(query)
    else:
        assert False, "both query and query_file are None"
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

def calc_paras_similarity(search_results: List[SLPLD], query_vec: Vec, df_pos_line_it: Iterable[DF_POS_LINES], model: SentenceTransformer, paragraph_search: bool, top_n: int) -> None:
    # for each paragraph in the file, calculate the similarity to the query
    para_texts = ['\n'.join(lines[pos[0] : pos[1]]) for _df, pos, lines in df_pos_line_it]
    para_vecs = model.encode(para_texts)

    i = -1
    for df, g in groupby(df_pos_line_it, key=lambda dpp: dpp[0]):
        slppds: List[SLPLD] = []
        for _df, pos, lines in g:
            i += 1
            para_vec = para_vecs[i]
            sim = np.inner(query_vec, para_vec)
            para_len = sum(len(L) for L in lines[pos[0] : pos[1]])
            slppds.append((sim, para_len, pos, lines, df))

        if not slppds:
            continue  # for df

        # pick up paragraphs for the file
        if paragraph_search:
            slppds = prune_overlapped_paragraphs(slppds)  # remove paragraphs that overlap
            slppds.sort(reverse=True)
            del slppds[top_n :]
        else:
            slppds = [max(slppds)]  # extract only the most similar paragraphs in the file

        # update search results
        if len(search_results) < top_n or slppds[0] > search_results[-1]:
            search_results.extend(slppds)
            if len(search_results) >= top_n:
                trim_search_results(search_results, top_n)


def chunked_para_iter(doc_file_it: Iterable[str], window: int, chunk_size: int, a_vv: bool) -> Iterator[Tuple[List[DF_POS_LINES], int]]:
    scanner = Scanner()

    df_pos_lines: List[DF_POS_LINES] = []
    df_count: int = 0
    for df in doc_file_it:
        if a_vv:
            print(ANSI_ESCAPE_CLEAR_CUR_LINE + "> reading: %s" % df, file=sys.stderr, flush=True)

        # read lines from document file
        try:
            lines = scanner.scan(df)
        except ScanErrorNotFile as e:
            continue
        except ScanError as e:
            print(ANSI_ESCAPE_CLEAR_CUR_LINE + "> Warning: %s" % e, file=sys.stderr, flush=True)
            continue

        # extract paragraphs
        for pos in sliding_window_iter(len(lines), window):
            df_pos_lines.append((df, pos, lines))

        df_count += 1

        if len(df_pos_lines) >= chunk_size:
            yield df_pos_lines, df_count
            df_pos_lines, df_count = [], 0
    else:
        if df_pos_lines:
            yield df_pos_lines, df_count
            df_pos_lines, df_count = [], 0


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding=sys.stderr.encoding, errors="replace")

    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--bin-dir":
            print(os.path.join(_script_dir, "bin"))
            return
        if a == "--model-dir":
            print(os.path.join(_script_dir, "models"))
            return
        if a == "--expand-wildcard":
            file_pats = argv[i+1:]
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
    query_vec = model.encode(['\n'.join(lines)])

    count_document_files = 0
    chunk_size = 500

    # search for document files that are similar to the query
    if a.verbose:
        print("", end="", file=sys.stderr, flush=True)
    search_results: List[SLPLD] = []
    t0 = time()
    try:
        for df_pos_lines, df_count in chunked_para_iter(expand_file_iter(a.file), a.window, chunk_size, a.vv):
            calc_paras_similarity(search_results, query_vec, df_pos_lines, model, a.paragraph_search, a.top_n)
            count_document_files += df_count
            if a.verbose:
                print_intermediate_search_result(search_results, count_document_files, time() - t0)
    except FileNotFoundError as e:
        if a.verbose:
            print(ANSI_ESCAPE_CLEAR_CUR_LINE, file=sys.stderr, flush=True)
        sys.exit(str(e))
    except KeyboardInterrupt:
        if a.verbose:
            print(ANSI_ESCAPE_CLEAR_CUR_LINE + "> Interrupted. Shows the search results up to now.\n" + "> number of document files: %d" % count_document_files, file=sys.stderr, flush=True)
    else:
        if a.verbose:
            print(ANSI_ESCAPE_CLEAR_CUR_LINE + "> number of document files: %d" % count_document_files, file=sys.stderr, flush=True)

    # output search results
    trim_search_results(search_results, a.top_n)
    if a.header:
        print("\t".join(["sim", "chars", "location", "text"]))
    for sim, para_len, (b, e), lines, df in search_results:
        para = lines[b:e]
        excerpt = excerpt_text(query_vec, para, model, a.excerpt_length)
        print("%.4f\t%d\t%s:%d-%d\t%s" % (sim, para_len, df, b + 1, e, excerpt))


if __name__ == "__main__":
    main()
