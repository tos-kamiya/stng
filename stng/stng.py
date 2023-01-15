from typing import Iterable, Iterator, List, Optional

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

from .iter_funcs import chunked_iter, sliding_window_iter
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
    workers: Optional[int]
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
  --workers=WORKERS, -j WORKERS         Worker process.
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


def find_similar_paragraphs(query_vec: Vec, doc_files: Iterable[str], model: SentenceTransformer, a: CLArgs) -> List[SLPLD]:
    scanner = Scanner()

    FLUSH_PARA_SIZE = 1000

    search_results: List[SLPLD] = []
    def flush_para(df_pos_paras):
        # for each paragraph in the file, calculate the similarity to the query
        para_texts = ['\n'.join(lines[pos[0] : pos[1]]) for _df, pos, lines in df_pos_paras]
        para_vecs = model.encode(para_texts)

        i = -1
        for df, g in groupby(df_pos_paras, key=lambda dpp: dpp[0]):
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
            if a.paragraph_search:
                slppds = prune_overlapped_paragraphs(slppds)  # remove paragraphs that overlap
                slppds.sort(reverse=True)
                del slppds[a.top_n :]
            else:
                slppds = [max(slppds)]  # extract only the most similar paragraphs in the file

            # update search results
            search_results.extend(slppds)
            if len(search_results) >= a.top_n:
                trim_search_results(search_results, a.top_n)

        assert i == len(df_pos_paras) - 1

    df_para_datas = []
    for df in doc_files:
        if a.vv:
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
        for pos in sliding_window_iter(len(lines), a.window):
            df_para_datas.append((df, pos, lines))

        if len(df_para_datas) >= FLUSH_PARA_SIZE:
            flush_para(df_para_datas)
            df_para_datas.clear()

    if df_para_datas:
        flush_para(df_para_datas)
        df_para_datas.clear()

    return search_results


def find_similar_paragraphs_i(arg_tuple):
    # (query_vec, dfs, model, a) = arg_tuple
    r = find_similar_paragraphs(*arg_tuple)
    return r, arg_tuple[0]


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
        for dfs in chunked_iter(expand_file_iter(a.file), chunk_size):
            search_results.extend(find_similar_paragraphs(query_vec, dfs, model, a))
            trim_search_results(search_results, a.top_n)
            count_document_files += len(dfs)
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
