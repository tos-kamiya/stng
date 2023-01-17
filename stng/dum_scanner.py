from typing import Iterator, Iterable, List, Optional, Tuple

import os
import platform
import re
import subprocess
import threading
import uuid

import html2text


DOC_FILE_SIZE_LIMIT = 128 * 1024 * 1024  # 128M


def to_lines(text: str) -> List[str]:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x20\x7f-\x9f]+", " ", text)
    text = re.sub(r"[\u0020\u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000]+", " ", text)
    lines = [L.strip() for L in text.split("\n")]
    lines = [L for L in lines if L]
    return lines


class FileTooLarge(ValueError):
    pass

class ExecGetOutputThread(threading.Thread):
    def __init__(self, filename: str):
        super().__init__()
        self.filename: str = filename
        self.retval: Optional[Tuple[str, Optional[Exception], Optional[str]]] = None

    def run(self) -> None:
        fn = self.filename

        if not (os.path.exists(fn) and os.path.isfile(fn)):
            self.retval = (fn, FileNotFoundError(), None)
            return

        ext = os.path.splitext(fn)[1].lower()
        if ext == '.pdf':
            cmd = ['pdftotext', '-nopgbrk', '-q', fn, '-']
        elif ext in ['.docx', '.odt', '.epub']:
            cmd = ['pandoc', '--from=%s' % ext[1:], '--to=plain', '--wrap=none', '--markdown-headings=atx', '--quiet', fn]
        else:
            cmd = None
        try:
            if cmd is None:
                s = os.path.getsize(fn)
                if s >= DOC_FILE_SIZE_LIMIT:
                    self.retval = (fn, FileTooLarge(), None)
                    return
                with open(fn, 'rb') as inp:
                    b = inp.read()
            else:
                b = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            t = b.decode(encoding='UTF-8', errors='replace')

            ext = os.path.splitext(fn)[1].lower()
            if ext in ['.html', '.htm']:
                t = html2text.html2text(t)

            self.retval = (fn, None, t)
        except Exception as e:
            self.retval = (fn, e, None)


def dum_scan(filename: str) -> Tuple[str, Optional[Exception], Optional[str]]:
    th = ExecGetOutputThread(filename)
    th.start()
    th.join()
    assert th.retval is not None
    return th.retval


def dum_scan_it(filename_it: Iterable[str], max_workers: int = 12) -> Iterator[Tuple[str, Optional[Exception], Optional[str]]]:
    que = []
    for fn in filename_it:
        if len(que) >= max_workers:
            th = que.pop(0)
            th.join()
            yield th.retval
        th = ExecGetOutputThread(fn)
        th.start()
        que.append(th)
    while que:
        th = que.pop(0)
        th.join()
        yield th.retval


if __name__ == '__main__':
    import sys
    from glob import glob

    def glob_it(file_patterns):
        for fp in file_patterns:
            for fn in glob(fp, recursive=True):
                yield fn

    args = sys.argv[1:]
    if not args or args[0] in ['-h', '--help']:
        print("Usage: dum_scanner.py <file>...")
        sys.exit(0)

    for fn, err, text in dum_scan_it(glob_it(args)):
        print("%s:" % fn)
        if err is not None:
            print("Error in reading: %s, %s" % (fn, err), file=sys.stderr, flush=True)
        else:
            print(text)
        print()
