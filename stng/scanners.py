from typing import Generator, List, Optional, TextIO

from contextlib import contextmanager
import os
import platform
import re
import subprocess
import unicodedata
import uuid


_script_dir: str = os.path.dirname(os.path.realpath(__file__))


@contextmanager
def open_file(filename: str, mode: str = "r") -> Generator[TextIO, None, None]:
    with open(filename, mode, encoding="utf-8", errors="replace") as fp:
        yield fp


_ja_nkf_abspath: Optional[str] = None
if platform.system() == "Windows" and os.path.exists(os.path.join(_script_dir, "nkf32.exe")):
    _ja_nkf_abspath = os.path.abspath(os.path.join(_script_dir, "nkf32.exe"))


if _ja_nkf_abspath:

    def read_text_file(file_name: str) -> str:
        b = subprocess.check_output([_ja_nkf_abspath, "-Lu", "--oc=UTF-8", file_name])
        return b.decode("utf-8")

else:

    def read_text_file(file_name: str) -> str:
        with open_file(file_name) as inp:
            return inp.read()


class ScanError(Exception):
    pass

class ScanErrorNotFile(ScanError):
    pass

class Scanner:
    def scan(self, file_name: str) -> List[str]:
        try:
            text = self._scan_i(file_name)
        except FileNotFoundError as e:
            raise e
        except ScanError as e:
            raise e
        except Exception as e:
            raise ScanError("ScanError: in reading file: %s" % repr(file_name)) from e
        return self.to_lines(text)

    def to_lines(self, text: str) -> List[str]:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x20\x7f-\x9f]+", " ", text)
        return [L.strip() for L in text.split("\n")]

    def _scan_i(self, file_name: str) -> str:
        assert file_name != "-"

        i = file_name.rfind(".")
        if i < 0:
            if os.path.isdir(file_name):
                raise ScanErrorNotFile
            raise ScanError("ScanError: file has NO extension: %s" % repr(file_name))

        if not os.path.exists(file_name):
            raise FileNotFoundError("Error: file not found: %s" % file_name)

        extension = file_name[i:].lower()

        if extension in [".html", "htm"]:
            return html_scan(file_name)
        elif extension == ".pdf":
            return pdf_scan(file_name)
        elif extension == ".docx":
            return docx_scan(file_name)
        else:
            return read_text_file(file_name)


if platform.system() != "Windows":

    def pdf_scan(file_name: str) -> str:
        try:
            import pdftotext
        except ImportError:
            raise ScanError("Error: pdftotext is not installed.")

        try:
            with open(file_name, "rb") as f:
                pdf = pdftotext.PDF(f)
        except pdftotext.Error as e:
            raise ScanError("ScanError: %s, file: %s" % (str(e), repr(file_name)))

        page_texts = [page for page in pdf]
        text = "".join(page_texts)
        # text = re.sub(r'(cid:\d+)', '', text)  # remove unknown glyphs

        return text

else:
    import tempfile

    _system_temp_dir = tempfile.gettempdir()

    def pdf_scan(file_name: str) -> str:
        tempf = os.path.join(_system_temp_dir, "%s.txt" % str(uuid.uuid4()))
        try:
            cmd = ["pdftotext.exe", file_name, tempf]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise ScanError("Error: pdftotext is not installed.")
        else:
            if p.returncode != 0:
                raise ScanError("ScanError: %s, file: %s, (%s)" % (p.stderr.decode("utf-8").rstrip(), repr(file_name), p.stdout))
            with open_file(tempf) as f:
                text = f.read()
            return text
        finally:
            if os.path.exists(tempf):
                os.remove(tempf)


def html_scan(file_name: str) -> str:
    import html2text

    with open_file(file_name) as inp:
        html_doc = inp.read()
    text = html2text.html2text(html_doc)
    return text


def docx_scan(file_name: str) -> str:
    import docx2txt
    import zipfile

    try:
        return docx2txt.process(file_name)
    except zipfile.BadZipFile:
        raise ScanError("ScanError: encrypted or corrupted .docx file: %s" % repr(file_name))
