"""MuJoCo model loading that tolerates non-ASCII (e.g. Chinese) install paths.

mj.MjModel.from_xml_path calls a C++ loader whose fopen() uses the ANSI codepage
on Windows, so it cannot open paths like ".../大二上/课题组项目/...". We read the
files with Python (Unicode safe) and feed them through from_xml_string with a
virtual asset dict.

Two constraints shape the implementation:

* from_xml_string has no notion of a base directory, so every file the XML can
  reference must be pre-registered.
* MuJoCo's virtual filesystem is a *flat, case-insensitive* namespace keyed by
  bare filename -- registering both "meshes/x.STL" and "x.STL" raises
  "Repeated file name in assets dict". So we register exactly one file per
  basename, and only for assets the XML actually references. That also matters
  for stanford_toddy, which ships unused alternate assemblies full of
  same-named meshes.

Asset paths follow MJCF resolution rules: <mesh> against meshdir, <texture> and
<hfield> against texturedir (defaulting to meshdir), <include> against the
including file's own directory.
"""

import pathlib
import re
import xml.etree.ElementTree as ET

import mujoco as mj

_ATTR = re.compile(r'\b(\w+)\s*=\s*"([^"]*)"')


def _compiler_dirs(xml_path: pathlib.Path, xml_text: str):
    """(base, meshdir, texturedir) for a file, all absolute."""
    match = re.search(r"<compiler\b[^>]*/?>", xml_text)
    attrs = dict(_ATTR.findall(match.group(0))) if match else {}

    base = xml_path.parent
    meshdir = (base / attrs["meshdir"]).resolve() if attrs.get("meshdir") else base
    # MJCF: texturedir defaults to meshdir when not given.
    texturedir = (base / attrs["texturedir"]).resolve() if attrs.get("texturedir") else meshdir
    return base, meshdir, texturedir


def _referenced_files(xml_path: pathlib.Path, _seen=None):
    """Yield candidate paths for every file the XML references, includes included.

    Resolution candidates are yielded in MJCF preference order; callers keep the
    first one that exists.
    """
    _seen = set() if _seen is None else _seen
    xml_path = xml_path.resolve()
    if xml_path in _seen:
        return
    _seen.add(xml_path)

    base, meshdir, texturedir = _compiler_dirs(xml_path, xml_path.read_text(encoding="utf-8"))
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return  # not well-formed; let MuJoCo report that itself

    for elem in root.iter():
        ref = elem.get("file")
        if ref:
            tag = elem.tag.lower()
            if tag == "mesh":
                candidates = [meshdir / ref, base / ref]
            elif tag in ("texture", "hfield"):
                candidates = [texturedir / ref, meshdir / ref, base / ref]
            else:
                candidates = [base / ref]  # includes and anything else
            yield from candidates

        if elem.tag.lower() == "include" and ref:
            yield from _referenced_files(base / ref, _seen)


def load_model(xml_path) -> mj.MjModel:
    xml_path = pathlib.Path(xml_path).resolve()
    xml_text = xml_path.read_text(encoding="utf-8")

    # Keyed by lowercased basename because that is MuJoCo's own VFS key. First
    # existing candidate wins, so the meshdir-resolution beats a stray same-named
    # file elsewhere in the tree.
    assets = {}
    for f in _referenced_files(xml_path):
        if f.is_file() and f.name.lower() not in assets:
            assets[f.name.lower()] = f.read_bytes()

    return mj.MjModel.from_xml_string(xml_text, assets)
