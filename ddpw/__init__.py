from .platform import Device, Platform
from .wrapper import Wrapper, wrapper


def __get_version() -> str:
    r"""Retrieve the packaged version number."""
    from importlib.metadata import metadata as __md

    return __md(__name__)["Version"]


def __get_licence() -> str:
    r"""Retrieve contents of the licence file in the "binary" distribution."""

    from importlib.metadata import metadata as __md
    from importlib.resources import files as __res_files
    from os.path import exists, join

    inst_dir = __res_files(__name__)
    licence_file = __md(__name__)["License-File"]
    dist_info_dir = f"{__name__}-{__version__}.dist-info"

    filename = join(inst_dir, "..", dist_info_dir, "licenses", licence_file)
    if exists(filename):
        return open(filename, "r").read()


__version__: str = __get_version()
__licence__: str = __get_licence()
