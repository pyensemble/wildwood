from collections import namedtuple
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext
import os
from os.path import dirname, exists, join
import errno


RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url", "checksum"])


def get_data_home(data_home=None) -> str:
    """Return the path of the wildwood data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data dir is set to a folder named 'wildwood_data' in the
    user home folder.

    Alternatively, it can be set by the 'WILDWOOD_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str, default=None
        The path to wildwood data directory. If `None`, the default path
        is `~/wildwood_data`.
    """
    if data_home is None:
        data_home = environ.get("WILDWOOD_DATA", join("~", "wildwood_data"))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def _mkdirp(d):
    """Ensure directory d exists (like mkdir -p on Unix)
    No guarantee that the directory is writable.
    """
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
