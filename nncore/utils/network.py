# Copyright (c) Ye Liu. Licensed under the MIT License.

import requests

from .path import dir_name, is_file, join, mkdir, remove
from .progress import ProgressBar


def download(url,
             out_path=None,
             out_dir=None,
             retries=3,
             exist='skip',
             verbose=True,
             chunk_size=1024,
             **kwargs):
    """
    Download file from an url.

    Args:
        url (str): The url to the file.
        out_path (str | None, optional): The path to output file. If not
            specified, the file will be saved in the current working directory.
            Default: ``None``.
        out_dir (str | None, optional): The directory of output file. This can
            be specified only when ``out_path`` is set to ``None``. Default:
            ``None``.
        retries (int, optional): The maximum number of retries after failure.
            Default: ``3``.
        exist (str, optional): What to do if file exists. Expected values are
            ``'skip'``, ``'overwrite'``, or ``None``. Default: ``'skip'``.
        verbose (bool, optional): Whether to use verbose mode. Default:
            ``True``.
        chunk_size (int, optional): The chunk size for downloading with
            progress bar. Default: ``1024``.

    Returns:
        str | None: The path to download file or None.
    """
    assert exist in ('skip', 'overwrite', None)

    if out_dir is not None:
        assert out_path is None, 'out_path should be None when setting out_dir'
        out_path = join(out_dir, url.split('/')[-1])

    if out_path is None:
        out_path = url.split('/')[-1]

    mkdir(dir_name(out_path))

    if is_file(out_path):
        if exist == 'skip':
            if verbose:
                print(f'File {out_path} already downloaded')
            return
        elif exist == 'overwrite':
            remove(out_path)
        else:
            raise FileExistsError("file '{}' exists".format(out_path))

    success = False
    for i in range(retries + 1):
        try:
            if verbose:
                print(f'Downloading from {url}')
                r = requests.get(url, stream=True)
                with open(out_path, 'wb') as f:
                    total = int(r.headers.get('content-length'))
                    for chunk in ProgressBar(
                            r.iter_content(chunk_size=chunk_size),
                            num_tasks=int(total / chunk_size) + 1,
                            percentage=True):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                print(f'Saved to {out_path}')
            else:
                with open(out_path, 'wb') as f:
                    f.write(requests.get(url, **kwargs).content)
            success = True
            break
        except Exception as e:
            if verbose and retries > 0:
                print(f'Error: {e}. Retrying [{i + 1}/{retries}]')
            remove(out_path)

    out = out_path if success else None
    return out
