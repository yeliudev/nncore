# Copyright (c) Ye Liu. Licensed under the MIT License.

import requests

from .path import dir_name, is_file, mkdir, remove
from .progress import ProgressBar


def download(url,
             out_path=None,
             retries=0,
             overwrite=True,
             progress=False,
             chunk_size=1024,
             **kwargs):
    """
    Download the file from an url.

    Args:
        url (str): The url to the file.
        out_path (str | None, optional): The path to output file. If not
            specified, the file will be saved in the current working directory.
            Default: ``None``.
        retries (int, optional): The maximum number of retries after failure.
            Default: ``0``.
        overwrite (bool, optional): Whether to overwrite it if the file exists.
            Default: ``True``.
        progress (bool, optional): Whether to display the progress bar.
            Default: ``False``.
        chunk_size (int, optional): The chunk size for downloading with
            progress bar. Default: ``1024``.

    Returns:
        success (bool): Whether the download is successful.
    """
    if out_path is None:
        out_path = url.split('/')[-1]
    else:
        mkdir(dir_name(out_path))

    if is_file(out_path):
        if overwrite:
            remove(out_path)
        else:
            raise FileExistsError("file '{}' exists".format(out_path))

    success = False
    for _ in range(retries + 1):
        try:
            if progress:
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
            else:
                with open(out_path, 'wb') as f:
                    f.write(requests.get(url, **kwargs).content)
            success = True
            break
        except Exception:
            remove(out_path)

    return success
