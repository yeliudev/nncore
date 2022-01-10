# Copyright (c) Ye Liu. All rights reserved.

from collections import OrderedDict
from math import ceil

import cv2

import nncore


@nncore.bind_getter('max_size')
class _Cache(object):

    def __init__(self, max_size):
        if max_size <= 0:
            raise ValueError('max_size must be a positive integer')

        self._max_size = max_size
        self._cache = OrderedDict()

    @property
    def size(self):
        return len(self._cache)

    def get(self, key, default=None):
        return self._cache.get(key, default)

    def set(self, key, value):
        if key in self._cache:
            return

        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = value


@nncore.bind_getter('vcap', 'width', 'height', 'fps', 'num_frames', 'fourcc',
                    'position')
class VideoReader(object):
    """
    A helper class for processing videos.

    This class provides convenient apis to access frames. There exists an
    issue of OpenCV's VideoCapture class that jumping to a certain frame may
    be inaccurate. It is fixed in this class by checking the position after
    jumping each time.

    Args:
        path (str): Path to the video.
        cache_size (int, optional): Maximum number of frames to cache. Default:
            ``16``.
    """

    def __init__(self, path, cache_size=16):
        if not path.startswith(('https://', 'http://')):
            nncore.is_file(path, raise_error=True)

        self._cache = _Cache(cache_size)
        self._vcap = cv2.VideoCapture(path)

        self._width = int(self._vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._vcap.get(cv2.CAP_PROP_FPS)
        self._num_frames = int(self._vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(cv2.CAP_PROP_FOURCC)
        self._position = 0

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self._vcap.release()

    def __len__(self):
        return self._num_frames

    def __next__(self):
        img = self.read()
        if img is None:
            raise StopIteration
        return img

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [
                self.get_frame(i)
                for i in range(*idx.indices(self._num_frames))
            ]

        if idx < 0:
            idx += self._num_frames
            if idx < 0:
                raise IndexError('index out of range')

        return self.get_frame(idx)

    def __iter__(self):
        self._set_position(0)
        return self

    @property
    def opened(self):
        return self._vcap.isOpened()

    @property
    def resolution(self):
        return (self._width, self._height)

    def _get_position(self):
        return int(round(self._vcap.get(cv2.CAP_PROP_POS_FRAMES)))

    def _set_position(self, idx):
        self._vcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        pos = self._get_position()
        for _ in range(idx - pos):
            self._vcap.read()
        self._position = idx

    def read(self):
        if self._cache:
            img = self._cache.get(self._position)
            if img is None:
                if self._position != self._get_position():
                    self._set_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    self._cache.set(self._position, img)
            else:
                ret = True
        else:
            ret, img = self._vcap.read()

        self._position += int(ret)
        return img

    def get_frame(self, idx=None):
        if idx is None:
            if self._position != 0:
                return self._cache.get(self._position - 1)
            else:
                return

        if idx < 0 or idx >= self._num_frames:
            raise IndexError(
                "'idx' must be between 0 and {}".format(self._num_frames - 1))

        if idx == self._position:
            return self.read()

        if self._cache:
            img = self._cache.get(idx)
            if img is not None:
                self._position = idx + 1
                return img

        self._set_position(idx)
        ret, img = self._vcap.read()

        if ret:
            self._cache.set(self._position, img)
            self._position += 1

        return img

    def dump_frames(self,
                    out_dir,
                    size=None,
                    scale=None,
                    interpolation='bilinear',
                    template='img_{:05d}.jpg',
                    interval=1,
                    start=0,
                    max_num=-1,
                    show_progress=False,
                    raise_error=True):
        """
        Dump the video to resized frame images.

        Args:
            out_dir (str): The output directory.
            size (tuple[int] | None, optional): The target frame size in the
                form of ``(width, height)``. Default: ``None``.
            scale (int | tuple[int] | None, optional): The scaling factor or
                the maximum size. If it is a number, the image will be
                rescaled by this factor. When it is a tuple containing 2
                numbers, the image will be rescaled as large as possible
                within the scale. In this case, ``-1`` means infinity. Default:
                ``None``.
            interpolation (str | int, optional): Interpolation method.
                Currently supported methods include ``nearest``, ``bilinear``,
                ``bicubic``, ``area``, and ``lanczos``. Default: ``bilinear``.
            template (str, optional): Filename template. Default:
                ``'img_{:05d}.jpg'``.
            interval (int, optional): The interval of dumped frames. Default:
                ``1``.
            start (int, optional): The starting frame index. Default: ``0``.
            max_num (int, optional): The maximum number of frames to be dumped.
                Default: ``-1``.
            show_progress (bool, optional): Whether to display the progress
                bar. Default: ``False``.
            raise_error (bool, optional): Whether to raise an error if a frame
                is not successfully decoded. Default: ``True``.
        """
        if max_num > 0:
            total_tasks = min(self._num_frames - start, max_num)
        else:
            total_tasks = self._num_frames - start

        if total_tasks <= 0:
            raise ValueError('start must be less than the total frame number')

        num_tasks = ceil(total_tasks / interval)

        if start > 0:
            self._set_position(start)

        prog_bar = nncore.ProgressBar(
            num_tasks=num_tasks, active=show_progress)

        for i in range(total_tasks):
            img = self.read()

            if i % interval != 0:
                continue

            if img is None:
                if raise_error:
                    raise ValueError(
                        'frame {} is not successfully decoded'.format(i))
                else:
                    prog_bar.update()
                    return

            if size is not None:
                img = nncore.imresize(img, size, interpolation=interpolation)

            if scale is not None:
                img = nncore.imrescale(img, scale, interpolation=interpolation)

            filename = nncore.join(out_dir, template.format(i))
            nncore.imwrite(img, filename)

            prog_bar.update()
