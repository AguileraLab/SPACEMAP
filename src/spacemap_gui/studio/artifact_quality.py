"""Versioned, tiled quality measurements for blur and fluorescent debris.

Per-channel extrema preserve a defect in one marker. Pooled measurements have
stable names across marker panels; absent/empty channels are not blurry tissue.
No viewer colour, visibility, or contrast setting enters these measurements.
"""
import numpy as np

SCHEMA = 'quality_v1'
METRICS = ('signal', 'contrast', 'edge_fine', 'edge_coarse', 'detail',
           'peak', 'ridge', 'local_mean', 'coarse_contrast')
WINDOWS_UM = (6., 18.)


def extractor(image, scale):
    # Lazy import keeps the legacy extractor independent of this schema.
    from .artifact_learning import Features

    class QualityFeatures(Features):
        feature_schema = SCHEMA
        tile_size = 256

        def __init__(self):
            super().__init__(image, scale)
            self.accelerator = None
            self.execution = {'device': 'cpu'}
            if len(self.he) == 3:
                self.names += ['quality:he:' + name for name in METRICS]
            if self.markers:
                self.names += ['quality:markers:' + mode + ':' + name
                               for mode in ('min', 'max', 'mean') for name in METRICS]
                self.names += ['quality:markers:signal_fraction']

        def enable_acceleration(self, device='auto'):
            from .artifact_sampling import backend
            self.accelerator, self.execution = backend(device)

        def tile(self, x, y, width, height):
            from scipy.ndimage import uniform_filter
            from .artifact_sampling import measurements as measure
            from skimage.color import rgb2hed
            fine, coarse = [max(3, int(round(um / (self.mpp * self.scale))) | 1) for um in WINDOWS_UM]
            # Derivatives then one box filter: radius + one pixel is sufficient.
            halo = coarse // 2 + 2
            x0, y0 = max(0, x - halo), max(0, y - halo)
            x1, y1 = min(self.w, x + width + halo), min(self.h, y + height + halo)
            crop = (slice(y - y0, y - y0 + height), slice(x - x0, x - x0 + width))
            result = np.empty((height, width, len(self.names)), np.float32)
            offset = 0

            def mean(a, size=fine):
                return uniform_filter(a, size=size, mode='reflect')

            def texture(a, size=fine):
                return np.sqrt(np.maximum(0, mean(a * a, size) - mean(a, size) ** 2))

            def write(fields):
                nonlocal offset
                for field in fields:
                    result[..., offset] = field[crop]
                    offset += 1

            def measurements(a):
                fields = measure(a, fine, coarse)
                return np.stack([f[crop] for f in fields], axis=-1), fields[7][crop] > .05

            he_quality = None
            if len(self.he) == 3:
                rgb = np.stack([np.clip(self.read(i, x0, y0, x1, y1) / self.stats[i], 0, 1)
                                for i in self.he], axis=-1)
                stains = rgb2hed(np.clip(rgb, 1e-4, 1)).astype(np.float32)
                h, e = stains[..., 0], stains[..., 1]
                write([rgb[..., i] for i in range(3)] + [h, e, mean(h), mean(e), texture(e)])
                he_quality = measurements(1 - rgb.mean(axis=-1))[0]
            # Accumulate one channel at a time: memory does not grow with a stack
            # of all channels' texture maps. Named features retain legacy cues.
            shape = (height, width, len(METRICS))
            low, high, total = np.full(shape, np.inf, np.float32), np.full(shape, -np.inf, np.float32), np.zeros(shape, np.float32)
            count = np.zeros((height, width), np.float32)
            def accumulate(values, signal, sd):
                nonlocal offset
                result[..., offset] = signal
                result[..., offset+1] = sd
                offset += 2
                active = values[..., 7] > .05
                np.minimum(low, np.where(active[..., None], values, np.inf), out=low)
                np.maximum(high, np.where(active[..., None], values, -np.inf), out=high)
                total[:] += np.where(active[..., None], values, 0)
                count[:] += active

            for start in range(0, len(self.markers), 8):
                channels = self.markers[start:start+8]
                arrays = [np.arcsinh(5*self.read(i,x0,y0,x1,y1)/self.stats[i]) for i in channels]
                values = None
                if self.accelerator:
                    yy,xx = np.mgrid[crop[0].start:crop[0].stop,crop[1].start:crop[1].stop]
                    positions = (np.repeat(np.arange(len(arrays)),height*width),
                                 np.tile(yy.ravel(),len(arrays)),np.tile(xx.ravel(),len(arrays)))
                    try:
                        values = self.accelerator.selected(arrays, positions, fine, coarse).reshape(len(arrays),height,width,len(METRICS))
                    except (RuntimeError, MemoryError) as exc:
                        import logging
                        logging.warning('Artifact prediction GPU unavailable; continuing on CPU: %s',exc)
                        self.accelerator = None
                        self.execution.update(device='cpu',fallback=str(exc))
                for index,a in enumerate(arrays):
                    measured = values[index] if values is not None else measurements(a)[0]
                    accumulate(measured, measured[...,0], measured[...,1])
            if he_quality is not None:
                result[..., offset:offset + len(METRICS)] = he_quality
                offset += len(METRICS)
            if self.markers:
                for values in (np.where(np.isfinite(low), low, 0), np.where(np.isfinite(high), high, 0),
                               total / np.maximum(count[..., None], 1)):
                    result[..., offset:offset + len(METRICS)] = values
                    offset += len(METRICS)
                result[..., offset] = count / len(self.markers)
            return result

    return QualityFeatures()


# ---------------------------------------------------------------- quality_v2: learns across markers
# A lint or a blur looks the same whichever marker it appears in. Each marker's measurements are first made relative
# to that marker's own typical values in this image (robust z: (value - median) / spread), then summarised across the
# markers: the most unusual one, the second most, the average and the share of markers unusual here. No measurement
# names a marker, so a defect taught in one marker is recognised in any other, and in another panel.
SCHEMA_V2 = 'quality_v2'
CALIBRATION_TILES = 12


class Pool:
    """Channel-independent summaries of per-marker measurements, built one marker at a time."""

    def __init__(self, n, centre, spread):
        shape = (n, len(METRICS))
        self.centre, self.spread = centre, spread
        self.low, self.high = np.full(shape, np.inf, np.float32), np.full(shape, -np.inf, np.float32)
        self.total, self.count = np.zeros(shape, np.float32), np.zeros(n, np.float32)
        self.top1, self.top2 = np.full(shape, -np.inf, np.float32), np.full(shape, -np.inf, np.float32)
        self.zsum, self.over = np.zeros(shape, np.float32), np.zeros(shape, np.float32)

    def add(self, values, channel, ids=slice(None)):
        active = values[:, 7] > .05
        a = active[:, None]
        self.low[ids] = np.minimum(self.low[ids], np.where(a, values, np.inf))
        self.high[ids] = np.maximum(self.high[ids], np.where(a, values, -np.inf))
        self.total[ids] += np.where(a, values, 0)
        self.count[ids] += active
        z = np.where(a, (values - self.centre[channel]) / self.spread[channel], -np.inf).astype(np.float32)
        top1, top2 = self.top1[ids], self.top2[ids]
        self.top2[ids] = np.where(z > top1, top1, np.maximum(top2, z))
        self.top1[ids] = np.maximum(top1, z)
        self.zsum[ids] += np.where(a, z, 0)
        self.over[ids] += np.where(a, np.abs(z) > 3, False)

    def result(self, channels):
        finite = lambda v: np.where(np.isfinite(v), v, 0)
        count = np.maximum(self.count, 1)[:, None]
        return np.concatenate([finite(self.low), finite(self.high), self.total / count, (self.count / max(channels, 1))[:, None],
                               np.clip(finite(self.top1), -50, 50), np.clip(finite(self.top2), -50, 50),
                               np.clip(self.zsum / count, -50, 50), self.over / count], axis=1)


def pooled_names():
    names = ['quality2:raw:' + mode + ':' + name for mode in ('min', 'max', 'mean') for name in METRICS]
    names.append('quality2:signal_fraction')
    names += ['quality2:z:' + mode + ':' + name for mode in ('max', 'second', 'mean', 'unusual') for name in METRICS]
    return names


def extractor_v2(image, scale):
    from .artifact_learning import Features
    from .artifact_sampling import measurements as measure

    class QualityFeatures2(Features):
        feature_schema = SCHEMA_V2
        tile_size = 256

        def __init__(self):
            super().__init__(image, scale)
            self.accelerator = None
            self.execution = {'device': 'cpu'}
            # No measurement is named after a marker: H&E, its quality, then the summaries across markers.
            self.names = self.names[:8] if len(self.he) == 3 else []
            if len(self.he) == 3:
                self.names += ['quality2:he:' + name for name in METRICS]
            if self.markers:
                self.names += pooled_names()
            self.fine, self.coarse = [max(3, int(round(um / (self.mpp * self.scale))) | 1) for um in WINDOWS_UM]
            self.calibrate()

        def enable_acceleration(self, device='auto'):
            from .artifact_sampling import backend
            self.accelerator, self.execution = backend(device)

        def calibrate(self):
            """Each marker's typical value and spread of every measurement in this image (its signal pixels in a
            grid of tiles), so markers are compared with themselves."""
            size = 160
            xs = np.linspace(0, max(self.w - size, 0), 4).astype(int)
            ys = np.linspace(0, max(self.h - size, 0), 3).astype(int)
            spots = [(x, y) for y in ys for x in xs][:CALIBRATION_TILES]
            self.centre, self.spread = {}, {}
            pooled = []
            for channel in self.markers:
                rows = []
                for x, y in spots:
                    a = np.arcsinh(5 * self.read(channel, x, y, min(self.w, x + size), min(self.h, y + size)) / self.stats[channel])
                    values = np.stack([f.ravel() for f in measure(a, self.fine, self.coarse)], axis=1)
                    rows.append(values[values[:, 7] > .05])
                values = np.concatenate(rows) if rows else np.empty((0, len(METRICS)))
                if len(values) >= 200:
                    centre = np.median(values, axis=0)
                    spread = 1.4826 * np.median(np.abs(values - centre), axis=0)
                    self.centre[channel], self.spread[channel] = centre.astype(np.float32), np.maximum(spread, 1e-3).astype(np.float32)
                    pooled.append(values[::max(1, len(values) // 2000)])
            fallback = np.concatenate(pooled) if pooled else np.zeros((1, len(METRICS)))
            centre = np.median(fallback, axis=0)
            spread = np.maximum(1.4826 * np.median(np.abs(fallback - centre), axis=0), 1e-3)
            for channel in self.markers:  # a marker with too little signal here uses all markers' values
                self.centre.setdefault(channel, centre.astype(np.float32))
                self.spread.setdefault(channel, spread.astype(np.float32))

        def _he(self, x0, y0, x1, y1):
            from scipy.ndimage import uniform_filter
            from skimage.color import rgb2hed
            rgb = np.stack([np.clip(self.read(i, x0, y0, x1, y1) / self.stats[i], 0, 1) for i in self.he], axis=-1)
            stains = rgb2hed(np.clip(rgb, 1e-4, 1)).astype(np.float32)
            h, e = stains[..., 0], stains[..., 1]
            mean = lambda v: uniform_filter(v, size=self.fine, mode='reflect')
            basic = [rgb[..., i] for i in range(3)] + [h, e, mean(h), mean(e), np.sqrt(np.maximum(0, mean(e * e) - mean(e) ** 2))]
            return basic, measure(1 - rgb.mean(axis=-1), self.fine, self.coarse)

        def tile(self, x, y, width, height):
            halo = self.coarse // 2 + 2
            x0, y0 = max(0, x - halo), max(0, y - halo)
            x1, y1 = min(self.w, x + width + halo), min(self.h, y + height + halo)
            crop = (slice(y - y0, y - y0 + height), slice(x - x0, x - x0 + width))
            parts = []
            if len(self.he) == 3:
                basic, quality = self._he(x0, y0, x1, y1)
                parts += [np.stack([f[crop] for f in basic], axis=-1).reshape(-1, 8),
                          np.stack([f[crop] for f in quality], axis=-1).reshape(-1, len(METRICS))]
            if self.markers:
                pool = Pool(height * width, self.centre, self.spread)
                for start in range(0, len(self.markers), 8):
                    channels = self.markers[start:start + 8]
                    arrays = [np.arcsinh(5 * self.read(i, x0, y0, x1, y1) / self.stats[i]) for i in channels]
                    values = None
                    if self.accelerator:
                        yy, xx = np.mgrid[crop[0].start:crop[0].stop, crop[1].start:crop[1].stop]
                        positions = (np.repeat(np.arange(len(arrays)), height * width), np.tile(yy.ravel(), len(arrays)), np.tile(xx.ravel(), len(arrays)))
                        try:
                            values = self.accelerator.selected(arrays, positions, self.fine, self.coarse).reshape(len(arrays), height * width, len(METRICS))
                        except (RuntimeError, MemoryError) as exc:
                            import logging
                            logging.warning('Artifact prediction GPU unavailable; continuing on CPU: %s', exc)
                            self.accelerator = None
                            self.execution.update(device='cpu', fallback=str(exc))
                    for index, (channel, a) in enumerate(zip(channels, arrays)):
                        measured = values[index] if values is not None else np.stack([f[crop].ravel() for f in measure(a, self.fine, self.coarse)], axis=1)
                        pool.add(measured, channel)
                parts.append(pool.result(len(self.markers)))
            return np.concatenate(parts, axis=1).reshape(height, width, -1).astype(np.float32)

        def sample_points(self, points, report=lambda *a: None, device='auto'):
            """The same measurements at training points, a window at a time, every marker read once per window."""
            import time
            from collections import defaultdict

            from .artifact_sampling import backend

            started = time.perf_counter()
            gpu, execution = backend(device)
            size, halo = self.tile_size, self.coarse // 2 + 2
            tiles = points // size
            windows = []
            for tx, ty in np.unique(tiles, axis=0):
                ids = np.flatnonzero((tiles == (tx, ty)).all(axis=1))
                x, y = int(tx * size), int(ty * size)
                windows.append((ids, max(0, x - halo), max(0, y - halo), min(self.w, x + size + halo), min(self.h, y + size + halo)))
            result = np.empty((len(points), len(self.names)), np.float32)
            offset = 0
            if len(self.he) == 3:
                for ids, x0, y0, x1, y1 in windows:
                    basic, quality = self._he(x0, y0, x1, y1)
                    yi, xi = points[ids, 1] - y0, points[ids, 0] - x0
                    result[ids, :8] = np.stack([f[yi, xi] for f in basic], axis=-1)
                    result[ids, 8:8 + len(METRICS)] = np.stack([f[yi, xi] for f in quality], axis=-1)
                offset = 8 + len(METRICS)
            if self.markers:
                pool = Pool(len(points), self.centre, self.spread)
                by_size = defaultdict(list)
                for window in windows:
                    by_size[(window[4] - window[2], window[3] - window[1])].append(window)
                for number, channel in enumerate(self.markers):
                    report(f'Reading marker {number + 1}/{len(self.markers)} · ' + ('GPU' if gpu else 'CPU'), .05 + .94 * number / len(self.markers))
                    for group in by_size.values():
                        for begin in range(0, len(group), 8):
                            batch = group[begin:begin + 8]
                            arrays, ids, positions = [], [], [[], [], []]
                            for b, (chosen, x0, y0, x1, y1) in enumerate(batch):
                                arrays.append(np.arcsinh(5 * self.read(channel, x0, y0, x1, y1) / self.stats[channel]))
                                ids.extend(chosen)
                                positions[0].extend([b] * len(chosen)); positions[1].extend(points[chosen, 1] - y0); positions[2].extend(points[chosen, 0] - x0)
                            values = None
                            if gpu:
                                try:
                                    values = gpu.selected(arrays, positions, self.fine, self.coarse)
                                except (RuntimeError, MemoryError) as exc:
                                    execution.update(device='cpu', fallback=str(exc)); gpu = None
                            if values is None:
                                values = np.concatenate([np.stack([f[points[chosen, 1] - y0, points[chosen, 0] - x0] for f in measure(a, self.fine, self.coarse)], axis=-1)
                                                         for a, (chosen, x0, y0, _, _) in zip(arrays, batch)])
                            pool.add(values, channel, np.asarray(ids))
                result[:, offset:] = pool.result(len(self.markers))
            if not np.isfinite(result).all():
                raise ValueError('Training measurements contain non-finite values. No model was saved.')
            execution.update(order='channel-major', windows=len(windows), channels=len(self.markers), seconds=round(time.perf_counter() - started, 3))
            report('Training examples ready', 1)
            return result, execution

    return QualityFeatures2()
