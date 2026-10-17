"""Bounded, channel-major training reads with optional CUDA texture measurements.

Sample the same coordinates and feature schema as tiled prediction. Finish a
channel's windows before opening the next so a wide marker panel cannot evict
itself from the image cache for every training tile.
"""
from collections import defaultdict
import logging
import time

import numpy as np
from scipy.ndimage import uniform_filter, sobel, laplace


def measurements(a, fine, coarse):
    """CPU reference; squared energies can round slightly below zero in filters."""
    def mean(v, size=fine):
        return uniform_filter(v, size=size, mode='reflect')
    mu, broad_mu = mean(a), mean(a, coarse)
    sd = np.sqrt(np.maximum(0, mean(a*a) - mu*mu))
    broad_sd = np.sqrt(np.maximum(0, mean(a*a, coarse) - broad_mu*broad_mu))
    gx, gy = sobel(a, axis=1, mode='reflect') / 8, sobel(a, axis=0, mode='reflect') / 8
    xx, yy, xy = mean(gx*gx), mean(gy*gy), mean(gx*gy)
    energy = gx*gx + gy*gy
    return (a, sd, np.sqrt(np.maximum(0, mean(energy))) / (sd+.03),
            np.sqrt(np.maximum(0, mean(energy, coarse))) / (broad_sd+.03),
            np.sqrt(np.maximum(0, mean(laplace(a, mode='reflect')**2))) / (sd+.03),
            np.clip((a-broad_mu)/(broad_sd+.03), -12, 12),
            np.sqrt(np.maximum(0, (xx-yy)**2 + 4*xy**2))/(xx+yy+.001), mu, broad_sd)


class TorchMeasurements:
    """Match SciPy's half-sample reflection (Torch's reflect padding differs)."""
    def __init__(self, device):
        import torch
        self.torch, self.device = torch, device
        self.indices = {}

    def pad(self, a, radius, axis):
        t = self.torch
        n = a.shape[axis]
        key = (n, radius)
        if key not in self.indices:
            i = t.arange(-radius, n+radius, device=self.device) % (2*n)
            self.indices[key] = t.where(i < n, i, 2*n-1-i)
        return a.index_select(axis, self.indices[key])

    def mean(self, a, size):
        from torch.nn.functional import avg_pool2d
        # SciPy accumulates in double and rounds each separable pass to float32.
        a = avg_pool2d(self.pad(a, size//2, -2).double().unsqueeze(1), (size, 1), stride=1).squeeze(1).float()
        return avg_pool2d(self.pad(a, size//2, -1).double().unsqueeze(1), (1, size), stride=1).squeeze(1).float()

    def selected(self, arrays, positions, fine, coarse):
        t = self.torch
        with t.no_grad():
            a = t.as_tensor(np.stack(arrays), device=self.device)
            bi, yi, xi = [t.as_tensor(v, device=self.device) for v in positions]
            mu, broad_mu = self.mean(a, fine), self.mean(a, coarse)
            sd = (self.mean(a*a, fine)-mu*mu).clamp_min(0).sqrt()
            broad_sd = (self.mean(a*a, coarse)-broad_mu*broad_mu).clamp_min(0).sqrt()
            px, py = self.pad(a, 1, -1), self.pad(a, 1, -2)
            dx, dy = px[..., 2:]-px[..., :-2], py[..., 2:, :]-py[..., :-2, :]
            dx, dy = self.pad(dx, 1, -2), self.pad(dy, 1, -1)
            gx = (dx[..., :-2, :]+2*dx[..., 1:-1, :]+dx[..., 2:, :])/8
            gy = (dy[..., :-2]+2*dy[..., 1:-1]+dy[..., 2:])/8
            lap = (py[..., :-2, :]-2*a+py[..., 2:, :]) + (px[..., :-2]-2*a+px[..., 2:])
            xx, yy, xy = self.mean(gx*gx, fine), self.mean(gy*gy, fine), self.mean(gx*gy, fine)
            energy = gx*gx+gy*gy
            # Only training points cross back to CPU, never a dense feature stack.
            fields = (a, sd, self.mean(energy, fine).clamp_min(0).sqrt()/(sd+.03),
                      self.mean(energy, coarse).clamp_min(0).sqrt()/(broad_sd+.03),
                      self.mean(lap*lap, fine).clamp_min(0).sqrt()/(sd+.03),
                      ((a-broad_mu)/(broad_sd+.03)).clamp(-12, 12),
                      ((xx-yy)**2+4*xy**2).clamp_min(0).sqrt()/(xx+yy+.001), mu, broad_sd)
            return t.stack([f[bi, yi, xi] for f in fields], dim=-1).cpu().numpy()


def backend(device='auto'):
    if device == 'cpu':
        return None, {'device': 'cpu', 'reason': 'CPU requested'}
    try:
        import torch
        if torch.cuda.is_available():
            # Respect the scheduler's visible devices and current CUDA device.
            index = torch.cuda.current_device()
            free, _ = torch.cuda.mem_get_info(index)
            if free >= 512*1024**2:
                probe = torch.zeros(1, device=f'cuda:{index}')
                del probe
                return TorchMeasurements(f'cuda:{index}'), {'device': 'cuda', 'name': torch.cuda.get_device_name(index), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda}
            reason = 'GPU memory is busy'
        else:
            reason = 'No usable CUDA device'
    except (ImportError, RuntimeError, OSError) as exc:
        reason = str(exc)
    return None, {'device': 'cpu', 'reason': reason}


def sample(reader, points, *, report=lambda *a: None, device='auto'):
    """Read all channels at the requested scale, in a cache-friendly order."""
    if getattr(reader, 'feature_schema', 'legacy') == 'quality_v2':
        return reader.sample_points(points, report=report, device=device)
    start = time.perf_counter()
    quality = getattr(reader, 'feature_schema', 'legacy') == 'quality_v1'
    fine, coarse = [max(3, int(round(um/(reader.mpp*reader.scale))) | 1) for um in (6., 18.)]
    halo = (coarse//2+2) if quality else fine//2+1
    size = getattr(reader, 'tile_size', 384)
    tiles = points//size
    groups = defaultdict(list)
    for tx, ty in np.unique(tiles, axis=0):
        indices = np.flatnonzero((tiles == (tx, ty)).all(axis=1))
        x, y = int(tx*size), int(ty*size)
        x0, y0 = max(0, x-halo), max(0, y-halo)
        x1, y1 = min(reader.w, x+size+halo), min(reader.h, y+size+halo)
        groups[(y1-y0, x1-x0)].append((indices, x0, y0, x1, y1))
    result = np.empty((len(points), len(reader.names)), np.float32)
    gpu, execution = backend(device) if quality else (None, {'device': 'cpu', 'reason': 'Legacy feature schema'})
    execution.update(order='channel-major', windows=sum(map(len, groups.values())), channels=len(reader.stats))
    report('Reading training examples · '+('GPU' if gpu else 'CPU'), 0)
    offset = 0
    if len(reader.he) == 3:
        from skimage.color import rgb2hed
        for group in groups.values():
            for ids, x0, y0, x1, y1 in group:
                report('Reading H&E examples', .02)
                rgb = np.stack([np.clip(reader.read(i,x0,y0,x1,y1)/reader.stats[i],0,1) for i in reader.he],axis=-1)
                stains = rgb2hed(np.clip(rgb,1e-4,1)).astype(np.float32)
                h, e = stains[...,0], stains[...,1]
                mean = lambda v: uniform_filter(v, size=fine, mode='reflect')
                fields = [rgb[...,i] for i in range(3)]+[h,e,mean(h),mean(e),np.sqrt(np.maximum(0,mean(e*e)-mean(e)**2))]
                yi, xi = points[ids,1]-y0, points[ids,0]-x0
                result[ids,:8] = np.stack([f[yi,xi] for f in fields],axis=-1)
                if quality:
                    fields = measurements(1-rgb.mean(axis=-1),fine,coarse)
                    start_he = 8+2*len(reader.markers)
                    result[ids,start_he:start_he+9] = np.stack([f[yi,xi] for f in fields],axis=-1)
        offset = 8
    low, high = np.full((len(points),9),np.inf,np.float32),np.full((len(points),9),-np.inf,np.float32)
    total, count = np.zeros((len(points),9),np.float32),np.zeros(len(points),np.float32)
    for number, channel in enumerate(reader.markers):
        for group in groups.values():
            for start_group in range(0,len(group),8):
                batch = group[start_group:start_group+8]
                report(f'Reading marker {number+1}/{len(reader.markers)} · '+('GPU' if gpu else 'CPU'), .05+.94*number/max(1,len(reader.markers)))
                arrays, indices, positions = [], [], [[],[],[]]
                for b, (ids,x0,y0,x1,y1) in enumerate(batch):
                    arrays.append(np.arcsinh(5*reader.read(channel,x0,y0,x1,y1)/reader.stats[channel]))
                    indices.extend(ids)
                    positions[0].extend([b]*len(ids));positions[1].extend(points[ids,1]-y0);positions[2].extend(points[ids,0]-x0)
                ids = np.asarray(indices)
                values = None
                if gpu:
                    try:
                        values = gpu.selected(arrays, positions, fine, coarse)
                    except (RuntimeError, MemoryError) as exc:
                        logging.warning('Artifact GPU preparation unavailable; continuing on CPU: %s', exc)
                        execution.update(device='cpu', fallback=str(exc));gpu=None
                if values is None:
                    rows=[]
                    for a, (chosen,x0,y0,_,_) in zip(arrays,batch):
                        yi,xi=points[chosen,1]-y0,points[chosen,0]-x0
                        if quality:
                            fields=measurements(a,fine,coarse)
                        else:
                            mu=uniform_filter(a,size=fine,mode='reflect')
                            fields=(a,np.sqrt(np.maximum(0,uniform_filter(a*a,size=fine,mode='reflect')-mu*mu)))
                        rows.append(np.stack([f[yi,xi] for f in fields],axis=-1))
                    values=np.concatenate(rows)
                result[ids,offset:offset+2]=values[:,:2]
                if quality:
                    active=values[:,7]>.05
                    low[ids]=np.minimum(low[ids],np.where(active[:,None],values,np.inf))
                    high[ids]=np.maximum(high[ids],np.where(active[:,None],values,-np.inf))
                    total[ids]+=np.where(active[:,None],values,0);count[ids]+=active
        offset+=2
    if quality and reader.markers:
        offset+=9 if len(reader.he)==3 else 0
        result[:,offset:offset+9]=np.where(np.isfinite(low),low,0)
        result[:,offset+9:offset+18]=np.where(np.isfinite(high),high,0)
        result[:,offset+18:offset+27]=total/np.maximum(count[:,None],1)
        result[:,offset+27]=count/len(reader.markers)
    if not np.isfinite(result).all():
        raise ValueError('Training measurements contain non-finite values. No model was saved.')
    execution['seconds']=round(time.perf_counter()-start,3)
    report('Training examples ready',1)
    return result, execution
