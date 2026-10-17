"""Small preference files must not queue behind image decoding/table work."""

from anyio import CapacityLimiter, to_thread
from anyio.lowlevel import RunVar


_limiter = RunVar('spacemap_viewer_settings_limiter')


async def run(function, *args):
    # Per event loop, including separate TestClients. Existing per-file locks and
    # sequence stamps still serialize writes and reject late/duplicate requests.
    try:
        limiter = _limiter.get()
    except LookupError:
        limiter = CapacityLimiter(4)
        _limiter.set(limiter)
    return await to_thread.run_sync(function, *args, limiter=limiter)
