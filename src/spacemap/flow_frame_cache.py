"""Frame-owned analysis caches that pandas must not deep-copy onto every Series."""
import weakref


class FrameCache(dict):
    def __init__(self, frame):
        super().__init__()
        self.owner = weakref.ref(frame)

    def __deepcopy__(self, memo):
        # Pandas propagates attrs during column access. Cache arrays belong to
        # the immutable source frame, never to the derived Series or row slice.
        return self

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other


def frame_cache(frame):
    cache = frame.attrs.get('_flow_cache')
    if not isinstance(cache,FrameCache) or cache.owner() is not frame:
        cache = FrameCache(frame)
        frame.attrs['_flow_cache'] = cache
    return cache
