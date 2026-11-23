"""Owner-only recovery of saved review URLs; no secrets in queue/list responses."""
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

from spacemap_gui.studio import review_store as store
from . import registry


def _path(root, grant_id):
    handle = registry.register_object('review_link', root=str(root), grant=grant_id)
    return handle, registry._state_folder() / 'review_link_keys' / handle


def remember(root, grant_id, secret):
    store.authorize(root, grant_id, secret)
    handle, path = _path(root, grant_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            output.write(secret)
        os.replace(temp, path)  # mkstemp creates an owner-only file (0600).
    finally:
        Path(temp).unlink(missing_ok=True)
    return handle + '.' + secret


def access_key(root, grant_id, existing_url=''):
    handle, path = _path(root, grant_id)
    if existing_url:
        # Legacy links stored only a hash. Accept the existing credential once,
        # never visit its URL and never regenerate/revoke the reviewer's link.
        parsed = urlsplit(str(existing_url).strip())
        parts = parsed.path.split('/review-public/', 1)
        key = parts[1].strip('/') if len(parts) == 2 else ''
        old_handle, separator, secret = key.partition('.')
        if old_handle != handle or not separator or '/' in secret:
            raise ValueError('Paste the original URL for this reviewer link.')
        return remember(root, grant_id, secret)
    try:
        secret = path.read_text()
    except FileNotFoundError:
        raise ValueError('Paste this older link once to display its URL. Its review list and decisions are preserved.') from None
    store.authorize(root, grant_id, secret)
    return handle + '.' + secret


def remember_address(root, grant_id, base):
    """Keep the last generated address, separately from the owner-only credential."""
    _, key_path = _path(root, grant_id)
    path = key_path.with_suffix('.address')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            output.write(str(base).rstrip('/'))
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def saved_url(request, root, grant_id):
    key = access_key(root, grant_id)
    link = next(link for link in store.grants(root) if link['id'] == grant_id)
    _, path = _path(root, grant_id)
    try:
        base = path.with_suffix('.address').read_text().strip()
    except FileNotFoundError:
        base = base_url(request, link['scope'].get('access', 'internal'))
    return base + '/review-public/' + key + '/'


def recover_visited_link(request, root, grant_id, secret):
    """A valid legacy link can remember itself when its reviewer next opens it."""
    _, path = _path(root, grant_id)
    if not path.is_file():
        remember(root, grant_id, secret)
        remember_address(root, grant_id, str(request.base_url).rstrip('/') + request.url.path.split('/review-public/')[0])


def base_url(request, access):
    if access not in ('internal', 'external'):
        raise ValueError('Choose Internal / VPN or HTTPS.')
    if access == 'internal':
        base = os.environ.get('SPACEMAP_REVIEW_INTERNAL_BASE_URL', '').strip().rstrip('/')
        if base:
            if not _valid_service_url(base, ('http', 'https')):
                raise ValueError('SPACEMAP_REVIEW_INTERNAL_BASE_URL must be an http:// or https:// API base, without credentials, query or fragment.')
            return base
        return str(request.base_url).rstrip('/') + request.url.path.split('/studio/')[0]
    base = os.environ.get('SPACEMAP_REVIEW_PUBLIC_BASE_URL', '').strip().rstrip('/')
    if not _valid_service_url(base, ('https',)):
        raise ValueError('Set up an HTTPS review service before creating an external link. the internal network/VPN links are available now.')
    return base


def _valid_service_url(base, schemes):
    """Validate configured addresses without contacting or exposing credentials."""
    try:
        parsed = urlsplit(base)
        port = parsed.port  # Also reject invalid/out-of-range ports.
        return (parsed.scheme in schemes and bool(parsed.hostname) and
                parsed.username is None and parsed.password is None and
                not parsed.query and not parsed.fragment and
                not any(c.isspace() or ord(c) < 32 for c in base) and
                '\\' not in base and port != 0)
    except ValueError:
        return False


def link_info(root, link):
    _, path = _path(root, link['id'])
    return {**link, 'access': link['scope'].get('access', 'internal'), 'url_saved': path.is_file()}
