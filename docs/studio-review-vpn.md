# Fixed Internal / VPN reviewer links

A stable review address needs a persistent internal host. Choosing a fixed
Studio port does not keep the link stable when the compute node changes.
`localhost` means the computer on which the reviewer's browser is running;
it is not a shared the internal network address.

SPACEMAP's launcher tries the requested port, then the next free port if another
process already owns it. This explains a move from 8502 to 8503. Leave that
behaviour available: the reviewer gateway can forward to either port.

## Request to send to your IT team

> Could you provide a persistent, internal-only hostname and reverse proxy for
> SPACEMAP Review, reachable by the internal network/VPN users? My Studio runs on temporary
> GPU/VNC nodes, so both the node and port can change. Reviewers need one fixed
> address while the proxy forwards to my current Studio node and port.
>
> This is for internal access, not a public website. Please confirm the permitted
> internal transport, provide a persistent host/service and fixed port, and an
> approved way for me to update the destination when my allocation changes.
> The proxy itself does not need a GPU. The last example backend was
> compute-node.internal:8503; the current destination must be checked at launch.
>
> Forward only `/spacemap-viewer/api/review-public/*` and GET/HEAD static assets
> under `/spacemap-viewer/api/window/*`, preserving the paths. Block other routes.
> Reviewer URLs contain access credentials, so please redact full paths/query
> strings from access and error logs. No copies of the research dataset are
> needed on the proxy. Please test a disposable link, image and review decision
> from a VPN client before deployment.

Do not place a permanent service on a login node or assume a VNC reservation
will stay available. IT needs to identify a suitable persistent host.

## How it will work

```text
Reviewer: one fixed the internal network/VPN URL
                   |
          persistent review proxy
                   |
       currently running Studio server
       compute-node.internal:8503 -> another-node:8502
```

Only the proxy destination changes. The reviewer's URL, queue and decisions stay
the same. Do not load-balance across unrelated Studio servers or guess the newest
available node: switch explicitly to the server running the intended project.

The fixed address still depends on a running backend. Review is temporarily
unavailable while Studio is stopped or the destination has not been updated.
Always-available review would require a separately maintained review service;
that service has not been deployed here.

## Configure Studio after IT supplies the address

Put the following variable in the environment used by each Studio launch,
replacing the example hostname and port with those provided by IT:

```bash
export SPACEMAP_REVIEW_INTERNAL_BASE_URL='http://review.internal.example:9080/spacemap-viewer/api'
```

This is the API base, not an individual review link. Internal HTTPS is also
accepted if provided by IT. The setting controls **Internal / VPN** links and is
independent of `SPACEMAP_REVIEW_PUBLIC_BASE_URL`, which controls external HTTPS
links. An environment variable does not create DNS, start a proxy, or update
its destination. Keep it in the normal launch environment so every new node
uses the same address.

Use the same account, shared project paths and existing SPACEMAP state directory
on the new node. That directory holds `viewer_registry.json`, `viewer_secret`
and `review_link_keys/`; the project holds the review queue and decisions.
Do not replace the secret or point `SPACEMAP_HOME` / `SPACEMAP_STATE_DIR` at a
fresh or node-local temporary folder during migration. Use the same checkout
version and mount paths on both nodes.

New links now use the fixed address. For an existing named link, select it in
**Review → Share**, keep **Internal / VPN**, and click **Update link**. Its credential,
queue and decisions remain intact. Send this fixed URL to the reviewer **once**.
Previously distributed URLs containing a node address cannot automatically
follow the new proxy when that old node is gone. Link expiry and revocation
still apply; a fixed address does not extend a link's lifetime.

On later node changes, update the proxy's backend destination. Reviewers keep
the fixed URL and can reload it after the backend is ready. There is no need
to create or update every named reviewer link again.

## Gateway template for IT

This is an example for an existing Caddy service, not an installed deployment.
Replace the hostname, listening port and backend in **both** handlers. Limit
ingress to the institution's approved the internal network/VPN network at the host/firewall.

```caddyfile
http://review.internal.example:9080 {
    header {
        Referrer-Policy no-referrer
        X-Content-Type-Options nosniff
        X-Robots-Tag "noindex, nofollow"
        Cache-Control "no-store"
    }
    @review path /spacemap-viewer/api/review-public/*
    handle @review {
        reverse_proxy compute-node.internal:8503
    }
    @assets {
        path /spacemap-viewer/api/window/*
        method GET HEAD
    }
    handle @assets {
        reverse_proxy compute-node.internal:8503
    }
    handle {
        respond "Not found" 404
    }
}
```

Caddy supports explicit HTTP site addresses and forwarding to another host/port.
Its configuration can be validated and reloaded after the destination changes.
See the official [site address documentation](https://caddyserver.com/docs/caddyfile/concepts#addresses),
[reverse proxy documentation](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy)
and [reload command](https://caddyserver.com/docs/command-line#caddy-reload).
The gateway administrator should follow their service's deployment process;
SPACEMAP does not modify the proxy or institutional network configuration.

## Verification

`tests/test_review_link_address.py` checks configured internal addresses,
node-dependent fallback, independent external configuration and invalid URLs.
`tests/test_review_link_migration.py` starts disposable HTTP backends behind a
fixed loopback test proxy. It records a decision on one backend, stops that
process, starts a new one on another port, and checks the same link, queue,
decision, image view specification, static reviewer page and revocation.

```bash
SPACEMAP_NETWORK_TESTS=1 PYTHONPATH=src:tests python -m pytest -q tests/test_review_link_migration.py
```

This tests actual HTTP across separate processes with shared test storage. It
does not establish the internal network DNS, firewall access, Caddy configuration validity or
connectivity between two real compute nodes. Those require the persistent host
and a final test from a reviewer on VPN. Existing research data and reviewer
links are not changed by these tests.
