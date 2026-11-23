# External HTTPS access for Studio Review

The message in Share means that an external address has not been configured.
the internal network/VPN links work on the internal network. Changing `http` to `https` in a
link, or setting an environment variable alone, does not make it reachable
outside that network.

For a fixed address **inside the internal network/VPN**, see
[Fixed Internal / VPN reviewer links](studio-review-vpn.md). This supports a
persistent internal gateway without requiring a public HTTPS deployment.

## What to ask your institution’s IT team for

You can send this request:

> Please provide an approved external HTTPS address for SPACEMAP's reviewer
> interface. It needs a gateway that can reach the running Studio server on
> the internal network. Please expose only `/spacemap-viewer/api/review-public/*` and the
> read-only static assets at `/spacemap-viewer/api/window/*`; block other paths.
> Reviewer links are expiring, revocable access credentials. Please avoid logging
> their full URLs or query strings. We need to test from outside the VPN using
> a disposable project before sharing research images.

For example, the upstream could be `compute-node.internal:8503`. IT must
confirm the current host/port and choose a stable upstream; a compute-node address
can change or become unavailable when its allocation ends. A public gateway must
actually have an approved network route to that upstream.

## Example for the gateway administrator

This is a configuration template, **not an installed or validated deployment**.
Use the institution's existing HTTPS gateway if available. For Caddy, replace
the example hostname and private upstream below. Keep the URL path intact.

```caddyfile
review.example.org {
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

Do not replace the allowlist with a proxy of the whole Studio. Reviewer image,
annotation and cell requests go through the scoped `review-public` API. The
ordinary `/studio`, `/image`, `/cells` and file-browser APIs are not public routes.
Use `handle`, not `handle_path`, because stripping the prefix breaks these URLs.
Configure URL redaction at any upstream load balancer as well; the example does
not enable Caddy access logging. Do not put reviewer links into analytics.

Caddy can manage a trusted certificate when DNS and certificate validation are
configured. IT may instead supply an institutional certificate. See the official
[HTTPS requirements](https://caddyserver.com/docs/automatic-https) and
[path handlers](https://caddyserver.com/docs/caddyfile/directives/handle).
For a gateway outside the trusted internal network, IT must also secure its
connection to the backend; a public certificate alone does not secure that hop.

## Connect Studio to the public address

Once that address works, set this in the environment of the process that serves
the Studio viewer API, **before starting it**:

```bash
export SPACEMAP_REVIEW_PUBLIC_BASE_URL='https://review.example.org/spacemap-viewer/api'
```

Then use your normal Studio launch command. An already-running process will not
inherit a later `export`; apply the change on its next planned restart, after
jobs and saves finish. Keep the same viewer state/registry directory so existing
link handles remain resolvable. No separate copy of the research dataset is
required for this gateway arrangement.

The variable is the **API base**, not `/window/index.html` or a particular link.
In **11 Review → Share → External browser**, Studio appends each link's scoped
path automatically. Setting this variable only enables link generation; it does
not verify network reachability or certificate validity.

## Reuse a named link

In **Review → Share**, select an existing reviewer name: its saved URL appears
automatically. Tick **Add selected items** when needed, choose **HTTPS · external
browser**, then **Update link**. The queue,
expiry and decisions stay the same. Send the displayed HTTPS address to the reviewer;
their open VPN page does not automatically move to the new host. Changing the address
is not revocation: both network routes can still reach the same active credential.

Without a configured fixed internal gateway, after moving Studio to another node,
open Studio at its new address and click
**Update link** with **Internal / VPN** selected. This refreshes the host/port while
preserving the reviewer's credential and decisions. Send that updated URL if the
old node is no longer running. The same user state directory must be available;
this does not migrate servers, datasets or credentials between unrelated accounts.

Links created before URL recovery was added can recover automatically when the
reviewer next opens the old link or its open page polls the updated server.
If that has not happened and only a secret hash remains, Studio cannot reconstruct
the URL and may ask for the original URL once.
Paste the original **Share** URL for that reviewer. Studio verifies it against that
link; it does not visit the URL or replace the credential. Recoverable keys are kept
in owner-only files in the Studio user's state directory, separate from the project
queue and list responses. Expired/revoked links cannot be reopened this way.

## Check before sending a real link

Use a disposable project and a browser outside the internal network/VPN. Check the reviewer
page, image tiles, H&E/markers, shared read-only Data and a decision received by
the owner. Add a new item to the same named link and confirm it appears. Verify
that ordinary Studio routes are blocked, unshared resources are denied, and a
revoked link stops working. This has not been tested on an external gateway in
the current workspace; local reviewer tests do not prove external reachability.

Until IT supplies the address and reachable gateway, choose **Internal / VPN** in
Share and have the reviewer connect to the institution's permitted network.
