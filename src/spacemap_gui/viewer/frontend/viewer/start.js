/* SPACEMAP viewer, part 12 of 12: starting the viewer (inside SPACEMAP, or as a window of its own), once every other part has loaded.
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

if (STANDALONE) {
  document.body.classList.add("standalone");
  followSession();
} else {
  window.addEventListener("message", function (event) {
    const data = event.data || {};
    if (data.type !== "streamlit:render") return;
    const spec = (data.args || {}).spec || {};
    setFrameHeight(spec.height || 760);
    // One id per browser, so a viewer window and SPACEMAP find each other again
    // after a restart: SPACEMAP publishes views under it.
    const browser = browserId();
    if (spec.session && browser && spec.session !== browser && state.reportedBrowser !== browser) {
      state.reportedBrowser = browser;
      setComponentValue({ event: "browser", browser: browser, n: Date.now() });
    }
    if (spec.command) acceptCommandView({spec, version:spec._studio_version || 0});
    else if (!state.spec || state.spec.id !== spec.id) init(spec);
    else state.spec = Object.assign(state.spec, { session: spec.session });
  });
  send("streamlit:componentReady", { apiVersion: 1 });
}
