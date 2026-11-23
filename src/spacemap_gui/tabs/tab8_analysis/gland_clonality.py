"""Classic adapter for the same Gland types worker and saved results as Studio."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from spacemap_gui.studio import gland_clonality as glands


def render_gland_clonality(project_root, df, table):
    from spacemap_gui.studio import jobs
    from spacemap import dataio

    prefix = "gland_" + str(table)
    initial = glands.settings(project_root, table)
    c = json.loads(json.dumps(initial["values"]))
    columns = list(df.columns)

    def col(key, label, optional=False, help=None):
        options = ([""] if optional else [""]) + columns
        c[key] = st.selectbox(label, options, index=options.index(c[key]) if c[key] in options else 0,
                             format_func=lambda v: v or "None", key=prefix+key, help=help)

    def labels(key, feature, title):
        options = sorted(glands._strings(df[c[feature]]).dropna().unique()) if c[feature] in df else []
        prior = [v for v in c[key] if v in options] or [v for v in options if v.lower() not in ("none", "nan", "null", "unassigned", "")]
        c[key] = st.multiselect(title, options, default=prior, key=prefix+key)

    configure, profiles, explore, compare = st.tabs(["Configure", "Profiles", "Explore", "Compare"])
    with configure:
        with st.expander("Annotation columns", expanded=not bool(c["regions"])):
            editable = pd.DataFrame(c["regions"], columns=["name", "roi_col", "bed_id_col", "bed_col", "bed_value"])
            edited = st.data_editor(editable, hide_index=True, num_rows="dynamic", use_container_width=True, key=prefix+"regions",
                column_config={"name": st.column_config.TextColumn("Compartment", required=True),
                    "roi_col": st.column_config.SelectboxColumn("Gland ID", options=columns, required=True),
                    "bed_id_col": st.column_config.SelectboxColumn("Bed ID", options=["", *columns]),
                    "bed_col": st.column_config.SelectboxColumn("Bed flag", options=["", *columns]),
                    "bed_value": st.column_config.TextColumn("Flag value", default="1")})
            c["regions"] = edited.fillna("").to_dict("records")
        scopes = [r["name"] for r in c["regions"] if r["name"]]
        c["scope"] = st.multiselect("Fit scope", scopes, default=[v for v in c["scope"] if v in scopes], key=prefix+"scope",
                                   help="Fit one shared set of composition types across the chosen compartments. These are not proven genetic clones.")
        a, b = st.columns(2)
        with a:
            col("interior_col", "Interior subtype column")
            labels("interior_classes", "interior_col", "Interior subtypes")
        with b:
            col("exterior_col", "Exterior cell-type column", True)
            labels("exterior_classes", "exterior_col", "Exterior classes")
        with st.expander("Cell roles and comparison columns"):
            a, b = st.columns(2)
            for i, (key, title, optional) in enumerate((("image_col", "Image ID", False), ("piece_col", "Tissue piece", True),
                ("cell_id_col", "Cell ID", False), ("label_col", "Segmentation label", True), ("x_col", "X", False), ("y_col", "Y", False),
                ("sample_col", "Independent sample", True), ("visit_col", "Visit", True), ("group_col", "Compare by", True), ("stratum_col", "Stratify by", True), ("parent_col", "Parent cells", True))):
                with (a if i % 2 == 0 else b):
                    col(key, title, optional)
            if c["parent_col"]:
                labels("parent_values", "parent_col", "Parent cell types")
            for source, pairs in (("group_col", [("group_a", "Group A"), ("group_b", "Group B")]),
                                  ("visit_col", [("baseline", "Before"), ("followup", "After")])):
                options = ["", *sorted(glands._strings(df[c[source]]).dropna().unique())] if c[source] in df else [""]
                a, b = st.columns(2)
                for target, (key, title) in zip((a, b), pairs):
                    with target:
                        c[key] = st.selectbox(title, options, index=options.index(c[key]) if c[key] in options else 0, key=prefix+key)
        with st.expander("Fit settings", expanded=True):
            a, b = st.columns(2)
            for i, (key, title, minimum) in enumerate((("min_gland", "Gland cells to fit", 1), ("min_bed", "Bed cells to fit", 1),
                 ("interior_k", "Gland types (K)", 1), ("exterior_k", "Niche types (K)", 1), ("seed", "Seed", 0))):
                with (a if i % 2 == 0 else b):
                    c[key] = st.number_input(title, min_value=minimum, value=c[key], step=1, key=prefix+key, help=glands.K_HELP.get(key))
            c["prior"] = st.number_input("Prior strength", min_value=0., value=float(c["prior"]), key=prefix+"prior",
                help="Adds this many cells in the pooled composition. Small glands are retained as smoothed predictions, not used to fit types.")
            c["standardize"] = st.checkbox("Use size-weighted Z-score", value=c["standardize"], key=prefix+"standardize",
                help="Fits mean and spread on well-sampled structures with log1p(cell count) weights. Relative enrichment is not marker positivity.")
            a, b = st.columns(2)
            with a: c["auto_k"] = st.checkbox("Auto gland K", value=c["auto_k"], key=prefix+"auto_k", help="Choose the best-scoring tested K during Fit instead of the manual Gland types (K).")
            with b: c["auto_bed_k"] = st.checkbox("Auto niche K", value=c["auto_bed_k"], key=prefix+"auto_bed_k", help="Choose the best-scoring tested K during Fit instead of the manual Niche types (K).")
        with st.expander("Output columns"):
            c["output_col"] = st.text_input("Gland type", c["output_col"], key=prefix+"output_col")
            c["niche_col"] = st.text_input("Bed niche", c["niche_col"], key=prefix+"niche_col")
            overwrite = st.checkbox("Overwrite gland results", key=prefix+"overwrite")
        images = sorted(df[c["image_col"]].astype("string").str.strip().dropna().unique()) if c["image_col"] in df else []
        picked = st.multiselect("Images to fit", images, default=images, key=prefix+"images")
        a, b = st.columns(2)
        with a: run = st.button("Fit gland types", type="primary", key=prefix+"run")
        with b: diagnose = st.button("Review optimal K", key=prefix+"diagnose", help=glands.K_HELP["review"])
        if run or diagnose:
            try:
                glands.settings(project_root, table, c, initial["revision"])
                config = glands.prepare(project_root, table, c, picked, overwrite, diagnose)
                job = jobs.start(str(project_root), "gland_types", config)
                st.session_state[prefix+"job"] = job["id"] if isinstance(job, dict) else job
            except (ValueError, OSError) as exc:
                st.error(str(exc))

    if st.session_state.get(prefix+"job"):
        @st.fragment(run_every="2s")
        def progress():
            job = jobs.status(project_root, st.session_state[prefix+"job"])
            if job["status"] in jobs.ACTIVE:
                st.warning(job.get("message", "Fitting gland types…"))
                st.progress(float(job.get("progress", 0)))
            else:
                if job["status"] == "done":
                    st.session_state[prefix+"chosen"] = job["id"]
                else:
                    st.session_state[prefix+"error"] = job.get("message", "Gland fit stopped.")
                del st.session_state[prefix+"job"]
                st.rerun()
        progress()
    if st.session_state.get(prefix+"error"):
        st.error(st.session_state[prefix+"error"])
    available = glands.runs(project_root, table)
    if not available:
        return
    ids = [r["id"] for r in available]
    chosen = st.session_state.get(prefix+"chosen", ids[0])
    job_id = st.selectbox("Saved gland run", ids, index=ids.index(chosen) if chosen in ids else 0,
                          format_func=lambda v: next(r["label"] for r in available if r["id"] == v), key=prefix+"saved_run")
    result, folder = glands.saved_result(project_root, table, job_id)
    cache_key = prefix+"runtime_"+job_id
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {"models": json.loads((folder / "gland_models.json").read_text()),
            "structures": pd.read_parquet(folder / "gland_structures.parquet")}
        if not result["diagnose"]:
            # Runtime dataframe for classic consumers; source file remains unchanged.
            anchor = result["values"]["cell_id_col"]
            added = dataio.load(table, columns=[anchor, *result["columns"]]).set_index(anchor)
            runtime_frame = df.copy(deep=False)
            for name in result["columns"]:
                runtime_frame[name] = df[anchor].map(added[name])
            st.session_state[cache_key]["dataframe"] = runtime_frame
    runtime = st.session_state[cache_key]
    if "dataframe" in runtime:
        st.session_state["gland_types_dataframe"] = runtime["dataframe"]
    with profiles:
        a, b = st.columns(2)
        a.metric("Glands", result["glands"]); b.metric("Nonempty beds", result["beds"])
        if result["diagnose"] or any(m and len(m.get("diagnostics", [])) > 1 for m in result["models"].values()):
            def use_k(kind, k):
                st.session_state[prefix+("interior_k" if kind == "interior" else "exterior_k")] = k
                st.session_state[prefix+("auto_k" if kind == "interior" else "auto_bed_k")] = False

            with st.expander("K review", expanded=True):
                st.caption("Saved K scores. Use a suggested K, then Fit to create results.")
                for kind in ("interior", "exterior"):
                    st.image(glands.k_review_figure(result["models"].get(kind), kind))
                    review = result["k_reviews"][kind]
                    if review["suggested_k"]:
                        st.button(f"Use {'gland' if kind == 'interior' else 'niche'} K={review['suggested_k']}",
                                  key=prefix+"use_k_"+kind, on_click=use_k, args=(kind, review["suggested_k"]),
                                  help="Sets this manual K and turns off Auto for it. Does not change cell labels until you Fit.")
        for kind in ("interior", "exterior"):
            if result["profiles"][kind]:
                st.image(str(folder / f"{kind}_profiles.png"))
        with st.expander("Individual silhouette scores"):
            if st.button("Show fit quality", key=prefix+"show_silhouette",
                         help="Shows individual gland and bed scores for the saved K. Uses the same fit sample as the K chart; small predicted structures are excluded. No model is refitted."):
                runtime["show_silhouette"] = True
            if runtime.get("show_silhouette"):
                for kind in ("interior", "exterior"):
                    data = glands.saved_silhouette(project_root, table, job_id, kind)
                    png = glands.silhouette_figure(data, result["palette"])
                    st.image(png)
                    a, b = st.columns(2)
                    a.download_button("Save PNG", png, file_name=kind+"_silhouette.png", mime="image/png", key=prefix+kind+"_silhouette_png")
                    b.download_button("Scores CSV", glands.silhouette_csv(data), file_name=kind+"_silhouette.csv", mime="text/csv", key=prefix+kind+"_silhouette_csv")
        with st.expander("Run details"):
            st.write(result["notes"])
            st.json(result["values"])
    with explore:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        image = st.selectbox("Image", result["images"], key=prefix+"map_image")
        found = glands.maps(project_root, table, job_id, image)
        points = pd.DataFrame(found["points"])
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Gland types", "Exterior niches"])
        for i, role in enumerate(("gland", "bed"), start=1):
            for name, group in points.fillna({role+"_name": "Unassigned"}).groupby(role+"_name"):
                fig.add_trace(go.Scattergl(x=group.x, y=group.y, mode="markers", name=name,
                    customdata=group[role+"_id"], marker=dict(size=3, color=found["palette"].get(name, "#888888")),
                    hovertemplate="%{customdata}<extra>%{fullData.name}</extra>"), row=1, col=i)
        fig.update_yaxes(autorange="reversed", scaleanchor="x"); fig.update_layout(height=600, margin=dict(l=10,r=10,t=40,b=10))
        selection = st.plotly_chart(fig, use_container_width=True, key=prefix+"maps", on_select="rerun")
        selected_points = selection.selection.points if selection else []
        if selected_points:
            structure = selected_points[0].get("customdata")
            if structure:
                from spacemap_gui.viewer import ui
                from urllib.parse import urlencode
                link = ui.studio_link("tab8")
                if link:
                    st.link_button("Review selected gland in Studio", link + "&" + urlencode({"gland_run": job_id, "gland_structure": structure}))
        s = runtime["structures"]; st.dataframe(s[s.image.eq(image)], hide_index=True, use_container_width=True)
    with compare:
        comp = st.selectbox("Compartment", ["", *result["values"]["scope"]], format_func=lambda v: v or "All fitted compartments", key=prefix+"comparison_comp")
        stratify = st.checkbox("Show strata as well", key=prefix+"stratify")
        try:
            comparison = glands.clinical(runtime["structures"], result["values"], compartment=comp, strata=stratify)
            if comparison["tests"]:
                tests = pd.DataFrame(comparison["tests"])
                a, b = st.columns(2)
                with a: view = st.selectbox("Visit / change", tests.view.unique(), key=prefix+"view")
                with b: stratum = st.selectbox("Stratum", tests.stratum.unique(), key=prefix+"stratum")
                st.image(glands.clinical_figure(comparison, view, stratum, result["palette"]))
                st.dataframe(tests[tests.view.eq(view) & tests.stratum.eq(stratum)], hide_index=True, use_container_width=True)
                import plotly.express as px
                trajectory = pd.DataFrame(comparison["trajectories"])
                if stratum != "All": trajectory = trajectory[trajectory.stratum.eq(stratum)]
                for kind in trajectory.kind.unique():
                    visits = [v for v in [result["values"]["baseline"], result["values"]["followup"]] if v]
                    visits += [v for v in trajectory.visit.unique() if v not in visits]
                    fig = px.bar(trajectory[trajectory.kind.eq(kind)], x="sample", y="value", color="feature", facet_col="visit",
                        barmode="stack", color_discrete_map=result["palette"], category_orders={"visit": visits}, labels={"value":"Weighted burden (%)"})
                    st.plotly_chart(fig, use_container_width=True, key=prefix+"trajectory_"+kind)
            else:
                st.info("Choose the sample, visit and comparison columns in Configure, then run.")
        except ValueError as exc:
            st.error(str(exc))
