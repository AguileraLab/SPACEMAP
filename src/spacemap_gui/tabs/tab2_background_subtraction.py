import streamlit as st
import os
from pathlib import Path
import re
import pandas as pd
import time
import csv
from datetime import datetime
from typing import Dict, Optional, List, TextIO
from spacemap.utils import log_time, hex_to_rgb
from spacemap.background_subtraction_utils import get_channel_info, subtract_background_and_save, is_supported_image_file, normalize_image_stem
from spacemap_gui.file_browser import render_file_browser
from spacemap_gui.ui_utils import directory_input_with_browse
from spacemap_gui.process_log import record_process_event
from spacemap_gui.tabs.tab1_registration import get_image_metadata_log, _prefix_keys

def render():
    from spacemap_gui.ui_utils import require_active_project
    project_root = require_active_project()
    if project_root is None:
        return

    st.header("Step 2: Background Subtraction")
    st.markdown(
        "Two routes to the same goal. Subtract blank cycles when the panel provides them; otherwise "
        "estimate each marker's background from its own pixel histogram."
    )

    blanks_tab, detect_tab = st.tabs(
        ["Blank Cycles", "Detect Background"]
    )
    with blanks_tab:
        _render_blank_subtraction(project_root)
    with detect_tab:
        from spacemap_gui.tabs.tab2_background_detection import render as render_detection

        render_detection(project_root)

    st.markdown("---")
    if st.button("Save Project", key="tab2_save_project", width='stretch'):
        from spacemap_gui.persistence import save_active_project_state
        save_active_project_state("tab2_background_subtraction")
        st.success("Project settings saved to configuration file.")


def _render_blank_subtraction(project_root):
    st.markdown("Remove noise by subtracting blank channels from marker channels.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("1. Directories")
        
        # Source Selection
        input_source = st.radio(
            "Select Input Source",
            ["Registered Images (OME-TIFF)", "Raw CODEX Images (QPTIFF)"],
            horizontal=True,
            help="Choose between processed OME-TIFFs (from Tab 1) or raw QPTIFFs."
        )
        
        if input_source == "Registered Images (OME-TIFF)":
            default_input_dir = st.session_state.get('registration_output_dir', str(project_root / 'outputs/registered'))
            input_dir = directory_input_with_browse(
                label="Input Directory (Registered Images)",
                text_key="tab2_input_dir_reg",
                default_value=str(default_input_dir),
                browser_key_prefix="tab2_input_dir_reg",
                start_dir_fallback=str(project_root),
                help_text="Directory containing registered OME-TIFF images."
            )
            suffixes = ('.ome.tiff', '.ome.tif')
        else:
            default_input_dir = str(project_root / 'data/codex')
            input_dir = directory_input_with_browse(
                label="Input Directory (Raw CODEX Images)",
                text_key="tab2_input_dir_raw",
                default_value=str(default_input_dir),
                browser_key_prefix="tab2_input_dir_raw",
                start_dir_fallback=str(project_root),
                help_text="Directory containing raw QPTIFF images."
            )
            suffixes = ('.qptiff',)
        
        input_path = Path(str(input_dir))
        
            
        selected_file = None
        
        if input_path.exists():
            files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(suffixes) and is_supported_image_file(f)])
            if files:
                selected_file = st.selectbox("Select Image", files, help="Choose the image to process.")
            else:
                st.warning(f"No supported files found for the selected source in directory.")
        else:
            st.error("Input directory not found.")

        if selected_file:
            file_path = input_path / selected_file
            file_tag = re.sub(r"[^A-Za-z0-9_]+", "_", Path(selected_file).stem)
            
            # Auto-Detection
            st.subheader("Channel Configuration")
            
            if st.button("Detect Channels", help="Read metadata to identify channels."):
                try:
                    channel_names, channel_colors, channel_meta, mpp = get_channel_info(str(file_path))
                    st.session_state['bg_channels'] = channel_names
                    st.session_state['bg_colors'] = channel_colors
                    st.session_state['bg_meta'] = channel_meta
                    st.session_state['bg_mpp'] = mpp
                    
                    # Clear previous per-channel widget state to force re-detection
                    for key in list(st.session_state.keys()):
                        if isinstance(key, str) and (key.startswith('bg_blank_') or key.startswith('bg_color_') or key.startswith('bg_picker_') or key.startswith('bg_text_')):
                            del st.session_state[key]
                            
                    st.success(f"Detected {len(channel_names)} channels.")
                except Exception as e:
                    st.error(f"Error detecting channels: {e}")

            if 'bg_channels' in st.session_state:
                channels = st.session_state['bg_channels']
                original_colors = st.session_state['bg_colors']
                meta = st.session_state.get('bg_meta', [{} for _ in channels])
                
                # Identify potential blanks
                blank_candidates = [c for c in channels if re.search(r'Blank|Empty|Control', c, re.IGNORECASE)]
                blank_options = ["None"] + blank_candidates

                st.markdown("Configure colors and assign blank channels for subtraction.")

                # Header
                c1, c2, c3, c4, c5, c6 = st.columns([3, 2, 1, 1.5, 1, 3])
                c1.markdown("**Channel**")
                c2.markdown("**Hex Code**")
                c3.markdown("**Color**")
                c4.markdown("**Dye**")
                c5.markdown("**Exp**")
                c6.markdown("**Blank Channel**")

                # Storage for selections
                marker_to_blank_map = {}
                edited_colors = []

                for i, (ch, col, m) in enumerate(zip(channels, original_colors, meta)):
                    # --- Color Logic (from Tab 1) ---
                    # Initialize session state for this channel if not exists
                    color_key = f"bg_color_{file_tag}_{i}"
                    if color_key not in st.session_state:
                        # Ensure col is hex string
                        hex_col = '#{:02x}{:02x}{:02x}'.format(*col) if isinstance(col, tuple) else col
                        st.session_state[color_key] = hex_col
                    
                    # Widget keys
                    picker_key = f"bg_picker_{file_tag}_{i}"
                    text_key = f"bg_text_{file_tag}_{i}"
                    
                    # Force sync widget state to main color key
                    st.session_state[picker_key] = st.session_state[color_key]
                    st.session_state[text_key] = st.session_state[color_key]

                    # Callbacks
                    def update_from_picker(key=color_key, p_key=picker_key, t_key=text_key):
                        st.session_state[key] = st.session_state[p_key]
                        st.session_state[t_key] = st.session_state[p_key]
                        
                    def update_from_text(key=color_key, t_key=text_key, p_key=picker_key):
                        st.session_state[key] = st.session_state[t_key]
                        st.session_state[p_key] = st.session_state[t_key]

                    # --- Blank Detection Logic ---
                    dye = str(m.get('dye', '')).lower()
                    default_blank = "None"
                    
                    # Don't assign blanks to blanks themselves
                    if ch not in blank_candidates:
                        for b in blank_candidates:
                            b_lower = b.lower()
                            if 'atto550' in dye and 'atto550' in b_lower:
                                default_blank = b
                                break
                            elif 'af750' in dye and 'af750' in b_lower:
                                default_blank = b
                                break
                            elif ('cy5' in dye or 'af647' in dye) and ('cy5' in b_lower or 'af647' in b_lower):
                                default_blank = b
                                break
                    
                    # Session state for blank selection to persist
                    blank_key = f"bg_blank_{file_tag}_{i}"
                    if blank_key not in st.session_state:
                        st.session_state[blank_key] = default_blank

                    # --- Render Row ---
                    c1, c2, c3, c4, c5, c6 = st.columns([3, 2, 1, 1.5, 1, 3])
                    c1.text(ch)
                    
                    c2.text_input(
                        f"txt_{i}", 
                        label_visibility="collapsed", 
                        key=text_key,
                        on_change=update_from_text
                    )
                    
                    c3.color_picker(
                        f"cp_{i}", 
                        label_visibility="collapsed", 
                        key=picker_key,
                        on_change=update_from_picker
                    )
                    
                    c4.text(m.get('dye', '-'))
                    c5.text(m.get('exposure', '-'))
                    
                    selected_blank = c6.selectbox(
                        f"blank_{i}",
                        blank_options,
                        key=blank_key,
                        label_visibility="collapsed"
                    )
                    
                    # Collect data
                    edited_colors.append(hex_to_rgb(st.session_state[color_key]))
                    if selected_blank != "None":
                        marker_to_blank_map[ch] = selected_blank

                # Convert marker_to_blank_map to blank_mapping format {Blank: [Markers]}
                final_mapping = {}
                for marker, blank in marker_to_blank_map.items():
                    if blank not in final_mapping:
                        final_mapping[blank] = []
                    final_mapping[blank].append(marker)
                
                # Output Config
                # Prefer tab2_output_dir if set, else bg_subtraction_output_dir, else default fallback
                fallback = str(project_root / 'outputs/background_subtracted')
                default_out = st.session_state.get('tab2_output_dir', st.session_state.get('bg_subtraction_output_dir', fallback))

                output_dir = directory_input_with_browse(
                    label="Output Directory (Background-Subtracted Images)",
                    text_key="tab2_output_dir",
                    default_value=str(default_out),
                    browser_key_prefix="tab2_output_dir",
                    start_dir_fallback=str(project_root),
                    help_text="Directory where processed images will be saved."
                )

                if "tab2_create_log" not in st.session_state:
                    st.session_state["tab2_create_log"] = True
                create_log = st.checkbox(
                    "Create Processing Log (CSV)",
                    help="If enabled, writes detailed CSV logs under Output Directory/log/bg_session_YYMMDD_HHMMSS",
                    key="tab2_create_log",
                )
                
                if st.button("Run Background Subtraction", type="primary", help="Execute background subtraction process."):
                    status_container = st.status("Processing...", expanded=True)
                    run_subtraction(
                        file_path, 
                        output_dir, 
                        final_mapping, 
                        channels, 
                        edited_colors, 
                        st.session_state.get('bg_mpp'),
                        status_container,
                        create_log
                    )

    with col2:
        st.subheader("Metadata Inspection")
        if selected_file:
            # Reconstruct file path for this scope
            file_path_inspect = input_path / selected_file
            
            if st.button("Inspect Metadata", help="View metadata tags like Fluorophores and Bands."):
                try:
                    import tifffile
                    with tifffile.TiffFile(file_path_inspect) as tif:
                        page = tif.pages[0]
                        desc = getattr(page, 'description', '')
                        
                        if desc and "PerkinElmer-QPI-ImageDescription" in desc:
                            st.success("PerkinElmer Metadata Found")
                            
                            fluors = re.findall(r"<Fluor>(.*?)</Fluor>", desc)
                            if fluors:
                                st.write("**Fluorophores:**", fluors)
                            else:
                                st.warning("No <Fluor> tags found.")
                                
                            bands = re.findall(r"<Band>(.*?)</Band>", desc)
                            if bands:
                                st.write("**Bands:**", bands)
                        else:
                            st.info("No PerkinElmer metadata found in description.")
                except Exception as e:
                    st.error(f"Error inspecting metadata: {e}")

        st.info("ℹ️ **Help:** Select an image, detect channels, identify 'Blank' channels, and assign them to specific markers for subtraction.")
        st.caption(
            "No blank cycles in this panel? Use Detect Background to estimate it from the image "
            "and export the threshold table RESOLVE reads."
        )

def run_subtraction(input_path, output_dir_str, mapping, channel_names, channel_colors, mpp, status_container, create_log):
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{normalize_image_stem(input_path)}_bg_subtracted.ome.tiff"
    output_path = output_dir / output_filename

    def log_step(msg):
        ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        full_msg = f"{ts} {msg}"
        status_container.write(full_msg)
        print(full_msg) # Terminal log
        record_process_event("tab2", "Background subtraction step", message=msg)

    # Logging Setup
    session_dir: Optional[str] = None
    log_path: Optional[str] = None
    session_start_dt = datetime.now()
    log_data: Dict[str, str] = {}
    
    if create_log:
        log_root = output_dir / "log"
        log_root.mkdir(exist_ok=True)
        session_dir = log_root / f"bg_session_{session_start_dt.strftime('%y%m%d_%H%M%S')}"
        session_dir.mkdir(exist_ok=True)
        log_path = session_dir / "process_log.csv"
        
        # Initial Log Data
        log_data = {
            "Date": session_start_dt.strftime("%Y-%m-%d"),
            "Time": session_start_dt.strftime("%H:%M:%S"),
            "Input_File": str(input_path),
            "Output_File": str(output_path),
        }
        # Input Metadata
        try:
            in_meta = get_image_metadata_log(str(input_path))
            log_data.update(_prefix_keys("Input_", in_meta))
        except Exception as e:
            from spacemap_gui.ui_utils import log_exception
            log_exception("reading input image metadata", e)

    status_container.update(label="Starting background subtraction...", state="running")
    record_process_event(
        "tab2",
        "Background subtraction started",
        details={"input": str(input_path), "output": str(output_path), "blank_channels": list(mapping)},
    )
    log_step(f"Input: {input_path.name}")
    log_step(f"Output: {output_filename}")
    start_time = time.time()
    
    try:
        subtract_background_and_save(
            input_path=str(input_path),
            output_path=str(output_path),
            blank_mapping=mapping,
            channel_names=channel_names,
            channel_colors=channel_colors,
            mpp=mpp
        )
        elapsed = time.time() - start_time
        log_step(f"Done in {elapsed:.2f}s")
        
        status_container.update(label="Complete!", state="complete", expanded=False)
        st.success(f"Saved to {output_path}")
        st.toast("Background Subtraction Complete!", icon="✅")
        st.session_state['bg_subtraction_output_dir'] = str(output_dir) # Sync legacy key if needed
        output_metadata = get_image_metadata_log(str(output_path)) if output_path.exists() else {}
        record_process_event(
            "tab2",
            "Background subtraction complete",
            status="complete",
            duration_s=elapsed,
            details={"input": str(input_path), "output": str(output_path), "output_metadata": output_metadata},
        )
        
        if create_log and log_path:
            log_data["Processing_Time_Sec"] = f"{elapsed:.3f}"
            log_data["Status"] = "Success"
            # Output Metadata
            try:
                out_meta = get_image_metadata_log(str(output_path))
                log_data.update(_prefix_keys("Output_", out_meta))
            except:
                pass
            
            # Write Log
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(log_data.keys()))
                writer.writeheader()
                writer.writerow(log_data)
        
    except Exception as e:
        status_container.update(label="Error", state="error")
        st.error(f"Error: {e}")
        log_step(f"ERROR: {e}")
        record_process_event(
            "tab2",
            "Background subtraction failed",
            message=str(e),
            status="error",
            duration_s=time.time() - start_time,
            details={"input": str(input_path), "output": str(output_path)},
        )
        
        if create_log and log_path:
            log_data["Status"] = f"Error: {e}"
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(log_data.keys()))
                writer.writeheader()
                writer.writerow(log_data)

if __name__ == "__main__":
    from spacemap_gui.ui_utils import require_active_project
    require_active_project()
    render()
