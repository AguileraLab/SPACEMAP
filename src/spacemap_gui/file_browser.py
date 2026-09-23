import os
import streamlit as st

def get_drives():
    """Get list of drives on Windows."""
    drives = []
    if os.name == 'nt':
        import string
        from ctypes import windll
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drives.append(f"{letter}:\\")
            bitmask >>= 1
    return drives

def render_file_browser(current_path, key_prefix, select_type='directory', file_extensions=None, show_hidden=False):
    """
    Renders a file browser and returns the selected path if a selection is made.
    
    Args:
        current_path (str): The starting path.
        key_prefix (str): Unique identifier for this browser instance.
        select_type (str): 'directory' or 'file'.
        file_extensions (list): Optional list of extensions to filter (e.g. ['.csv', '.txt']).
        show_hidden (bool): Whether to show hidden files/folders.
        
    Returns:
        str: The selected path if the user clicks "Select", otherwise None.
    """
    # Internal state for navigation
    nav_key = f"{key_prefix}_nav_path"
    if nav_key not in st.session_state:
        # Initialize with provided path or current working directory
        initial = str(current_path) if current_path and os.path.exists(current_path) else os.getcwd()
        if os.path.isfile(initial):
            initial = os.path.dirname(initial)
        st.session_state[nav_key] = initial

    current_dir = st.session_state[nav_key]
    
    # Ensure current_dir exists
    if not os.path.exists(current_dir):
        current_dir = os.getcwd()
        st.session_state[nav_key] = current_dir

    st.markdown(f"**Browsing:** `{current_dir}`")

    # Navigation Controls
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 4])
    
    with col_nav1:
        if st.button("Up", key=f"{key_prefix}_up"):
            parent = os.path.dirname(current_dir)
            if parent and parent != current_dir:
                st.session_state[nav_key] = parent
                st.rerun()
                
    with col_nav2:
        if os.name == 'nt':
            # Drive selector for Windows
            drives = get_drives()
            current_drive = os.path.splitdrive(current_dir)[0] + "\\"
            selected_drive = st.selectbox("Drive", drives, index=drives.index(current_drive) if current_drive in drives else 0, key=f"{key_prefix}_drive", label_visibility="collapsed")
            if selected_drive != current_drive:
                st.session_state[nav_key] = selected_drive
                st.rerun()
        else:
            # Linux/Mac Navigation
            c_home, c_root = st.columns(2)
            with c_home:
                if st.button("Home", key=f"{key_prefix}_home"):
                    st.session_state[nav_key] = os.path.expanduser("~")
                    st.rerun()
            with c_root:
                if st.button("ROOT", key=f"{key_prefix}_root"):
                    st.session_state[nav_key] = "/"
                    st.rerun()

    # List contents
    try:
        items = os.listdir(current_dir)
        # Filter hidden
        if not show_hidden:
            items = [i for i in items if not i.startswith('.')]
        
        items = sorted(items)
        
        dirs = [i for i in items if os.path.isdir(os.path.join(current_dir, i))]
        files = [i for i in items if os.path.isfile(os.path.join(current_dir, i))]
        
        # Filter files by extension
        if file_extensions:
            files = [f for f in files if any(f.lower().endswith(ext.lower()) for ext in file_extensions)]
            
    except PermissionError:
        st.error(f"Permission Denied: {current_dir}")
        dirs = []
        files = []
    except Exception as e:
        st.error(f"Error reading directory: {e}")
        dirs = []
        files = []

    # Display Directories
    st.markdown("##### Folders")
    
    # Use a selectbox for folders to save space
    if dirs:
        # Add a placeholder option so "Select This Folder" is unambiguous
        dir_options = ["(Stay in current)"] + dirs
        selected_dir = st.selectbox("Go to subfolder:", dir_options, key=f"{key_prefix}_dir_select")
        
        if selected_dir != "(Stay in current)":
            new_path = os.path.join(current_dir, selected_dir)
            st.session_state[nav_key] = new_path
            st.rerun()
    else:
        st.info("No subdirectories found.")

    # Display Files (if selecting file)
    selected_file = None
    if select_type == 'file':
        st.markdown("##### Files")
        if files:
            selected_file_name = st.selectbox("Select File:", [""] + files, key=f"{key_prefix}_file_select")
            if selected_file_name:
                selected_file = os.path.join(current_dir, selected_file_name)
        else:
            st.info("No matching files found.")

    st.markdown("---")
    
    # Action Buttons
    col_act1, col_act2 = st.columns([1, 1])
    
    with col_act1:
        if select_type == 'directory':
            if st.button(f"Select This Folder", key=f"btn_{key_prefix}_confirm", type="primary"):
                return current_dir
        else:
            # File selection
            if st.button(f"Select File", key=f"btn_{key_prefix}_confirm", type="primary", disabled=(not selected_file)):
                return selected_file
                
    with col_act2:
        if st.button("Cancel", key=f"{key_prefix}_cancel"):
            return "CANCEL"

    return None
