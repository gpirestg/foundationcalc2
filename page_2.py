import streamlit as st
from pathlib import Path
import datetime
import configparser


# Select and manage foundation loads template
def run():
    st.title("Loads Template File")
    
    # Define folder path
    #folder_path = Path(r"C:\Users\GDP\OneDrive - Tony Gee and Partners LLP\Documents\Automation\__2025__Digital__\DEV - WIP\P&E Streamlit Toms App\structural_app2\loads_template")
    folder_path = Path("structural_app2/loads_template/")
    files = list(folder_path.glob("*"))

    #st.set_page_config(layout="wide")
    st.subheader("📄 Select Foundation Loads Template File")

    selected_files = []

    # === FILE SELECTION ===
    if files:
        file_options = []
        for idx, f in enumerate(files):
            creation_time = datetime.datetime.fromtimestamp(f.stat().st_ctime)
            label = f"{f.name} – {creation_time.strftime('%d %b %Y %H:%M')}"
            file_options.append((label, f))

        for i, (label, file) in enumerate(file_options):
            if st.checkbox(label, key=f"file_{i}"):
                selected_files.append(file)

    # === FILE ACTIONS ===
    if len(selected_files) == 1:
        selected_file = selected_files[0]
        st.success(f"✅ Selected: {selected_file.name}")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Accept"):
                loads_file_path = selected_file.resolve()
                st.session_state.loads_file_path = str(loads_file_path)
                st.info(f"You Accepted: {selected_file.name}")

                # === Update settings.cfg with selected file ===
                config = configparser.ConfigParser()
                #cfg_path = "settings.cfg"
                cfg_path = "/mount/src/foundationcalc/structural_app2/settings.cfg"

                # Create new config or load existing one
                config.read(cfg_path)
                if "DEFAULT" not in config:
                    config["DEFAULT"] = {}

                config["DEFAULT"]["TEMPLATE"] = str(selected_file.name)  # or full path if you prefer

                with open(cfg_path, "w") as configfile:
                    config.write(configfile)

                st.success(f"`settings.cfg` updated with TEMPLATE = {selected_file.name}")


        with col2:
            with open(selected_file, "rb") as f:
                st.download_button("⬇️ Download", data=f, file_name=selected_file.name)

        with col3:
            if st.button("🗑️ Delete"):
                selected_file.unlink()
                st.warning(f"{selected_file.name} deleted.")
                st.rerun()

        upload_disabled = True  # Disable upload if one is selected

    elif len(selected_files) > 1:
        st.warning("⚠️ Please select only one file.")
        upload_disabled = True
    else:
        st.info("👈 Select a file to enable actions.")
        upload_disabled = False

    # === FILE UPLOAD SECTION ===
    # === ONLY SHOW UPLOAD IF NO FILE IS SELECTED ===
    if len(selected_files) == 0:
        st.divider()
        st.subheader("📤 Upload New Template")

        uploaded_file = st.file_uploader("Upload .xlsx file", type=["xlsx"])

        if uploaded_file:
            destination = folder_path / uploaded_file.name
            with open(destination, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.rerun()
