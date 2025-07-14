import streamlit as st
import configparser
import pandas as pd
import os

def run():
    st.title("Loads Template Sheet")

    # Define where your template files are located
    #TEMPLATE_FOLDER = r"C:\Users\GDP\OneDrive - Tony Gee and Partners LLP\Documents\Automation\__2025__Digital__\DEV - WIP\P&E Streamlit Toms App\structural_app2\loads_template"
    TEMPLATE_FOLDER = "/mount/src/foundationcalc/structural_app2/loads_template/"

    # === Load TEMPLATE from settings.cfg ===
    #cfg_path = "settings.cfg"
    cfg_path = "/mount/src/foundationcalc/structural_app2/settings.cfg"
    config = configparser.ConfigParser()

    if not os.path.exists(cfg_path):
        st.error(f"`{cfg_path}` not found.")
        return

    config.read(cfg_path)
    template_file_name = config["DEFAULT"].get("TEMPLATE", "")
    full_template_path = os.path.join(TEMPLATE_FOLDER, template_file_name)
    
    if not os.path.exists(full_template_path):
        st.error(f"Excel file not found at: `{full_template_path}`")
        return

    st.success(f"Loaded Excel file: `{template_file_name}`")
    # Optional: show current selected sheet from config
    current_sheet = config["DEFAULT"].get("SHEET", "Not set")
    st.info(f"📄 Currently selected sheet: `{current_sheet}`")

    # === List available sheet names ===
    try:
        xls = pd.ExcelFile(full_template_path)
        sheet_names = xls.sheet_names
    except Exception as e:
        st.error(f"Failed to read Excel file: {e}")
        return

    # Safe init
    if "sheet_loaded" not in st.session_state:
        st.session_state.sheet_loaded = False

    # Select sheet
    selected_sheet = st.selectbox("Select a sheet to preview", sheet_names)

    # Load button
    if st.button("📄 Load Sheet"):
        try:
            df = pd.read_excel(full_template_path, sheet_name=selected_sheet)
            st.dataframe(df)
            st.session_state.sheet_loaded = True
        except Exception as e:
            st.error(f"Error reading sheet: {e}")
            st.session_state.sheet_loaded = False

    # Accept button only after successful load
    if st.session_state.get("sheet_loaded"):
        if st.button("✅ Accept Sheet"):
            config["DEFAULT"]["SHEET"] = selected_sheet
            with open(cfg_path, "w") as configfile:
                config.write(configfile)
            st.success(f"✅ Sheet name `{selected_sheet}` saved to `settings.cfg`")

