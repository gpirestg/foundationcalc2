import streamlit as st
import page_1
import page_2
import page_3

# Set a simple username and password (you could improve this later)
USERNAME = "admin"
PASSWORD = "1234"

# Initialize session state for login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# === LOGIN PAGE ===
if not st.session_state.authenticated:
    # Only show background and logo during login
    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://www.tonygee.com/wp-content/uploads/2025/01/p.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        label {
            color: white !important;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align: center; margin-top: 10px;">
            <img src="https://www.tonygee.com/wp-content/uploads/2021/06/SocialImg.jpg"
                 style="width: 200px; max-width: 90%; height: auto; transform: translateX(-10px);">
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h1 style='color:white; text-align:center;'>Login Page</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.authenticated = True
            st.success("Logged in successfully.")
            st.rerun()
        else:
            st.error("Invalid credentials")

# === MAIN APP (after login) ===
if st.session_state.authenticated:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Optimisation Page", "Loads Template File", "Loads Template Sheet", "🔓 Logout"])

    if page == "Home":
        st.title("📊 Structural Foundation Optimisation Tool")
        st.write("Welcome! Use the sidebar to navigate through the application!")
        st.markdown("---")

        st.subheader("Optimisation Page:")
        st.write("Run foundation size optimisation based on the current input parameters.")

        st.subheader("Loads Template File:")
        st.write("Select and accept the Excel file containing the load data.")

        st.subheader("Loads Template Sheet:")
        st.write("Choose which sheet from the Excel file to work with and preview its content.")

        st.markdown("---")

    elif page == "Optimisation Page":
        page_1.run()

    elif page == "Loads Template File":
        page_2.run()

    elif page == "Loads Template Sheet":
        page_3.run()

    elif page == "🔓 Logout":
        st.session_state.authenticated = False
        st.rerun()
