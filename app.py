# ======================
# LANGUAGE
# ======================
lang = st.sidebar.selectbox("🌐 Language", ["English", "Français"])

def t(en, fr):
    return en if lang == "English" else fr

st.sidebar.title(t("Settings", "Paramètres"))

# ======================
# STYLE (UPDATED 🔥)
# ======================
st.markdown("""
<style>
...
</style>
""", unsafe_allow_html=True)
