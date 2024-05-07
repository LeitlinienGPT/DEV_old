import streamlit as st
from faq import faq

def sidebar():
    with st.sidebar:
        st.markdown(
            "## Tipps zur Benutzung\n"
            "1. Stelle eine Frage an die [AWMF Leitlinien](https://register.awmf.org/de/leitlinien/aktuelle-leitlinien).\n"  
            "2. Stelle Rückfragen und Chatte mit den Leitlinien.\n"  
            "3. Verifiziere die Antwort anhand der Quellen.\n"
            "4. Setze die chat history zurück für eine neue Frage!\n"
        )

        st.markdown("---")
        st.markdown("# Über die App")
        st.markdown(
            "👨🏽‍⚕️LeitlinienGPT👩🏻‍⚕️ ermöglicht einen Chat mit den AWMF Leitlinien. \n"
            "Die Antwort ist möglichst qualitativ mit Referenz zur entsprechenden Leitlinie. "
        )
        st.markdown("Die Website ist immer offen für neue Anregungen. \n")
        st.markdown("Gebt mir gerne feedback unter marlon.hamm@outlook.de! \n")
        # st.markdown("Für Interessierte kann der Code auf [GitHub](XXX) eingesehen werden. ")
        
        st.markdown("Made by [Marlon Hamm](https://www.linkedin.com/in/m-hamm/) & [Paolo Oppelt](https://www.linkedin.com/in/paolo-oppelt/) 😃")

        st.markdown("---")

        faq()
