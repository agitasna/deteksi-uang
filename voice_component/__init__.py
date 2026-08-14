import streamlit.components.v1 as components


_component = components.declare_component(
    "voice_recognition",
    path="voice_component/frontend"
)


def voice_recognition(
    speak_text="",
    key=None
):

    return _component(
        speak_text=speak_text,
        default=None,
        key=key
    )
