import streamlit.components.v1 as components


_component = components.declare_component(
    "voice_recognition",
    path="voice_component/frontend"
)


def voice_recognition():

    return _component(
        default=None
    )