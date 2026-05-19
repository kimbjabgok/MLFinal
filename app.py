from dragdiff_repro.ui.gradio_app import build_demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True, debug=True, show_api=False)
