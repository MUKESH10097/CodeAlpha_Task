import gradio as gr
from deep_translator import GoogleTranslator

def translate(text, source, target):
    if not text.strip():
        return "Please enter some text."
    try:
        translated = GoogleTranslator(source=source, target=target).translate(text)
        return translated
    except Exception as e:
        return f"Error: {str(e)}"

languages = GoogleTranslator(source='auto', target='english').get_supported_languages(as_dict=True)
lang_names = list(languages.keys())

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🌐 Language Translator")

    with gr.Row():
        source_lang = gr.Dropdown(choices=lang_names, label="Source Language", value="english")
        target_lang = gr.Dropdown(choices=lang_names, label="Target Language", value="tamil")

    text_input = gr.Textbox(lines=3, placeholder="Enter text here...", label="Input Text")
    output = gr.Textbox(label="Translated Text")

    translate_btn = gr.Button("Translate")

    translate_btn.click(fn=translate, inputs=[text_input, source_lang, target_lang], outputs=output)

demo.launch()
