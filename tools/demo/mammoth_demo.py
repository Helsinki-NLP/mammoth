from dataclasses import dataclass
import requests
import streamlit as st

MAMMOTH = '🦣'
FAT_UNDER = '▁'

@dataclass
class ModelSpecs:
    id: int
    task: str
    loaded: bool

    @staticmethod
    def format_model(model):
        suffix = ' <<' if model.loaded else ''
        return f'{model.task}{suffix}'


class Translator:
    def __call__(self):
        st.title(f'{MAMMOTH} MAMMOTH translation demo')
        with st.form('Translation demo'):
            model = st.selectbox(
                'Model',
                st.session_state.models,
                format_func=ModelSpecs.format_model,
            )
            source = st.text_area(
                'Source text',
                height=None,
            )
            submitted = st.form_submit_button('▶️  Translate')
            if submitted:
                target_text = self.submit(source, model.id)
            else:
                target_text = ''
            st.text_area(
                'Target text',
                value=target_text,
                height=None,
            )

    def submit(self, query, model):
        try:
            response = requests.request(
                'POST',
                'http://127.0.0.1:5000/translator/translate',
                json=[{
                    'src': query,
                    'id': model,
                }],
            )
            data = response.json()
        except Exception as e:
            print(response.content)
            raise e
        tokenized = data[0][0]['tgt']
        return self.detokenize(tokenized)

    def detokenize(self, tokenized):
        result = tokenized.replace(' ', '').replace(FAT_UNDER, ' ')
        return result

    def get_models(self):
        data = requests.request(
            'GET',
            'http://127.0.0.1:5000/translator/models',
        ).json()
        models = [
            ModelSpecs(model_specs['model_id'], model_specs['opts']['task_id'], model_specs['loaded'])
            for model_specs in data
        ]
        st.session_state.models = models
        return models


translator = Translator()
if 'models' not in st.session_state:
    st.session_state.models = translator.get_models()
translator()
