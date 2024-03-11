from dataclasses import dataclass
import requests
import streamlit as st  # type: ignore

st.set_page_config(layout="wide")

MAMMOTH = '🦣'
FAT_UNDER = '▁'

ARCHITECTURE_HTML = """
<h3>Decoder</h3>
<div class="arch">
    <div class="layer langspec">
        <div class="compo compo-en">en</div>
        <div class="compo compo-fr">fr</div>
        <div class="compo compo-ru">ru</div>
    </div>
    <div class="layer taskspec">
        <div class="compo compo-defmod">defmod</div>
        <div class="compo compo-pargen">pargen</div>
        <div class="compo compo-texsim">texsim</div>
        <div class="compo compo-translate">translate</div>
    </div>
    <div class="layer langspec">
        <div class="compo compo-en">en</div>
        <div class="compo compo-fr">fr</div>
        <div class="compo compo-ru">ru</div>
    </div>
</div>
<h3>Encoder</h3>
<div class="arch">
    <div class="layer full">
        <div class="compo">fully shared</div>
    </div>
</div>

<style>
    .arch {
        display: grid;
        row-gap: 5px;
    }
    .layer {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        column-gap: 5px;
    }
    .taskspec {
        grid-template-columns: repeat(4, 1fr);
    }
    .langspec {
        grid-template-columns: repeat(3, 1fr);
    }
    .full {
        grid-template-columns: 1fr;
    }
    .compo {
        display: grid;
        border: 2px solid gray;
        text-align: center;
    }
    .compo-__LANG__ {
        border: 5px solid green;
        font-weight: bold;
    }
    .compo-__TASK__ {
        border: 5px solid red;
        font-weight: bold;
    }
    .full div {
        border: 5px solid blue;
    }
</style>
"""


def render(template, model_task):
    task, lang = model_task.split('_')
    if task == 'translate':
        _, lang = lang.split('-')
    template = template.replace('__TASK__', task)
    template = template.replace('__LANG__', lang)
    return template


@dataclass
class ModelSpecs:
    id: int
    task: str
    loaded: bool

    @staticmethod
    def format_model(model):
        return model.task


class Translator:
    def __call__(self):
        st.title(f'{MAMMOTH} MAMMOTH translation demo')
        col1, col2 = st.columns([0.6, 0.4], gap="large")
        with col1:
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
        with col2:
            st.markdown(
                render(ARCHITECTURE_HTML, model.task),
                unsafe_allow_html=True,
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
