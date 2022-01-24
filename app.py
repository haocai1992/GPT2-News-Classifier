import streamlit as st
from gpt2_funcs import load_model, read_input, run_model

st.title("GPT-2 News Classifier")
st.write('This app uses GPT-2 language model to identify the category of any input news. It aims to save readers time accessing the news of their interest.')
st.markdown('The source code for this app can be found in this GitHub repo: [GPT2-News-Classifier](https://www.google.com).')

example_text = """
The US has ordered the relatives of its embassy staff in Ukraine to leave amid rising tension in the region. The State Department has also given permission for non-essential staff to leave and urged US citizens in Ukraine to consider departing. In a statement, it said there were reports that Russia is planning significant military action against Ukraine. Russia has denied claims that it is planning to invade Ukraine.
"""

input_text = st.text_area(
    label="Input/Paste News here:",
    value="",
    height=30,
    placeholder="Example:{}".format(example_text)
    )

if input_text == "":
    input_text = example_text

if st.button("Run GPT-2!"):
    if len(input_text) < 300:
        st.write("Please input more text!")
    else:
        with st.spinner("Running..."):
            model = load_model(path="./model/gpt2-text-classifier-model.pt")
            model_input = read_input(input_text)
            model_output = run_model(model, *model_input)
            st.write("Predicted News Category (with Probability):")
            st.write(model_output)