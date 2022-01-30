import streamlit as st
import time
import json
import boto3
import os

def predict(text):
    data = {"text": text}
    sagemaker_client = boto3.client('sagemaker-runtime', region_name='us-east-1')
    try:
        response = sagemaker_client.invoke_endpoint(
            EndpointName=os.environ['SAGEMAKER_ENDPOINT_NAME'], 
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(data)
        )
    except sagemaker_client.exceptions.ClientError as e:
        if "ExpiredTokenException" in str(e):
            raise Exception("""
                ExpiredTokenException.
                You can refresh credentials by restarting the Docker container.
                Only occurs during development (due to passing of temporary credentials).
            """)
        else:
            raise e
    body_str = response['Body'].read().decode("utf-8")
    body = json.loads(body_str)
    return body['text']


def main():
    st.title("GPT-2 News Classifier")
    st.write('This app uses GPT-2 language model to identify the category of any input news. It aims to save readers time accessing the news of their interest.')
    st.markdown('The source code for this app can be found in this GitHub repo: [GPT2-News-Classifier](https://github.com/haocai1992/GPT2-News-Classifier).')

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
                model_output = predict(input_text)
                st.write("Predicted News Category (with Probability):")
                st.write(model_output)

if __name__ == "__main__":
    debug = os.getenv('DASHBOARD_DEBUG', 'false') == 'true'
    if debug:
        main()
    else:
        try:
            main()
        except Exception as e:
            st.error('Internal error occurred.')