import streamlit as st
from openai import OpenAI
import time
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from retry import retry
import google.generativeai as genai
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

# Configure API key
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    #raise EnvironmentError("GOOGLE_API_KEY not set in .env")
    api_key = st.secrets("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

PROMPT = \
'''
    You are a career‑coach assistant. Below are EXAMPLES of the format to follow.
    Each example has a Question followed by the Answer. 
    If a question is outside job‑seeking topics, politely refuse. After the examples, answer the new question.

    Example 1:
    Question: How can I improve my resume to stand out?
    Answer: Quantify your achievements, tailor each section to the job description, and include keywords that applicant‑tracking systems (ATS) scan for.

    Example 2:
    Question: What’s the weather like today?
    Answer: I’m sorry—I can only provide advice on job searching, interview preparation, and skill development. Please ask a question related to those areas.

    Now answer the following:
    Question: {}
    Answer:
'''

SYSTEM_INSTRUCTION = \
'''
You are a career coach specializing in job search strategies, interview preparation, and skill development. 
Answer ONLY questions within that domain; otherwise, politely refuse as in the examples.
'''

# We do not set any safety settings
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Gemini model
def create_gemini_model(system_instruction: str = "You are a helpful assistant"):
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system_instruction
    )

# We call the model for inference
@retry(tries=5, delay=1, backoff=2)
def generate_gemini_response(messages: list, system_instruction: str = "You are a helpful assistant") -> str:
    model = create_gemini_model(system_instruction)
    max_output_tokens = 8096
    response = model.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(max_output_tokens=max_output_tokens),
        safety_settings=SAFETY_SETTINGS,
        stream=False
    )
    return response.text

placeholderstr = "Please input your question for the coach"
user_name = "Didier"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.05)

st.set_page_config(
    page_title='Job Seeking App',
    page_icon="💼",
    layout='wide',
    initial_sidebar_state='auto'
)

# Show title and description.
st.title(f"👩🏻‍💼 {user_name}'s Job Seeking App")

#def main():
tab_chat, tab_extra = st.tabs(["👩🏻‍💼 Job Seeking Coaching", "💼 Dashboard for Job Opportunities"])

with tab_chat:
    
    st.write("### Ask for advice to our extremely smart coach if you are about to apply for any jobs, and you have questions about the process.")
    st_c_chat = st.container(border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                if user_image:
                    st_c_chat.chat_message(msg["role"],avatar=user_image).markdown((msg["content"]))
                else:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            elif msg["role"] == "assistant":
                st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            else:
                try:
                    image_tmp = msg.get("image")
                    if image_tmp:
                        st_c_chat.chat_message(msg["role"],avatar=image_tmp).markdown((msg["content"]))
                except:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))

    def generate_response(prompt):
        # output = "Your text in binary is: "
        # output += ' '.join(format(x, 'b') for x in bytearray(prompt, 'utf-8'))
        msgs = []
        msgs.append(PROMPT.format(prompt))
        response = generate_gemini_response(
            messages=msgs,
            system_instruction=SYSTEM_INSTRUCTION
        )
        
        return response
        
        
    # Chat function section (timing included inside function)
    def chat(prompt: str):
        st_c_chat.chat_message("user",avatar=user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt)
        # response = f"You type: {prompt}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))

    
    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)

with tab_extra:
    st.header("💼 Linkedin Job Opportunities' Dashboard")
    st.write("#### You can filter for the areas you are interested in, and visualize a summary in the shape of a word cloud of what is more commonly encountered in their job descriptions.")

    
    df_raw = pd.read_csv("./data/linkedin_jobs_april_20.csv")
    st.write(f"##### There are a total of {len(df_raw)} job opportunities in our database crawled on April 20, 2025.")

    # Dropping ID
    df = df_raw.drop(columns=["job_id"])

    # Renaming the columns
    rename_map = {
        "job_title":            "Job Title",
        "job_description":      "Job Description",
        "job_url":              "Job URL",
        "job_location":         "Job Location",
        "time_posted":          "Time Posted",
        "num_applicants":       "Applicants",
        "seniority":            "Seniority",
        "employment_type":      "Employment Type",
        "category_minor":       "Category (Minor)",
        "category_major":       "Category (Major)",
        "work_mode":            "Work Mode",
        "company_name":         "Company",
        "company_linkedin_url": "Company LinkedIn URL",
        "company_about":        "Company About",
        "company_field":        "Company Field",
        "company_size":         "Company Size",
        "company_address":      "Company Address",
        "company_type":         "Company Type",
        "company_founded":      "Company Founded",
        "company_specialties":  "Company Specialties",
        "company_website":      "Company Website",
    }
    df.rename(columns=rename_map, inplace=True)

    # Columns for filtering
    filter_cols = [
        "Job Location",
        "Seniority",
        "Employment Type",
        "Category (Minor)",
        "Category (Major)",
        "Work Mode",
        "Company",
    ]

    st.subheader("Filter the Data")

    selected_column = st.selectbox("Select column to filter by", filter_cols)

    unique_values = df[selected_column].dropna().unique()
    selected_value  = st.selectbox("Select value", unique_values)

    filtered_df = df[df[selected_column] == selected_value]

    st.dataframe(filtered_df, use_container_width=True) 

    # Generating the word cloud based on the filter
    st.subheader("Wordcloud of the Filtered Job Descriptions")
    def tokenize(text: str):
        return re.findall(r"\b\w+\b", text.lower())
    
    row_sets = [set(tokenize(desc)) for desc in filtered_df["Job Description"]]

    common_words = set.intersection(*row_sets) if row_sets else set()

    custom_stopwords = STOPWORDS.union(common_words)

    all_job_descriptions = " ".join(filtered_df["Job Description"].tolist())

    wc = WordCloud(
            width=1600, height=800, scale=2,
            background_color="white", colormap="Dark2",
            stopwords=custom_stopwords,
            prefer_horizontal=0.9, max_words=300,
            random_state=42
        ).generate(all_job_descriptions)
    
    st.image(wc.to_array(), use_container_width=True)     

# if __name__ == "__main__":
#     main()
