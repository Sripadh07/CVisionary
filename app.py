import logging
import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
from tavily import TavilyClient
import fitz
import docx
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import time
import re
import mimetypes
import chardet
import json
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def classify_resume_length(resume_text):
    words=[word for word in re.split(r'\W+',resume_text) if word]
    word_count=len(words)
    if word_count < 300:
        return "Small Resume"
    elif 300 <= word_count <= 500:
        return "Medium Resume"
    else:
        return "Large Resume"



def check_encoding(file):
    detect = chardet.detect(file)
    return detect["encoding"]

def get_file_extension(uploaded_file):
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    
    if mime_type:
        if "pdf" in mime_type:
            return "pdf"
        elif "word" in mime_type or "msword" in mime_type:
            return "docx"
        elif "text" in mime_type:
            return "txt"
        elif "image" in mime_type:
            return "image"
        elif "tex" in mime_type:
            return "tex"
    
    return uploaded_file.name.split(".")[-1].lower()

def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(input)
    cleared_text = response.text.replace("```", "").replace("json", "").strip()
    return cleared_text

def input_pdf_text(uploaded_file):
    file_type = get_file_extension(uploaded_file)
    if file_type == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return '\n'.join([page.get_text("text") for page in doc])
    elif file_type == "docx":
        doc = docx.Document(uploaded_file)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif file_type == "txt":
        file= uploaded_file.read()
        detect=check_encoding(file)
        try:
            return file.decode(detect).strip()
        except (UnicodeDecodeError,TypeError):
            return "could not decode the file"
    elif file_type == "image":
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image).strip()
    elif file_type == "tex":
        content=uploaded_file.read().decode("utf-8")
        cleaned_content = re.sub(r"\\(textbf|textit|texttt|text|section|subsection|subsubsection|paragraph|subparagraph|chapter|part|documentclass|usepackage|begin|end|item|itemize|enumerate|description|table|figure|caption|label|ref|cite|bibliography|thebibliography|author|title|date|maketitle|includegraphics|textwidth|linewidth|hline|vline|centering|textwidth|linewidth)\{.*?\}", "", content)
        cleaned_content = re.sub(r"\\[a-zA-Z]+\{(.*?)\}", r"\1", cleaned_content)
        cleaned_content = re.sub(r"\{(.*?)\}", r"\1", cleaned_content)
        cleaned_content=re.sub(r"\s+", " ", cleaned_content).strip()
        return cleaned_content
    else:
        return None

def check_role_existence(job_role, company):
    prompt = (
        f"Does the company {company} hire for the role of {job_role}? "
        "Respond with only 'Yes' or 'No'. Avoid any additional text."
    )
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    answer = response.text.strip().lower()
    return answer == "yes"


def get_job_description(job_role, company):
    query = f"job description for {job_role} at {company}"
    
    response = client.search(query=query, max_results=2)
    
    if not response.get("results"):
        return "Job description not found."
    
    # Combine content from all search results
    job_description = [result["content"] for result in response["results"] if result.get("content")]
    
    if not job_description:
        return "Job description not available."
    
    return job_description

def generate_prompt(resume_text, job_role, company, role_exists):
    if role_exists:
        job_description = get_job_description(job_role, company)
        prompt=f'''Act as an advanced ATS with expertise in tech fields.Evaluate the resume for the specified job role at the given company.The role exists,so tailor the evaluation strictly to the company's job description and requirements.Provide:1.A percentage match based on the job description.2.A list of missing keywords essential for the role.3.A personalized feedback message using direct language like "your resume","you should","your skills",etc.,highlighting strengths,areas for improvement,and specific actions to align with the job.Return the response in JSON:"JD_Match":"%","MissingKeywords":[],"Profile_Summary":"Personalized feedback","Role_Exists":"Yes".The details are Resume:{resume_text},job_role:{job_role},company:{company},job_description:{job_description}" '''
    else:
        prompt =f'''Act as an advanced ATS with expertise in tech fields.Evaluate the resume for the specified job role.Evaluate based on general industry standards.Provide:1.A percentage match based on industry standards.2.A list of missing keywords relevant to the job.3.A personalized feedback message using direct language like "your resume","you should","your skills",etc.,highlighting strengths,areas for improvement,and specific actions to align with the job.and set the role_exists to no. Return the response in JSON:"JD_Match":"%","MissingKeywords":[],"Profile_Summary":"Personalized feedback","Role_Exists":"No".The details are resume:{resume_text},job_role:{job_role}"'''
    return prompt
chat_styles = """
<style>
/* Full Page Background */
html, body {
    height: 100vh;
    width: 100vw;
    margin: 0;
    padding: 0;
    background: #f5f5f5;
    color: #333;
    font-family: 'Poppins', sans-serif;
}

/* Chat Bubbles */
.chat-bubble {
    padding: 12px;
    margin: 8px 0;
    border-radius: 10px;
    max-width: 75%;
    word-wrap: break-word;
    font-size: 15px;
}
.user-message {
    background: #007bff;
    color: white;
    text-align: right;
    align-self: flex-end;
    margin-left: auto;
}
.bot-message {
    background: #e9ecef;
    color: #333;
    text-align: left;
    align-self: flex-start;
}

/* Title */
h1 {
    text-align: center;
    font-size: 2.2em;
    color: #007bff;
    margin-bottom: 20px;
}

/* Input Fields */
input[type="text"], textarea {
    width: 100%;
    background: #ffffff;
    border: 1px solid #ccc;
    color: #333;
    padding: 10px;
    border-radius: 8px;
    outline: none;
    transition: border 0.2s ease-in-out;
    font-size: 14px;
}
input[type="text"]:focus, textarea:focus {
    border-color: #007bff;
}

/* Buttons */
button {
    background: #007bff;
    border: none;
    color: white;
    padding: 12px 20px;
    font-size: 16px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s ease-in-out;
}
button:hover {
    background: #0056b3;
}

/* Smooth Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""
st.markdown(chat_styles, unsafe_allow_html=True)    


st.title("CVisionary")
st.text("Improve Your Resume for ATS")

analysis_type = st.radio("Select Analysis Type", ["Role-Based Analysis", "Role & Company Analysis"], horizontal=True)

jobr = st.text_input("Enter the Job Role")
company=''
if analysis_type == "Role & Company Analysis":
    company = st.text_input("Enter the Company Name")
uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf","docx","jpg","jpeg","png","txt","tex"], help="Please upload the File")

submit = st.button("Submit")
if submit:
    if uploaded_file is not None:
        start_time = time.time()
        text = input_pdf_text(uploaded_file)
        st.session_state["resume_text"] = text  # Store resume text
        logging.info(f"Extracted Resume Text (first 200 chars): {text[:200]}") 
        
        # Classify resume size
        resume_size = classify_resume_length(text)  
        st.session_state["resume_size"] = resume_size  # Store result
        
        # Debugging: Log classification
        logging.info(f"Classified Resume Size: {resume_size}") 

        st.write(f"**Resume Size:** {resume_size}")

        if analysis_type == "Role & Company Analysis":
            role_exists = check_role_existence(jobr, company)
            if not role_exists:
                st.warning(f"There isn't any {jobr} role at {company}. Please ignore if the job role exists.")
        else:
            role_exists = True

        formatted_prompt = generate_prompt(text, jobr, company, role_exists)
        response = get_gemini_response(formatted_prompt)

        end_time = time.time()  # End measuring time
        total_time = end_time - start_time
        
        response_dict = json.loads(response)
        
        st.session_state["processing_time"] = total_time

        # Store results in session state to persist across interactions

        st.write(f"**Processing Time:** {total_time:.2f} seconds")

        st.session_state["JD_Match"] = response_dict.get("JD_Match", "N/A")
        st.session_state["MissingKeywords"] = response_dict.get("MissingKeywords", [])
        st.session_state["Profile_Summary"] = response_dict.get("Profile_Summary", "")

# Persist results even after chatbot interaction
if "JD_Match" in st.session_state:
    st.subheader("Results")
    st.metric(label="JD Match Percentage", value=st.session_state["JD_Match"])

if "MissingKeywords" in st.session_state:
    st.subheader("Missing Keywords")
    missing_keywords = st.session_state["MissingKeywords"]
    if missing_keywords:
        keyword_buttons = ""
        for keyword in missing_keywords:
            keyword_buttons += f"<button>{keyword}</button> "
        
        st.markdown(
            f"""
            <div style="overflow-x: auto; white-space: nowrap; background-color: #f0f0f0; padding: 10px; border-radius: 8px;">
                {keyword_buttons}
            </div>
            """,
            unsafe_allow_html=True
        )

if "Profile_Summary" in st.session_state:
    st.subheader("Profile Summary")
    st.markdown(st.session_state["Profile_Summary"])


logging.info(f"Resume Size Stored: {st.session_state.get('resume_size', 'Not classified')}")
logging.info(f"Processing Time: {st.session_state.get('processing_time', 0):.2f} seconds")



st.subheader("AI Career Mentor Chatbot")

# Ensure chat history is stored
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

resume_text = st.session_state.get("resume_text")


# Display chat history
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for sender, message in st.session_state.chat_history:
    icon = "👤" if sender == "You" else "🤖"
    align_class = "user-message" if sender == "You" else "bot-message"
    st.markdown(
        f"<div class='chat-bubble {align_class}'>"
        f"<strong>{icon} {sender}</strong><br>{message}"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)
user_input = st.chat_input("Ask your career-related question:")

if user_input:
    if not resume_text:
        st.warning("Please upload and analyze your resume first!")
    else:
        # Show the user's query on the right side
        st.session_state.chat_history.append(("You", user_input))
        
        # Display a loading message
        with st.spinner("Processing your query..."):
            time.sleep(2)  # Simulating a delay
            chat_history = "\n".join([f"{sender}: {message}" for sender, message in st.session_state.chat_history])
            
            if analysis_type == "Role-Based Analysis":
                chatbot_prompt = f"Act as an AI career mentor.Analyze the resume based on the job role.Keep the previous conversation in context and provide a detailed response.Previous Chat:{chat_history} Resume:{resume_text},Job Role:{jobr},User Query:{user_input}.Ensure the response is informative,concise,and actionable."
            else:
                chatbot_prompt = f"Act as an AI career mentor.Analyze the resume based on the job role and company.Align the response strictly with the company's job description and industry standards.Keep the previous conversation in context.Previous Chat:{chat_history} Resume:{resume_text},Job Role:{jobr},Company:{company},User Query:{user_input}.Ensure the response is professional,specific,and provides actionable insights."

            chatbot_response = get_gemini_response(chatbot_prompt)
            
            # Trim response to last complete sentence
            last_period_index = chatbot_response.rfind('.')
            if last_period_index != -1:
                chatbot_response = chatbot_response[:last_period_index + 1]

            st.session_state.chat_history.append(("CVisionary Response:", chatbot_response))
            st.rerun()