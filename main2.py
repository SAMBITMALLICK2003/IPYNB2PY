__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import re
import json
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from dotenv import load_dotenv
from crewai import LLM
from langchain_google_genai import ChatGoogleGenerativeAI
import datetime
import tempfile

load_dotenv()

KEY = os.getenv("GOOGLE_API_KEY")

my_llm = LLM(
    api_key=KEY,
    model="gemini/gemini-2.0-flash-exp",
    max_completion_tokens=8192
)


@CrewBase
class AutoCodeRefactor:
    agent_config = "config/agents.yaml"
    task_config = "config/tasks.yaml"

    @agent
    def Python_Developer(self) -> Agent:
        developer = Agent(
            role="Python Developer and Refactor Expert",
            goal="Refactor Python code from Jupyter Notebooks into well-structured and optimized Python files with functions.",
            backstory="You are a highly skilled Python developer with expertise in code refactoring, function design, and code optimization. You are adept at understanding complex ipynb code structures and transforming them into clean, efficient, and maintainable code. You excel at creating functions to encapsulate logic and improve readability.",
            allow_delegation=False,
            verbose=True,
            llm=my_llm
        )
        return developer

    @task
    def Extract_and_Refactor(self) -> Task:
        return Task(
            description="""Extract python code from the provided Jupyter Notebook : {context} and refactor into multiple optimized functions, ensuring each function has necessary comments. Ensure that no global variable is used and pass all the required parameter in the function. The refactored code should be well-structured, easy to understand, and optimized for performance. Provide the refactored python code ready to be written into a file. Do not write any additional comments outisde the function. The code should be fully contained with the code markdown format. Do not include explanations of what you are doing. Only include the code inside the markdown. \n""",
            expected_output="A Python script as code markdown that refactors the input code into multiple functions. Each function should be well-structured, contain necessary comments and follows best practices. All the required parameter should be passed in function instead of using global variables.",
            agent=self.Python_Developer()
        )


    @crew
    def refactor_crew(self) -> Crew:
        return Crew(
            agents=[self.Python_Developer()],
            tasks=[self.Extract_and_Refactor()],
            process=Process.sequential,
            verbose=True
        )

    def read_ipynb_file(self, filepath):
        """Reads an ipynb file and returns the content as a string."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            code_cells = [cell['source'] for cell in data['cells'] if cell['cell_type'] == 'code']
            code_string = "\n".join("".join(cell) for cell in code_cells)
            return code_string
        except Exception as e:
            st.error(f"Error reading ipynb file: {e}")
            return None


    def write_to_file(self, code, filename):
        """Writes the generated code to a new file."""
        with open(filename, "w", encoding='utf-8') as f:
             f.write(code)
        st.success(f"Refactored code has been written to {filename}")


# Streamlit UI
st.title("Jupyter Notebook Refactor Tool")

uploaded_file = st.file_uploader("Upload your .ipynb file", type=["ipynb"])

if uploaded_file:
    if st.button("Start Refactoring"):
        with st.spinner("Refactoring in progress..."):
            try:
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp_file:
                  tmp_file.write(uploaded_file.getvalue())
                  tmp_file_path = tmp_file.name

                auto_code_instance = AutoCodeRefactor()
                ipynb_content = auto_code_instance.read_ipynb_file(tmp_file_path)
                os.unlink(tmp_file_path)

                if ipynb_content:
                    inputs = {"context": f"{ipynb_content}"}
                    result = auto_code_instance.refactor_crew().kickoff(inputs)
                    pattern = r"```python\s*(.*?)\s*```"  # Regular expression
                    matches = re.findall(pattern, result.raw, re.DOTALL)

                    if matches:
                        refactored_code = matches[0]
                        st.code(refactored_code, language="python")

                        # Download button
                        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        file_name = f"refactored_code_{current_time}.py"
                        st.download_button(label="Download Refactored Code",
                                          data=refactored_code,
                                          file_name=file_name,
                                          mime="text/x-python")

                    else:
                        st.error("No code was generated or result is not a string.")
                else:
                    st.error("Could not read ipynb file.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
