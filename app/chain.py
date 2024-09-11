import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self, llm):
        # Correctly initializing ChatGroq with a proper API key from environment or directly
        self.llm = llm

    def extracte_job(self, cleaned_text):
        # Defining the prompt template
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the
            following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        
        # Create the chain
        chain_extract = prompt_extract | llm  # Ensure this operation is compatible
        # OR use a proper call method if the operator is not suitable

        # Invoke the chain and parse the response
        try:
            res = chain_extract.invoke({'page_data': cleaned_text})
            json_parser = JsonOutputParser()
            json_res = json_parser.parse(res.content)  # Assuming res has a content attribute
            return json_res if isinstance(json_res, list) else [json_res]
        except OutputParserException as e:
            raise OutputParserException(f"Parsing error: {str(e)}")

    def write_email(self, job, links):
        # Defining the prompt template for the email
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Omar Talibi , a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools.
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability,
            process optimization, cost reduction, and heightened overall efficiency.
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase AtliQ's portfolio: {link_list}
            Remember you are Mohan, BDE at AtliQ.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )

        # Create the chain
        chain_email = prompt_email | self.llm  # Ensure this operation is compatible
        # OR use a proper call method if the operator is not suitable
        
        # Invoke the chain and return the response
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content  # Assuming res has a content attribute

# Example usage
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key="YOUR_KEY"  # Ensure this is correctly set in your .env file
)

chain = Chain(llm)

# Example calls
# cleaned_text = "some cleaned text from the scraped data"
# jobs = chain.extracte_job(cleaned_text)
# email = chain.write_email(job=jobs[0], links=["link1", "link2"])
