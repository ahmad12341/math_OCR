from groq import Groq
# from openai import OpenAI
# import google.generativeai as genai
from enum import Enum
import json
from typing import List, Union

# local_llama_model = Ollama(model="llama3", temperature=0)
# gemini_model = genai.GenerativeModel('gemini-1.5-flash')

class LLM(Enum):
    gpt4o = "gpt-4o"
    gpt4o_mini = "gpt-4o-mini"  
    llama3_70b = "llama3-70b-8192" 
    llama3_8b = "llama3-8b-8192" 
    default = "default"
    gemini = "gemini-1.5-flash"

def get_string(key):
    with open('background_knowledge.json', 'r') as file:
        strings = json.load(file)
        return strings.get(key, None)  # Returns None if key is not found

def getSystemPrompt(equation: str, category :List[Union[str, None]]):
    myStr = str(equation)
    system_prompt = get_string("solver system prompt")
    backgroundKnowledge = ""

    if category == None:
        return system_prompt + "this is the problem that you need to solve" + "\n"+ "####" + myStr + "####"
    
    elif category in "Primes, Highest Common Factor, and Lowest Common Multiple":
        backgroundKnowledge = get_string("HCF,Prime and LCM")
    
    elif category in "Basic Algebra and Algebraic Manipulation":
        backgroundKnowledge = get_string("Basic Algebra and Algebraic Manipulation")        
    
    elif category in "Linear Equations and Simple Inequalities":
        backgroundKnowledge = get_string("Linear Equations and Simple Inequalities") 

    elif category in "Expansion and Factorisation of Quadratic Expressions":
        backgroundKnowledge = get_string("Expansion and Factorisation of Quadratic Expressions")     

    elif category in "Further Expansion and Factorisation of Algebraic Expressions":
        backgroundKnowledge = get_string("Further Expansion and Factorisation of Algebraic Expressions")                 

    elif category in "Algebraic Fractions and Formulae":
        backgroundKnowledge = get_string("Algebraic Fractions and Formulae")   

    return system_prompt + "Here is some background knowledge that may be required for the question " + backgroundKnowledge + "this is the problem that you need to solve" + myStr

def queryLLM(myModel: LLM, api_key: str, equation: str, category: List[Union[str, None]]):
    systemPrompt = getSystemPrompt(equation, category)
    userPrompt = "Please answer the following question: "

    # if myModel == LLM.default:
    #     return local_llama_model.invoke(systemPrompt + userPrompt)
    
    # elif myModel == LLM.gemini:
    #     genai.configure(api_key=api_key)
    #     response = gemini_model.generate_content(systemPrompt + userPrompt)
    #     return response.text

    # elif myModel == LLM.gpt4o:
    #     llmClient = OpenAI(api_key=api_key)  # OpenAI Client Initialization

    # elif myModel == LLM.gpt4o_mini:
    #     llmClient = OpenAI(api_key=api_key)  # OpenAI Client Initialization

    if myModel == LLM.llama3_70b:
        llmClient = Groq(api_key=api_key)  # Groq Client Initialization

    elif myModel == LLM.llama3_8b:
        llmClient = Groq(api_key=api_key)  # Groq Client Initialization
   
    response = ""
    mres = llmClient.chat.completions.create(
            model= myModel.value,
            messages=[
                {
                    "role": "system",
                    "content": systemPrompt
                },
                {
                    "role": "user",
                    "content": userPrompt
                }
            ],
            temperature=0, # with temperature setting it is including the fields that are not very relevant to the user question set to 0 to check its effect on it
            max_tokens=8192,
            top_p=0,
            seed = 0,
            stream=True,
            stop=None,
        )
    for chunk in mres:
            response+=chunk.choices[0].delta.content or ""

    return response


def getEquationType(myModel: LLM, api_key: str, equation: str):
    systemPrompt = get_string("classifier system prompt")
    userPrompt = "Please check the following mathematical expression: " + equation

    # if myModel == LLM.default:
    #     return local_llama_model.invoke(systemPrompt + userPrompt)
    
    # elif myModel == LLM.gemini:
    #     genai.configure(api_key=api_key)
    #     response = gemini_model.generate_content(systemPrompt + userPrompt)
    #     return response.text

    # elif myModel == LLM.gpt4o:
    #     llmClient = OpenAI(api_key=api_key)  # OpenAI Client Initialization

    # elif myModel == LLM.gpt4o_mini:
    #     llmClient = OpenAI(api_key=api_key)  # OpenAI Client Initialization

    if myModel == LLM.llama3_70b:
        llmClient = Groq(api_key=api_key)  # Groq Client Initialization

    elif myModel == LLM.llama3_8b:
        llmClient = Groq(api_key=api_key)  # Groq Client Initialization
   
    response = ""
    mres = llmClient.chat.completions.create(
            model= myModel.value,
            messages=[
                {
                    "role": "system",
                    "content": systemPrompt
                },
                {
                    "role": "user",
                    "content": userPrompt
                }
            ],
            temperature=0, # with temperature setting it is including the fields that are not very relevant to the user question set to 0 to check its effect on it
            max_tokens=8192,
            top_p=0,
            seed = 0,
            stream=True,
            stop=None,
        )
    for chunk in mres:
            response+=chunk.choices[0].delta.content or ""
    print("*******************************")
    print("Equation Identifieer SAYS: ")
    print(response)
    print('\n')
    print("*******************************") 

    if "Highest Common Factor, and Lowest Common Multiple" in response: 
        return "Primes, Highest Common Factor"  
    elif "Basic Algebra" in response:
        return "Basic Algebra and Algebraic Manipulation"
    elif "Linear Equations" in response:
        return "Linear Equations and Simple Inequalities"  
    elif "Factorisation of Quadratic Expressions" in response:
        return "Expansion and Factorisation of Quadratic Expressions"      
    elif "Factorisation of Algebraic Expressions" in response:
        return "Further Expansion and Factorisation of Algebraic Expressions"        
    elif "Algebraic Fractions and Formulae" in response:
        return "Algebraic Fractions and Formulae"      

