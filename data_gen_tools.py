import os, shutil
from typing import List
import requests  # for downloading image from OPENAI DALLIE MODEL
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm.notebook import tqdm
from tqdm import tqdm
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def copy_database(src_dir, dst_dir, db_name):
    # delete if database exist
    shutil.rmtree(os.path.join(dst_dir, db_name), ignore_errors=True)
    
    shutil.copytree(src=os.path.join(src_dir, db_name), dst=os.path.join(dst_dir, db_name))


# DATA GENERATION
def generate_synthetic_real_state_data(model_name: str, num_houses: int, columns: List[str]) -> pd.DataFrame:
    """
    Generates synthetic real estate data for a specified number of houses.

    This function uses a language model to generate a JSON-formatted description of residential real estate
    properties in the United Kingdom. The descriptions include various details about the properties and
    their surroundings, such as distance from the city center, availability of transport, school proximity,
    noise levels, kitchen size, living room size, nearby parks, gardens, and views.

    Parameters:
    -----------
    num_houses : int
        The number of houses for which synthetic data needs to be generated.
    columns : List[str]
        A list of column names that should be included in the generated data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the generated synthetic real estate data.
    """
    
    # Create an empty DataFrame with the specified columns
    real_state_data = pd.DataFrame(columns=columns)
    
    # Define a template for generating the prompt to create JSON-formatted descriptions of properties
    data_gen_prompt_template = """
    Generate JSON formatted description for {num_houses} residential real estate in London, Manchester and Edinbrugh cities in the United Kingdom.
    In the JSON file include data for these features: {columns}. While describing the house and neighborhood,
    be concise, creative and realistic. Include information such as how big the house is? distance from the city center, availability of transport,
    schools for children, the size of living room and kitchen, availability of transport types nearby, parking space, garden, and
    outside views and others characteristics that house buyers are usually interseted in.
    """
    
    # Create a LangChain prompt template from the defined template
    data_gen_prompt = PromptTemplate.from_template(data_gen_prompt_template)
    
    # Initialize the OpenAI chat completion model with the specified parameters
    llm = OpenAI(model_name=model_name,
                 temperature=0.0,
                 max_tokens=4050)
    
    # Generate the description of the houses using the model and the formatted prompt
    output = llm(data_gen_prompt.format(num_houses=num_houses, columns=columns))
    
    # Post process LLM output and parse it as CSV
    output = llm_output_parser(output)
    
    return output


def llm_output_parser(input_data: str):
    # Replaces all occurrences of triple backticks and "json" with empty strings in synthetic_real_state_data,
    # which is expected to contain JSON-like data, likely from a language model output.
    cleaned_data = input_data.replace("```", "").replace("json", "")
    
    # Evaluates the cleaned string as Python code.
    # This results in list of dictionary (similar to JSON).
    # This output is passed to pandas' DataFrame constructor, turning it into a tabular format.
    return pd.DataFrame(eval(cleaned_data)['residential_real_estate'])


def generate_real_state_images(model_name: str, synthetic_data: pd.DataFrame, output_dir: str) -> None:
    """
    Generates and saves images of real estate properties based on synthetic data descriptions.

    This function takes a DataFrame containing synthetic real estate data and generates images for each
    property using the DALL-E model. It constructs a descriptive text prompt for each property, sends
    the prompt to the DALL-E API to generate an image, and then saves the resulting image to the specified
    output directory.

    Parameters:
    -----------
    synthetic_data : pd.DataFrame
        A DataFrame containing the synthetic real estate data. Each row should represent a property
        and include columns like 'bedrooms', 'bathrooms', 'description', and 'neighbourhood'.
    output_dir : str
        The directory where the generated images will be saved. The directory will be created if it
        does not already exist.

    Returns:
    --------
    None
        The function saves the images as PNG files in the specified output directory and does not return any value.

    """
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the number of properties (houses) in the synthetic data
    n = len(synthetic_data)
    print(f"Generating {n} images")
    
    # Loop through each row in the DataFrame, where each row represents a property
    for index, row in tqdm(synthetic_data.iterrows(), desc="Generate images", total=n):
        # Construct the text prompt for DALL-E using the property's features
        prompt = """Generate a realistic real estate house with the following description.

        The house has {} bedrooms, and {} bathrooms. {} {}"""
        
        # Format the prompt string with data from the current row
        prompt = prompt.format(row['bedrooms'], row['bathrooms'], row['description'], row['neighbourhood'])
        # print(f"Prompt: \n{prompt}")
        
        # Generate the image using the DALL-E API with the specified prompt and model
        response = openai.Image.create(
            prompt=prompt,
            model=model_name,  # Specify the version of the DALL-E model to use
            n=1,  # Number of images to generate per prompt
            size="512x512"  # Dimensions of the generated image
        )
        
        # Extract the URL of the generated image from the API response
        image_url = response['data'][0]['url']
        
        # Download the image from the URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        # Construct the output file name using the row index, padded to three digits
        output_file = os.path.join(output_dir, f"{index}".zfill(3) + ".png")
        
        # Save the image as a PNG file in the specified output directory
        img.save(output_file)
        # print(f"Image saved at: {output_file}")
