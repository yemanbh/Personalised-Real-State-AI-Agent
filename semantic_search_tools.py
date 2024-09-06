import os, shutil
import openai

def copy_database(src_dir, dst_dir, db_name):
    """
    Copies a database directory from a source to a destination directory, 
    replacing the existing database at the destination if it exists.

    Args:
        src_dir (str): The source directory containing the database.
        dst_dir (str): The destination directory where the database should be copied.
        db_name (str): The name of the database directory to copy.

    Returns:
        None
    """
    # Remove the existing database directory at the destination if it exists
    shutil.rmtree(os.path.join(dst_dir, db_name), ignore_errors=True)
    
    # Copy the database directory from the source to the destination
    shutil.copytree(src=os.path.join(src_dir, db_name), dst=os.path.join(dst_dir, db_name))

# Function to format the input and send to LLM
def capture_user_preferences(preferences):

    query = f"""
    I am looking for a  house which has {preferences["bedrooms"]} bedroom, {preferences["bathrooms"]}-bathroom property located in the city of {preferences["city"]}.
    The house should be about {preferences["area"]} square feet area and price about £{preferences["price"]}.
    The most important things I am looking for include {preferences["important_features"]} and {preferences["amenities"]} amenities.
    with regard to transportation, I prefer {preferences["transportation"]}. I am looking for a house in a neighborhood like this:{ preferences["neighborhood"]}
    """

    return query

# features = ["bedrooms", "bathrooms", "area", "zipcode", "price", "city", "street",
#             "description", "neighbourhood"]

def generate_personalized_descriptions(model_name, matched_listing, preferences):
  # Function to generate personalized descriptions using LLM
  # personalized_descriptions = []
  # for listing in matched_listings:
  # - Title: {matched_listing['title']}

  real_state_title = "REAL STATE TITLE"
  prompt = f"""
  Here is a real estate listing matching with user preference bellow:
  - Bedrooms: {matched_listing['bedrooms']}
  - Bathrooms: {matched_listing['bathrooms']}
  - Area: {matched_listing['area']} sqft
  - Price: {matched_listing['price']} pounds
  - City: {matched_listing['city']}
  - Postcode: {matched_listing['zipcode']}
  - Street: {matched_listing['street']}
  - Description: {matched_listing['description']}

  The buyer has the following preferences:
  - Bedrooms: {preferences['bedrooms']}
  - Bathrooms: {preferences['bathrooms']}
  - Area: {preferences['area']} sqft
  - Price: {preferences['price']} pounds
  - City: {preferences['city']}
  - Important Features: {preferences['important_features']}
  - Preferred Amenities: {preferences['amenities']}
  - Transportation Preferences: {preferences['transportation']}
  - Neighborhood Preference: {preferences['neighborhood']}

  Based on these preferences, rewrite the listing description to make it more appealing to the buyer.
  Highlight features that match their preferences without changing any factual information.
  """

  response = openai.ChatCompletion.create(
      model=model_name,
      messages=[
          {"role": "system", "content": "You are an expert real estate matching assistant."},
          {"role": "user", "content": prompt}
      ],
      max_tokens=200,
      temperature=0.0,
  )

  personalized_description = response['choices'][0]['message']['content'].strip()
  personalized_description = f"Title: {matched_listing['title']} \n\n " + personalized_description

  return personalized_description






