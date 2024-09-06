import gradio as gr


def house_matcher_app(match_house):
    """
    Creates a Gradio interface for matching houses based on user preferences.

    Parameters:
    - match_house: Function that takes user inputs and returns matching house details.
    """
    
    with gr.Blocks() as matcher_app:
        # Header
        gr.Markdown("# HomeMatch: Personalized Real Estate Agent\n")
        
        gr.Markdown("## Property Details")
        
        # User Inputs
        with gr.Row():
            bedrooms = gr.Number(label="Number of Bedrooms", value=3)
            bathrooms = gr.Number(label="Number of Bathrooms", value=2)
            area = gr.Number(label="Area (in sqft)", value=1000)
        
        with gr.Row():
            price = gr.Number(label="Price (£)", value=200000)
            city = gr.Textbox(label="City", value="London")
        
        gr.Markdown("## Preferences")
        
        important_features = gr.Textbox(
            label="What are the 3 most important things for you in choosing a property?",
            placeholder="e.g., Location, Price, Safety, Space"
        )
        amenities = gr.Textbox(
            label="Which amenities would you like?",
            placeholder="e.g., Pool, Gym, Parking"
        )
        transportation = gr.Textbox(
            label="Which transportation options are important to you and how close should they be from the house?",
            placeholder="e.g., Proximity to subway, Bus stops"
        )
        neighborhood = gr.Textbox(
            label="How urban do you want your neighborhood to be?",
            placeholder="e.g., Very urban, Suburban"
        )
        
        # Search Button
        submit_button = gr.Button("Search")
        
        # Output Fields for Results
        with gr.Row():
            with gr.Column():
                real_state_1_image = gr.Image(label="House Image")
                real_state_1_desc = gr.Textbox(label="House Description", interactive=False)
            
            with gr.Column():
                real_state_2_image = gr.Image(label="House Image")
                real_state_2_desc = gr.Textbox(label="House Description", interactive=False)
            
            with gr.Column():
                real_state_3_image = gr.Image(label="House Image")
                real_state_3_desc = gr.Textbox(label="House Description", interactive=False)
        
        # Define the action to take when the button is clicked
        submit_button.click(
            fn=match_house,
            inputs=[bedrooms, bathrooms, area, price, city, important_features, amenities, transportation,
                    neighborhood],
            outputs=[real_state_1_image, real_state_1_desc, real_state_2_image, real_state_2_desc, real_state_3_image,
                     real_state_3_desc]
        )
    
    # Launch the Gradio app
    matcher_app.launch()