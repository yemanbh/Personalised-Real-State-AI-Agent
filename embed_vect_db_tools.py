import os, io, re, shutil
from PIL import Image
import pandas as pd
from tqdm.notebook import tqdm
from lancedb.pydantic import Vector, LanceModel
from tqdm import tqdm
import torch
from torchvision.transforms.functional import pil_to_tensor
from transformers import CLIPModel, CLIPTokenizerFast, CLIPProcessor


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


def get_clip_model(clip_model, device):
    """
    Loads and prepares a CLIP model and its associated components for embedding and
    understanding text and images.

    Args:
        clip_model (str): The identifier for the pre-trained CLIP model to load.
        device (torch.device): The device (CPU or GPU) on which the model should be loaded.

    Returns:
        tuple: A tuple containing:
            - CLIPModel: The loaded CLIP model.
            - CLIPProcessor: The processor for the CLIP model.
            - CLIPTokenizerFast: The tokenizer for the CLIP model.
    """
    # Load the CLIP tokenizer, model, and processor from pre-trained weights
    tokenizer = CLIPTokenizerFast.from_pretrained(clip_model)
    model = CLIPModel.from_pretrained(clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model)
    
    return model, processor, tokenizer


def embed_image(clip_model,
                clip_processor,
                image,
                device='cpu',
                normalise=False):
    """
    Generates an embedding for an image using the CLIP model.

    Args:
        image (torch.Tensor): The input image as a tensor, typically preprocessed.
        normalise (bool): Whether to normalize the embedding using L2 normalization. Default is False.

    Returns:
        torch.Tensor: The embedding of the image.
    """
    # Unsqueeze image tensor to add a batch dimension and move it to the correct device
    images = torch.unsqueeze(image, dim=0).to(device)
    # print(images.shape)
    
    # Use CLIP processor to get the input tensors for the image
    inputs = clip_processor(text=None, images=images, return_tensors='pt')
    
    # Generate the image embedding
    with torch.no_grad():
        img_feature = clip_model.get_image_features(**inputs)
    
    # Move the embedding to CPU and detach from computation graph
    img_feature = img_feature.detach().cpu()[0]
    
    # Optionally normalize the embedding using L2 normalization
    if normalise:
        img_feature = img_feature / torch.norm(img_feature, p=2)
    
    return img_feature


def embed_text(clip_model,
               clip_tokenizer,
               text,
               normalise=False):
    """
    Generates an embedding for text using the CLIP model. This function handles text inputs longer
    than 77 tokens by splitting them into chunks.

    Args:
        text (str): The input text to be embedded.
        normalise (bool): Whether to normalize the embedding using L2 normalization. Default is False.

    Returns:
        torch.Tensor: The combined embedding of the text chunks.
    """
    # Remove newline characters from the text
    text = text.replace("\n", "")
    
    # Tokenize the text into tokens using the CLIP tokenizer
    tokens = clip_tokenizer.tokenize(text)
    max_len = 77  # CLIP text model limit
    
    # Split the tokens into chunks of 77 tokens each
    token_chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
    
    # print(f"tokens:{token_chunks}")
    
    # Reconstruct text chunks from tokens (this step depends on the tokenizer implementation)
    text_chunks = ["".join(chunk).replace("</w>", " ") for chunk in token_chunks]
    # print(f"text_chunks={text_chunks}")

    # TODO: REMOVE ANY SMALL TOKEN AT THE END
    
    # Embed each text chunk using the CLIP model
    inputs = clip_tokenizer(text=text_chunks, padding=True, truncation=True, return_tensors='pt')
    text_features = clip_model.get_text_features(**inputs)
    
    # Combine the embeddings of the chunks by averaging
    text_features = torch.mean(text_features, dim=0)
    
    # Optionally normalize the final embedding using L2 normalization
    if normalise:
        text_features = text_features / torch.norm(text_features, p=2)
    
    return text_features.detach().cpu()


def get_text_image_embedding(clip_model,
                             clip_processor,
                             clip_tokenizer,
                             text,
                             image,
                             device='cpu',
                             normalise=True, alpha=0.8):
    """
    Generates and combines embeddings for both text and image, and returns the concatenated result.

    Args:
        text (str): The input text to be embedded.
        image (torch.Tensor): The input image as a tensor, typically preprocessed.
        normalise (bool): Whether to normalize both text and image embeddings using L2 normalization. Default is True.

    Returns:
        torch.Tensor: The concatenated embedding of the text and image.
    """
    # Generate text embedding
    text_embedding = embed_text(clip_model, clip_tokenizer, text, normalise=normalise)
    
    # Generate image embedding
    img_embedding = embed_image(clip_model, clip_processor, image, device=device, normalise=normalise)
    
    # final embedding containing text and image
    final_embedding = alpha * text_embedding + (1 - alpha) * img_embedding
    
    # Concatenate the text and image embeddings into a single tensor
    return final_embedding


class RealState(LanceModel):
    """
    A class representing a real estate listing with multimodal embeddings.

    Attributes:
        vector (Vector(512)): Multimodal embedding vector combining image and text data.
        bedrooms (int): Number of bedrooms in the property.
        bathrooms (int): Number of bathrooms in the property.
        area (float): Area of the property in square feet.
        zipcode (str): Postal code of the property's location.
        price (float): Listing price of the property.
        image (bytes): Image of the property in bytes format.
        title (str): Title of the property listing.
        city (str): City where the property is located.
        street (str): Street address of the property.
        description (str): Detailed description of the property.
    """
    
    vector: Vector(512)  # Multimodal embedding
    # img_vector: Vector(img_emb_dim)  # Image-only embedding (commented out)
    # text_vector: Vector(text_emb_dim)  # Text-only embedding (commented out)
    bedrooms: int
    bathrooms: int
    area: float
    zipcode: str
    price: float
    image: bytes
    title: str
    city: str
    street: str
    description: str
    
    def to_pil(self):
        """
        Converts the image byte data to a PIL Image object.

        Returns:
            PIL.Image.Image: The image as a PIL Image object.
        """
        return Image.open(io.BytesIO(self.image))
    
    @classmethod
    def pil_to_byte(cls, img):
        """
        Converts a PIL Image object to byte format.

        Args:
            img (PIL.Image.Image): The PIL Image object to convert.

        Returns:
            bytes: The image in byte format.
        """
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()


def create_housematch_vec_db(df,
                             clip_model,
                             clip_processor,
                             clip_tokenizer,
                             images_dir,
                             table,
                             device):
    """
    Creates a vector database for real estate listings by embedding images and text descriptions.

    Args:
        df (pd.DataFrame): DataFrame containing real estate listings data.
        clip_model: The CLIP model for generating embeddings.
        clip_processor: The processor for the CLIP model.
        clip_tokenizer: The tokenizer for the CLIP model.
        images_dir (str): Directory containing property images.
        table: The table where data entries will be added.
        device: The device (CPU/GPU) to run the embeddings on.

    Returns:
        None
    """
    features = ["title", "bedrooms", "bathrooms", "area", "zipcode", "price", "city", "street",
                "description", "neighbourhood"]
    
    data = []
    
    for i, row in tqdm(df.iterrows(), desc="creating vector database", total=len(df)):
        # Load image
        image_path = os.path.join(images_dir, f"{i}".zfill(3) + ".png")
        image = Image.open(image_path)
        tensor_image = pil_to_tensor(image)
        
        # Construct house description
        house_description = f"{row['description']} {row['neighbourhood']}"
        print(f"house description: {house_description}")
        
        # Prepare entry for database
        entry = {feature: row[feature] for feature in features}
        entry['vector'] = get_text_image_embedding(clip_model,
                                                   clip_processor,
                                                   clip_tokenizer,
                                                   house_description,
                                                   tensor_image,
                                                   device,
                                                   normalise=True).tolist()
        # Optional: Uncomment for separate embeddings
        # entry['img_vector'] = embed_image(clip_model, clip_processor, tensor_image, device, normalise=True).tolist()
        # entry['text_vector'] = embed_text(house_description, normalise=True).tolist()
        entry['image'] = RealState.pil_to_byte(image)
        
        data.append(entry)
    
    table.add(data)


def clean_synthetic_data(df):
    """
    Cleans the synthetic data by removing non-numeric characters from the area and price columns
    and converting them to floats.

    Args:
        df (pd.DataFrame): DataFrame containing real estate listings data.

    Returns:
        pd.DataFrame: DataFrame with cleaned area and price columns.
    """
    # Remove non-numeric characters from 'price' and 'area', then convert to float
    df['price'] = df['price'].map(lambda val: float("".join(re.findall(r"\d+", val))))
    df['area'] = df['area'].map(lambda val: float("".join(re.findall(r"\d+", val))))
    
    return df
