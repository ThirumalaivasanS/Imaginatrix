import base64
import os
import random
import boto3
import json
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import base64
import os
import random
import boto3
import json
from PIL import Image
import matplotlib.pyplot as plt


# Define the Streamlit app layout
st.set_page_config(page_title="Imaginatrix", layout="wide", initial_sidebar_state="expanded")

st.header("Imaginatrix", divider='rainbow')
st.subheader("Your go-to tool for image generation and enhancement")

# Define the Streamlit sidebar menu
selected_menu = st.sidebar.radio("Select a Menu", ["Image Generation", "Enhanced Resolution", "Mock Data Generating"])


def generate_image(prompt: str, seed: int, index: int):
    payload = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 12,
        "seed": seed,
        "steps": 80,
    }

    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    body = json.dumps(payload)
    
    model_id = 'stability.stable-diffusion-xl-v0'
    
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    artifact = response_body.get("artifacts")[0]
    image_encoded = artifact.get("base64").encode("utf-8")
    image_bytes = base64.b64decode(image_encoded)

    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/generated-{index}.png"
    with open(file_name, "wb") as f:
        f.write(image_bytes)

    return file_name

# Define the main function for the Streamlit app
def main():
    if selected_menu == "Image Generation":
        st.markdown("Harnessing the power of the AWS Bedrocks platform's Stability AI model, we can seamlessly conjure up vivid and imaginative images. By furnishing the model with a prompt that serves as the creative compass, this innovative technology employs advanced algorithms to craft dynamic and unique visual compositions. The synergy of human imagination and the precision of the Stability AI model results in a fascinating interplay, unveiling a realm of artistic possibilities and pushing the boundaries of what can be generated through the marriage of creative prompts and cutting-edge AI technology")
        # Update the prompt with your specific requirements
        prompt_data = st.text_input("Enter the prompt:")
        if st.button("Generate Image"):
            seed = random.randint(0, 100000)
            generated_image_path = generate_image(prompt=prompt_data, seed=seed, index=0)

            # Display the generated image using Streamlit
            st.image(Image.open(generated_image_path), caption="Generated Image", use_column_width=True)

    if selected_menu == "Mock Data Generating":
        st.markdown("User-provided image data serves as the foundation for generating innovative synthetic or mock images. Through a sophisticated process, new visual data is crafted, leveraging the input data to produce realistic and diverse imagery. This approach enhances the dataset, facilitating improved training and testing in various applications, such as machine learning and computer vision")
        st.markdown("### Upload your image data")
        uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            st.markdown("### Uploaded Images:")
            for uploaded_file in uploaded_files:
                st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

        # st.markdown("### Actions:")
        if st.button("Generate Mock Data"):
            # Create a directory to store augmented images
            augmented_images_dir = 'C:/TMV/CLARITY/mdata'  # Use forward slashes for paths
            os.makedirs(augmented_images_dir, exist_ok=True)

            # Create an ImageDataGenerator for augmentation
            datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            # Iterate over each uploaded image and generate augmented images
            for uploaded_file in uploaded_files:
                img = image.load_img(uploaded_file, target_size=(150, 150))  # Adjust target size as needed
                x = image.img_to_array(img)
                x = x.reshape((1,) + x.shape)

                # Generate augmented images and save to the augmented directory
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_images_dir, save_prefix='aug',
                                          save_format='jpeg'):
                    i += 1
                    if i > 4:  # Generate 4 augmented images per original image
                        break

            # Visualize some of the augmented images in a grid
            for i, augmented_image_filename in enumerate(os.listdir(augmented_images_dir)[:8]):
                augmented_image_path = os.path.join(augmented_images_dir, augmented_image_filename)
                img = image.load_img(augmented_image_path)
                # Reduce the size of the displayed augmented images
                st.image(img, caption=f"Synthetic Image Data {i}", use_column_width=True, width=150)

    if selected_menu == "Enhanced Resolution":
        st.markdown("Work In Progress")
        # Add your enhanced resolution functionality here

# Run the Streamlit app
if __name__ == "__main__":
    main()
