# Creative AI: Generating Art with Neural Style Transfer 🎨

This project explores the fascinating intersection of art and AI by implementing Neural Style Transfer. It uses a pre-trained deep learning model from TensorFlow Hub to combine the content of one image (e.g., a photo) with the artistic style of another image (e.g., a painting).

This notebook demonstrates how to load pre-trained models, preprocess images for style transfer, apply the style transfer algorithm, and visualize the creative results.

**Model:** Arbitrary Image Stylization model from TensorFlow Hub (`magenta/arbitrary-image-stylization-v1-256`)
**Input:** A content image and a style image.
**Focus:** Demonstrating Neural Style Transfer, using pre-trained models from TensorFlow Hub, image loading and preprocessing, model inference for image generation, and result visualization.
**Repository:** [https://github.com/Jayasurya227/Creative-AI-Generating-Art-with-Neural-Style-Transfer](https://github.com/Jayasurya227/Creative-AI-Generating-Art-with-Neural-Style-Transfer)

***

## Key Techniques & Concepts Demonstrated

Based on the implementation within the notebook (`Creative_AI_Generating_Art_with_Neural_Style_Transfer (1).ipynb`), the following key concepts and techniques are applied:

* **Neural Style Transfer:** An optimization technique used to take two images—a content image and a style reference image—and blend them so the output image retains the core content of the content image but adopts the artistic style (texture, color palette, brush strokes) of the style image.
* **Pre-trained Models (TensorFlow Hub):** Leveraging a sophisticated, pre-trained arbitrary image stylization model available directly from TensorFlow Hub, eliminating the need for extensive model training.
* **TensorFlow Hub:** Using TF Hub (`hub.load`) as a repository to easily load and utilize pre-trained machine learning models.
* **Image Loading & Preprocessing:**
    * Using `matplotlib.pyplot` and `PIL` to load images.
    * Defining a function (`load_img`) to read images, convert them to `float32` tensors, scale pixel values to [0, 1], and add a batch dimension.
* **Model Inference:** Running the loaded style transfer model by providing the preprocessed content and style images as input. The model outputs the stylized image tensor.
* **Image Post-processing:** Converting the output tensor back into a displayable image format (e.g., NumPy array with `uint8` pixel values).
* **Visualization:** Using Matplotlib to display the original content image, the style image, and the final generated stylized image side-by-side for comparison.

***

## Analysis Workflow

The notebook follows a straightforward process for applying Neural Style Transfer:

1.  **Setup & Dependencies:** Importing necessary libraries (TensorFlow, TensorFlow Hub, NumPy, Matplotlib, PIL).
2.  **Load Pre-trained Model:** Loading the `magenta/arbitrary-image-stylization-v1-256` model from TensorFlow Hub.
3.  **Define Image Loading Function:** Creating the `load_img` function to handle image loading and preprocessing (scaling, adding batch dimension).
4.  **Load Content & Style Images:** Specifying file paths for the content image (e.g., `content.jpg`) and style image (e.g., `style.jpg`) and loading them using the defined function.
5.  **Apply Style Transfer:** Passing the preprocessed content and style image tensors to the loaded TensorFlow Hub model.
6.  **Retrieve & Post-process Output:** Getting the stylized image tensor from the model's output and converting it back to a standard image format (NumPy `uint8` array).
7.  **Visualize Results:** Displaying the content image, style image, and the generated stylized image using Matplotlib.

***

## Technologies Used

* **Python**
* **TensorFlow:** Core deep learning framework.
* **TensorFlow Hub (`tensorflow_hub`):** For loading the pre-trained style transfer model.
* **NumPy:** For numerical operations and tensor/array manipulation.
* **Matplotlib:** For image loading and visualization.
* **Pillow (`PIL`):** For image file handling.
* **Jupyter Notebook / Google Colab:** For the interactive development environment.

***

## Prerequisites

* **Content Image:** An image file to provide the content (e.g., `content.jpg`).
* **Style Image:** An image file to provide the artistic style (e.g., `style.jpg`).

*(Ensure these image files are present in the working directory or update the file paths in the notebook accordingly.)*

***

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jayasurya227/Creative-AI-Generating-Art-with-Neural-Style-Transfer.git](https://github.com/Jayasurya227/Creative-AI-Generating-Art-with-Neural-Style-Transfer.git)
    cd Creative-AI-Generating-Art-with-Neural-Style-Transfer
    ```
2.  **Install dependencies:**
    (It is recommended to use a virtual environment)
    ```bash
    pip install tensorflow tensorflow_hub numpy matplotlib Pillow jupyter
    ```
3.  **Prepare Images:** Place your desired `content.jpg` and `style.jpg` (or other named images, updating the notebook paths if necessary) in the repository directory.
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook "Creative_AI_Generating_Art_with_Neural_Style_Transfer (1).ipynb"
    ```
5.  **Run Cells:** Execute the cells sequentially. The TensorFlow Hub model will be downloaded automatically the first time it's loaded. The final cell will display the content, style, and generated stylized images.

***

## Author & Portfolio Use

* **Author:** Jayasurya227
* **Portfolio:** This project ([https://github.com/Jayasurya227/Creative-AI-Generating-Art-with-Neural-Style-Transfer](https://github.com/Jayasurya227/Creative-AI-Generating-Art-with-Neural-Style-Transfer)) demonstrates the creative application of AI and deep learning through Neural Style Transfer. It showcases the ability to work with pre-trained models from repositories like TensorFlow Hub and apply them to image generation tasks. Suitable for GitHub, resumes/CVs, LinkedIn, and interviews, especially for roles involving creative AI, computer vision, or deep learning applications.
* **Notes:** Recruiters can see the practical use of a popular generative AI technique, leveraging existing powerful models for artistic image creation. It highlights skills in using TensorFlow, TF Hub, and image processing.
