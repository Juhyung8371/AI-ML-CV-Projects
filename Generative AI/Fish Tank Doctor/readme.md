# Fish Tank Doctor - Fish Tank Health Diagnosis AI (work in progress)

# <ins>Introduction</ins>

Maintaining a balanced environment within a fish tank is an intricate task, often presenting challenges for both novice and experienced fish keepers. Fish tank care encompasses a spectrum of challenges, as it requires a keen understanding of factors such as water parameters, species care, and overall tank maintenance. Inexperienced hobbyists may struggle to identify subtle signs of issues in their fish tanks.

I developed the Fish Tank Doctor, an AI-powered fish tank diagnosis tool, to empower inexperienced fish keepers to overcome these challenges with precision and ease. Leveraging the capabilities of machine learning and computer vision, this tool stands as a reliable companion in the quest for fish-keeping mastery. It can offer a comprehensive analysis of visual cues and provide tailored insights into the conditions of the tank.

# <ins>Literature Review</ins>

The ultimate goal of this project is to make a system that takes in an image of an aquarium as the input and produces comprehensive text feedback to help improve the user's fish husbandry skills. A suitable solution would be an image captioning model. A visual question-answering model would be good as well if I want to extend the scope to allow the user to interact with the model and get more personalized information. Those models are subcategories of the vision-language model (VLM). In general, VLMs are asked to perform certain tasks with given image inputs, and they output natural language text. Examples of tasks include image captioning, visual question answering, object detection, visual generation, visual summarization, etc. 

A VLM typically consists of 3 key elements: 
- an image encoder
- a text encoder
- a strategy to fuse information from the two encoders

1. **[VQA: Visual Question Answering](https://arxiv.org/abs/1505.00468) 2015**
   - VQA aims to answer open-ended and free-form questions, unlike older works such as [this paper](https://arxiv.org/abs/1410.0210), which limited the object category to 894 pre-defined objects and [this paper](https://www.researchgate.net/publication/273387445_Visual_Turing_test_for_computer_vision_systems) which has a set of questions four possible categories.
   - It consists of VGGNet (a type of CNN network) as the image encoder and LSTM as the question text encoder. The visual and textual features are then combined using Element-wise Multiplication.
   - This study introduced the visual-question answering task, released a benchmark dataset (the VQA dataset) for future VQA models, and demonstrated the effectiveness of multimodal fusion techniques for integrating visual and textual information in a unified framework.
   
   <br>
   <img src="readme_images/vqa_model.png" height="200">

2. **[Show and Tell](https://arxiv.org/abs/1411.4555)** 2015:
   - The image features are extracted using a Convolutional Neural Network (CNN) as an encoder, and a Long Short-Term Memory (LSTM) network is employed as a decoder to generate the sequential captions. 
   - Similarly, in [Show, Attend and Tell](https://arxiv.org/abs/1502.03044), an attention mechanism is used to dynamically focus on different parts of the image, improving the model's ability to capture finer details. 
   - I implemented a similar architecture [here](https://github.com/Juhyung8371/AI-Projects/tree/main/Generative%20AI/5%20Image%20Captioning%20-%20Visual%20Attention) using InceptionResNetV2 and visual attention.
   
   <br>
   <img src="readme_images/showattentionandtell_model.png" height="200">

3. **[Image Transformer](https://arxiv.org/abs/1802.05751) 2018:**
   - The authors extend the transformer architecture, initially designed for sequence-to-sequence tasks in NLP, to process images. The key idea is to treat the image as a 1D sequence of fixed-size patches, similar to how words in a sentence are treated. 
   - It showcases the effectiveness of transformer-based architectures on image-based tasks (image generation and image super-resolution in this case) beyond traditional tasks like NLP.

4. **[ViLBERT](https://arxiv.org/abs/1908.02265) 2019**:
   - ViLBERT (Vision-and-Language Bidirectional Encoder Representations from Transformers) aims to learn task-agnostic visio-linguistic representations that can be fine-tuned for various downstream tasks like image captioning, visual commonsense reasoning, and VQA. 
   - The model uses a combination of Masked-Language Modeling (MLM) and Image-Text Matching (ITM) objectives to enable said downstream tasks. 
   - It extends BERT architecture to a multi-modal two-stream model, processing both visual and textual inputs in separate streams that interact through co-attentional transformer layers.
   - The model uses the [Faster R-CNN (2018)](https://arxiv.org/abs/1506.01497) to extract regional visual features.
   - This work is a significant milestone in bridging the gap between image and text.
   
   <br>
   <img src="readme_images/vilbert_model.png" height="150">

5. **[ViT](https://arxiv.org/abs/2010.11929) 2020**:
   - ViT challenges the dominance of CNNs in image recognition tasks, and made significant impact in the computer vision field. It did not replace CNNs but rather provided an additional tool in the toolbox for researchers.
   - The principles ViT introduced have inspired the development of hybrid models that combine vision and language processing, contributing to advancements in multimodal tasks.
   - Vision Transformer (ViT) treats the entire image as a sequence of fixed-size non-overlapping patches, aiming to capture both local and global spatial dependencies in the image.
   - According to this Medium article [CNNs vs ViT](https://medium.com/@faheemrustamy/vision-transformers-vs-convolutional-neural-networks-5fe8f9e18efc), ViT could outperform CNNs given large dataset, but it is much more resource-intensive process than CNNs. 
   
   <br>
   <img src="readme_images/vit_model.png" height="250">

6. **[SOHO](https://arxiv.org/abs/2104.03135) 2021**:
   - Seeing Out of tHe bOx (SOHO) proposes a solution to common limitations of region-based image features:
     - It can detect objects but may not capture the context, which can lead to misunderstandings. 
     - Object detection is limited to the number of pre-defined features in the model.
     - The image detection model may suffer from data problems such as low quality, noise, over-sampling, and reliance on a large amount of annotated data. 
   - SOHO leverages a trainable CNN visual encoder (ResNet), which takes the whole image as input and produces visual features at image-level instead of region-level. 
   - Then, it assigns visual words from the Visual Dictionary to different regions or segments within the image. The model learns to predict object boundaries based on the distribution of visual words in the image, enabling more accurate and fine-grained object detection results.
   - SOHO represents objects in images using object boundaries, facilitating more accurate and context-aware visual-text operations.
   
   <br>
   <img src="readme_images/soho_model.png" height="250">

7. **[CLIP](https://arxiv.org/abs/2103.00020) 2021**:
   - Contrastive Language–Image Pre-training (CLIP) uses a contrastive learning strategy to learn how to recognize the similarities and differences between data to pre-train the model. 
   - It jointly trains the image encoder (ViT) and text encoder (BERT) by putting the images and texts in a shared embedding space and putting like pairs closer to each other.  
   - It can effectively caption the semantic gap/relationship - demonstrating remarkable performance on tasks such as zero-shot image classification and natural language-based image retrieval.
   - Read more about CLIP in this [article](https://viso.ai/deep-learning/clip-machine-learning/).
   
   <br>
   <img src="readme_images/clip_model.png" height="250">

8. **[LiT](https://arxiv.org/abs/2111.07991) 2021**:
   - This paper proposes a Locked-image Tuning (LiT) strategy, which utilizes contrastive learning methods like CLIP, but it only fine-tunes the text encoder while the image encoder is locked (not learning).
   - By locking the image encoder, LiT preserves the pre-trained visual data, reduces overfitting, prevents catastrophic forgetting of image data, and saves resources required for training the image encoder, which is not small in multimodal training. 
   - LiT is particularly strong for unlabeled data, making it a valuable tool for zero-shot transfer learning.

9. **[Frozen](https://arxiv.org/abs/2106.13884) 2021**:
   - Frozen is the pioneering work in the in-context few-shot learning topic. 
   - It uses the Frozen PrefixLM technique where the language model (transformer-based) is frozen (not learning), and only the vision encoder (NF-ResNet-50 model) is updated. The visual data is used as the prefix of the text data in the training process.
   - By freezing the language models during training, the model benefits from the rich semantic information encoded in the pre-trained language model weights.
   
   <br>
   <img src="readme_images/frozen_model.png" height="200">

10. **[SimVLM](https://arxiv.org/abs/2108.10904) 2021**:
   - The Simple Visual Language Model (SimVLM) proposes a minimalist pre-training framework that exploits large-scale weak supervision to reduce training complexity. It is trained [end-to-end](https://www.baeldung.com/cs/end-to-end-deep-learning) with a single prefix language modeling objective.
   - In this model, images and some beginning texts are considered as prefixes for the rest of the textual descriptions, enabling bidirectional attention for the prefix and autoregressive learning on the rest of the text. 
   - The model's backbone is the transformer architecture, where the image feature extraction is a combination of ViT and ResNet, inspired by [CoAtNet](https://arxiv.org/abs/2106.04803).
   - This model is very powerful in image captioning and VQA tasks. However, it is important to note that a model solely relying on PrefixLM may struggle to adapt to other downstream tasks like object detection. 

   <br>
   <img src="readme_images/simvlm_model.png" height="250">

11. **[Flamingo](https://arxiv.org/abs/2204.14198) 2022**:
   - To address the challenges associated with information lost during the fine-tuning step, Flamingo freezes both pre-trained visual and textual encoders and inserts gated cross-attention modules to bridge these two frozen models.
   - The Perceiver Resampler module is used to reduce the complexity of vision-text cross-attention by taking the images or video features from the vision encoder and producing a fixed number of visual tokens.
   - Flamingo sets a new state-of-the-art in few-shot learning on a wide range of open-ended vision and language tasks by leveraging a 70B-sized frozen language model (contemporary model sizes are from 1B to 5B).
   
   <br>
   <img src="readme_images/flamingo_model.png" height="250">

12. **[CoCa](https://arxiv.org/abs/2205.01917) 2022**:
    - Contrastive Captioner (CoCa) efficiently combines contrastive and captioning objectives with Selective Cross-Attention Mechanism where we omit the cross-attention in first half of the text decoder layers to encode unimodal text representations, and cascades the remaining decoder layers for multimodal image-text representations. 
    - The two training objectives are computed efficiently with minimal overhead by sharing the same computational graph.
    - This work bridges the gap among various pretraining approaches by proposing said strategies. 
    
   <br>
   <img src="readme_images/coca_model.png" height="250">

13. **[GIT](https://arxiv.org/abs/2205.14100) 2022**:
    - The Generative Image-to-text Transformer's (GIT) uniqueness lies in its simplicity, generative nature, and scalability, which led to its state-of-the-art performance across various vision-language tasks. 
    - GIT consists of only one image encoder and a text decoder under a single langauge modeling task. This architecture is much simpler compared to existing works, which typically contain complex structures (uni/multi-modal encoder/decoder) and depends on external modules such as object detectors/taggers and optical character recognition. 
    - GIT adopts a generative approach to tackle vision-language tasks as opposed to more common classification or detection frameworks. This approach provides a consistent network architecture between pre-training and fine-tuning stages, simplifying the model. It generates text directly based on the images or video inputs, facilitating a more unified method for handling various tasks without needing task-specific adaptations. This innovative application showcases the robustness of generating descriptive or explanatory text for images over identifying labels. 
    - GIT adopts a large contrastively pre-trained image encoder with a relatively small, randomly initialized text decoder. Also, it uses the generation task to pre-train both the image encoder and text decoder. 
    - GIT can scale up both the model size and the pre-training data to enhance the model's performance significantly. The image encoder is based on CLIP/ViT-B for GIT base (size 129M), CLIP/ViT-L for GIT large (size 347M), [Florence](https://arxiv.org/abs/2111.11432) for GIT (size 681M), and [DaViT](https://arxiv.org/abs/2204.03645) for GIT2 (size 5.1B).

    <br>
    <img src="readme_images/git_model.png" height="250">

14. **[BLIP](https://arxiv.org/abs/2201.12086) 2022**:
    - BLIP proposes Multimodal mixture of Encoder-Decoder (MED), a multi-task model which can operate in one of the three functionalities: Unimodal encoder, which is ViT for image and BERT for text; Image-grounded text encoder; Image-grounded text decoder.
    - It uses CapFilt method, a new dataset bootstrapping method for learning from noisy image-text pairs. 
    - With those techniques, BLIP became a unified vision-language pretraining framework that can effectively learn from noisy image-text pairs. 
    - Their following work, [BLIP2](https://arxiv.org/abs/2301.12597) from 2023, shifts focus to efficiency and robustness by leveraging frozen pre-trained image models and language models, which are bridged by the Querying Transformer. BLIP2 achieved state-of-the-art performance on various vision-language tasks. 
    - Check this [article](https://medium.com/@enrico.randellini/image-and-text-features-extraction-with-blip-and-blip-2-how-to-build-a-multimodal-search-engine-a4ceabf51fbe) that compares two strategies.
    
    <br>
    <img src="readme_images/blip_model.png" height="250"> <img src="readme_images/blip2_model.png" height="250">

15. **[PaLI](https://arxiv.org/abs/2209.06794) 2022**:
    - Language model scales well (500B+ parameters) while vision models did not experience as dramatic effect so far beyond 1B parameters. PaLI challenged this idea by creating largest vanilla ViT architecture to date (ViT-e) to-date with 4B parameters. 
    - The key characteristics of PaLI is the reuse of large unimodal backbones for language [(mT5-XXL)](https://arxiv.org/abs/2010.11934)and vision (ViT-e) modeling. 
    - The study found three things:
      - PaLI performs very well in multiple language tasks. It can do 109 languages, which required them to define a new benchmark task and a new multilingual image-language dataset WebLI with 10B images, 12B alt-texts, and 29B image-OCR pairs. 
      - The langauge model's performance scales with its size. 
      - Balancing the vision model with the langauge model can efficiently improve the performance, proving that vision models do benefit from scaling. 
    - With a simple and scalable solution, PaLI outperforms previous state-of-the-art models in various vision-language tasks. 
    - The most recent version, [PaLI-3](https://arxiv.org/abs/2310.09199) from 2023, is much smaller but stronger. It uses contrastively pretrained ViT-G/14 model (2B parameters) using the [SigLIP](https://arxiv.org/abs/2303.15343?s=09) training recipe.
    
    <br>
    <img src="readme_images/pali_model.png" height="250"> <img src="readme_images/pali3_model.png" height="250">

16. **[BEiT](https://arxiv.org/abs/2106.08254) 2021**:
    - BEiT (Bidirectional Encoder representation from Image Transformers) is a self-supervised ViT inspired by BERT. 
    - The model learns to recover the visual tokens of the original image, instead of the raw pixels, through masked image modeling (MIM) process, where the objective is to maximize the log-likelihood of the correct visual tokens from the masked image.
    - The image is tokenized with discrete variational autoencoder (dVAE) from [DALL-E](https://arxiv.org/abs/2102.12092), which significantly outperforms naïve pixel-level auto-encoding techniques that suffers from high memory requirement, inability to capture the big picture (too focused on miniscule details), and short-range dependencies between pixels. 
    - The most recent work, [BEiT-3](https://arxiv.org/abs/2208.10442) from 2023, uses Multiway Transformers from [VLMo](https://arxiv.org/abs/2111.02358) as the backbone to enable deep fusion and modality-specific encoding, allowing the model to be transferred to various vision-language downstream tasks. 
      - BEiT-3 handles text and images in the same way without dramatic changes by regarding the image as a foreign language (Imglish).
      - As shown in the high-level model architecture, BEiT-3 uses a pool of modality experts to capture more modality-specific information.
        - BEIT-3, a general-purpose multimodal foundation model, achieved state-of-the-art performance across a wide range of vision and vision-language benchmarks.
    
    <br>
    <img src="readme_images/beit_model.png" height="250"> <img src="readme_images/beit3_model.png" height="250">

[//]: # (GPT-4V)
[//]: # (LLaVA-1.5)

# <ins>Model Selection</ins>

The paper [Vision-Language Pre-training: Basics, Recent Advances, and Future Trends](https://arxiv.org/abs/2210.09263) is a comprehensive resource for learning the recent trend of vision-langauge models and pre-training strategies. 

[//]: # (https://huggingface.co/blog/vision_language_pretraining)

[//]: # (As per evaluation methods, examples include BLEU, ROUGE, METEOR, and CIDEr according to this [article]&#40;https://encord.com/blog/vision-language-models-guide/&#41;.)

[//]: # ()
[//]: # (https://arxiv.org/pdf/2202.10936.pdf A Survey of Vision-Language Pre-Trained Models)

[//]: # ()
[//]: # (Empirical data:)

[//]: # ([paper with code image captioning comparison]&#40;https://paperswithcode.com/task/image-captioning&#41;)

[//]: # ([Hugging Face's GIT-bBase implementation]&#40;https://huggingface.co/docs/transformers/model_doc/git&#41;)

[//]: # (GIT-Base is 129M parameters)

[//]: # ()
[//]: # (SalesForce)

[//]: # ([Hugging Face's BLIP-2 implementation]&#40;https://huggingface.co/docs/transformers/model_doc/blip-2&#41;)

[//]: # (BLIP-2 2.7B parameters)

# <ins>Data Collection</ins>

## Data Ethics

In the development of the Fish Tank Doctor, a paramount consideration was the ethical and responsible collection of data. Recognizing the sensitive nature of user-generated content and the importance of privacy, I meticulously adhered to ethical principles throughout the data-gathering process. There are many resources about data ethics online like [this](https://www.intechopen.com/chapters/1121510), [this](https://medium.com/analytics-vidhya/data-ethics-in-artificial-intelligence-machine-learning-72467b9c70f3), [this](https://www.dataversity.net/machine-learning-data-governance-and-data-ethics/), and [this](https://online.hbs.edu/blog/post/data-ethics). 

I needed two types of data images of various fish tank setups and corresponding tank diagnoses in text. 

First, raw images were obtained through web-scraping. I followed these [guidelines](https://blog.apify.com/is-web-scraping-legal/) to conduct ethical web scraping. 

* The data scraper acts as a good citizen of the web and does not seek to overburden the targeted website.
* The information copied was publicly available and not behind a password authentication barrier.
* The information copied was primarily factual in nature, and the taking did not infringe on the rights — including copyrights — of another.
* The information was used to create a transformative product and was not used to steal market share from the target website by luring away users or creating a substantially similar product.

Second, tank diagnosis data was obtained through my work and voluntary participation from online fish-keeping community members. Here are some data ethics practices I employed:

1. User Consent and Anonymity:
   
    Explicit consent was obtained before any data collection began to ensure voluntary participation. All collected data is anonymized by removing any personally identifiable information from the dataset to preserve the subject's privacy. They were also informed about their right to retract their shared data at any time. 
   
2. Transparency and Accountability:
   
    A commitment to transparency was maintained throughout the data collection process. Data subjects were informed about the purpose of data collection, the types of information gathered, the intended use of data, and the tool's functionality.
    
3. Inclusivity and Diversity:
   
    Striving for a representative dataset, the collection process was designed to be inclusive of various fish tank setups. This approach not only promotes diversity in the dataset but also prevents biases, ensuring that the Fish Tank Doctor delivers reliable results across various scenarios.

## Automatic Data Annotation 

Data collection is one of the most resource-intensive steps of a machine learning project, and challenges includes complexity, subjectivity, and diversity, often requiring domain expertise. The recent advancement of Large Language Models (LLMs) presents a revolutionary opportunity to automate the data annotation process. [This paper](https://arxiv.org/abs/2402.13446) surveyed the potential of the latest LLMs for data annotation task which includes LLM-Based Data Annotation, Assessing LLM-Generated Annotations, Learning with LLM-Generated Annotations, and Challenges and Ethical Considerations.

<img src="readme_images/llm_data_annotation.png" height="450">

Data scientists spend around 80% of their time preparing and managing data for analysis, according to [this Forbes article](https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/?sh=79dcbd16f637). Privacy/copyright concern is another hurdle. I attempt to solve those issues and streamline the data collection process with automatic labeling and synthetic data.


## Synthetic Data

Synthetic data in machine learning refers to artificially generated data that mimics the characteristics of real-world data. 

| Advantages of Synthetic Data |                                                                                       Note                                                                                        |
|:----------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|         More Ethical         |         Most importantly, it doesn’t expose sensitive data or breach copyright since it's made up. Therefore, it also removes the regulatory hurdles of collecting them.          |
|        More Flexible         |                                        It allows organizations to share or distribute datasets without compromising sensitive information.                                        |
|      Faster and Cheaper      | By minimizing regulatory hurdles, data collection time, data cleaning time, etc., one can quickly obtain a large amount of data at a lower cost, leading to greater productivity. |
|    Better Quality Control    |                     Having full control over the synthesis process, you can make changes and tests easier, such as adjusting data diversity and complexity.                     |


| Disadvantages of Synthetic Data |                                                                                                          Note                                                                                                          |                                                                                           Potential Solutions                                                                                            |
|:-------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|         Limited Realism         | Synthetic data may not perfectly capture the complexities and nuances of real-world scenarios. Creating high-quality, complex synthetic data requires a deep understanding of the data distribution and relationships. |                                                                         Combine synthetic data with real data in a balanced way.                                                                         |
|  Risk of Overfitting and Bias   |                                                   Over-reliance on synthetic data can lead to overfitting and bias since the data is created by a man-made machine.                                                   |                   Employ bias detection and mitigation techniques like bias metrics, bias indicators, regularization, over/under-sampling, etc. Also, fine-tune it with real-world data.                   |
|  Privacy and Security Concerns   |                          If the synthetic data generation process doesn't adequately mimic the real data, it could unintentionally reveal information about the original, sensitive dataset.                           | Regularly monitor the model to identify and address any potential privacy or security vulnerabilities. Anonymize the original data and adhere to legal and ethical guidelines when using synthetic data. |

### Example of Caption Generation

<img src='readme_images/overstocked_tank_example.jpg' height=300>

This is an example image of an overstocked aquarium. 

|   Captioner    |                                                   Concise Caption                                                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               Elaborated Caption                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|:--------------:|:-------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| OpenAI GPT API |          Overstocked, Water change needed, More filtration, Bigger habitat, Algae control, Plant diversity          | It seems like your aquarium is facing a few challenges that need attention. First, having an overstocked aquarium can lead to issues like inadequate space for the inhabitants and increased waste, impacting water quality. To address this, prioritize a gradual rehoming of some fish or other aquatic creatures to achieve a balanced ecosystem. Additionally, regular water changes are crucial to maintain water quality, so schedule these consistently. Considering the need for more filtration, upgrading your filtration system or adding supplementary filtration devices can improve water circulation and quality. A larger habitat would not only accommodate the current residents better but also allow for better territorial separation. Introduce algae-eating organisms or adjust lighting to control algae growth effectively. Lastly, enhance plant diversity as they contribute to a healthier ecosystem by absorbing excess nutrients and providing hiding spots. Gradual changes and careful monitoring will help transform your aquarium into a thriving, well-balanced environment. |
|     Human      | Overstocked, Water change needed, Filter needed, Bigger tank needed, Algae, Bacterial bloom, Stay away from window. |                                                                                                                                                                                                                         This tank is so overcrowded with goldfish that I cannot even count their numbers. Rehome the fish, or you will need a pond to keep them all. Also, please do not put the tank under direct sunlight as it can cause algae bloom. The water looks foggy, so the fish are probably suffering from algae bloom already, but it is more likely a bacterial bloom due to too much fish waste from the fish. As an immediate remedy, do a 50% water change daily until the water clears. Otherwise, those fish will die from ammonia poisoning. Where is the filter? This tank will probably need like three 30 gallon rated sponge filters to keep the fish alive."                                                                                                                                                                                                                          |

Human and GPT have their pros and cons. Humans are much better at visual analysis and more robust in dealing with edge cases like unrelated input images. GPT is better at elaborating on Concise Caption, producing more consistent and knowledgeable text responses. For example, GPT suggests enhancing plant diversity when there is no plant to begin with. Also, the tone of the human text is a bit aggressive for a helpful assistant. Therefore, the most efficient way to produce the real-world caption is to let humans analyze the image and let GPT elaborate on it. 

### Example of Data Augmentation

* Augment the image using image processing techniques like rotation, flip, zooming, translation, noise addition, cropping, brightness adjustment, and contrast adjustment. 
* For text paraphrasing, use GPT-3.5 Turbo again because it's very good at text generation tasks like paraphrasing.  

## Data Pipeline

After carefully considering said topics, following is what I got for the data collection process:

1. Image collection
   * Automate it using a custom web-scraping code. 
   * Future work: Train a GAN model to generate images. 
2. Image filtering and concise captioning
   * Use GPT-4 Vision for the initial image captioning task. 
   * The request prompt is engineered to produce aquarium maintenance insights in a concise and consistent list format to minimize the output token cost while maximizing the output information. 
   * Prompt: `"You are an expert aquarist. List good maintenance signs and potential improvements for the aquarium in this image using under five words each, * as a bullet point, and without headers."`
   * Sometimes, the AI will say that they can't help me or have trouble with the captioning task. I can add a prompt like `"Say 'STOPPED' if the image is not about aquariums or if you can't help me with this task."` but that requires a lot of tokens. Instead, I can just stop the AI mid-generation with the stop-sequence parameter and filter out those incomplete captions with other scripts (detect missing new lines or bullet points).
3. Combine with real-world data
   * Combine the real-world data (ones ethically collected and ones produced myself) with the synthetic ones to ensure data diversity and enhance realism in data. 
4. Caption elaboration
   * I let GPT-3.5 Turbo, a cheaper and dialog fine-tuned model, elaborate on the list from the image captioning task. 
   Prompt: `"You are an expert aquarist. In a paragraph, elaborate on this aquarium:" + the short caption`
5. Human check
   * Check and fix for any errors or shortcomings in the captions before augmenting them to produce more data. 
6. Data augmentation 
   * Augment the image using image processing techniques like rotation, flip, zooming, translation, noise addition, cropping, brightness adjustment, and contrast adjustment. 
   * For text paraphrasing, use GPT-3.5 Turbo again because it's very good at text generation tasks like paraphrasing.  
7. Human check
   * The final check for quality assurance. 

<img src='readme_images/caption_gui_example.gif' height=350>

In addition, to streamline the manual image captioning part, I developed a GUI, `caption_gui.py`, to easily save and load the captions to a CSV file format. This tool automates most of the tedious work of file organization and formatting so the caption writers can focus on producing quality captions. 

The save format is HuggingFace's `ImageFolder` caption dataset generation method's metadata format, which is described [here](https://huggingface.co/docs/datasets/main/en/image_dataset#image-captioning):

```
file_name,additional_feature
0001.png,First caption text
0002.png,Second caption text
0003.png,Third caption text
```



[//]: # ()
[//]: # (https://cameronrwolfe.substack.com/p/the-story-of-rlhf-origins-motivations history)

[//]: # ()
[//]: # (https://encord.com/blog/guide-to-rlhf/)

[//]: # ()
[//]: # (https://huggingface.co/blog/rlhf)

[//]: # ()
[//]: # (https://www.linkedin.com/advice/0/how-can-you-effectively-evaluate-nlp-model-image automatic metrics)
