---
license: cc-by-4.0
language:
- en
pretty_name: HelpSteer2
size_categories:
- 10K<n<100K
tags:
  - human-feedback
dataset_info:
  features:
    - name: prompt
      dtype: string
    - name: response
      dtype: string
    - name: helpfulness
      dtype: int32
    - name: correctness
      dtype: int32
    - name: coherence
      dtype: int32
    - name: complexity
      dtype: int32
    - name: verbosity
      dtype: int32
  splits:
    - name: train                  
      num_examples: 20324
    - name: validation                  
      num_examples: 1038
---

# HelpSteer2: Open-source dataset for training top-performing reward models


HelpSteer2 is an open-source Helpfulness Dataset (CC-BY-4.0) that supports aligning models to become more helpful, factually correct and coherent, while being adjustable in terms of the complexity and verbosity of its responses.
This dataset has been created in partnership with [Scale  AI](https://scale.com/). 

When used with a [Llama 3 70B Base Model](https://huggingface.co/meta-llama/Meta-Llama-3-70B), we achieve 88.8% on RewardBench, which makes it the 4th best Reward Model as of 12 Jun 2024.
This model is available on HF at [Llama3-70B-SteerLM-RM](https://huggingface.co/nvidia/Llama3-70B-SteerLM-RM).

Reward Models was trained using the open-source [NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner) following [SteerLM training user guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/steerlm.html).

HelpSteer2 is a follow-up to the popular [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) dataset and we recommend using HelpSteer2 instead of HelpSteer.

HelpSteer2 Paper : [HelpSteer2: Open-source dataset for training top-performing reward models](http://arxiv.org/abs/2406.08673)


## RewardBench Primary Dataset LeaderBoard


 | Model  | Type of Model|  Overall | Chat | Chat-Hard | Safety | Reasoning | 
|:-----------------------------|:----------------|:-----|:----------|:-------|:----------|:-----------------------|
  | _**Nemotron-4-340B-RM**_  | Trained with Permissive Licensed Data | **92.0**  | 95.8 |   **87.1** | 91.5  | 93.7 | 
  | ArmoRM-Llama3-8B-v0.1 | Trained with GPT4 Generated Data|  90.8 | 96.9     | 76.8  | 92.2 | 97.3  | 
  | Cohere May 2024   | Proprietary LLM |   89.5  | 96.4     | 71.3      | 92.7 | 97.7  | 
  | _**Llama3-70B-SteerLM-RM**_  | Trained with Permissive Licensed Data | 88.8  | 91.3 |   80.3 | **92.8**  | 90.7 | 
  | Google Gemini Pro 1.5 | Proprietary LLM |  88.1 | 92.3  | 80.6 | 87.5  | 92.0  | 
  | RLHFlow-Llama3-8B | Trained with GPT4 Generated Data |   87.1   |  **98.3**  |  65.8   |    89.7     |   94.7 | 
  | Cohere March 2024 | Proprietary LLM | 87.1|  94.7 | 65.1 | 90.3 | **98.7** |
  | GPT-4-0125-Preview|Proprietary LLM |   85.9   | 95.3     | 74.3      | 87.2     | 86.9      | 
  | Claude 3 Opus 0229 | Proprietary LLM | 80.7 | 94.7 | 60.3 | 89.1 | 78.7 | 
  
Last updated: 12 Jun 2024

Note that we only consider the first four categories in RewardBench, because the optional fifth category (Prior Sets) is 
1. Heavily towards models trained on Anthropic HHH, Anthropic Helpful, OpenAI Summarize and Stanford Human Preferences (constituent datasets for the Prior Sets category) and therefore can be easily gamed (see About page on RewardBench)
2. Extremely noisy with many constituent datasets (e.g. Anthropic Helpful, OpenAI summarize) not being able to reach val accuracy beyond ~0.70 even if training on the training set alone, suggesting unchecked errors in annotation.
3. Not reported by several models such as Google Gemini Pro 1.5 and Claude 3 Opus 0229, making comparisons unfair since Prior Sets typically has lower scores than other categories

## Dataset Description

HelpSteer contains 21, 362 samples, each containing a prompt, a response as well as five human-annotated attributes of the response, each ranging between 0 and 4 where higher means better for each attribute. Consecutive samples (e.g. sample 1 with 2, 3 with 4 ...) share the same prompt so this can be used for preference pairs based on the helpfulness score (e.g. training DPO or Preference RM), in addition to training SteerLM Regression RM. 

About 29% of all prompts used are multi-turn. In this case, the prompt consists of all of the user turns and all but the last assistant turn, which is contained in the response field. This is done because the attribute values only assessed only for the last assistant turn.

For multi-turn prompts, the structure of prompts look like below. This is how we used it for training SteerLM Regression Reward Models, but you need to use it in other settings, please adjust as required.

```json
{user turn 1}
<extra_id_1>Assistant
{assistant turn 1}
<extra_id_1>User
{user turn 2}
```

These attributes are:

1. **Helpfulness**: Overall helpfulness of the response to the prompt.
2. **Correctness**: Inclusion of all pertinent facts without errors. 
3. **Coherence**: Consistency and clarity of expression. 
4. **Complexity**: Intellectual depth required to write response (i.e. whether the response can be written by anyone with basic language competency or requires deep domain expertise).
5. **Verbosity**: Amount of detail included in the response, relative to what is asked for in the prompt.


## Example

```json
{
  "prompt": "c#", 
  "response": "C# is a high-level, object-oriented programming language developed by Microsoft as part of its .NET initiative. It was created as a modern alternative to Java and supports a variety of programming paradigms, including imperative, functional, and event-driven. C# is primarily used for Windows application development, but it can also be used for web, mobile, and game development. The language is designed to be safe, secure, and efficient, and it provides developers with a rich set of libraries and tools for building robust and scalable applications. C# is also widely used in the game development industry, particularly in the development of games for the Xbox 360 and Xbox One consoles.", 
  "helpfulness": 3, 
  "correctness": 4, 
  "coherence": 4, 
  "complexity": 2, 
  "verbosity": 1
}
```


## Using the Huggingface Datasets

```python
from datasets import load_dataset

ds = load_dataset("nvidia/HelpSteer2")

train = ds['train'] # len(train) = 20324 (95%)
val = ds['validation']     # len(val) = 1038 (5%)
```

## Source

1. Prompts are collected based on mostly user-contributed ShareGPT prompts and with a small proportion (~5%) that are human generated by Scale AI. 
2. Responses are generated by early versions of a mix of 10 different inhouse LLMs (note: none from properitary LLM providers such as OpenAI). We generate 2 responses per prompts (each from a different model) using sampling techniques to give diverse yet reasonable responses.
3. Annotations of various attributes were done by Scale AI. Annotators rated each response on a Likert 5 scale (between 0 and 4) for each attribute (helpfulness, correctness, coherence, complexity and verbosity).

## Annotation methodology (short)	

1. We engaged a select group of contractors via Scale AI. These contractors were provided with comprehensive guidelines that defined each attribute and the criteria for every rating level, together with some annotated examples. These guidelines and examples are detailed in the Appendix of the accompanying paper.
2. The annotation process involved approximately 1000 U.S.-based human annotators. Candidates first underwent preliminary assignments, including assessments of English proficiency, to determine eligibility for working on the project. Subsequently, they participated in an introductory training course on the task which ended with a test that involved annotating 35 sample responses. This process ensured not only a thorough understanding of the task requirements but also the delivery of high-quality annotations.
3. Every sample was independently annotated by a minimum of three annotators and up to five annotators, if the initial annotators do not agree with each other sufficiently (2 points or less on helpfulness). The final annotations (mean of 3.41 annotators) were obtain by taking the mean of the three annotators who agree with each other most, rounded to the nearest integer. 
4. Post-annotations, Scale AI performed extensive quality assurance, with each annotation reaching a minimum of two human reviews in addition to automated checks. After receiving the annotations from Scale AI, we conducted our independent quality assurance to make sure that the quality of the annotations was up to our expectations. As a result, many annotations were filtered away to retain only 20, 324 samples. 


## Ethical statement	
Annotators for the dataset were contracted through Scale AI. Scale AI engages the Anker Methodology, GISC Impact Sourcing Standard, and UN Sustainable Development Goals to provide a fair and competitive pay. The specific pay is calculated based on many factors, including the specific project, the specialized skillset and expertise required, regional costs of living and then transparently listed on Scale AI platform. Scale AI also provides multiple channels for questions and support, including 24/7 support teams, community discussion channels with specially trained moderators, and a “speak up” hotline where contractors can report concerns anonymously. Worker concerns can be submitted to and are reviewed by our Remotasks support team, and pay disputes are reviewed by support specialists trained in this area. 


## Contact

E-Mail: [Zhilin Wang](mailto:zhilinw@nvidia.com)

## Citation

If you find this dataset useful, please cite the following works

```bibtex
@misc{wang2024helpsteer2,
      title={HelpSteer2: Open-source dataset for training top-performing reward models}, 
      author={Zhilin Wang and Yi Dong and Olivier Delalleau and Jiaqi Zeng and Gerald Shen and Daniel Egert and Jimmy J. Zhang and Makesh Narsimhan Sreedhar and Oleksii Kuchaiev},
      year={2024},
      eprint={2406.08673},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```