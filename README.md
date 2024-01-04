# Fine-tuning LLM for Various E-commerce Tasks

In this repo we will fine-tune llama2-7B on four different e-commerce tasks using LoRA. 
The tasks are following:
* Product Classification
* Product title NER
* Description generation based on product title
* Products review summarization
We have different baselines for each task: BERT for classification, BERT for NER, GPT2/BART/T5 for description generation, BART/T5 for summarization.

The questions we want to explore:
* How many training samples do we need to achieve SOTA results with fine-tuning llama2-7B for each task?
* Is LLM a good choice for e-commerce tasks compared to traditional baseline models?
* Fine-tuning LLM on a mixed training dataset for all tasks, or fine-tuning LLM for single task one by one, which one has better performance?
* Can we get performance gain if we merge LoRA weights with different task? Is LoRA merging a good way to explore?
* What are correlations between LoRA weights for every two tasks, and the correlations between every two tasks?