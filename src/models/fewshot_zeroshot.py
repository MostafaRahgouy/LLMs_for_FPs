import torch
from tqdm import tqdm
from typing import Dict, List
import time
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers

huggingface_access_token = ""  # todo: add your huggingface key here


class RetrievalModel:
    def __init__(self) -> None:
        print("Retriever Called!")
        self.original_docs = None
        self.vectorizer = None
        self.documents_embeddings = None

    def fit(self, documents: List[Dict]):
        self.original_docs = documents
        self.vectorizer = TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True, norm=None)
        combined_docs = ["Problem:" + item['input'] + "\nSolution: " + item['output'] for item in self.original_docs]
        self.documents_embeddings = self.vectorizer.fit_transform(combined_docs)

    def retrieve(self, query: str, k: int) -> List[Dict]:
        query_embed = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_embed, self.documents_embeddings)
        top_k_indexes = torch.topk(torch.Tensor(similarities), k).indices[0].numpy()
        top_k_items = [self.original_docs[index] for index in top_k_indexes]
        return top_k_items


class NShotModeling:
    output_data_key_text: str = None

    def __init__(self, model_name: str, max_tokens: int) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens

    def is_it_done(self, input_dataset: List[Dict]) -> bool:
        for input_data_block in input_dataset:
            if input_data_block['check'] == False:
                return False
        return True

    def init_output(self, input_dataset: List[Dict]) -> List[Dict]:
        output_data = []
        for index in range(len(input_dataset)):
            output_data.append({"check": False,
                                "input": input_dataset[index]['input'],
                                "output": input_dataset[index]['output']
                                })
        return output_data

    def predict(self, input_dataset: List[Dict]) -> List[Dict]:
        output_data = self.init_output(input_dataset=input_dataset)

        assert self.is_it_done(output_data) == False

        while not self.is_it_done(output_data):
            for index, input_data_block in enumerate(tqdm(output_data)):
                if output_data[index]['check'] != True:
                    try:
                        text_output = self.call_model(input_data_block=input_data_block)
                        output_data[index][self.output_data_key_text] = text_output
                        output_data[index]['check'] = True
                    except Exception as err:
                        print(f"UNexpected {err}, {type(err)}")
                        print("Going to sleep for 10 second!")
                        time.sleep(10)
        return output_data

    def call_model(self, input_data_block: Dict) -> str:
        pass

    def fill_prompt(self, input_data_block: Dict) -> str:
        pass


class ZeroShotOpenAI(NShotModeling):
    output_data_key_text: str = "openai_out_text"

    def call_model(self, input_data_block: Dict) -> str:
        prompt = self.fill_prompt(input_data_block=input_data_block)
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=1.0,
        )
        return response['choices'][0]['message']['content']

    def fill_prompt(self, input_data_block: Dict) -> str:
        prompt_template = """Answer the following question with an approximation of a numeric value.

        Format output as JSON with just `description` and `answer` keys.

        Question:
        ```
        {input}
        ```
        Answer:
        """

        question = input_data_block['input']
        filled_prompt = prompt_template.replace("{input}", question)
        return filled_prompt


class FewShotOpenAI(ZeroShotOpenAI):

    def init_output(self, input_dataset: List[Dict]) -> List[Dict]:
        output_data = []
        for index in range(len(input_dataset)):
            output_data.append({"check": False,
                                "input": input_dataset[index]['input'],
                                "output": input_dataset[index]['output'],
                                "retrieval_candidates": input_dataset[index]['retrieval_candidates'],
                                })
        return output_data

    def fill_prompt(self, input_data_block: Dict) -> str:
        prompt_template = """Answer the following question by reasoning step-by-step as a program.
        Format program as JSON with just a "program" key.

        Use the following question-program examples as samples.

            {fewshot}

            {input}
                
            answer:
                
        	"""

        fewshot_prompt_template = """
        {input}

        answer:
        {output}
        """
        fewshot_prompt = ""
        for retrieval_data_block in input_data_block['retrieval_candidates']:
            template = fewshot_prompt_template.replace("{input}", retrieval_data_block['input']).replace("{output}",
                                                                                                         retrieval_data_block[
                                                                                                             'output'])
            fewshot_prompt += template
        filled_prompt = prompt_template.replace("{fewshot}", fewshot_prompt).replace("{input}",
                                                                                     input_data_block['input'])
        return filled_prompt


class ZeroShotLLaMA(NShotModeling):
    output_data_key_text: str = "llama_out_text"

    def __init__(self, model_name: str, max_tokens: int) -> None:
        super().__init__(model_name=model_name, max_tokens=max_tokens)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, token=huggingface_access_token)
        llama = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            token=huggingface_access_token,
        )

        self.model = transformers.pipeline(
            "text-generation",
            model=llama,
            tokenizer=self.tokenizer
        )

    def call_model(self, input_data_block: Dict) -> str:
        prompt = self.fill_prompt(input_data_block=input_data_block)
        max_token = len(prompt) + 30
        sequences = self.model(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            # max_length=self.max_tokens,
            max_length=max_token,
            temperature=0.01
        )
        output = sequences[0]['generated_text'][len(prompt):]
        return output

    def fill_prompt(self, input_data_block: Dict) -> str:
        prompt_template = """Answer the following question with an approximation of a single numeric value.

        Output should be:
         {"answer": a single numeric value}
         
        
        question:
        {input}
        
        Output:
        """
        question = input_data_block['input']
        filled_prompt = prompt_template.replace("{input}", question)
        return filled_prompt


class FewShotLLaMA(ZeroShotLLaMA):

    def init_output(self, input_dataset: List[Dict]) -> List[Dict]:
        output_data = []
        for index in range(len(input_dataset)):
            output_data.append({"check": False,
                                "input": input_dataset[index]['input'],
                                "output": input_dataset[index]['output'],
                                "retrieval_candidates": input_dataset[index]['retrieval_candidates'],
                                })
        return output_data

    def fill_prompt(self, input_data_block: Dict) -> str:
        prompt_template = """Answer the following question by reasoning step-by-step as a program.

        Use the following question-program examples to find the question's program.

                {fewshot}

                Question:
                ```
                {input}
                ```
                Program:
        		"""

        fewshot_prompt_template = """
        ### Problem:
        ```
        {input}
        ```
        ### Solution:
        ```
        {output}
        ```"""
        fewshot_prompt = ""
        for retrieval_data_block in input_data_block['retrieval_candidates']:
            template = fewshot_prompt_template.replace("{input}", retrieval_data_block['input']).replace("{output}",
                                                                                                         retrieval_data_block[
                                                                                                             'output'])
            fewshot_prompt += template
        filled_prompt = prompt_template.replace("{fewshot}", fewshot_prompt).replace("{input}",
                                                                                     input_data_block['input'])
        return filled_prompt


def zero_few_shot_experimentation(test: List[Dict],
                                  train: List[Dict] = None,
                                  few_shot_no: int = 5,
                                  model_name: str = "NONE",
                                  max_tokens: int = 0,
                                  model_type: str = 'zeroshot-openai') -> List[Dict]:
    """
        model_types: 
            - `zeroshot-openai`
            - `fewshot-openai`
            - `zeroshot-llama`
            - `fewshot-llama`
        model_name:
            - `gpt-3.5-
    """
    if model_type.startswith("fewshot"):
        retriever = RetrievalModel()
        retriever.fit(train)
        model_inputs = []
        for _, query in enumerate(test):
            candidates = retriever.retrieve(query["input"], k=few_shot_no)
            query['retrieval_candidates'] = candidates
            model_inputs.append(query)

        if model_type.endswith("llama"):
            llm = FewShotLLaMA(model_name=model_name, max_tokens=max_tokens)
        else:
            llm = FewShotOpenAI(model_name=model_name, max_tokens=max_tokens)
    else:
        model_inputs = test
        if model_type.endswith("llama"):
            llm = ZeroShotLLaMA(model_name=model_name, max_tokens=max_tokens)
        else:
            llm = ZeroShotOpenAI(model_name=model_name, max_tokens=max_tokens)

    results = llm.predict(input_dataset=model_inputs)
    return results
