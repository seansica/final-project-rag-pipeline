import time
from typing import Dict, List, Optional, Any
# import enum
# from dotenv import load_dotenv
from loguru import logger
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from langchain_huggingface import HuggingFacePipeline
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_community.vectorstores import Qdrant

from .vectordb.manager import VectorDatabaseManager
from .vectordb.utils import SupportedGeneratorModels, ModelType, Team

# class SupportedGeneratorModels(enum.Enum):
#     MistralInstructV2 = "mistralai/Mistral-7B-Instruct-v0.2"

# class ModelType(enum.Enum):
#     Mistral = "mistral"
#     Cohere = "cohere"

# class Team(enum.Enum):
#     Engineering = "engineering"
#     Marketing = "marketing"

class RAGSystem:
    def __init__(
        self,
        vector_db_manager: VectorDatabaseManager,
        engineering_template_path: str = "templates/engineering_template.txt",
        marketing_template_path: str = "templates/marketing_template.txt",
        cohere_api_key: Optional[str] = None,
        use_cohere: bool = True,
        use_mistral: bool = False,
        mistral_model_name: SupportedGeneratorModels = SupportedGeneratorModels.MistralInstructV2,
        top_k: int = 4,
    ):

        if use_cohere and use_mistral:
            raise ValueError("use_cohere and use_mistral cannot both be True. Choose one LLM.")

        # load_dotenv()

        self.config = {
            'llm': {},
            'top_k': top_k,
            'engineering_template_path': engineering_template_path,
            'marketing_template_path': marketing_template_path,
        }

        # self.hf_token = os.getenv('HF_TOKEN')

        self.vdm = vector_db_manager
        self.top_k = top_k
        # self.vectorstore = vdm.vectorstore
        # self.retriever = vector_db_manager.vectorstore.as_retriever(search_kwargs={"k": top_k})

        if use_mistral:
            self.llm = self._init_mistral(mistral_model_name)

        if use_cohere:
            if not cohere_api_key:
                raise ValueError("Cohere API key is required when use_cohere=True")
            self.llm = self._init_cohere(cohere_api_key)

        self.engineering_template = self._load_template(engineering_template_path)
        self.marketing_template = self._load_template(marketing_template_path)

        self.engineering_prompt = ChatPromptTemplate.from_template(
            self.engineering_template
        )
        self.marketing_prompt = ChatPromptTemplate.from_template(
            self.marketing_template
        )

        self.output_parser = StrOutputParser()

    @property
    def retriever(self):
        return self.vdm.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    def get_config(self) -> dict:
        summary = {
            **self.config,
            'vectorstore': self.vdm.get_config()
        }
        return summary

    def _load_template(self, template_path: str) -> str:
        try:
            with open(template_path, "r") as f:
                template = f.read()
            return template
        except FileNotFoundError as e:
            logger.exception(e)
            logger.warning(
                f"Template file {template_path} not found. Using default template."
            )
            return e

    def _init_mistral(self, model_name: SupportedGeneratorModels):
        logger.info(f"Initializing Mistral model: {model_name}")

        quant_cfg = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            quantization_config=quant_cfg,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        temperature = 1.0
        max_new_tokens = 500
        top_p = 0.95
        repetition_penalty = 1.2

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
        )

        pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

        self.config['llm'] = {
            'family': ModelType.Mistral,
            'model_name': model_name,
            'quantization': {'load_in_4bit': True},
            'tokenizer': tokenizer.name_or_path,
            'temperature': temperature,
            'top_p': top_p,
            'max_new_tokens': max_new_tokens,
            'repetition_penalty': repetition_penalty,
        }

        return HuggingFacePipeline(pipeline=pipe)

    def _init_cohere(self, api_key: str):
        logger.info("Initializing Cohere model")
        model_name = "command-r"
        self.config['llm'] = {
            'family': ModelType.Cohere,
            'model_name': model_name,
            'api_key_provided': bool(api_key)
        }
        return ChatCohere(cohere_api_key=api_key, model=model_name, timeout_seconds=60)

    def format_docs(self, docs):
        """Format a list of documents into a string."""
        return "\n\n".join(
            f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)
        )

    def query_vectorstore(self, query: str) -> List[Any]:
        """Get retrieved documents for a query."""
        return self.retriever.invoke(query)

    def get_retrieval_metadata(self, query: str) -> List[Dict[str, Any]]:
        """Get metadata about the retrieved documents."""
        docs = self.query_vectorstore(query)
        return [doc.metadata for doc in docs]

    def invoke(self, team: Team, query: str) -> str:
        """Generate an answer for the engineering team."""

        if not isinstance(team, Team):
            raise ValueError(f"Invalid team: {team}")

        if not self.llm:
            raise ValueError("LLM not initialized")

        prompt = self.engineering_prompt if team == Team.Engineering else self.marketing_prompt

        # Create chain
        chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | self.output_parser
        )

        # Run chain without retry:
        # return chain.invoke(query)
        
        # Run chain with retry logic:
        max_retries = 3
        retry_delay = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                return chain.invoke(query)
            except Exception as e:
                if attempt < max_retries - 1:  # Don't sleep after the last attempt
                    logger.warning(f"Error during invoke (attempt {attempt+1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise

    def get_document_sources(self, query: str) -> List[str]:
        docs = self.query_vectorstore(query)
        sources = []

        for doc in docs:
            source_info = ""
            if "doc_source" in doc.metadata:
                source_info += doc.metadata["doc_source"]
            if "source_id" in doc.metadata:
                source_info += f": {doc.metadata['source_id']}"
            sources.append(source_info)

        return sources
