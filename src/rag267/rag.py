import time
from typing import Dict, List, Optional, Any, Union, Callable
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
from langchain_core.retrievers import BaseRetriever

from .vectordb.manager import VectorDatabaseManager
from .vectordb.utils import SupportedGeneratorModels, ModelType, Team


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
        retriever_type: str = "similarity",
        retriever_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if use_cohere and use_mistral:
            raise ValueError("use_cohere and use_mistral cannot both be True. Choose one LLM.")

        # Default retriever_kwargs if none provided
        if retriever_kwargs is None:
            retriever_kwargs = {"k": top_k}
        
        # If k not in kwargs but top_k provided, add it
        if "k" not in retriever_kwargs and top_k is not None:
            retriever_kwargs["k"] = top_k

        self.config = {
            'llm': {},
            'retriever': {
                'type': retriever_type,
                'kwargs': retriever_kwargs,
            },
            'engineering_template_path': engineering_template_path,
            'marketing_template_path': marketing_template_path,
        }

        self.vdm = vector_db_manager
        self.top_k = top_k
        self.retriever_type = retriever_type
        self.retriever_kwargs = retriever_kwargs

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
    def retriever(self) -> BaseRetriever:
        """Configure and return the appropriate retriever based on retriever_type"""
        if self.retriever_type == "similarity":
            return self.vdm.vectorstore.as_retriever(search_kwargs=self.retriever_kwargs)
        elif self.retriever_type == "mmr":
            return self.vdm.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs=self.retriever_kwargs
            )
        elif self.retriever_type == "similarity_score_threshold":
            kwargs = self.retriever_kwargs.copy()
            score_threshold = kwargs.pop("score_threshold", 0.5)
            return self.vdm.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={**kwargs, "score_threshold": score_threshold}
            )
        elif self.retriever_type == "multi_query":
            from langchain.retrievers import MultiQueryRetriever
            
            llm_for_queries = self.llm  # Default to same LLM
            # If specified in kwargs, use that instead
            if "llm_for_queries" in self.retriever_kwargs:
                llm_key = self.retriever_kwargs["llm_for_queries"]
                # Initialize the specific LLM for query generation
                # This is simplified - you'd need proper initialization based on model type
                if llm_key == "mistral":
                    from langchain_community.llms import HuggingFacePipeline as HFPipeline
                    llm_for_queries = HFPipeline(model_name=SupportedGeneratorModels.MistralInstructV2.value)
                elif llm_key == "cohere":
                    api_key = self.retriever_kwargs.get("api_key", None)
                    llm_for_queries = ChatCohere(cohere_api_key=api_key, model="command-r")
            
            kwargs = {k: v for k, v in self.retriever_kwargs.items() 
                     if k not in ["llm_for_queries", "api_key"]}
            
            base_retriever = self.vdm.vectorstore.as_retriever(search_kwargs=kwargs)
            
            return MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm_for_queries
            )
        else:
            logger.warning(f"Unknown retriever type: {self.retriever_type}, falling back to similarity search")
            return self.vdm.vectorstore.as_retriever(search_kwargs=self.retriever_kwargs)

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

    def clean_response(self, response: str) -> str:
        """Clean up LLM response to extract only the model's answer.
        
        Removes the instruction prompt and any other artifacts, keeping only the 
        generated content from the LLM.
        """
        # If the response contains [/INST], extract only the text after it
        if "[/INST]" in response:
            return response.split("[/INST]", 1)[1].strip()
        
        # For other formats, just return the original response
        return response

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

        # Run chain with retry logic:
        max_retries = 3
        retry_delay = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                response = chain.invoke(query)
                # Clean the response to extract only the model's answer
                return self.clean_response(response)
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