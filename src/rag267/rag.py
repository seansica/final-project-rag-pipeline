import os
from typing import Dict, List, Optional, Any, Callable
import enum
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
from langchain_community.vectorstores import Qdrant


class SupportedGeneratorModels(enum.Enum):
    MistralInstructV2 = "mistralai/Mistral-7B-Instruct-v0.2"


class RAGSystem:
    def __init__(
        self,
        vectorstore: Qdrant,
        engineering_template_path: str = "templates/engineering_template.txt",
        marketing_template_path: str = "templates/marketing_template.txt",
        cohere_api_key: Optional[str] = None,
        use_mistral: bool = True,
        use_cohere: bool = True,
        mistral_model_name: SupportedGeneratorModels = SupportedGeneratorModels.MistralInstructV2,
        top_k: int = 4,
    ):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        self.top_k = top_k

        self.mistral_llm = None
        self.cohere_llm = None

        if use_mistral:
            self.mistral_llm = self._init_mistral(mistral_model_name)

        if use_cohere:
            if not cohere_api_key:
                raise ValueError("Cohere API key is required when use_cohere=True")
            self.cohere_llm = self._init_cohere(cohere_api_key)

        self.engineering_template = self._load_template(engineering_template_path)
        self.marketing_template = self._load_template(marketing_template_path)

        self.engineering_prompt = ChatPromptTemplate.from_template(
            self.engineering_template
        )
        self.marketing_prompt = ChatPromptTemplate.from_template(
            self.marketing_template
        )

        self.output_parser = StrOutputParser()

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

    def reload_templates(
        self,
        engineering_template_path: str = "templates/engineering_template.txt",
        marketing_template_path: str = "templates/marketing_template.txt",
    ):
        self.engineering_template = self._load_template(engineering_template_path)
        self.marketing_template = self._load_template(marketing_template_path)

        # Re-create prompt templates
        self.engineering_prompt = ChatPromptTemplate.from_template(
            self.engineering_template
        )
        self.marketing_prompt = ChatPromptTemplate.from_template(
            self.marketing_template
        )

        logger.info("Templates reloaded successfully")

    def _init_mistral(self, model_name: SupportedGeneratorModels):
        logger.info(f"Initializing Mistral model: {model_name}")

        quant_cfg = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_name="auto",
            quantization_config=quant_cfg,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1000,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
        )

        pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

        return HuggingFacePipeline(pipeline=pipe)

    def _init_cohere(self, api_key: str):
        logger.info("Initializing Cohere model")
        return ChatCohere(cohere_api_key=api_key, model="command-r")

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

    def answer_for_engineers(self, query: str) -> str:
        """Generate an answer for the engineering team."""
        if not self.mistral_llm:
            raise ValueError("Mistral model not initialized")

        # Create chain
        chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | self.engineering_prompt
            | self.mistral_llm
            | self.output_parser
        )

        # Run chain
        return chain.invoke(query)

    def answer_for_marketing(self, query: str) -> str:
        if not self.cohere_llm:
            llm = self.mistral_llm if self.mistral_llm else None
            if not llm:
                raise ValueError("No LLM initialized")
        else:
            llm = self.cohere_llm

        chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | self.marketing_prompt
            | llm
            | self.output_parser
        )

        return chain.invoke(query)

    def generate_responses(self, query: str) -> Dict[str, str]:
        """Generate responses for both engineering and marketing teams."""
        engineering_response = self.answer_for_engineers(query)
        marketing_response = self.answer_for_marketing(query)

        return {"engineering": engineering_response, "marketing": marketing_response}

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
