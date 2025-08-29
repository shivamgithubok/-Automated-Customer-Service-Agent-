from setuptools import setup, find_packages

setup(
    name="rag-agent-service",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "langchain>=0.1.0",
        "langchain-google-genai>=0.0.3",
        "langchain-community>=0.0.10",
        "faiss-cpu>=1.7.4",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "tiktoken>=0.5.1",
    ],
    author="Shivam",
    author_email="your.email@example.com",
    description="A RAG-based document processing and query service",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
