"""
DocuMind - PDF Processor Module
===============================
Extracts text from uploaded PDF documents, preserves page-level metadata,
validates document contents, and splits text into meaningful chunks for RAG.
"""

import io
from typing import List, Union, BinaryIO
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessingError(Exception):
    """Custom exception raised when PDF extraction or processing fails."""
    pass


class PDFProcessor:
    """
    Handles PDF validation, page-by-page text extraction, and text chunking
    while preserving page numbers and source document references.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDFProcessor with text splitting parameters.

        Args:
            chunk_size (int): Max characters per chunk. Default is 1000.
            chunk_overlap (int): Number of overlapping characters between chunks. Default is 200.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_text_from_pdf(
        self,
        file_input: Union[str, BinaryIO, io.BytesIO],
        file_name: str = "Uploaded_Document.pdf"
    ) -> List[Document]:
        """
        Extracts text from a single PDF, generating Document objects per non-empty page
        with rich metadata (source filename and page number).

        Args:
            file_input: File path, BytesIO, or file-like object of the PDF.
            file_name: The display name of the PDF for source referencing.

        Returns:
            List[Document]: List of LangChain Document objects per page.

        Raises:
            PDFProcessingError: If the PDF cannot be opened, is encrypted, or has no extractable text.
        """
        try:
            reader = PdfReader(file_input)
        except Exception as e:
            raise PDFProcessingError(f"Failed to read PDF '{file_name}': {str(e)}")

        if reader.is_encrypted:
            try:
                # Attempt to decrypt with empty password for passively encrypted PDFs
                reader.decrypt("")
            except Exception:
                raise PDFProcessingError(f"PDF '{file_name}' is password protected and cannot be opened.")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise PDFProcessingError(f"PDF '{file_name}' is empty (0 pages).")

        documents = []
        total_text_length = 0

        for page_idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            cleaned_text = page_text.strip()
            total_text_length += len(cleaned_text)

            if cleaned_text:
                doc = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": file_name,
                        "page": page_idx + 1,  # 1-indexed page number for human readability
                        "total_pages": total_pages
                    }
                )
                documents.append(doc)

        if total_text_length == 0:
            raise PDFProcessingError(
                f"No extractable text found in '{file_name}'. "
                "The PDF might contain scanned images without OCR."
            )

        return documents

    def process_multiple_pdfs(
        self,
        uploaded_files: List[BinaryIO]
    ) -> tuple[List[Document], dict]:
        """
        Extracts, validates, and chunks multiple uploaded PDF files.

        Args:
            uploaded_files: List of Streamlit UploadedFile or file-like objects.

        Returns:
            tuple[List[Document], dict]: A tuple containing:
                - List of chunked Document objects ready for vector storage.
                - Summary statistics dictionary (total_files, total_pages, total_chunks, errors).
        """
        all_page_docs: List[Document] = []
        errors: List[str] = []
        total_pages_count = 0

        for file in uploaded_files:
            file_name = getattr(file, "name", "uploaded_doc.pdf")
            try:
                # Seek to beginning if file was previously read
                if hasattr(file, "seek"):
                    file.seek(0)
                page_docs = self.extract_text_from_pdf(file, file_name=file_name)
                all_page_docs.extend(page_docs)
                if page_docs:
                    total_pages_count += page_docs[0].metadata.get("total_pages", len(page_docs))
            except PDFProcessingError as err:
                errors.append(str(err))
            except Exception as err:
                errors.append(f"Unexpected error in '{file_name}': {str(err)}")

        if not all_page_docs and errors:
            raise PDFProcessingError("\n".join(errors))

        # Split documents into smaller semantic chunks
        chunked_docs = self.split_documents(all_page_docs)

        summary = {
            "total_files": len(uploaded_files),
            "processed_files": len(uploaded_files) - len(errors),
            "total_pages": total_pages_count,
            "total_chunks": len(chunked_docs),
            "errors": errors
        }

        return chunked_docs, summary

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Document objects into overlapping chunks using RecursiveCharacterTextSplitter.
        Preserves original metadata (source, page) and adds chunk indexing.

        Args:
            documents: List of page-level Document objects.

        Returns:
            List[Document]: Chunked Document objects.
        """
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)

        # Attach unique chunk index within each document
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx

        return chunks
