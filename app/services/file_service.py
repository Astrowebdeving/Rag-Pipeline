"""File Processing Service

Handles extraction of text from various file formats including
PDF, text files, and potentially other document types.
"""

import os
import time
import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# File processing imports with availability flags
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available - PDF processing disabled")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available - advanced PDF processing disabled")


class FileProcessingService:
    """Service for extracting text from various file formats."""
    
    def __init__(self):
        """Initialize the file processing service."""
        self.supported_extensions = ['txt', 'pdf']
        if not PDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            self.supported_extensions = ['txt']
            logger.warning("No PDF libraries available - only text files supported")
    
    def get_supported_extensions(self) -> list:
        """Get list of supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        return self.supported_extensions.copy()
    
    def extract_text_from_file(self, file_path: str, filename: str) -> str:
        """Extract text from uploaded files based on file type.
        
        Args:
            file_path: Path to the file to process
            filename: Original filename (used to determine file type)
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file type is not supported
            Exception: If text extraction fails
        """
        start_time = time.time()
        
        # Get file extension to determine processing method
        file_extension = filename.lower().split('.')[-1]
        
        logger.info(f"Starting text extraction from {filename} ({file_extension})")
        
        try:
            if file_extension == 'txt':
                text = self._extract_text_file(file_path)
            elif file_extension == 'pdf':
                text = self._extract_pdf_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            extraction_time = time.time() - start_time
            logger.info(f"Text extraction completed in {extraction_time:.2f} seconds")
            logger.info(f"Extracted {len(text)} characters from {filename}")
            
            return text
            
        except Exception as e:
            logger.error(f"Text extraction failed for {filename}: {str(e)}")
            raise
    
    def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text files.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File contents as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Failed to read text file: {str(e)}")
            raise
    
    def _extract_pdf_file(self, file_path: str) -> str:
        """Extract text from PDF files using available libraries.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        if not PDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            raise ValueError("PDF processing not available - install PyPDF2 or pdfplumber")
        
        # Try pdfplumber first (usually better text extraction)
        if PDFPLUMBER_AVAILABLE:
            try:
                return self._extract_pdf_with_pdfplumber(file_path)
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {str(e)}")
                if PDF_AVAILABLE:
                    logger.info("Falling back to PyPDF2")
                else:
                    raise
        
        # Fall back to PyPDF2
        if PDF_AVAILABLE:
            try:
                return self._extract_pdf_with_pypdf2(file_path)
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed: {str(e)}")
                raise
        
        raise Exception("No PDF processing libraries available")
    
    def _extract_pdf_with_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber (advanced PDF parsing).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        import pdfplumber
        
        text_content = []
        
        with pdfplumber.open(file_path) as pdf:
            logger.info(f"Processing PDF with {len(pdf.pages)} pages using pdfplumber")
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Heuristic multi-column handling: split page into left/right halves
                    # only if the word distribution suggests columns
                    words = []
                    try:
                        words = page.extract_words()
                    except Exception:
                        words = []

                    use_two_cols = False
                    if words:
                        width = getattr(page, 'width', None)
                        if width:
                            mid_x = width / 2.0
                            left = sum(1 for w in words if ((w.get('x0', 0) + w.get('x1', 0)) / 2.0) < mid_x)
                            right = len(words) - left
                            # Both sides should have a reasonable share to consider it two-column
                            if left > 0 and right > 0 and min(left, right) / max(left, right) > 0.3:
                                use_two_cols = True

                    if use_two_cols:
                        width = page.width
                        height = page.height
                        left_box = (0, 0, width * 0.5, height)
                        right_box = (width * 0.5, 0, width, height)
                        left_page = page.crop(left_box)
                        right_page = page.crop(right_box)
                        left_text = left_page.extract_text() or ''
                        right_text = right_page.extract_text() or ''
                        page_text = (left_text.strip() + ("\n\n" if left_text and right_text else "") + right_text.strip()).strip()
                    else:
                        page_text = page.extract_text() or ''

                    if page_text:
                        text_content.append(page_text)
                        logger.debug(f"Extracted {len(page_text)} characters from page {page_num}{' (2-col)' if use_two_cols else ''}")
                    else:
                        logger.warning(f"No text found on page {page_num}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                    continue
        
        extracted_text = '\n\n'.join(text_content)
        
        if not extracted_text.strip():
            raise Exception("No text could be extracted from PDF using pdfplumber")
        
        return extracted_text
    
    def _extract_pdf_with_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2 (basic PDF parsing).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        import PyPDF2
        
        text_content = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            logger.info(f"Processing PDF with {len(pdf_reader.pages)} pages using PyPDF2")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                        logger.debug(f"Extracted {len(page_text)} characters from page {page_num}")
                    else:
                        logger.warning(f"No text found on page {page_num}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                    continue
        
        extracted_text = '\n\n'.join(text_content)
        
        if not extracted_text.strip():
            raise Exception("No text could be extracted from PDF using PyPDF2")
        
        return extracted_text
    
    def validate_file(self, file_obj, max_size_mb: int = 50) -> tuple:
        """Validate uploaded file.
        
        Args:
            file_obj: Flask file object
            max_size_mb: Maximum file size in MB
            
        Returns:
            Tuple of (is_valid, error_message, file_info)
        """
        try:
            # Check if file exists
            if not file_obj or not file_obj.filename:
                return False, "No file selected", None
            
            # Read file content to check size
            file_obj.seek(0)
            file_content = file_obj.read()
            file_size = len(file_content)
            
            # Check if file is empty
            if file_size == 0:
                return False, "Uploaded file is empty", None
            
            # Check file size limit
            max_size_bytes = max_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                return False, f"File size exceeds {max_size_mb}MB limit", None
            
            # Validate file extension
            file_extension = file_obj.filename.lower().split('.')[-1]
            if file_extension not in self.supported_extensions:
                return False, f"Unsupported file type. Supported: {', '.join(self.supported_extensions)}", None
            
            # Return file info for processing
            file_info = {
                'content': file_content,
                'size': file_size,
                'extension': file_extension,
                'filename': file_obj.filename
            }
            
            return True, None, file_info
            
        except Exception as e:
            logger.error(f"File validation failed: {str(e)}")
            return False, f"File validation error: {str(e)}", None
    
    def save_temp_file(self, file_content: bytes, file_extension: str) -> str:
        """Save file content to temporary file.
        
        Args:
            file_content: File content as bytes
            file_extension: File extension
            
        Returns:
            Path to temporary file
        """
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
            temp_file_path = temp_file.name
            temp_file.write(file_content)
            temp_file.close()
            
            logger.info(f"File saved to temporary path: {temp_file_path}")
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Failed to save temporary file: {str(e)}")
            raise
    
    def cleanup_temp_file(self, temp_file_path: str):
        """Clean up temporary file.
        
        Args:
            temp_file_path: Path to temporary file to delete
        """
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except OSError as e:
                logger.warning(f"Could not clean up temporary file {temp_file_path}: {str(e)}")


# Global instance
file_service = FileProcessingService() 