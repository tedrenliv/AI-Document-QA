# Requirements Document

## Introduction

This feature adds local AI processing capabilities to the existing mainintegratedWORD.py application by integrating Ollama with embeddinggemma:latest model. This will provide users with an alternative to Google's Gemini API, allowing them to run AI-powered document Q&A entirely on their local machine without requiring internet connectivity or external API keys.

## Requirements

### Requirement 1

**User Story:** As a user, I want to choose between Google Gemini API and local Ollama processing, so that I can use the application offline or avoid external API dependencies.

#### Acceptance Criteria

1. WHEN the application starts THEN the system SHALL display a radio button or dropdown to select between "Google Gemini" and "Local Ollama" processing modes
2. WHEN "Local Ollama" mode is selected THEN the system SHALL disable the API key input field
3. WHEN "Google Gemini" mode is selected THEN the system SHALL enable the API key input field and require it for processing
4. IF the user switches between modes THEN the system SHALL update the UI accordingly without requiring application restart

### Requirement 2

**User Story:** As a user, I want the application to automatically detect if Ollama is available on my system, so that I know whether local processing is possible.

#### Acceptance Criteria

1. WHEN the application starts THEN the system SHALL check if Ollama is running and accessible on the default port (11434)
2. IF Ollama is not available THEN the system SHALL display a warning message and disable the "Local Ollama" option
3. WHEN Ollama becomes available THEN the system SHALL enable the "Local Ollama" option
4. IF embeddinggemma:latest model is not available THEN the system SHALL display an informative error message with installation instructions

### Requirement 3

**User Story:** As a user, I want to use embeddinggemma:latest for document processing through Ollama, so that I can get AI-powered answers using my local resources.

#### Acceptance Criteria

1. WHEN "Local Ollama" mode is selected and a question is asked THEN the system SHALL use Ollama API to communicate with embeddinggemma:latest model
2. WHEN processing with Ollama THEN the system SHALL format the prompt appropriately for the embeddinggemma model
3. IF Ollama processing fails THEN the system SHALL display a clear error message explaining the issue
4. WHEN using Ollama THEN the system SHALL maintain the same chunking and context preparation as the Google Gemini implementation

### Requirement 4

**User Story:** As a user, I want the same logging functionality regardless of which AI backend I choose, so that I can track my Q&A history consistently.

#### Acceptance Criteria

1. WHEN using Local Ollama mode THEN the system SHALL log questions and answers to the same logbook.txt file
2. WHEN logging Ollama responses THEN the system SHALL include a marker indicating the processing mode used
3. IF logging fails THEN the system SHALL display a warning but continue with normal operation
4. WHEN viewing logs THEN the user SHALL be able to distinguish between responses from different AI backends

### Requirement 5

**User Story:** As a user, I want clear feedback about processing status when using local Ollama, so that I understand what's happening during potentially slower local processing.

#### Acceptance Criteria

1. WHEN processing starts with Ollama THEN the system SHALL display a progress indicator or status message
2. WHEN Ollama is processing THEN the system SHALL disable the Ask button to prevent multiple simultaneous requests
3. IF Ollama processing takes longer than expected THEN the system SHALL provide feedback that processing is still ongoing
4. WHEN processing completes THEN the system SHALL re-enable the Ask button and clear status indicators