
---

```markdown
# SaclayAI Chatbot API

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)

Welcome to the **SaclayAI Chatbot API**! This is a FastAPI-based chatbot designed to answer questions about **Paris-Saclay University**. The chatbot leverages advanced natural language processing (NLP) techniques, including Retrieval-Augmented Generation (RAG), to provide accurate and context-aware responses.

---

## Features

- **Multi-language Support**: Answers questions in the same language as the query (e.g., French, English, Spanish).
- **Retrieval-Augmented Generation (RAG)**: Combines information from uploaded PDFs, web scraping, and external APIs (Wikipedia, DuckDuckGo).
- **Conversational Memory**: Maintains context across multiple interactions.
- **Rate Limiting Handling**: Automatically detects and handles API rate limits.
- **Easy Integration**: Simple REST API for seamless integration with other applications.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [API Endpoints](#api-endpoints)
4. [Environment Variables](#environment-variables)
5. [Deployment](#deployment)
6. [Contributing](#contributing)
7. [License](#license)

---

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kempsly/saclayapi_bot.git
   cd saclayapi_bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory and add the following variables:
   ```env
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   ```

4. **Run the application**:
   ```bash
   uvicorn chatbot:app --host 0.0.0.0 --port 8000
   ```

5. **Access the API**:
   Open your browser and go to `http://localhost:8000/docs` to interact with the API.

---

## Usage

### Interacting with the Chatbot

Send a **POST** request to the `/chat` endpoint with a JSON payload containing your question:

```json
{
  "input_text": "Tell me about Paris-Saclay University"
}
```

#### Example using `curl`:
```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"input_text": "Tell me about Paris-Saclay University"}'
```

#### Example Response:
```json
{
  "response": "Paris-Saclay University is a French research university located in the Paris-Saclay cluster..."
}
```

---

## API Endpoints

| Endpoint       | Method | Description                          |
|----------------|--------|--------------------------------------|
| `/chat`        | POST   | Send a question to the chatbot.      |
| `/health`      | GET    | Check the health of the API.         |

---

## Environment Variables

The following environment variables are required for the API to function:

| Variable            | Description                          |
|---------------------|--------------------------------------|
| `GROQ_API_KEY`      | API key for Groq's LLM service.      |
| `OPENAI_API_KEY`    | API key for OpenAI embeddings.       |
| `LANGCHAIN_API_KEY` | API key for LangChain tracing.       |

---

## Deployment

### Deploying on Render

1. Fork this repository to your GitHub account.
2. Go to [Render](https://render.com) and create a new **Web Service**.
3. Connect your GitHub account and select this repository.
4. Set the following environment variables in Render:
   - `GROQ_API_KEY`
   - `OPENAI_API_KEY`
   - `LANGCHAIN_API_KEY`
5. Deploy the service.

Once deployed, your API will be accessible at the provided Render URL.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework.
- [LangChain](https://www.langchain.com/) for the NLP tools.
- [Groq](https://groq.com/) for the LLM service.
- [OpenAI](https://openai.com/) for embeddings.

---

## Contact

For questions or feedback, please contact:
- **Kempsly** - [GitHub](https://github.com/kempsly)

```

---

### Key Features of This README

1. **Badges**: Visual indicators for technologies used (FastAPI, Python, GitHub).
2. **Table of Contents**: Easy navigation for users.
3. **Installation Guide**: Step-by-step instructions for setting up the project.
4. **Usage Examples**: Clear examples for interacting with the API.
5. **API Documentation**: Details about available endpoints.
6. **Environment Variables**: Explanation of required configuration.
7. **Deployment Guide**: Instructions for deploying on Render.
8. **Contributing Section**: Encourages collaboration.
9. **License and Acknowledgments**: Proper attribution and licensing information.

---

### How to Use

1. Save the above content in a file named `README.md` in the root of your project.
2. Push it to your GitHub repository:
   ```bash
   git add README.md
   git commit -m "Add README.md"
   git push origin main
   ```
