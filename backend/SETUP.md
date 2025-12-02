# Backend Setup Guide

## Environment Variables

This application requires an OpenAI API key for the chatbot functionality.

### Setup Steps:

1. **Copy the example environment file:**

   ```bash
   cp .env.example .env
   ```

2. **Get your OpenAI API key:**

   - Go to [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a new API key
   - Copy the key

3. **Add your API key to `.env` file:**

   ```
   OPENAI_API_KEY=your-actual-api-key-here
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the backend:**
   ```bash
   python app.py
   ```

## Important Notes

- **Never commit your `.env` file to Git** - it contains sensitive information
- The `.env` file is already in `.gitignore` to prevent accidental commits
- Make sure you have sufficient credits in your OpenAI account for the chatbot to work

## Security

- Keep your API keys private
- Don't share your `.env` file
- Rotate your API keys regularly
- Monitor your OpenAI usage at https://platform.openai.com/usage
