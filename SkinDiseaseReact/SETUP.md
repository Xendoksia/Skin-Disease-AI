# Frontend Setup

## Environment Variables

This React application uses environment variables to configure the backend API URL.

### Setup Steps:

1. **Copy the example environment file:**

   ```bash
   cp .env.example .env
   ```

2. **Configure the API URL in `.env` file:**

   For **development** (local):

   ```
   VITE_API_BASE_URL=http://localhost:5000/api
   ```

   For **production**:

   ```
   VITE_API_BASE_URL=https://your-backend-domain.com/api
   ```

3. **Install dependencies:**

   ```bash
   npm install
   ```

4. **Run the development server:**

   ```bash
   npm run dev
   ```

5. **Build for production:**
   ```bash
   npm run build
   ```

## Important Notes

- **Never commit your `.env` file to Git** - it may contain sensitive configuration
- The `.env` file is already in `.gitignore` to prevent accidental commits
- Environment variables in Vite must be prefixed with `VITE_` to be exposed to the client
- Changes to `.env` require restarting the development server

## Default Configuration

If no `.env` file is found, the application defaults to:

- API URL: `http://localhost:5000/api`

This works fine for local development.
