# Efficient Frontier Explorer — Streamlit App

This repository contains a Streamlit application that visualizes and analyzes portfolio efficient frontiers. It is themed to match the design of the [investgptai.io](https://investgptai.io/index.html) website and includes functionality for uploading your own price CSV, running optimizations on sample data, plotting portfolios, and generating a ChatGPT summary of the results.

## Features

- Upload a CSV with daily price data and generate the efficient frontier for your own assets, or fall back to a sample S&P 500 dataset.
- Choose the test/train split, number of portfolios (frontier size), and the risk measure (variance, semi-variance or CVaR).
- Optionally specify minimum annualized return targets.
- Visualize daily returns, the efficient frontier on train and test data, and summary statistics.
- Provide an OpenAI API key to generate a narrative summary of the frontier using ChatGPT. The app supports both the 0.28.x and 1.x versions of the `openai` Python library.
- Custom styling via `.streamlit/config.toml` to blend seamlessly with your existing website.

## Getting Started Locally

1. **Install dependencies** (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:

   ```bash
   streamlit run app.py
   ```

   If `streamlit` isn't available on your PATH, you can use:

   ```bash
   python -m streamlit run app.py
   ```

3. Open the provided URL in your browser (Streamlit will usually open it automatically). Upload a CSV if you have one, adjust the parameters, optionally enter your OpenAI API key, and click **Run optimization**.

## Deployment on Streamlit Community Cloud

To deploy this app on Streamlit Community Cloud and embed it in your static website, follow these steps:

1. **Create a new GitHub repository** and add the following files from this folder:
   - `app.py`
   - `.streamlit/config.toml`
   - `requirements.txt`
   - `icon.png`
   - `README.md`

2. Commit and push the repository to GitHub.

3. Log into [Streamlit Cloud](https://share.streamlit.io/) and click **New app**. Select your repository and choose `app.py` as the main file.

4. Deploy the app. After a few moments, Streamlit will provide a URL such as `https://your-username-your-repo.streamlit.app/`.

5. To make the tool available on your site, create a new HTML page (e.g., `efficient-frontier.html`) in your GitHub Pages repository and embed the app using an `<iframe>`:

   ```html
   <iframe src="https://your-username-your-repo.streamlit.app/" style="width:100%; height:800px; border:none;" allowfullscreen></iframe>
   ```

   Adjust the `height` as needed. Keep the same header and footer styling to maintain consistency across your site.

## Notes

- The app uses the `skfolio` library for portfolio optimization; be sure to include it in your environment. If you need other risk measures or optimization methods, you can modify `app.py` accordingly.
- If the `openai` library is not installed or an API key is not provided, the ChatGPT summary section will show a helpful message and skip the summary.

Enjoy exploring portfolio efficient frontiers!