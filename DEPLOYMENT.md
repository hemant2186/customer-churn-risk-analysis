# Deployment Guide

This app is ready for Streamlit Community Cloud. You only need a GitHub account and a Streamlit account connected to GitHub.

## 1. Push to GitHub

Create a new public GitHub repository named something like `customer-churn-retention-strategy`.

From this project folder, run:

```bash
git add .
git commit -m "Prepare churn project for deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-retention-strategy.git
git push -u origin main
```

If `git remote add origin` says the remote already exists, use:

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/customer-churn-retention-strategy.git
git push -u origin main
```

## 2. Deploy on Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io/).
2. Sign in with GitHub.
3. Click **Create app**.
4. Select **Yup, I have an app**.
5. Choose your repository.
6. Set the branch to `main`.
7. Set the main file path to `app.py`.
8. Choose an app URL, for example `customer-churn-retention`.
9. Click **Deploy**.

The app should build from `requirements.txt` and load the saved model from `models/churn_pipeline.joblib`.

## 3. Add the link to your resume

Use a short project line like:

```text
Customer Churn Prediction App: https://YOUR-APP-NAME.streamlit.app
```

On GitHub, add the deployed Streamlit URL to the repository website field so recruiters can find it immediately.

## Troubleshooting

- If the app cannot load the model, confirm `models/churn_pipeline.joblib` was committed.
- If the app cannot install dependencies, confirm `requirements.txt` is in the repository root.
- If Streamlit asks for the entrypoint file, use `app.py`.
- If the deployed app is outdated, reboot it from the Streamlit app settings after pushing changes.
