# Streamlit Cloud Deployment Guide

## Troubleshooting Deployment Issues

If you're seeing "The app's code is not connected to a remote GitHub repository":

### 1. Verify Repository Access

- **Repository URL**: `https://github.com/MishalQamar/videoQA`
- **Branch**: `main`
- **Main file**: `app.py`

### 2. Check Repository Visibility

Streamlit Community Cloud needs access to your repository:
- If the repository is **private**, ensure Streamlit Cloud has access
- Go to repository Settings → Collaborators → Add `streamlit-cloud` as collaborator (if needed)
- Or make the repository **public** temporarily to test

### 3. Verify Files are on GitHub

Check that these files exist on GitHub:
- ✅ `app.py`
- ✅ `requirements.txt`
- ✅ `.streamlit/config.toml`
- ✅ `README.md`

Visit: https://github.com/MishalQamar/videoQA

### 4. Deployment Steps

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Repository**: Select `MishalQamar/videoQA` from dropdown
   - If it doesn't appear, click "Browse repositories" and search for it
5. **Branch**: `main`
6. **Main file path**: `app.py`
7. **Python version**: 3.9 (default)
8. **Click "Deploy!"**

### 5. Common Issues

**Issue**: Repository not showing in dropdown
- **Solution**: Make sure you're signed in with the correct GitHub account that has access to the repository

**Issue**: "Cannot find app.py"
- **Solution**: Verify the branch name is `main` (not `master`)

**Issue**: Deployment fails
- **Solution**: Check the logs in Streamlit Cloud for error messages
- Verify `requirements.txt` has all dependencies

### 6. Add API Key (Optional)

After deployment, add your OpenAI API key:
1. In Streamlit Cloud app settings
2. Go to "Secrets"
3. Add:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

**Note**: Users can also enter their own API key in the app's sidebar.

